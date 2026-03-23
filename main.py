import asyncio
import gc
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import PermissionType
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.config.default import CONFIG_METADATA_2
from astrbot.core.provider.entities import ProviderType
from astrbot.core.provider.provider import EmbeddingProvider
from astrbot.core.provider.register import (
    provider_registry,
    register_provider_adapter,
)

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MAX_TEXT_LENGTH = 8192
DEFAULT_MAX_BATCH_SIZE = 32


def _looks_like_relative_path_escape(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def _normalize_text_input(text: str, max_length: int) -> str:
    normalized = (text or "").strip()
    if not normalized:
        raise ValueError("[STEmbedding] Text must not be empty")
    if len(normalized) > max_length:
        raise ValueError(
            f"[STEmbedding] Text is too long: {len(normalized)} > {max_length}"
        )
    return normalized


def _normalize_text_list(
    texts: list[str], max_length: int, max_batch_size: int
) -> list[str]:
    if not texts:
        raise ValueError("[STEmbedding] Text list must not be empty")
    if len(texts) > max_batch_size:
        raise ValueError(
            f"[STEmbedding] Batch size is too large: {len(texts)} > {max_batch_size}"
        )
    return [_normalize_text_input(text, max_length) for text in texts]


def _exception_chain_text(exc: BaseException) -> str:
    parts: list[str] = []
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        try:
            text = str(cur)
        except Exception:
            text = repr(cur)
        if text:
            parts.append(text)
        cur = cur.__cause__ or cur.__context__
    return "\n".join(parts)


def _looks_like_torch_weights_only_error(exc: BaseException) -> bool:
    text = _exception_chain_text(exc).lower()
    if not text:
        return False

    if "weights only load failed" in text:
        return True
    if "weightsunpickler" in text and "weights_only" in text:
        return True
    if "torch.load" in text and "weights_only" in text and "default value" in text:
        return True

    return False


def _parse_weights_only_override(value) -> bool | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return bool(value)

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"", "auto"}:
            return None
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False

    return None


@contextmanager
def _torch_load_weights_only_default(weights_only: bool) -> Iterator[None]:
    """
    临时修改 torch.load 的默认 weights_only 行为。

    - 仅当当前 torch.load 支持 weights_only 参数时生效
    - 退出时恢复原始函数，避免影响进程内其他加载逻辑
    """
    try:
        import torch
    except Exception:
        yield
        return

    try:
        sig = inspect.signature(torch.load)
    except Exception:
        yield
        return

    if "weights_only" not in sig.parameters:
        yield
        return

    original_torch_load = torch.load

    serialization_module = None
    original_serialization_load = None
    try:
        import torch.serialization as serialization_module  # type: ignore

        if hasattr(serialization_module, "load"):
            original_serialization_load = serialization_module.load
    except Exception:
        serialization_module = None
        original_serialization_load = None

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", weights_only)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_load
    if (
        serialization_module is not None
        and original_serialization_load is original_torch_load
    ):
        serialization_module.load = patched_load

    try:
        yield
    finally:
        torch.load = original_torch_load
        if serialization_module is not None and original_serialization_load is not None:
            serialization_module.load = original_serialization_load


def _load_sentence_transformer(model_path: str, weights_only: bool | None):
    from sentence_transformers import SentenceTransformer

    if weights_only is None:
        return SentenceTransformer(model_path)

    with _torch_load_weights_only_default(weights_only):
        return SentenceTransformer(model_path)


# ============================================================
# Embedding Provider
# ============================================================
class STEmbeddingProvider(EmbeddingProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)

        # -------- 模型路径处理（Pathlib）--------
        base_path = provider_config.get("STEmbedding_path", DEFAULT_MODEL_NAME)

        data_dir = Path(StarTools.get_data_dir())
        base_path = Path(base_path)
        if not base_path.is_absolute() and _looks_like_relative_path_escape(base_path):
            raise ValueError(
                "[STEmbedding] Relative model paths must stay within the plugin data directory"
            )

        self.STEmbedding_path = (
            base_path if base_path.is_absolute() else data_dir / base_path
        )

        # -------- 运行状态 --------
        self.model = None
        self._model_lock = asyncio.Lock()
        self._encode_lock = asyncio.Lock()
        self.max_text_length = int(
            self.provider_config.get("max_text_length", DEFAULT_MAX_TEXT_LENGTH)
        )
        self.max_batch_size = int(
            self.provider_config.get("max_batch_size", DEFAULT_MAX_BATCH_SIZE)
        )

        # -------- sentence_transformers 环境检测（一次）--------
        self._env_available = True
        self._env_error: str | None = None
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            self._env_available = False
            self._env_error = "未安装 sentence-transformers，请执行：pip install sentence-transformers"

        logger.info(
            f"[STEmbedding] Provider 初始化完成，"
            f"env_available={self._env_available}, "
            f"model_path={self.STEmbedding_path}, "
            f"max_text_length={self.max_text_length}, "
            f"max_batch_size={self.max_batch_size}, "
            f"torch_load_weights_only={provider_config.get('STEmbedding_torch_load_weights_only', 'auto')}"
        )

    # ====================================================
    # 内部工具
    # ====================================================

    def _ensure_env_available(self):
        if not self._env_available:
            raise RuntimeError(f"[STEmbedding] 环境不可用: {self._env_error}")
        if self.max_text_length <= 0:
            raise RuntimeError("[STEmbedding] max_text_length must be greater than 0")
        if self.max_batch_size <= 0:
            raise RuntimeError("[STEmbedding] max_batch_size must be greater than 0")

    async def _encode(self, payload: str | list[str]):
        await self._ensure_model_loaded()

        loop = asyncio.get_running_loop()
        async with self._encode_lock:
            return await loop.run_in_executor(None, self.model.encode, payload)

    async def _ensure_model_loaded(self):
        """
        Lazy Loading + 并发安全
        """
        self._ensure_env_available()

        if self.model is not None:
            return

        async with self._model_lock:
            if self.model is not None:
                return

            logger.info(f"[STEmbedding] 开始加载模型: {self.STEmbedding_path}")
            loop = asyncio.get_running_loop()

            weights_only_override = _parse_weights_only_override(
                self.provider_config.get("STEmbedding_torch_load_weights_only", "auto")
            )

            try:
                try:
                    self.model = await loop.run_in_executor(
                        None,
                        _load_sentence_transformer,
                        str(self.STEmbedding_path),
                        weights_only_override,
                    )
                    if weights_only_override is None:
                        logger.info("[STEmbedding] 模型加载成功")
                    else:
                        logger.info(
                            f"[STEmbedding] 模型加载成功（weights_only={weights_only_override}）"
                        )
                except Exception as e:
                    if (
                        weights_only_override is None
                        and _looks_like_torch_weights_only_error(e)
                    ):
                        logger.warning(
                            "[STEmbedding] 检测到 PyTorch weights_only 加载失败，"
                            "正在使用 weights_only=False 重试（仅可信模型）"
                        )
                        self.model = await loop.run_in_executor(
                            None,
                            _load_sentence_transformer,
                            str(self.STEmbedding_path),
                            False,
                        )
                        logger.info("[STEmbedding] 模型加载成功（weights_only=False）")
                    else:
                        raise
            except ImportError:
                logger.info("[STEmbedding] sentence_transformers导入失败")
                raise
            except Exception as e:
                logger.error("[STEmbedding] 模型加载失败", exc_info=True)
                if weights_only_override is None:
                    raise RuntimeError(f"模型加载失败: {e}") from e
                raise RuntimeError(
                    f"模型加载失败（weights_only={weights_only_override}）: {e}"
                ) from e

    def _cleanup_resources(self) -> bool:
        """
        统一的模型 / 显存 / 内存清理逻辑
        """
        try:
            try:
                import torch

                if self.model is not None and hasattr(self.model, "to"):
                    self.model.to("cpu")
            except ImportError:
                pass

            self.model = None
            gc.collect()

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except ImportError:
                pass

            return True
        except Exception:
            logger.error("[STEmbedding] 资源清理失败", exc_info=True)
            return False

    # ====================================================
    # Embedding API
    # ====================================================

    async def get_embedding(self, text: str) -> list[float]:
        normalized = _normalize_text_input(text, self.max_text_length)
        embedding = await self._encode(normalized)
        return embedding.tolist()

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        normalized = _normalize_text_list(
            texts,
            self.max_text_length,
            self.max_batch_size,
        )
        embeddings = await self._encode(normalized)
        return embeddings.tolist()

    def get_dim(self) -> int:
        return int(self.provider_config.get("embedding_dimensions", 384))

    # ====================================================
    # 卸载
    # ====================================================

    async def unload_model(self) -> bool:
        async with self._model_lock:
            if self.model is None:
                logger.info("[STEmbedding] 模型未加载，无需卸载")
                return True

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._cleanup_resources)

    def force_unload_sync(self) -> bool:
        if self.model is None:
            return True
        return self._cleanup_resources()


# ============================================================
# Provider 注册函数（只负责注册）
# ============================================================
def register_STEmbeddingProvider():
    try:
        register_provider_adapter(
            "STEmbedding",
            "Sentence Transformers Embedding Provider",
            provider_type=ProviderType.EMBEDDING,
        )(STEmbeddingProvider)
        logger.info("[STEmbedding] Provider 已注册")
    except ValueError:
        logger.info("[STEmbedding] Provider 已存在，跳过注册")


# ============================================================
# Star 插件本体
# ============================================================
@register("STEmbedding", "Lishining", "我的STEmbedding", "1.0.0")
class STEmbedding(Star):
    _registered = False

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self.auto_start = self.config.get("auto_start", 0) == 1

    # --------------------------------------------------------
    def _register_config(self):
        if self._registered:
            return False
        # ---- 防御性获取配置节点----
        try:
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
                "config_template"
            ]["STEmbedding"] = {
                "id": "STEmbedding",
                "type": "STEmbedding",
                "provider": "Local",
                "STEmbedding_path": DEFAULT_MODEL_NAME,
                "STEmbedding_torch_load_weights_only": "auto",
                "provider_type": "embedding",
                "enable": True,
                "embedding_dimensions": 384,
                "max_text_length": DEFAULT_MAX_TEXT_LENGTH,
                "max_batch_size": DEFAULT_MAX_BATCH_SIZE,
            }
        except KeyError:
            logger.error("[STEmbedding] AstrBot 配置结构异常，无法注册 Provider")
            return False

        try:
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"][
                "STEmbedding_path"
            ] = {
                "description": "SentenceTransformer 模型路径",
                "type": "string",
            }
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"][
                "STEmbedding_torch_load_weights_only"
            ] = {
                "description": "torch.load 的 weights_only（auto/true/false）。PyTorch 2.6+ 默认 true，部分旧模型需 false（仅可信模型）",
                "type": "string",
            }
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"][
                "max_text_length"
            ] = {
                "description": "Maximum length allowed for a single text input.",
                "type": "int",
            }
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"][
                "max_batch_size"
            ] = {
                "description": "Maximum number of texts allowed in one batch request.",
                "type": "int",
            }
        except KeyError:
            logger.error("[STEmbedding] AstrBot 配置结构异常，无法注册 Provider")
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
                "config_template"
            ].pop("STEmbedding", None)
            return False

        # ---- Provider 注册 ----
        already_registered = False
        if isinstance(provider_registry, list):
            for p in provider_registry:
                if getattr(p, "type", None) == "STEmbedding":
                    already_registered = True
                    break

        if not already_registered:
            register_STEmbeddingProvider()

        self._registered = True
        logger.info("[STEmbedding] 配置与 Provider 注册完成")
        return True

    def _unregister_config(self):
        try:
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
                "config_template"
            ].pop("STEmbedding", None)
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"].pop(
                "STEmbedding_path", None
            )
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"].pop(
                "STEmbedding_torch_load_weights_only", None
            )
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"].pop(
                "max_text_length", None
            )
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"].pop(
                "max_batch_size", None
            )
        except KeyError:
            pass

        self._registered = False
        logger.info("[STEmbedding] 配置已清理")

    # --------------------------------------------------------
    # Commands
    # --------------------------------------------------------
    @filter.command_group("ste")
    def ste(self):
        pass

    @ste.command("help")
    async def help(self, event: AstrMessageEvent):
        """获取代码帮助"""
        help_text = [
            "STEmbedding 插件",
            "/ste register                      注册 Provider",
            "/ste redb                          重新加载数据库",
            "/ste kbnep                         获取所有数据库以及其对应的embedding_provider_id",
            "/ste ukbw [embedding_provider_id]  卸载掉embedding_provider_id的权重(防止不运行时消耗过大)",
        ]
        yield event.plain_result("\n".join(help_text))

    @ste.command("redb")
    @filter.permission_type(PermissionType.ADMIN)
    async def redb(self, event: AstrMessageEvent):
        """重新加载数据库,防止在astrbot初始化后出现,STEmbeddingProvider未注册数据库加载失败,从而无法加载数据库的情况"""
        await self.context.kb_manager.load_kbs()
        yield event.plain_result("[STEmbedding] 数据库已重新加载")

    @ste.command("register")
    @filter.permission_type(PermissionType.ADMIN)
    async def register_cmd(self, event: AstrMessageEvent):
        """主动将STEmbedding注册到嵌入式向量提供商"""
        yield event.plain_result("[STEmbedding] 正在注册 Provider")
        if self._register_config():
            yield event.plain_result("[STEmbedding] 注册 Provider 成功")
        await self.context.kb_manager.load_kbs()

    @ste.command("kbnep")
    @filter.permission_type(PermissionType.ADMIN)
    async def get_kb_name_epid(self, event: AstrMessageEvent):
        """获取所有数据库以及其对应的编码器"""
        outputtext = []
        for kb_helper in self.context.kb_manager.kb_insts.values():
            outputtext.append(
                f"数据库名称:{kb_helper.kb.kb_name}, 编码器:{kb_helper.kb.embedding_provider_id}"
            )
        yield event.plain_result("可用数据库:\n" + "\n".join(outputtext))
        logger.info("[STEmbedding] 可用数据库:\n" + "\n".join(outputtext))

    @ste.command("ukbw")
    @filter.permission_type(PermissionType.ADMIN)
    async def uninstall_kbw(self, event: AstrMessageEvent, embedding_provider_id: str):
        """清理权重,防止用不到权重时,权重会占用太多内存"""
        pm = self.context.provider_manager.get_provider_by_id(embedding_provider_id)
        if isinstance(pm, STEmbeddingProvider):
            yield event.plain_result("[STEmbedding] 正在清理权重")
            logger.info("[STEmbedding] 正在清理权重")
            await pm.unload_model()
            yield event.plain_result("[STEmbedding] 清理权重成功")
            logger.info("[STEmbedding] 清理权重成功")
        else:
            yield event.plain_result(
                f"[STEmbedding] 编码器实例:{embedding_provider_id},不为STEmbeddingProvider"
            )
            logger.info(
                f"[STEmbedding] 编码器实例:{embedding_provider_id},不为STEmbeddingProvider"
            )

    # --------------------------------------------------------
    # 生命周期
    # --------------------------------------------------------
    async def initialize(self):
        if not self.auto_start:
            logger.info("[STEmbedding] 未启用自加载")
            return
        logger.info("[STEmbedding] 插件初始化中")
        if self._register_config():
            logger.info("[STEmbedding] 注册 Provider 成功")
        else:
            logger.error("[STEmbedding] 插件初始化失败")

    async def terminate(self):
        logger.info("[STEmbedding] 插件终止中")
        self._unregister_config()
        logger.info("[STEmbedding] 插件终止完成")

    # --------------------------------------------------------
    # 在astrbot启动时
    # --------------------------------------------------------
    @filter.on_astrbot_loaded()
    async def init_db(self):
        """如果启动自动加载,将在astrbot启动后自动刷新数据库"""
        if not self.auto_start:
            return
        if not self._registered:
            logger.info("[STEmbedding] 刷新数据库失败,未注册编码器")
        try:
            await self.context.kb_manager.load_kbs()
            logger.info("[STEmbedding] 插件初始化完成,已重新刷新数据库")
        except:
            raise
