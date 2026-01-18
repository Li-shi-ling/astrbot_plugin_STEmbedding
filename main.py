import asyncio
import gc
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.config.default import CONFIG_METADATA_2
from astrbot.core.provider.entities import ProviderType
from astrbot.core.provider.provider import EmbeddingProvider
from astrbot.core.provider.register import (
    provider_registry,
    register_provider_adapter,
)

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ============================================================
# Embedding Provider
# ============================================================
class STEmbeddingProvider(EmbeddingProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)

        # -------- 模型路径处理（Pathlib）--------
        base_path = provider_config.get(
            "STEmbedding_path",
            DEFAULT_MODEL_NAME
        )

        data_dir = Path(StarTools.get_data_dir())
        base_path = Path(base_path)

        self.STEmbedding_path = (
            base_path if base_path.is_absolute()
            else data_dir / base_path
        )

        # -------- 运行状态 --------
        self.model = None
        self._model_lock = asyncio.Lock()

        # -------- sentence_transformers 环境检测（一次）--------
        self._env_available = True
        self._env_error: str | None = None
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            self._env_available = False
            self._env_error = (
                "未安装 sentence-transformers，请执行：pip install sentence-transformers"
            )

        logger.info(
            f"[STEmbedding] Provider 初始化完成，"
            f"env_available={self._env_available}, "
            f"model_path={self.STEmbedding_path}"
        )

    # ====================================================
    # 内部工具
    # ====================================================

    def _ensure_env_available(self):
        if not self._env_available:
            raise RuntimeError(f"[STEmbedding] 环境不可用: {self._env_error}")

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

            try:
                from sentence_transformers import SentenceTransformer
                self.model = await loop.run_in_executor(
                    None,
                    SentenceTransformer,
                    str(self.STEmbedding_path)
                )
                logger.info("[STEmbedding] 模型加载成功")
            except ImportError:
                logger.info("[STEmbedding] sentence_transformers导入失败")
                raise
            except Exception as e:
                logger.error("[STEmbedding] 模型加载失败", exc_info=True)
                raise RuntimeError(f"模型加载失败: {e}") from e

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
        except:
            logger.error("[STEmbedding] 资源清理失败", exc_info=True)
            return False

    # ====================================================
    # Embedding API
    # ====================================================

    async def get_embedding(self, text: str) -> list[float]:
        await self._ensure_model_loaded()

        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            None, self.model.encode, text
        )
        return embedding.tolist()

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        await self._ensure_model_loaded()

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, self.model.encode, texts
        )
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
            return await loop.run_in_executor(
                None,
                self._cleanup_resources
            )

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
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["config_template"]["STEmbedding"] = {
                "id": "STEmbedding",
                "type": "STEmbedding",
                "provider": "Local",
                "STEmbedding_path": DEFAULT_MODEL_NAME,
                "provider_type": "embedding",
                "enable": True,
                "embedding_dimensions": 384,
            }
        except KeyError:
            logger.error("[STEmbedding] AstrBot 配置结构异常，无法注册 Provider")
            return False

        try:
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"]["STEmbedding_path"] = {
                "description": "SentenceTransformer 模型路径",
                "type": "string",
            }
        except KeyError:
            logger.error("[STEmbedding] AstrBot 配置结构异常，无法注册 Provider")
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["config_template"].pop("STEmbedding", None)
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
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["config_template"].pop("STEmbedding", None)
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"].pop("STEmbedding_path", None)
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
        help_text = [
            "STEmbedding 插件",
            "/ste register                      注册 Provider",
            "/ste redb                          重新加载数据库",
            "/ste kbnep                         获取所有数据库以及其对应的embedding_provider_id",
            "/ste ukbw [embedding_provider_id]  卸载掉embedding_provider_id的权重(防止不运行时消耗过大)",
        ]
        yield event.plain_result("\n".join(help_text))

    @ste.command("redb")
    async def redb(self, event: AstrMessageEvent):
        """重新加载数据库,防止在astrbot初始化后出现,STEmbeddingProvider未注册数据库加载失败,从而无法加载数据库的情况"""
        await self.context.kb_manager.load_kbs()
        yield event.plain_result("[STEmbedding] 数据库已重新加载")

    @ste.command("register")
    async def register_cmd(self, event: AstrMessageEvent):
        yield event.plain_result("[STEmbedding] 正在注册 Provider")
        if self._register_config():
            yield event.plain_result("[STEmbedding] 注册 Provider 成功")
        await self.context.kb_manager.load_kbs()

    @ste.command("kbnep")
    async def get_kb_name_epid(self, event: AstrMessageEvent):
        """获取所有数据库以及其对应的编码器"""
        outputtext = []
        for kb_helper in self.context.kb_manager.kb_insts.values():
            outputtext.append(
                f"数据库名称:{kb_helper.kb.kb_name}, 编码器:{kb_helper.kb.embedding_provider_id}"
            )
        yield event.plain_result(f"可用数据库:\n" + "\n".join(outputtext))
        logger.info(f"[STEmbedding] 可用数据库:\n" + "\n".join(outputtext))

    @ste.command("ukbw")
    async def uninstall_kbw(self, event: AstrMessageEvent, embedding_provider_id: str):
        pm = self.context.provider_manager.get_provider_by_id(embedding_provider_id)
        if isinstance(pm, STEmbeddingProvider):
            yield event.plain_result(f"[STEmbedding] 正在清理权重")
            logger.info(f"[STEmbedding] 正在清理权重")
            await pm.unload_model()
            yield event.plain_result(f"[STEmbedding] 清理权重成功")
            logger.info(f"[STEmbedding] 清理权重成功")
        else:
            yield event.plain_result(f"[STEmbedding] 编码器实例:{embedding_provider_id},不为STEmbeddingProvider")
            logger.info(f"[STEmbedding] 编码器实例:{embedding_provider_id},不为STEmbeddingProvider")

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
