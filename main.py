import asyncio
import os
import gc

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

# ============================================================
# Provider 注册函数（显式调用才会注册）
# ============================================================
def register_STEmbeddingProvider():
    """
    显式调用才会注册 STEmbedding Provider
    """

    @register_provider_adapter(
        "STEmbedding",
        "Sentence Transformers Embedding 提供商适配器",
        provider_type=ProviderType.EMBEDDING,
    )
    class STEmbeddingProvider(EmbeddingProvider):

        def __init__(self, provider_config: dict, provider_settings: dict) -> None:
            super().__init__(provider_config, provider_settings)
            self.provider_config = provider_config
            self.provider_settings = provider_settings
            # -------- 模型路径处理 --------
            base_path = provider_config.get(
                "STEmbedding_path",
                "./paraphrase-multilingual-MiniLM-L12-v2/"
            )
            if os.path.isabs(base_path):
                self.STEmbedding_path = base_path
            else:
                self.STEmbedding_path = os.path.join(
                    str(StarTools.get_data_dir()), base_path
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
                        None, SentenceTransformer, self.STEmbedding_path
                    )
                    logger.info("[STEmbedding] 模型加载成功")
                except ImportError:
                    logger.error("[STEmbedding] sentence_transformers的导入失败")
                    raise
                except Exception as e:
                    logger.error("[STEmbedding] 模型加载失败", exc_info=True)
                    raise RuntimeError(f"模型加载失败: {e}") from e

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

                try:
                    try:
                        import torch
                        if hasattr(self.model, "to"):
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

                    logger.info("[STEmbedding] 模型卸载完成")
                    return True
                except Exception:
                    logger.error("[STEmbedding] 模型卸载失败", exc_info=True)
                    return False

        def force_unload_sync(self) -> bool:
            try:
                if self.model is None:
                    return True

                try:
                    import torch
                    if hasattr(self.model, "to"):
                        self.model.to("cpu")
                except ImportError:
                    pass

                self.model = None
                gc.collect()

                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                return True
            except Exception:
                logger.error("[STEmbedding] 强制卸载失败", exc_info=True)
                return False

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
            return
        # -------- 注册配置模板 --------
        CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["config_template"]["STEmbedding"] = {
            "id": "STEmbedding",
            "type": "STEmbedding",
            "provider": "Local",
            "STEmbedding_path": "./paraphrase-multilingual-MiniLM-L12-v2/",
            "provider_type": "embedding",
            "enable": True,
            "embedding_dimensions": 384,
        }
        CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"]["STEmbedding_path"] = {
            "description": "SentenceTransformer 模型路径",
            "type": "string",
        }
        # -------- 是否已注册 Provider --------
        already_registered = False
        if isinstance(provider_registry, list):
            for p in provider_registry:
                if getattr(p, "type", None) == "STEmbedding":
                    already_registered = True
                    break
        if not already_registered:
            try:
                register_STEmbeddingProvider()
                logger.info("[STEmbedding] Provider 已动态注册")
            except ValueError as e:
                logger.info(f"[STEmbedding] Provider 已存在: {e}")
        else:
            logger.info("[STEmbedding] Provider 已存在，跳过注册")
        self._registered = True

    def _unregister_config(self):
        logger.info("[STEmbedding] 开始清理配置")
        CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["config_template"].pop(
            "STEmbedding", None
        )
        CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"].pop(
            "STEmbedding_path", None
        )
        self._registered = False
        logger.info("[STEmbedding] 配置清理完成")

    # --------------------------------------------------------
    # Commands
    # --------------------------------------------------------
    @filter.command_group("ste")
    def ste(self):
        pass

    @ste.command("help")
    async def help(self, event: AstrMessageEvent):
        help_text = [
            "你好，这是 STEmbedding 插件",
            "请使用/ste register 进行STEmbedding注册",
            "使用/ste redb 进行数据库重新载入(如果数据库没有被检测到,请使用这个)"
        ]
        yield event.plain_result("\n".join(help_text))

    @ste.command("redb")
    async def redb(self, event: AstrMessageEvent):
        await self.context.kb_manager.load_kbs()
        yield event.plain_result("[STEmbedding] 注册完成，数据库已重新加载")

    @ste.command("register")
    async def register_cmd(self, event: AstrMessageEvent):
        yield event.plain_result("[STEmbedding] 正在注册配置与 Provider")
        self._register_config()
        await self.context.kb_manager.load_kbs()
        yield event.plain_result("[STEmbedding] 注册完成，数据库已重新加载")

    # --------------------------------------------------------
    # 生命周期
    # --------------------------------------------------------
    async def initialize(self):
        if not self.auto_start:
            logger.info("[STEmbedding] 未启用开机自启")
            return
        logger.info("[STEmbedding] 插件初始化中")
        self._register_config()
        await self.context.kb_manager.load_kbs()
        logger.info("[STEmbedding] 插件初始化完成")

    async def terminate(self):
        logger.info("[STEmbedding] 插件终止中")
        self._unregister_config()
        logger.info("[STEmbedding] 插件终止完成")
