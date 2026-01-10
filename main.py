import asyncio
import os

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.config.default import CONFIG_METADATA_2
from astrbot.core.provider.entities import ProviderType
from astrbot.core.provider.provider import EmbeddingProvider
from astrbot.core.provider.register import (
    provider_cls_map,
    provider_registry,
    register_provider_adapter,
)

# 延迟导入检查，不立即抛出异常
sentence_transformers_available = False
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    logger.warning("未检测到 sentence-transformers 库，插件功能将受限")


@register("STEmbedding", "Lishining", "我的STEmbedding", "1.0.0")
class STEmbedding(Star):
    # 类属性
    _registered = False

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.model = None
        self.available = sentence_transformers_available

    @classmethod
    def _register_config(cls):
        """注册STEmbedding配置到全局配置中"""
        if cls._registered:
            return

        # 注册配置模板
        CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["config_template"]["STEmbedding"] = {
            "id": "STEmbedding",
            "type": "STEmbedding",
            "provider": "Local",
            "STEmbedding_path": "./paraphrase-multilingual-MiniLM-L12-v2/",
            "provider_type": "embedding",
            "enable": True,
            "embedding_dimensions": 384,
        }

        # 注册配置项描述
        CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"]["STEmbedding_path"] = {
            "description": "SentenceTransformer模型的路径",
            "type": "string",
        }

        st_in_use = False
        if isinstance(provider_registry, list):
            for pm in provider_registry:
                if hasattr(pm, "type") and pm.type == "STEmbedding":
                    st_in_use = True
        else:
            st_in_use = False

        if not st_in_use and sentence_transformers_available:
            try:
                # 保持适配器类在方法内部定义,为了防止ASTRbot的插件的类别检测检测到STEmbeddingProvider,然后导致插件无法动态导入
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

                        # 处理路径：如果是相对路径，则基于 data_dir；如果是绝对路径，直接使用
                        base_path = self.provider_config.get(
                            "STEmbedding_path",
                            "./paraphrase-multilingual-MiniLM-L12-v2/"
                        )
                        if os.path.isabs(base_path):
                            self.STEmbedding_path = base_path
                        else:
                            self.STEmbedding_path = os.path.join(
                                str(StarTools.get_data_dir()), base_path
                            )

                        # ⚠️ 关键点：不在 __init__ 中加载模型
                        self.model = None
                        self._model_lock = asyncio.Lock()

                        logger.info(
                            f"[STEmbedding] Provider 初始化完成，模型将延迟加载: {self.STEmbedding_path}"
                        )

                    async def _ensure_model_loaded(self) -> None:
                        """确保模型已加载（Lazy Loading，线程安全）"""
                        if self.model is not None:
                            return

                        async with self._model_lock:
                            # Double Check，防止并发重复加载
                            if self.model is not None:
                                return

                            loop = asyncio.get_running_loop()
                            logger.info(f"[STEmbedding] 开始加载模型: {self.STEmbedding_path}")

                            try:
                                from sentence_transformers import SentenceTransformer
                                self.model = await loop.run_in_executor(
                                    None, SentenceTransformer, self.STEmbedding_path
                                )
                                logger.info("[STEmbedding] 模型加载成功")
                            except ImportError:
                                logger.error(
                                    f"[STEmbedding] 模型加载失败,没有sentence_transformers包,请使用pip install sentence-transformers"
                                )
                                raise
                            except Exception as e:
                                logger.error(
                                    f"[STEmbedding] 模型加载失败: {e}", exc_info=True
                                )
                                raise

                    async def get_embedding(self, text: str) -> list[float]:
                        """获取单个文本的嵌入向量 - 异步执行"""
                        await self._ensure_model_loaded()

                        loop = asyncio.get_running_loop()
                        try:
                            embedding = await loop.run_in_executor(
                                None, self.model.encode, text
                            )
                            return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                        except Exception as e:
                            logger.error(
                                f"[STEmbedding] 获取嵌入向量失败: {e}", exc_info=True
                            )
                            raise

                    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
                        """获取多个文本的嵌入向量 - 异步执行"""
                        await self._ensure_model_loaded()

                        loop = asyncio.get_running_loop()
                        try:
                            embeddings = await loop.run_in_executor(
                                None, self.model.encode, texts
                            )
                            return embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)
                        except Exception as e:
                            logger.error(
                                f"[STEmbedding] 获取批量嵌入向量失败: {e}", exc_info=True
                            )
                            raise

                    def get_dim(self) -> int:
                        """获取嵌入向量的维度"""
                        return self.provider_config.get("embedding_dimensions", 384)
            except ValueError as e:
                logger.info(f"[STEmbedding] STEmbedding已经注册: {e}")
        else:
            logger.info("[STEmbedding] STEmbedding已经注册")

        # 保存适配器类的引用
        cls._registered = True

    @classmethod
    def _unregister_config(cls):
        """取消注册STEmbedding配置"""
        logger.info("[STEmbedding] 开始清理配置...")

        # 1. 从配置模板中移除STEmbedding
        config_template = CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["config_template"]
        removed_template = config_template.pop("STEmbedding", None)
        if removed_template:
            logger.debug("[STEmbedding] 已从配置模板中移除STEmbedding")

        # 2. 从配置项中移除STEmbedding_path
        config_items = CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"]
        removed_item = config_items.pop("STEmbedding_path", None)
        if removed_item:
            logger.debug("[STEmbedding] 已从配置项中移除STEmbedding_path")

        # 3. 重置状态
        cls._registered = False
        logger.info("[STEmbedding] 配置清理完成")

    @filter.command("STEmbedding")
    async def stembedding(self, event: AstrMessageEvent):
        """STEmbedding 命令处理器"""
        if not self.available:
            yield event.plain_result(
                "STEmbedding插件功能受限，请安装sentence-transformers库：pip install sentence-transformers"
            )
            return
        yield event.plain_result("你好，这是 STEmbedding 插件")

    @filter.command("cs")
    async def cs(self, event: AstrMessageEvent):
        """cs 命令处理器"""
        logger.info(f"provider_cls_map:{list(provider_cls_map.keys())}")

    async def initialize(self):
        """插件初始化方法 - 注册配置"""
        if not self.available:
            logger.warning("[STEmbedding] sentence-transformers库未安装，插件功能受限")
            return

        logger.info("[STEmbedding] 插件正在初始化，注册配置...")
        self._register_config()
        logger.info("[STEmbedding] 配置和适配器已注册")

    async def terminate(self):
        """插件终止方法 - 清理所有注册的配置"""
        logger.info("[STEmbedding] 插件正在终止，清理配置...")
        self._unregister_config()
        logger.info("[STEmbedding] 插件终止完成")
