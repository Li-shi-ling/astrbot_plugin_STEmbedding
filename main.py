from astrbot.core.provider.register import provider_registry, provider_cls_map
from astrbot.core.provider.register import register_provider_adapter
from astrbot.core.provider.provider import EmbeddingProvider
from astrbot.core.config.default import CONFIG_METADATA_2
from astrbot.core.provider.entities import ProviderType
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.star import StarTools
from astrbot.api import AstrBotConfig, logger
import os

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("未检测到 sentence-transformers 库, 请使用pip install sentence-transformers安装")
    raise ImportError("未检测到 sentence-transformers 库, 请使用pip install sentence-transformers安装")


@register("STEmbedding", "Lishining", "我的STEmbedding", "1.0.0")
class STEmbedding(Star):
    # 类属性
    _registered = False
    _init = False

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)

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

        if not cls._init:
            # 注册适配器
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

                    # 处理路径：如果是相对路径，则基于data_dir；如果是绝对路径，直接使用
                    base_path = self.provider_config.get("STEmbedding_path", "./paraphrase-multilingual-MiniLM-L12-v2/")
                    if os.path.isabs(base_path):
                        self.STEmbedding_path = base_path
                    else:
                        self.STEmbedding_path = os.path.join(str(StarTools.get_data_dir()), base_path)

                    logger.info(f"[STEmbedding] 正在加载模型: {self.STEmbedding_path}")
                    try:
                        self.model = SentenceTransformer(self.STEmbedding_path)
                        logger.info(f"[STEmbedding] 模型加载成功: {self.model}")
                    except Exception as e:
                        logger.error(f'[STEmbedding] 模型加载失败: {e}', exc_info=True)
                        raise

                async def get_embedding(self, text: str) -> list[float]:
                    """获取单个文本的嵌入向量"""
                    embedding = self.model.encode(text)
                    # 确保返回的是Python list而不是numpy array
                    return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

                async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
                    """获取多个文本的嵌入向量"""
                    embeddings = self.model.encode(texts)
                    # 确保返回的是Python list而不是numpy array
                    return embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)

                def get_dim(self) -> int:
                    """获取嵌入向量的维度"""
                    return self.provider_config.get("embedding_dimensions", 384)

        # 保存适配器类的引用
        cls._init = True
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

        # 4. 重置状态
        cls._registered = False
        logger.info("[STEmbedding] 配置清理完成")

    @filter.command("STEmbedding")
    async def stembedding(self, event: AstrMessageEvent):
        """STEmbedding 命令处理器"""
        yield event.plain_result("你好，这是 STEmbedding 插件")

    @filter.command("cs")
    async def cs(self, event: AstrMessageEvent):
        """cs 命令处理器"""
        logger.info(f"provider_cls_map:{list(provider_cls_map.keys())}")

    async def initialize(self):
        """插件初始化方法 - 注册配置"""
        logger.info("[STEmbedding] 插件正在初始化，注册配置...")
        self._register_config()
        logger.info("[STEmbedding] 配置和适配器已注册")

    async def terminate(self):
        """插件终止方法 - 清理所有注册的配置"""
        logger.info("[STEmbedding] 插件正在终止，清理配置...")

        # 检查是否还有其他插件在使用STEmbedding
        try:
            st_in_use = False

            # 遍历所有已注册的适配器，检查STEmbedding是否还被其他适配器使用
            for pm in provider_registry:
                if hasattr(pm, 'type') and pm.type == "STEmbedding":
                    # 如果找到STEmbedding适配器
                    st_in_use = True

            if not st_in_use and not self._registered:
                logger.info("[STEmbedding] 适配器未注册，无需清理")
                return

        except Exception as e:
            logger.warning(f"[STEmbedding] 检查适配器状态时出错: {e}")

        # 执行清理
        self._unregister_config()
        logger.info("[STEmbedding] 插件终止完成")
