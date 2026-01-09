from astrbot.core.provider.register import register_provider_adapter
from astrbot.core.provider.provider import EmbeddingProvider
from astrbot.core.config.default import CONFIG_METADATA_2
from astrbot.core.provider.entities import ProviderType
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.star import StarTools
from astrbot.api import AstrBotConfig
from astrbot.api import logger
import os
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("未检测到 sentence-transformers 库, 请使用pip install sentence-transformers安装")
    raise ImportError("未检测到 sentence-transformers 库, 请使用pip install sentence-transformers安装")

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
    "description": "SentenceTransformer模型的路径",
    "type": "string",
}

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
        self.STEmbedding_path = os.path.join(str(StarTools.get_data_dir()),self.provider_config.get("STEmbedding_path", "./paraphrase-multilingual-MiniLM-L12-v2/"))
        try:
            self.model = SentenceTransformer(self.STEmbedding_path)
        except Exception as e:
            logger.error(f'[STEmbedding] 模型加载失败 e:{e}', exc_info=True)

    async def get_embedding(self, text: str) -> list[float]:
        return self.model.encode(text)

    async def get_embeddings(self, text: list[str]) -> list[list[float]]:
        return list(self.model.encode(text))

    def get_dim(self) -> int:
        return 384

@register("STEmbedding", "Lishining", "STEmbedding", "1.0.0")
class STEmbedding(Star):
    def __init__(self, context: Context):
        super().__init__(context)


    @filter.command("STEmbedding")
    async def stembedding(self, event: AstrMessageEvent):
        yield event.plain_result("你好")
