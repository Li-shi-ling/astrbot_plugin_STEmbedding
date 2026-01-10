# STEmbedding - AstrBot 嵌入向量生成插件

一个为 [AstrBot](https://github.com/your-repo/astrbot) 框架设计的 Sentence Transformers 嵌入向量生成插件，提供本地部署的文本嵌入功能。

## 功能特性

- 🚀 **本地化部署**: 使用 Sentence Transformers 模型在本地生成文本嵌入向量
- 🔧 **无缝集成**: 作为 AstrBot 的 Provider 适配器，可直接在框架配置中使用
- 📦 **自动配置注册**: 插件启动时自动注册配置项到 AstrBot 全局配置
- 🧹 **资源清理**: 插件卸载时自动清理注册的配置和适配器
- 🔌 **即插即用**: 简单的安装和配置流程

## 安装要求

### 系统依赖
- Python 3.10+
- AstrBot 框架

### Python 依赖
```bash
pip install sentence-transformers
```

插件会自动检查依赖，如果缺少 `sentence-transformers` 库，会在初始化时提示安装。

## 配置说明

### 自动注册的配置项
插件初始化时会自动向 AstrBot 注册以下配置：

```yaml
provider_group:
  metadata:
    provider:
      config_template:
        STEmbedding:
          id: "STEmbedding"
          type: "STEmbedding"
          provider: "Local"
          STEmbedding_path: "./paraphrase-multilingual-MiniLM-L12-v2/"
          provider_type: "embedding"
          enable: true
          embedding_dimensions: 384
      
      items:
        STEmbedding_path:
          description: "SentenceTransformer模型的路径"
          type: "string"
```

### 配置参数详解

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `STEmbedding_path` | string | `"./paraphrase-multilingual-MiniLM-L12-v2/"` | Sentence Transformer 模型路径，支持相对路径和绝对路径 |
| `embedding_dimensions` | integer | `384` | 嵌入向量的维度 |
| `enable` | boolean | `true` | 是否启用该 provider |
| `provider_type` | string | `"embedding"` | Provider 类型 |

### 路径说明
- **相对路径**: 相对于 AstrBot 的 `data_dir` 目录
- **绝对路径**: 直接使用指定的完整路径

## 使用方法

### 1. 作为嵌入向量提供者
在 AstrBot 配置文件中引用 STEmbedding：

```yaml
# config.yaml
embedding_provider:
  type: "STEmbedding"
  config:
    STEmbedding_path: "./your-model-directory/"
    embedding_dimensions: 384
```

### 2. 插件命令
插件提供以下命令：

#### `STEmbedding` 命令
```bash
# 测试插件是否正常工作
STEmbedding
```
响应: "你好，这是 STEmbedding 插件"

#### `cs` 命令
```bash
# 查看当前注册的 provider 类
cs
```
用于调试，查看所有已注册的 provider 类

### 3. 在代码中使用
```python
# 在 AstrBot 的其他插件中调用
provider = await self.get_provider("embedding")
embedding = await provider.get_embedding("你好，世界")
embeddings = await provider.get_embeddings(["文本1", "文本2", "文本3"])
dimensions = provider.get_dim()
```

## API 接口

### STEmbeddingProvider 类

#### 初始化
```python
def __init__(self, provider_config: dict, provider_settings: dict)
```
- `provider_config`: 提供者配置字典
- `provider_settings`: 提供者设置字典

#### 方法

##### `async get_embedding(text: str) -> list[float]`
获取单个文本的嵌入向量

**参数:**
- `text`: 输入文本字符串

**返回:**
- `list[float]`: 嵌入向量列表

##### `async get_embeddings(texts: list[str]) -> list[list[float]]`
获取多个文本的嵌入向量

**参数:**
- `texts`: 文本字符串列表

**返回:**
- `list[list[float]]`: 嵌入向量列表的列表

##### `get_dim() -> int`
获取嵌入向量的维度

**返回:**
- `int`: 向量维度

## 模型支持

### 预训练模型
插件默认使用 `paraphrase-multilingual-MiniLM-L12-v2` 模型，支持多种语言。

### 自定义模型
支持任何兼容的 Sentence Transformers 模型：

1. 从 [Hugging Face Model Hub](https://huggingface.co/models?library=sentence-transformers) 下载模型
2. 将模型放置在本地目录
3. 在配置中指定路径

### 推荐模型
- `paraphrase-multilingual-MiniLM-L12-v2` (默认，多语言，384维)
- `all-MiniLM-L6-v2` (英语，384维)
- `paraphrase-albert-small-v2` (英语，768维)
- `distiluse-base-multilingual-cased-v2` (多语言，512维)

## 开发说明

### 生命周期方法
- `initialize()`: 插件启动时调用，注册配置和适配器
- `terminate()`: 插件停止时调用，清理配置和适配器

### 日志
插件使用 AstrBot 的日志系统，日志前缀为 `[STEmbedding]`。

## 故障排除

### 常见问题

#### 1. 导入错误：缺少 sentence-transformers
```bash
# 安装依赖
pip install sentence-transformers
```

#### 2. 模型加载失败
- 检查模型路径是否正确
- 确认模型文件完整
- 检查磁盘空间和权限

#### 3. 配置未注册
- 确认插件已正确加载
- 检查插件初始化日志
- 重启 AstrBot

#### 4. 内存不足
- 使用更小的模型
- 增加系统内存
- 分批处理文本

### 日志级别
```python
# 查看详细日志
logger.setLevel("DEBUG")
```

## 版本历史

### v1.0.0
- 初始版本发布
- 支持 Sentence Transformers 模型
- 自动配置注册和清理
- 提供基本的嵌入向量生成功能

### v1.0.8
- 修改注册和清理方法,防止报错

### v1.0.9
- 修改编码方法为线程池

### v1.1.0 (当前版本)
- 修改模型读取代码

## 贡献指南

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 支持与联系

- 提交 Issue: [GitHub Issues](https://github.com/your-repo/astrbot-stembedding/issues)
- 文档: [项目 Wiki](https://github.com/your-repo/astrbot-stembedding/wiki)
- 讨论: [GitHub Discussions](https://github.com/your-repo/astrbot-stembedding/discussions)

## 致谢

- [Sentence Transformers](https://www.sbert.net/) - 用于生成嵌入向量
- [AstrBot](https://github.com/your-repo/astrbot) - 提供插件框架
- 所有贡献者和用户
