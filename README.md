# STEmbedding - AstrBot 嵌入向量生成插件

一个为 [AstrBot](https://github.com/AstrBotDevs/astrbot) 框架设计的 Sentence Transformers 嵌入向量生成插件，提供本地部署的文本嵌入功能。

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

```json
{
  "provider_group": {
    "metadata": {
      "provider": {
        "config_template": {
          "STEmbedding": {
            "id": "STEmbedding",
            "type": "STEmbedding",
            "provider": "Local",
            "STEmbedding_path": "./paraphrase-multilingual-MiniLM-L12-v2/",
            "STEmbedding_torch_load_weights_only": "auto",
            "provider_type": "embedding",
            "enable": true,
            "embedding_dimensions": 384
          }
        },
        "items": {
          "STEmbedding_path": {
            "description": "SentenceTransformer模型的路径",
            "type": "string"
          },
          "STEmbedding_torch_load_weights_only": {
            "description": "torch.load 的 weights_only（auto/true/false）。PyTorch 2.6+ 默认 true，部分旧模型需 false（仅可信模型）",
            "type": "string"
          }
        }
      }
    }
  }
}
```

### 配置参数详解

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `STEmbedding_path` | string | `"./paraphrase-multilingual-MiniLM-L12-v2/"` | Sentence Transformer 模型路径，支持相对路径和绝对路径 |
| `STEmbedding_torch_load_weights_only` | string | `"auto"` | `torch.load` 的 `weights_only` 行为：`auto/true/false`。PyTorch 2.6+ 若遇到 `Weights only load failed` 可设为 `false`（仅可信模型） |
| `embedding_dimensions` | integer | `384` | 嵌入向量的维度 |
| `enable` | boolean | `true` | 是否启用该 provider |
| `provider_type` | string | `"embedding"` | Provider 类型 |

### 路径说明
- **相对路径**: 相对于 AstrBot 的 `data_dir` 目录
- **绝对路径**: 直接使用指定的完整路径

## 使用方法

### 1. 作为嵌入向量提供者
可以在插件配置里面选择开机自启
第一次需要用户手动操作启动,使用指令`/ste register`将提供商注册到嵌入向量提供者
然后可以通过嵌入式模型的创建页面创建这个嵌入向量提供者

### 2. 插件命令
使用 /ste help 获取帮助

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

#### 2.1 PyTorch 2.6 `Weights only load failed`
当你看到类似错误：`Weights only load failed ... weights_only ...`，说明当前模型权重文件不兼容 PyTorch 2.6 默认的 `weights_only=True` 安全加载。

- 推荐：在 Provider 配置中设置 `STEmbedding_torch_load_weights_only` 为 `false`（仅可信模型）
- 或者：升级/更换为支持 `safetensors` 的模型与依赖组合

#### 3. 配置未注册
- 确认插件已正确加载
- 检查插件初始化日志
- 重启 AstrBot

#### 4. 内存不足
- 使用更小的模型
- 增加系统内存
- 分批处理文本

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

### v1.1.0
- 修改模型读取代码

### v1.1.1
- 修改模型读取代码为异步,更改第三方包检测代码

### v1.1.2
- 添加重新加载数据库的代码,防止找不到本地编码器导致的数据库加载失败

### v1.1.4
- 完全重构代码,保持逻辑的一致性

### v1.2.0
- 修改卸载参数为异步方法

### v1.2.3
- 修改刷新数据库逻辑移动到astrbot初始化后

### v1.2.4 (当前版本)
- 兼容torch 2.6的weights_only

## 支持与联系

- 提交 Issue: [GitHub Issues](https://github.com/Li-shi-ling/astrbot_plugin_STEmbedding/issues)
- QQ群: 1083090761

## 致谢

- [Sentence Transformers](https://www.sbert.net/) - 用于生成嵌入向量
- [AstrBot](https://github.com/AstrBotDevs/AstrBot) - 提供插件框架

[![Moe Counter](https://count.getloli.com/get/@li-shi-ling?theme=minecraft)](https://github.com/Li-shi-ling/astrbot_plugin_STEmbedding)
