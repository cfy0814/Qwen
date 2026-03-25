# Qwen 系列开源大语言模型本地部署与应用

本项目基于官方案例实现**Qwen1.5/Qwen2.5/Qwen3** 轻量级模型的本地加载、推理及 ChatBot 网页应用搭建，涵盖模型基础使用、对话推理、Gradio 可视化 Web Demo 搭建全流程，清晰展示大语言模型本地部署的核心逻辑与实操步骤。

## 项目介绍

本项目以 Qwen1.5-0.5B、Qwen2.5-0.5B、Qwen3-0.6B 轻量级模型为核心（参数量均在 10 亿以内），通过本项目可掌握：

1. 开源大语言模型的本地权重加载、Tokenizer 配置方法
2. 基于 Transformers 的模型对话推理实现
3. 基于 Gradio+FastAPI 的 ChatBot 网页应用搭建逻辑
4. 不同版本 Qwen 模型的适配与切换技巧

## 环境配置

### 基础环境要求

- Python ≥ 3.8
- PyTorch ≥ 1.12（推荐 2.0 及以上版本，CPU/GPU 均支持）
- CUDA ≥ 11.4（仅 GPU 环境需要，CPU 环境可忽略）

## 模型权重下载

### 推荐方式：ModelScope 魔搭社区下载

安装`modelscope`后，通过代码一键下载指定模型，自动保存到本地，避免手动下载文件缺失：

```
from modelscope import snapshot_download

# 下载Qwen1.5-0.5B-Chat
model_dir = snapshot_download('qwen/Qwen1.5-0.5B-Chat', cache_dir='./')
# 下载Qwen2.5-0.5B-Instruct
model_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct', cache_dir='./')
# 下载Qwen3-0.6B
model_dir = snapshot_download('qwen/Qwen3-0.6B', cache_dir='./qwen')
```

- `cache_dir='./models'`：指定模型保存到项目根目录的`qwen`文件夹，可自行修改
- 下载完成后，模型路径示例：`./qwen/Qwen/Qwen1.5-0.5B-Chat`

## 快速开始

### 1. 模型基础推理

以 Qwen3-0.6B 为例，实现基础对话推理，Qwen1.5/2.5 仅需移除`trust_remote_code=True`并修改模型路径即可：

```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = r"D:\lianxi\Qwen\qwen\Qwen\Qwen3-0.6B"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型和Tokenizer（Qwen3需开启trust_remote_code=True）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True  # Qwen3专属，1.5/2.5无需添加
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True  # Qwen3专属，1.5/2.5无需添加
)

# 定义对话提示语
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "简单介绍一下你自己"}
]

# 格式化对话并推理
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
# 解码并输出结果
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("模型回复：\n", response)
```

### 2. 搭建 ChatBot 网页应用

基于 Gradio 实现可视化对话界面，运行后在浏览器访问即可实现交互式聊天，步骤如下：

1. 找到项目中的web_demo.py文件，修改模型默认路径

   ```
   # 替换为自己的本地模型路径（示例：Qwen3-0.6B）
   DEFAULT_CKPT_PATH = r"D:\lianxi\Qwen\qwen\Qwen\Qwen3-0.6B"
   ```

2. Qwen3 需在web_demo.py文件的模型加载部分添加trust_remote_code=True。（1.5/2.5 无需添加）

   ```
   model = AutoModelForCausalLM.from_pretrained(
       DEFAULT_CKPT_PATH,
       torch_dtype="auto",
       device_map="auto",
       trust_remote_code=True  # Qwen3专属
   ).eval()
   tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CKPT_PATH, trust_remote_code=True)
   ```

3. 运行 Web Demo：

   ```
   python web_demo.py
   ```

4. 浏览器访问地址（默认）：`http://localhost:7860`（或终端输出的其他地址），即可开始对话。

## 不同版本 Qwen 模型适配差异

|       模型版本        |  transformers 版本要求  |        核心加载参数差异         |   模型标识（ModelScope）   |
| :-------------------: | :---------------------: | :-----------------------------: | :------------------------: |
|   Qwen1.5-0.5B-Chat   |          ≥4.32          |   无需 trust_remote_code=True   |   qwen/Qwen1.5-0.5B-Chat   |
| Qwen2.5-0.5B-Instruct |          ≥4.32          |   无需 trust_remote_code=True   | qwen/Qwen2.5-0.5B-Instruct |
|      Qwen3-0.6B       | ≥4.51.0（推荐 4.57.0+） | 必须开启 trust_remote_code=True |      qwen/Qwen3-0.6B       |

**核心注意**：Qwen3 因采用新架构，Hugging Face Transformers 未原生集成，需通过`trust_remote_code=True`加载模型目录中的自定义代码，否则会报`KeyError: 'qwen3'`错误。

## ChatBot 构建逻辑

本项目的 ChatBot 由**模型权重服务**、**API 服务**、**Web 服务**三部分构成：

1. **模型权重服务**：加载本地模型权重，提供推理能力；
2. **API 服务**：基于 FastAPI 框架封装模型推理逻辑，提供可调用的接口；
3. **Web 服务**：基于 Gradio 框架实现前端可视化界面，包含对话输入框、发送按钮、历史记录等，无需前端开发知识，几行代码即可实现。

**用户交互流程**：用户在网页输入问题 → 前端将请求发送给 FastAPI 接口 → 接口调用本地模型进行推理 → 推理结果返回前端 → 展示给用户。

