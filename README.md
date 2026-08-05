![Awesome PyTRIO Train](./assets/cover.png)

<div align="center">
  <h1>Awesome PyTRIO Train</h1>
  <p>从 SFT 到 DPO、GRPO，用轻量代码跑通大模型后训练。</p>
  <p>
    <a href="https://pytrio.cn/"><img alt="PyTRIO" src="https://img.shields.io/badge/PyTRIO-Remote_Training-6C5CE7?style=flat" /></a>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-306998?style=flat" />
    <a href="https://github.com/Zeyi-Lin/awesome-pytrio-train/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Zeyi-Lin/awesome-pytrio-train?style=flat" /></a>
    <a href="https://github.com/Zeyi-Lin/awesome-pytrio-train"><img alt="visitors" src="https://komarev.com/ghpvc/?username=zeyi-lin-awesome-pytrio-train&amp;label=visitors&amp;color=1283c3&amp;style=flat" /></a>
  </p>
</div>

## 这个仓库是什么？

这是一个面向实践的 [PyTRIO](https://pytrio.cn/) 大模型后训练案例集。目标不是只展示
API，而是用完整、可运行的任务把数据处理、prompt、loss、reward、训练循环、实验记录、
权重保存和评测串起来。

仓库主要做三件事：

- **把训练流程讲清楚**：从文本与多模态 SFT，逐步走到偏好优化和在线强化学习。
- **提供可直接运行的代码**：每个示例都有独立依赖、训练脚本、评测脚本和详细 README。
- **降低实验门槛**：训练、采样和权重保存由 PyTRIO 远程服务完成，本地不需要 GPU。

本地 Python 专注于数据和实验逻辑，PyTRIO 负责远程模型计算：

| 本地负责 | PyTRIO 远程服务负责 |
| --- | --- |
| 数据下载与清洗、prompt 构造、loss/reward 逻辑、训练循环、指标记录 | 模型前向与反向传播、优化器更新、并发采样、LoRA 训练、权重与训练状态保存 |

## 学习路线

建议按下面的顺序阅读和运行示例：

1. **文本分类 SFT**：理解自回归右移、assistant-only loss mask 和基础 LoRA 训练闭环。
2. **多模态 LaTeX OCR SFT**：在 SFT 基础上加入图片编码和多模态 `ModelInput`。
3. **HH-RLHF DPO**：学习 chosen/rejected 偏好数据、reference logprob 和本地 custom loss。
4. **GSM8K GRPO**：完成 rollout、reward、group-relative advantage 与在线策略更新。

## 示例目录

| 阶段 | 示例 | 数据集 | 核心内容 |
| --- | --- | --- | --- |
| SFT | [文本分类 LoRA SFT](./sft/text_classification/README.md) | ModelScope `testUser/SFT-Text-Classification` | 标签生成、assistant-only loss、训练前后准确率对比 |
| SFT | [多模态 LaTeX OCR LoRA SFT](./sft/latext_ocr/README.md) | Hugging Face `linxy/LaTeX_OCR` | `ImageChunk`、多模态 prompt、并发 OCR 评测 |
| DPO | [HH-RLHF DPO](./dpo/README.md) | ModelScope `Anthropic/hh-rlhf` | chosen/rejected、reference model、`forward_backward_custom` |
| GRPO | [GSM8K GRPO](./grpo/README.md) | Hugging Face `openai/gsm8k` | group rollout、reward、advantage、`importance_sampling` |

## 快速启动

### 1. 克隆仓库

```bash
git clone https://github.com/Zeyi-Lin/awesome-pytrio-train.git
cd awesome-pytrio-train
```

### 2. 进入一个示例并安装依赖

每个示例使用独立的 `requirements.txt`。以 GRPO 为例：

```bash
cd grpo
pip install -r requirements.txt
```

### 3. 登录 PyTRIO

```bash
trio login
```

也可以直接提供 API Key：

```bash
trio login -k YOUR_API_KEY
```

CLI 命令是 `trio`。登录完成后，Python 代码使用 `import pytrio as trio`。

### 4. 运行示例

在对应示例目录中运行：

```bash
# 文本分类 SFT
python train.py

# LaTeX OCR SFT 快速试跑
python train.py --max-samples 32 --swanlab-mode disabled

# DPO 快速试跑
python train.py --steps 2 --batch-size 1 --sample-size 32 --swanlab-mode disabled

# GRPO 快速试跑
python train.py --steps 2 --limit 8 --batch-size 2 --group-size 4 --max-tokens 256
```

上面的命令需要分别在 `sft/text_classification`、`sft/latext_ocr`、`dpo` 和
`grpo` 目录运行。完整的数据准备、训练参数、评测方式和输出说明请查看对应 README。

## 训练闭环

四个示例遵循相同的 PyTRIO 基础流程：

1. 准备数据，并转换为 `trio.ModelInput` / `trio.Datum`。
2. 通过 `trio.ServiceClient()` 创建训练或采样客户端。
3. 提交远程前向、反向与优化器更新任务。
4. 使用 `save_weights_for_sampler()` 保存推理权重。
5. 用 `SamplingClient` 评测基座模型或训练后的权重。

不同训练方法的主要区别在于监督信号：

| 方法 | 监督信号 | PyTRIO 训练入口 |
| --- | --- | --- |
| SFT | 目标 assistant token 与 loss weights | `forward_backward(..., "cross_entropy")` |
| DPO | chosen/rejected 的相对偏好 | `forward_backward_custom(...)` |
| GRPO | rollout reward 对应的 group-relative advantage | `forward_backward(..., "importance_sampling")` |

## 仓库结构

```text
awesome-pytrio-train/
├── sft/
│   ├── text_classification/  # 文本分类 SFT
│   └── latext_ocr/           # 多模态 LaTeX OCR SFT
├── dpo/                      # HH-RLHF DPO
├── grpo/                     # GSM8K GRPO
├── assets/                   # README 图片资源
└── README.md
```

每个示例目录都尽量保持自包含，方便直接运行，也方便把单个案例复制到自己的项目中改造。

## 相关链接

- [PyTRIO 官网](https://pytrio.cn/)
- [PyTRIO 文档](https://docs.pytrio.cn/)

## 贡献

欢迎通过 [Issues](https://github.com/Zeyi-Lin/awesome-pytrio-train/issues) 提交问题、训练记录和
案例建议，也欢迎通过 Pull Request 补充新的 SFT、偏好优化、蒸馏或 Agentic RL 示例。

如果这个仓库对你有帮助，欢迎点一个 Star，让更多人更轻松地开始大模型后训练实验。
