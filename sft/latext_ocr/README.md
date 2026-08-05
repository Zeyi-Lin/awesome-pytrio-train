# 多模态 LaTeX OCR LoRA SFT

本示例使用 [PyTRIO](https://pytrio.cn) 在远程训练服务上对
`Qwen/Qwen3.5-4B` 进行多模态 LoRA 监督微调，使模型把公式图片转写为
LaTeX。训练和评测数据来自 Hugging Face 数据集
[`linxy/LaTeX_OCR`](https://huggingface.co/datasets/linxy/LaTeX_OCR) 的
`small` 子集。

本地只负责下载及处理数据、提交训练任务和记录实验，不需要本地 GPU。

## 功能

- 自动下载并缓存 `small` 子集的 Parquet 文件
- 将图片编码为 PyTRIO 多模态 `ImageChunk`
- 仅对目标 LaTeX 和结束标记计算 SFT loss
- 保存可供 `SamplingClient` 直接评测的 LoRA 权重
- 支持基座模型与微调后模型的并发评测

## 环境准备

建议使用 Python 3.10 或更高版本，安装依赖

```bash
cd sft/latext_ocr
pip install -r requirements.txt
```

首次使用 PyTRIO 时，需要先登录：

```bash
trio login
```

也可以直接提供 API Key：

```bash
trio login -k YOUR_API_KEY
```

## 快速试跑

先用少量样本检查数据、训练和权重保存流程：

```bash
python train.py \
  --max-samples 32 \
  --swanlab-mode disabled
```

脚本会自动把数据下载到 `data/LaTeX_OCR`。首次运行需要能够访问
Hugging Face；后续运行会复用本地 Parquet 文件。

## 训练

使用默认配置训练一个 epoch：

```bash
python train.py
```

自定义训练配置：

```bash
python train.py \
  --model Qwen/Qwen3.5-4B \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --rank 32 \
  --max-length 8192 \
  --checkpoint-name latex-ocr-small-sft
```

训练完成后，终端会打印类似下面的权重路径：

```text
saved_weights=YOUR_PYTRIO_WEIGHT_PATH
```

该路径由 `save_weights_for_sampler()` 生成，用于后续推理和评测；它不是
本地文件路径。

### 训练参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3.5-4B` | PyTRIO 基座模型 |
| `--epochs` | `1` | 训练轮数 |
| `--batch-size` | `1` | 每次远程更新的样本数 |
| `--learning-rate` | `1e-4` | Adam 学习率 |
| `--rank` | `32` | LoRA rank |
| `--max-samples` | `0` | 最多使用的样本数，`0` 表示全部 |
| `--max-length` | `8192` | 最大多模态序列长度，超长样本会被过滤 |
| `--dataset-dir` | `data/LaTeX_OCR` | 数据集缓存目录 |
| `--seed` | `42` | LoRA 初始化和数据打乱随机种子 |
| `--checkpoint-name` | `latex-ocr-small-sft` | 保存的推理权重名称 |

## 评测

### 评测基座模型

```bash
python eval.py --max-samples 30
```

### 评测微调后的模型

把训练输出的 `saved_weights` 路径传给 `--model-path`：

```bash
python eval.py \
  --model-path "YOUR_PYTRIO_WEIGHT_PATH" \
  --max-samples 30
```

默认关闭 Qwen 的思考模式。需要评测思考模式时使用：

```bash
python eval.py --enable-thinking --max-samples 30
```

评测脚本默认使用 `small/test`，以 16 个并发请求采样，并将逐样本结果写入
`latex_ocr_eval_results.jsonl`。终端会输出：

- `final_accuracy`：规范化 LaTeX 后的完全匹配率
- `avg_edit_similarity`：规范化 LaTeX 字符串的平均编辑相似度

### 评测参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3.5-4B` | 基座模型 |
| `--model-path` | 无 | PyTRIO 推理权重路径；不传则评测基座模型 |
| `--split` | `test` | `train`、`validation` 或 `test` |
| `--max-samples` | `0` | 最多评测的样本数，`0` 表示全部 |
| `--max-tokens` | `512` | 单条样本最大生成 token 数 |
| `--temperature` | `0.0` | 采样温度 |
| `--seed` | `42` | 采样随机种子 |
| `--concurrency` | `16` | 最大并发请求数 |
| `--enable-thinking` | 关闭 | 是否启用 Qwen 思考模式 |
| `--dataset-dir` | `data/LaTeX_OCR` | 数据集缓存目录 |
| `--output` | `latex_ocr_eval_results.jsonl` | JSONL 结果文件路径 |

## 文件说明

```text
latext_ocr/
├── train.py          # 多模态 LoRA SFT
├── eval.py           # 基座模型及微调权重评测
├── requirements.txt  # Python 依赖
└── README.md
```
