# 文本分类 LoRA SFT

[![visitors](https://komarev.com/ghpvc/?username=zeyi-lin-awesome-pytrio-train-text-classification&label=visitors&color=1283c3&style=flat)](https://github.com/Zeyi-Lin/awesome-pytrio-train/tree/main/sft/text_classification)

本示例使用 [PyTRIO](https://pytrio.cn) 在远程训练服务上对
`Qwen/Qwen3.5-4B` 进行文本分类 LoRA 监督微调。模型接收待分类文本和候选标签，
并且只能输出一个候选标签原文。

本地负责读取 JSONL、构造 assistant-only loss、提交训练以及对比基座模型和微调
模型的准确率；远程服务负责训练与采样，不需要本地 GPU。

## 功能

- 支持每条样本使用不同的候选标签集合
- 对超长文本进行 token 级截断并保留完整提示词结构
- prompt token 不参与 loss，只训练分类标签和结束标记
- 训练完成后立即对比基座模型与微调模型准确率
- 保存可供 `SamplingClient` 使用的 LoRA 推理权重
- 提供独立评测脚本，统计 accuracy lift、improved 和 regressed 样本数

## 环境准备

建议使用 Python 3.10 或更高版本，并在虚拟环境中安装依赖：

```bash
cd sft/text_classification
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

首次使用 PyTRIO 时，需要先登录：

```bash
trio login
```

也可以直接提供 API Key：

```bash
trio login -k YOUR_API_KEY
```

## 数据集

训练和测试数据来自 ModelScope 的
[`testUser/SFT-Text-Classification`](https://modelscope.cn/datasets/testUser/SFT-Text-Classification)。
如果当前目录中没有 `train.jsonl` 和 `test.jsonl`，可执行：

```bash
modelscope download \
  --dataset testUser/SFT-Text-Classification \
  train.jsonl \
  --local_dir ./

modelscope download \
  --dataset testUser/SFT-Text-Classification \
  test.jsonl \
  --local_dir ./
```

每行是一个 JSON 对象，使用以下字段：

```json
{
  "text": "待分类文本",
  "category": ["标签A", "标签B"],
  "output": "标签A"
}
```

`category` 也可以是单个字符串；`output` 必须是候选标签之一。

## 训练

```bash
python train.py
```

训练脚本当前使用文件顶部的 Python 常量配置，而不是 CLI 参数。默认配置为：

| 常量 | 默认值 | 说明 |
| --- | --- | --- |
| `base_model` | `Qwen/Qwen3.5-4B` | PyTRIO 基座模型 |
| `train_dataset_path` | `train.jsonl` | 训练集路径，相对于运行目录 |
| `test_dataset_path` | `test.jsonl` | 训练后验证集路径，相对于运行目录 |
| `batch_size` | `32` | 每次远程更新的样本数 |
| `epoch` | `1` | 训练轮数 |
| `train_size` | `200` | 使用的训练样本数 |
| `eval_size` | `20` | 训练完成后立即评测的样本数 |
| `max_seq_len` | `4096` | prompt 与 completion 的最大总长度 |

脚本中 LoRA rank 为 `32`，Adam 学习率为 `1e-4`，保存名称为
`text_classification`。需要调整这些配置时，修改 `train.py` 顶部常量或相应客户端参数。

训练完成后会打印类似下面的推理权重路径，并在同一次运行中输出基座模型与
微调模型的验证准确率：

```text
Saved Weights: YOUR_PYTRIO_WEIGHT_PATH
Eval Accuracy: ...
Base Accuracy: ...
```

`Saved Weights` 由 `save_weights_for_sampler()` 生成，用于后续评测；它不是本地
文件路径。

## 独立评测

把训练输出的权重路径传给 `--model-path`：

```bash
python eval.py \
  --model-path "YOUR_PYTRIO_WEIGHT_PATH" \
  --eval-size 100
```

查看每条样本的基座模型预测、微调模型预测和标签：

```bash
python eval.py \
  --model-path "YOUR_PYTRIO_WEIGHT_PATH" \
  --eval-size 100 \
  --verbose
```

评测脚本会输出：

- `Base Accuracy`：基座模型准确率
- `PyTRIO Weight Accuracy`：微调权重准确率
- `Accuracy Lift`：微调模型相对基座模型的准确率变化
- `Improved`：基座错误、微调正确的样本数
- `Regressed`：基座正确、微调错误的样本数

### 评测参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--base-model` | `Qwen/Qwen3.5-4B` | 基座模型，必须与训练权重兼容 |
| `--model-path` / `--weights-path` | 必填 | `save_weights_for_sampler()` 返回的权重路径 |
| `--dataset-path` | `test.jsonl` | JSONL 测试集路径 |
| `--eval-size` | `100` | 最多评测的样本数 |
| `--max-seq-len` | `4096` | prompt 与 generation 的最大总长度 |
| `--max-new-tokens` | `20` | 单条样本最大生成 token 数 |
| `--progress-every` | `1` | 每隔多少条打印进度，`0` 表示只打印最终结果 |
| `--seed` | `42` | 采样随机种子 |
| `--verbose` | 关闭 | 输出每条样本的预测和标签 |

## 文件说明

```text
text_classification/
├── train.py          # 文本分类 LoRA SFT 与训练后验证
├── eval.py           # 基座模型和微调权重对比评测
├── train.jsonl       # 训练数据
├── test.jsonl        # 测试数据
├── requirements.txt  # Python 依赖
└── README.md
```
