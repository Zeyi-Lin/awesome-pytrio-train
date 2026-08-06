# GSM8K GRPO

[![visitors](https://komarev.com/ghpvc/?username=zeyi-lin-awesome-pytrio-train-grpo&label=visitors&color=1283c3&style=flat)](https://github.com/Zeyi-Lin/awesome-pytrio-train/tree/main/grpo)

本示例使用 [PyTRIO](https://pytrio.cn) 在远程训练服务上对
`Qwen/Qwen3.5-4B` 进行 LoRA GRPO（Group Relative Policy Optimization），
训练和评测数据来自 Hugging Face
[`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) 的 `main` 配置。

模型需要逐步求解数学题，并把最终数值答案写入 `\boxed{}`。本地负责数据加载、
并发 rollout、reward 与 group-relative advantage 计算；远程服务负责采样和训练，
不需要本地 GPU。

## 功能

- 自动从 Hugging Face 下载并缓存 GSM8K
- 对同一道题并发采样多个 completion
- 根据答案正确性、数值误差和 `\boxed{}` 格式计算 reward
- 跳过组内 reward 完全相同、没有训练信号的 group
- 使用 `importance_sampling` loss 更新远程 LoRA
- 定期保存完整训练状态和 sampler 推理权重
- 支持异步并发评测基座模型或训练后权重

## 环境准备

建议使用 Python 3.10 或更高版本，并在虚拟环境中安装依赖：

```bash
cd grpo
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

```bash
python train.py \
  --steps 2 \
  --limit 8 \
  --batch-size 2 \
  --group-size 4 \
  --max-tokens 256
```

首次运行需要能够访问 Hugging Face；后续运行会复用本地 datasets 缓存。

## 训练

默认读取前 200 条训练数据。未指定 `--steps` 时，训练步数为
`ceil(limit / batch_size)`：

```bash
python train.py
```

训练 50 个 step，并提高每题的 rollout 数量：

```bash
python train.py \
  --steps 50 \
  --batch-size 2 \
  --group-size 16 \
  --rollout-concurrency 8 \
  --max-tokens 512
```

每到 `--save-interval` 以及最后一个 step，脚本会保存两类产物：

```text
Saved training state: ... path=YOUR_TRAINING_STATE_PATH
Saved LoRA sampler weights: ... path=YOUR_SAMPLER_WEIGHT_PATH
```

- `save_state()` 生成完整训练状态，用于断点续训场景。
- `save_weights_for_sampler()` 生成 sampler 权重，可传给 `eval.py --checkpoint-path`。

### 训练参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--base-model` | `Qwen/Qwen3.5-4B` | PyTRIO 基座模型；也可用 `TRIO_BASE_MODEL` 环境变量设置 |
| `--rank` | `16` | LoRA rank |
| `--limit` | `200` | 使用的 GSM8K train 样本数 |
| `--steps` | 自动计算 | 优化步数；不传时遍历所选样本一轮 |
| `--batch-size` | `4` | 每步 prompt 数量 |
| `--group-size` | `4` | 每个 prompt 的 rollout 数，必须至少为 2 |
| `--rollout-concurrency` / `--concurrency` | `8` | 最大并发 rollout 请求数 |
| `--max-tokens` | `512` | 单条 rollout 最大生成 token 数 |
| `--temperature` | `1.0` | rollout 采样温度 |
| `--top-p` | `1.0` | nucleus sampling 参数 |
| `--learning-rate` | `4e-5` | Adam 学习率 |
| `--seed` | `42` | 采样与 NumPy 随机种子 |
| `--save-interval` | `50` | 每隔多少个完成的 step 保存一次；最后一步始终保存 |
| `--weights-name` | 自动生成 | 保存名称前缀，脚本会附加超参数、step 和产物类型 |
| `--list-models` | 关闭 | 训练前打印 PyTRIO 支持的模型列表 |

## Reward 规则

回答首先尝试提取最后一个 `\boxed{...}`，否则使用最后一个数字：

- 数值完全正确且使用 `\boxed{}`：reward 为 `1.0`
- 数值完全正确但没有 `\boxed{}`：reward 为 `0.85`
- 数值不正确但可解析：按相对误差给予部分 reward
- 使用 `\boxed{}` 但答案错误：额外获得少量格式 reward
- 无法提取数值：reward 为 `0.0`

每条 completion 的 advantage 为其 reward 减去同组平均 reward。组内 reward
完全相同时 advantage 全为零，该组会被跳过。

## 评测

### 评测基座模型

```bash
python eval.py --limit 100
```

提高并发数：

```bash
python eval.py --limit 100 --concurrency 16
```

### 评测训练后的模型

`--checkpoint-path` 需要传入训练输出中的 sampler 权重路径：

```bash
python eval.py \
  --checkpoint-path "YOUR_SAMPLER_WEIGHT_PATH" \
  --limit 100 \
  --concurrency 16
```

评测采用温度为 0 的确定性采样，并输出：

- `exact_acc`：数值答案完全正确率
- `wrong_answer`：正常结束但答案错误的样本数
- `max_tokens_truncated`：生成达到 token 上限的样本数

### 评测参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--base-model` | `Qwen/Qwen3.5-4B` | 基座模型；也可用 `TRIO_BASE_MODEL` 环境变量设置 |
| `--checkpoint-path` | 无 | sampler 权重路径；不传则评测基座模型 |
| `--limit` | `100` | 评测样本数 |
| `--split` | `test` | GSM8K 数据划分 |
| `--max-tokens` | `512` | 单条样本最大生成 token 数 |
| `--concurrency` | `8` | 最大并发评测请求数 |

## 文件说明

```text
grpo/
├── train.py          # 异步 rollout 与 GRPO 训练
├── eval.py           # 基座模型及 sampler 权重评测
├── requirements.txt  # Python 依赖
└── README.md
```
