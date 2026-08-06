# HH-RLHF DPO

[![visitors](https://komarev.com/ghpvc/?username=zeyi-lin-awesome-pytrio-train-dpo&label=visitors&color=1283c3&style=flat)](https://github.com/Zeyi-Lin/awesome-pytrio-train/tree/main/dpo)

本示例使用 [PyTRIO](https://pytrio.cn) 在远程训练服务上对
`Qwen/Qwen3.5-4B` 进行 LoRA DPO（Direct Preference Optimization），偏好数据来自
[`Anthropic/hh-rlhf`](https://modelscope.cn/datasets/Anthropic/hh-rlhf)。

脚本会把每条数据解析为共享 prompt 下的 `chosen` / `rejected` 回复，使用基座模型
计算 reference logprob，再通过 `forward_backward_custom()` 在本地计算 DPO loss。
本地不需要 GPU，但 custom loss 依赖本地 PyTorch。

## 功能

- 自动从 ModelScope 下载并缓存 HH-RLHF 子集
- 校验 chosen/rejected 是否共享同一个对话上下文
- 只在最终 assistant 回复 token 上计算偏好损失
- 使用基座模型作为固定 reference model
- 通过 PyTRIO custom loss 更新远程 LoRA
- 使用 SwanLab 记录 DPO loss、偏好准确率和 reward margin
- 保存可供 `SamplingClient` 使用的推理权重

## 环境准备

建议使用 Python 3.10 或更高版本，并在虚拟环境中安装依赖：

```bash
cd dpo
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

## 数据集

默认使用 HH-RLHF 的 `helpful-base/train`，首次运行时自动下载到
`datasets/hh-rlhf`，之后复用本地缓存。可用子集包括：

- `helpful-base`
- `helpful-online`
- `helpful-rejection-sampled`
- `harmless-base`

`red-team-attempts` 不是 chosen/rejected 偏好数据，因此不支持用于本示例训练。

## 快速试跑

先用少量数据验证下载、reference logprob、custom loss 和权重保存流程：

```bash
python train.py \
  --steps 2 \
  --batch-size 1 \
  --sample-size 32 \
  --swanlab-mode disabled
```

## 训练

使用默认配置训练 10 个 step：

```bash
python train.py
```

较完整的训练示例：

```bash
python train.py \
  --base-model Qwen/Qwen3.5-4B \
  --steps 100 \
  --batch-size 16 \
  --sample-size 5000 \
  --max-length 1024 \
  --dpo-beta 0.1 \
  --learning-rate 1e-5 \
  --swanlab-mode online
```

同时训练多个 HH-RLHF 子集：

```bash
python train.py \
  --dataset-subsets helpful-base harmless-base \
  --sample-size 5000
```

训练完成后会打印类似下面的推理权重路径：

```text
Saved weights: YOUR_PYTRIO_WEIGHT_PATH
```

该路径由 `save_weights_for_sampler()` 生成，用于后续创建 PyTRIO
`SamplingClient`；它不是本地文件路径。

### 训练参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset-repo` | `Anthropic/hh-rlhf` | ModelScope 数据集仓库 |
| `--dataset-revision` | `master` | 数据集 revision |
| `--dataset-dir` | `datasets/hh-rlhf` | 本地数据与 datasets 缓存目录 |
| `--dataset-subsets` | `helpful-base` | 一个或多个 HH-RLHF 子集 |
| `--split` | `train` | `train` 或 `test` |
| `--force-download` | 关闭 | 强制重新下载数据文件 |
| `--sample-size` | `1000` | 随机抽样数量，`<=0` 表示全部 |
| `--seed` | `42` | 数据打乱和 LoRA 初始化随机种子 |
| `--base-model` | `Qwen/Qwen3.5-4B` | PyTRIO 基座及 reference 模型 |
| `--lora-rank` | `32` | LoRA rank |
| `--steps` | `10` | 优化步数 |
| `--batch-size` | `2` | 每步 preference pair 数量 |
| `--max-length` | `2048` | chosen/rejected 完整序列最大长度 |
| `--dpo-beta` | `0.1` | DPO 偏好强度系数 |
| `--learning-rate` | `1e-5` | Adam 学习率 |
| `--beta1` | `0.9` | Adam beta1 |
| `--beta2` | `0.95` | Adam beta2 |
| `--adam-eps` | `1e-8` | Adam epsilon |
| `--enable-thinking` | 关闭 | 是否启用模型 chat template 的思考模式 |
| `--save-weights-name` | `dpo-hh-rlhf-qwen35-4b-sync` | 推理权重名称 |
| `--swanlab` / `--no-swanlab` | 开启 | 是否启用 SwanLab |
| `--swanlab-project` | `dpo-hh-rlhf-pytrio` | SwanLab 项目名 |
| `--swanlab-name` | `dpo-qwen35-4b-sync` | SwanLab 实验名 |
| `--swanlab-workspace` | 无 | SwanLab workspace |
| `--swanlab-mode` | SwanLab 默认值 | `online`、`local`、`offline` 或 `disabled` |

## 训练指标

终端与 SwanLab 主要记录：

- `dpo/loss`：pairwise DPO loss
- `dpo/accuracy`：当前策略对 chosen 的偏好比例
- `dpo/margin`：chosen 与 rejected 的平均 reward margin
- `dpo/chosen_reward`、`dpo/rejected_reward`：两类回复的隐式 reward
- `data/pairs`、`data/tokens`：当前 step 的有效样本和 token 数

## 文件说明

```text
dpo/
├── train.py          # 同步 HH-RLHF DPO 训练
├── requirements.txt  # Python 依赖
└── README.md
```
