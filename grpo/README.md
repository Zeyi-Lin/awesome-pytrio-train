# GRPO: GSM8k

## 安装环境

```bash
pip install -r requirements.txt
```

## 训练

训练50个step：

```bash
python train.py --steps 50 --batch-size 2 --group-size 16 --max-tokens 512 --eval-limit 0
```

## 评估

用base模型评估前100条：

```bash
python eval.py --limit 100
```

开16并发加速评估：

```bash
python eval.py --limit 100 --concurrency 16
```

用训练好的模型评估前100条：

```bash
python eval.py --checkpoint-path trio://run_xxx/sampler_weights/xxx --limit 100 --concurrency 16
```
