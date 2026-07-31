# SFT: Text Classification

## 安装环境

```bash
pip install -r requirements.txt
```

## 下载数据集

```bash
modelscope download --dataset testUser/SFT-Text-Classification train.jsonl --local_dir ./
modelscope download --dataset testUser/SFT-Text-Classification test.jsonl --local_dir ./
```

## 训练

```bash
python train.py
```

## 评估

```bash
python eval.py --model-path YOUR_PYTRIO_WEIGHT_PATH
```