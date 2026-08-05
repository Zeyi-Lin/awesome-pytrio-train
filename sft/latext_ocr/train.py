"""用 PyTRIO 在 LaTeX_OCR/small 上对 Qwen3.5-4B 做多模态 LoRA SFT。

运行：
    python train.py
    python train.py --epochs 3 --batch-size 2
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pytrio as trio
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoImageProcessor


DATASET_ID = "linxy/LaTeX_OCR"
IMAGE_PAD = "<|image_pad|>"
PROMPT = (
    "Transcribe the mathematical formula in this image into LaTeX. "
    "Output only the LaTeX."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0, help="0 表示全部")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--dataset-dir", default="data/LaTeX_OCR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-name", default="latex-ocr-small-sft")
    return parser.parse_args()


def load_train_dataset(dataset_dir: str):
    """首次运行时下载 small 子集，之后只读取本地 parquet。"""
    local_dir = Path(dataset_dir).expanduser().resolve()
    files = sorted((local_dir / "small").glob("train-*.parquet"))
    if not files:
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=["README.md", "small/*.parquet"],
        )
        files = sorted((local_dir / "small").glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"本地没有找到 train parquet：{local_dir}")

    return load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in files]},
        split="train",
    )


def encode_image(image: Image.Image, processor) -> trio.ImageChunk:
    """把 PIL 图片转换为 PyTRIO 的多模态 ImageChunk。"""
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    # TRIO 需要提前知道视觉编码器将产生多少个 token。
    patches = processor.get_number_of_image_patches(
        image.height,
        image.width,
        images_kwargs={},
    )
    return trio.ImageChunk(
        data=buffer.getvalue(),
        format="png",
        expected_tokens=patches // processor.merge_size**2,
    )


def process_example(example, tokenizer, processor) -> trio.Datum:
    """将一条 image/text 样本转换为只训练答案部分的 SFT Datum。"""
    image_chunk = encode_image(example["image"], processor)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image", "image": "formula"},
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # apply_chat_template 先生成图片占位符，再替换为真实 ImageChunk。
    parts = prompt.split(IMAGE_PAD)
    if len(parts) != 2:
        raise ValueError(f"期望 1 个图片占位符，实际得到 {len(parts) - 1} 个")
    before_image, after_image = parts
    prompt_chunks = [
        trio.types.EncodedTextChunk(
            tokens=tokenizer.encode(before_image, add_special_tokens=False)
        ),
        image_chunk,
        trio.types.EncodedTextChunk(
            tokens=tokenizer.encode(after_image, add_special_tokens=False)
        ),
    ]

    prompt_length = len(trio.ModelInput(chunks=prompt_chunks))
    completion = tokenizer.encode(
        str(example["text"]).strip() + "<|im_end|>",
        add_special_tokens=False,
    )

    # 自回归右移：prompt 最后一个位置开始预测 completion。
    model_input = trio.ModelInput(
        chunks=[
            *prompt_chunks,
            trio.types.EncodedTextChunk(tokens=completion[:-1]),
        ]
    )
    target_tokens = np.zeros(len(model_input), dtype=np.int64)
    weights = np.zeros(len(model_input), dtype=np.float32)
    start = prompt_length - 1
    target_tokens[start : start + len(completion)] = completion
    weights[start : start + len(completion)] = 1.0

    return trio.Datum(
        model_input=model_input,
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


def loss_per_token(result, batch: list[trio.Datum]) -> float:
    """使用服务返回的 logprobs 计算有监督 token 的平均 NLL。"""
    logprobs = np.concatenate(
        [output["logprobs"].tolist() for output in result.loss_fn_outputs]
    )
    weights = np.concatenate(
        [datum.loss_fn_inputs["weights"].tolist() for datum in batch]
    )
    return float(-np.dot(logprobs, weights) / weights.sum())


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size 必须大于 0")

    # 1. 下载并从本地加载 small/train 数据集。
    dataset = load_train_dataset(args.dataset_dir)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # 2. 与 TRIO 建立连接并创建 LoRA 训练客户端。
    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.model,
        rank=args.rank,
        seed=args.seed,
    )

    # 3. 获取与远程模型一致的 tokenizer 和图片处理器。
    tokenizer = training_client.get_tokenizer()
    processor_source = getattr(tokenizer, "name_or_path", args.model)
    image_processor = AutoImageProcessor.from_pretrained(
        processor_source,
        use_fast=False,
    )

    # 4. 数据集很小，一次性转换为 PyTRIO Datum，避免每个 epoch 重复编码。
    processed_examples = [
        process_example(example, tokenizer, image_processor) for example in dataset
    ]
    processed_examples = [
        datum
        for datum in processed_examples
        if len(datum.model_input) <= args.max_length
    ]
    if not processed_examples:
        raise RuntimeError("没有可训练样本，请检查 max-samples/max-length")

    # 5. 每个 epoch 打乱 Datum，然后按 batch_size 切片训练。
    step = 0
    for epoch in range(args.epochs):
        indices = np.random.default_rng(args.seed + epoch).permutation(
            len(processed_examples)
        )
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            batch = [processed_examples[index] for index in batch_indices]
            fwdbwd = training_client.forward_backward(batch, "cross_entropy")
            optim = training_client.optim_step(
                trio.AdamParams(learning_rate=args.learning_rate)
            )
            result = fwdbwd.result()
            optim.result()

            step += 1
            loss = loss_per_token(result, batch)
            print(
                f"epoch={epoch + 1} step={step} loss={loss:.4f}",
                flush=True,
            )

    # 7. 保存可直接传给 SamplingClient 的推理权重。
    saved = training_client.save_weights_for_sampler(
        name=args.checkpoint_name
    ).result()
    print(f"saved_weights={saved.path}")
    print(f"python latex_ocr_multimodal_eval.py --max-samples 30 --model-path {saved.path}")


if __name__ == "__main__":
    main()
