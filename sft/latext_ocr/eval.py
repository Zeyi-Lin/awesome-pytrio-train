"""评估 Qwen3.5-4B 在 LaTeX_OCR/small 上的多模态 OCR 效果。

评估基座模型：
    python eval.py

评估训练后权重：
    python eval.py --model-path "YOUR_SAVED_MODEL_PATH"

开启推理模式：
    python eval.py --enable-thinking
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
from pathlib import Path

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
    parser.add_argument(
        "--model-path",
        default=None,
        help="save_weights_for_sampler 返回的路径；不传则评估基座模型",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 表示全部")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否开启 Qwen 的推理模式（默认关闭）",
    )
    parser.add_argument("--dataset-dir", default="data/LaTeX_OCR")
    parser.add_argument("--output", default="latex_ocr_eval_results.jsonl")
    return parser.parse_args()


def download_and_load_split(split: str, dataset_dir: str):
    local_dir = Path(dataset_dir).expanduser().resolve()
    parquet_files = sorted((local_dir / "small").glob(f"{split}-*.parquet"))
    if not parquet_files:
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=["README.md", "small/*.parquet"],
        )
        parquet_files = sorted(
            (local_dir / "small").glob(f"{split}-*.parquet")
        )
    if not parquet_files:
        raise FileNotFoundError(f"本地没有找到 {split} parquet：{local_dir}")
    return load_dataset(
        "parquet",
        data_files={split: [str(path) for path in parquet_files]},
        split=split,
    )


def encode_image(image: Image.Image, processor) -> trio.ImageChunk:
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
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


def build_prompt(
    tokenizer,
    image_chunk: trio.ImageChunk,
    enable_thinking: bool,
) -> trio.ModelInput:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "formula"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    parts = rendered.split(IMAGE_PAD)
    if len(parts) != 2:
        raise ValueError(f"期望 1 个图片占位符，实际得到 {len(parts) - 1} 个")

    chunks: list[trio.types.EncodedTextChunk | trio.ImageChunk] = []
    for index, text in enumerate(parts):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if tokens:
            chunks.append(trio.types.EncodedTextChunk(tokens=tokens))
        if index == 0:
            chunks.append(image_chunk)
    return trio.ModelInput(chunks=chunks)


def strip_math_wrappers(text: str) -> str:
    text = text.strip()
    fenced = re.fullmatch(
        r"```(?:latex|tex|math)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        text = fenced.group(1).strip()

    wrappers = (
        ("$$", "$$"),
        (r"\[", r"\]"),
        (r"\(", r"\)"),
        ("$", "$"),
    )
    changed = True
    while changed:
        changed = False
        for left, right in wrappers:
            if text.startswith(left) and text.endswith(right):
                text = text[len(left) : -len(right)].strip()
                changed = True
                break
    return text


def clean_prediction(text: str) -> str:
    text = text.split("<|im_end|>", 1)[0]
    # 开启推理模式时，只使用 </think> 后的最终答案评估。
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return strip_math_wrappers(text)


def normalize_latex(text: str) -> str:
    text = strip_math_wrappers(text)
    # 这些命令只影响包裹、间距或字形写法，不改变公式内容。
    text = re.sub(r"\\(?:,|!|;|:|>| )", "", text)
    text = re.sub(
        r"\\(?:quad|qquad|enspace|thinspace|medspace|thickspace)\b",
        "",
        text,
    )
    text = re.sub(r"\\(?:left|right)\b", "", text).replace("~", "")
    aliases = {
        r"\widetilde": r"\tilde",
        r"\dfrac": r"\frac",
        r"\tfrac": r"\frac",
        r"\ast": "*",
        r"\bigtriangleup": r"\triangle",
        r"\lbrace": r"\{",
        r"\rbrace": r"\}",
    }
    for source, target in aliases.items():
        text = text.replace(source, target)

    # 统一 \mathcal{H} 与旧式 {\cal H}，并去掉重音命令外的冗余分组。
    text = re.sub(r"\\mathcal\s*\{\s*([A-Za-z])\s*\}", r"\\cal \1", text)
    text = re.sub(
        r"\{\s*(\\(?:tilde|hat|bar|vec)\s*\{\s*[^{}]+\s*\})\s*\}",
        r"\1",
        text,
    )
    return re.sub(r"\s+", "", text)


def edit_similarity(left: str, right: str) -> float:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(len(left), len(right), 1)


async def evaluate_sample(
    position: int,
    sample,
    sampler,
    tokenizer,
    image_processor,
    sampling_params: trio.SamplingParams,
    semaphore: asyncio.Semaphore,
    enable_thinking: bool,
) -> dict:
    reference = str(sample["text"]).strip()
    image_chunk = encode_image(sample["image"], image_processor)
    prompt = build_prompt(tokenizer, image_chunk, enable_thinking)

    async with semaphore:
        response = await sampler.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
        )

    prediction = clean_prediction(response.sequences[0].text)
    normalized_prediction = normalize_latex(prediction)
    normalized_reference = normalize_latex(reference)
    return {
        "index": position,
        "prediction": prediction,
        "reference": reference,
        "exact_match": prediction == reference,
        # 控制台中的 correct/accuracy 使用该指标，避免仅因空白差异误判。
        "normalized_exact_match": normalized_prediction == normalized_reference,
        "edit_similarity": edit_similarity(
            normalized_prediction,
            normalized_reference,
        ),
    }


async def main() -> None:
    args = parse_args()
    if args.concurrency <= 0:
        raise ValueError("concurrency 必须大于 0")

    dataset = download_and_load_split(args.split, args.dataset_dir)
    sample_count = len(dataset)
    if args.max_samples > 0:
        sample_count = min(sample_count, args.max_samples)

    sampler = await trio.ServiceClient().create_sampling_client_async(
        base_model=args.model,
        model_path=args.model_path,
    )
    tokenizer = sampler.get_tokenizer()
    processor_source = getattr(tokenizer, "name_or_path", args.model)
    image_processor = AutoImageProcessor.from_pretrained(
        processor_source,
        use_fast=False,
    )
    params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        stop="<|im_end|>",
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(
            evaluate_sample(
                position,
                dataset[position],
                sampler,
                tokenizer,
                image_processor,
                params,
                semaphore,
                args.enable_thinking,
            )
        )
        for position in range(sample_count)
    ]

    correct = 0
    similarity_sum = 0.0
    results = []
    # task 已全部启动并受 semaphore 限制并发；按列表顺序 await 只影响打印顺序。
    for completed, task in enumerate(tasks, start=1):
        record = await task
        results.append(record)
        correct += int(record["normalized_exact_match"])
        similarity_sum += record["edit_similarity"]
        print(
            f"sample={record['index'] + 1}/{sample_count} "
            f"correct={record['normalized_exact_match']} "
            f"accuracy={correct / completed:.2%}",
            flush=True,
        )

    output_path = Path(args.output).expanduser()
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in sorted(results, key=lambda item: item["index"]):
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    accuracy = correct / sample_count if sample_count else 0.0
    average_similarity = similarity_sum / sample_count if sample_count else 0.0
    print(
        f"final_accuracy={accuracy:.2%} ({correct}/{sample_count}) "
        f"avg_edit_similarity={average_similarity:.2%}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
