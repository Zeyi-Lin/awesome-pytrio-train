"""Evaluate a PyTRIO base or checkpoint model on Hugging Face GSM8K.

Before running:
    python -m pip install -U pytrio transformers datasets numpy
    trio login

Examples:
    python eval.py --limit 100
    python eval.py --limit 100 --concurrency 16
    python eval.py --checkpoint-path trio://run_xxx/sampler_weights/xxx
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from train import (
    BASE_MODEL,
    DATASET_ID,
    gsm8k_gold,
    load_gsm8k,
    render_prompt,
    score_completion,
    stop_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a base or checkpoint model on GSM8K.")
    parser.add_argument("--base-model", default=os.getenv("TRIO_BASE_MODEL", BASE_MODEL))
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional PyTRIO sampler checkpoint path, for example trio://run_xxx/sampler_weights/xxx.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=8)
    return parser.parse_args()


def load_runtime() -> tuple[Any, Any]:
    try:
        import pytrio as trio
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖。请先运行：python -m pip install -U pytrio transformers datasets numpy"
        ) from exc
    return trio, load_dataset


async def eval_one(
    trio: Any,
    sampler: Any,
    tokenizer: Any,
    row: dict[str, str],
    row_number: int,
    params: Any,
    semaphore: asyncio.Semaphore,
) -> tuple[int, bool]:
    async with semaphore:
        gold = gsm8k_gold(row["answer"])
        prompt_tokens = render_prompt(tokenizer, row["question"])
        result = await sampler.sample_async(
            prompt=trio.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=params,
            return_text=True,
        )
        sequence = result.sequences[0]
        text = (
            sequence.text
            if sequence.text is not None
            else tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        )
        score = score_completion(text, gold)
        return row_number, bool(score["exact"])


async def evaluate_async(
    trio: Any,
    sampler: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    max_tokens: int,
    concurrency: int,
) -> None:
    params = trio.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=stop_sequences(tokenizer),
    )
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            eval_one(
                trio=trio,
                sampler=sampler,
                tokenizer=tokenizer,
                row=row,
                row_number=index,
                params=params,
                semaphore=semaphore,
            )
        )
        for index, row in enumerate(rows, start=1)
    ]

    correct = 0
    completed = 0
    total = len(tasks)
    print(f"Start async eval: total={total}, concurrency={concurrency}")
    for task in asyncio.as_completed(tasks):
        row_number, is_correct = await task
        completed += 1
        correct += int(is_correct)
        running_acc = correct / completed
        print(
            f"eval {completed}/{total} | row={row_number} | "
            f"correct={is_correct} | acc={running_acc:.4f}"
        )

    final_acc = correct / total if total else 0.0
    print(f"Final | exact_acc={final_acc:.4f}")


async def main_async() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be greater than 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be greater than 0")

    trio, load_dataset = load_runtime()
    print(f"Loading first {args.limit} examples from {DATASET_ID}/{args.split}...")
    rows = load_gsm8k(load_dataset, args.limit, split=args.split)

    service = trio.ServiceClient()
    if args.checkpoint_path:
        print(f"Evaluating checkpoint: {args.checkpoint_path}")
        sampler = await service.create_sampling_client_async(
            base_model=args.base_model,
            model_path=args.checkpoint_path,
        )
    else:
        print(f"Evaluating base model: {args.base_model}")
        sampler = await service.create_sampling_client_async(base_model=args.base_model)
    tokenizer = sampler.get_tokenizer()
    await evaluate_async(
        trio=trio,
        sampler=sampler,
        tokenizer=tokenizer,
        rows=rows,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    asyncio.run(main_async())
