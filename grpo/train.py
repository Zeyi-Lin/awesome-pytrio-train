"""PyTRIO + Hugging Face GSM8K + GRPO minimal training script.

Before running:
    python -m pip install -U pytrio transformers datasets numpy addict
    trio login

Quick smoke test:
    python grpo_gsm8k_hf.py --limit 8 --batch-size 2 --group-size 2 --max-tokens 128

Default run uses the first 200 train examples from:
    https://huggingface.co/datasets/openai/gsm8k
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any


BASE_MODEL = "Qwen/Qwen3.5-4B"
DATASET_ID = "openai/gsm8k"
SYSTEM_PROMPT = (
    "You are a careful math solver. Solve step by step, and put the final "
    "numeric answer inside \\boxed{}."
)
QUESTION_SUFFIX = "\n\nSolve the problem. The final answer must be written as \\boxed{number}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3.5-4B on GSM8K with GRPO via PyTRIO.")
    parser.add_argument("--base-model", default=os.getenv("TRIO_BASE_MODEL", BASE_MODEL))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--limit", type=int, default=200, help="Use the first N train examples.")
    parser.add_argument("--steps", type=int, default=None, help="Defaults to one pass over --limit.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights-name", default="qwen35-4b-gsm8k-grpo-hf-200")
    parser.add_argument("--eval-limit", type=int, default=16, help="Use 0 to skip eval.")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--eval-max-tokens", type=int, default=512)
    parser.add_argument("--eval-print-samples", type=int, default=3)
    parser.add_argument("--eval-concurrency", type=int, default=8)
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def load_runtime() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pytrio as trio
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖。请先运行：python -m pip install -U pytrio transformers datasets numpy"
        ) from exc

    return trio, np, load_dataset


def load_gsm8k(load_dataset: Any, limit: int, split: str = "train") -> list[dict[str, str]]:
    if limit <= 0:
        raise ValueError("--limit must be greater than 0")

    dataset = load_dataset(DATASET_ID, "main", split=f"{split}[:{limit}]")

    rows: list[dict[str, str]] = []
    for row in dataset:
        if "question" not in row or "answer" not in row:
            raise KeyError(f"Unexpected GSM8K row keys: {sorted(row)}")
        rows.append({"question": str(row["question"]), "answer": str(row["answer"])})
        if len(rows) >= limit:
            break

    if not rows:
        raise RuntimeError("GSM8K train split is empty")
    return rows


def render_prompt(tokenizer: Any, question: str) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question + QUESTION_SUFFIX},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return tokenizer.encode(text, add_special_tokens=False)


def gsm8k_gold(answer: str) -> str:
    match = re.search(r"####\s*(.+)", answer)
    if not match:
        raise ValueError(f"No GSM8K final answer found: {answer!r}")
    return match.group(1).strip()


def boxed_answer(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return matches[-1].strip() if matches else None


def last_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?(?:\d+(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?", text)
    return matches[-1] if matches else None


def parse_number(text: str) -> Decimal | None:
    text = text.replace(",", "").replace("$", "").strip().rstrip(".")
    text = re.sub(r"\s+", "", text)
    if re.fullmatch(r"[-+]?\d+/\d+", text):
        try:
            value = Fraction(text)
            return Decimal(value.numerator) / Decimal(value.denominator)
        except (ValueError, ZeroDivisionError, InvalidOperation):
            return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def score_completion(completion: str, gold: str) -> dict[str, Any]:
    boxed = boxed_answer(completion)
    answer = boxed or last_number(completion)
    if answer is None:
        return {"reward": 0.0, "exact": False, "answer": None, "boxed": False}

    pred_value = parse_number(answer)
    gold_value = parse_number(gold)
    exact = pred_value is not None and gold_value is not None and pred_value == gold_value

    if exact:
        reward_value = 1.0 if boxed else 0.85
    elif pred_value is not None and gold_value is not None:
        scale = max(abs(float(gold_value)), 1.0)
        rel_error = abs(float(pred_value - gold_value)) / scale
        reward_value = max(0.0, 0.45 * (1.0 - min(rel_error, 1.0)))
        if boxed:
            reward_value += 0.10
    else:
        reward_value = 0.10 if boxed else 0.0

    return {
        "reward": min(1.0, reward_value),
        "exact": exact,
        "answer": answer,
        "boxed": boxed is not None,
    }


def reward(completion: str, gold: str) -> float:
    return float(score_completion(completion, gold)["reward"])


def rollout_group(
    trio: Any,
    sampler: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    gold: str,
    params: Any,
    group_size: int,
) -> list[dict[str, Any]]:
    result = sampler.sample(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=group_size,
        sampling_params=params,
        return_text=True,
    ).result()

    samples = []
    rewards = []
    for seq in result.sequences:
        text = seq.text if seq.text is not None else tokenizer.decode(seq.tokens, skip_special_tokens=True)
        tokens = list(seq.tokens)
        logprobs = [float(x) for x in seq.logprobs]
        if not tokens or len(tokens) != len(logprobs):
            continue
        score = score_completion(text, gold)
        sample = {
            "tokens": tokens,
            "logprobs": logprobs,
            "text": text,
            **score,
        }
        samples.append(sample)
        rewards.append(float(score["reward"]))

    if not samples:
        return []

    mean_reward = sum(rewards) / len(rewards)
    for sample in samples:
        sample["advantage"] = sample["reward"] - mean_reward
    return samples


def build_datum(trio: Any, np: Any, prompt_tokens: list[int], sample: dict[str, Any]) -> Any:
    obs_len = len(prompt_tokens) - 1
    input_tokens = prompt_tokens + sample["tokens"][:-1]
    target_tokens = [0] * obs_len + sample["tokens"]
    old_logprobs = [0.0] * obs_len + sample["logprobs"]
    advantages = [0.0] * obs_len + [sample["advantage"]] * len(sample["tokens"])

    if not (len(input_tokens) == len(target_tokens) == len(old_logprobs) == len(advantages)):
        raise ValueError("GRPO datum fields are not aligned")

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(old_logprobs, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
        },
    )


def stop_sequences(tokenizer: Any) -> list[str]:
    return list(dict.fromkeys(x for x in [getattr(tokenizer, "eos_token", None), "<|im_end|>"] if x))


def preview(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def reward_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


async def evaluate_one(
    trio: Any,
    sampler: Any,
    tokenizer: Any,
    row: dict[str, str],
    index: int,
    params: Any,
    semaphore: asyncio.Semaphore,
) -> tuple[int, str, str, dict[str, Any]]:
    async with semaphore:
        gold = gsm8k_gold(row["answer"])
        prompt_tokens = render_prompt(tokenizer, row["question"])
        result = await sampler.sample_async(
            prompt=trio.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=params,
            return_text=True,
        )
        seq = result.sequences[0]
        text = seq.text if seq.text is not None else tokenizer.decode(seq.tokens, skip_special_tokens=True)
        return index, gold, text, score_completion(text, gold)


async def evaluate_async(
    trio: Any,
    service: Any,
    base_model: str,
    model_path: str,
    tokenizer: Any,
    rows: list[dict[str, str]],
    max_tokens: int,
    print_samples: int,
    concurrency: int,
) -> None:
    sampler = await service.create_sampling_client_async(
        base_model=base_model,
        model_path=model_path,
    )
    params = trio.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=stop_sequences(tokenizer),
    )
    semaphore = asyncio.Semaphore(concurrency)
    print(f"\nAsync eval on {len(rows)} examples | concurrency={concurrency}")
    if print_samples > 0:
        print(f"打印前{min(print_samples, len(rows))}条 eval 样本：")

    results = await asyncio.gather(
        *(
            evaluate_one(
                trio=trio,
                sampler=sampler,
                tokenizer=tokenizer,
                row=row,
                index=index,
                params=params,
                semaphore=semaphore,
            )
            for index, row in enumerate(rows, start=1)
        )
    )

    for index, gold, text, score in results:
        if index <= print_samples:
            print(
                f"[eval sample {index}] exact={score['exact']} reward={score['reward']:.3f} "
                f"answer={score['answer']!r} gold={gold}"
            )
            print(f"  {preview(text, 700)}")

    rewards = [float(score["reward"]) for _, _, _, score in results]
    exacts = [bool(score["exact"]) for _, _, _, score in results]
    exact_acc = sum(exacts) / len(exacts) if exacts else 0.0
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"Eval result | exact_acc={exact_acc:.3f} | reward={mean_reward:.3f}")


def format_metrics(metrics: dict[str, Any]) -> str:
    fields = ["loss_sum", "loss_mean", "token_count"]
    parts = []
    for field in fields:
        value = metrics.get(field, float("nan"))
        if isinstance(value, (int, float)):
            parts.append(f"{field}={float(value):.4f}")
        else:
            parts.append(f"{field}={value}")
    return " | ".join(parts)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")
    if args.group_size < 2:
        raise SystemExit("--group-size must be at least 2 for GRPO")
    if args.eval_concurrency <= 0:
        raise SystemExit("--eval-concurrency must be greater than 0")

    trio, np, load_dataset = load_runtime()
    np.random.seed(args.seed)

    service = trio.ServiceClient()
    if args.list_models:
        print(json.dumps(service.get_supported_models(), ensure_ascii=False, indent=2))

    print(f"Loading first {args.limit} examples from {DATASET_ID}...")
    rows = load_gsm8k(load_dataset, args.limit)

    trainer = service.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    tokenizer = trainer.get_tokenizer()
    sample_params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        stop=stop_sequences(tokenizer),
    )
    adam = trio.AdamParams(learning_rate=args.learning_rate)

    steps = args.steps or math.ceil(len(rows) / args.batch_size)
    for step in range(steps):
        start = (step * args.batch_size) % len(rows)
        batch = [rows[(start + offset) % len(rows)] for offset in range(args.batch_size)]

        sampler = trainer.save_weights_and_get_sampling_client()
        datums = []
        rewards = []
        exacts = []
        group_stds = []
        trainable_groups = 0
        skipped = 0

        for row in batch:
            prompt_tokens = render_prompt(tokenizer, row["question"])
            gold = gsm8k_gold(row["answer"])
            samples = rollout_group(
                trio=trio,
                sampler=sampler,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                gold=gold,
                params=sample_params,
                group_size=args.group_size,
            )

            rewards.extend(float(sample["reward"]) for sample in samples)
            exacts.extend(bool(sample["exact"]) for sample in samples)
            group_std = reward_std([float(sample["reward"]) for sample in samples])
            group_stds.append(group_std)
            if not samples or group_std <= 1e-12:
                skipped += 1
                continue
            trainable_groups += 1
            datums.extend(build_datum(trio, np, prompt_tokens, sample) for sample in samples)

        if datums:
            fwd = trainer.forward_backward(datums, loss_fn="importance_sampling")
            opt = trainer.optim_step(adam)
            metrics = fwd.result().metrics
            opt.result()
        else:
            metrics = {"loss_mean": float("nan")}

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        exact_acc = sum(exacts) / len(exacts) if exacts else 0.0
        mean_group_std = sum(group_stds) / len(group_stds) if group_stds else 0.0
        print(
            f"step {step + 1}/{steps} | reward={mean_reward:.3f} | exact={exact_acc:.3f} | "
            f"group_reward_std={mean_group_std:.3f} | trainable_groups={trainable_groups}/{len(batch)} | "
            f"skipped_uniform={skipped}/{len(batch)} | datums={len(datums)} | {format_metrics(metrics)}"
        )

    weights = trainer.save_weights_for_sampler(name=args.weights_name).result()
    print(f"Saved LoRA sampler weights: {weights.path}")

    if args.eval_limit > 0:
        eval_rows = load_gsm8k(load_dataset, args.eval_limit, split=args.eval_split)
        asyncio.run(
            evaluate_async(
                trio=trio,
                service=service,
                base_model=args.base_model,
                model_path=weights.path,
                tokenizer=tokenizer,
                rows=eval_rows,
                max_tokens=args.eval_max_tokens,
                print_samples=args.eval_print_samples,
                concurrency=args.eval_concurrency,
            )
        )


if __name__ == "__main__":
    main()
