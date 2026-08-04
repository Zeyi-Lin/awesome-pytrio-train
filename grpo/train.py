"""PyTRIO + Hugging Face GSM8K + GRPO minimal training script.

Before running:
    python -m pip install -U pytrio transformers datasets numpy addict
    trio login

Quick smoke test:
    python train.py --steps 50 --batch-size 2 --group-size 16 --max-tokens 512

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
    parser.add_argument(
        "--rollout-concurrency",
        "--concurrency",
        dest="rollout_concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent rollout requests.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Save a full training state every N completed steps; the final step is always saved.",
    )
    parser.add_argument(
        "--weights-name",
        default=None,
        help=(
            "Optional saved-weight name prefix. Training hypestaterparameters, step, and artifact "
            "type are appended automatically."
        ),
    )
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


def load_gsm8k(
    load_dataset: Any,
    limit: int,
    split: str = "train",
) -> list[dict[str, str]]:
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


async def rollout_group(
    trio: Any,
    sampler: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    gold: str,
    params: Any,
    group_size: int,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    async with semaphore:
        result = await sampler.sample_async(
            prompt=trio.ModelInput.from_ints(prompt_tokens),
            num_samples=group_size,
            sampling_params=params,
            return_text=True,
        )

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


def reward_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _name_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _float_name(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace("+", "").replace(".", "p")


def build_weights_name(args: argparse.Namespace, step: int, artifact_type: str) -> str:
    if artifact_type not in {"train", "sampler"}:
        raise ValueError(f"Unsupported artifact type: {artifact_type}")

    model_name = args.base_model.rsplit("/", 1)[-1]
    prefix = _name_slug(args.weights_name or f"{model_name}-gsm8k-grpo")
    return "-".join(
        [
            prefix,
            f"r{args.rank}",
            f"n{args.limit}",
            f"bs{args.batch_size}",
            f"g{args.group_size}",
            f"c{args.rollout_concurrency}",
            f"mt{args.max_tokens}",
            f"t{_float_name(args.temperature)}",
            f"p{_float_name(args.top_p)}",
            f"lr{_float_name(args.learning_rate)}",
            f"s{args.seed}",
            f"si{args.save_interval}",
            f"step{step:06d}",
            artifact_type,
        ]
    )


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


async def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")
    if args.group_size < 2:
        raise SystemExit("--group-size must be at least 2 for GRPO")
    if args.rollout_concurrency <= 0:
        raise SystemExit("--rollout-concurrency must be greater than 0")
    if args.save_interval <= 0:
        raise SystemExit("--save-interval must be greater than 0")
    if args.steps is not None and args.steps <= 0:
        raise SystemExit("--steps must be greater than 0")

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
    rollout_semaphore = asyncio.Semaphore(args.rollout_concurrency)

    steps = args.steps or math.ceil(len(rows) / args.batch_size)
    for step in range(steps):
        start = (step * args.batch_size) % len(rows)
        batch = [rows[(start + offset) % len(rows)] for offset in range(args.batch_size)]

        sampler = await trainer.save_weights_and_get_sampling_client_async()
        datums = []
        rewards = []
        exacts = []
        group_stds = []
        trainable_groups = 0
        skipped = 0

        rollout_inputs = [
            (render_prompt(tokenizer, row["question"]), gsm8k_gold(row["answer"]))
            for row in batch
        ]
        rollout_results = await asyncio.gather(
            *(
                rollout_group(
                    trio=trio,
                    sampler=sampler,
                    tokenizer=tokenizer,
                    prompt_tokens=prompt_tokens,
                    gold=gold,
                    params=sample_params,
                    group_size=args.group_size,
                    semaphore=rollout_semaphore,
                )
                for prompt_tokens, gold in rollout_inputs
            )
        )

        for (prompt_tokens, _), samples in zip(rollout_inputs, rollout_results, strict=True):

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

        completed_step = step + 1
        if completed_step % args.save_interval == 0 or completed_step == steps:
            state_name = build_weights_name(args, completed_step, "train")
            state = trainer.save_state(name=state_name).result()
            print(f"Saved training state: name={state_name} | path={state.path}")

            sampler_name = build_weights_name(args, completed_step, "sampler")
            weights = trainer.save_weights_for_sampler(name=sampler_name).result()
            print(f"Saved LoRA sampler weights: name={sampler_name} | path={weights.path}")


if __name__ == "__main__":
    asyncio.run(main())
