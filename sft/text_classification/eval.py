"""评估文本分类 SFT 权重相对 base model 的准确率提升。

运行示例：
python eval.py --model-path YOUR_PYTRIO_WEIGHT_PATH
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytrio as trio


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SCRIPT_DIR / "test.jsonl"
SYSTEM_PROMPT = (
    "你是一个严格的文本分类器。你必须从用户给出的候选标签中选择且只选择一个标签。"
    "最终回答只能包含候选标签原文，不要解释、不要复述文本、不要输出标点、不要输出 JSON。"
)


@dataclass(frozen=True)
class EvalCase:
    text: str
    categories: list[str]
    label: str


@dataclass(frozen=True)
class Prediction:
    raw: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTRIO 文本分类权重评估")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--model-path",
        "--weights-path",
        dest="model_path",
        required=True,
        help="PyTRIO 权重路径，即 save_weights_for_sampler 输出的 path",
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="每多少条样本打印一次中间结果；0 表示只打印最终结果",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.eval_size < 1:
        raise ValueError("--eval-size must be >= 1")
    if args.max_seq_len < 8:
        raise ValueError("--max-seq-len must be >= 8")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be >= 0")
    return args


def load_examples(path: Path, limit: int) -> list[EvalCase]:
    examples: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            categories = row["category"]
            if isinstance(categories, str):
                categories = [categories]
            examples.append(
                EvalCase(
                    text=row["text"],
                    categories=[str(category) for category in categories],
                    label=str(row["output"]).strip(),
                )
            )
            if len(examples) >= limit:
                break

    if not examples:
        raise ValueError(f"No examples loaded from {path}")
    return examples


def render_prompt_parts(example: EvalCase) -> tuple[str, str]:
    categories = ", ".join(example.categories)
    prefix = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n文本："
    )
    suffix = (
        f"\n候选标签：{categories}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    return prefix, suffix


def encode_prompt(tokenizer: Any, example: EvalCase, max_tokens: int) -> list[int]:
    prefix, suffix = render_prompt_parts(example)
    prompt_tokens = tokenizer.encode(f"{prefix}{example.text}{suffix}", add_special_tokens=False)
    if len(prompt_tokens) <= max_tokens:
        return list(prompt_tokens)

    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    text_budget = max_tokens - len(prefix_tokens) - len(suffix_tokens)
    if text_budget < 1:
        raise ValueError("--max-seq-len is too small for the classification prompt")

    text_tokens = tokenizer.encode(example.text, add_special_tokens=False)
    return list(prefix_tokens) + list(text_tokens[:text_budget]) + list(suffix_tokens)


def clean_generation(text: str) -> str:
    text = text.split("<|im_end|>", 1)[0].split("</s>", 1)[0]
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = text.replace("<think>", "")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = lines[0] if lines else text.strip()
    return text.strip(" \t\r\n'\"`。，、：:；;！!？?()[]{}<>")


def short_text(text: str, max_chars: int = 40) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= max_chars else f"{text[:max_chars - 3]}..."


def extract_label(text: str, candidates: list[str]) -> str:
    cleaned = clean_generation(text)
    normalized = cleaned.casefold()
    for candidate in candidates:
        if normalized == candidate.casefold():
            return candidate

    pattern = re.compile(
        "|".join(rf"(?<![A-Za-z]){re.escape(label)}(?![A-Za-z])" for label in candidates),
        re.IGNORECASE,
    )
    matches = pattern.findall(cleaned)
    if len(matches) == 1:
        for candidate in candidates:
            if matches[0].casefold() == candidate.casefold():
                return candidate
    return cleaned


def predict(
    sampling_client: Any,
    tokenizer: Any,
    params: trio.SamplingParams,
    example: EvalCase,
    max_seq_len: int,
) -> Prediction:
    prompt_tokens = encode_prompt(
        tokenizer,
        example,
        max_tokens=max_seq_len - (params.max_tokens or 0),
    )
    response = sampling_client.sample(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=params,
    ).result()
    raw = response.sequences[0].text
    return Prediction(raw=raw, label=extract_label(raw, example.categories))


def main(args: argparse.Namespace) -> None:
    examples = load_examples(args.dataset_path, args.eval_size)
    params = trio.SamplingParams(
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        temperature=0.0,
        stop=["<|im_end|>"],
    )

    print("Creating PyTRIO sampling clients...")
    service_client = trio.ServiceClient()
    base_client = service_client.create_sampling_client(base_model=args.base_model)
    tuned_client = service_client.create_sampling_client(
        base_model=args.base_model,
        model_path=args.model_path,
    )

    print("Loading tokenizer...")
    tokenizer = base_client.get_tokenizer()
    print(f"Start eval: {len(examples)} examples from {args.dataset_path}")

    total = len(examples)
    base_correct = 0
    tuned_correct = 0
    improved = 0
    regressed = 0

    for index, example in enumerate(examples, start=1):
        case_start = time.time()
        base_pred = predict(base_client, tokenizer, params, example, args.max_seq_len)
        tuned_pred = predict(tuned_client, tokenizer, params, example, args.max_seq_len)
        base_hit = base_pred.label == example.label
        tuned_hit = tuned_pred.label == example.label

        base_correct += int(base_hit)
        tuned_correct += int(tuned_hit)
        improved += int(not base_hit and tuned_hit)
        regressed += int(base_hit and not tuned_hit)

        if args.progress_every and (index == 1 or index == total or index % args.progress_every == 0):
            message = (
                f"Eval {index:03d}/{total} | "
                f"base {base_correct}/{index}={base_correct / index:.2%} | "
                f"tuned {tuned_correct}/{index}={tuned_correct / index:.2%} | "
                f"lift {tuned_correct - base_correct:+d} | "
                f"improved {improved} | regressed {regressed} | "
                f"{time.time() - case_start:.1f}s"
            )
            if args.verbose:
                message += (
                    f" | base_pred={short_text(base_pred.label)!r} | "
                    f"tuned_pred={short_text(tuned_pred.label)!r} | "
                    f"label={example.label!r}"
                )
            print(message, flush=True)

    base_acc = base_correct / total
    tuned_acc = tuned_correct / total
    delta = tuned_acc - base_acc

    print("#" * 50)
    print(f"Base Accuracy: {base_correct}/{total} = {base_acc:.2%}")
    print(f"PyTRIO Weight Accuracy: {tuned_correct}/{total} = {tuned_acc:.2%}")
    print(f"Accuracy Lift: {delta:+.2%} ({tuned_correct - base_correct:+d} examples)")
    print(f"Improved: {improved} | Regressed: {regressed}")
    print("#" * 50)


if __name__ == "__main__":
    start = time.time()
    main(parse_args())
    print(f"# eval cost {time.time() - start:.2f}s")
