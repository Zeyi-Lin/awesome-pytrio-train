import pytrio as trio
import numpy as np
import json

base_model = "Qwen/Qwen3.5-4B"
train_dataset_path = "train.jsonl"
test_dataset_path = "test.jsonl"
system_prompt = "你是一个严格的文本分类器。你必须从用户给出的候选标签中选择且只选择一个标签。最终回答只能包含候选标签原文，不要解释、不要复述文本、不要输出标点、不要输出 JSON。"
batch_size = 32
epoch = 1
train_size = 400
eval_size = 20
max_seq_len = 8192

# 1. 与TRIO建立连接
service_client = trio.ServiceClient()

# 2. 创建1个训练客户端
training_client = service_client.create_lora_training_client(
    base_model=base_model,
    rank=32,
)

# 3. 数据集构建-文本分类
def load_examples(path: str = "train.jsonl") -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                examples.append(json.loads(line))
    return examples

examples = load_examples(train_dataset_path)[:train_size]
eval_examples = load_examples(test_dataset_path)[:eval_size]

# 4. 获取Tokenizer
print("Loading tokenizer...")
tokenizer = training_client.get_tokenizer()
print("Tokenizer finish")

# 5. 处理数据集，转换为训练需要的格式
def render_prompt_parts(example: dict) -> tuple[str, str]:
    categories = example["category"]
    if isinstance(categories, list):
        categories = ", ".join(categories)

    prefix = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n文本："
    )
    suffix = (
        f"\n候选标签：{categories}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    return prefix, suffix


def encode_prompt(example: dict, tokenizer, max_tokens: int) -> list[int]:
    prefix, suffix = render_prompt_parts(example)
    prompt_tokens = tokenizer.encode(f"{prefix}{example['text']}{suffix}", add_special_tokens=False)
    if len(prompt_tokens) <= max_tokens:
        return prompt_tokens

    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    text_tokens = tokenizer.encode(example["text"], add_special_tokens=False)
    text_tokens = text_tokens[:max_tokens - len(prefix_tokens) - len(suffix_tokens)]
    return prefix_tokens + text_tokens + suffix_tokens


def process_example(example: dict, tokenizer) -> trio.Datum:
    completion_tokens = tokenizer.encode(f"{example['output']}<|im_end|>\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    prompt_tokens = encode_prompt(example, tokenizer, max_seq_len - len(completion_tokens))
    prompt_weights = [0] * len(prompt_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    # 转换为trio训练需要的格式
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "weights": np.asarray(weights, dtype=np.float32),
            "target_tokens": np.asarray(target_tokens, dtype=np.int32),
        },
    )


processed_examples = [process_example(ex, tokenizer) for ex in examples]

# 6. 训练
print("Start Training")
step = 0
steps_per_epoch = (len(processed_examples) + batch_size - 1) // batch_size
total_steps = epoch * steps_per_epoch
for ep in range(epoch):
    for start in range(0, len(processed_examples), batch_size):
        batch = processed_examples[start:start + batch_size]
        fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")  # 前向反向计算
        optim_future = training_client.optim_step(trio.AdamParams(learning_rate=1e-4))  # Adam优化器更新

        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        step += 1
        print(f"Epoch {ep+1}/{epoch} Step[{step}/{total_steps}] Loss per token: {fwdbwd_result.metrics['loss_mean']:2f}")

# 7. 验证集评估
print("Start Eval")
sft_weights = training_client.save_weights_for_sampler(name="text_classification").result()
print(f"Saved Weights: {sft_weights.path}")
sampling_base_client = service_client.create_sampling_client(base_model=base_model)
sampling_sft_client = service_client.create_sampling_client(
    base_model=base_model,
    model_path=sft_weights.path,
)
params = trio.SamplingParams(max_tokens=20, temperature=0.0)
correct = 0
base_correct = 0

def clean_prediction(text: str) -> str:
    text = text.split("<|im_end|>")[0].strip()
    return text.splitlines()[0].strip() if text else ""


for idx, example in enumerate(eval_examples, start=1):
    prompt = trio.ModelInput.from_ints(encode_prompt(example, tokenizer, max_seq_len - 20))
    future = sampling_sft_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
    base_future = sampling_base_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
    result = future.result()
    base_result = base_future.result()

    pred = clean_prediction(result.sequences[0].text)
    base = clean_prediction(base_result.sequences[0].text)
    label = example["output"].strip()
    correct += int(pred == label)
    base_correct += int(base == label)
    print(f"Eval {idx}/{len(eval_examples)} pred={repr(pred)} base={repr(base)} label={repr(label)}")

print(f"Eval Accuracy: {correct}/{len(eval_examples)} = {correct / len(eval_examples):.2%}")
print(f"Base Accuracy: {base_correct}/{len(eval_examples)} = {base_correct / len(eval_examples):.2%}")