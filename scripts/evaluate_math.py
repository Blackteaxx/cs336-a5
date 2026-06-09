# 我们使用了 https://huggingface.co/datasets/garg-aayush/sft-cs336-assign5-datasets
# 使用 GPT-OSS 构造的数据集

from vllm import LLM, SamplingParams
from typing import Callable,  List
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import json

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truth: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    # generate outputs
    outputs = vllm_model.generate(
        prompts=prompts,
        sampling_params=eval_sampling_params,
    )

    # compute evaluation metrics
    rewards = [reward_fn(output.outputs[0].text, gt) for output, gt in zip(outputs, ground_truth)]

    # print avg evaluation rewards
    keys = rewards[0].keys()
    for key in keys:
        avg_reward = sum(reward[key] for reward in rewards) / len(rewards)
        print(f"Average Reward for {key}: {avg_reward}")

    # serialize results to disk
    results = {}
    per_instance_results = []
    for output, reward, gt in zip(outputs, rewards, ground_truth):
        per_instance_results.append({
            "prompt": output.prompt,
            "output": output.outputs[0].text,
            "ground_truth": gt,
            "reward": reward,
        })
    results["length"] = len(per_instance_results)
    results["average_rewards"] = {key: sum(reward[key] for reward in rewards) / len(rewards) for key in keys}
    results["per_instance_results"] = per_instance_results

    # --- 补全的内容开始 ---
    # print categories
    #  (1) correct with both format and answer reward 1,
    #  (2) format reward 1 and answer reward 0,
    #  (3) format reward 0 and answer reward 0?
    # Observing at least 10 cases where format reward is 0

    category_1_count = 0
    category_2_count = 0
    category_3_count = 0
    format_zero_cases = []

    for instance in per_instance_results:
        reward_dict = instance["reward"]
        # 假设 reward 字典中的键为 "format" 和 "answer"
        format_reward = reward_dict.get("format_reward", 0.0)
        answer_reward = reward_dict.get("answer_reward", 0.0)

        # 统计各个类别
        if format_reward == 1.0 and answer_reward == 1.0:
            category_1_count += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            category_2_count += 1
        elif format_reward == 0.0 and answer_reward == 0.0:
            category_3_count += 1

        # 收集 format reward 为 0 的案例
        if format_reward == 0.0:
            format_zero_cases.append(instance)

    # 打印分类统计结果
    print("\n" + "="*40)
    print("Evaluation Categories Summary:")
    print(f"(1) Format: 1, Answer: 1 -> {category_1_count} cases")
    print(f"(2) Format: 1, Answer: 0 -> {category_2_count} cases")
    print(f"(3) Format: 0, Answer: 0 -> {category_3_count} cases")
    print("="*40)

    # 打印至少 10 个 format reward 为 0 的观察案例（如果总数不足10个则打印全部）
    print(f"\nObserving {min(10, len(format_zero_cases))} cases where format reward is 0:")
    for i, case in enumerate(format_zero_cases[:10]):
        print(f"\n--- Case {i+1} ---")
        print(f"Prompt: {case['prompt']}")
        print(f"Output: {case['output']}")
        print(f"Ground Truth: {case['ground_truth']}")
        print(f"Reward: {case['reward']}")
    print("="*40 + "\n")
    # --- 补全的内容结束 ---

    with open("outputs/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)




# Read val dataset
with open("data/sft-cs336-assign5-datasets/sft-reason/val.jsonl", "r") as f:
    val_data = json.load(f)

# Read Prompt
with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
    prompt = f.read()

# load llm
llm = LLM(model="model/Qwen2.5-Math-1.5B", tensor_parallel_size=2)

# sampling params
eval_sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=1024,
    stop=["</answer>"],
    top_p=1.0,
    include_stop_str_in_output=True
)

# Prompts and Ground Truth
prompts = [prompt.format(question=item["problem"]) for item in val_data]
ground_truth = [item["expected_answer"] for item in val_data]

# evaluate
evaluate_vllm(
    vllm_model=llm,
    reward_fn=r1_zero_reward_fn,
    prompts=prompts,
    ground_truth=ground_truth,
    eval_sampling_params=eval_sampling_params
)
