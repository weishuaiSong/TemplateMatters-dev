# multi_gpu_runner.py
import torch
import argparse
import os
import json
from multiprocessing import Process, set_start_method
from pathlib import Path
from tm.qa_models import ImageQAModel, build_prompt_func
from tqdm import tqdm

set_start_method("spawn", force=True)


def run_inference(args, full_dataset, start_idx, end_idx, output_file):
    """在指定 GPU 上处理数据集的一部分，并保存部分结果"""
    template_path = Path(args.template_path)
    with open(template_path) as f:
        templates = json.load(f)["MultiChoiceImageQa"]

    vqa_model = ImageQAModel(
        args.model_name, enable_choice_search=True,
        torch_device=args.torch_device, precision=torch.bfloat16
    )

    # 初始化每个模板的正确性列表
    num_templates = len(templates)
    template_correct = [[] for _ in range(num_templates)]

    # 处理指定范围内的样本
    for i in tqdm(range(start_idx, end_idx), desc=f"GPU {args.torch_device} Processing"):
        item = full_dataset[i]
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]
        image = item["image"]

        # 对每个模板进行预测
        for t_idx, template in enumerate(tqdm(templates, desc="Templates", leave=False)):
            prompt_func = build_prompt_func(template)
            result = vqa_model.multiple_choice_qa(
                image=image,
                question=question,
                choices=choices,
                answer=answer,
                prompt_func=prompt_func,
            )
            correct = result["accuracy"]
            template_correct[t_idx].append(correct)

    # 保存部分结果
    part_result = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "template_correct": {
            f"prompt_{t_idx+1}": template_correct[t_idx]
            for t_idx in range(num_templates)
        }
    }
    with open(output_file, "w") as f:
        json.dump(part_result, f, indent=2)


def calculate_split_indices(total_len, num_parts, part_idx):
    """计算每个部分的起始和结束索引"""
    part_len = total_len // num_parts
    start_idx = part_idx * part_len
    end_idx = total_len if part_idx == num_parts - 1 else start_idx + part_len
    return start_idx, end_idx


def run_on_gpu(rank, gpu_id, args, full_dataset, start_idx, end_idx):
    # 关键：在模型加载前指定当前进程使用的 GPU
    torch.cuda.set_device(gpu_id)
    args.torch_device = gpu_id   # 传递实际设备号给模型

    output_file = os.path.join(
        args.output_path,
        f"{args.model_name}_{args.dataset_name}_results_rank{rank}.json"
    )
    run_inference(args, full_dataset, start_idx, end_idx, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--template_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--gpu_ids", type=str, required=True, help="Comma separated GPU ids, e.g., 0,2,3")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    num_gpus = len(gpu_ids)

    from tm.qa_datasets import SingleImageQADataset
    full_dataset = SingleImageQADataset(args.dataset_name).get_dataset()
    total_len = len(full_dataset)

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # 启动多进程
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx, end_idx = calculate_split_indices(total_len, num_gpus, i)
        p = Process(target=run_on_gpu, args=(i, gpu_id, args, full_dataset, start_idx, end_idx))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # ---------- 合并所有部分结果 ----------
    # 读取所有部分文件
    part_files = []
    for i in range(num_gpus):
        fname = os.path.join(args.output_path, f"{args.model_name}_{args.dataset_name}_results_rank{i}.json")
        if os.path.exists(fname):
            with open(fname) as f:
                part_files.append(json.load(f))

    if not part_files:
        raise RuntimeError("No partial result files found.")

    # 从第一个部分文件中获取模板数量
    num_templates = len(part_files[0]["template_correct"])
    # 为每个模板准备总长度为 total_len 的列表，初始填充 None
    final_template_correct = {f"prompt_{t_idx+1}": [None] * total_len for t_idx in range(num_templates)}

    # 填充数据
    for part in part_files:
        start = part["start_idx"]
        end = part["end_idx"]
        for t_name, correct_list in part["template_correct"].items():
            # 确保列表长度与区间一致
            assert len(correct_list) == end - start
            final_template_correct[t_name][start:end] = correct_list

    # 检查是否所有位置都已填充（可选）
    for t_name, lst in final_template_correct.items():
        if any(v is None for v in lst):
            print(f"Warning: {t_name} has missing values.")

    # 构建最终输出结构
    final_output = {
        args.model_name: {
            args.dataset_name: final_template_correct
        }
    }

    # 保存最终合并结果
    final_output_file = os.path.join(
        args.output_path,
        f"{args.model_name}_{args.dataset_name}_results.json"
    )
    with open(final_output_file, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Results saved to {final_output_file}")