import json
from tqdm import tqdm
from tm.qa_models import ImageQAModel, build_prompt_func
from tm.qa_datasets import SingleImageQADataset
from pathlib import Path
import argparse
import torch

# Qwen-style 模型使用 messages + process_vision_info + apply_chat_template 做 logprob 打分
QWEN_STYLE_MODELS = {"qwenvl", "qwen2.5vl", "qwen2.5omni", "qwen3vl", "qwenvl-chat"}
# InternVL：HF 版本有 processor（可算 POSIX），chat 版本无 processor（仅准确率）
INTERNVL_STYLE_MODELS = {"internvl-chat-v1.5", "internvl2.5-8b", "internvl3-8b", "molmo2-8b"}
# Molmo 7B-D：使用 processor.process(images, text)，需单独 logprob 路径
MOLMO_STYLE_MODELS = {"molmo-7b-d-0924"}

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="llavav1.5-7b", help="Model name to load")
parser.add_argument("--dataset_name", type=str, default="tma-subset", help="Dataset name")
parser.add_argument("--template_path", type=str, default="TemplateMatters/templates/heldout_templates.json", help="Path to template JSON file")
parser.add_argument("--output_path", type=str, default="posix_results", help="Output directory for results")
args = parser.parse_args()

template_path = Path(args.template_path)
with open(template_path) as f:
    templates = json.load(f)["MultiChoiceImageQa"]

vqa_model = ImageQAModel(
    args.model_name, enable_choice_search=True, torch_device=0, precision=torch.bfloat16
)
dataset = SingleImageQADataset(args.dataset_name).get_dataset()

results = []
use_qwen_style = args.model_name in QWEN_STYLE_MODELS
use_internvl_style = args.model_name in INTERNVL_STYLE_MODELS
use_molmo_style = args.model_name in MOLMO_STYLE_MODELS

for item in tqdm(dataset, desc="Processing dataset"):
    question = item["question"]
    choices = item["choices"]
    answer = item["answer"]
    image = item["image"]

    responses = []
    logprobs = []
    corrects = []  # 每个模板下该题是否答对

    for template in tqdm(templates, desc="Generating responses", leave=False):
        prompt_func = build_prompt_func(template)
        result = vqa_model.multiple_choice_qa(
            image=image,
            question=question,
            choices=choices,
            answer=answer,
            prompt_func=prompt_func,
        )
        responses.append(result.get("free_form_answer", ""))
        logprobs.append(result.get("log_probability", result.get("log_prob", 0.0)))
        # 有 answer 时 multiple_choice_qa 返回 accuracy (0/1)，否则用预测与标准答案比较
        pred = result.get("multiple_choice_answer")
        corrects.append(result.get("accuracy", 1 if pred == answer else 0))

    N = len(templates)
    logprob_matrix = []

    if use_internvl_style and hasattr(vqa_model.model, "processor"):
        # InternVL HF（processor + apply_chat_template），与 Qwen 类似的 logprob 打分
        processor = vqa_model.model.processor
        model = vqa_model.model.model
        for i in tqdm(range(N), desc="Scoring templates", leave=False):
            row = []
            prompt_i = build_prompt_func(templates[i])(question, choices)
            messages_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_i},
                    ],
                }
            ]
            prompt_inputs = processor.apply_chat_template(
                messages_prompt,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            _device = next(model.parameters()).device
            _dtype = next(model.parameters()).dtype
            prompt_inputs = {k: v.to(_device) if v is not None else v for k, v in prompt_inputs.items()}
            if "pixel_values" in prompt_inputs and prompt_inputs["pixel_values"] is not None:
                prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].to(_dtype)
            prompt_length = prompt_inputs["input_ids"].shape[1]

            for j in range(N):
                response_j = responses[j].strip()
                messages_full = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_i},
                        ],
                    },
                    {"role": "assistant", "content": response_j},
                ]
                inputs = processor.apply_chat_template(
                    messages_full,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(_device) if v is not None else v for k, v in inputs.items()}
                if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                    inputs["pixel_values"] = inputs["pixel_values"].to(_dtype)
                seq_len = inputs["input_ids"].shape[1]
                if seq_len <= prompt_length:
                    print("WARNING: no response tokens detected! prompt_len:", prompt_length, "full_len:", seq_len)

                model_kwargs = {k: v for k, v in inputs.items() if v is not None}
                with torch.no_grad():
                    outputs = model(**model_kwargs)
                log_probs = torch.log_softmax(outputs.logits, dim=-1)[0]
                total_log_prob = 0.0
                for k in range(seq_len - prompt_length):
                    pred_pos = prompt_length + k - 1
                    token_pos = prompt_length + k
                    token_id = inputs["input_ids"][0, token_pos].item()
                    total_log_prob += log_probs[pred_pos, token_id].item()
                row.append(total_log_prob)
            logprob_matrix.append(row)
    elif use_internvl_style:
        # InternVL chat 版（OpenGVLab remote-code）：用 teacher-forcing 计算 logprob 矩阵
        for i in tqdm(range(N), desc="Scoring templates", leave=False):
            row = []
            prompt_i = build_prompt_func(templates[i])(question, choices)
            for j in range(N):
                row.append(vqa_model.model.logprob_of_response(image=image, prompt=prompt_i, response=responses[j]))
            logprob_matrix.append(row)
    elif use_molmo_style:
        # Molmo 7B-D：processor.process(images, text)，teacher-forcing 计算 logprob
        processor = vqa_model.model.processor
        model = vqa_model.model.model
        _device = next(model.parameters()).device
        molmo_image = image.convert("RGB") if hasattr(image, "mode") and image.mode != "RGB" else image

        for i in tqdm(range(N), desc="Scoring templates", leave=False):
            row = []
            prompt_i = build_prompt_func(templates[i])(question, choices)
            prompt_inputs = processor.process(images=[molmo_image], text=prompt_i)
            prompt_inputs = {k: (v.to(_device).unsqueeze(0) if v is not None else v) for k, v in prompt_inputs.items()}
            prompt_length = prompt_inputs["input_ids"].shape[1]

            for j in range(N):
                response_j = responses[j].strip()
                full_text = prompt_i + response_j
                inputs = processor.process(images=[molmo_image], text=full_text)
                inputs = {k: (v.to(_device).unsqueeze(0) if v is not None else v) for k, v in inputs.items()}
                seq_len = inputs["input_ids"].shape[1]
                if seq_len <= prompt_length:
                    print("WARNING: no response tokens detected! prompt_len:", prompt_length, "full_len:", seq_len)

                model_kwargs = {k: v for k, v in inputs.items() if v is not None}
                with torch.no_grad():
                    outputs = model(**model_kwargs)
                log_probs = torch.log_softmax(outputs.logits, dim=-1)[0]
                total_log_prob = 0.0
                for k in range(seq_len - prompt_length):
                    pred_pos = prompt_length + k - 1
                    token_pos = prompt_length + k
                    token_id = inputs["input_ids"][0, token_pos].item()
                    total_log_prob += log_probs[pred_pos, token_id].item()
                row.append(total_log_prob)
            logprob_matrix.append(row)
    elif use_qwen_style:
        from qwen_vl_utils import process_vision_info

        processor = vqa_model.model.processor
        model = vqa_model.model.model

        for i in tqdm(range(N), desc="Scoring templates", leave=False):
            row = []
            prompt_i = build_prompt_func(templates[i])(question, choices)
            messages_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_i},
                    ],
                }
            ]
            prompt_text = processor.apply_chat_template(
                messages_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages_prompt)
            prompt_inputs = processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            prompt_length = prompt_inputs.input_ids.shape[1]

            for j in range(N):
                response_j = responses[j].strip()
                full_text = prompt_text + response_j
                inputs = processor(
                    text=[full_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                seq_len = inputs.input_ids.shape[1]

                if seq_len <= prompt_length:
                    print("WARNING: no response tokens detected! prompt_len:", prompt_length, "full_len:", seq_len)

                model_kwargs = dict(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                )
                if hasattr(inputs, "pixel_values"):
                    model_kwargs["pixel_values"] = inputs.pixel_values
                if hasattr(inputs, "image_grid_thw"):
                    model_kwargs["image_grid_thw"] = inputs.image_grid_thw

                with torch.no_grad():
                    outputs = model(**model_kwargs)
                log_probs = torch.log_softmax(outputs.logits, dim=-1)[0]
                total_log_prob = 0.0
                for k in range(seq_len - prompt_length):
                    pred_pos = prompt_length + k - 1
                    token_pos = prompt_length + k
                    token_id = inputs.input_ids[0, token_pos].item()
                    total_log_prob += log_probs[pred_pos, token_id].item()
                row.append(total_log_prob)
            logprob_matrix.append(row)
    else:
        # LLaVA-style: USER/ASSISTANT prompt, processor(text=..., images=image)
        for i in tqdm(range(N), desc="Scoring templates", leave=False):
            row = []
            for j in range(N):
                current_prompt = build_prompt_func(templates[i])(question, choices)
                response = responses[j]
                full_prompt = "USER: <image>\n" + current_prompt + "\nASSISTANT:"
                full_text = full_prompt + response

                inputs = vqa_model.model.processor(
                    text=full_text,
                    images=image,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(vqa_model.model.model.device)
                prompt_inputs = vqa_model.model.processor(
                    text=full_prompt,
                    images=image,
                    return_tensors="pt",
                )
                prompt_length = prompt_inputs.input_ids.shape[1]

                with torch.no_grad():
                    outputs = vqa_model.model.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        pixel_values=inputs.pixel_values,
                    )
                log_probs = torch.log_softmax(outputs.logits, dim=-1)[0]
                total_log_prob = 0.0
                for k in range(inputs.input_ids.shape[1] - prompt_length):
                    pred_position = prompt_length + k - 1
                    token_position = prompt_length + k
                    token_id = inputs.input_ids[0, token_position].item()
                    total_log_prob += log_probs[pred_position, token_id].item()
                row.append(total_log_prob)
            logprob_matrix.append(row)

    psi = 0.0
    for i in range(N):
        for j in range(N):
            psi += abs(logprob_matrix[j][i] - logprob_matrix[j][j]) / 200
    posix = psi / (N * (N - 1))

    results.append({
        "question": question,
        "answer": answer,
        "posix": posix,
        "responses": responses,
        "corrects": corrects,
        "logprob_matrix": logprob_matrix,
    })

all_posix_values = [entry["posix"] for entry in results]
average_posix = sum(all_posix_values) / len(all_posix_values)

# 准确率：每题、每模板的 corrects[i]；按模板聚合得每个模板的准确率
N = len(templates)
num_samples = len(results)
per_template_correct = [sum(entry["corrects"][i] for entry in results) for i in range(N)]
per_template_accuracy = [c / num_samples for c in per_template_correct] if num_samples else []
average_accuracy = sum(entry["corrects"][i] for entry in results for i in range(N)) / (num_samples * N) if num_samples and N else 0.0
best_template_accuracy = max(per_template_accuracy) if per_template_accuracy else 0.0
worst_template_accuracy = min(per_template_accuracy) if per_template_accuracy else 0.0
template_accuracy_gap = best_template_accuracy - worst_template_accuracy

output_dir = Path(args.output_path)
output_dir.mkdir(parents=True, exist_ok=True)
filename = f"{args.model_name}_{args.dataset_name}_average_posix.json"
save_path = output_dir / filename

summary = {
    "average_posix": average_posix,
    "average_accuracy": average_accuracy,
    "best_template_accuracy": best_template_accuracy,
    "worst_template_accuracy": worst_template_accuracy,
    "template_accuracy_gap": template_accuracy_gap,
    "per_template_accuracy": per_template_accuracy,
}
with open(save_path, "w") as f:
    json.dump(summary, f, indent=2)

# 可选：保存每条样本的详细结果（含 corrects），文件名加 _details
details_path = output_dir / f"{args.model_name}_{args.dataset_name}_posix_details.json"
with open(details_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Summary saved to: {save_path}")
print(f"  average_posix: {average_posix:.4f}")
print(f"  average_accuracy: {average_accuracy:.4f}")
print(f"  best_template_accuracy: {best_template_accuracy:.4f}, worst: {worst_template_accuracy:.4f}, gap: {template_accuracy_gap:.4f}")
print(f"Details saved to: {details_path}")
