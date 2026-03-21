import tempfile
from typing import Union
import torch
from PIL import Image
from transformers import image_utils

from .base_qa_model import QAModelInstance, QAModel
from .utils import image_to_base64, load_image
import torch.nn.functional as F
from .base_qa_model import QAModel

imageqa_models = {

    "llavav1.5-7b": ("LLaVA", "llava-hf/llava-1.5-7b-hf"),
    "llavav1.5-13b": ("LLaVA", "llava-hf/llava-1.5-13b-hf"),
    "llavav1.6-7b": ("LLaVA", "llava-hf/llava-v1.6-vicuna-7b-hf"),
    "llavav1.6-13b": ("LLaVA", "llava-hf/llava-v1.6-vicuna-13b-hf"),
    "qwenvl": ("QwenVL", "Qwen/Qwen-VL"),
	"qwen2.5vl": ("Qwen25VL", "Qwen/Qwen2.5-VL-7B-Instruct"),
	"qwen2.5omni": ("Qwen25Omni", "Qwen/Qwen2.5-Omni-7B"),
	"qwen3vl"  : ("Qwen3VL", "Qwen/Qwen3-VL-8B-Instruct"),
    "qwenvl-chat": ("QwenVLChat", "Qwen/Qwen-VL-Chat"),
    "internvl-chat-v1.5": ("InternVLChat", 'failspy/InternVL-Chat-V1-5-quantable'),
    # InternVL2.5 目前为 OpenGVLab remote-code 的 InternVLChatConfig（非 InternVLConfig），不能用 AutoModelForImageTextToText；
    # POSIX 通过 InternVLChat.logprob_of_response 的 teacher-forcing forward 计算。
    "internvl2.5-8b": ("InternVLChat", "OpenGVLab/InternVL2_5-8B"),
    "internvl3-8b": ("InternVLChat", "OpenGVLab/InternVL3-8B"),
    "idefics2-8b": ("IDEFICS2", "HuggingFaceM4/idefics2-8b"),
    "molmo-7b-d-0924": ("Molmo", "allenai/Molmo-7B-D-0924"),
    "molmo2-8b": ("Molmo2", "allenai/Molmo2-8B"),
    "llava-onevision-qwen2-7b-ov-hf": ("LlavaOnevision", "llava-hf/llava-onevision-qwen2-7b-ov-hf"),
    "llava-onevision-1.5-8b-instruct": ("LLaVAOneVision15", "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"),

    "llavav1.5-7b-10-templated": ('LLaVA', "/mnt/ali-sh-1/usr/tusen/tmp-dev/shijian/template-scaling/LLaVA/checkpoints/hf_models/llava-v1.5-7b-lora-10-templated"),
    "llavav1.5-7b-100-templated": ('LLaVA', "shijianS01/llava-v1.5-7b-lora-100-templated"),
    "llavav1.5-7b-1k-templated": ('LLaVA', "shijianS01/llava-v1.5-7b-lora-1k-templated"),
    "llavav1.5-7b-5k-templated": ('LLaVA', "shijianS01/llava-v1.5-7b-lora-5k-templated"),
    "llavav1.5-7b-10k-templated": ('LLaVA', "shijianS01/llava-v1.5-7b-lora-10k-templated"),
    "llavav1.5-7b-15k-templated": ('LLaVA', "shijianS01/llava-v1.5-7b-lora-15k-templated"),

    "llavav1.5-13b-10-templated": ('LLaVA', "/mnt/ali-sh-1/usr/tusen/tmp-dev/shijian/template-scaling/LLaVA/checkpoints/hf_models/llava-v1.5-13b-lora-10-templated"),
    "llavav1.5-13b-100-templated": ('LLaVA', "shijianS01/llava-v1.5-13b-lora-100-templated"),
    "llavav1.5-13b-1k-templated": ('LLaVA', "shijianS01/llava-v1.5-13b-lora-1k-templated"),
    "llavav1.5-13b-5k-templated": ('LLaVA', "shijianS01/llava-v1.5-13b-lora-5k-templated"),
    "llavav1.5-13b-10k-templated": ('LLaVA', "shijianS01/llava-v1.5-13b-lora-10k-templated"),
    "llavav1.5-13b-15k-templated": ('LLaVA', "shijianS01/llava-v1.5-13b-lora-15k-templated"),

}


def set_imageqa_model_key(model_name, key):
    imageqa_models[model_name] = (imageqa_models[model_name][0], key)


def list_imageqa_models():
    return list(imageqa_models.keys())

def calculate_log_probs(scores):
    """Compute the average log probability correctly.
    
    Steps:
    1. Apply softmax to logits to get probabilities.
    2. Select the maximum probability.
    3. Compute the log probability of the selected maximum.
    4. Convert the log probability back to normal probability using exp.
    5. Return the average probability.
    """
    log_probs = []
    
    for logits in scores:
        probs = F.softmax(logits, dim=-1)  # Step 1: Apply softmax
        max_prob, _ = torch.max(probs, dim=-1)  # Step 2: Select the max probability
        log_prob = torch.log(max_prob).item()  # Step 3: Compute log probability
        log_probs.append(log_prob)
    
    avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else None  # Compute average log probability

    # Step 4: Convert log probability back to normal probability
    avg_prob = torch.exp(torch.tensor(avg_log_prob)).item() if avg_log_prob is not None else None
    return avg_prob  # Step 5: Return final probability

class ImageQAModel(QAModel):
   
    def __init__(
        self,
        model_name: str,
        model: QAModelInstance = None,
        torch_device: Union[int, str] = -1,
        precision=torch.bfloat16,
        choice_format='letter',
        enable_choice_search: bool = True,
        cache_path: str = None,

    ):
        from transformers import AutoProcessor
        super().__init__(model_name, choice_format, enable_choice_search, cache_path)

        if isinstance(torch_device, str):
            if torch_device != "auto":
                torch_device = torch.device(torch_device)
            else:
                pass
        else:
            if torch_device == -1:
                torch_device = torch.device(
                    "cuda") if torch.cuda.is_available() else "cpu"
            else:
                torch_device = torch.device(f"cuda:{torch_device}")

        if model is None:
            print(f"Loading {model_name}...")
            class_name, ckpt = imageqa_models[model_name]
            self.model_precision = precision
            self.model = eval(class_name)(
                ckpt, torch_device, self.model_precision)
            print(f"Finish loading {model_name}")
        else:
            print(f"Using provided model...")
            self.model = model

    def _data_to_str(self, data):
        if isinstance(data, str):
            return data
        else:
            return image_to_base64(data)


class LLaVA(QAModelInstance):
    def __init__(self, ckpt="llava-hf/llava-1.5-7b-hf", torch_device=torch.device("cuda"), model_precision=torch.float32):

        if ckpt == "llava-hf/llava-v1.6-vicuna-13b-hf" or ckpt == "llava-hf/llava-v1.6-vicuna-7b-hf":
            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                ckpt,
                torch_dtype=model_precision,
                low_cpu_mem_usage=True,
                device_map=torch_device
            ).eval()
            self.processor = LlavaNextProcessor.from_pretrained(
                ckpt, device_map=torch_device)
        elif ckpt in {
                "llava-hf/llava-1.5-7b-hf",
                "/mnt/ali-sh-1/usr/tusen/tmp-dev/shijian/template-scaling/LLaVA/checkpoints/hf_models/llava-v1.5-7b-lora-10-templated",
                "shijianS01/llava-v1.5-7b-lora-100-templated",
                "shijianS01/llava-v1.5-7b-lora-1k-templated",
                "shijianS01/llava-v1.5-7b-lora-5k-templated",
                "shijianS01/llava-v1.5-7b-lora-10k-templated",
                "shijianS01/llava-v1.5-7b-lora-15k-templated",
        }:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            self.model = LlavaForConditionalGeneration.from_pretrained(
                ckpt,
                torch_dtype=model_precision,
                low_cpu_mem_usage=True,
                device_map=torch_device,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                "llava-hf/llava-1.5-7b-hf", device_map=torch_device)
        elif ckpt in {
                "llava-hf/llava-1.5-13b-hf",
                "/mnt/ali-sh-1/usr/tusen/tmp-dev/shijian/template-scaling/LLaVA/checkpoints/hf_models/llava-v1.5-13b-lora-10-templated",
                "shijianS01/llava-v1.5-13b-lora-100-templated",
                "shijianS01/llava-v1.5-13b-lora-1k-templated",
                "shijianS01/llava-v1.5-13b-lora-5k-templated",
                "shijianS01/llava-v1.5-13b-lora-10k-templated",
                "shijianS01/llava-v1.5-13b-lora-15k-templated"
        }:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            self.model = LlavaForConditionalGeneration.from_pretrained(
                ckpt,
                torch_dtype=model_precision,
                low_cpu_mem_usage=True,
                device_map=torch_device,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                "llava-hf/llava-1.5-13b-hf", device_map=torch_device)
        else:
            raise ValueError("Not Implemented")

    def qa(self, image, prompt):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            image=image

        prompt = "USER: <image>\n" + prompt + "\nASSISTANT:"
        if isinstance(self.model, torch.nn.DataParallel):
            inputs = self.processor(prompt, image, return_tensors='pt').to(
                next(self.model.parameters()).device)
            out = self.model.module.generate(
                **inputs, max_new_tokens=200, do_sample=False)
        else:
            inputs = self.processor(
                text=prompt, images=image, return_tensors='pt').to(self.model.device)
            out = self.model.generate(
                **inputs, max_new_tokens=200, do_sample=False)
        answer = self.processor.decode(
            out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

        return answer,0

class QwenVL(QAModelInstance):
    def __init__(self, ckpt="Qwen/Qwen-VL", torch_device=torch.device("cuda"), model_precision=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt, trust_remote_code=True)
        if model_precision == torch.float32:
            self.model = AutoModelForCausalLM.from_pretrained(
                ckpt,
                device_map=torch_device,
                trust_remote_code=True,
                fp32=True,
                low_cpu_mem_usage=True,
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                ckpt,
                device_map=torch_device,
                trust_remote_code=True,
                bf16=True,
                low_cpu_mem_usage=True,
            ).eval()

    def qa(self, image, prompt):
        if isinstance(image, Image.Image):
            # Check if the image is a PIL.Image object and save to a temporary file if so
            with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
                image.save(tmp.name)
                image_path = tmp.name

                # Use the temporary image path for the tokenizer
                query = self.tokenizer.from_list_format([
                    {'image': image_path},
                    {'text': prompt},
                ])

                inputs = self.tokenizer(
                    query, return_tensors='pt').to(self.model.device)
                out = self.model.generate(**inputs,return_dict_in_generate=True,output_scores=True)

        else:
            # If `image` is not a PIL.Image object, use it directly
            query = self.tokenizer.from_list_format([
                {'image': image},
                {'text': prompt},
            ])

            inputs = self.tokenizer(
                query, return_tensors='pt').to(self.model.device)
            out = self.model.generate(**inputs,return_dict_in_generate=True)

        answer = self.tokenizer.decode(
            out.sequences[0][inputs["input_ids"].size(1):], skip_special_tokens=True
        ).strip()
        log_probs=calculate_log_probs(out.scores)
        return answer,log_probs
    
class Qwen25VL(QAModelInstance):
    def __init__(
        self,
        ckpt="Qwen/Qwen2.5-VL-7B-Instruct",
        torch_device="cuda",
        model_precision=torch.bfloat16,
        use_flash_attention=True,          # 新增参数，控制是否启用 Flash Attention
    ):
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration
        )

        self.processor = AutoProcessor.from_pretrained(
            ckpt,
            trust_remote_code=True
        )

        # 准备模型加载参数
        model_kwargs = {
            "device_map": torch_device,
            "torch_dtype": model_precision,
            "trust_remote_code": True,
        }
        if use_flash_attention:
            # 建议使用 float16/bfloat16 以获得最佳性能
            if model_precision not in [torch.float16, torch.bfloat16]:
                print("Warning: Flash Attention works best with float16 or bfloat16. Current dtype: {}".format(model_precision))
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ckpt,
            **model_kwargs
        ).eval()

    def qa(self, image, prompt):
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return answer,1


class Qwen25Omni(QAModelInstance):
    """Qwen2.5-Omni-7B 视觉语言模型，使用 Thinker 仅做文本生成（不加载 Talker），适用于图像 QA 评测。"""
    def __init__(
        self,
        ckpt="Qwen/Qwen2.5-Omni-7B",
        torch_device="cuda",
        model_precision=torch.bfloat16,
        use_flash_attention=True,
    ):
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            ckpt,
            trust_remote_code=True,
        )

        model_kwargs = {
            "device_map": torch_device,
            "torch_dtype": model_precision,
            "trust_remote_code": True,
        }
        if use_flash_attention:
            if model_precision not in [torch.float16, torch.bfloat16]:
                print("Warning: Flash Attention works best with float16 or bfloat16. Current dtype: {}".format(model_precision))
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            ckpt,
            **model_kwargs
        ).eval()

    def qa(self, image, prompt):
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return answer, 1


class Qwen3VL(QAModelInstance):
    def __init__(
        self,
        ckpt="Qwen/Qwen3-VL-8B-Instruct",
        torch_device="cuda",
        model_precision=torch.bfloat16,
        use_flash_attention=True,          # 新增参数，控制是否启用 Flash Attention
    ):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(
            ckpt,
            trust_remote_code=True
        )

        # 准备 from_pretrained 的参数字典
        model_kwargs = {
            "torch_dtype": model_precision,
            "device_map": torch_device,    # 或 "auto"
            "trust_remote_code": True,
        }
        if use_flash_attention:
            # 确保 dtype 为 float16/bfloat16 以获得最佳性能
            if model_precision not in [torch.float16, torch.bfloat16]:
                print("Warning: Flash Attention works best with float16 or bfloat16. Current dtype: {}".format(model_precision))
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForImageTextToText.from_pretrained(
            ckpt,
            **model_kwargs
        ).eval()

    def qa(self, image, prompt):
        from qwen_vl_utils import process_vision_info  # 若Qwen3也依赖此工具

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 应用聊天模板，生成包含特殊标记的文本
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 处理图像（和可能的视频）
        image_inputs, video_inputs = process_vision_info(messages)

        # 编码输入，必须包含 padding 和返回张量
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # 生成
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,          # 明确关闭采样
        )

        # 去除输入部分，只保留新生成的 token
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return answer, 1



class QwenVLChat(QAModelInstance):
    def __init__(self, ckpt="Qwen/Qwen-VL-Chat", torch_device=torch.device("cuda"), model_precision=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig

        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt, trust_remote_code=True)
        if model_precision == torch.float32:
            self.model = AutoModelForCausalLM.from_pretrained(
                ckpt,
                device_map=torch_device,
                trust_remote_code=True,
                fp32=True,
                low_cpu_mem_usage=True,
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                ckpt,
                device_map=torch_device,
                trust_remote_code=True,
                bf16=True,
                low_cpu_mem_usage=True,
            ).eval()

        # Specify hyperparameters for generation
        self.model.generation_config = GenerationConfig.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True)

    # def qa(self, image, prompt):
    #     if isinstance(image, Image.Image):
    #         # Check if the image is a PIL.Image object and save to a temporary file if so
    #         with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
    #             image.save(tmp.name)
    #             image_path = tmp.name

    #             # Use the temporary image path for the tokenizer
    #             query = self.tokenizer.from_list_format([
    #                 {'image': image_path},
    #                 {'text': prompt},
    #             ])

    #             answer, history = self.model.chat(
    #                 self.tokenizer, query=query, history=None)
    #     else:
    #         # If `image` is not a PIL.Image object, use it directly
    #         query = self.tokenizer.from_list_format([
    #             {'image': image},
    #             {'text': prompt},
    #         ])

    #         answer, history = self.model.chat(
    #             self.tokenizer, query=query, history=None)

    #     return answer

    def qa(self, image, prompt):
        if isinstance(image, Image.Image):
            # Check if the image is a PIL.Image object and save to a temporary file if so
            with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
                image.save(tmp.name)
                image_path = tmp.name

                # Use the temporary image path for the tokenizer
                query = self.tokenizer.from_list_format([
                    {'image': image_path},
                    {'text': prompt},
                ])

                inputs = self.tokenizer(
                    query, return_tensors='pt').to(self.model.device)
                out = self.model.generate(**inputs,return_dict_in_generate=True,output_scores=True)

        else:
            # If `image` is not a PIL.Image object, use it directly
            query = self.tokenizer.from_list_format([
                {'image': image},
                {'text': prompt},
            ])

            inputs = self.tokenizer(
                query, return_tensors='pt').to(self.model.device)
            out = self.model.generate(**inputs,return_dict_in_generate=True)

        answer = self.tokenizer.decode(
            out.sequences[0][inputs["input_ids"].size(1):], skip_special_tokens=True
        ).strip()
        log_probs=calculate_log_probs(out.scores)
        return answer,log_probs


class InternVLChat(QAModelInstance):
    """InternVL 官方用法：AutoModel + model.chat(tokenizer, pixel_values, question, generation_config)。
    见 https://huggingface.co/OpenGVLab/InternVL2_5-8B Quick Start：仅给出 AutoModel/AutoTokenizer + chat()，无 AutoProcessor。"""
    def __init__(self, ckpt="OpenGVLab/InternVL-Chat-V1-5", torch_device=torch.device("cuda"), model_precision=torch.bfloat16):
        from transformers import AutoTokenizer, AutoModel
        model_kwargs = dict(
            torch_dtype=model_precision,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )
        # InternVL2.5 / 3 推荐 use_flash_attn
        if "InternVL2" in ckpt or "InternVL3" in ckpt:
            model_kwargs["use_flash_attn"] = True
        self.model = AutoModel.from_pretrained(ckpt, **model_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, use_fast=False)
        self._max_num = 12 if ("InternVL2" in ckpt or "InternVL3" in ckpt) else 6

        # 从 tokenizer 里推断 image special tokens（避免源码里 <img> 标签被平台吞掉）
        specials = getattr(self.tokenizer, "additional_special_tokens", []) or []
        def _pick(substrs):
            for t in specials:
                s = str(t)
                if any(sub in s.lower() for sub in substrs):
                    return s
            return None
        self._img_start_token = _pick(["img_start", "image_start", "<img>"]) or "<img>"
        self._img_end_token = _pick(["img_end", "image_end", "</img>"]) or "</img>"
        self._img_context_token = _pick(["img_context", "image_context", "img_patch", "<img_context>"]) or "<IMG_CONTEXT>"

        # 兼容部分环境下 InternLM2ForCausalLM 缺少 generate() 的问题：
        # OpenGVLab 的 remote code 在 chat() 内部会调用 self.language_model.generate(...)
        # 若 transformers 版本/基类不含该方法，这里动态绑定 GenerationMixin.generate。
        try:
            from transformers.generation.utils import GenerationMixin
            lm = getattr(self.model, "language_model", None)
            if lm is not None and not hasattr(lm, "generate"):
                lm.generate = GenerationMixin.generate.__get__(lm, lm.__class__)
        except Exception:
            # 若 generation utils 不可用，则保持原状，让上层报错提示升级 transformers
            pass

    def qa(self, image, prompt):
        # 与官方一致：load_image -> pixel_values，question = '<image>\n' + prompt，model.chat(...)
        if isinstance(image, Image.Image):
            with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
                image.save(tmp.name)
                image_path = tmp.name
                pixel_values = load_image(image_path, max_num=self._max_num).to(torch.bfloat16).cuda()
        else:
            pixel_values = load_image(image, max_num=self._max_num).to(torch.bfloat16).cuda()

        question = "<image>\n" + prompt
        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )
        response = self.model.chat(
            self.tokenizer, pixel_values, question, generation_config)
        return response, 0

    def _build_internvl_query(self, question: str, num_patches: int) -> str:
        # 复刻 OpenGVLab InternVLChatModel.chat 的 prompt 逻辑：用对话模板 + 用 image token 串替换第一个 <image>
        template = self.model.conv_template.copy()
        template.system_message = getattr(self.model, "system_message", template.system_message)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = self._img_start_token + (self._img_context_token * self.model.num_image_token * num_patches) + self._img_end_token
        # 只替换第一个 <image> 占位符
        if "<image>" in query:
            return query.replace("<image>", image_tokens, 1)
        return query

    @torch.no_grad()
    def logprob_of_response(self, image, prompt: str, response: str) -> float:
        # 用 teacher-forcing 方式计算 response 在给定 prompt+image 下的 logprob
        if isinstance(image, Image.Image):
            with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
                image.save(tmp.name)
                image_path = tmp.name
                pixel_values = load_image(image_path, max_num=self._max_num).to(torch.bfloat16).cuda()
        else:
            pixel_values = load_image(image, max_num=self._max_num).to(torch.bfloat16).cuda()

        question = "<image>\n" + prompt
        query_prompt = self._build_internvl_query(question, num_patches=pixel_values.shape[0])
        # prompt_len：只含 prompt（assistant 为空）到输入 token 长度
        prompt_inputs = self.tokenizer(query_prompt, return_tensors="pt")
        prompt_input_ids = prompt_inputs["input_ids"].to(self.model.device)
        prompt_attn = prompt_inputs["attention_mask"].to(self.model.device)

        # full：把 response 文本接在 query_prompt 后面（不再追加额外角色标签，保持与模型 prompt 一致）
        full_text = query_prompt + (response.strip() if response is not None else "")
        full_inputs = self.tokenizer(full_text, return_tensors="pt")
        input_ids = full_inputs["input_ids"].to(self.model.device)
        attention_mask = full_inputs["attention_mask"].to(self.model.device)

        # 计算 inputs_embeds 并注入 vit embeds（复刻 InternVLChatModel.generate）
        vit_embeds = self.model.extract_feature(pixel_values.to(self.model.device))
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        flat_embeds = input_embeds.reshape(B * N, C)
        flat_ids = input_ids.reshape(B * N)
        selected = (flat_ids == self.model.img_context_token_id)
        if selected.sum() == 0:
            raise ValueError("No image context tokens found in input_ids; cannot score InternVL response.")
        flat_embeds[selected] = vit_embeds.reshape(-1, C).to(flat_embeds.device)
        input_embeds = flat_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)

        prompt_length = prompt_input_ids.shape[1]
        seq_len = input_ids.shape[1]
        total = 0.0
        for k in range(seq_len - prompt_length):
            pred_pos = prompt_length + k - 1
            tok_pos = prompt_length + k
            tok_id = input_ids[0, tok_pos].item()
            total += log_probs[pred_pos, tok_id].item()
        return total


class InternVLHF(QAModelInstance):
    """HF 原生 InternVL（AutoProcessor + AutoModelForImageTextToText），支持 apply_chat_template 与 POSIX logprob。"""
    def __init__(
        self,
        ckpt="OpenGVLab/InternVL3-8B-hf",
        torch_device="cuda",
        model_precision=torch.bfloat16,
    ):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            ckpt,
            device_map=torch_device,
            torch_dtype=model_precision,
            trust_remote_code=True,
        ).eval()

    def qa(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=next(self.model.parameters()).dtype)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return answer, 0


class LlavaOnevision(QAModelInstance):
    """LLaVA-OneVision Qwen2 模型，使用 LlavaOnevisionForConditionalGeneration + apply_chat_template。"""
    def __init__(
        self,
        ckpt="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        torch_device="cuda",
        model_precision=torch.bfloat16,
    ):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            ckpt,
            torch_dtype=model_precision,
            low_cpu_mem_usage=True,
            device_map=torch_device,
        ).eval()

    def qa(self, image, prompt):
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
        ).to(self.model.device, dtype=next(self.model.parameters()).dtype)
        generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.decode(
            generated_ids_trimmed[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        return answer, 1


class LLaVAOneVision15(QAModelInstance):
    """LLaVA-OneVision-1.5 模型，使用 qwen_vl_utils.process_vision_info（Qwen 风格）。
    该模型 preprocessor_config 指定 Qwen2_5_VLProcessor，但仓库无 processor 类。
    使用 Qwen2.5-VL 的 processor，并将其 tokenizer 替换为模型自带的（词表一致）。"""
    def __init__(
        self,
        ckpt="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        torch_device="cuda",
        model_precision=torch.bfloat16,
    ):
        from transformers import AutoProcessor, AutoModelForCausalLM, Qwen2Tokenizer

        # 使用 Qwen2.5-VL processor（preprocessor_config 指定 Qwen2_5_VLProcessor）
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
        # 模型仓库 tokenizer_config 标明 Qwen2Tokenizer；AutoTokenizer 无法识别 Llavaonevision1_5Config，故直接指定类
        self.processor.tokenizer = Qwen2Tokenizer.from_pretrained(ckpt)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            torch_dtype=model_precision,
            device_map=torch_device,
            trust_remote_code=True,
        ).eval()

    def qa(self, image, prompt):
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return answer, 1


class Molmo2(QAModelInstance):
    """Molmo2 视觉语言模型（如 Molmo2-8B），基于 Qwen3，使用 AutoProcessor + AutoModelForImageTextToText。"""
    def __init__(
        self,
        ckpt="allenai/Molmo2-8B",
        torch_device="cuda",
        model_precision=torch.bfloat16,
    ):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(
            ckpt,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            ckpt,
            device_map=torch_device,
            torch_dtype=model_precision,
            trust_remote_code=True,
        ).eval()

    def qa(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=next(self.model.parameters()).dtype)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return answer, 1


class Molmo(QAModelInstance):
    """Molmo 7B-D 视觉语言模型，使用 processor.process(images, text) + generate_from_batch。"""
    def __init__(
        self,
        ckpt="allenai/Molmo-7B-D-0924",
        torch_device="cuda",
        model_precision=torch.bfloat16,
    ):
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        self.processor = AutoProcessor.from_pretrained(
            ckpt,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            device_map=torch_device,
            torch_dtype=model_precision,
            trust_remote_code=True,
        ).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(
            ckpt, trust_remote_code=True
        )

    def qa(self, image, prompt):
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor.process(
            images=[image],
            text=prompt,
        )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        gen_config = self.model.generation_config
        gen_config.max_new_tokens = 512
        gen_config.use_cache = True
        output = self.model.generate_from_batch(
            inputs,
            generation_config=gen_config,
            tokenizer=self.processor.tokenizer,
        )
        generated_tokens = output[0, inputs["input_ids"].size(1):]
        answer = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()
        return answer, 1


class IDEFICS2(QAModelInstance):
    def __init__(self, ckpt="HuggingFaceM4/idefics2-8b", torch_device=torch.device("cuda"), model_precision=torch.float32):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = AutoModelForVision2Seq.from_pretrained(
            ckpt,
            torch_dtype=model_precision,
            #_attn_implementation="flash_attention_2",
            device_map=torch_device
        )

    def _extract_assistant_content(self, text: str):
        parts = text.split('\nAssistant:', 1)
        if len(parts) > 1:
            return 'Assistant:' + parts[1]
        return text

    def qa(self, image, prompt):

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ]
            }
        ]

        input_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True)

        if isinstance(image, Image.Image):
            inputs = self.processor(text=input_prompt, images=[
                                    image], return_tensors="pt")
        else:
            inputs = self.processor(text=input_prompt, images=[
                                    image_utils.load_image(image)], return_tensors="pt")

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True)

        return self._extract_assistant_content(generated_texts[0]),0
