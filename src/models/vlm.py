"""Qwen2.5-VL-3B loaded once in 4-bit — the only model on the GPU.

Everything (indexing-time summaries, query-image captions, final answers) goes
through this one instance. Loading it a second time would blow the 6 GB budget,
so it is cached behind ``lru_cache`` and never instantiated more than once.
"""
from __future__ import annotations

from functools import lru_cache

import torch

import config


@lru_cache(maxsize=1)
def _load():
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. The VLM must run on the RTX 3050. "
            "Check `nvidia-smi`, and that torch was installed from the cu121 wheel."
        )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.VLM_MODEL_ID,
        quantization_config=bnb,
        device_map="cuda:0",
        attn_implementation="sdpa",   # flash-attn is unavailable on Windows
        token=config.HF_TOKEN,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        config.VLM_MODEL_ID,
        min_pixels=config.VLM_MIN_PIXELS,
        max_pixels=config.VLM_MAX_PIXELS,
        token=config.HF_TOKEN,
    )
    return model, processor


def get_vlm():
    """Return the (model, processor) tuple, loading them on first use."""
    return _load()


def _prepare(messages):
    from qwen_vl_utils import process_vision_info

    model, processor = _load()
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    return model, processor, inputs


@torch.inference_mode()
def chat(messages, max_new_tokens: int = 512) -> str:
    """Run one blocking generation and return the decoded assistant text."""
    model, processor, inputs = _prepare(messages)
    generated = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return text.strip()


def stream(messages, max_new_tokens: int = 512):
    """Yield decoded text chunks as they are generated (for the dashboard)."""
    from threading import Thread

    from transformers import TextIteratorStreamer

    model, processor, inputs = _prepare(messages)
    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    kwargs = dict(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer
    )
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    try:
        for piece in streamer:
            yield piece
    finally:
        thread.join()


def ask_about_image(image, prompt: str, max_new_tokens: int = 320) -> str:
    """Convenience wrapper: one image + one text prompt -> text answer."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return chat(messages, max_new_tokens=max_new_tokens)
