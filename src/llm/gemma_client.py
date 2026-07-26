"""Gemma 12B client for clinical text parsing (no forecasting).

Gemma 4 12B is not available on Hugging Face as of 2026-06; this module uses
``google/gemma-3-12b-it`` (Gemma 3 12B instruction-tuned), which fits on an
RTX 5090 in bfloat16 (~24 GB VRAM).
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Callable

from src.llm.parser import LLMContextParser

# Closest official Gemma 12B instruction model (Gemma 4 not released).
DEFAULT_MODEL_ID = 'google/gemma-3-12b-it'


@lru_cache(maxsize=1)
def _load_model_and_tokenizer(model_id: str) -> tuple[Any, Any]:
    """Lazy-load Gemma on GPU (cached for process lifetime)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = (
        os.environ.get('HF_TOKEN')
        or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map='auto' if torch.cuda.is_available() else None,
        token=token,
    )
    model.eval()
    return model, tokenizer


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned.strip()


def create_gemma_client(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> Callable[[str, str], str]:
    """Return llm_client(system_prompt, user_text) -> json_str."""

    def _client(system_prompt: str, user_text: str) -> str:
        import torch

        model, tokenizer = _load_model_and_tokenizer(model_id)
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_text},
        ]
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f'{system_prompt}\n\nUser: {user_text}\n\nJSON:'

        inputs = tokenizer(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        new_tokens = output_ids[0, input_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        cleaned = _strip_json_fences(raw)
        # Validate JSON early so parser can surface errors clearly.
        json.loads(cleaned)
        return cleaned

    return _client


def create_parser(
    *,
    use_gemma: bool = True,
    model_id: str = DEFAULT_MODEL_ID,
) -> LLMContextParser:
    """Build an ``LLMContextParser`` with optional Gemma backend."""
    if use_gemma:
        return LLMContextParser(llm_client=create_gemma_client(model_id))
    return LLMContextParser()
