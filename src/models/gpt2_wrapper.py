"""GPT-2 loader using lxt's AttnLRP-aware GPT-2 implementation."""
from __future__ import annotations

import torch
from transformers import GPT2TokenizerFast

from lxt.explicit.models.gpt2 import GPT2LMHeadModel, attnlrp


def load_gpt2_with_attnlrp(name: str = "gpt2", device: str = "cuda"):
    tokenizer = GPT2TokenizerFast.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    attnlrp.register(model)
    model.to(device)
    return model, tokenizer
