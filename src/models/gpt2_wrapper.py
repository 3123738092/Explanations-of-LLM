"""GPT-2 loader using lxt's AttnLRP-aware GPT-2 implementation."""
from __future__ import annotations

import torch
from transformers import GPT2TokenizerFast

from lxt.explicit.models.gpt2 import GPT2LMHeadModel, attnlrp


def load_gpt2_with_attnlrp(
    name: str = "gpt2",
    device: str = "cuda",
    tokenizer_name: str | None = None,
):
    tok_name = tokenizer_name or name
    tokenizer = GPT2TokenizerFast.from_pretrained(tok_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(name, torch_dtype=torch.float32, local_files_only=True)
    model.eval()
    attnlrp.register(model)
    model.to(device)
    return model, tokenizer
