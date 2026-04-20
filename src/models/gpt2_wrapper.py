"""GPT-2 loader with AttnLRP rule registration via the `lxt` package."""
from __future__ import annotations

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# lxt patches GPT-2 modules in-place so that a standard .backward() produces
# AttnLRP relevance (see Achtibat et al., ICML 2024).
from lxt.models.gpt2 import attnlrp as gpt2_attnlrp


def load_gpt2_with_attnlrp(name: str = "gpt2", device: str = "cuda"):
    tokenizer = GPT2TokenizerFast.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    gpt2_attnlrp.register(model)
    model.to(device)
    return model, tokenizer
