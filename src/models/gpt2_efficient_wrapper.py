"""GPT-2 loader using lxt's efficient AttnLRP monkey patch."""
from __future__ import annotations

import torch
from transformers import GPT2TokenizerFast
from transformers.models.gpt2 import modeling_gpt2

from lxt.efficient import monkey_patch


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_gpt2_efficient_with_attnlrp(
    name: str = "gpt2",
    device: str = "cuda",
    dtype: str = "bfloat16",
    tokenizer_name: str | None = None,
):
    monkey_patch(modeling_gpt2, verbose=False)

    tok_src = tokenizer_name if tokenizer_name else name
    tokenizer = GPT2TokenizerFast.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = modeling_gpt2.GPT2LMHeadModel.from_pretrained(
        name,
        torch_dtype=_DTYPES[dtype],
        attn_implementation="eager",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    return model, tokenizer
