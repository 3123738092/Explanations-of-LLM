"""Load GPT-2 with LXT efficient AttnLRP monkey patching."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast
from transformers.models.gpt2 import modeling_gpt2


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _load_lxt_monkey_patch():
    """Import LXT from the environment, falling back to the vendored copy."""
    try:
        from lxt.efficient import monkey_patch

        return monkey_patch
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        vendored = repo_root / "third_party" / "LRP-eXplains-Transformers"
        if vendored.is_dir() and str(vendored) not in sys.path:
            sys.path.insert(0, str(vendored))
        from lxt.efficient import monkey_patch

        return monkey_patch


def load_gpt2_efficient_with_attnlrp(
    name: str = "gpt2",
    device: str = "cuda",
    dtype: str = "bfloat16",
    tokenizer_name: str | None = None,
):
    monkey_patch = _load_lxt_monkey_patch()
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
