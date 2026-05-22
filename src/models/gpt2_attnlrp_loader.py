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
# 不是在改某一个模型实例，而是在 from_pretrained 加载权重之前，改掉 HuggingFace GPT-2 模块里若干类的 forward（以及 attention 函数表）。之后 new 出来的 GPT2LMHeadModel 都会走改过的逻辑。
    monkey_patch = _load_lxt_monkey_patch()  #实则返回了monkey_patch函数
    monkey_patch(modeling_gpt2, verbose=False) #在类定义层面替换 GPT-2 里 attention/MLP 等的 forward/backward

    tok_src = tokenizer_name if tokenizer_name else name
    tokenizer = GPT2TokenizerFast.from_pretrained(tok_src) # tokenizer.pad_token 是 tokenizer 对象的一个属性，表示用于填充的特殊 token。当输入序列的长度小于最大长度时，tokenizer 会自动使用 pad_token 来填充，以确保输入序列的长度等于最大长度。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = modeling_gpt2.GPT2LMHeadModel.from_pretrained(
        name,
        torch_dtype=_DTYPES[dtype],
        attn_implementation="eager", #eager模式是torch.compile的默认模式，表示eager执行，即立即执行，不会缓存计算图。特别细的一个点，没有用flash attention
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    return model, tokenizer
