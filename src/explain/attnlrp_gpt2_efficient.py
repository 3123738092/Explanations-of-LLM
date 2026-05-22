"""Extract GPT-2 AttnLRP token and layer relevance.

LXT provides the efficient AttnLRP backward rules through its GPT-2 monkey patch.
This file keeps the main explanation flow direct: tokenize, run GPT-2, choose the
next-token target, run patched backward, and read out token/layer relevance.

Part 3 parameter summaries are optional and live in ``parameter_attribution.py``
so the main Part 1 flow remains easy to read.
"""
from __future__ import annotations

from typing import Any

import torch

from src.explain.parameter_attribution import (
    collect_parameter_relevance,
    parameter_gradients_enabled,
)


def _capture_block_hidden_states(model) -> tuple[dict[int, torch.Tensor], list[Any]]:
    """Retain each GPT-2 block output so layer relevance can be read after backward."""
    hidden_states: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.requires_grad:
                hidden.retain_grad()
                hidden_states[layer_idx] = hidden

        return hook

    for layer_idx, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(make_hook(layer_idx)))
    return hidden_states, handles


def explain_sample(
    model,
    tokenizer,
    sample: dict,
    device: str = "cuda",
    parameter_attribution: bool = False,
    save_parameter_tensors: bool = False,
) -> dict:
    """Run AttnLRP for one prompt and return token/layer/optional parameter relevance."""
    #第一步：编码输入文本
    encoded = tokenizer(sample["prompt"], return_tensors="pt").to(device)
    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    #第二步：获取输入嵌入，将输入（嵌入向量）单独抓出来以后期计算relevance
    input_embeddings = model.get_input_embeddings()(input_ids)
    input_embeddings.requires_grad_(True)
    input_embeddings.retain_grad()
    
    #第三步：同理把layer 层的hidden state抓取出来
    block_hidden_states, hook_handles = _capture_block_hidden_states(model)
    try:
        with parameter_gradients_enabled(model, parameter_attribution):
            model.zero_grad(set_to_none=True)
            logits = model(inputs_embeds=input_embeddings, use_cache=False).logits
            next_token_logits = logits[0, -1]
            target_id = int(next_token_logits.argmax(dim=-1).item())

            #构造长度为V的向量，向量中每个元素都是-1/V
            contrastive_seed = torch.full_like(
                next_token_logits,
                -1.0 / next_token_logits.numel(),
            )
            #将target_id位置的元素设置为1
            contrastive_seed[target_id] = 1.0
            next_token_logits.backward(contrastive_seed)

            token_relevance = (
                (input_embeddings * input_embeddings.grad)
                .float()
                .sum(-1) #对最后一个维度求和
                .squeeze(0)
                .detach()
                .cpu()
            )

            n_layers = len(model.transformer.h)
            n_tokens = token_relevance.shape[0]
            layer_relevance = torch.zeros(n_layers, n_tokens)
            for layer_idx, hidden in block_hidden_states.items():
                if hidden.grad is not None:
                    layer_relevance[layer_idx] = (
                        (hidden * hidden.grad)
                        .float()
                        .sum(-1)
                        .squeeze(0)
                        .detach()
                        .cpu()
                    )

            parameter_result = (
                collect_parameter_relevance(model, save_tensors=save_parameter_tensors)
                if parameter_attribution
                else {}
            )
    finally:
        for handle in hook_handles:
            handle.remove()
        model.zero_grad(set_to_none=True)

    return {
        "tokens": tokens,
        "token_relevance": token_relevance,
        "layer_relevance": layer_relevance,
        "target_token": tokenizer.convert_ids_to_tokens([target_id])[0],
        "target_id": target_id,
        "input_ids": input_ids.detach().cpu(),
        **parameter_result,
    }
