import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.models.gpt2 import modeling_gpt2

from lxt.efficient import monkey_patch
from lxt.utils import pdf_heatmap, clean_tokens

monkey_patch(modeling_gpt2, verbose=True)

SAMPLE_INDEX = 45  # 0-based index after max_length filtering
SAMPLE_INDEX_ALT = 1
MAX_LENGTH = 512


def format_prompt(question: str, context: str) -> str:
  return f"Context: {context}\nQuestion: {question}\nAnswer:"


def load_squad_v2_sample(index: int, tokenizer) -> dict:
  ds = load_dataset("squad_v2", split="validation")
  kept = 0
  for ex in ds:
    prompt = format_prompt(ex["question"], ex["context"])
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if ids.shape[1] > MAX_LENGTH:
      continue
    if kept == index:
      answer = ex["answers"]["text"][0] if len(ex["answers"]["text"]) > 0 else None
      return {
        "id": ex["id"],
        "prompt": prompt,
        "answer": answer,
      }
    kept += 1
  raise ValueError(f"Sample index {index} not found within max_length filter")


def run_contrastive_explain(
  model_name: str,
  tokenizer_name: str,
  dtype,
  output_pdf: str,
  sample_index: int,
) -> None:
  model = modeling_gpt2.GPT2LMHeadModel.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=dtype,
    attn_implementation="eager",
  )

  for param in model.parameters():
    param.requires_grad = False

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  sample = load_squad_v2_sample(sample_index, tokenizer)

  print(f"Using sample id: {sample['id']}")
  print(f"Answer: {sample['answer']}")

  input_ids = tokenizer(sample["prompt"], return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
  input_embeds = model.get_input_embeddings()(input_ids)
  output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits

  max_logit, max_index = torch.max(output_logits[0, -1, :], dim=-1)

  # Contrastive explanation
  mask = torch.ones_like(output_logits[0, -1, :]) * -1 / output_logits[0, -1, :].size(-1)
  mask[max_index] = 1
  output_logits[0, -1, :].backward(mask)

  relevance = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0]
  relevance = relevance / relevance.abs().max()

  tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
  tokens = clean_tokens(tokens)
  pdf_heatmap(tokens, relevance, path=output_pdf, backend="pdflatex")


if __name__ == "__main__":
  run_contrastive_explain(
    model_name="openai-community/gpt2",
    tokenizer_name="openai-community/gpt2",
    dtype=torch.bfloat16,
    output_pdf="heatmap_contrastive_squad045_base.pdf",
    sample_index=SAMPLE_INDEX,
  )
  run_contrastive_explain(
    model_name="FT-model/modelA_SQuAD_15600",
    tokenizer_name="FT-model/modelA_SQuAD_15600",
    dtype=torch.float32,
    output_pdf="heatmap_contrastive_squad045_finetuned.pdf",
    sample_index=SAMPLE_INDEX,
  )
  run_contrastive_explain(
    model_name="openai-community/gpt2",
    tokenizer_name="openai-community/gpt2",
    dtype=torch.bfloat16,
    output_pdf="heatmap_contrastive_squad001_base.pdf",
    sample_index=SAMPLE_INDEX_ALT,
  )
  run_contrastive_explain(
    model_name="FT-model/modelA_SQuAD_15600",
    tokenizer_name="FT-model/modelA_SQuAD_15600",
    dtype=torch.float32,
    output_pdf="heatmap_contrastive_squad001_finetuned.pdf",
    sample_index=SAMPLE_INDEX_ALT,
  )