import torch
from transformers import AutoTokenizer
from transformers.models.gpt2 import modeling_gpt2

from lxt.efficient import monkey_patch
from lxt.utils import pdf_heatmap, clean_tokens

monkey_patch(modeling_gpt2, verbose=True)

model = modeling_gpt2.GPT2LMHeadModel.from_pretrained('openai-community/gpt2', device_map='cuda', torch_dtype=torch.bfloat16, attn_implementation="eager")

# deactive gradients on parameters to save memory
for param in model.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

prompt = """context:
         In the course of the 10th century, the initially destructive incursions of Norse war bands into the rivers of France evolved into more permanent encampments that included local women and personal property. The Duchy of Normandy, which began in 911 as a fiefdom, was established by the treaty of Saint-Clair-sur-Epte between King Charles III of West Francia and the famed Viking ruler Rollo, and was situated in the former Frankish kingdom of Neustria. The treaty offered Rollo and his men the French lands between the river Epte and the Atlantic coast in exchange for their protection against further Viking incursions. The area corresponded to the northern part of present-day Upper Normandy down to the river Seine, but the Duchy would eventually extend west beyond the Seine. The territory was roughly equivalent to the old province of Rouen, and reproduced the Roman administrative structure of Gallia Lugdunensis II (part of the former Gallia Lugdunensis).
       question: Who did Rollo sign the treaty of Saint-Clair-sur-Epte with?
       answer:"""

input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
input_embeds = model.get_input_embeddings()(input_ids)

# Inference
output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits

# Take the maximum logit at last token position. You can also explain any other token, or several tokens together!
max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)

# Backward pass (the relevance is initialized with the value of max_logits)
max_logits.backward()

# Obtain relevance. (Works at any layer in the model!)
relevance = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()  # Cast to float32 for higher precision

relevance = relevance / relevance.abs().max()

# Remove special characters that are not compatible wiht LaTeX
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
tokens = clean_tokens(tokens)

# Save heatmap as PDF
pdf_heatmap(tokens, relevance[0], path='heatmap_nomask.pdf', backend='xelatex')
