# Data Sources

Both datasets used in this project are publicly available.

## SQuAD v2

- **Dataset:** Stanford Question Answering Dataset v2.0
- **Description:** Extractive QA with unanswerable questions (130,319 train / 3,958 dev)
- **Paper:** Rajpurkar et al., "Know What You Don't Know: Unanswerable Questions for SQuAD", ACL 2018
- **Access:** https://huggingface.co/datasets/rajpurkar/squad_v2
- **License:** CC BY-SA 4.0

## SciQ

- **Dataset:** Science Question Answering Dataset
- **Description:** 4-choice multiple-choice science questions (11,679 train / 1,000 test)
- **Paper:** Welbl et al., "Crowdsourcing Multiple Choice Science Questions", W-NUT 2017
- **Access:** https://huggingface.co/datasets/allenai/sciq
- **License:** CC BY-NC 4.0

## Model

- GPT-2 base (124M parameters): https://huggingface.co/openai-community/gpt2

---

The datasets are loaded automatically via `datasets.load_dataset()` from Hugging Face Hub when running the pipeline. No manual data download is required.
