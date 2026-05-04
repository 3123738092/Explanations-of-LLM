# 运行说明

在仓库根目录 `Explanations-of-LLM/` 下执行（需已安装项目依赖，GPU 环境请将 `device` 设为 `cuda`）。

## 参数归因相关改动（文件与算法）

在不改动 `lxt` AttnLRP 激活传播规则的前提下，在解释阶段利用 **patched backward 后的参数梯度** 计算对目标 logit 的参数贡献：**`贡献 ≈ param.grad ⊙ param`**（与文档中的 Input×Gradient 式捷径一致），并做按层 / 模块 / attention head 的聚合。

| 文件 | 作用 |
|------|------|
| `src/explain/attnlrp_gpt2_efficient.py` | 可选开启参数 `requires_grad`；在 contrastive backward 后收集各层参数的贡献与摘要。 |
| `src/visualize/heatmap.py` | 参数可视化：`save_attention_head_heatmap`、`save_layer_parameter_trend`。 |
| `main.py` | 读取 `parameter_attribution` 配置，写出 `parameter_relevance/` 与各类型热力图（见下文输出目录）。 |
| `configs/gpt2_efficient.yaml` 等 | 增加 `parameter_attribution` 块（`enabled`、`save_tensors`、`top_modules`）。 |

## 常用命令

**预训练 GPT-2（Hugging Face Hub `gpt2`）**

```bash
python main.py --config configs/gpt2_efficient.yaml
```

**本仓库内的 SQuAD 微调 checkpoint**

```bash
python main.py --config configs/gpt2_efficient_finetuned_squad.yaml
```

**本仓库内的 SciQ 微调 checkpoint**

提示模板与 `train_gpt2_qa.py` 中 SciQ 样本格式一致（`Question` / `Context` / `Options` / `Answer:`，四选一字母作答）；默认从 Hugging Face 读取 `allenai/sciq`。

```bash
python main.py --config configs/gpt2_efficient_finetuned_sciq.yaml
```

SciQ 相关说明：

- 微调产物目录若**只有** `pytorch_model.bin` 与 `config.json`、**没有** tokenizer 文件时，需在 YAML 里设置 **`model.tokenizer_name`**（如 `gpt2`），与预训练 GPT-2 词表对齐。
- **`data.dataset`** 设为 **`sciq`** 时会走 `src/data/sciq_loader.py`；默认 **`split: validation`**，与常见微调评估集一致。
- 选项打乱使用 **`sciq_shuffle_seed` + 样本行索引**（与微调 `--sciq_shuffle_seed` 对齐）。若在 YAML 中省略该项，`main.py` 内默认仍为 **42**；若你微调时改过种子，须在配置里显式写出同一数值。
- 若要与 **`train_gpt2_qa.py` 使用的本地 parquet** 完全一致（行顺序与打乱），可设置 **`data.sciq_parquet_path`**（相对仓库根目录的路径即可）；读取 parquet 需要安装 **pandas**。

`--config` 指向任意 YAML 配置文件的路径（可用绝对路径）。

## 输出位置

结果写入配置中的 `output.dir`。

| 路径 | 内容 |
|------|------|
| `output.figures_dir/tokens/` | token 相关性热力图（`*_tokens.png`） |
| `output.figures_dir/layers/` | 层 × token 相关性热力图（`*_layers.png`） |
| `output.figures_dir/parameter_heads/` | attention head 参数归因（`*_param_heads.png`） |
| `output.figures_dir/parameter_layers/` | 按层的模块参数归因曲线（`*_param_layers.png`） |
| `<output.dir>/relevance/` | 每条样本的激活归因 `.pt` |
| `<output.dir>/parameter_relevance/` | 参数归因摘要 `.pt`（若开启 `parameter_attribution`） |
| `<output.dir>/faithfulness_summary.json` | faithfulness 汇总 |

## YAML 配置要点

可按需复制一份 `configs/*.yaml` 修改下列字段。

| 区块 | 字段 | 含义 |
|------|------|------|
| **model** | `family` | 目前使用 `gpt2_efficient`。 |
| | `name` | 模型：Hub 名（如 `gpt2`）或本地目录（相对仓库根目录或绝对路径），目录内需含 `config.json` 与权重文件。 |
| | `tokenizer_name` | 可选。checkpoint 无 tokenizer 时单独指定（如 SciQ 微调目录常用 `gpt2`）。 |
| | `device` | `cuda` 或 `cpu`。 |
| | `dtype` | `float32` / `float16` / `bfloat16`，视显存与 checkpoint 精度选择。 |
| **data** | `dataset` | `squad_v2` 或 **`sciq`**。 |
| | `split` | 数据集拆分；SQuAD / SciQ 解释管线一般用 **`validation`**。 |
| | `num_samples` | 处理的样本条数。 |
| | `max_length` | 超过该 token 长度的样本会被跳过。 |
| | `hf_dataset` | SciQ 时用 Hugging Face 数据集 id，默认 **`allenai/sciq`**。 |
| | `sciq_parquet_path` | SciQ 可选；指向本地 parquet 以与微调数据完全一致（需 pandas）。 |
| | `sciq_shuffle_seed` | SciQ 可选；选项打乱基准种子，省略则默认 **42**。 |
| **parameter_attribution** | `enabled` | 是否计算并可视化参数层归因。 |
| | `save_tensors` | 是否保存完整参数相关张量（体积大，默认建议 `false`）。 |
| | `top_modules` | 写入汇总 JSON 的前若干模块条数。 |
| **faithfulness** | `steps` | MoRF/LeRF faithfulness 的步数。 |
| | `strategy` | 保留字段，当前管线主要使用 `steps`。 |
| **output** | `dir` | 输出根目录。 |
| | `figures_dir` | 图片根目录（其下自动分子目录 `tokens`、`layers`、`parameter_heads`、`parameter_layers`）。 |

`explain` 块保留与论文管线一致，一般无需改。
