# 运行说明

在仓库根目录 `Explanations-of-LLM/` 下执行（需已安装项目依赖，GPU 环境请将 `device` 设为 `cuda`）。

## 参数归因相关改动（文件与算法）

在不改动 `lxt` AttnLRP 激活传播规则的前提下，在解释阶段利用 **patched backward 后的参数梯度** 计算对目标 logit 的参数贡献：**`贡献 ≈ param.grad ⊙ param`**（与文档中的 Input×Gradient 式捷径一致），并做按层 / 模块 / attention head 的聚合。

| 文件 | 作用 |
|------|------|
| `src/explain/attnlrp_gpt2_efficient.py` | 可选开启参数 `requires_grad`；在 contrastive backward 后收集各层参数的贡献与摘要。 |
| `src/visualize/heatmap.py` | 新增参数可视化：`save_attention_head_heatmap`、`save_layer_parameter_trend`。 |
| `main.py` | 读取 `parameter_attribution` 配置，写出 `parameter_relevance/` 与 `*_param_heads.png` / `*_param_layers.png`。 |
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

`--config` 指向任意 YAML 配置文件的路径（可用绝对路径）。

## 输出位置

结果写入配置中的 `output.dir`：热力图在 `output.figures_dir`，相关性张量在 `<output.dir>/relevance/`，若开启参数归因还有 `<output.dir>/parameter_relevance/` 与汇总 JSON。

## YAML 配置要点

可按需复制一份 `configs/*.yaml` 修改下列字段。

| 区块 | 字段 | 含义 |
|------|------|------|
| **model** | `family` | 目前使用 `gpt2_efficient`。 |
| | `name` | 模型：Hub 名（如 `gpt2`）或本地目录（相对仓库根目录或绝对路径），目录内需含 `config.json`、权重与 tokenizer。 |
| | `device` | `cuda` 或 `cpu`。 |
| | `dtype` | `float32` / `float16` / `bfloat16`，视显存与 checkpoint 精度选择。 |
| **data** | `split` | SQuAD v2 拆分，一般为 `validation`。 |
| | `num_samples` | 处理的样本条数。 |
| | `max_length` | 超过该 token 长度的样本会被跳过。 |
| **parameter_attribution** | `enabled` | 是否计算并可视化参数层归因。 |
| | `save_tensors` | 是否保存完整参数相关张量（体积大，默认建议 `false`）。 |
| | `top_modules` | 写入汇总 JSON 的前若干模块条数。 |
| **faithfulness** | `steps` | MoRF/LeRF faithfulness 的步数。 |
| | `strategy` | 保留字段，当前管线主要使用 `steps`。 |
| **output** | `dir` | 输出根目录。 |
| | `figures_dir` | 图片保存目录（可为 `dir` 的子路径）。 |

`explain` 块保留与论文管线一致，一般无需改。
