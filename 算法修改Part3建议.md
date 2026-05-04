# AttnLRP Parameter Attribution Specification

## 1. 任务目标 (Objective)
本指南旨在说明如何基于 **AttnLRP (Attention-Aware Layer-Wise Relevance Propagation)** 框架修改现有算法，以计算模型中每一层**模型参数（Weights 和 Biases）**对最终输出 Logit 的贡献值（Contribution Values/Relevance）。

**当前状态：** 算法目前仅支持计算输入特征（Input Tokens/Activations）的相关性。
**最终目的：** 算法需要能够精确定位并量化具体的模型权重（如 $W_q, W_k, W_v, W_{mlp}$ 等）对特定预测结果的物理贡献。

---

## 2. 核心理论基石 (Theoretical Foundation)
在进行参数归因之前，必须严格遵守 AttnLRP 处理 Transformer 架构的三大核心反向传播规则。**这些针对中间激活值（Activations）的传播规则是保证相关性 $R$ 在深层网络中不爆炸、不丢失纯度的基石，不可做任何破坏性修改：**

1. **Softmax 规则 (保留偏置)：** $R_{i}^{l-1} = x_{i} \left(R_{i}^{l} - s_{i} \sum_{j} R_{j}^{l}\right)$ 
2. **LayerNorm 规则 (恒等传递)：** $R_{i}^{l-1} = R_{i}^{l}$
3. **双线性矩阵乘法规则 (Shapley Value 均分)：** 对于 $O = A \times V$，分配规则为 $R_{ji}^{l-1} = \sum_{p} A_{ji} V_{ip} \frac{R_{jp}^{l}}{2(O_{jp} + \epsilon)}$

只有在主干道（激活值）使用上述规则获取了纯净的中间层相关性 $R_{l}$ 后，才能进行下面的参数相关性计算。

---

## 3. 参数归因的算法逻辑 (Algorithmic Logic for Parameters)

在线性变换层 $z_{j} = \sum_{i} W_{ji} x_{i} + b_{j}$ 中，权重 $W$ 与输入 $x$ 在数学上处于完全对称的地位。因此，分配给参数的“奖金（Relevance）”同样遵循 LRP 的 $\epsilon$-rule。

### 核心差异：共享权重的跨时间步累加 (Summation over Sequence Dimension)
输入变量 $x$ 是独立的 Token，而模型参数 $W$ 和 $b$ 是在整个序列（Sequence Length, 记为 $t$）上共享的。因此，**一个权重参数的总贡献，必须是它在处理所有输入 Token 时所做贡献的总和。**

### 3.1 权重参数 (Weights) 的分配公式
假设在第 $l$ 层的输出端，我们已经通过 AttnLRP 规则截获了纯净的相关性矩阵 $R_{t,j}^{l}$，并且已知该层的前向输出为 $z_{t,j}$，输入为 $x_{t,i}$。
单个权重参数 $W_{ji}$ 的贡献度 $R_{W_{ji}}$ 计算公式为：

$$R_{W_{ji}} = \sum_{t} \frac{x_{t,i} W_{ji}}{z_{t,j} + \epsilon} R_{t,j}^{l}$$

* $t$: 序列/时间步维度 (Sequence length)
* $x_{t,i} W_{ji}$: 该特定权重在时间步 $t$ 的前向物理贡献
* $z_{t,j}$: 该节点在时间步 $t$ 的总输出

### 3.2 偏置参数 (Biases) 的分配公式
同理，偏置项 $b_j$ 本身就是其物理贡献，公式为：

$$R_{b_{j}} = \sum_{t} \frac{b_{j}}{z_{t,j} + \epsilon} R_{t,j}^{l}$$

---

## 4. 工程实现捷径：梯度与权重的阿达玛乘积 (Implementation Shortcut)

在深度泰勒分解（DTD）和 LRP 的实际工程落地中，**不需要**在代码中手动编写上述带有除法和求和的复杂公式。

根据链式法则与 $\epsilon$-rule 的等价性，当我们在 PyTorch/TensorFlow 中将 AttnLRP 的三套数学规则注入到自定义的 `autograd.Function` 中（即劫持标准的 `.backward()` 计算图）后，参数的归因计算可以极度简化。

### 算法执行流程：
1. **初始化源点：** 选定目标 Token，提取其分类头输出的原始 Logit 值。构建一个与词表大小相同的向量，仅目标位置置为该 Logit，其余置 $0$。将其作为整个网络相关性反向传播的起点 $R^L$。
2. **执行反向传播：** 触发被 AttnLRP 规则修改过的 `backward()` 过程。
3. **拦截参数梯度：** 随着反向传播的完成，模型中每一层的权重参数 $W$ 和偏置 $b$ 都会获得一个计算好的梯度属性（如 `W.grad`）。注意：由于我们修改了后向传播图，此时的梯度已经**不再是标准的损失函数梯度**，而是**相关性传播的比例因子**。
4. **计算最终贡献 (阿达玛乘积)：** * 权重贡献矩阵：$R_{W} = W.\text{grad} \odot W$
   * 偏置贡献向量：$R_{b} = b.\text{grad} \odot b$

*(注：$\odot$ 表示逐元素相乘 Hadamard product)*

---

## 5. 预期输出结果 (Expected Output)

修改算法后，对于模型中的任意一层（例如 LLaMa 中的 `q_proj`, `k_proj`, `v_proj`, `up_proj` 等），您将能够获得一个与其参数形状完全一致的贡献度张量（Contribution Tensor）。

例如，若 $W_q$ 的形状为 `[4096, 4096]`，算法应输出一个同样为 `[4096, 4096]` 的张量。该张量中的每一个浮点数，都精确量化了该特定参数对最终目标 Logit 的正向推动或反向抑制作用。