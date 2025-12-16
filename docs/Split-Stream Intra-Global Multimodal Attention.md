# SIGMA: 解耦模态内一致性与全局对齐的多模态注意力架构

> **摘要**: 现有的多模态大模型（MLLM）普遍采用单一混合流的注意力机制，导致**模态内特征（Intra-Modal Features，如文本语法或图片结构）**容易受到**跨模态相互作用（Inter-Modal Interactions）**的**稀释**与**干扰**。
>
> 本文介绍了一种名为 **SIGMA (Split-Stream Intra-Global Gated Multimodal Attention)** 的新架构。它通过物理分头（Split-Head）策略，构建了并行的 **Intra-Modal Stream（模态内流）** 和 **Global-Context Stream（全局流）**，并配合**模态特异性门控（Modality-Specific Gating）**，实现了模态内一致性与全局对齐的解耦控制。

---

## 1. 核心动机：解开“拿铁”中的咖啡与牛奶

在 LLaVA [1] 和 Qwen-VL [2] 等主流 MLLM 架构中，Attention 机制通常是**全连接（Full Self-Attention）**的。文本 Token 和图片 Token 在每一层都进行着密集的交互。

这就好比一杯**拿铁咖啡**：咖啡（文本特征）和牛奶（视觉特征）一旦在 Source 端混合，后续层级就很难再将其分离。

这种“混合流”带来了两个严重问题，也是本工作最主要的insights：
1.  **模态内特征稀释 (Intra-Modal Dilution)**：当模型需要专注模态内部建模（如纯文本推理或纯视觉一致性）时，异构模态的 Token 依然占据着 Attention 的带宽，引入分布外噪音。这个点起源于早期由LLM到MLLM的性能劣化观察，比如哪怕只是给模型一些空白图片也可能会显著影响指令能力，主要原因就是大量的图片token稀释了文本的attention。
2.  **控制失效**：即便是引入了 Gated Attention [3]，如果应用在混合流上，也只是调节了“拿铁”的总量，而无法单独剔除“牛奶”或“咖啡”。

**SIGMA 的核心思想**：与其在混合后试图“去噪”，不如在计算前进行**物理分流**。

---

## 2. SIGMA 架构详解

SIGMA 不引入额外的 Transformer 层，而是对现有的 Attention Layer 进行重构。其核心由三个部分组成：**物理分流 (Split)**、**双流掩码 (Dual Masks)** 和 **模态特异性门控 (Bias Injection)**。

### 2.1 整体架构图

SIGMA 采用了**双流并进 (Dual-Stream)** 的对称设计。

```text
                                     Input Sequence X
                                            |
                       +--------------------+--------------------+
                       |                                         |
             (STREAM 1: INTRA)                         (STREAM 2: GLOBAL)
          "Intra-Modal Consistency"                 "Global-Context Alignment"
                       |                                         |
       +---------------+---------------+         +---------------+---------------+
       |                               |         |                               |
       v                               v         v                               v
 [ Attn Projs ]                  [ Gate Proj ] [ Attn Projs ]              [ Gate Proj ]
 (Heads 0..15)                   (Linear)      (Heads 16..31)              (Linear)
       |                               |         |                               |
       v                               v         v                               v
 [ Apply Mask ]                  [ Add Bias ]  [ Apply Mask ]              [ Add Bias ]
 (Block Diag)                   (+ B[intra])    (Full Causal)              (+ B[global])
       |                               |         |                               |
       v                               v         v                               v
  [ SDPA ]                       [ Sigmoid ]   [ SDPA ]                    [ Sigmoid ]
       |                               |         |                               |
       v                               v         v                               v
    O_Intra                         g_intra   O_Global                        g_global
       |                               |         |                               |
       |                               |         |                               |
       +------------->(*) <------------+         +------------->(*) <------------+
                 Element-wise                              Element-wise
                      Mul                                       Mul
                       |                                         |
                       v                                         v
                 Gated_O_Intra                             Gated_O_Global
                       |                                         |
                       +--------------------+--------------------+
                                            |
                                            v
                                     [ Concatenation ]
```

### 2.2 关键技术组件

#### A. 物理分流与差异化掩码 (Split-Head & Dual Masks)
我们将 $H$ 个注意力头划分为两组（暂时对半分的，其实可以调一调）：
*   **Intra-Modal Stream (Heads 0~15)**: 应用 **块对角掩码 (Block-Diagonal Mask)**。文本仅关注文本历史，图片仅关注图片历史。此流保证了模态内特征的纯净性。
*   **Global-Context Stream (Heads 16~31)**: 应用 **标准自回归掩码 (Causal Mask)**。允许跨模态的任意交互（如 $T \leftarrow I$ 或 $I \leftarrow T$）。此流负责捕获图文对齐信息。

![image-20251215031624149](https://xianyunlp.oss-cn-hangzhou.aliyuncs.com/uPic/sigma_attention.png)

#### B. 模态特异性流偏置 (Modality-Specific Stream Bias)
区别于标准的 Gated Attention [3]，我们引入了显式的 Bias 项 $\mathbf{B} \in \mathbb{R}^{2 \times 2 \times D/2}$ 来加速门控分化。
$$ \text{Logits} = W_{gate} x + \mathbf{B}[\text{Stream}][\text{Modality}] $$

*   $\mathbf{B}[\text{Intra}][\text{Text}]$ 初始化为正值，赋予文本 Token 默认的模态内建模权重。
*   $\mathbf{B}[\text{Global}][\text{Text}]$ 初始化为负值，默认抑制跨模态干扰，实现“按需开启”。

#### C. 门控拼接 (Gated Concatenation)
最终输出由两个流拼接而成：
$$ O_{Final} = \text{Concat}(\mathbf{g}_{intra} \odot O_{intra}, \quad \mathbf{g}_{global} \odot O_{global}) $$

---

## 3. 性能与开销分析

我们将 SIGMA 与标准 Attention 以及目前 SOTA 的 Gated Attention 进行对比。

| 指标               | Standard Attention [1,2] | Gated Attention [3]     | **SIGMA (Ours)**                 |
| :----------------- | :----------------------- | :---------------------- | :------------------------------- |
| **模态解耦方式**   | 无 (全混合)              | 软性 (仅靠门控数值抑制) | **硬性 (物理分流 + 掩码隔离)**   |
| **特征纯净度**\*   | 低 (受异构模态干扰)      | 中 (混合特征被缩放)     | **高 (Intra流 0 污染)**          |
| **参数量变化**     | Baseline                 | + $D^2$ (Gate Proj)     | + $D^2$ (Gate Proj) + 极小 Bias  |
| **计算量 (FLOPs)** | Baseline                 | + $O(L \cdot D^2)$      | + $O(L \cdot D^2)$               |
| **FlashAttn**      | 支持                     | 支持                    | **支持 (Block/Causal Mask)** [4] |

> \* **特征纯净度 (Feature Purity)**：指 Hidden State 中仅包含同模态信息的程度。SIGMA 的 Intra 流在物理上保证了这一点。

---

## 4. 核心实现逻辑 (PyTorch-style Pseudocode)

以下代码展示了 SIGMA 的数据流向。Gate 的计算仅依赖于 Query 侧输入 $x$，而 Attention 的计算则利用 FlashAttention 的并行能力。

```python
class SIGMA_Attention(nn.Module):
    def __init__(self, config):
        # ... 初始化 ...
        # 新增门控投影与特异性 Bias (Innovative Point)
        self.gate_intra_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.gate_global_proj = nn.Linear(hidden_size, hidden_size // 2)
        
        # Bias Shape: [Stream(2), Modality(2), Head_Dim]
        self.stream_bias = nn.Parameter(torch.zeros(2, 2, split_head_dim)) 
        self.split_idx = num_heads // 2

    def forward(self, x, modality_id):
        # 1. 投影 Q, K, V
        # shape: [B, H, S, d]
        q, k, v = self.proj_qkv(x)
        
        # 2. 并行 Attention 计算 (FlashAttention [4])
        # 利用切片操作实现分流，无需冗余的 chunk 赋值
        # Stream 1: Intra-Modal (Heads 0 : H/2) -> 只看同模态
        out_intra = flash_attn(
            q[:, :self.split_idx], k[:, :self.split_idx], v[:, :self.split_idx], 
            mask=block_diag_mask
        )
        # Stream 2: Global-Context (Heads H/2 : H) -> 看全文
        out_global = flash_attn(
            q[:, self.split_idx:], k[:, self.split_idx:], v[:, self.split_idx:], 
            mask=causal_mask
        )
        
        # 3. 门控生成 (Query-Dependent with Bias Injection)
        # 查表获取 Bias: [B, S, D/2]
        bias_intra = embedding(modality_id, self.stream_bias[0]) 
        bias_global = embedding(modality_id, self.stream_bias[1])
        
        # 内容投影 + 身份偏置 -> 激活
        g_intra = sigmoid(self.gate_intra_proj(x) + bias_intra)
        g_global = sigmoid(self.gate_global_proj(x) + bias_global)
        
        # 4. 独立控制与拼接
        # 通过 Concat 保持特征维度的独立性
        final_output = torch.cat([
            out_intra * g_intra, 
            out_global * g_global
        ], dim=-1)
        
        return self.o_proj(final_output)
```

---

## 5. 实验与结论

老实说本工作的进展说不上顺利，**Split-Stream Intra-Global Gated Multimodal Attention**其实我很早就提出来了，在部分任务上展现出了一些优势但是主任务上劣化了，中间因为资源问题搁置了蛮久。近期受qwen Gated Attention[3]的启发和激励，正在艰难讨资源重启，祝我好运吧QAQ。

- 注意力分流：这是本工作最早的出发点，不过早期的版本差别也比较大，有空后面会先把一些失败的尝试放出来。
- 门控：门控的部分之前的实现没有这么优雅（核心思想合作差不多，gate分了3类，global、text-text、image-image，不过是通过修改softmax分数实现的，与flashattention不太兼容，导致比较慢），而且一度训不出效果，现在回过头来看，应该和Gated Attention[3]的实验观察类似：训练量小了可能很难产生效果。



## 6. 引用 (References)

[1] Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2024). "Visual Instruction Tuning." *NeurIPS 2024*.
[2] Bai, J., et al. (2023). "Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities." *arXiv preprint arXiv:2308.12966*.
[3] Qwen Team. (2025). "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free." *NeurIPS 2025*.
[4] Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*.

---

## 7. Cite Us

本工作为OpenLLMAI-research 2025年release的第3个工作，欢迎关注。不过。不同于前面的RFPO和WDB-GRPO，本工作目前尚未得到充分的实验验证，欢迎讨论，但谨慎参考。

**SIGMA**: Xianyu et al. (OpenLLMAI). . "SIGMA: Decoupling Intra-Modal Consistency and Global Context in Multimodal LLMs." Available at: [https://github.com/your-repo/sigma](https://github.com/your-repo/sigma)



前作：

https://openllmai.notion.site/rfpo

https://openllmai.notion.site/wdb-grpo