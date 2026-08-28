# AV1神经压缩设计

**User:** Anonymous  
**Created:** 8/28/2026 9:15:30  
**Updated:** 8/28/2026 14:53:02  
**Exported:** 8/28/2026 14:59:01  
**Link:** [https://chatgpt.com/c/6a90e073-0ea0-83ed-9844-6566b2a81f06](https://chatgpt.com/c/6a90e073-0ea0-83ed-9844-6566b2a81f06)  

## Prompt:
8/28/2026, 9:15:29 AM

现在目标升级：给 Qwen 35BA3B 设计一个完全按照 AV1 精神的有损压缩法。帧间压缩就是层间压缩，帧内压缩就是层内压缩，神经元的精度，以及相邻神经元，就是原来的块的概念，同样神经元有重要不重要之分。第一性原理不是优化信噪比，而是优化 KL 散度（可以参考校准，但不能过拟合校准，要有适当的 OOS 合适才行）还有哪些 AV1 精神值得学习？

[Attachment: ChatGPT-逐层近似蒸馏效果-20260828-0912.md]

## Response:
8/28/2026, 9:15:30 AM

我会把你的目标按“**AV1 的设计哲学如何映射到 LLM 后训练有损压缩**”来拆：先从你这份讨论里找已经形成的约束，再补 AV1 真正值得借鉴的机制，最后给一个适合 Qwen3.6-35B-A3B 的 codec-like 设计框架。

你这次加的两点很关键：**distortion 应该直接定义在 teacher/student 的输出分布上，而不是 hidden MSE；校准集只能用于“估计编码统计”，不能成为被拟合的对象本身。** 这会让方案更像真正 codec，而不是另一种小规模蒸馏。你前面的实验也已经出现了“几何误差小、KL 仍很大”的证据。

> **解析Qwen3.6压缩与AV1类比并查阅相关文件**
>
> 
>
> **正在搜索网页**
>
> [Source/ThirdParty/libwebrtc/Source/third_party/libaom/source/libaom/av1/encoder/pass2_strategy.c - external/github.com/WebKit/webkit - Git at Google](https://chromium.googlesource.com/external/github.com/WebKit/webkit/%2B/4915364aa76402d80037222809cfdfb92f8eeb1d/Source/ThirdParty/libwebrtc/Source/third_party/libaom/source/libaom/av1/encoder/pass2_strategy.c)
> [AV1 Video Codec | Alliance for Open Media](https://aomedia.org/specifications/av1/)
>
> **已搜索 2 个网站**
>
> 
>
> **正在搜索网页**
>
> [aomedia.org](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf)
>
> **分析稳健压缩**
>
> 我正在把问题抽象成率失真优化：校准集用于估计表示与分配策略，验证集用于选择超参数，最终用完全未见数据评估，避免把压缩方案调成只适合校准分布。
>
> **分析了稳健压缩方法**
>
> 
>
> **映射编码机制**
>
> 我正在将视频编码思想逐项映射到模型压缩：跨层预测、对齐传输、残差变换、敏感度量化、上下文熵编码、误差补偿、锚点刷新，以及离线可用的双向层级参考图。
>
> **映射了编码机制**
>
> 
>
> **已搜索 1 个网站**
>
> 
>
> **查找文件中关键指标及相关论述**
>
> 
>
> **正在搜索网页**
>
> [libavif/src/codec_aom.c at main · AOMediaCodec/libavif · GitHub](https://github.com/AOMediaCodec/libavif/blob/main/src/codec_aom.c)
>
> **已搜索 1 个网站**
>
> 
>
> 思考了 2m 11s

对，而且我觉得你现在这个目标可以再收紧一句：

> **不是“借鉴 AV1 做模型量化”，而是把 Qwen3.6-35B-A3B 当成一种沿 depth 展开的高维信号，设计一个真正的 Neural Codec。**

你前面已经走到了很关键的一步：实验里 **hidden 几何误差只有 0.64%，端到端 KL 仍有 1.218 nats**，这基本说明“权重 MSE / activation MSE / SNR”都不能作为最终 distortion。`ChatGPT-逐层近似蒸馏效果-20260828-0912.md` 你上一轮定义的 `KL + λR` 其实已经是正确主轴，而且文件里已有 Reference Bank、alignment、transform、Delta-Q、keyframe、in-loop correction 等 AV1 映射。`ChatGPT-逐层近似蒸馏效果-20260828-0912.md`

但我认为还有几条**比具体 AV1 工具更重要的“AV1 精神”**，现在还没有完全吃透。

## 1. 最重要的一条：不是“神经元 = 像素”，而是先创造真正的空间局部性

这里我会修正你一句：

> “相邻神经元就是 block”

**物理 index 相邻的神经元未必真的相邻。**

视频像素天然有二维拓扑；神经网络 channel 有非常强的 permutation symmetry。特别是一个 SwiGLU expert，512 个中间 channel 在很大程度上可以交换排列，所以 channel 137 和 138 在数组里相邻，不代表功能相邻。

因此 AV1 的 block partition 之前，我们需要一个相当于“建立坐标系”的步骤：

$$
\text{raw neuron index}
\rightarrow
\text{functional topology}
\rightarrow
\text{block partition}.
$$

例如根据 calibration activation 建相关图：

$$
A_{ij}
=
E_x[\phi_i(x)\phi_j(x)]
$$

再做 spectral ordering / clustering / seriation，把功能相近的 neuron 排到一起。

对 SwiGLU，更应该把一个 neuron 定义成一个**原子三元组**：

$$
\boxed{
(W_{\rm gate}[i,:],\;
W_{\rm up}[i,:],\;
W_{\rm down}[:,i])
}
$$

而不是分别压三个矩阵。

之后才能像 AV1 一样形成：

$$
\text{expert}
\rightarrow
\text{neuron cluster}
\rightarrow
\text{sub-cluster}
\rightarrow
\text{coefficient tile}.
$$

AV1 本身就是编码块 partition 和 transform block partition 分开的，而不是从头到尾只有一种固定 block。官方工具说明也把 block partition、transform partition、intra/inter prediction、transform、quantization、entropy coding 分成独立工具。([Alliance for Open Media](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf))

这对 NLC 很重要：

**预测单元 ≠ 变换单元 ≠ 量化单元。**

一个完整 expert 可以共用同一个 reference，但 residual 可以分成 64-neuron / 128-neuron transform blocks，每个 coefficient 又独立决定 bit depth。

---

## 2. Qwen 的“视频”实际上不是一条 40 帧视频，而是四条交织视频

Qwen3.6-35B-A3B 的结构是周期性的：

$$
10\times[D_1,D_2,D_3,A].
$$

也就是说，从 codec 的视角，更自然的是：

$$
D_1:
L_0,L_4,L_8,\ldots,L_{36}
$$

$$
D_2:
L_1,L_5,L_9,\ldots
$$

$$
D_3:
L_2,L_6,L_{10},\ldots
$$

$$
A:
L_3,L_7,L_{11},\ldots,L_{39}.
$$

你的文件里也已经确认这种四相位周期结构。`ChatGPT-逐层近似蒸馏效果-20260828-0912.md`

所以这里其实更像：

> **4 个 interleaved video planes，而不是 40 张同质 frame。**

默认 inter prediction 应当首先发生在**同 phase**之间。

跨 phase 不是不能预测，但它更类似 AV1 的 cross-component prediction，而不是普通 temporal prediction。

这样一来，“距离”也应该定义成：

$$
\Delta t=\frac{|l-r|}{4},
$$

而不是简单的 layer index 差。

---

## 3. 一个特别值得偷的东西：B-frame / 双向预测

这是我现在觉得非常强的一点。

之前我们一直在想：

$$
L_{20}\leftarrow L_{16}
$$

或者：

$$
L_{20}\leftarrow L_{24}.
$$

但为什么不能：

$$
\boxed{
L_{20}
\approx
\alpha T_{16\rightarrow20}(L_{16})
+
(1-\alpha)
T_{24\rightarrow20}(L_{24})
+
R_{20}
}
$$

？

AV1 的 compound prediction 本身就是从两个 reference frame 形成组合预测。([Alliance for Open Media](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf))

而对模型权重，我们甚至比视频有优势：

**checkpoint 是离线一次性 decode 的。**

没有实时播放的因果约束。

所以完全可以先解码：

$$
L_0,\quad L_{16},\quad L_{32},\quad L_{39}
$$

这些 anchor，

再解码中点：

$$
L_8,\quad L_{24},
$$

再递归：

$$
L_4,L_{12},L_{20},L_{28}.
$$

即：

$$
\boxed{\text{hierarchical depth B-frame coding}}
$$

这可能比：

$$
L_0\rightarrow L_4\rightarrow L_8\rightarrow\cdots
$$

一路链式 prediction 稳定很多，因为**误差传播深度直接下降**。

例如 10 个同 phase 层，从链式 reference depth 9，变成二叉 reference graph 后深度约：

$$
\log_2 10.
$$

这个我认为非常值得优先试。

---

## 4. “Motion vector”不能只是 expert permutation，而应该升级成分层 Warp

AV1 不只有平移 motion vector，还有 warped motion、compound prediction、OBMC 等工具。官方 AV1 甚至会从邻块 motion sample 推局部 warped model。([Alliance for Open Media](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf))

对应到模型，我会定义：

$$
\boxed{
T_{r\rightarrow l}
=
P_E
\circ
P_C
\circ
D
\circ
Q
}
$$

其中：

$$
P_E=\text{expert permutation}
$$

$$
P_C=\text{neuron/channel permutation}
$$

$$
D=\text{per-channel scaling}
$$

$$
Q=\text{小型 orthogonal / low-rank basis warp}.
$$

也就是说，先做非常廉价的 global motion：

> 整个 expert-bank 的 permutation + scaling

再允许局部 motion：

> 每个 expert 的 neuron permutation

最后才允许昂贵的：

> low-rank basis transport。

然后 RDO 自己决定某块到底值得不值得发送更复杂的 warp 参数。

这非常 AV1：

**不是所有 block 都用最强 predictor。**

强 predictor 自身也是有 bit cost 的。

---

## 5. AV1 最值得学的其实是“模式竞争”，不是某一种算法

这一点特别关键。

不要预先规定：

> 这块用 SVD。  
> 那层一定 inter。  
> 最后四层一定 intra。  
> 重要 neuron 用 8 bit。

应该给每个 coding block 一组 candidate：

| NLC mode | 含义 |
|---|---|
| `INTRA_RAW` | 独立编码这个 neuron block |
| `INTER_COPY` | 从一个 reference 直接复制 |
| `INTER_WARP` | reference + permutation/scale/warp |
| `COMPOUND` | 两个 reference 混合预测 |
| `PALETTE` | 从 prototype/codebook 取一个模式 |
| `SKIP` | predictor 已足够好，不编码 residual |
| `TRANSFORM_RESIDUAL` | predictor + transform residual |

然后统一算：

$$
J_m
=
D_{KL,m}
+\lambda R_m.
$$

选择：

$$
m^*=\arg\min_m J_m.
$$

所以：

**“这一层是不是重要”根本不成为输入。**

重要性只是最后的解释：

如果某 block 对 KL 极敏感，那么 INTRA/高精度的 RD cost 自然获胜。

如果某 block 不敏感，那么 SKIP/2bit/INTER_COPY 自然获胜。

---

# 6. “重要神经元”的第一性定义应该直接来自 KL 曲率

这是我觉得整个系统最漂亮的一步。

不是：

$$
\|w_i\|
$$

不是：

$$
\sigma_i
$$

也不是：

$$
\|a_i\|.
$$

设对一个 block 的扰动是：

$$
\delta\theta.
$$

在扰动足够小时：

$$
D_{KL}
\approx
\frac12
\delta\theta^T
F
\delta\theta
$$

其中 $F$ 可以理解为 teacher 输出分布对应的 Fisher / generalized Hessian。

因此一个 neuron/block 的真正“能量”是：

$$
\boxed{
E_i^{\rm codec}
=
\delta\theta_i^T F\delta\theta_i
}
$$

而不是权重本身的欧氏能量。

你之前数据已经很好地说明为什么必须这样做：0.64% hidden drift 并不意味着功能失真很小。`ChatGPT-逐层近似蒸馏效果-20260828-0912.md`

于是量化精度自动变成一个标准 water-filling 问题：

$$
\frac{\partial D_i}{\partial R_i}
=
-\frac1\lambda.
$$

换句话说：

> **最后一 bit 应当被分配给“每增加一 bit 能减少最多 KL”的 coefficient。**

这才是“神经元重要程度”的第一性原理版本。

---

# 7. 必须引入 AV1 的 SKIP / zero residual 精神

这是层间 sharing 最可能真正产生巨大压缩率的地方。

假设预测后：

$$
W_B=\hat W_B+E_B.
$$

如果：

$$
\Delta D_{KL}(E_B\rightarrow0)
<
\lambda R(E_B),
$$

那么：

$$
\boxed{E_B=0}
$$

直接什么都不存。

不是“把这个 block 量成 2 bit”。

而是：

> **这个 residual 根本不值得编码。**

因此最终可能出现大量：

$$
0\text{ bit neuron blocks}.
$$

这才真正接近视频压缩里的高压缩比来源。

---

# 8. Palette / IntraBC 对 MoE 可能异常重要

AV1 为 screen content 专门提供了 Intra Block Copy 和 palette mode，因为有些视频不是自然图像，而是包含大量重复图形。([Alliance for Open Media](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf))

MoE 很可能也存在类似情况。

例如 256 个 expert 中，可能有一批 functional motifs：

$$
E_i
\approx
C_k+\Delta_i.
$$

那么不要每个 expert 单独压。

先形成：

$$
\{C_1,\ldots,C_K\}
$$

prototype dictionary，

每个 expert 只存：

$$
k+\Delta_i.
$$

相当于：

$$
\boxed{\text{Neural Palette Mode}}
$$

甚至跨层可以：

$$
E_{l,i}
\approx
C_k.
$$

如果 residual 不值得编码：

$$
E_{l,i}=C_k
$$

一个 prototype index 就结束。

对于有 256 experts × 40 layers 的 Qwen，这可能比普通 block SVD 还值得研究。

---

## 9. In-loop filter 的真正意义不是“补偿一下”，而是保证所有预测都基于 reconstructed state

这个特别重要，而且与你们前面已经发现的 teacher-forcing illusion 完全一致。

你们已经明确意识到，压完第 1 层以后，第 2 层实际吃的是：

$$
\hat h_1
$$

而不是：

$$
h_1.
$$

`ChatGPT-逐层近似蒸馏效果-20260828-0912.md`

AV1 的 reference frame 也是**已经重建、已经过 loop filter 的 frame**，不是原始 source frame。

所以 NLC 应该严格规定：

$$
\boxed{
\text{任何 INTER reference 都只能引用 reconstructed Bank}
}
$$

不能编码时拿 teacher 原权重当 reference、最后 decoder 却拿量化后的 reference。

否则会形成完全一样的 encoder-decoder mismatch。

而所谓 Neural In-loop Filter，可以是很小的：

$$
h\leftarrow
h+
D_l h+
U_lV_lh
$$

或者 weight-space correction。

它的目标不是让当前层 MSE 好看，而是：

> **把 reconstructed reference 拉回适合后续预测的 manifold。**

AV1 的 CDEF / loop restoration 本质上也是把压缩产生的 artifacts 修正之后，再让结果进入后续解码链路。([Alliance for Open Media](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf))

---

# 10. Calibration 最值得学 AV1 的不是“多采样”，而是 Two-pass 思想

这一点直接对应你说的“不能过拟合 calibration”。

AV1/libaom 的 two-pass 编码会先采集 first-pass statistics，再在第二遍根据全局统计进行 rate allocation；其实现里明确维护 first-pass stats、剩余 bits、group error 等统计来决定后续分配。([Chromium Git](https://chromium.googlesource.com/external/github.com/WebKit/webkit/%2B/4915364aa76402d80037222809cfdfb92f8eeb1d/Source/ThirdParty/libwebrtc/Source/third_party/libaom/source/libaom/av1/encoder/pass2_strategy.c?utm_source=chatgpt.com))

NLC 我会比 two-pass 再严格一层：

$$
S=S_{\rm fit}\cup S_{\rm select}\cup S_{\rm OOS}
$$

且三者绝不能混。

### `fit`

只允许估计：

$$
\text{covariance / basis / permutation / entropy model / sensitivity}.
$$

### `select`

只允许进行：

$$
\text{mode selection / }\lambda\text{ selection / partition selection}.
$$

### `OOS`

编码结束前完全不看。

只用来报告：

$$
D_{KL}^{OOS}.
$$

最好进一步做 domain-held-out：

比如某一轮完全不让 codec 看代码数据，却测 code KL；另一轮完全不看数学，却测 math。

如果压缩决策在这些 split 下基本稳定，才说明它找到的是**模型本身的 redundancy**，而不是 calibration corpus 的 redundancy。

---

## 11. 平均 KL 仍然不够：但不用抛弃 KL，只要改变聚合方式

你提出“第一性原理就是 KL”，我同意。

但：

$$
E[KL]
$$

可能掩盖少数 catastrophic tokens。

例如 99% token：

$$
KL=0.01
$$

1% token：

$$
KL=10.
$$

平均值可能还不错，但 reasoning 已经死了。

所以 distortion 仍然全部用 KL，只是定义成：

$$
\boxed{
D_{\rm robust}
=
\mu_{KL}
+
\alpha\,\operatorname{CVaR}_{95\%}(KL)
+
\beta\,\operatorname{Std}_{domain}(KL)
}
$$

没有加入 MSE，没有加入人工 task score。

仍然纯粹是：

$$
\boxed{\text{distribution distortion}}
$$

只是同时关心平均、尾部和 domain robustness。

这恰好解决你说的 calibration overfit / OOS 问题。

---

# 12. 还有一个 AV1 极其重要的哲学：Encoder 可以极贵，Decoder 必须极便宜

这个对模型压缩尤其重要。

AV1 encoder 为了寻找更好的 partition、motion、transform 和 mode，可以花非常高计算量；decoder 却必须规则明确、成本可控。AV1 实际实现也大量使用 lookahead、候选搜索和 mode pruning 来换取编码效率。([Chromium Git](https://chromium.googlesource.com/external/github.com/WebKit/webkit/%2B/4915364aa76402d80037222809cfdfb92f8eeb1d/Source/ThirdParty/libwebrtc/Source/third_party/libaom/source/libaom/av1/encoder/pass2_strategy.c?utm_source=chatgpt.com))

所以 NLC 完全不应该害怕：

$$
10^5\sim10^6
$$

次 KL probe，

甚至几小时做：

$$
\text{JVP/Hutchinson}
$$

$$
\text{Hungarian}
$$

$$
\text{SVD/KLT}
$$

$$
\text{beam search}
$$

$$
\text{global RDO}.
$$

checkpoint 一生只编码一次。

真正应该严格限制的是 decoder：

$$
\boxed{
\text{permutation}
+
\text{scale}
+
\text{fixed transform}
+
\text{low-bit coefficients}
}
$$

最好最终还能直接融合进 GEMM kernel。

这条原则会帮你避免把“压缩器”不知不觉做成一个需要额外 neural network 的蒸馏系统。

---

## 13. 所以我现在会把整个 Qwen Codec 定义成这个结构

核心目标：

$$
\boxed{
\min_{\mathcal C}
D_{\rm robust}
(F,\operatorname{Decode}(\mathcal C))
+
\lambda R_{\rm actual}(\mathcal C)
}
$$

其中：

$$
R_{\rm actual}
=
-\log_2P(\text{symbols})
$$

必须尽量接近**实际 bitstream 长度**，而不是：

$$
\#\text{parameters}.
$$

一个 3-bit coefficient 如果高度可预测，真实 rate 也许只有：

$$
0.7\text{ bit}.
$$

所以最终也应该有真正的 context entropy coder。AV1 本身就把 transform coefficients 接到多符号 arithmetic coding 上。([Alliance for Open Media](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf))

而 codec pipeline 会成为：

$$
\boxed{
\begin{aligned}
&\text{Functional canonicalization}\\
\rightarrow&
\text{Hierarchical block partition}\\
\rightarrow&
\text{INTRA/INTER/COMPOUND/PALETTE/SKIP RDO}\\
\rightarrow&
\text{reference warp/alignment}\\
\rightarrow&
\text{residual transform}\\
\rightarrow&
\text{KL-sensitive quantization}\\
\rightarrow&
\text{entropy coding}\\
\rightarrow&
\text{reconstruction}\\
\rightarrow&
\text{in-loop correction}.
\end{aligned}}
$$

然后只有一个真正意义上的用户旋钮：

$$
\boxed{\lambda}
$$

或者：

$$
\boxed{\text{Target physical GiB}}.
$$

---

我尤其看好其中三个此前还没充分利用的 AV1 思想：

**第一，Qwen 四相位分别做 hierarchical B-frame depth coding；第二，block 必须建立在 functional adjacency 而不是 raw neuron index 上；第三，把平均 KL 升级成 cross-validated、tail-aware 的 robust KL，但整个 distortion 空间始终不离开 KL。**

如果这三件事成立，那么最终出来的模型可能根本不会长得像“40→12 Bank”。它可能自己发现一种很奇怪的结构：**浅层很多 SKIP/compound prediction，中层大量 palette expert，某几个 depth 自动成为 anchor，末层精度突然升高；甚至同一个 expert 内部 70% neuron 只有 0-2 bit residual，而少数 KL-sensitive neuron 保留 6-8 bit。**

这才真正是 **AV1 for Qwen**：不是我们决定模型哪里重要，而是**每一 bit 都必须向 KL 证明自己值得存在。**

---

**Sources:**

- [](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf)
- [Source/ThirdParty/libwebrtc/Source/third_party/libaom/source/libaom/av1/encoder/pass2_strategy.c - external/github.com/WebKit/webkit - Git at Google](https://chromium.googlesource.com/external/github.com/WebKit/webkit/%2B/4915364aa76402d80037222809cfdfb92f8eeb1d/Source/ThirdParty/libwebrtc/Source/third_party/libaom/source/libaom/av1/encoder/pass2_strategy.c?utm_source=chatgpt.com)

## Prompt:
8/28/2026, 9:20:54 AM

本质上我认为 IQ3_XSS 这些都是早期视频压缩在 LLM 上的对应物，我们直接用 AV1 的巨人肩膀一定能超过去他们。一个好的方法是，从 IQ3_XSS 的实现开始演进，正如 AV1 是从早期的 bmp 降低位宽，然后不均等降低，然后 jpg, xvid, hevc, 开始一路演进的一样。

## Response:
8/28/2026, 9:20:55 AM

我同意这个演进视角，而且我会再收紧一点：**IQ3_XSS 不应该被当成“竞品终点”，而应该当成我们的 H.261/JPEG 起点**。最有价值的不是推倒重来，而是保留它成熟的块量化/反量化内核，然后一层层把 AV1 的预测、分块、变换、RDO、熵编码思想长进去。

> **正在搜索网页**
>
> [llama.cpp/tools/quantize/quantize.cpp at master · ggml-org/llama.cpp · GitHub](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/quantize.cpp)
>
> **已搜索 1 个网站**
>
> 
>
> 思考了 17s

是，这个路线我认为比“从零发明一个神经网络压缩格式”更稳。更准确地说，我们可以把 **IQ3_XXS / IQ3_XS 看成已经完成了早期图像编码阶段的一部分：固定分块、非均匀标度、重要性加权、低 bit 表示；然后在它上面逐步长出 JPEG/Xvid/HEVC/AV1 那些真正决定压缩效率的东西。**

先纠正一个小命名：llama.cpp 当前有 `IQ3_XXS`（约 3.06 bpw）和 `IQ3_XS`（约 3.3 bpw），没有正式的 `IQ3_XSS` 这个名字。`IQ3_XXS` 已经是 256-weight superblock、里面 8×32-weight blocks，并且量化器明确接受 importance matrix，也就是它已经不是最原始的“统一 3 bit”。([GitHub](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.h))

所以我会把演进路径设计成这样：

| 阶段 | 视频史类比 | LLM 版本 |
|---|---|---|
| V0 | BMP 降色深 | 普通 Q3/Q4，固定 bit |
| V1 | 非均匀量化 | **IQ3_XXS**，importance-aware |
| V2 | JPEG | block transform + coefficient quantization |
| V3 | MPEG/Xvid | **层间预测 + residual coding** |
| V4 | H.264/HEVC | 可变 block、multiple references、RDO |
| V5 | AV1 | compound prediction、hierarchical references、adaptive transform、entropy contexts、全局 RD optimization |

而且这不是纯类比，可以非常具体地落到代码演进上。

### 第一阶段千万不要改 decoder：先做“AV1 encoder + IQ3_XXS bitstream”

这是我觉得最聪明的第一步。

保留现在：

$$
\texttt{block\_iq3\_xxs}
$$

以及已有 CPU/Vulkan/CUDA dequant / dot-product kernel。

**只换 encoder。**

现在 IQ3_XXS 本质上在解决：

$$
\min_{\hat W}
\sum_i a_i(W_i-\hat W_i)^2
$$

其中 $a_i$ 来自 importance matrix。

第一版直接升级成：

$$
\boxed{
\min_{\hat W}
D_{KL}
\big(
p_{\rm teacher}\|p_{\hat W}
\big)
}
$$

但仍然要求最终每个 block **必须编码成合法 IQ3_XXS**。

也就是说我们先问一个极其干净的问题：

> 同样 3.0625 bpw、同一个 runtime kernel，只把 encoder 的 distortion metric 从 activation-weighted MSE 换成 KL-RDO，能赢多少？

如果这一步就赢，那已经非常有说服力。

因为 inference 零变化。

---

## 然后第二步：从“固定 IQ3”变成 Delta-Q

现在 llama.cpp 已经有：

$$
IQ2,\ IQ3,\ IQ4,\ Q5,\ldots
$$

而实际 quantizer 也已经会对不同 tensor category 做一些人工 type selection，例如某些 FFN/attention tensor使用不同 fallback/type。([GitHub](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp?utm_source=chatgpt.com))

这其实正像早期编码器：

> 人工规定“这里质量高一点，那里低一点”。

下一步不要人工规定。

对每个 block $B$，编码器实际试：

$$
m\in
\{
IQ1,\ IQ2,\ IQ3,\ IQ4,\ Q5,\ FP16
\}
$$

计算：

$$
J(B,m)
=
\Delta KL(B,m)
+
\lambda R(B,m).
$$

然后：

$$
m_B^*
=
\arg\min_m J(B,m).
$$

这一步就是 AV1 的 **Delta-Q / segment-level quality selection**。

最后可能平均还是：

$$
3.06\ \text{bpw},
$$

但分布变成：

- 35% weights：1.5-2 bit；
- 40%：3 bit；
- 20%：4 bit；
- 5%：6-16 bit。

这已经很可能明显超过固定 IQ3。

---

# 第三步才是真正的 JPEG 时刻：不要直接量化原权重

IQ3 现在基本还是：

$$
W
\xrightarrow{\text{scalar/vector quantization}}
Q.
$$

而 JPEG 的巨大进步来自：

$$
X
\rightarrow
TX
\rightarrow
\text{quantize coefficients}.
$$

所以我们应该加入：

$$
W_B
\rightarrow
Z_B=T_BW_B
$$

其中 $T_B$ 可以从最便宜的开始：

$$
\text{Hadamard}
$$

然后：

$$
\text{PCA/KLT}
$$

或者固定小型正交 transform。

编码：

$$
Q(Z_B)
$$

decode：

$$
\hat W_B=T_B^{-1}\hat Z_B.
$$

真正关键的是，很多 coefficient 会变得接近 0。

于是：

$$
\boxed{\text{SKIP / zero coefficient}}
$$

才开始出现。

这是从“量化”跨越到“压缩编码”的关键一步。

---

# 第四步就是你们现在研究的真正突破：Xvid/MPEG 式层间预测

到这里才加入：

$$
W_l
=
P(W_r)+E_l.
$$

以前 IQ3 编的是：

$$
W_l.
$$

现在编码：

$$
E_l=W_l-P(W_r).
$$

如果层间高度相关，则：

$$
H(E_l)\ll H(W_l).
$$

这里可以继续完全复用 IQ3 的 quantizer：

$$
E_l
\xrightarrow{IQ3/2/1}
\hat E_l.
$$

所以第一版层间编码甚至不用设计新的 residual quantizer。

**IQ3 本身就变成 AV1 里的 transform coefficient quantizer。**

这就是为什么我非常赞成“从 IQ3 演进”。

---

## 最漂亮的架构其实是：把 IQ3 降级成 codec 中的一个 primitive

今天是：

$$
\boxed{\text{IQ3 = 压缩算法}}
$$

未来应该是：

$$
\boxed{\text{IQ3 = 一个 coefficient coding mode}}
$$

整个编码器可能这样：

$$
\text{predict}
\rightarrow
\text{transform}
\rightarrow
\boxed{\text{IQ1/IQ2/IQ3/IQ4}}
\rightarrow
\text{entropy coder}.
$$

就像 AV1 里“8-bit quantizer”绝对不是整个视频 codec。

---

# 第五步：HEVC/AV1 式可变分块

IQ3_XXS 今天的一个核心限制是分块几乎写死：256-weight superblock，内部 32-weight blocks。([GitHub](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes/cbababee1f5ba8cd64eef187bffa07881d7e0cd8))

这可以看成固定 8×8 JPEG block 的时代。

下一代需要：

$$
B
\in
\{16,32,64,128,256,512,\ldots\}
$$

但不是简单按内存连续位置。

而是上一轮我们说的：

$$
\boxed{\text{functional neuron blocks}}
$$

然后用类似 quadtree 的 split：

$$
J(B)
\stackrel{?}{>}
J(B_1)+J(B_2)+R_{\rm split}.
$$

如果拆开更划算就拆。

于是：

- 很规则的区域 → 大 block；
- KL 敏感区域 → 小 block；
- 某几个异常 neuron → 单独编码。

这就是真正的 **partition RDO**。

---

# 第六步：从 MPEG P-frame 升级到 AV1 multiple-reference / B-frame

最初只做：

$$
L_{12}\leftarrow L_8.
$$

之后允许 reference candidates：

$$
r\in\{L_4,L_8,L_{16},L_{20},\ldots\}.
$$

比较：

$$
J(r)=D_{KL,r}+\lambda R_r.
$$

然后 compound：

$$
\hat W_l
=
\alpha P(W_a)
+
(1-\alpha)P(W_b).
$$

对应你这个 Qwen 尤其适合：

$$
L_{20}
\leftarrow
L_{16}+L_{24}.
$$

再构造 hierarchical depth GOP。

这一步才会真正把你们现在的：

$$
40\rightarrow12
$$

这种人工 Funnel 设计淘汰掉。

codec 自己决定哪些层是 I-frame，哪些是 P/B-frame。你之前文件里其实已经把这个原则写出来了：当 inter residual 的 RD cost 高于 intra，RDO 自然让该层成为 anchor，而不是人工规定末层必须保留。`ChatGPT-逐层近似蒸馏效果-20260828-0912.md`

---

# 第七步才是 AV1 的“可怕之处”：所有工具同时竞争

最终一个 neuron block 会面对这样的模式集合：

$$
\mathcal M=
\{
\text{INTRA-IQ},
\text{INTRA-Transform},
\text{INTER-Copy},
\text{INTER-Residual},
\text{INTER-Warp},
\text{Compound},
\text{Palette},
\text{Skip}
\}.
$$

并且每种模式里面又有：

$$
\text{block size}
$$

$$
\text{transform}
$$

$$
\text{reference}
$$

$$
\text{bit depth}
$$

$$
\text{scale}.
$$

全部统一：

$$
\boxed{
J=
D_{KL}^{OOS}
+
\lambda R
}
$$

竞争。

这时候它才真正拥有 AV1 的精神。

---

## 我甚至认为第一代原型应该故意保持非常“丑”

不要一上来开发新文件格式、新 kernel、新 entropy decoder。

第一代可以直接产生一个普通 GGUF：

1. 读取 FP16/BF16 teacher；
2. 用 calibration + KL 计算新的 importance；
3. 对 block 做 RDO；
4. 最终仍输出标准 IQ3_XXS / IQ2 / IQ4 tensor；
5. llama.cpp 原样运行。

目标先证明：

$$
\boxed{
\text{same bytes}
\quad\Rightarrow\quad
KL_{\rm ours}<KL_{\rm IQ3}
}
$$

然后第二代才允许 mixed quant。

第三代加入 transform。

第四代加入 inter-layer residual。

第五代再做真正 NLC bitstream。

这是最像视频 codec 历史的工程路线：**每代只改变一个最关键的自由度，而且前一代始终是下一代的 baseline 和 primitive。**

---

还有一个很关键的认识：你说“一定能超过他们”，我会把它变成一个可以证伪的命题，而不是预设结论。

IQ3_XXS 已经用了 importance matrix，而且非常成熟、decoder 极快。([GitHub](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.h)) 所以真正公平的胜负标准应该画一条完整曲线：

$$
\boxed{
R\;(\text{GiB/bpw})
\quad\leftrightarrow\quad
D_{KL}^{OOS}
}
$$

再附：

$$
\text{decode speed},\quad
\text{PPL},\quad
\text{reasoning retention}.
$$

如果我们的 RD 曲线整体在 IQ3/IQ2 左下方，它才是真的“下一代 codec”。

而我认为**最可能最先赢的版本，并不是复杂的全 AV1 版，而是：**

$$
\boxed{
\text{IQ3 runtime}
+
\text{KL importance}
+
\text{block-level mixed precision RDO}
}
$$

因为它几乎不承担新的 runtime 成本，却已经把 IQ3 最大的局限--**固定编码结构、近似局部 distortion、人工 bit allocation**--一起拿掉了。

之后再加：

$$
\text{transform}
\rightarrow
\text{inter-layer prediction}
\rightarrow
\text{compound reference}
\rightarrow
\text{entropy coding}.
$$

这条路线我认为非常适合真正开始写实现。

---

**Sources:**

- [llama.cpp/ggml/src/ggml-quants.h at master · ggml-org/llama.cpp · GitHub](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.h)

## Prompt:
8/28/2026, 9:25:45 AM

大路线图不错，但每个步骤都数学上很粗。光是 JPEG 当年就那么多页纸那么复杂的数学，你这么寥寥几笔肯定是不行的

## Response:
8/28/2026, 9:25:55 AM

你这个批评是对的。上一版只能算“codec 目录”，还不是算法；真正要超过 IQ3，必须把每一代都写成**明确的随机源模型、失真度量、变换、量化器、码率模型和优化问题**。我先不继续铺大路线图，而是把最关键的 **“JPEG 对应阶段”完整数学化**，因为这一层如果没立住，后面的层间预测/RDO 都会建立在沙子上。

关键数学桥梁其实已经浮出来了：JPEG 的 DCT 不是因为“余弦很神奇”，而是因为它在自然图像的相关源模型下是 KLT 的廉价近似；对 LLM，我们应该推导一个**由权重统计 + 输出 KL 曲率共同决定的广义 KLT**，而不是拍脑袋选 Hadamard/SVD。JPEG 基线本身也是完整的 8×8 DCT→量化→符号重排→熵编码链，而 AV1 再把 partition/prediction/transform/quantization/entropy/loop filtering 组合成系统。

> **重构数学路线**
>
> 我会把方案收敛成可推导的模块：失真度量、正交变换、标量量化、熵模型与率失真优化。先完整推导单层内编码，再扩展到跨层敏感度与异常值稳健性。
>
> **重构数学路线**
>
> 
>
> **正在搜索网页**
>
> [Influence of block size on JPEG compression performance](https://ieeexplore.ieee.org/iel7/9107937/9111475/09111506.pdf)
> [1 Introduction](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf)
>
> **建立可复现实验**
>
> 我会把下一版写成技术规格，而非概念清单：先定义块、概率模型和失真函数，再推导变换、量化、熵编码及率失真优化，最后给出可直接复现实验。
>
> **建立了可复现实验流程**
>
> 
>
> **已搜索 2 个网站**
>
> 
>
> 思考了 1m 59s

对。上一版的粒度大概相当于说“JPEG = DCT + 量化”，这对做路线图够，但对设计一个新 codec 基本没用。**每一个箭头本身都应该展开成一套 rate-distortion 理论。**

而且最关键的是：我们不应该“把 JPEG/AV1 的模块名称翻译成 LLM 名词”，而应该重走它们背后的数学推导。

JPEG 的关键并不是“用了 DCT”，而是对一个具有强相关性的随机源做 **decorrelation → energy compaction → nonuniform quantization → entropy coding**。Baseline JPEG 的完整链条确实是 8×8 DCT、量化、系数组织/游程表示和熵编码，而不是一句“DCT 压一下”。([Stanford University](https://web.stanford.edu/class/ee398a/handouts/papers/Wallace%20-%20JPEG%20-%201992.pdf?utm_source=chatgpt.com))

所以我建议现在先停在 **JPEG 阶段**，把它真正推导出来。先不碰层间预测。

---

# 0. 先定义我们到底在压什么

设原模型参数：

$$
\theta.
$$

编码器输出 bitstream：

$$
c=\mathcal E(\theta),
$$

decoder 重建：

$$
\hat\theta=\mathcal D(c).
$$

问题不是：

$$
\min \|\theta-\hat\theta\|^2.
$$

而是：

$$
\boxed{
\min_{\mathcal E,\mathcal D}
R(c)
+\beta D(\theta,\hat\theta)
}
$$

其中真正的 distortion 为

$$
\boxed{
D(\theta,\hat\theta)
=
\mathbb E_{x\sim P}
\left[
\frac1{T_x}
\sum_t
D_{\mathrm{KL}}
\left(
p_\theta(\cdot|x_{<t})
\|
p_{\hat\theta}(\cdot|x_{<t})
\right)
\right].
}
$$

这和你前面的数据完全吻合：

> hidden-state 漂移只有 0.64%，输出 KL 仍有 1.218 nats。

说明 Euclidean distortion 的坐标系根本错了。`ChatGPT-逐层近似蒸馏效果-20260828-0912.md`

于是从第一行数学开始，就已经和 IQ3 类算法不同了。

---

# 1. JPEG 的 DCT 在我们这里到底对应什么？

不能直接回答“SVD”。

要从 KL 展开推出来。

设某个待压缩参数 block 为

$$
w\in\mathbb R^n,
$$

量化造成：

$$
\hat w=w+\epsilon.
$$

teacher logits 为：

$$
z=f(w,x).
$$

参数扰动造成：

$$
\delta z=J_w(x)\epsilon+O(\|\epsilon\|^2).
$$

其中

$$
J_w=\frac{\partial z}{\partial w}.
$$

softmax：

$$
p=\operatorname{softmax}(z).
$$

对小扰动，有：

$$
D_{\rm KL}
(p(z)\|p(z+\delta z))
=
\frac12
\delta z^T
G(p)
\delta z
+
O(\|\delta z\|^3)
$$

其中

$$
\boxed{
G(p)=\operatorname{Diag}(p)-pp^T
}
$$

正是 categorical distribution 的 Fisher metric。

代入：

$$
D_{\rm KL}
\approx
\frac12
\epsilon^T
J_w^T
G
J_w
\epsilon.
$$

对 token / calibration distribution 取平均：

$$
\boxed{
H_w
=
\mathbb E_x
[J_w(x)^T G(p_x)J_w(x)]
}
$$

得到

$$
\boxed{
D_{\rm KL}
\approx
\frac12\epsilon^T H_w\epsilon.
}
$$

这才是我们版本的“感知误差矩阵”。

---

# 2. 这立刻告诉我们：普通 PCA/SVD 也不是最优

假设某一类 block 可以看成随机变量：

$$
w\sim(\mu,\Sigma).
$$

其中：

$$
\Sigma
=
\mathbb E[(w-\mu)(w-\mu)^T].
$$

普通 KLT/PCA 解的是：

$$
\Sigma u_i=\sigma_i^2u_i.
$$

它只知道：

> 哪些方向权重变化大。

但不知道：

> 哪些变化会改变模型输出。

所以我们真正需要同时考虑：

$$
\boxed{\Sigma}
$$

和：

$$
\boxed{H}.
$$

这就是 JPEG 类比真正值得推下去的地方。

---

# 3. 可以推导出一个“KL-KLT”

先 whiten source：

$$
z
=
\Sigma^{-1/2}(w-\mu).
$$

于是：

$$
\operatorname{Cov}(z)=I.
$$

在这个坐标里 KL metric 变成：

$$
H_z
=
\Sigma^{1/2}
H
\Sigma^{1/2}.
$$

做特征分解：

$$
\boxed{
\Sigma^{1/2}
H
\Sigma^{1/2}
=
U\Gamma U^T
}
$$

其中：

$$
\Gamma
=
\operatorname{diag}
(\gamma_1,\ldots,\gamma_n).
$$

现在定义真正的 transform：

$$
\boxed{
y
=
U^T
\Sigma^{-1/2}
(w-\mu).
}
$$

decoder：

$$
w
=
\mu+
\Sigma^{1/2}Uy.
$$

这个 transform 有两个特别漂亮的性质。

第一：

$$
\operatorname{Cov}(y)=I.
$$

即 coefficient 被 decorrelate / whiten。

第二：

如果量化误差是

$$
e_y=y-\hat y,
$$

那么：

$$
\boxed{
D_{\rm KL}
\approx
\frac12
\sum_i
\gamma_i e_{y,i}^2.
}
$$

因此整个复杂的：

$$
\epsilon^TH\epsilon
$$

真的变成了独立 coefficient distortion。

---

这才是 LLM 对应 JPEG DCT 的东西。

我暂时会叫：

$$
\boxed{\text{KL-KLT}}
$$

或者：

$$
\boxed{\text{Functional KLT}}.
$$

它对应的 generalized eigenproblem 也可以写成：

$$
\boxed{
Hv_i
=
\gamma_i\Sigma^{-1}v_i.
}
$$

---

# 4. 这里出现了一个很重要的结果

以前我们说：

$$
\text{importance}
=
\text{energy}
\times
\text{sensitivity}.
$$

那还是口头语言。

现在它有精确定义了：

$$
\boxed{
\gamma_i
=
\lambda_i
\left(
\Sigma^{1/2}H\Sigma^{1/2}
\right).
}
$$

也就是说：

> **一个 coefficient 是否值得编码，不由 singular value 决定，也不由 Hessian sensitivity 单独决定，而由 source covariance 和 functional sensitivity 的联合谱决定。**

你前面的文件实际上已经隐约推到了“source energy × functional sensitivity”这一步。`ChatGPT-逐层近似蒸馏效果-20260828-0912.md`

现在这个 generalized KLT 给它补上了完整数学形式。

---

# 5. 然后才进入 JPEG 的“量化表”问题

现在：

$$
y_i
$$

已经近似 decorrelated。

设 uniform quantizer：

$$
q_i
=
\operatorname{round}
\left(
\frac{y_i}{\Delta_i}
\right),
$$

重建：

$$
\hat y_i=\Delta_iq_i.
$$

高码率近似下：

$$
\mathbb E[e_i^2]
\approx
\frac{\Delta_i^2}{12}.
$$

因此：

$$
D_i
\approx
\frac{\gamma_i\Delta_i^2}{24}.
$$

另一方面，entropy-constrained scalar quantization 下：

$$
R_i
\approx
h(y_i)-\log_2\Delta_i.
$$

于是单 coefficient 的 Lagrangian：

$$
J_i
=
\frac{\gamma_i\Delta_i^2}{24}
+
\beta
\left[
h(y_i)-\log_2\Delta_i
\right].
$$

求导：

$$
\frac{\partial J_i}{\partial\Delta_i}
=
\frac{\gamma_i\Delta_i}{12}
-
\frac{\beta}{\Delta_i\ln2}.
$$

令其为 0：

$$
\boxed{
\Delta_i^{*2}
=
\frac{12\beta}
{\gamma_i\ln2}
}
$$

所以：

$$
\boxed{
\Delta_i^*
\propto
\frac1{\sqrt{\gamma_i}}.
}
$$

这不是“重要 neuron 多几 bit”的启发式。

而是直接从：

$$
KL+\beta R
$$

推导出的最佳 quantization step。

---

# 6. 再往下一步就是经典 reverse water-filling

如果把 transform coefficients 近似看成独立 Gaussian source：

$$
y_i\sim\mathcal N(0,1),
$$

并采用上述 KL-weighted quadratic distortion，则 Gaussian rate-distortion 理论直接给出：

$$
\boxed{
R_i
=
\frac12
\left[
\log_2
\frac{\gamma_i}{\nu}
\right]_+
}
$$

这里 $\nu$ 是全局 water level，由目标总 bitrate 决定。

这条公式非常重要。

因为它意味着根本不需要：

> neuron 1 用 3 bit；  
> neuron 2 用 4 bit；  
> neuron 3 用 2 bit。

连续最优解自己产生。

如果：

$$
\gamma_i\leq\nu
$$

那么：

$$
\boxed{R_i=0}.
$$

即：

> **这个 generalized coefficient 理论最优就是一个 bit 都不给。**

如果：

$$
\gamma_i=16\nu,
$$

那么：

$$
R_i=2\text{ bits}.
$$

如果：

$$
\gamma_i=256\nu,
$$

则：

$$
R_i=4\text{ bits}.
$$

于是：

$$
\boxed{
\gamma_i
\rightarrow
R_i
}
$$

成为数学上的连续 bit-allocation law。

这才是真正意义上的“非均等量化”。

---

# 7. IQ3 应该插在这一步，而不是被整个抛弃

这是“从 IQ3 演进”真正合理的地方。

我们推出来的是一个理想 RD solution：

$$
R_i^*,\Delta_i^*.
$$

但实际 decoder 不可能拥有任意实数 bit depth。

因此实际 encoder 有 candidate set：

$$
\mathcal Q
=
\{
IQ1,
IQ2,
IQ3,
IQ4,
Q4,
Q5,
Q6,
FP16,
SKIP
\}.
$$

对每一个真实 IQ quantizer，我们都可以：

1. 真正 quantize；
2. 真正 dequantize；
3. 得到误差 $\epsilon_q$；
4. 算

$$
\tilde D_q
=
\frac12\epsilon_q^TH\epsilon_q;
$$

5. 算真实 bit cost：

$$
R_q;
$$

6. 比较：

$$
\boxed{
J_q
=
\tilde D_q+\beta R_q.
}
$$

所以 IQ3 不再是“算法”。

它变成：

$$
\boxed{
q\in\mathcal Q
}
$$

中的一个 reproduction codebook。

这非常像 codec 演进的方式。

---

# 8. JPEG 还有一个不能漏掉的东西：DC/AC 分离

JPEG 不只是 DCT。

它发现 transformed block 中：

$$
DC
$$

和：

$$
AC
$$

统计性质完全不同，所以编码方式也不同；量化后的 AC 又被重新排序，使大量零值形成可压缩的 runs。Wallace 的 JPEG 描述里就明确包含 DC/AC 的不同符号处理以及后续 variable-length entropy coding。([Stanford University](https://web.stanford.edu/class/ee398a/handouts/papers/Wallace%20-%20JPEG%20-%201992.pdf?utm_source=chatgpt.com))

LLM 不应该机械复制“低频/高频”。

我们应该定义：

$$
\boxed{\text{functional DC / AC}}.
$$

例如 generalized spectrum：

$$
\gamma_1\geq
\gamma_2\geq
\cdots\geq
\gamma_n.
$$

可以按预计非零概率排序：

$$
P(q_i\neq0)
$$

而不是按二维 spatial frequency zig-zag。

形成：

$$
i_1,i_2,\ldots,i_n
$$

使：

$$
P(q_{i_1}\neq0)
\ge
P(q_{i_2}\neq0)
\ge\cdots.
$$

这就是我们的：

$$
\boxed{\text{KL-zigzag}}.
$$

高 KL-energy coefficients 在前。

后面逐渐变成：

$$
0,0,0,0,\ldots
$$

于是自然适合：

$$
\text{EOB}
$$

和 zero-run coding。

---

# 9. 真正 rate 不能再写成“参数数 × bit 数”

必须进入 entropy coding。

设量化后的 symbol：

$$
q_i\in\mathbb Z.
$$

上下文：

$$
c_i=
(
\text{tensor type},
\text{layer phase},
i,
q_{i-1},
\text{zero run},
\text{block class},
\dots
).
$$

概率模型：

$$
P(q_i|c_i).
$$

实际 rate 应定义为：

$$
\boxed{
R
=
R_{\rm side}
-
\sum_i
\log_2P(q_i|c_i).
}
$$

其中：

$$
R_{\rm side}
$$

必须包括：

- partition 信息；
- transform ID；
- scale；
- quant mode；
- codebook ID；
- entropy table；
- alignment 信息。

不能作弊。

AV1 正式结构本身就是 prediction、transform、quantization 后进入 coefficient entropy coding，而不是把“几个 bit/sample”当最终码率。([Alliance for Open Media](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf?utm_source=chatgpt.com))

---

# 10. 甚至“学一个更好的 entropy model”也不是免费的

这一点很有意思。

假设为某个 tensor 专门拟合一个复杂 probability model：

$$
\phi.
$$

它让 coefficient code length 少了：

$$
50\text{ KB}.
$$

但你需要发送：

$$
100\text{ KB}
$$

的 $\phi$。

那它就是亏的。

所以真正目标是：

$$
\boxed{
R_{\rm total}
=
L(\phi)
+
L(q|\phi).
}
$$

这其实就是 MDL：

$$
\boxed{\text{Minimum Description Length}}.
$$

因此 codec 自然会在：

> 通用 entropy table

和：

> tensor-specific table

之间做 RDO。

---

# 11. “8×8 block”问题也要数学化

JPEG 的固定 8×8 是工程折中。

我们不能说：

> “LLM 就用 32/64 neuron block。”

应该比较：

$$
J(B)
=
D^*(B)+\beta R^*(B).
$$

如果允许把 block $B$ 分成：

$$
B_1,B_2,
$$

那么 split 条件是：

$$
\boxed{
J(B_1)+J(B_2)
+
\beta R_{\rm split}
<
J(B).
}
$$

则 split。

否则不 split。

递归下去：

$$
256
\rightarrow
128+128
$$

$$
128
\rightarrow64+64
$$

等等。

于是 block size 本身也是 RD optimum。

这已经开始从 JPEG 进入 HEVC/AV1 的 variable partition 思想；AV1 明确把 coding-block partition 和 transform-block partition 当作两套工具。([Alliance for Open Media](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf?utm_source=chatgpt.com))

---

# 12. 而且 prediction block 与 transform block 不能是同一个概念

例如未来层间阶段：

一个 256-neuron 单元可能共用一个 reference：

$$
P_B(W_r).
$$

但它的 residual：

$$
E_B
$$

可能最适合：

$$
64+64+32+32+64
$$

这样的 transform partition。

所以至少需要区分：

$$
\boxed{\text{Prediction Unit}}
$$

$$
\boxed{\text{Transform Unit}}
$$

$$
\boxed{\text{Quantization Group}}.
$$

这就是上一轮我一句“block”根本说不清的地方。

---

# 13. KL-KLT 最大的实际问题：$H$ 根本存不下来

这才开始进入真正算法工程。

如果 block 只有：

$$
n=256,
$$

则：

$$
H\in\mathbb R^{256\times256}
$$

还可以。

但整模型 Fisher 不可能 materialize。

幸运的是我们从来不需要显式 $H$。

只需要：

$$
Hv.
$$

因为：

$$
Hv
=
\mathbb E_x[
J^TGJv
].
$$

计算路径：

$$
v
\xrightarrow{JVP}
Jv
\xrightarrow{G}
GJv
\xrightarrow{VJP}
J^TGJv.
$$

因此 generalized eigensolver 可以用：

$$
\text{Lanczos / LOBPCG}
$$

只调用 operator：

$$
v\mapsto Hv.
$$

不构造 Hessian。

---

# 14. 对线性层还能进一步得到 Kronecker 结构

假设：

$$
y=Wx.
$$

量化误差：

$$
\Delta W.
$$

则：

$$
\delta y=\Delta Wx.
$$

如果 downstream KL sensitivity 为：

$$
S_y,
$$

局部失真：

$$
D
\approx
\frac12
\mathbb E[
x^T\Delta W^T
S_y
\Delta Wx
].
$$

利用 vec：

$$
\boxed{
H_W
\approx
A\otimes S
}
$$

其中：

$$
A=\mathbb E[xx^T].
$$

这非常重要。

因为这意味着我们甚至可能把一个巨型：

$$
(d_{\rm out}d_{\rm in})^2
$$

Fisher 问题拆成：

$$
d_{\rm in}^2+d_{\rm out}^2.
$$

对 Qwen 这种模型，这是能不能实际做出来的区别。

---

# 15. SwiGLU 的“一个 neuron”也应该严格定义

对于 intermediate neuron $j$，不能把三个矩阵分别编码。

一个真正的 functional atom 应该是：

$$
\boxed{
\theta_j
=
\begin{bmatrix}
W_{\rm gate}[j,:]\\
W_{\rm up}[j,:]\\
W_{\rm down}[:,j]
\end{bmatrix}
}
$$

因为函数实际上是：

$$
W_{\rm down}[:,j]\,
\sigma(W_{\rm gate}[j,:]x)
\,
(W_{\rm up}[j,:]x).
$$

三者共同定义一个 neuron。

所以“JPEG pixel”对应的最小单位不是单 weight，而更接近：

$$
\boxed{\text{functional neuron atom}}.
$$

然后：

$$
8/16/32\text{ atoms}
$$

形成 transform block。

这个定义本身就比传统按连续 memory address 切 256 weights 更合理。

---

# 16. 但现在有一个危险：KL 二次展开只在小误差成立

尤其：

$$
2\sim3\text{ bit}
$$

绝对不一定属于 infinitesimal perturbation。

所以：

$$
\frac12\epsilon^TH\epsilon
$$

只能作为 **fast RDO surrogate**。

不能作为最终 distortion。

真正 encoder 应该两级：

### Stage A：廉价搜索

对几百万候选使用：

$$
\tilde D
=
\frac12\epsilon^TH\epsilon.
$$

大量剪枝。

### Stage B：full RDO

对少量候选真正 forward teacher/student：

$$
\boxed{
D_{\rm exact}
=
D_{\rm KL}
(p_\theta\|p_{\hat\theta})
}
$$

最终模式选择依据：

$$
\boxed{
J_{\rm exact}
=
D_{\rm exact}+\beta R.
}
$$

这特别像现代视频 encoder：

> 快速 proxy 决定搜索范围，但最后昂贵候选才做完整 RD 比较。

---

# 17. Calibration/OOS 也必须数学化，否则 KL 优化会变成蒸馏

我建议至少三分：

$$
\boxed{
S_{\rm fit},
S_{\rm RDO},
S_{\rm OOS}
}
$$

### Fit set

只能估计：

$$
\Sigma,H,
$$

transform、entropy statistics。

### RDO set

只能选择：

$$
\beta,\quad
block partition,\quad
quantizer,\quad
mode.
$$

### OOS

编码器在冻结 bitstream 以前**绝对不能看**。

最后只报告：

$$
D_{\rm KL}^{OOS}.
$$

你们前面发现的另一个问题也必须保留：后续压缩不能继续使用 teacher hidden state，否则产生 teacher-forcing illusion；实际下一层看到的是已经重建模型的 activation。`ChatGPT-逐层近似蒸馏效果-20260828-0912.md`

---

# 18. 还应该对 $\Sigma,H$ 做统计收缩

85k token 并不意味着 2048 维 Fisher 就能被准确估计。

定义：

$$
\hat\Sigma_{\rm shrunk}
=
(1-\alpha)\hat\Sigma
+
\alpha
\frac{\operatorname{tr}\hat\Sigma}{n}I.
$$

$H$ 同理：

$$
\hat H_{\rm shrunk}
=
(1-\rho)\hat H
+
\rho\tau I.
$$

然后 calibration 分成多个 bootstrap folds。

如果 generalized eigenvalue：

$$
\gamma_i
$$

稳定，但 eigenvector：

$$
u_i
$$

在不同 fold 剧烈旋转，则说明那一小片 eigenspace 近简并。

此时不要拟合单独 eigenvector。

应该把它作为一个 subspace：

$$
\mathcal U_k
$$

整体编码。

这是防 calibration overfit 很重要的一步。

---

# 19. 最终“JPEG-generation Neural Codec”已经可以写成一个完整优化问题

对于每一 source class $c$：

$$
\boxed{
\Sigma_c,\ H_c
}
$$

求：

$$
\Sigma_c^{1/2}H_c\Sigma_c^{1/2}
=
U_c\Gamma_cU_c^T.
$$

transform：

$$
y
=
U_c^T
\Sigma_c^{-1/2}
(w-\mu_c).
$$

然后 encoder 联合求：

$$
\boxed{
\min_{
\mathcal P,
Q,
\Delta,
Z
}
D_{\rm KL}^{\rm exact}
+
\beta
\left[
R_{\rm tree}
+
R_Q+
R_Z+
R_{\rm coeff}
\right]
}
$$

其中：

- $\mathcal P$：block partition；
- $Q$：IQ1/IQ2/IQ3/IQ4/SKIP 等 reproduction mode；
- $\Delta$：scale/quant step；
- $Z$：zero/nonzero coefficient structure；
- $R_{\rm coeff}$：真实 entropy-coded coefficient bits。

而 continuous relaxation 给：

$$
\boxed{
R_i^*
=
\frac12
\left[
\log_2\frac{\gamma_i}{\nu}
\right]_+
}
$$

作为搜索器的理论先验。

---

## 这才是我认为正确的研发方式

不是现在就说：

> JPEG → MPEG → HEVC → AV1。

而是：

### **Paper 1 / Codec V1：KL-optimal intra transform coding**

光这一代就应该完整解决：

$$
\boxed{
\begin{aligned}
&\text{source model}\\
&\text{KL geometry}\\
&\text{generalized KLT}\\
&\text{reverse water-filling}\\
&\text{practical IQ codebook projection}\\
&\text{block RDO}\\
&\text{entropy rate}\\
&\text{exact-KL correction}\\
&\text{OOS robustness}
\end{aligned}
}
$$

这本身已经完全够一篇很重的论文。

如果这一代在**完全不使用层间相似性**的情况下，就能让：

$$
R\leftrightarrow KL_{\rm OOS}
$$

曲线严格压过 IQ3_XXS，那么我们才有资格进入下一代：

$$
\boxed{\text{MPEG-generation：predictive layer coding}}
$$

而到了那一步，数学复杂度会再增加一层：不是简单

$$
W_l-W_r
$$

而是要建立 **depth-domain Gauss-Markov source、最佳 reference、prediction operator、innovation covariance、误差传播、B-frame reference graph 和 predictive rate-distortion function**。

那一章确实还能再写这么多，甚至更多。

所以你指出的问题本质上是对的：**我们现在不缺 AV1 功能列表，缺的是一本“Neural Codec Theory”。而第一章应该不是 AV1，而是把 JPEG/KLT/rate-distortion 在 KL 几何下重新推导一遍。**

---

**Sources:**

- [The JPEG still picture compression standard - Consumer Electronics, IEEE Transactions on](https://web.stanford.edu/class/ee398a/handouts/papers/Wallace%20-%20JPEG%20-%201992.pdf?utm_source=chatgpt.com)
- [1 Introduction](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf?utm_source=chatgpt.com)

## Prompt:
8/28/2026, 2:31:28 PM

如果我直接复用硬件AV1解码器，自己写编码器，把解码器出来的内容用于推理，你觉得有希望吗

## Response:
8/28/2026, 2:31:31 PM

有希望，但我会把“复用硬件 AV1 解码器”拆成两个完全不同的目标：**拿它当超高速权重解压器**，我很看好；**把 AV1 解码出来的 YUV 数据零转换直接当普通 LLM 权重喂给现有 GEMM**，我不太看好。真正有意思的是两者之间的第三条路：让模型权重本身变成一种“AV1-native 10-bit sample field”。

真正的关键不是“AV1 能不能压权重”，而是**把 AV1 夹在模型存储和 GEMM 之间，数据路径是否闭合**。硬件上这条路已经有基础：Intel 的 AV1 硬解常见输出是 NV12/P010，NVIDIA NVDEC 也暴露 NV12/P010 surface；AMD 的 AV1 decoder 能直接产出 DX/Vulkan surface，所以可以避免先回 CPU。

> **正在搜索网页**
>
> [Features and Formats](https://www.intel.com/content/www/us/en/docs/onevpl/developer-reference-media-intel-hardware/1-1/features-and-formats.html)
>
> **评估整体可行性**
>
> 我会把它看成“视频压缩承载权重、硬件解码后重排”的系统。概念上可行，但 AV1 输出的是像素面，不能直接恢复任意浮点张量；更适合低比特整数权重的加载或流式解码。
>
> **评估了整体可行性**
>
> 
>
> **正在搜索网页**
>
> [NVIDIA VIDEO CODEC SDK - DECODER](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/pdf/NVDEC_VideoDecoder_API_ProgGuide.pdf)
> [AMF/amf/doc/AMF_FRC_API.md at master · GPUOpen-LibrariesAndSDKs/AMF · GitHub](https://github.com/GPUOpen-LibrariesAndSDKs/AMF/blob/master/amf/doc/AMF_FRC_API.md)
> [9/7/2018  AV1 Codec ISO Media File Format Binding](https://downloads.aomedia.org/assets/pdf/AV1-ISO-Base-Media-File-Format-Binding-Specification.pdf)
>
> **已搜索 1 个网站**
>
> 
>
> **正在搜索网页**
>
> [PowerPoint Presentation](https://downloads.aomedia.org/assets/pdf/symposium-2019/slides/RyanLei-Av1QualityEvaluation.pdf)
> [CDEF is a nonlinear filtering process that prevents excessive blurring when a filtering process is applied across an edge. It is achieved by ignoring pixels that are too different from the current pixel that is to be filtered. When the difference between the current pixel and its neighbouring pixel *d* is within a given threshold, *f(d,S,D)* is set equal to *d*; otherwise, *f(d,S,D)* is set equal to 0. Specifically, the strength *S* determines the maximum allowable difference, and damping *D* determines the sample positions that ignore the filter tap.](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf)
> [AV2 Bitstream & Decoding Process Specification](https://av2.aomedia.org/v13-public/index.html)
>
> **分析编码路径**
>
> 我会将权重先量化并映射为合法像素，再交给自定义编码器生成标准AV1码流；硬件解码只能恢复像素面，不能直接输出变换系数或浮点张量，因此还需要GPU上的解包、反量化和布局重排。
>
> **分析了编码路径**
>
> 
>
> **已搜索 3 个网站**
>
> 
>
> **正在搜索网页**
>
> [Alliance for Open Media  DRAFT  v0.1](https://aomedia.org/docs/aomenc-command-line-parameters-draft.pdf)
> [PowerPoint Presentation](https://downloads.aomedia.org/assets/pdf/symposium-2019/slides/RyanLei-Av1QualityEvaluation.pdf)
> [AV2 Bitstream & Decoding Process Specification](https://av2.aomedia.org/v13-public/index.html)
>
> **设计端到端方案**
>
> 我会按层或专家分块，将相邻权重排列成伪帧，量化为8/10位样本，并保存缩放、置换和布局元数据。编码器可用下游误差优化模式选择；解码后再执行GPU解包、反量化和矩阵乘法。
>
> **验证压缩收益**
>
> 我会先测量实际比特率，而不是假设视频压缩有效。随机权重通常难以压缩；只有经过量化、置换、变换并形成局部相关性后，AV1才可能明显优于普通权重格式。
>
> **设计端到端方案并验证压缩收益**
>
> 
>
> **已搜索 3 个网站**
>
> 
>
> 思考了 1m 25s

有希望，而且是个很有意思的方向。但我会把它定义成：

$$
\boxed{\text{AV1 bitstream} \rightarrow \text{fixed-function HW decode} \rightarrow \text{AV1-native weight surface} \rightarrow \text{GEMM}}
$$

而不是简单的“把模型文件伪装成视频”。

最有价值的一点是：**你自己写 encoder，就可以完全不用 AV1 原来的 PSNR/SSE 目标。** AV1 bitstream 只规定 decoder 怎么重建；encoder 为什么选择这个 partition、reference、transform、quantizer，decoder 根本不关心。所以 encoder 完全可以做：

$$
\boxed{
\arg\min_{\text{legal AV1 decisions}}
D_{KL}(p_{\theta}\|p_{\hat\theta})
+\lambda R
}
$$

最后却交给全球几十亿块已经存在的 AV1 硬件 decoder 来执行反变换、预测、熵解码和重建。

**这一点是非常成立的。**

现在 Intel 的硬件 AV1 decode 常见输出就是 NV12/P010；NVIDIA NVDEC 也能输出 NV12/P010；AMD 的 AV1 decoder 可以直接生成 DX11/DX12/Vulkan surface。也就是说，至少在现代 GPU 上，解码结果可以留在 GPU-side surface，不需要先拷回 CPU。([intel.com](https://www.intel.com/content/www/us/en/docs/onevpl/developer-reference-media-intel-hardware/1-1/features-and-formats.html?utm_source=chatgpt.com))

但这里有一道非常重要的分叉。

---

## 1. 最弱版本：AV1 只作为模型文件压缩

例如：

$$
13\text{ GB AV1}
\rightarrow
35\text{ GB reconstructed weights}
\rightarrow
\text{推理}.
$$

这个技术上很好做，但意义有限。

它解决的是：

- 磁盘大小；
- 下载大小；
- 模型加载；
- CPU 解压开销。

却没有解决：

$$
\boxed{\text{VRAM 占用}}.
$$

模型一旦全部 decode 出来，压缩收益就消失了。

所以我不认为这应该是最终目标。

---

# 2. 真正有意思的是：**不要把 AV1 decode 回 FP16**

例如我们规定 decoder 输出 P010：

$$
s_i\in[0,1023].
$$

不把它转换回：

$$
w_i\in FP16
$$

存储。

而直接定义：

$$
\boxed{
w_i
=
a_g(s_i-z_g)
}
$$

其中：

- $s_i$：AV1 decoder 输出的 sample；
- $z_g$：block zero point；
- $a_g$：block scale。

也就是说：

$$
\boxed{\text{P010 surface 本身就是模型权重}}
$$

GEMM kernel 读取 P010：

$$
s_i
$$

现场做：

$$
a(s-z)
$$

然后乘 activation。

这样模型的“重建格式”就不再是 GGUF IQ3，也不是 FP16，而是：

$$
\boxed{\text{AV1-native W10}}
$$

甚至更值得首先实验的是 **NV12 的 8-bit luma plane**：

$$
s_i\in[0,255].
$$

因为这天然接近：

$$
INT8/W8A16
$$

计算。

---

# 3. 实际上可以把权重映射为一段“伪视频”

这个映射非常漂亮。

假设某层参数为：

$$
W_l.
$$

先通过我们自己的 functional transform：

$$
Z_l=T_l(W_l).
$$

然后做 companding：

$$
S_l
=
\operatorname{clip}
\left(
\operatorname{round}
\frac{Z_l}{\Delta_l}
+z_l
\right).
$$

于是：

$$
S_l\in\{0,\ldots,255\}^{H\times W}.
$$

把它当成 AV1 的 luma plane。

### 横纵空间

不应该按照原始内存排列。

应该把：

$$
\text{functional neighboring neurons}
$$

排到二维上。

例如：

$$
x=\text{within-neuron coefficient},
$$

$$
y=\text{functionally ordered neuron}.
$$

### 时间轴

就是 depth：

$$
S_0,S_4,S_8,\ldots
$$

成为一段视频。

于是 AV1 原生地做：

$$
S_8
\approx
P(S_4)+R.
$$

这正好就是我们想要的：

$$
\boxed{\text{layer prediction}}.
$$

---

# 4. Qwen3.6 特别适合这么做

甚至我不会编码成一个 40-frame video。

应该编码成四条 sequence：

$$
\begin{aligned}
V_0&=(L_0,L_4,L_8,\ldots,L_{36}),\\
V_1&=(L_1,L_5,L_9,\ldots,L_{37}),\\
V_2&=(L_2,L_6,L_{10},\ldots,L_{38}),\\
V_3&=(L_3,L_7,L_{11},\ldots,L_{39}).
\end{aligned}
$$

这样 AV1 temporal prediction 天然变成：

$$
\boxed{\text{同 phase 层间 prediction}}.
$$

然后 custom encoder 决定：

- keyframe；
- reference frame；
- compound prediction；
- block partition；
- transform；
- quantizer；
- skip；
- coefficient coding。

而选择标准不是视觉质量：

$$
SSE_{\rm pixel}
$$

而是：

$$
\boxed{KL_{\rm model}}.
$$

这其实非常接近你最开始想要的东西。

---

# 5. 一个意外的优势：AV1 decoder 根本不知道自己在解模型

这意味着我们不需要改任何 ASIC。

encoder 可以认为：

> 这个 64×64 block 如果使用 frame 12 做 reference，经过某 transform、Q=137 后，增加 832 bits，但 OOS KL 只增加 $2.1\times10^{-5}$。

于是计算：

$$
J=
D_{KL}+\lambda R.
$$

最终吐出**完全合法的 AV1 bitstream**。

硬件 decoder 看到的只是：

> 哦，某个 inter block，某个 reference，某个 transform，某些 coefficient。

然后照标准做。

所以：

$$
\boxed{\text{Neural intelligence 全部留在 encoder}}
$$

$$
\boxed{\text{decoder 100\% commodity}}
$$

这是这条路线最吸引我的地方。

---

# 6. 但有几个非常硬的限制

其中最大的一个是：

## AV1 decoder 输出的是像素，不是 transform coefficients

我们不能要求硬件说：

> 给我解码到 residual coefficient 这里就停。

一般 API 给你的最终结果是 reconstructed surface。

例如 Intel 的常见 AV1 硬解格式明确是 8-bit NV12 和 10-bit P010；NVIDIA 同样把 NV12/P010 作为 decoder surface format。([Intel](https://www.intel.com/content/www/us/en/docs/onevpl/developer-reference-media-intel-hardware/1-1/features-and-formats.html?utm_source=chatgpt.com))

所以所有 AV1 内部：

$$
\text{entropy decode}
\rightarrow
\text{inverse quant}
\rightarrow
\text{inverse transform}
\rightarrow
\text{prediction}
\rightarrow
\text{reconstruction}
$$

已经发生。

这意味着我们的模型必须接受：

$$
\boxed{\text{最终 8/10-bit reconstructed samples}}
$$

作为权重表示。

---

# 7. 因此必须避免一些“视频特有的好心办坏事”

例如 AV1 有：

- deblocking；
- CDEF；
- loop restoration；
- film grain 等。

对视频来说这些可能改善观感。

对一个 weight tensor：

> “我替你平滑一下边缘”

可能直接毁掉模型。

幸运的是这些工具大量是可以通过 bitstream/encoder configuration 控制的。AOM 的 encoder 参数就明确提供关闭 CDEF 和 loop restoration 的控制；AV1 的这些环路滤波本来就是独立工具。([Alliance for Open Media](https://aomedia.org/docs/aomenc-command-line-parameters-draft.pdf?utm_source=chatgpt.com))

第一版我会：

$$
\boxed{
\text{Deblock}=0,\quad
\text{CDEF}=0,\quad
\text{Restoration}=0,\quad
\text{Film Grain}=0
}
$$

把 AV1 当成：

$$
\text{predictor + transform + quantizer + entropy coder}.
$$

以后再试一个很有趣的问题：

> 某些 filter 是否反而能成为 neural in-loop filter？

但第一代绝对不要。

---

# 8. YUV 是一个烦人的地方，但不是致命问题

跨平台硬件最保险的是：

$$
4:2:0.
$$

例如 Main-profile 常见输出：

$$
Y+UV.
$$

这对权重不自然。

最简单的设计：

### Y plane

真正存权重：

$$
\boxed{Y=\text{weights}}.
$$

### U/V

一开始设成常量。

它们在 bitstream 中因为高度平坦，压缩成本应该很低。

虽然 decoder surface 中还是占空间，但压缩码流成本不高。

以后甚至可以利用 U/V 存：

- per-block scale；
- zero point；
- transform class；
- importance side-channel。

因为它们恰好是低空间分辨率：

$$
\boxed{\text{一个 chroma sample 对多个 Y weights}}
$$

这反而很像 block metadata。

我觉得这个映射非常有意思。

---

# 9. 真正最大的现实问题反而是**解码带宽**

这里必须很清醒。

GPU 的视频 decoder 是为：

$$
4K/8K\times几十/几百 FPS
$$

设计的。

而 GPU HBM/GDDR GEMM 读权重的带宽是：

$$
\text{数百 GB/s}\sim\text{TB/s}.
$$

二者不是一个量级。

所以如果你的想法是：

> 每生成一个 token，都把 Qwen 需要的所有权重从 AV1 bitstream 重新硬解一次

我基本不看好。

因为即使只有 active 3B 参数：

$$
3\times10^9
$$

个 sample。

8-bit 重建就是约：

$$
3\text{ GB/token}.
$$

20 token/s 就要求：

$$
60\text{ GB/s}
$$

**持续 AV1 reconstructed payload throughput**。

普通视频 decode block 的设计目标不是这个量级。

---

# 10. 所以“AV1 streaming inference”必须有 cache

正确架构更像：

$$
\boxed{
\text{compressed AV1 VRAM}
\rightarrow
\text{video decoder}
\rightarrow
\text{decoded weight cache}
\rightarrow
\text{GEMM}
}
$$

即类似 CPU 的：

$$
RAM\rightarrow L3\rightarrow L2.
$$

AV1 bitstream 是 backing store。

decoded weight surface 是 cache。

比如保存：

$$
4\sim8
$$

个 Bank 的 decoded W8 数据。

对于顺序层：

$$
L_0\rightarrow L_1\rightarrow\cdots
$$

decoder 可以提前：

$$
\text{prefetch }L_{l+1}.
$$

这样 decode 和 compute overlap。

---

# 11. 不过 MoE 给这个方案同时带来了优势和问题

优势：

Qwen3.6 每 token 只激活少量 experts。

理论上：

$$
256
$$

个 expert 不需要全部重建。

这听起来非常适合：

$$
\boxed{\text{compressed expert store + decoded expert cache}}.
$$

但是问题是：

普通 AV1 decoder 最舒服的单位是：

$$
\boxed{\text{完整 frame}}
$$

不是：

> frame 里面随机挑 expert 17、83、201，只解这三块。

AV1 确实有 tiles，但硬件 API 的随机 partial-decode 能力不能假设跨厂商一致。

因此如果真要发挥 MoE 稀疏性，我反而会设计：

$$
\boxed{\text{expert group = independently decodable frame/tile group}}
$$

而不是：

$$
\text{整个 MoE bank = 1 frame}.
$$

这会是 codec/container 层面的重要设计。

---

# 12. 我认为最值得做的第一个实验非常简单

**甚至先不要写 KL encoder。**

拿 Qwen 的一个 MoE tensor。

### A. 做二维 layout

例如：

$$
512\times2048
$$

或者重新排列后的 functional layout。

### B. 映射成 10-bit

$$
w
\rightarrow
s=
\operatorname{round}(w/\Delta)+512.
$$

### C. 生成合法 10-bit AV1

关闭所有视觉 enhancement filter。

先只用：

$$
\text{INTRA}.
$$

### D. 硬件 decode 到 P010

保持在 GPU。

### E. 写一个极小 kernel

直接读取 P010：

$$
s
$$

做：

$$
w=\Delta(s-512)
$$

然后 GEMM。

**不生成 FP16 weight tensor。**

如果这个闭环能跑：

$$
\boxed{
\text{AV1}
\rightarrow
\text{P010}
\rightarrow
\text{GEMM}
}
$$

这条路线就已经被证明在系统层面成立。

---

下一步再变成：

$$
L_0 = I
$$

$$
L_4 = P(L_0)
$$

$$
L_8 = P/B(L_0,L_{12})
$$

真正开始吃 AV1 inter prediction。

最后才把 encoder 的：

$$
\text{SSE}
$$

完全替换成：

$$
\boxed{
D_{KL}^{OOS}+\lambda R.
}
$$

---

## 我的判断

我会分别给三个目标这样的评价：

| 目标 | 判断 |
|---|---|
| AV1 压模型文件，启动时全解压 | **非常可行，但创新价值一般** |
| AV1 解成 W8/W10 GPU surface，直接用于定制 GEMM | **非常值得研究** |
| 每 token 实时 AV1 解整个模型/全部 active weights | **带宽上大概率不成立** |
| AV1 bitstream + decoded expert/layer cache + prefetch | **我很看好** |
| 自写 KL-RDO encoder + 原封不动商业 AV1 decoder | **概念上最漂亮，也最值得验证** |

所以我会把目标进一步改成：

$$
\boxed{\textbf{AV1-Native Neural Weight Format}}
$$

不是“AV1 帮我们解压 GGUF”。

而是让：

$$
\boxed{\text{AV1 reconstructed samples 本身就是推理权重格式}}
$$

这样你真正获得的是一个非常奇怪但可能非常强的系统：**几十亿台设备已经免费带了一块专门为你的模型权重做 prediction + inverse transform + entropy decode 的 ASIC，而我们只需要写 encoder 和一个很薄的 AV1-surface→GEMM 层。**

这条我认为值得实际做 POC。

## Prompt:
8/28/2026, 2:43:09 PM

如果用hevc会不会带宽大，我用的是6800，实时解压有希望吗

## Response:
8/28/2026, 2:43:09 PM

如果你说的是 **Radeon RX 6800（RDNA2 / Navi 21）**，HEVC 比 AV1 更现实，因为这张卡有成熟的硬件 HEVC 解码，而没有 AV1 硬解。关键不是显存带宽本身，而是 **VCN 解码单元能持续吐出多少“重建权重字节/秒”**，以及能否和 GEMM 并行。

查到一个对你这张卡非常重要的事实：AMD 当前资料把 RX 6800 列为 **2 个 VCN**，HEVC 8/10-bit 都能硬解；而 AV1 的脚注明确是**只在其中一个 VCN 上支持**。所以如果你的目标是“把视频解码器当权重解压 ASIC”，**HEVC 在 RX 6800 上反而可能比 AV1 更适合做第一版系统实验**。

> **正在搜索网页**
>
> [Radeon™ RX 6800 Graphics Card](https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6800.html)
> [Features - Features - 3.0 English - PG447](https://docs.amd.com/r/en-US/2026.1/pg447-vcu2-solutions/Features?contentId=wxLZNY4z1kyeMdkM~oKpZQ)
>
> **评估缓存与解码**
>
> 我在估算显存占用、专家权重缓存和按页换入的带宽需求。关键约束是路由局部性与缓存未命中率；同时，我在核对硬件是否支持单色 HEVC，否则可能需要用亮度平面承载权重。
>
> **评估了缓存与解码性能**
>
> 
>
> **已搜索 2 个网站**
>
> 
>
> **计算4K/8K视频带宽需求**
>
>
> **计算了4K/8K视频带宽需求**
>
> 
>
> **正在搜索网页**
>
> [Decoder Software Features - Decoder Software Features - 3.0 English - PG447](https://docs.amd.com/r/en-US/pg447-vcu2-solutions/Decoder-Software-Features?contentId=DHSAQE1WdFbbJFPMFRhp0Q)
> [Radeon™ RX 6800 Graphics Card](https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6800.html)
>
> **核对解码吞吐**
>
> 我需要区分压缩输入、解码写出和显存搬运；4K60 约为每秒数百 MB，通常低于显存带宽。还要确认 RX 6800 的 HEVC 分辨率、并发实例与延迟限制，并修正此前关于 AV1 支持的表述。
>
> **估算带宽瓶颈**
>
> 按每 token 激活约 3B 参数估算，解码写出约 3GB；若压缩读取为 0.75–1.5GB，10 token/s 就需要约 37.5–45GB/s。显存带宽或许足够，但固定功能视频解码器的吞吐与逐帧延迟更可能成为瓶颈。
>
> **核对了解码吞吐并估算带宽瓶颈**
>
> 
>
> **正在搜索网页**
>
> [Radeon™ RX 6800 Graphics Card](https://www.amd.com/zh-cn/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6800.html)
> [Adobe Premiere Pro - AMD Radeon RX 6800 (XT) Performance | Puget Systems](https://www.pugetsystems.com/labs/articles/adobe-premiere-pro-amd-radeon-rx-6800-xt-performance-1989/)
> [FAQ](https://nation.sapphiretech.com/zh-cn/faq)
> [SVP Transcoding and RX6000 series (Page 1) — Using SVP — SmoothVideo Project](https://www.svp-team.com/forum/viewtopic.php?id=6255)
> [AMD Radeon RX 6800: performance tests and specs | NR](https://nanoreview.net/en/gpu/radeon-rx-6800)
>
> **判断实际可行性**
>
> 我会把 HEVC 当作冷数据压缩层，而不是每个 token 都完整解码。RX 6800 的显存带宽足够应付几十 GB/s，但硬件解码吞吐、启动延迟和随机分页会限制性能；应优先缓存热点专家，并按较大连续块预取。
>
> **判断了实际可行性**
>
> 
>
> **已搜索 2 个网站**
>
> 
>
> 思考了 1m 24s

如果是 **RX 6800**，我现在的判断是：

> **HEVC 做“每 token 全量实时解压”基本不够；但做“专家/层级按需解压 + GPU 内 decoded cache”是有现实希望的，而且 RX 6800 很适合做原型。**

先更正我上一条的一点：AMD 官方现在明确列 RX 6800 **支持 AV1 硬解**；同时列出 RX 6800 有 **2 个 VCN**，而 AV1 只有其中一个 VCN 支持。HEVC 8/10-bit 则属于成熟的硬件解码路径，所以在这张卡上用 HEVC 做第一版实验非常合理。([AMD](https://www.amd.com/en/products/graphics/radeon-for-creators/video-editing.html?utm_source=chatgpt.com))

### 先看真正的瓶颈

RX 6800 的显存带宽是：

$$
512\ {\rm GB/s}
$$

所以 **GDDR6 本身不是主要问题**。([AMD](https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6800.html?utm_source=chatgpt.com))

假设硬解一个普通 4K60 HEVC surface，3840×2160×60：

$$
497.7M\ {\rm pixels/s}.
$$

8-bit 4:2:0 的最终重建数据量大约：

$$
0.75\ {\rm GB/s}.
$$

P010 10-bit 因为通常装在 16-bit container 中，则约：

$$
1.49\ {\rm GB/s}.
$$

即使解码器能跑到高于 4K60、甚至两个 VCN 并行，量级仍然更像：

$$
\boxed{\text{几 GB/s}}
$$

而不是 RX 6800 显存的：

$$
512\ {\rm GB/s}.
$$

AMD 没有公布一个我敢直接拿来当“RX 6800 arbitrary-data HEVC 解压 GB/s”的官方数字，因此这里不能把双 VCN简单理解成精确的 2×。但是数量级判断很清楚：**VCN throughput 才是需要实测的瓶颈。**

---

## 为什么“每 token 重解整个 A3B”还是不行

Qwen3.6-35B-A3B 大约：

$$
3B\ {\rm active\ parameters/token}.
$$

如果重建格式是 W8：

$$
3B\times1{\rm byte}
\approx3{\rm GB/token}.
$$

10 token/s 就要求：

$$
30{\rm GB/s}.
$$

20 token/s：

$$
60{\rm GB/s}.
$$

如果用 P010 这种 16-bit 容器作为 W10 surface，物理输出甚至接近：

$$
6{\rm GB/token}.
$$

所以：

$$
\boxed{\text{VCN每token重建全部active weights}}
$$

基本没戏。

不是显存 512 GB/s 不够，而是 **HEVC ASIC 每秒解出来的像素/样本数量不够**。

---

# 但 MoE 改变了结论

因为真正需要动态换入换出的，不应该是全部 3B active parameters。

Attention、DeltaNet、Norm 等可以：

$$
\boxed{\text{永久驻留 VRAM}}
$$

只把：

$$
\boxed{\text{256 routed experts}}
$$

做成 HEVC compressed backing store。

Qwen 一个 2048→512→2048 的 SwiGLU expert 大约：

$$
3\times2048\times512
\approx3.15M
$$

参数。

W8 重建以后：

$$
\boxed{\sim3.15\ {\rm MB/expert}}.
$$

一层每 token 选 8 个：

$$
8\times3.15
\approx25.2{\rm MB}.
$$

40 层：

$$
\approx1.0{\rm GB/token}.
$$

这就比 3 GB/token 好很多了。

而更关键的是：

**你根本不应该每 token 都重新解这 1 GB。**

---

# 正确结构应该是 Expert L2 Cache

我会做：

$$
\boxed{
\text{HEVC compressed expert store}
\rightarrow
\text{VCN decode}
\rightarrow
\text{W8 decoded expert cache}
\rightarrow
\text{Vulkan GEMM}
}
$$

例如 RX 6800 的 16GB：

$$
\begin{array}{rl}
2\sim4{\rm GB}&:\text{永久驻留 mixer/attention/其他权重}\\
6\sim10{\rm GB}&:\text{decoded expert cache}\\
其余&:\text{KV/state/workspace}
\end{array}
$$

如果给 expert cache 6GB：

$$
\frac{6{\rm GB}}{3.15{\rm MB}}
\approx1900
$$

个 decoded experts。

整个模型 routed experts 是约：

$$
40\times256=10240.
$$

也就是说缓存大概能装：

$$
19\%
$$

的专家。

但重要的是访问分布大概率**远远不是均匀的**。

如果实际 router 存在明显热点：

$$
P(e)
$$

是 heavy-tail，那么 1900 个 expert 可能覆盖远超过 19% 的实际访问。

这件事情你现在的 8.5 万 token 激活数据应该直接就能测。

---

# 可以把可行条件写成一个特别简单的公式

设：

- $B_e$：每 token routed expert 的 decoded bytes；
- $f$：tokens/s；
- $m$：decoded cache miss rate；
- $V$：HEVC 实测有效 reconstruction throughput。

要求：

$$
\boxed{
mB_ef<V.
}
$$

Qwen routed experts 粗略：

$$
B_e\approx1{\rm GB/token}.
$$

假设目标：

$$
f=10{\rm tok/s}.
$$

那么没有 cache：

$$
10{\rm GB/s}.
$$

假如你实测 RX6800 双 VCN 能为这种数据提供：

$$
V=2{\rm GB/s},
$$

那么要求：

$$
m<20\%.
$$

即：

$$
\boxed{\text{cache hit}>80\%}.
$$

如果只有：

$$
V=1{\rm GB/s},
$$

要求：

$$
\boxed{\text{hit}>90\%}.
$$

如果双 VCN 实际能做到：

$$
4{\rm GB/s},
$$

那只需要：

$$
\boxed{\text{hit}>60\%}.
$$

所以现在问题突然变成了一个**非常具体、可以今天就测的系统问题**：

> Qwen3.6 在真实 generation 中，给 2GB / 4GB / 6GB decoded-expert cache，LRU/LFU 下的 expert cache hit rate 是多少？

这个数字可能直接决定整个 HEVC 方案生死。

---

## 而且可以做得比普通 LRU 好很多

因为 Router 在 GEMM **之前**就已经告诉你下一步需要哪些 experts。

也就是说你拥有：

$$
\boxed{\text{oracle-like prefetch information}}
$$

某层：

$$
R_l(h)\rightarrow\{e_1,\ldots,e_8\}.
$$

在当前 mixer/其他 expert 计算期间，VCN 就可以开始：

$$
\operatorname{decode}(e_1,e_2,\ldots).
$$

于是形成 pipeline：

$$
\begin{array}{ccc}
\text{VCN} &:& \text{decode layer }l+1\\
\text{GPU shader} &:& \text{compute layer }l
\end{array}
$$

如果：

$$
T_{\rm decode}
<
T_{\rm compute}+T_{\rm lookahead},
$$

很多 miss latency 可以完全藏起来。

这比普通视频 playback 对我们更有利。

---

# 甚至两个 VCN 可以做“双通道专家服务器”

RX 6800 官方资料列的是 **2 VCN**；AV1 特别注明只有一个 VCN 支持，而 HEVC 没有这个 AV1 限制说明。([AMD](https://www.amd.com/en/products/graphics/radeon-for-creators/video-editing.html?utm_source=chatgpt.com))

所以我会专门实验：

$$
\boxed{
VCN_0:\text{ stream A}
\qquad
VCN_1:\text{ stream B}
}
$$

不要做一条巨大 HEVC stream。

把 experts 拆成多个独立 bitstreams/GOP：

$$
S_0,S_1,\ldots,S_K.
$$

然后把请求：

$$
e_{37},e_{201},e_{98},\ldots
$$

分配到两个 decoder queue。

这比希望一个单流 decoder 自己并行得更好。

但“两个 HEVC 流能够线性 2×”这一点必须在你的具体 Windows/Linux + Vulkan/VAAPI/D3D11 环境上实测，我不会提前假设成立。

---

# 我反而建议先用 HEVC 8-bit，而不是 Main10

第一版目标应该是：

$$
\boxed{\text{NV12 Y plane}\rightarrow W8 GEMM}
$$

理由很简单。

P010 虽然有 10-bit：

$$
0\ldots1023
$$

但 surface 一般是 16-bit container。

于是 decode output bandwidth 和 cache 占用都会接近翻倍。

而 W8：

$$
w=s(q-z)
$$

已经很成熟。

第一代应该证明：

$$
\boxed{\text{VCN}\rightarrow\text{GPU surface}\rightarrow\text{GEMM}}
$$

能跑。

以后 KL-RDO 告诉我们哪些区域需要 >8 bit，再考虑 Main10。

---

## HEVC 相比 AV1，我现在反而更推荐你先做

不是因为 HEVC 压缩率一定更好--通常不是。

而是对你的 **RX 6800 原型机**：

$$
\boxed{
\text{HEVC}:
\text{简单一些 + 8/10-bit成熟 + 双VCN潜力}
}
$$

而 AV1：

$$
\boxed{
\text{更强codec工具集，但RX6800只有一个VCN支持AV1}
}
$$

所以研发顺序完全可以是：

$$
\boxed{
\text{HEVC proof-of-system}
\rightarrow
\text{HEVC KL encoder}
\rightarrow
\text{AV1 KL encoder}
}
$$

底层 cache/prefetch/GEMM surface 接口全部复用。

---

### 我现在最想测的甚至不是压缩率，而是这 4 个数

直接写一个 HEVC synthetic benchmark，在 RX 6800 上测：

1. 单 VCN 持续 decode 的 **Y-plane GB/s**；
2. 双独立 HEVC stream 的总 **GB/s**；
3. decode surface → Vulkan compute shader **零拷贝是否成立以及同步成本**；
4. 一边 VCN decode、一边 Vulkan GEMM 时，两者是否真正 overlap。

只要第 2 项能有**数 GB/s**，第 3、4 项成立，再结合 expert cache 命中率，我认为：

$$
\boxed{\text{HEVC-backed realtime MoE inference}}
$$

就不再只是概念，而是真有工程实现希望。

而“每 token 把整个模型重新 HEVC 解一次”，我会直接放弃--**cache + routing-aware prefetch 才是 RX6800 上真正可能成立的版本。**

---

**Sources:**

- [Video Production Advanced by AMD Radeon™ Graphics](https://www.amd.com/en/products/graphics/radeon-for-creators/video-editing.html?utm_source=chatgpt.com)

## Prompt:
8/28/2026, 2:46:19 PM

哦，解压不动。我想省显存，这条路不通

## Response:
8/28/2026, 2:46:23 PM

对，如果目标是**真正省显存**，那“压缩存着、推理时实时解回 W8/W10”这条路基本把收益吃掉了：

$$
\text{省下的 VRAM}
\longleftrightarrow
\text{被解码带宽重新买回来}.
$$

所以 AV1 **硬件 decoder** 这条支线可以砍掉，但 **AV1 的编码思想完全不用砍**。甚至现在方向反而更清楚了：

> **最终格式必须能够直接参与矩阵乘，而不是先恢复成完整权重。**

这意味着真正值得做的只有三类。

### 1. 最核心：Compressed-domain GEMM

类似 IQ3_XXS 为什么能省显存：

$$
\boxed{
\text{3-bit weights remain 3-bit in VRAM}
}
$$

计算时 kernel 小块读取：

$$
q_B
$$

在寄存器/LDS 中临时反量化：

$$
\hat W_B=D(q_B)
$$

然后马上：

$$
y+=\hat W_Bx.
$$

从来没有：

$$
\boxed{\text{完整 }\hat W\text{ 存进 VRAM}}
$$

所以我们的 AV1-like codec 最终也必须满足：

$$
\boxed{
\text{bitstream/block representation}
\rightarrow
\text{register decode}
\rightarrow
\text{FMA}
}
$$

而不是：

$$
\text{bitstream}
\rightarrow
\text{full reconstructed tensor}
\rightarrow
\text{GEMM}.
$$

这就是硬约束。

---

### 2. AV1 的“层间压缩”可以保留，但 predictor 必须直接进入 GEMM

比如：

$$
W_l
=
W_r+\Delta_l.
$$

绝不能实际构造：

$$
\hat W_l=W_r+\hat\Delta_l
$$

再存下来。

应该直接计算：

$$
W_lx
=
W_rx+\Delta_lx.
$$

这一下非常关键。

如果 reference Bank 已经驻留：

$$
W_r
$$

而 residual 只有 1–2 bit：

$$
\Delta_l,
$$

那么推理直接做：

$$
\boxed{
y_l=
GEMM(W_r,x)
+
GEMM_{\rm compressed}(\Delta_l,x)
}
$$

完全不需要恢复 $W_l$。

甚至如果多层共享：

$$
W_l=W_0+\Delta_l,
$$

那么物理显存就是：

$$
\boxed{
W_0+\Delta_1+\Delta_2+\cdots
}
$$

这才是真正的“P-frame 神经网络”。

你们前面做的 recurrent MoE/shared Bank，本质上已经很接近这个结构。

---

### 3. 更进一步：transform coefficient 也直接计算

假设我们以后得到：

$$
W=T^{-1}q.
$$

普通 codec 会先：

$$
q\rightarrow W
$$

再：

$$
Wx.
$$

但如果 $T$ 选得聪明：

$$
Wx
=
T^{-1}q\,x
$$

可以改写计算顺序，或者把 transform 融入左右输入。

例如若：

$$
W=P\tilde WQ
$$

其中 $P,Q$ 是 permutation / Hadamard / block orthogonal transform，

那么：

$$
Wx
=
P\tilde W(Qx).
$$

于是：

1. 先便宜地算 $Qx$；
2. 用压缩的 $\tilde W$ 做 GEMM；
3. 再做 $P$。

这意味着 transform coding **不一定需要 inverse-transform 整个权重矩阵**。

这是非常重要的一条路：

$$
\boxed{\text{codec transform 与 GEMM algebra 联合设计}}
$$

而不是照搬 JPEG decoder。

---

## 所以我们现在可以正式加入一个约束

之前目标：

$$
\min D_{KL}+\lambda R.
$$

现在必须升级为：

$$
\boxed{
\min_{\mathcal C}
D_{KL}
+
\lambda_M M(\mathcal C)
+
\lambda_B B_{\rm inference}(\mathcal C)
+
\lambda_F FLOPs(\mathcal C)
}
$$

其中：

- $M$：常驻显存；
- $B_{\rm inference}$：推理每 token 必须从 VRAM 读取的字节；
- FLOPs：额外解码/补偿计算。

因为对 LLM 来说，**rate 不只是磁盘 bitstream 长度。**

真正的 rate 应该是：

$$
\boxed{\text{inference-time memory traffic}}
$$

甚至这可能比“checkpoint 文件大小”更第一性。

---

## 这也解释了为什么 IQ3_XXS 是个好起点

IQ3 的最大优点其实不只是 3 bit。

而是：

$$
\boxed{
\text{它的 compressed representation 本身就是 runtime representation}.
}
$$

这正是我们必须保留的性质。

所以新的路线应该是：

$$
\text{IQ3}
$$

↓

$$
\text{KL-optimal IQ3}
$$

↓

$$
\text{mixed-precision block IQ}
$$

↓

$$
\text{transform-domain IQ}
$$

↓

$$
\boxed{\text{predictive / inter-layer IQ}}
$$

而不是最后变成传统 AV1 bitstream。

AV1 应该贡献的是：

> prediction、reference、partition、transform、skip、compound、RDO。

**不应该贡献它的像素 decoder。**

---

而且这样一来，你最初那个“40→12 Bank”研究突然重新变得非常核心。

例如如果：

$$
W_l=W_{g(l)}+\Delta_l
$$

其中 12 个：

$$
W_{g(l)}
$$

用 IQ3/4 保存，

40 个 residual：

$$
\Delta_l
$$

平均只需要：

$$
0.5\sim1.5\text{ bpw},
$$

那就可能得到：

$$
\boxed{
12\times\text{full-ish banks}
+
40\times\text{tiny residual streams}
}
$$

同时 GEMM：

$$
W_lx
=
W_{g(l)}x
+
\Delta_lx.
$$

这才是真正把 **AV1 的帧间预测** 和 **IQ3 的 compressed compute** 合起来了。

所以刚才 HEVC 那条路虽然死了，但其实帮我们确认了一个特别重要的设计原则：

$$
\boxed{\textbf{Never fully decompress weights.}}
$$

我会把它列成 Neural Codec 的第一条硬性规范。

## Prompt:
8/28/2026, 2:47:59 PM

甚至能不能通过巧妙的设计，不但不是overhead，而是反倒加速gemm，时间空间一起省！─=≡Σ(((つ•̀ω•́)つ

## Response:
8/28/2026, 2:47:59 PM

可以，而且我认为这反而应该成为**最高级目标**：不是“压缩后尽量少拖慢”，而是让压缩表示本身更适合计算，使

$$
\boxed{\text{更少显存}+\text{更少内存流量}+\text{更少乘加}}
$$

同时成立。关键在于不能把 codec 和 GEMM 分开设计，而要做 **compute-native codec**。

而且这里有一个很反直觉、但非常重要的方向：**3 bit 未必比 4 bit 快。** 对 RX 6800 的 RDNA2，硬件 ISA 原生有 `V_DOT8_I32_I4`——一次直接做 8 个 INT4 点积；如果我们设计一个“数学压缩率更高、但运行时落到规则 INT4/INT8 dot 指令”的 codec，完全可能比复杂 IQ3 解码**既更小又更快**。

> **正在搜索网页**
>
> [vulkan: K-quant MMVQ path 10-15x slower than f32 dequant on AMD RDNA3 · Issue #21151 · ggml-org/llama.cpp · GitHub](https://github.com/ggml-org/llama.cpp/issues/21151)
> [Dot Product Unit - Dot Product Unit - AM004](https://docs.amd.com/r/en-US/am004-versal-dsp-engine/Dot-Product-Unit)
>
> **规划本地推理测试**
>
> 我会把重点放在生成阶段的矩阵向量乘：比较权重读取量、解码开销、寄存器占用，以及 RDNA2 上 INT4 点积是否能抵消更复杂格式的收益。
>
> **评估权重复用**
>
> 我判断原生 4 位格式可能凭借直接点积指令和简单解码胜过 3 位格式。跨层共享权重主要节省容量；只有当活跃权重能留在缓存中时，才可能明显减少显存读取。
>
> **规划本地推理测试并评估权重复用**
>
> 
>
> **已搜索 2 个网站**
>
> 
>
> **正在搜索网页**
>
> [Radeon™ RX 6800 Graphics Card](https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6800.html)
>
> **设计可行方案**
>
> 我会优先测试四位原生点积、分块缩放和稀疏异常值分离；再评估低秩残差。跨层共享只有在能复用缓存权重，或通过因子分解避免额外矩阵乘时，才可能真正提升每秒令牌数。
>
> **构造共享变换**
>
> 我会尝试让每层表示为“廉价变换＋共享四位核心矩阵”，把置换、符号、缩放或小型蝶形变换推到激活端，避免为残差再执行一次完整矩阵乘。
>
> **设计可行方案并构造共享变换**
>
> 
>
> **已搜索 1 个网站**
>
> 
>
> 思考了 1m 6s

可以，而且我现在反而觉得：**如果做不到“压缩同时加速”，这个 Neural Codec 还没有走到真正成熟。** 对自回归 LLM，尤其 batch 小的时候，本来就高度受权重读取影响；压缩减少显存流量，只要“解码”设计得足够贴着硬件，就可能比传统 GEMM 更快。

对 RX 6800 还有个特别有利的事实：RDNA2 原生提供 `V_DOT4_I32_I8` 和 `V_DOT8_I32_I4`，也就是一条指令做 4 个 INT8 或 8 个 INT4 点积。([amd.com](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf?utm_source=chatgpt.com)) 所以我甚至会提出一个看似反常的原则：

$$
\boxed{\text{不要追求最低 bpw，追求最低 bytes}\times\text{decode cost}}
$$

## 1. 可能真正比 IQ3 快的是“有效 3 bit，计算 INT4”

IQ3 的问题之一恰恰是它为了 3-bit 左右的密度，需要 lookup、sign extraction、unpacking 等复杂操作。近期 llama.cpp 的 AMD profiling 就观察到 IQ3_XS shader 使用更多 VGPR、occupancy 降低；另一些 Vulkan quant 路径甚至会因为复杂 unpack/repack 严重拖慢。([GitHub](https://github.com/ggml-org/llama.cpp/issues/20848?utm_source=chatgpt.com))

所以我们不要设计：

$$
\text{3-bit bitstream}\rightarrow
\text{复杂解码}\rightarrow FP16\rightarrow FMA.
$$

而设计：

$$
\boxed{
\text{总模型有效 }2.5\!-\!3.2\ {\rm bpw}
}
$$

但真正进入 ALU 的核心矩阵永远是规则：

$$
\boxed{\mathrm{INT4}}
$$

直接：

$$
V\_DOT8\_I32\_I4.
$$

怎么做到“4-bit compute、<4-bit storage”？

靠的不是把一个 INT4 再压成 3 bit，而是 AV1 式的：

$$
\text{prediction}
+\text{SKIP}
+\text{shared prototype}
+\text{structured zeros}.
$$

例如：

$$
W_l\simeq T_l(Q_{g(l)})
$$

其中 $Q_g$ 是共享 INT4 核心，$T_l$ 只有极少 metadata。

所以全模型平均可能只有 2.8 bpw，但每次 GEMM 仍然吃**硬件最喜欢的 INT4**。

这可能比 IQ3漂亮得多。

---

# 2. 更强：把 inverse transform 推到 activation 上

这是我非常看好的设计。

假设编码器找到：

$$
W_l
\approx
D_l^{out}P_l^{out}
\,Q_g\,
P_l^{in}D_l^{in}
$$

其中：

- $Q_g$：共享 INT4 matrix；
- $D$：对角 scale；
- $P$：permutation/sign flip，甚至简单 butterfly/Hadamard。

普通 codec 会先重建：

$$
W_l.
$$

我们绝对不这样做。

因为：

$$
W_lx
=
D_l^{out}P_l^{out}
Q_g
\left(
P_l^{in}D_l^{in}x
\right).
$$

所以实际执行：

$$
x'
=
P_l^{in}D_l^{in}x
$$

然后一次：

$$
z=Q_gx'
$$

最后：

$$
y=
D_l^{out}P_l^{out}z.
$$

于是：

$$
\boxed{\text{完全没有 weight decompression}}
$$

而且主计算仍然只有**一次 INT4 GEMM**。

这可以叫：

$$
\boxed{\textbf{Compute-domain Warped Prediction}}
$$

AV1 里的 warped reference，不再重建“像素”，而是把 warp 挪到 GEMM 两边。

这比：

$$
W_l=W_r+\Delta W_l
$$

高级得多，因为后者通常需要：

$$
W_rx+\Delta W_lx
$$

两次乘法。

而这个仍然只有一次主体 GEMM。

---

# 3. 真正的杀手可能是“结构化 SKIP”

AV1 最赚钱的 mode 之一从来不是“把东西量得更精妙”，而是：

$$
\boxed{\text{这里什么都不用编码}}
$$

对应 LLM：

假设 SwiGLU 一个 functional neuron：

$$
\theta_j=
(
W_g[j,:],
W_u[j,:],
W_d[:,j]
).
$$

如果 KL-RDO 判断这个 neuron 的 marginal utility 很低：

$$
\Delta KL_j
<
\lambda\,C_j,
$$

就不是给它 2 bit。

而是：

$$
\boxed{\text{整个 neuron 删除}}
$$

于是 intermediate width：

$$
512
\rightarrow
m_e.
$$

例如某 expert 自动变成：

$$
m_e=352.
$$

那么 gate/up：

$$
2048\times512
\rightarrow2048\times352
$$

down：

$$
512\times2048
\rightarrow352\times2048.
$$

这不是单纯：

$$
\text{VRAM减少31\%}
$$

而是 FFN 的：

$$
\boxed{\text{FLOPs也减少约31\%}}.
$$

这才是真正：

$$
\boxed{\text{space ↓,\ compute ↓,\ traffic ↓}}
$$

同时发生。

注意必须是**结构化 neuron atom 删除**，而不是随机 weight sparse。

---

# 4. 所以“神经元 block”最终应该和 GEMM tile 对齐

codec 不应该只优化：

$$
D_{KL}+\lambda R.
$$

应该直接优化：

$$
\boxed{
J=
D_{KL}
+\lambda_M M
+\lambda_B B_{\rm VRAM}
+\lambda_C C_{\rm kernel}
}
$$

其中 $C_{\rm kernel}$ 是**真实硬件成本**。

例如删除 17 个随机 neuron：

$$
512\rightarrow495
$$

可能在 GPU 上几乎没得到速度收益。

但：

$$
512\rightarrow448
$$

刚好少掉一个 64-channel tile，

就可能真实变快。

所以 RDO 的最小单位不应该只是数学 block，而应该是：

$$
\boxed{\text{hardware execution tile}}.
$$

例如宽度候选：

$$
512,448,384,320,256,\ldots
$$

这就是我们的“AV1 superblock partition”与 GPU GEMM tile 统一。

---

# 5. MoE 上还有一个更漂亮的招：**专家别名可以减少 GEMM 次数**

这个我觉得非常值得单独研究。

原来一个 token 选择 8 个 experts：

$$
y
=
\sum_{e=1}^8 g_eE_e(x).
$$

假设 codec 自己发现：

$$
E_3\approx \alpha_3 P,
$$

$$
E_{17}\approx\alpha_{17}P,
$$

$$
E_{91}\approx\alpha_{91}P.
$$

也就是说它们实际上共享同一个 prototype $P$，只差一个 cheap scale。

如果 router 同时选择这三个 expert，那么原来需要：

$$
E_3(x),\quad E_{17}(x),\quad E_{91}(x)
$$

三个 expert GEMM。

现在：

$$
g_3E_3(x)
+
g_{17}E_{17}(x)
+
g_{91}E_{91}(x)
$$

近似变成：

$$
\left(
g_3\alpha_3+
g_{17}\alpha_{17}+
g_{91}\alpha_{91}
\right)
P(x).
$$

于是：

$$
\boxed{\text{3 个 expert GEMM}\rightarrow\text{1 个}}
$$

！！！

这个东西不只是模型压缩。

它直接减少 active experts。

---

## 更一般地

设：

$$
E_e(x)
=
\alpha_eP_{c(e)}(x)
+
R_e(x).
$$

则：

$$
y=
\sum_e g_e\alpha_eP_{c(e)}(x)
+
\sum_e g_eR_e(x).
$$

把相同 prototype 的项聚合：

$$
\boxed{
y=
\sum_c
\left(
\sum_{e:c(e)=c}g_e\alpha_e
\right)
P_c(x)
+
\text{small residuals}.
}
$$

如果：

$$
R_e=0
$$

或足够稀疏，

那么原来的：

$$
8\text{ active experts}
$$

实际可能只需要：

$$
5,\quad4
$$

个 physical prototype GEMM。

这就是：

$$
\boxed{\textbf{MoE route coalescing}}
$$

我认为这可能是 Qwen 35B-A3B 上**时间+空间同时省的最强机制之一**。

---

# 6. residual 也不要设计成第二个 GEMM

之前我们讨论：

$$
W_l=W_g+\Delta_l.
$$

如果直接：

$$
W_gx+\Delta_lx
$$

，很可能变慢。

正确做法应该优先把 residual 约束成**可融合 correction**：

### Level 0

只有 scale：

$$
W_l=D_oW_gD_i.
$$

主体只有一次 GEMM。

### Level 1

scale + permutation：

$$
W_l=D_oP_oW_gP_iD_i.
$$

仍然一次 GEMM。

### Level 2

blockwise affine/codebook correction：

$$
q_l=q_g+\delta_{\rm small}.
$$

在寄存器中修正后直接 INT4 dot。

### Level 3

少量 block residual。

只有异常 tile 才追加计算。

### Level 4

low-rank：

$$
UV^Tx.
$$

最后才允许。

也就是说 residual mode 本身要有一个：

$$
\boxed{\text{compute hierarchy}}
$$

而不是所有 residual 都一视同仁。

---

# 7. 还有一个 RX 6800 特别值得利用的东西：128 MB Infinity Cache

RX 6800 有 **128 MB Infinity Cache** 和最高 **512 GB/s GDDR6**。([AMD](https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6800.html?utm_source=chatgpt.com))

这意味着 layer/expert sharing 不只省“静态显存”。

如果我们把物理 prototype 工作集设计到：

$$
\lesssim128\text{ MB}
$$

或者至少 active working set 落在这个数量级，

那么反复使用的：

$$
Q_g
$$

可能大量命中 cache。

于是原来每个 logical layer：

$$
\text{GDDR6}\rightarrow W_l
$$

变成：

$$
\boxed{
\text{Infinity Cache}\rightarrow Q_g
}
$$

这时候即使 FLOPs 没变，

**实际速度也可能提高。**

所以物理 Bank 大小甚至应该把：

$$
128\text{ MB}
$$

作为 RDO 中一个非常真实的“相变点”。

---

# 8. 这说明我们真正要优化的不是“bpw”

我现在会把最终目标改成：

$$
\boxed{
\min
D_{KL}
+
\lambda_1
\underbrace{\text{resident bytes}}_{\text{空间}}
+
\lambda_2
\underbrace{\text{DRAM bytes/token}}_{\text{带宽}}
+
\lambda_3
\underbrace{\text{dot instructions/token}}_{\text{计算}}
+
\lambda_4
\underbrace{\text{decode instructions}}_{\text{codec overhead}}
}
$$

这比：

$$
KL+\lambda\,bpw
$$

更完整。

因为：

> 3.0 bpw、但需要几十条 unpack 指令，

完全可能输给：

> 4.0 bpw、直接一条 `V_DOT8_I32_I4`。

现有 AMD 上 IQ/Vulkan 的性能问题正好说明了这一点：压得更小并不自动意味着更快，复杂 decoder 可能增加 VGPR、降低 occupancy，甚至让有效带宽变差。([GitHub](https://github.com/ggml-org/llama.cpp/issues/20848?utm_source=chatgpt.com))

---

## 所以我现在最看好的最终形态不是“AV1 + IQ3”

而是：

$$
\boxed{\textbf{Compute-Native Neural Codec}}
$$

它输出的不是一个等待解压的 tensor，而是一个**天然可以执行的程序化矩阵表示**：

$$
\boxed{
W_l
\equiv
\{
\text{INT4 prototype},
\text{permutation},
\text{scale},
\text{structured skip mask},
\text{rare correction}
\}
}
$$

运行时：

$$
\text{cheap transform}(x)
\rightarrow
\boxed{\text{native INT4 dot}}
\rightarrow
\text{cheap transform}(y).
$$

MoE 再加：

$$
\boxed{\text{prototype route coalescing}}
$$

把 8 个逻辑 experts 合并成更少的 physical GEMM。

这就真的可能出现：

$$
\boxed{
35B\rightarrow 10\!-\!15B\ \text{physical}
}
$$

同时：

$$
\boxed{
3B\ \text{active}
\rightarrow
<3B\ \text{physical compute}
}
$$

而且显存流量进一步下降。

这比“AV1 for weights”又高了一个层次：**AV1 给我们的是压缩决策哲学；最终格式则必须由 GPU ISA 反过来决定。** 对你的 RX 6800，`INT4 DOT + structured skip + prototype sharing + route coalescing` 这四个东西，我会优先级拉到最高。

---

**Sources:**

- [vulkan: IQ3 dequant shaders use 64 VGPRs on AMD — 40% occupancy limits bandwidth on HBM GPUs · Issue #20848 · ggml-org/llama.cpp · GitHub](https://github.com/ggml-org/llama.cpp/issues/20848?utm_source=chatgpt.com)
- [Radeon™ RX 6800 Graphics Card](https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6800.html?utm_source=chatgpt.com)

## Prompt:
8/28/2026, 2:52:27 PM

还是需要数学变换，这个是该花的代价，不能省

## Response:
8/28/2026, 2:52:32 PM

对。这里不能为了“kernel 简单”把最值钱的数学结构扔掉。**如果没有变换，最终很容易退化成“更聪明的 IQ3”；真正跨代的地方恰恰是先把权重映射到一个更可压、更可算的坐标系。**

我会把原则写成：

$$
\boxed{
\text{允许付出 }O(n\log n)\text{ 或很小的 }O(nr)
\text{ 变换代价，只要它换来更大的 GEMM/带宽下降}
}
$$

而不是追求“零变换 overhead”。

更具体地，真正应该优化的是总执行时间：

$$
\boxed{
T_{\rm total}
=
T_{\rm transform}
+
T_{\rm compressed\ GEMM}
+
T_{\rm correction}
}
$$

要求的是

$$
T_{\rm total}
<
T_{\rm IQ3/GEMM},
$$

而不是单独要求

$$
T_{\rm transform}=0.
$$

这两者差别非常大。

---

我现在最看好的数学形式其实是：

$$
\boxed{
W
\approx
P_oD_o
\,U_o\,
Q\,
U_i^TD_iP_i
+
R
}
$$

其中 $Q$ 才是真正存储的大矩阵，而且设计成硬件喜欢的 INT4 / structured-low-bit；$P,D$ 是 permutation/sign/scale；$U_i,U_o$ 是数学变换；$R$ 是少量不可解释 residual。

于是计算不是重建 $W$，而是：

$$
x_1=D_iP_ix
$$

$$
x_2=U_i^Tx_1
$$

$$
z=Qx_2
$$

$$
z_1=U_oz
$$

$$
y=P_oD_oz_1+R x.
$$

核心原则仍然是：

$$
\boxed{\text{Never reconstruct }W}
$$

但**绝对允许 transform activation**。

这才是正确方向。

---

尤其值得注意的是，$U$ 不应该随便选 Hadamard。

前面我们推出来的真正候选应该从：

$$
\Sigma^{1/2}H\Sigma^{1/2}
$$

出发。

也就是说 simultaneously 考虑：

$$
\Sigma=\text{source covariance}
$$

和：

$$
H=\text{KL/Fisher sensitivity}.
$$

理想变换：

$$
\Sigma^{1/2}H\Sigma^{1/2}
=
U\Gamma U^T.
$$

这种 **Functional KLT / KL-KLT** 能把原本复杂的 functional distortion：

$$
\delta w^T H\delta w
$$

尽量变成近似可分离的：

$$
\sum_i\gamma_i\delta z_i^2.
$$

这样后面的量化、裁剪、bit allocation 才真正简单。

所以数学变换并不是附属品。

它实际上承担了三件事情：

$$
\boxed{
\text{decorrelation}
+
\text{importance diagonalization}
+
\text{energy compaction}
}
$$

这三件事完成得越好，后面的 codec 越便宜。

---

而且这里可能出现你说的“**花数学代价反而加速 GEMM**”。

例如变换后：

$$
Q=
\begin{bmatrix}
Q_{\rm important}&0\\
0&\approx0
\end{bmatrix}.
$$

原来：

$$
512
$$

维 FFN 实际可能被 transform 集中到：

$$
384
$$

个有效坐标。

那么运行：

$$
U_i^Tx
$$

可能花几千次 add/mul，

但接下来 GEMM 从：

$$
2048\times512
$$

变成：

$$
2048\times384.
$$

少掉的是几十万到几百万级操作。

这笔账完全可能非常划算。

因此正确判断条件是：

$$
\boxed{
C(U_ix)+C(Qx)+C(U_oy)
<
C(Wx)
}
$$

而不是：

> “U 会不会产生 overhead？”

当然会。

问题是它换回来多少。

---

甚至我会把 transform 分成几个级别，而不是只允许一种。

| Transform | 代价 | 表达能力 |
|---|---:|---:|
| permutation/sign | 极低 | 很弱 |
| diagonal scaling | 极低 | 弱 |
| butterfly | $O(n\log n)$ | 中 |
| Hadamard-like | $O(n\log n)$ | 中 |
| block orthogonal | $O(nb)$ | 较强 |
| low-rank update | $O(nr)$ | 较强 |
| dense KLT | $O(n^2)$ | 最强但通常太贵 |

真正的 AV1 精神应该是：

$$
\boxed{\text{transform mode 也参加 RDO}}
$$

而不是规定所有 block 都用 Hadamard。

比如一个 block 可以候选：

$$
T\in
\{
I,
P,
PD,
H,
B_{16},
B_{32},
I+UV^T
\}.
$$

然后算：

$$
J_T
=
D_{KL}
+
\lambda_M M
+
\lambda_B B
+
\lambda_C C_T.
$$

复杂 transform 如果能把 residual 压得极小，就值得用。

如果一个简单 permutation 已经够，就没必要花 KLT。

---

更进一步，**transform block size 也应该自动选择**。

例如 512 个 neuron：

$$
512
$$

可能整体做一次 KLT 太贵。

可以分成：

$$
8\times64.
$$

但如果某 128-neuron 子空间高度相关，就做：

$$
128\times128
$$

block transform。

这实际上就是神经网络版：

$$
\boxed{\text{adaptive transform partition}}
$$

跟 AV1 不同 block 可选不同 transform 的思想完全一致。

---

我甚至觉得“变换的代价”本身可以再被压缩。

假设不同层：

$$
U_l
$$

非常接近。

不要存 40 个 dense $U_l$。

做：

$$
\boxed{
U_l
=
U_0
\prod_{k=1}^{K}
G_{l,k}
}
$$

其中 $G$ 是少量 Givens rotations / butterfly stages。

或者：

$$
U_l
=
U_{\rm shared}
(I+A_lB_l^T).
$$

这样：

> **连坐标系本身也做层间预测。**

这已经非常 AV1 了。

不仅压 signal：

$$
W_l
$$

还压它的 transform parameters：

$$
T_l.
$$

---

最终我会把模型表示理解为：

$$
\boxed{
\text{原模型}
=
\text{便宜的数学坐标变换}
+
\text{极小的低位核心矩阵}
+
\text{少量高价值 residual}
}
$$

而不是：

$$
\text{原模型}
\approx
\text{一堆低 bit 原权重}.
$$

前者才可能真正产生数量级变化。

更极端一点，我们最终追求的其实不是“压缩矩阵”，而是找到一个等价近似：

$$
\boxed{
Wx
\approx
T_o^{-1}
Q
T_i x
}
$$

使得：

$$
\operatorname{size}(Q)\ll\operatorname{size}(W)
$$

同时：

$$
\operatorname{cost}(T_i)
+
\operatorname{cost}(Q)
+
\operatorname{cost}(T_o)
<
\operatorname{cost}(W).
$$

这就同时是：

**有损压缩问题 + 矩阵算法问题 + 编译器/kernel 问题。**

所以你这句话我会直接立为第二条硬规范：

$$
\boxed{\textbf{Mathematical transforms are allowed—even encouraged—if they reduce total rate-distortion-compute cost.}}
$$

第一条是：

$$
\boxed{\textbf{Never fully decompress weights.}}
$$

这两条结合起来，方向就开始真正从“IQ3 优化”走向一个新的计算表示了。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)