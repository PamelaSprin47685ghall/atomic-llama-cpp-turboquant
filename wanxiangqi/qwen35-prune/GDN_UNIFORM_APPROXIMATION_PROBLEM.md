# 2.0 research note: Qwen3.5 Gated Delta Net 128D -> 64D

## Status of this document

This is a **2.0 problem-formulation / expert-review document**, not a 1.0
release requirement and not a claim that a final theorem has already been
proved.  The 1.0 test line exact-copies GDN at 128D/4096 and ships expert +
vocabulary compression independently.

The project has since relaxed the original requirement that GDN compression be
strictly data-free.  For 2.0, calibration is allowed when it is low-capacity,
stable and checked OOS; synchronized QKV local recurrence replay and held-out
perplexity gates are also allowed.  The static/full-domain formulation below
is retained because it provides useful bounds, priors and failure modes, not
because every 2.0 candidate must satisfy the original theorem-first policy.

The original theory-first constraints studied in this note are:

- no dynamic backpropagation;
- no corpus-defined empirical objective;
- no greedy black-box search over validation loss;
- prefer static, closed-form, one-shot spectral, or deterministic top-k rules;
- the approximation should come with a computable **a priori uniform error
  upper bound** over the full admissible input domain;
- overfitting prevention should follow from the formulation itself, not from a
  train/validation split.

The main purpose of this note is to make every assumption and every remaining
gap explicit enough for an approximation-theory / operator-theory expert to
critique.

---

## 1. Architectural object to be compressed

Consider one recurrent qwen35moe block.  The input to the recurrent attention
branch is the output of the block's attention RMSNorm.

Use the following dimensions:

\[
d = 2048,\qquad m = 128,\qquad r = 64,\qquad L = 4.
\]

The teacher has:

- 16 Q/K heads of dimension 128;
- 32 V heads of dimension 128;
- 4-tap depthwise convolution;
- GDN state dimension 128 per V head;
- a post-recurrence gated RMSNorm;
- a final `ssm_out` projection from 4096 to 2048.

The target student has the same stock graph but dimension 64:

- 16 Q/K heads of dimension 64;
- 32 V heads of dimension 64;
- the same 4-tap depthwise-conv structure;
- GDN state dimension 64;
- a 64D post-recurrence RMSNorm;
- `ssm_out` from 2048 to 2048.

The beta/alpha branches are not dimension-reduced by this compression and can
therefore be kept exactly for a local same-input comparison.

---

## 2. Exact full input domain induced by RMSNorm

The actual input sequence seen by a recurrent block is constrained by the
preceding RMSNorm.  This can be described more sharply than by a Euclidean
ball.  Let

\[
D_\gamma=\operatorname{diag}(\gamma_{\rm attn})\in\mathbb R^{d\times d}.
\]

Before multiplication by \(D_\gamma\), RMSNorm maps an arbitrary vector \(u\)
to

\[
z(u)=\frac{u}{\sqrt{\|u\|_2^2/d+\epsilon}},
\qquad \|z(u)\|_2<\sqrt d.
\]

Every direction and every radius strictly below \(\sqrt d\) is attained.
Therefore the closure of the exact RMSNorm image is

\[
\boxed{
\mathcal E_\gamma
=
\left\{D_\gamma z:\ \|z\|_2\le\sqrt d\right\}.
}
\]

Taking the closure does not change any supremum used below.  The formula also
remains valid when some entries of \(\gamma_{\rm attn}\) are zero.  Its support
function is available in closed form:

\[
\boxed{
\sup_{x\in\mathcal E_\gamma}|w^\top x|
=
\sqrt d\,\|D_\gamma w\|_2.
}
\]

Define the full sequence domain

\[
\mathcal X_\gamma
=
\left\{
(x_t)_{t\ge0}:\ x_t\in\mathcal E_\gamma\ \text{for every }t
\right\}.
\]

This is still a product-domain relaxation of the set jointly reachable through
all preceding layers, because arbitrary individually valid RMSNorm outputs are
allowed to occur independently across time.  It nevertheless contains every
legal token-prefix trajectory and is completely data-independent.

---

## 3. Primary approximation target

Let \(F_T\) be the teacher recurrent block as a causal operator and \(F_S\) a
stock 64D student block.  The primary local operator problem is

\[
\boxed{
\inf_{F_S\in\mathcal F_{64}^{\rm stock}}
\ \sup_{x\in\mathcal X_\gamma}
\ \sup_{t\ge 0}
\|F_T(x)_{t}-F_S(x)_{t}\|_2.
}
\]

The supremum over \(t\ge0\) is intentional.  A finite calibration/context
length is not part of the definition.

The current concrete approximation family under study is a **coordinate
subspace family**: the student keeps 64 teacher Q/K coordinates per Q/K head
and 64 teacher V coordinates shared by all V heads.  This is not claimed to be
the globally best 64D stock family; it is being considered because it admits
static realizability and potentially strong uniform certificates.

An expert may propose a larger closed-form stock-realizable family if it
retains comparable guarantees.

---

## 4. Exact per-channel stock functions before Q/K normalization

For branch \(a\in\{q,k,v\}\), head \(h\), coordinate \(i\), write the teacher
projection-plus-depthwise-conv preactivation as

\[
A^a_{h,i,t}
=
\sum_{\tau=0}^{L-1}
c^a_{h,i,\tau}
\langle w^a_{h,i},x_{t-\tau}\rangle,
\]

and the post-SiLU channel function as

\[
U^a_{h,i,t}=\operatorname{SiLU}(A^a_{h,i,t}).
\]

Using the exact support function of \(\mathcal E_\gamma\),

\[
|A^a_{h,i,t}|
\le
\sqrt d\,\|D_\gamma w^a_{h,i}\|_2\,\|c^a_{h,i}\|_1.
\]

Since \(|\operatorname{SiLU}(z)|\le |z|\), define the data-independent channel
envelope

\[
M^a_{h,i}
:=
\sqrt d\,\|D_\gamma w^a_{h,i}\|_2\,\|c^a_{h,i}\|_1,
\]

which yields

\[
|U^a_{h,i,t}|\le M^a_{h,i}
\qquad
\text{for every }x\in\mathcal X_\gamma,\ t\ge0.
\]

This is a genuine full-domain statement computed entirely from weights.

For the post-recurrence gate projection \(z\), which does not use the same
depthwise conv path, an analogous envelope is

\[
M^z_{h,i}=\sqrt d\,\|D_\gamma w^z_{h,i}\|_2.
\]

---

## 5. Uniform omission bound for a coordinate subset

Let \(S\subset\{1,\ldots,m\}\), \(|S|=r\), be retained coordinates and
\(O=S^c\) the omitted coordinates.  If the selected channels are copied
exactly, the raw vector difference is supported only on \(O\).  Therefore

\[
\|U^a_{h,O,t}\|_2
\le
\left(
\sum_{i\in O}(M^a_{h,i})^2
\right)^{1/2}
\equiv E^a_h(S).
\]

This is the first candidate certificate component.

### Possible spectral tightening

For omitted coordinates define the linear 4-frame operator

\[
B^a_{h,O}
=
\begin{bmatrix}
c_{i,0}w_i^\top & c_{i,1}w_i^\top & \cdots & c_{i,L-1}w_i^\top
\end{bmatrix}_{i\in O}.
\]

For the stacked normalized coordinates

\[
Z_t=[z_t;z_{t-1};\ldots;z_{t-L+1}],
\qquad
\|Z_t\|_2\le\sqrt{Ld},
\]

and \(\mathcal D_\gamma=I_L\otimes D_\gamma\), one also has

\[
\|A^a_{h,O,t}\|_2
\le
\sqrt{Ld}\,\|B^a_{h,O}\mathcal D_\gamma\|_2.
\]

Together with the global SiLU Lipschitz constant this can give a tighter
spectral envelope than the coordinatewise \(\ell_2\) tail.  However, selecting
the best 64 rows for this spectral norm is no longer automatically a closed
form top-k problem.  The separable envelope above is therefore attractive as
a provable baseline even when it is looser.

---

## 6. Q/K L2 normalization: global and truncation-specific bounds

Q and K use `ggml_l2_norm`, whose mathematical map is

\[
N_\varepsilon(u)
=
\frac{u}{\max(\|u\|_2,\varepsilon)}.
\]

This is the Euclidean projection of \(u/\varepsilon\) onto the unit ball, and
is therefore globally \(1/\varepsilon\)-Lipschitz:

\[
\|N_\varepsilon(u)-N_\varepsilon(v)\|_2
\le
\frac1\varepsilon\|u-v\|_2.
\]

Let \(P_S:\mathbb R^r\to\mathbb R^m\) be the coordinate injection.  The
embedded student Q vector is

\[
\widetilde q
=P_S N_\varepsilon(P_S^\top U^q).
\]

Equivalently this is \(N_\varepsilon(P_SP_S^\top U^q)\), embedded into the
teacher space.  Hence

\[
\boxed{
\|q-\widetilde q\|_2
\le
\frac{E^q_h(S)}\varepsilon.
}
\]

The same holds for K.  For coordinate truncation there is also a useful
dimension-free geometric cap.  Write \(u=(s,o)\) in selected/omitted
coordinates and \(v=(s,0)\).  A direct case split according to whether
\(\|u\|_2\) and \(\|s\|_2\) are above or below \(\varepsilon\) gives

\[
\boxed{
\|N_\varepsilon(u)-N_\varepsilon(v)\|_2\le\sqrt2.
}
\]

Therefore a sharper static certificate is

\[
\boxed{
\delta^q_h(S)
=
\min\left\{\sqrt2,\frac{E^q_h(S)}\varepsilon\right\},
\qquad
\delta^k_h(S)
=
\min\left\{\sqrt2,\frac{E^k_h(S)}\varepsilon\right\}.
}
\]

The \(\sqrt2\) cap matters because the model epsilon may be small enough that
the naive \(E/\varepsilon\) expression saturates for almost every coordinate
set.  This is not merely a numerical inconvenience: it warns that a
full-domain Q/K-normalization certificate can be intrinsically coarse unless
the omitted channel map is structurally small.

There is no corpus-dependent constant here.

### Exact obstruction criterion

For a fixed coordinate set \(S\), let \(\Phi(X)=U^q_h(X)\) be the exact
pre-normalization Q channel map on an admissible four-step input window.  If
there exists \(X\) such that

\[
P_S^\top\Phi(X)=0,
\qquad
\|\Phi(X)\|_2\ge\varepsilon,
\]

then the embedded student normalized query is zero while the teacher query has
unit norm.  Hence

\[
\boxed{
\sup_X\|q(X)-\widetilde q(X)\|_2\ge1.
}
\]

The same statement holds for K.  Thus a genuinely small uniform error is
impossible for a coordinate subset whenever the selected channels admit such
a full-domain annihilating input.  Whether the actual model has this property
for every 64-coordinate subset is an important structural question for expert
review; empirical data cannot answer it.

---

## 7. Uniform contraction of the recurrent state

For one V head, beta is produced by a sigmoid.  The exact RMSNorm support
function gives a static upper bound on its preactivation:

\[
w_\beta^\top x_t
\le
\sqrt d\,\|D_\gamma w_\beta\|_2.
\]

Hence

\[
\boxed{
0\le\beta_t\le
\bar\beta
:=
\sigma\!\left(\sqrt d\,\|D_\gamma w_\beta\|_2\right)
<1.
}
\]

The decay branch has the form

\[
g_t
=a\,\operatorname{softplus}(w_\alpha^\top x_t+b),
\]

where the loaded `ssm_a` value \(a\) is expected to be strictly negative.

Again using the exact support function,

\[
w_\alpha^\top x_t+b
\ge
b-\sqrt d\,\|D_\gamma w_\alpha\|_2
\equiv \ell.
\]

Softplus is increasing.  Since \(a<0\),

\[
g_t
\le
a\,\operatorname{softplus}(\ell)<0.
\]

Therefore

\[
\boxed{
e^{g_t}\le
\rho
:=
\exp\left(a\,\operatorname{softplus}(\ell)\right)
<1
}
\]

uniformly on the full input domain.

This strict \(\rho<1\) is the key fact that makes an infinite-horizon bound
possible.

### Assumption to verify from the actual model

Every recurrent head used by the target model must satisfy `ssm_a < 0` with a
finite margin.  The tool should eventually emit both \(\rho_h\) and
\(\bar\beta_h\) for every recurrent layer/head as part of the certificate.

---

## 8. Teacher state bound for arbitrary prefix length

Up to a transpose convention that does not affect spectral/Frobenius norm
bounds, the delta recurrence may be written

\[
S_t
=
e^{g_t}S_{t-1}(I-\beta_t k_tk_t^\top)
+\beta_t v_tk_t^\top.
\]

Because Q/K normalization gives \(\|k_t\|_2\le1\), and
\(0\le\beta_t\le\bar\beta_h<1\),

\[
\|I-\beta_t k_tk_t^\top\|_2\le1.
\]

Define the full-domain V envelope for head \(h\):

\[
V_h
=
\left(\sum_{i=1}^{m}(M^v_{h,i})^2\right)^{1/2}.
\]

Then

\[
\|S_t\|_2
\le
\rho_h\|S_{t-1}\|_2+\bar\beta_hV_h.
\]

For a state originating from the normal zero state and any arbitrarily long
legal prefix,

\[
\boxed{
\|S_t\|_2
\le
\frac{\bar\beta_hV_h}{1-\rho_h}
\equiv \bar S_h.
}
\]

The same bound covers a recurrent cache state carried across chunks, provided
that state itself was generated by a legal prefix.

This is the proposed mechanism for replacing all finite-context calibration
assumptions.

---

## 9. Uniform state-error recurrence under coordinate truncation

For one V head let \(P_V\) inject the selected V coordinates and \(P_K\) inject
the selected Q/K coordinates associated with that head.  Embed the student
state as

\[
\widetilde S_t=P_V S'_t P_K^\top.
\]

Let

\[
\Delta_t=\|S_t-\widetilde S_t\|_2.
\]

Using the recurrence above, \(\|k\|,\|\widetilde k\|\le1\), and

\[
\|kk^\top-\widetilde k\widetilde k^\top\|_2
\le
(\|k\|+\|\widetilde k\|)\|k-\widetilde k\|
\le2\delta^k,
\]

one obtains the conservative one-step estimate

\[
\Delta_t
\le
\rho_h\Delta_{t-1}
+\bar\beta_h\delta^v_h
+\bar\beta_h\left(2\rho_h\bar S_h+V_h\right)\delta^k_h,
\]

where

\[
\delta^v_h(S_V)=E^v_h(S_V).
\]

Therefore the infinite-horizon state error obeys

\[
\boxed{
\Delta_\infty
\le
\frac{
\bar\beta_h\delta^v_h
+\bar\beta_h(2\rho_h\bar S_h+V_h)\delta^k_h
}{1-\rho_h}.
}
\]

This is a central candidate theorem to check carefully.  A tighter derivation
may improve the constants, but the important structural property is the
geometric denominator \(1-\rho_h\), obtained without any data.

---

## 10. Combine the recurrence scale and post-GDN RMSNorm exactly

Let \(\widehat q_t\) denote Q after `ggml_l2_norm` but before the delta-net
dimension scale, and define

\[
a_t=S_t\widehat q_t,
\qquad
a'_t=S'_t\widehat q'_t.
\]

The delta-net implementation supplies \(a_t/\sqrt m\) to the teacher
post-GDN RMSNorm and \(a'_t/\sqrt r\) to the student RMSNorm.  Before those
dimension scales, the embedded state/query error satisfies

\[
\boxed{
\eta_h
:=
\|a_t-P_Va'_t\|_2
\le
\Delta_\infty+\bar S_h\delta^q_h.
}
\]

It is preferable not to bound the \(1/\sqrt n\) scale and RMSNorm separately.
For dimension \(n\), their exact composition is the radial map

\[
\boxed{
G_{n,\epsilon}(a)
:=
R_{n,\epsilon}\!\left(\frac a{\sqrt n}\right)
=
\frac{\sqrt n\,a}{\sqrt{\|a\|_2^2+n^2\epsilon}}.
}
\]

Its radial and tangential Jacobian eigenvalues are both bounded by their value
at the origin, so

\[
\boxed{
\operatorname{Lip}(G_{n,\epsilon})
=
\frac1{\sqrt n\sqrt\epsilon}.
}
\]

This avoids double-counting a standalone dimension-scale defect and then a
second RMSNorm Lipschitz penalty.

---

## 11. Static minimax compensation for the 128D -> 64D norm defect

The stock `ssm_norm.weight` can be multiplied by a static scalar
\(\alpha>0\) on the selected coordinates.  For an embedded student vector of
norm \(s\), the exact teacher/student radial-scale mismatch is

\[
D^{\rm comb}_{m,r,\epsilon}(\alpha;B)
=
\sup_{0\le s\le B}
s
\left|
\frac{\sqrt m}{\sqrt{s^2+m^2\epsilon}}
-
\alpha\frac{\sqrt r}{\sqrt{s^2+r^2\epsilon}}
\right|.
\]

One may safely take \(B=\bar S_h\), because
\(\|a'_t\|_2\le\|S'_t\|_2\|\widehat q'_t\|_2\le\bar S_h\).

Let \(\Gamma=\operatorname{diag}(\gamma_{\rm ssm})\) be the teacher
post-GDN RMSNorm weight and \(\Gamma_S\) its selected diagonal.  Decomposing
through \(G_{m,\epsilon}(P_Va'_t)\) yields

\[
\boxed{
\begin{aligned}
&\|\Gamma G_{m,\epsilon}(a_t)
-P_V\alpha\Gamma_SG_{r,\epsilon}(a'_t)\|_2\\
&\quad\le
\frac{\|\gamma_{\rm ssm}\|_\infty}{\sqrt m\sqrt\epsilon}\,\eta_h
+
\|\gamma_{{\rm ssm},S}\|_\infty
D^{\rm comb}_{m,r,\epsilon}(\alpha;\bar S_h).
\end{aligned}
}
\]

Thus the pure dimension-change term is a **one-dimensional static minimax
problem**:

\[
\boxed{
\alpha_h^*
=
\arg\min_{\alpha>0}
D^{\rm comb}_{m,r,\epsilon}(\alpha;\bar S_h).
}
\]

The same `ssm_norm.weight` is shared across the 32 V heads in the stock
architecture, so production compression may require one common \(\alpha\) per
recurrent layer rather than one \(\alpha_h\) per head.  The corresponding
problem is to minimize the maximum of the 32 head-specific defect functions.
It remains one-dimensional and data-independent.  Expert review is requested
on whether either form admits a convenient equioscillation closed form; an
interval-certified scalar minimization would also preserve the requested
static character.

---

## 12. Gated RMSNorm and gate-coordinate omission

Define the post-norm vectors

\[
u_t=\Gamma G_{m,\epsilon}(a_t),
\qquad
\widetilde u'_t=P_V\alpha\Gamma_SG_{r,\epsilon}(a'_t),
\]

and the gate vectors

\[
g_t=\operatorname{SiLU}(z_t),
\qquad
\widetilde g'_t=P_VP_V^\top g_t.
\]

The second equality is exact because selected gate-projection rows are copied
without approximation.  The teacher and embedded student features are

\[
h_t=u_t\odot g_t,
\qquad
\widetilde h'_t=\widetilde u'_t\odot\widetilde g'_t.
\]

Let

\[
M^z_{h,\infty}=\max_i M^z_{h,i}.
\]

Because \(\|G_{r,\epsilon}(a')\|_2\le\sqrt r\),

\[
\|\widetilde u'_t\|_\infty
\le
\alpha\|\gamma_{{\rm ssm},S}\|_\infty\sqrt r.
\]

Using

\[
\|a\odot b-c\odot d\|_2
\le
\|a-c\|_2\|b\|_\infty
+\|c\|_\infty\|b-d\|_2,
\]

the complete per-head feature certificate is

\[
\boxed{
\begin{aligned}
\delta^h_h
&:=\|h_t-\widetilde h'_t\|_2\\
&\le
M^z_{h,\infty}
\left[
\frac{\|\gamma_{\rm ssm}\|_\infty}{\sqrt m\sqrt\epsilon}\,\eta_h
+
\|\gamma_{{\rm ssm},S}\|_\infty
D^{\rm comb}_{m,r,\epsilon}(\alpha;\bar S_h)
\right]\\
&\quad+
\alpha\|\gamma_{{\rm ssm},S}\|_\infty\sqrt r\,E^z_h(S_V).
\end{aligned}
}
\]

Every term is static and computable from model weights, dimensions and
normalization epsilons.  No sampled trajectory appears in the bound.

---

## 13. Final ssm_out projection and quantization error

Flatten the 32 V heads and let \(W_o\in\mathbb R^{2048\times4096}\) be the
teacher `ssm_out` operator after dequantization.  Let \(P_H\) inject all
selected V-head coordinates into the 4096D teacher feature space.

If the student stored exactly

\[
W'_o=W_oP_H,
\]

then

\[
\|W_oh-W'_oh'\|_2
\le
\|W_o\|_2\,\|h-P_Hh'\|_2.
\]

For arbitrary selected columns in a quantized GGUF, exact storage may be
impossible.  Let the actual requantized student weight be \(\widehat W'_o\) and

\[
E_Q=\widehat W'_o-W_oP_H.
\]

Then the deterministic serialization error contributes

\[
\boxed{
\|E_Qh'\|_2
\le
\|E_Q\|_2\,\|h'\|_2
\le
\|E_Q\|_F\,\|h'\|_2.
}
\]

The student feature norm is itself statically bounded.  For one V head,

\[
\|h'_h\|_2
\le
\alpha\|\gamma_{{\rm ssm},S}\|_\infty\sqrt r\,
M^{z,S}_{h,\infty},
\]

where \(M^{z,S}_{h,\infty}\) is the maximum gate envelope among selected
coordinates.  Therefore, after flattening all 32 V heads,

\[
\boxed{
\|h'\|_2
\le
H'_{\max}
:=
\alpha\|\gamma_{{\rm ssm},S}\|_\infty
\sqrt{
r\sum_{h=1}^{32}(M^{z,S}_{h,\infty})^2
}.
}
\]

The Frobenius version is cheap and fully static; a spectral norm gives a
tighter certificate if desired.

This is the preferred way to allow arbitrary coordinate sets without pretending
that packed-weight rewriting is exact.  In particular, Q4_K repacking can
change block-level scale/min metadata even when every selected 32-value
subgroup is source-aligned, so a nonzero deterministic \(E_Q\) may remain.

---

## 14. Candidate block certificate

Let

\[
\delta_H
=
\left(\sum_{h=1}^{32}(\delta^h_h)^2\right)^{1/2}.
\]

Combining the previous pieces gives the candidate block certificate

\[
\boxed{
\sup_{x\in\mathcal X_\gamma}\sup_{t\ge0}
\|F_T(x)_t-F_S(x)_t\|_2
\le
\|W_o\|_2\,\delta_H
+
\|E_Q\|_2\,H'_{\max}.
}
\]

If desired, replace either spectral norm by its Frobenius norm to obtain a
cheaper but looser fully explicit certificate.

The proposed certificate \(\mathcal U\) contains only:

- model weight norms;
- 4-tap conv coefficients;
- normalization epsilons;
- the strict recurrent contraction factors \(\rho_h\);
- selected/omitted coordinate tail envelopes;
- the static RMSNorm dimension-defect minimax term;
- the final `ssm_out` operator norm;
- an explicit quantization error term.

It contains **no samples and no empirical expectations**.

---

## 15. Closed-form coordinate selection from a separable relaxation

The certificate naturally produces terms like

\[
A_q\sqrt{\sum_{i\notin S}Q_i^2}
+
A_k\sqrt{\sum_{i\notin S}K_i^2}
\]

for a Q/K head, where the same coordinate set must serve Q and K.

By Cauchy-Schwarz,

\[
A_q\sqrt{T_q(S)}+A_k\sqrt{T_k(S)}
\le
\sqrt2
\sqrt{
A_q^2T_q(S)+A_k^2T_k(S)
}.
\]

Therefore define the static coordinate score

\[
s_i^{QK}
=
A_q^2Q_i^2+A_k^2K_i^2.
\]

Then

\[
A_q^2T_q(S)+A_k^2T_k(S)
=
\sum_{i\notin S}s_i^{QK}.
\]

Hence the 64-coordinate set minimizing this relaxed certificate is obtained
**exactly** by keeping the 64 largest \(s_i^{QK}\).

No iterative search is needed.

The same construction can combine V and post-recurrence gate terms.  For
example, after absorbing head-dependent propagation constants into static
weights,

\[
s_i^V
=
\sum_h
\left[
(A^v_hM^v_{h,i})^2
+(A^z_hM^z_{h,i})^2
\right],
\]

and the globally shared 64 V coordinates are the 64 largest \(s_i^V\).

### What is actually guaranteed

This top-k rule is globally optimal only for the **chosen separable majorant**,
not for the sharp certificate above and not for the true minimax operator
error.  In particular, the sharper Q/K factor
\(\min\{\sqrt2,E/\varepsilon\}\), selected-coordinate RMSNorm factors, and
quantization error are generally nonseparable.  A closed-form top-k rule may
therefore minimize a further upper bound in which those terms are replaced by
selection-independent majorants.

For two **uncapped grouped tail terms with fixed constants**, the Cauchy step
loses at most the explicit \(\sqrt2\) factor relative to their sum.  More
generally \(p\) such terms produce at most a \(\sqrt p\) factor.  This factor
does **not** compare the top-k majorant to the sharp capped certificate or to
the true minimax error; no such ratio is currently claimed.

This kind of statement is preferred over a heuristic importance score because
both the upper bound and the relaxation factor are explicit.

---

## 16. Why strong RRQR / interpolative decomposition is not automatically valid

Strong rank-revealing QR and interpolative decomposition can give excellent
row/column subset guarantees for linear operators.  However the stock GDN
channel is

\[
\operatorname{SiLU}(\text{projection followed by depthwise conv}),
\]

and the recurrence is nonlinear in normalized Q/K.

An RRQR theorem that reconstructs omitted teacher rows as linear combinations
of selected rows would require the student to realize those linear
combinations **after** the nonlinear channel map.  The stock 64D architecture
does not obviously provide that reconstruction before the recurrence.

Therefore RRQR/ID should not be inserted merely because it has a good linear
matrix theorem.  It is valid only if a reviewer can supply an explicit
stock-realizable lifting/reconstruction argument through the nonlinear
recurrence.

The simpler coordinate-tail certificate above avoids that gap.

---

## 17. Local GDN operator bound versus whole-model logit bound

The formulation above gives a route to a uniform bound on one recurrent block
for every RMSNorm-bounded input sequence.

Extending this to a useful uniform bound on final model logits is harder.

The model contains top-k MoE routing.  Top-k routing is discontinuous at expert
score ties.  Therefore a global small-perturbation Lipschitz constant for the
entire network does not exist without an additional routing-margin assumption.

Two possible goals must be distinguished:

### Goal A: certified GDN block/operator compression

Prove

\[
\sup_{x\in\mathcal X_\gamma,t\ge0}
\|F^{\rm GDN}_T(x)_t-F^{\rm GDN}_S(x)_t\|
\le\epsilon_{\rm GDN}.
\]

This appears feasible without data and without assumptions about later MoE
routing.

### Goal B: certified final-logit compression

Prove a nontrivial uniform bound on

\[
\sup_{\text{all token sequences},t}
\|\operatorname{logits}_T-\operatorname{logits}_S\|.
\]

For this, one must either:

- derive a bound that tolerates arbitrary routing changes (likely very loose);
- prove a global routing margin, which may be impossible over the entire
  relaxed domain;
- or define a different end-to-end metric that remains meaningful under
  routing discontinuities.

Expert guidance is especially requested here.

---

## 18. Quantization-feasible coordinate sets

There are two distinct feasible families and they should not be conflated.

### 18.1 Arbitrary mathematical coordinates

Allow any 64 of 128 coordinates.  QKV/conv/gate rows and F32 norm weights can
be gathered directly.  Quantized `ssm_out` must be requantized, and the
resulting \(\epsilon_Q\) is included in the certificate.

This gives the cleanest approximation problem.

### 18.2 Quant-group-compatible coordinates

Restrict the coordinate set so every selected 32-value output group maps to a
real source quantization subgroup.  This avoids the more serious error of
pretending that an arbitrary 32-coordinate pattern is represented by one
source subgroup.

For Q8_0, an aligned 32-value block can be copied byte-for-byte and therefore
adds no new storage error.

For Q4_K, subgroup compatibility is **not** enough for zero error.  A Q4_K
256-value block shares block-level `d/dmin`; when eight selected source
subgroups are repacked into a new block, those shared scale/min values are
recomputed and the subgroup scale/min codes are quantized again.  Thus the
4-bit value codes may be preserved while the represented floating values
change.  A deterministic \(E_Q\) term is still required in general.

True byte-exact Q4_K copying requires the destination 256-value block to be an
entire source block in the same order, which may be incompatible with the
64-per-head geometry.  Any 32-coordinate grouping introduced here is a
serialization constraint only, not GDN approximation theory.

If quant-group compatibility materially worsens the analytic certificate,
arbitrary coordinates plus explicit dequantize/select/requantize error may be
preferable.

---

## 19. Overfitting policy

The static planner should obey the following rule:

> No corpus statistic may appear in any coordinate score, contraction
> constant, normalization compensation, or acceptance criterion.

Allowed inputs to the planner are:

- model weights;
- tensor types / quantization layout;
- architecture dimensions;
- fixed mathematical constants and epsilons.

After the model has been produced, corpora may be used for:

- perplexity audits;
- teacher-KL audits;
- checking whether the uniform certificate is extremely conservative;
- debugging implementation mistakes.

Those audits must not feed back into the selected 64D coordinates if the goal
is to preserve the stated data-independent guarantee.

### Non-vacuity policy

A formal upper bound is useful only if it is numerically non-vacuous on the
actual model.  In particular:

- Q/K normalization errors may saturate at the \(\sqrt2\) geometric cap;
- \(1/(1-\rho_h)\) may be large when the worst-case alpha decay is weak;
- the product ellipsoid allows adversarial temporal combinations that may
  never be reached by real token trajectories;
- Q4_K repacking error may dominate a small functional truncation bound.

The certificate generator should therefore emit every intermediate constant,
not only the final number.  If the final bound is vacuous, the correct
conclusion is that this theorem does not certify the requested compression on
the full domain.  Held-out perplexity must not be used to reinterpret a
vacuous certificate as a proof.

---

## 20. Proposed static algorithm if the certificate is accepted

For every recurrent layer:

1. Read the attention-RMSNorm weight and use the exact ellipsoid support
   function \(\sqrt d\|D_\gamma w\|_2\).
2. For every Q/K/V channel, compute
   \(M=\sqrt d\|D_\gamma w\|_2\|c\|_1\).
3. Compute gate envelopes
   \(M^z=\sqrt d\|D_\gamma w^z\|_2\).
4. Compute every static \(\bar\beta_h<1\) and recurrent contraction factor
   \(\rho_h<1\) from beta/alpha weights, dt bias and negative `ssm_a`.
5. Compute \(V_h\),
   \(\bar S_h=\bar\beta_hV_h/(1-\rho_h)\), Q/K normalization caps, and the
   infinite-horizon state-error coefficients.
6. Construct a **separable majorant** of the sharp certificate and keep the
   top 64 coordinates for each paired Q/K head according to its exact static
   top-k rule.
7. Construct the corresponding shared-V/gate majorant and keep the top 64 V
   coordinates.  Record explicitly which sharp terms were upper-bounded to
   obtain separability.
8. Solve the one-dimensional combined scale+RMSNorm minimax problem for the
   layer-shared \(\alpha\).
9. Evaluate the sharper post-selection certificate, including the
   \(\sqrt2\) Q/K caps rather than only the optimization majorant.
10. Rewrite the stock 64D tensors and compute the actual deterministic
    `ssm_out` error \(E_Q\) after Q8_0 copy or Q4_K repack/requantization.
11. Emit a machine-readable certificate containing every intermediate
    constant, the optimization majorant, the sharper final block bound, and
    the quantization term separately.
12. Run empirical perplexity/KL only as a post-hoc audit.

The computational cost is essentially tensor-norm scans plus sorting 128
scores per head; there is no sequence replay and no optimization loop.

---

## 21. Questions for expert review

The most useful review would address these points.

1. **Exact input domain.** The closure of one RMSNorm output is the ellipsoid
   \(\mathcal E_\gamma=\{D_\gamma z:\|z\|\le\sqrt d\}\).  Is there a
   substantially tighter *temporal* reachable-set description than the product
   domain \(\mathcal X_\gamma\) that remains static and tractable?
2. **Q/K normalization geometry.** Is the \(\sqrt2\) truncation cap sharp for
   \(N_\varepsilon(u)=u/\max(\|u\|,\varepsilon)\)?  Can the exact channel-map
   structure yield a stronger computable uniform bound than
   \(\min\{\sqrt2,E/\varepsilon\}\)?
3. **Obstruction/lower bound.** Can the annihilating-input criterion in
   Section 6 be strengthened into a practical lower bound proving that certain
   64-coordinate subsets, or all such subsets, must incur order-one uniform
   Q/K error?
4. **State recurrence constants.** Can the current
   \(\bar\beta(2\rho\bar S+V)\delta_k\) term be materially tightened by using
   more of the positive-semidefinite structure of
   \(I-\beta kk^\top\) or coupling beta/alpha extrema instead of maximizing
   them separately?
5. **Infinite horizon.** Are there hidden cases in the exact llama.cpp GDN
   recurrence/cache semantics that invalidate the simple \(\rho<1\)
   geometric argument?
6. **Combined dimension defect.** Does
   \(\min_\alpha D^{\rm comb}_{128,64,\epsilon}(\alpha;B)\) have a useful
   equioscillation closed form, especially when one shared \(\alpha\) must
   cover 32 head-specific values of \(B\)?
7. **Coordinate top-k majorant.** Is there a stronger static/closed-form subset
   theorem that directly handles capped normalization terms or provides a
   finite approximation ratio to the sharp certificate, without requiring a
   non-stock reconstruction after SiLU?
8. **V subset coupling.** What is the cleanest way to aggregate the 32 V heads
   and gate tails while retaining an explicit approximation factor for one
   shared 64-coordinate set?
9. **Final `ssm_out`.** For Q4_K repacking, is there a cheap structured
   spectral bound on \(E_Q\) that is substantially tighter than Frobenius norm
   and that exposes the block-scale/min structure?
10. **Whole-model theorem.** Is a useful end-to-end uniform logit bound possible
    in the presence of top-k MoE routing without a global routing-margin
    assumption?
11. **Better function class.** Is there a strictly larger 64D stock-realizable
    approximation class with a static closed-form solution and a comparable
    full-domain certificate, so that coordinate truncation is unnecessarily
    restrictive?

---

## 22. What would count as a successful replacement theorem

A stronger proposed method is preferable if it provides all of the following:

1. an explicitly defined full input domain independent of sampled data;
2. a stock-realizable 64D student construction;
3. a finite, computable, non-vacuous uniform error upper bound;
4. treatment of arbitrarily long recurrent prefixes;
5. explicit handling of Q/K normalization and 128D->64D RMSNorm;
6. explicit handling of quantization/requantization error;
7. a static solution rule (closed form, deterministic top-k, or one-shot
   spectral factorization preferred);
8. a theorem explaining any relaxation/approximation factor introduced by the
   solution rule.

Empirical performance can still be excellent or poor; it is not part of the
proof obligation above.

