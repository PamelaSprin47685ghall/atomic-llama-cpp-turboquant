# D1 两步原型实现设计（v1，写代码前最后一步）

日期：2026-09-26。run-id：20260926T0421。承接 `input-table.md` + `gate-preregistration.md`。

## 源码事实链（本轮新核验，行号均属 16e28b8ac 基座）

1. **现状 draft 每步的 host 往返清单**（`common/speculative.cpp:1855-1927`）：
   - `llama_decode(ctx_dft, batch)`（host 路径）
   - `common_sampler_sample` → `llama_sampler_sample`（llama-sampler.cpp:808）→
     `llama_get_sampled_token_ith` → `output_reorder()`（仅 host 数组交换）→
     读 `sampling.sampled.data[row]`。该 4 字节由 `copy_tensor_async_ints`
     （llama-context.cpp:4410 → 3077-3100 `ggml_backend_tensor_get_async`）在 decode 尾部
     异步入队；**完成点由下一次 `ggml_backend_sched_synchronize` 保证**（decode 尾部
     或下一次 decode 的输入同步），无独立 host 自旋。
   - `llama_get_embeddings_nextn_ith` → hidden 行（hc_dim×F32 ≈ 4×n_embd×4 B）回读 host
     → memcpy 进 `batch.embd`。
2. **设备通道已具备的闭环件**：
   - hidden：`decode_predefined_hidden`（llama-predefined-hidden.cpp:229）接受
     RESULT/CARRY/SEED→INPUT 的设备 copy_ranges；draft 图输出 `t_h_nextn` 已由
     `predefined_hidden_capture`（llama-context.cpp:4388）在 decode 内捕获进
     `mtp_persistent_result`。`feedback` range（speculative.cpp:1853-1857）已实现
     "第 i 步用第 i-1 步 RESULT 行"——即 **device_hidden 模式下 hidden 已不回 host**。
   - token：draft 图含 `greedy_argmax`（`build_sampling` 对 MTP ctx 生效），`t_sampled`
     是设备张量；唯一 host 环节是 `copy_tensor_async_ints` 的 4 B 回读 + host 把 id
     写进下一 batch.token。
3. **因此 D1 两步原型的真正缺口只有一个**：第二步的 `batch.token[0]` 目前必须由 host
   填写（`common_batch_add`），因为 `llama_predefined_decode_hidden` 要求 `batch.token`
   非 null（llama-predefined-hidden.cpp:232 `!batch.token → return -1`），且
   `llm_graph_input_embd_h::set_input`（llama-graph.cpp:119-125）用
   `ggml_backend_tensor_set(tokens, ubatch->token, …)` 从 host 上传 token id。

## 两步原型 v1 设计（最小机制差异，满足 §8.3 硬门）

**机制**：新增 opt-in `GGML_TP5_MTP_DEVICE_TOKENS=1`（与 `GGML_TP5_MTP_DEVICE_HIDDEN=1`
联合生效）。在 `decode_predefined_hidden` 增加一条设备 token range 语义：
`batch.token` 允许为 null，当 `predefined_hidden->bound_input` 且调用方提供
"token 来源 = 本 context 上一步 `t_sampled` 设备张量"时，`set_input` 不再从 host 上传，
而是把 `t_sampled`（I32, 设备）作为 `inp->tokens` 的设备侧写入（设备 copy 或图内直连，
以实现时 backend 支持为准；第一版用 `ggml_backend_tensor_copy_async` 设备到设备）。

**两步执行序**（draft() 内，n_max=2）：
1. 第一步：现状路径（token=已提交 `dp.id_last`，host 填写一次——该 token 本就在 host）。
   hidden 走 CARRY→INPUT 设备通道（已有）。图执行后 `t_sampled` 与 `t_h_nextn` 均在设备。
2. 第二步：不调 `llama_decode`/`common_sampler_sample`。直接以设备 token range +
   第一步 RESULT→INPUT hidden range 调 `llama_predefined_decode_hidden`。
   第二步产出的 `t_sampled`/`t_h_nextn` 留设备，供 cycle 边界 accept/commit 消费
   （commit 仍走 K/V-only catch-up，不改）。
3. **host 参与仅剩**：cycle 边界读第二步 token id 一次（4 B，用于 result push 与
   target verify batch 组装——这是语义需要，不是执行依赖）。

**与撤回原型（3488 ms）的差异**：无 unrolled 大图；无逐 token hidden host 物化；
无逐 token `common_sampler_sample` 调用；新增仅一条设备 token copy。

## 生死门测量计划（预注册，§8.3）

- 原型：`GGML_TP5_MTP_DEVICE_HIDDEN=1 GGML_TP5_MTP_DEVICE_TOKENS=1 --spec-draft-n-max 2`
- 对照 A：现状 n=2（`GGML_TP5_MTP_DEVICE_HIDDEN=0`，host hidden+token 往返）
- 对照 B（机制消融）：`GGML_TP5_MTP_DEVICE_HIDDEN=1 GGML_TP5_MTP_DEVICE_TOKENS=0`，n=2
  ——分离 "hidden 闭环" 与 "token 闭环" 的各自贡献
- 判据：原型 vs 对照 A 配对完整请求 predicted_ms；不优即停，不扩六步
- 正确性：全文精确、171/171、自然 stop、accepted 0/1/2、EOG、取消后下一请求

## 风险与回退

- `t_sampled` 是 graph 输出张量，其 buffer 归 scheduler 所有；第二步读取时第一步图
  可能已被 reset。v1 缓解：第一步后立即把 `t_sampled` 设备拷入一个持久 1×I32 slot
  （复用 predefined_hidden store 模式），再作为第二步输入——不 alias 未退休 scratch。
- 契约不满足（p_min>0、多序列、共享 KV、非 greedy chain）：env 未设时零改动，
  现状路径原样保留；实现后任何失败在修改状态前回退。
