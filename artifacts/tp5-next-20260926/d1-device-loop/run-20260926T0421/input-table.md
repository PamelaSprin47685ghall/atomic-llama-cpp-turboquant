# D1 两步设备闭环：源码证据输入表

日期：2026-09-26。对象：融合基座 `16e28b8ac`（candidate-k1a worktree 同源）。
依据手册 §8.2 逐行核对源码，非猜测。

## 现状执行流（draft() 每步，`common/speculative.cpp:1796-1960`）

每 draft step i：
1. `llama_decode(ctx_dft, batch)`（fused-default 未设 `GGML_TP5_MTP_DEVICE_HIDDEN`，host 路径）
2. `common_sampler_sample(smpl, ctx_dft, i_last, true)` —— backend greedy chain 已挂接
   （speculative.cpp:1425-1436，`backend_sampling && p_min<=0` → `llama_set_sampler(ctx_dft, greedy)`）；
   `llama_sampler_sample`（llama-sampler.cpp:808）优先读 `llama_get_sampled_token_ith` 的
   **异步回读结果**（llama-context.cpp:3098 `ggml_backend_tensor_get_async`，单 token 4 字节），
   命中则跳过 CPU 采样——所以现状已是"GPU argmax + 4 字节回读"，剩余 host 往返是
   decode 调用本身 + token 写回下一 batch + hidden 行回读。
3. `llama_get_embeddings_nextn_ith` → **hidden 行（hc_dim×F32 ≈ 数 KB）回读 host**，memcpy 进 `batch.embd`
4. 下一 batch = {token(id), pos+1, embd(h_row)} → 回到 1

G1 实测 draft 13.3 ms/cycle（六步），其中每步的高层 decode/sample/readback 往返是 D1 对象。

## 关键源码事实（修正输入表第一版）

| 事实 | 源码 | 对 D1 的含义 |
|---|---|---|
| draft 图已含设备 argmax | llama-model.cpp:2933 `build_sampling()` 对 MTP ctx 同样执行；greedy `backend_apply` 生成 `ggml_argmax`（llama-sampler.cpp:994-1002）；`t_sampled` 被 `set_outputs` 标记输出（llama-graph.cpp:2964-2968） | 第二步的 candidate 已在设备上产出；D1 不需要新采样机制，需要的是**不回 host 的 token 消费** |
| token 回读仅 4 B/步 | llama-context.cpp:3098-3100 `ggml_backend_tensor_get_async(backend, tensor, sampled.data+row, 0, sizeof(token))` | token 回读本身便宜；贵的是 decode() 调用链与 hidden 回读 |
| hidden 回读是主要载荷 | speculative.cpp:1895 `get_embeddings_nextn_ith` + memcpy（hc_dim = hc_mult × n_embd × F32/行） | 两步闭环的核心收益面：第二步 hidden 不回 host |
| 设备 hidden 通道已存在 | `enable_device_hidden()`（speculative.cpp:1462）+ `llm_graph_input_embd_h::set_input` 的 `ubatch->device_hidden` 分支（llama-graph.cpp:127-150，`ggml_backend_device_copy_ranges` 设备到设备拷贝）；历史 device-chain-smoke 正确（2064.35 ms SMOKE_PASS） | hidden 设备通道不需要新写；需要把第二步 hidden 从"回读再上传"改为"RESULT→INPUT 直通" |
| 撤回原型差异 | `withdrawn-mtp-chain-repair.json`：host-hidden unrolled 两步 3488.112 ms vs 普通 n=2 3019.149 ms（`post-chain-withdrawal-smoke.json` draft_n=114 确认 n=2 口径） | 撤回原型把 hidden 回读后重新物化；D1 生死门对照数 = 同 horizon host 往返实现 |

## 输入表（§8.2 逐项，含源码证据）

| 输入/输出 | 每步规则 | 源码证据 | D1 两步版须证明 |
|---|---|---|---|
| token | 第一步 = `dp.id_last`（host 已提交）；第二步 = 第一步设备 argmax 结果 | speculative.cpp:1813、1888；llama-sampler.cpp:808-817（sampled_token 优先）；llama-context.cpp:3098 | 第二步 token 留设备：argmax 输出（I32）直接作为第二步 `inp->tokens` 的行来源（设备侧 index 写入或 graph 内 GET_ROWS 接线）；无效 token 不越界 |
| embedding | `ggml_get_rows(tok_embd_w, inp->tokens)`，权重在 draft 卡 | qwen4exp.cpp:610-612 | index 张量是 graph input（I32，rows=capacity）；设备写 index 即闭环，不复制 embedding 权重 |
| hidden | 第一步 = target RESULT 行（predefined_hidden SEED/RESULT→INPUT）；第二步 = 第一步 `t_h_nextn` | qwen4exp.cpp:744 `res->t_h_nextn`；llama-graph.cpp:127-150 device_hidden 分支；llama-predefined-hidden.cpp ABI | 第二步 hidden 用设备 copy_ranges 从第一步 h_nextn 行进 INPUT；rank/owner/generation 由既有 ABI 保证 |
| position/KV index/mask | pos = `dp.n_past + i + 1`；KV 写 draft memory | speculative.cpp:1927；graph_mtp `build_inp_pos`/`build_attn_inp_kv` | 两步图一次预制两行 metadata；环形 KV/边界走既有 inp_attn 契约 |
| candidate/valid/EOG/cancel | greedy backend chain、p_min<=0、单序列 | speculative.cpp:1425-1436、1888-1910 | 原型只覆盖该契约；第二步 valid=0 不写无效 KV、不让 rank 空等 |
| catch-up | K/V-only（n_outputs==0 → DECODER_MTP_KV） | llama-context.cpp:2482-2487；speculative.cpp:1761 | 不改；全接受仍执行，用 target 真实 hidden |

## 生死门（§8.3，预注册）

1. 原型 = n=2、同权重/context/sampler/放置，第二步 token+hidden 全程不回 host
   （唯一 host 参与 = cycle 边界 accept/commit 与第一步已提交 token）。
2. 对照 = 同 horizon 现状实现（每步 decode→sample 回读→hidden 回读→batch 重建）。
3. 判据 = 两步完整请求 predicted_ms 优于对照，且节省的往返成本 > 新增设备 copy/同步开销。
   不优于对照即停，不扩六步、不切五卡。
4. 正确性 = 全文精确、171/171、自然 stop、accepted 覆盖 0…2、EOG、取消后下一请求。
