# 任务书：消除 DFlash / DSpark 投机解码 GPU-CPU-GPU 折返跑（Zero-Copy Direct Hidden Pipeline）

## 1. 任务背景与核心瓶颈

在当前 `llama.cpp` 及本 fork 架构中，`draft-dflash` 和 `draft-dspark` 投机解码驱动（`common/speculative.cpp`）在每个解码步均存在 **4 次总线跨设备往返（GPU -> Host RAM -> GPU）** 与 CPU 冗余数据重排，严重拖累投机验证与草稿生成的吞吐与延迟：

```
[Target Model Forward (GPU)]
      │
      ▼  【折返 1：隐层跨总线回传】
  src/llama-context.cpp: extract_layer_inputs() 逐层调用 ggml_backend_tensor_get_async 将 VRAM 数据拷至 Host RAM
      │
      ▼  【CPU 瓶颈：Host 数据搬运与交织】
  common/speculative.cpp: 循环 std::memcpy 将多层 target_layer_ids 交织拼装入 features_buf
      │
      ▼  【折返 2：Encoder 输入重新上送】
  llama_encode(ctx_dft, enc_batch) 将 features_buf 从 Host RAM 重新拷贝到 GPU
      │
      ▼
[DFlash Encoder Forward (GPU)]
      │
      ▼  【折返 3：Encoder 输出回传】
  llama_get_embeddings_nextn(ctx_dft) 将生成的隐层从 GPU 读回 Host RAM (inp_g)
      │
      ▼  【折返 4：Decoder KV 注入重新上送】
  llama_decode(ctx_dft, batch_inject) 将 inp_g 作为 batch_inject.embd 重新送回 GPU
      │
      ▼
[DFlash Decoder K/V Update (GPU)]
```

### 优化目标
当 Target 模型与 Draft 模型驻留在同一硬件设备（如同一 Vulkan 物理设备或同一 CUDA/HIP 设备，或 APU 统一物理内存）时，**完全消除上述 4 次跨设备数据拷贝与 CPU 重排**，实现纯 GPU 显存内直通（Zero-Copy / Device-to-Device Copy / In-Graph View）。

---

## 2. 参考实现（现有成熟先例）

本仓库在 Qwen MTP 路径已成功落地了 GPU 内部隐层直通：
- `common/speculative.cpp:1653` (`enable_device_hidden()`)
- `src/llama-predefined-hidden.cpp` (`llama_predefined_decode_hidden`)
- `src/llama-batch.h` (`llama_device_hidden_input` 挂载在 `llama_ubatch::device_hidden`)
- 环境变量：`GGML_TP5_MTP_DEVICE_HIDDEN=1`

DFlash/DSpark 的优化应复用或参照该机制的设计模式。

---

## 3. 详细修改范围与架构设计

### 阶段一：消除折返 3+4（Encoder 输出直通 Decoder KV 注入）

#### 涉及文件
- `src/llama-context.h`, `src/llama-context.cpp`, `src/llama-ext.h`
- `common/speculative.cpp`

#### 机制设计
1. **暴露 Device-Side Tensor 句柄**：
   在 `src/llama-ext.h` 中新增 API：
   ```cpp
   LLAMA_API struct ggml_tensor * llama_get_embeddings_nextn_tensor(struct llama_context * ctx);
   ```
   返回当前 `ctx` 内部计算图生成的 `res->t_h_nextn` 或相关尾部隐藏层 `ggml_tensor *`，而非读取到 CPU 缓冲区的 raw pointer。
2. **Device-to-Device 注入**：
   在 `common/speculative.cpp` 中的 DFlash/DSpark `process()` 循环：
   若 Target 与 Draft 在同一后端设备：
   - 不再调用 `llama_get_embeddings_nextn(ctx_dft)` 进行 host readback。
   - 直接获取 encoder 输出 tensor 的 device buffer 指针/句柄。
   - `batch_inject` 采用 device-tensor view 或调用后端级 `ggml_backend_tensor_copy` 将其拷入 decoder 的对应输入槽位。

---

### 阶段二：消除折返 1+2（Target 多层隐层直接在 GPU 侧交织并送入 Encoder）

#### 涉及文件
- `src/llama-context.cpp`, `src/llama-context.h`
- `common/speculative.cpp`
- 各后端 kernel（Vulkan: `ggml/src/ggml-vulkan/vulkan-shaders/`，CUDA: `ggml/src/ggml-cuda/`）

#### 机制设计
1. **暴露 Target 层输入 Device Tensor**：
   在 `src/llama-ext.h` 增加：
   ```cpp
   LLAMA_API struct ggml_tensor * llama_get_embeddings_layer_inp_tensor(struct llama_context * ctx, uint32_t lid);
   ```
   直接返回 `res->get_layer_inp((int) lid)`（即 Target 在各层开头打桩记录的 `t_layer_inp[lid]`）。
2. **GPU 侧 Gather & Interleave 算子**：
   - 目标：将 5 个（或 N 个）Target 层的隐藏层输出 `[n_embd_tgt, n_tokens]` 交织组合为 `[n_embd_enc, n_tokens]`（其中 `n_embd_enc = N * n_embd_tgt`）。
   - 实现方案 A（图节点连接）：在 Draft Encoder 构建计算图时，若开启直通，直接通过 `ggml_concat` 或 `ggml_view` 将 Target 暴露的层张量组装为 Encoder 的输入节点（两图处于同一 `ggml_context` 或跨 context 共享张量 descriptor）。
   - 实现方案 B（微型 Compute Shader / Kernel）：如果 Target 与 Draft 上下文相互独立，通过一个轻量级 GPU 算子，在 Device 显存内直接做 strided copy/scatter，写入 Draft Encoder 的输入缓冲区。
3. **消除 Host 端 features_buf 分配与 memcpy**：
   - `features_buf`、`llama_get_embeddings_layer_inp()`、`std::memcpy` 全部旁路。
   - 若 Target 隐藏层存在非有限值（NaN/Inf），将原本在 CPU 做的 `std::isfinite` 校验下沉为 GPU 算子或验证为安全后通过 clamp 节点在计算图内抑制。

---

### 阶段三：降级与兼容保护（Fail-Closed & Fallback）

1. **同设备判定**：
   ```cpp
   bool can_direct = (llama_model_get_device(model_tgt, 0) == llama_model_get_device(model_dft, 0)) &&
                     (backend_tgt == backend_dft) &&
                     !is_cpu_backend;
   ```
   若 Target 与 Draft 分布在不同物理卡或 CPU 后端，必须无缝退回原有的 Host 读回与拷贝路径。
2. **开关控制**：
   - 引入环境变量：`GGML_SPEC_DIRECT_HIDDEN`（默认 `1` 开启自动探测，`0` 强制传统 Host 折返跑以便进行 A/B 对拍与回归分析）。

---

## 4. 验收标准与交付物

1. **功能正确性**：
   - 在开启 `GGML_SPEC_DIRECT_HIDDEN=1` 与 `=0` 两种状态下，使用同一个 Target + DSpark Draft（如 `LFM2.5-8B-A1B` + `LFM2.5-8B-A1B-DSpark-Q8_0` 或 `Qwen3.8-27B` + `DSpark`），在 `seed=42, temp=0` 条件下输出完全相同的 token 序列与 logit。
2. **性能与延迟降低**：
   - 在单路解码吞吐（tok/s）上有可观测提升（预期提升 15%~40%，消除总线抖动）。
   - 监控 Host 端 CPU 占用与内存带宽，主线程在 `speculative.cpp:process()` 耗时大幅压低。
3. **代码规范**：
   - 遵循仓库现有规范：禁止无意义重构或额外引入臃肿抽象。
   - 所有新增 API 在 `include/llama.h` 或 `src/llama-ext.h` 中提供清晰的生命周期与空指针契约说明。
   - 编译必须在 Vulkan、CPU 以及 CUDA/ROCm 矩阵下均无警告、无符号未定义报错。
