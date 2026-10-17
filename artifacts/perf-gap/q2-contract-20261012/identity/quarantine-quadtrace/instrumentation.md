# r12 插桩记录（GAP-1 fork 裁决三实验）

全部改动位于 /var/tmp/tp5-verify-rebuild-src/ggml/src/ggml-vulkan/ggml-vulkan.cpp（隔离树，工作树零接触）。

## 实验 1｜统一四元组打印（GAP1-QUAD）
- 辅助函数 `ggml_vk_gap1_quad`（~11141）：打印 tag/name/seq + ggml_buf(ggml_backend_buffer_t*) + ctx(buffer->context) + dev_buffer(vk_buffer_struct*) + vkhandle(raw VkBuffer)。写侧与 dump 侧走同一函数，字段语义逐字节一致。
- 写侧调用（~22869，set_tensor 的 GAP1-W 块内，BF16 张量前 20 条）：tag="W"，seq=调用计数。
- dump 侧调用（~16976，dump_site_weights 的 dev_id 计算后）：tag="DUMP"，seq=dev_id（rank 序号）。

## 实验 2｜fence 后读回（GAP1-FENCE）
- write_2d 非 HostVisible 分支，waitForFences 之后、resetFences 之前（~11195 块）。
- env GGML_VK_GAP1_READBACK=1 门控，前 200 次写各读回 64 字节（ggml_vk_buffer_read）。
- 时序：fence signaled → 手动 resetFences → read submit+wait（read 内部自管 fence）→ 原有 resetFences 再执行（终态 unsignaled，与原语义一致）。
- 打印 off/buf_size/rb/first8（8 个 u32 hex）。

## 实验 3｜dump 侧张量自检（GAP1-DUMPID）
- dump_site_weights 内（~16983）：打印 tensor 指针、data 指针、view_offs、sub_off、sub_size——与 r7 dispatchtrace 的张量身份对照。

## 前向声明
- `static void ggml_vk_buffer_read(...)` 声明加在 ~11149（write_2d 定义于 read 之前）。
