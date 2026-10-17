# GAP-1 fork fix r10 verdict（更新）

## 判定：权重 registry 修复未生效——W 仍全零。根因在 Vulkan buffer write 的 host→device 可见性层，不在 meta 路由层。

## r10 证据

- 构建：成功（libggml-base `0ca5f460…`）
- W dump：50 文件，5 rank 全零（与 r2–r9 一致）
- 修复内容（全局权重→simple tensor registry）编译进二进制但未改变结果

## 关键代码发现（本轮新证据）

`ggml_vk_buffer_write_2d` 的 host 路径：
```cpp
if (dst->memory_property_flags & eHostVisible) {
    GGML_ASSERT(dst->memory_property_flags & eHostCoherent);
    memcpy((uint8_t *)dst->ptr + offset, src, width * height);
}
```
**HostVisible + HostCoherent 路径直接 memcpy 到 `dst->ptr + offset`**。r8 的 readback（`ggml_backend_tensor_get`）走同一路径看到非零。但 dispatch 的 GPU 读（`vkCmdCopyBuffer` staging）看到零。

## 剩余问题

`ggml_backend_vk_buffer_set_tensor` 写入的 `vk_tensor_offset(tensor)` 与 `ggml_vk_tensor_subbuffer` 读取的 offset 是否一致？两者的 buffer 链解析是否指向同一 `vk_buffer`？

**下一步**：在 `ggml_backend_vk_buffer_set_tensor` 打印 `buf_ctx->dev_buffer` 指针 + `vk_tensor_offset` + `dst->ptr`，与 W dump 的 `subbuffer{buf,off}` 对照——一次运行即可定位写入与读取是否指向同一 VkBuffer/offset。
