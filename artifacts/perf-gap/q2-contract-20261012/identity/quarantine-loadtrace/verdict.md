# GAP-1 修复闭环第二步：loadtrace 运行时判别 verdict（r5，最终）

## 判定：假设 (iii) 身份/布局分叉——写入执行到正确的 simple tensor，但 dispatch 侧 A 绑定指向**不同的 VkBuffer**

## 证据链（r5 运行，loadtrace 2688 行 + 50 dump）

### 1. MIRRORED fan-out 完全正常工作（假设 (i) 与 (ii) 均排除）

- **384 次 MIRRORED set_tensor 调用**——每个 hc_ 权重张量（48 层 × 8 张量 = 384）都被调用，包括 `blk.0.hc_ffn_down.weight`：
  ```
  [loadtrace] MIRRORED set_tensor name=blk.0.hc_ffn_down.weight axis=10 n_bufs=5 offset=0 size=6553600
  ```
- **零 SIMPLE_TENSOR_NULL**（假设 (ii) 排除）——5 个 rank 的 simple tensor 全部非空且有效：
  ```
  rank0: simple{buf=0x570564428160 data=0x6c8f000} set(data,0,6553600)
  rank1: simple{buf=0x57056442a2a0 data=0x6c8f000} set(data,0,6553600)
  rank2: simple{buf=0x57056442c4c0 ...}
  rank3: simple{buf=0x57056442e700 ...}
  rank4: simple{buf=0x570564430940 ...}
  ```
- 写入尺寸 6553600 = 完整 BF16 权重字节数（320×10240×2）✓

### 2. 但 dispatch 侧 A 绑定指向不同的 buffer（假设 (iii) 坐实）

dispatch 时 W_down 的 subbuffer（hc-dump weightsA）：
```
subbuffer{buf=0x5705644297d0, off=1049041280, size=6553600}
```

**三方对照**：
| | buffer 句柄 | 位置 |
|---|---|---|
| 加载期写入目标（rank0 simple tensor） | `0x570564428160` | data=0x6c8f000（~108MB 区域） |
| dispatch 读取绑定（W_down） | `0x5705644297d0` | off=1049041280（~1000MB 偏移） |

**`0x5705644297d0` 不在加载期 5 个 rank buffer 句柄集合中**（`0x570564428160/a2a0/c4c0/e700/30940`）。写入和读取发生在**两个不同的 VkBuffer 对象**上。

### 3. r4 的插桩笔误（如实记录）

r4 的 loadtrace 零行——三处插桩的过滤条件写成 `strncmp(name, "hc_", 3)`，但真实张量名是 `blk.N.hc_...`（llama-arch.cpp 注册为 `"blk.%d.hc_ffn_down"` 等）。r4 的「无 set_tensor 记录」是插笔误不是加载链缺陷。r5 修正为 `strstr(name, ".hc_")` 后 2688 行正常输出。

## 结论

**权重上传链（MIRRORED fan-out）本身工作正常。** 缺陷在**身份/布局分叉**：加载期 simple tensor 与 dispatch 期 A 绑定指向不同的 VkBuffer。候选机理（供 Engineer）：

1. **stc_static vs stc_compute 路由**：meta buffer 有三个 simple tensor 容器（stc_static / stc_compute_0 / stc_compute_1）。权重加载写入的是 stc_static 的 buffer，但 dispatch 侧的图张量可能解析到了 stc_compute 的 buffer——两者是不同分配。
2. **tallocr/ggml-alloc 布局分叉**：图构建期的 tensor 分配器给 W_down 分配了新的 buffer（~1GB 偏移处），而非复用加载期已写入的 buffer。

**修复方向**：确保 dispatch 侧的 A 绑定解析到 stc_static 中已写入的 simple tensor（或把 fan-out 写入目标改为 dispatch 绑定的那个 buffer）。
