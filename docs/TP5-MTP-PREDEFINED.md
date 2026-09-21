# TP5 / MTP：最大容量预定义执行体

## 本批代码的边界

本批是**资源层、执行参数 ABI、MTP 主机状态存储和定义所有权**的实现。
只改源码；没有配置构建目录、编译、执行测试或启动 GPU / 模型。

**尚不是完整的“单张最大 GPU 图”。** 最大形状图构建、其他算子的有效行数下沉
仍未全部闭合。设备侧 hidden handoff 已在后续源码批次接入（第 8 节）；
多行 Q5_K/terminal ADD direct producer 与有效载荷消费见第 9 节。
这些新增路径尚未编译或运行。当前逻辑形状检查
继续保留，不能仅把 `ne == n_tokens` 改成 `<=` 并宣称完成。

为避免把资源冻结误当作完整后端支持，资源预分配入口通过
`GGML_TP5_MTP_MAX_CAPACITY=1` 显式启用，默认不启用。它目前限定单序列
Qwen4EXP TP5 target 与同架构 MTP，拒绝超容量，而不是自动 resize / 缩短 draft。
普通 MTP 的固定主机工作区与 meta 定义所有权修复不依赖该开关。

此文件与其他并行工作者正在修改的 `TP5.md` 分开，以免覆盖协议或性能记录。

## 1. 一份容量，不是一组 shape cache

`ggml/include/ggml-predefined.h` 定义：

- 不变的 `ggml_predefined_limits` / `ggml_predefined_capacity`；
- 64 字节 `ggml_predefined_frame`，只保存有效序列、token、输出行、位置上限、
  draft/accept 数量、epoch 与参数槽；
- 固定的 dispatch 规则，以及 12 字节 indirect dispatch 参数；
- 串行阶段的临时空间取最大值、真实持久状态单独保留的布局规则。

示例：`S=1, D=3, U=32, C=256, O=4`。

| 阶段 | 有效输入 | 有效输出 |
| --- | --- | --- |
| Prompt | `1..32` 行 | 实际请求的输出行，通常最后一行 |
| Target verify | `1 + actual_drafts`，最多 4 行 | 同样数量的 logits |
| MTP step | 1 行 | 1 行 |
| Accepted catch-up | `1 + actual_accepted` 行 | 0 行 logits |

输入容量为 32 不意味着总要生成 32 行 logits。F16/F32 只改变 wire 的字节数，
不改变 F32 hidden / logits 的容量和 accepted-state 语义。

所有大小计算先做溢出、microbatch、输出容量和上下文上限检查。失败不改输出记录。
一次 verification 必须完整放下；不在运行中退回另一组未经定义的形状。

`ggml_predefined_dispatch_arguments()` 生成的是本轮有效工作量。例如
2560 宽、每组覆盖 64 元素：4 行是 160 组，1 行是 40 组，不是容量 32 行的 1280 组。
该函数目前是待 Vulkan 接入的纯规划代码，不会自行插入 GPU 命令。

## 2. 一个共享高水位池，显式冻结

`ggml_gallocr_buffer_pool_set_retain_capacity()` 保留已经预留的最大物理块。
阶段变化仍可清理旧的逻辑分配计划，但不再因小阶段而缩掉物理空间。

`ggml_gallocr_buffer_pool_set_capacity_sealed()` 在定义结束后禁止增长。
申请更大图时，先检查所有 chunk；任何一个越界都在释放、替换原 buffer 之前失败。
不得在失败分支里自动解封再试。

这两个策略由 scheduler 的 compute-pool mutex 串行保护。原有的跨 context
资源使用完成依赖没有删除：共用 scratch 不代表两个计算可以同时往里面写。

`src/llama-predefined.cpp` 的启动入口在 sampler / NextN 输出选项确定后，重新做
target 与 draft 的最大尺寸 reserve，保留高水位，预留输出空间，再封住池并给两个
context 挂上同一份不可变容量记录。reserve 构图用于规划，不执行推理。

**实际复用粒度是相同 `ggml_backend_buffer_type_t`。** 两个不同 GGUF 的 meta
设备可能拥有不同的 buffer type 和模型 userdata。即使底下是同一组 GPU，当前
pool 也不会把不同 type 的物理分配自动合并。因此本批保证“同 type 各阶段取 max”，
不宣称所有 sidecar 组合已经只剩一份物理 GPU scratch。后续合并需让物理存储可共享，
同时保留各模型独立的 split hook / tensor wrapper；不能直接冒用 target 的 meta buft。

## 3. MTP 状态工作区已经接入现有流程

`common/speculative-mtp-workspace.h` 与 `common/speculative.cpp`：

- batch 在初始化时同时拥有 token 和 embedding 数组，不再额外裸 `malloc`，
  也不在 catch-up 时释放重分配；`llama_batch` 只是该 RAII 存储的借用视图；
- verification hidden 以一份最大行数矩阵存储，各序列只保存范围；
- 每序列另外保留跨 batch 的 pending hidden 与 catch-up 第一行的 seed；
- 删除第二份 staged hidden 矩阵；后续 catch-up 行直接索引
  `verified[first + row - 1]`；
- draft 活动标记、defer 标记初始化分配，后续复用；
- 长度变化、部分接受、reset、暂停不缩放这些 owning vectors。

这里**保留**正确的 accepted-state catch-up：token 接受不证明 MTP 自己预测的
hidden/KV 等于 target 验证所得的 hidden/KV。第 0 行仍用上轮 seed，第 N 行仍用
本轮 target 的前一行 hidden。不用错误的“只提交 KV 指针”替代该计算。

默认路径仍使用上述主机桥。第 8 节的显式 device-hidden 路径复用相同范围元数据，
不分配这些主机 hidden 矩阵。它是设备内复制，不称为“零拷贝”。

## 4. 定义属于 owner，不属于进程静态槽

`ggml-backend-meta.cpp` 中的 `s_predefined_*` 函数级 static 单槽已经改成
backend context 自己持有的一个定义。新图重建、资源释放前先使旧定义失效。
没有增加 `map<shape, graph>`。

`llama-context.cpp` 不再把所有单 token 图都赋予同一个
`0x5100000000000000` UID。真实定义获得唯一 48 位父 ID，重用期间保持不变，
实际重建才换 ID。这样 target / MTP 不会因同一个 UID 误用另一个图；meta 的
`uid << 16` 也不会把这个父 ID 整体移掉。reserve 的测量图同样使用独立 ID。

这是正确的生命周期修复，**不等于** token 数变化时不再构图。目前的实际 shape
复用检查仍然有效，热路径可否少建图必须等下一层完整接入后再确认。

## 5. 两个参数槽不是两份图 / 两份工作区

`ggml_predefined_slots_*` 管理两个很小的 frame 槽位。

1. 取得空槽，填好新 epoch 的参数。
2. 每个 rank 真正提交成功后记录其 native timeline 值。
3. 只按实际提交的 rank/value 判断退休；partial submit 不等不存在的值。
4. 已提交槽不可 cancel。旧 epoch 的回执不能释放新 epoch 的槽。
5. 有未退休槽时 `slots_free()` 返回 false，调用者必须连同对应 GPU 参数分配一起保留。

这组 API 不等待、不自旋、不创建 semaphore。接入时每 rank 应使用会话统一的
完成 timeline；不能把不同 communicator 的两个独立 timeline 数值混作同一计数器。
共享 scratch 本身仍受执行阶段所有权约束，两个参数槽不授予并发别名写权限。

## 6. 后端接线清单

下面未完成的部分是启用“单最大 GPU 图”之前的必要条件，不是可跳过的优化项：

1. target / draft / catch-up 固定入口图按容量定义；把有效长度作为输入，
   而不是修改同一张图的 `ne/nb` 后让旧 CB 继续跑。
2. 将 active row、有效 KV 长度、输出行索引下沉到所有相关算子。无效行不得写 KV、
   更新 GDN/recurrent 状态或进入 MoE 工作列表。不能靠零 token 填充。
3. 把静态 dispatch 规则绑定固定参数 arena，预提交时只更新当前未使用的 frame /
   arguments；保留真实 `INDIRECT_COMMAND_READ` 可见性依赖，不逐算子增加 host barrier。
4. 多行 direct producer 与 RELAY 的 CPU reduce / P2 必须共用有效载荷长度，
   不因 verification 的行数大于 1 就退回 P1。
5. 设备 hidden 的源码接线见第 8 节；尚需统一编译、CPU 回归和设备数值验证。
   后续固定最大图仍应沿用这些独立生命周期及 target-conditioned catch-up。
6. 不同 meta buffer type 的物理 scratch 合并必须显式处理模型 wrapper / hook，
   不能只按设备名称或同显卡列表把 buffer 当同一个对象。

## 7. 已写、尚未执行的检查

新增目标：

- `test-predefined-capacity`：最大容量、有效帧、F16/F32、输出行独立、溢出、
  indirect 工作量、两个参数槽、partial submit / stale epoch / 真实退休。
- `test-mtp-workspace`：两序列 seed、target hidden 配对、全拒绝/部分接受、
  长度反复变化、范围重叠拒绝、容量不变、RAII batch 同时持有两个输入。
- `test-alloc` 新增 `test_predefined_maximum_pool`：相同 buft 的两阶段取 max，
  release 不缩、seal 越界失败不替换旧内存、失败后小计划仍可分配。

CMake source / install header 列表已接入。**本批没有运行 CMake、编译器、shader
生成器、上述测试或模型**，最终构建和验证由用户统一安排。

未来完整后端验收应要求：改变 draft/accept 数量不重建图、不重录 CB、不 resize，
有效工作量随有效行数变化，无效行状态写入为零；并分别报告 build、CPU 回归与
GPU 数值/性能结果，不能用静态接口检查冒充这些结果。

## 8. 后续源码批次：设备侧 hidden 有效行交接

开关：`GGML_TP5_MTP_DEVICE_HIDDEN=1`，默认关闭。与
`GGML_TP5_MTP_MAX_CAPACITY` 分开：前者去掉 MTP 热路径的 hidden 主机桥，后者
是最大资源预留。两者都不能冒充尚未完成的全算子最大图执行器。

当前准入是单序列 Qwen4EXP TP5 target + 单头 MTP、相同的逐 rank 物理设备、
target unmasked hidden + draft masked hidden、非 RERoT。不支持时启动失败，
不静默使用 host copy 或跨设备搬运。普通 MTP 默认路径不改变。

### 8.1 持久数据和临时数据分开

`src/llama-predefined-hidden.*` 给 target/draft 各自持有一个持久 hidden buffer：

- target RESULT：按 target batch 最大行数分配，实际只捕获本次输出行；
- draft RESULT：只需一行；catch-up 无 logits，draft step 一行输出；
- CARRY / SEED：各一行，分别保存跨 batch carry 与 catch-up 的第一行输入；
- INPUT：只是固定大小的借用范围表，**没有额外的最大输入矩阵**。

模型图的临时 `t_h_nextn` 在同一生产队列上复制到 RESULT，随后才允许图工作区
被复用。完整 decode 成功才公开 `valid_rows`；失败不会暴露部分捕获为有效结果。
每次 decode 的 RESULT 使用单调 generation；来自 RESULT 的范围必须携带匹配的
generation，旧 acceptance 不能因新一轮恰好同样行数而读错结果。reset 不倒退代数。
尚未完成的 copy 在无 queued token 时也有独立 pending 标记，退出/跨 owner
交接仍须退休，不会因 token 计数清零而提前释放持久 buffer。

### 8.2 所有热路径分支已经接到同一行计划

`common/speculative.cpp` 的 device 分支包含：

1. target 结果到达后，保存旧 CARRY 为 SEED，再选择新 CARRY；
2. acceptance 后从 **target RESULT** 选择 accepted carry；
3. catch-up 的第 0 行来自 SEED，其余行来自 target RESULT 的前一行；
4. draft 第一步使用 CARRY，后续步使用 draft RESULT；
5. reset / checkpoint 的显式 carry 读写保留原有序列化语义。

普通 draft / verify / catch-up 不调用 hidden getter，不经 CPU memcpy hidden，
不向 batch allocator 提供假的 host embedding。主机只保留 token、position、
接受数量与范围元数据。显式 `get_embeddings_nextn*()` 或保存状态仍可按需读回，
不能把这些 API 的显式读取算成热路径意外回退。

### 8.3 有效行数进入输入层，不修改物理张量形状

batch allocator 的 source-row map 是唯一坐标依据。每个 ubatch 的范围由
`llama_predefined_hidden_slice()` 与原 batch 行区间求交，只复制实际行数。
目前限定单序列连续行；乱序、缺失坐标、重叠、越界在记录之前拒绝，不猜偏移。

`llm_graph_input_embd_h::set_input()` 直接把这几个区间复制进实际 `h` 输入。
没有先拼 INPUT 再复制一遍的双重搬运，也没有改 `ne/nb` 欺骗旧 CB。
这完成的是**输入行数下沉**，其他算子仍走现有精确形状图。

### 8.4 Native copy 及同步边界

`ggml-device-copy.*` 是无 host fallback 的 F32 范围复制接口。meta 保留各模型
独立 wrapper/split hook，只有 MIRRORED 张量才按 rank 映射到同一物理设备。
所有 rank 先 preflight，再记录；不在逐 rank 检查失败后退回 CPU。

Vulkan 实现追加到既有 compute stream，一批范围共用前后依赖，不为每行
临时提交、创建描述符或插 HOST barrier。跨 context 仍使用已有的完成依赖，
本批没有宣称消除了 target/MTP 阶段同步，也没有并发别名使用 shared scratch。
失败后的实际已记录工作仍须按原安全路径退休，不能假定 false 表示 GPU 什么也没做。

已扩展但未执行：`test-mtp-workspace` 的无 host hidden 存储、变化长度、有效行
切片、错误映射、溢出用例；`test-alloc` 的 device-only 范围检查及禁止 CPU fallback。
另含 RESULT 的旧代、零代、失败捕获拒绝检查。
本批仍然**没有编译、shader 生成或执行测试**。

## 9. 后续源码批次：多行 producer、有效载荷和固定列块

本批接入真实 Vulkan 记录路径，不只是参数结构：

- Q5_K projection 的 native float 与 MMVQ 两种算术路径，增加同一个固定 4-column
  传输变体，不为 `2/3/4/…` token 分别创建一组传输 pipeline；
- 多行 MoE 的最终 `routed + shared` ADD 原位完成 host payload 写出，无单独 P1；
- 普通 RELAY P2 从 generation 一起发布的长度字段取得有效元素数，只复制有效前缀。

传输仍由原来的 direct-host opt-in 控制，不强开、不拆单行融合区，也不因通信
强制切换 native float/MMVQ。不能把“存在 batched shader”当作全模型均已命中：
当前普通 Q5_K GEMV 选择器上限为 18 列；超过上限仍走原 GEMM/P1 路径。
新的 ADD 只接受两个输入及输出都是连续、相同形状、对齐的 F32 张量，且不与
ADD+RMS partial 融合抢占。未覆盖的类型、布局、算子仍保留原路径。

### 9.1 固定容量与本轮长度

route slot 仍为每 stage 256 字节，不增加 per-shape route table。有效结构扩展为
64 字节，旧 `bank/ready/epoch/flags` 四字布局不变，新增：

```text
byte 16: active_elements
byte 20: capacity_elements
byte 32: dispatch_x, dispatch_y, dispatch_z
```

`active_elements` 来自实际 stage 长度；capacity 排除 host payload bank 尾部的
64-byte status 区域。额外记录的 descriptor capacity 则约束这一算子本身的输入
和输出范围，不能因为整个 bank 更大，就允许 shader 写过 tensor 边界。

CPU reduce 仍按同一有效长度处理。CPU 下行在写 generation 前写
`inbox[3] = active_elements`，不会占用 LateBind 的 `inbox[2]` 独立 Q generation。
普通 P2 命中 generation 后检查 `0 < active_elements <= recorded_capacity`；
非法长度进入原 sticky-error/NaN 失败路径，不写成功 completion。失败时全容量
毒化只是 fail-closed，不会提交无效 token 的 KV/recurrent 结果。

### 9.2 一次定义 dispatch，长度从稳定参数槽读取

`ggml-vulkan-tp5-rows.h` 定义无 Vulkan 依赖的列块/容量/indirect-arguments 规则。
Q5_K 使用 `ceil(active_rows / 4)` 个列块，ADD 使用
`ceil(active_elements / 1024)` 个工作组；两者记录 `dispatchIndirect`，不把
当前有效行数烘焙进 dispatch 命令。矩阵宽度、物理 stride、descriptor range
和列块大小保持定义时的值。

最后不足四列时，两个 Q5_K shader 都在 **任何 B 或 bias 读取之前**判断列是否
有效，subgroup reduction 同样不处理无效列；不用补零 token，不启动空列工作组。
ADD 的尾块也在读写前做有效范围判断。单行现有 GEMV/fused-region pipeline
保持原来的选择方式，未改成最大批量跑一遍。

host-coherent route/indirect 参数在所属提交之前一次写好。新 recorder 复用普通
pipeline 的 descriptor/push-constant 记录，不使用原 GPU-generated indirect
helper 的逐算子 updateBuffer 和 transfer barrier。真正由 GPU 生成的 indirect
参数仍走原有依赖路径，本批没有删除它们的内存依赖。

### 9.3 重放和生命周期

每个已记录 producer 保存自身的 row-program recipe（容量、宽度、行块）。旧图
重放时恢复该 recipe，再生成该 stage 的 arguments，不能沿用刚刚运行的另一张
图的 tile/容量。定义切换并不增加一个 shape→graph map。

新图第一次记录时，在记录 dispatch 前准备参数；记录结束发布的是 CPU 元数据，
**不会**在它已经可能提交后再次写相同参数。frame/route 的复用仍受原有提交与
producer-completion 规则约束，不允许在消费者仍读取时改 slot。

该工作解决了传输算子自己的动态有效范围，**尚未使整个 ggml 图固定为最大形状**。
前端仍按精确逻辑形状构建，Attention、GDN/recurrent、MoE packing 和其他计算
算子的容量/有效范围还需要逐项接通。因此不能据此宣称 `1→4→2` 已零建图。

已登记但未编译/执行 `test-tp5-row-program`：固定定义的长度反复变化、四列尾块、
descriptor 容量拒绝、工作组上限、整数溢出、bank 末尾 status 隔离、ADD 尾块保护。
这些是纯 CPU 规则用例，不代替两种 wire、两种 Q5_K 算术与真实 MTP 的数值验收。
