# MTP / TP5 进展收回（2026-10-31）

此目录保存本轮候选、补丁、测试材料和历史交接；**不是把这些互不兼容的候选全部启用的可运行配置**。本分支的运行代码仍为 `delivery/tp5-mtp-ready-20261030` 的 ready snapshot。未验收候选不得合入其生产路径或作为性能结果。

## 源码和基线

| 远端分支 | 内容与状态 |
|---|---|
| `delivery/tp5-mtp-ready-20261030` | ready snapshot 源码，历史隔离安装 512×2 为 51.23/52.09 tok/s；本次 runtime 对象复编逐位一致，未做整个干净重建及 GPU 复验。 |
| `archive/mtp-f1c5faf-20261031` | 候选 FA、maxfp、prefab、router、dual-window 的真实基线 `f1c5fafc0449b22969ddabfcc9c361940d94d033`。 |
| `archive/mtp-nout-d6d57375-20261031` | nout 真实基线 `d6d57375e88f4e223ab434eeb742f6a7f1412f83`。 |
| `candidates/fa-truelen-20261031` | 两层 FA 补丁，PARTIAL：CPU mask、attention 和 tiny reader 有局部实测；reader 全场景与设备 true-len 尾块未验。 |
| `candidates/nout-reuse-20261031` | nout 安全增量，PARTIAL：两个 CPU 翻译单元编译，真实多请求复用尚未证。 |
| `candidates/maxfp-safety-20261031` | share+safety，PARTIAL：CPU ggml-base 与 context 单元编译；失败注入、双窗隔离及内存收益未证。 |
| `candidates/prefab-share-20261031` | 080352 唯一谱系，PARTIAL：历史 tiny CPU 运行与源码/二进制溯源未闭合；目标模型和 GPU 未验。 |
| `candidates/router-scratch-20261031` | M2 allocator scratch 候选，PARTIAL：真实模型消费端命中和数值门未证。 |
| `candidates/dualwin-device-wait-20261031` | 085001 路线静态候选，未编译、未设备验证，与 async overlay 不可盲叠。 |

`fa/`、`provider/`、`nout/`、`maxfp/`、`router/`、`prefab/`、`dualwin/`、`union/`、`latebind/` 保留原补丁/报告；补丁对应各自基线，不能按目录顺序混合 apply。`nout/nout_floor1_driver.cpp` 是用户停止派工时收回的未完成驱动，未编译运行。`history/` 是原交接及证据索引，包含历史时间点状态；以本文件和原始记录判读，不把历史 running 视为当前状态。

## 本轮证据与限制

- FA：对真实 stock `f1c5faf` 与候选分别编译 CPU attention，无 runtime 通道的输出逐位相同；候选有效宽度及零值/越界检查在局部 harness 通过。原测试把输出布局和 mask 掩盖的 511 边界判错，收录的是本机修正后的 `fa/verify/host_cpu_fa_off.cpp`。reader 的 tiny 合成权重真实 llama_decode 路径中，views-live、views-clear、cov-off、outputs 和 plain-rebind 场景退出 0；outputs 的拒绝计数是聚合值，不能证明两个分支均独立命中；未对目标模型/完整 stock 和 GPU 证明。禁止未修复并验证前在 GPU 打开 `GGML_TP5_FA_MASK_KVLEN=1`。
- provider：CPU getter fixture 20/20、joiner 62/62、audit 15/15；私有库与 runtime 未修改对照重链逐位复现，HIP DSO 不变。8×2 微诊断及正式首对 OFF_1/ON_1 logits bitwise、五卡上同 PCI 共享时钟流包络有交集；这不是 kernel 并行或吞吐收益证明。重启恢复后 OFF_2 rc0 且与 OFF_1 bitwise；ON_2 未完成，内核报 `0000:03:00.0 device lost from bus`，故四腿 GATE3 未完成。不要重烧同配置，先解决设备稳定性。
- LateBind：历史候选的 UNSET≡ON 表面门已通过；没有独立 pristine stock 对照，不能把绕过 MTP/service 的分流当成 OFF≡stock，也未在交付运行代码中默认开启。
- 联合性能：未取得新的 qualified formal 512×2 加速；历史 row-tail 56.77/58.93 tok/s 属另一候选，并未包含在 ready snapshot 源码交付中。

## 操作约束

12022 只写码、静态审查，不编译或测试；13022 CPU 构建/测试串行；HIP 使用真实双锁且仅 1/1 车道。不得改 selected 安装、不得用 RSZ_RT、不得绕过 nrows==1 别名保护或牺牲精度。所有派工依用户要求已停止。
