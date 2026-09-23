# ctest 全量运行导致本机重启——排查记录（2026-09-22 23:24）

**结论：不是本仓库改动引入的问题。** `ctest` 全量轮次中 `test-llama-archs` 在 `0000:0f:00.0`
（Vulkan0 / card1）上触发 GPUVM page fault → ring 超时 → GPU mode1 reset，重启与该 reset 同时发生。
全量轮次里被 GPU 击落的只有 `test-llama-archs` 一个用例；本班次的 B00/B02 改动（RELAY 热路径
不再 patch 旧 batch、recipe 拆分）**都不在这条路径上**，且已通过的定向回归见下。

## 一、现场证据（boot 0 = 当前开机，故障 23:24:01）

```
amdgpu 0000:0f:00.0: [gfxhub] page fault (src_id:0 ring:40 vmid:6 pasid:6970)
amdgpu 0000:0f:00.0:   in page starting at address 0x000080010034c000 from client 0x1b (UTCL2)
amdgpu 0000:0f:00.0:  Process test-llama-arch pid 8904 thread test-llama-arch pid 8904   (x12)
[drm:gfx_v10_0_bad_op_irq [amdgpu]] *ERROR* Illegal opcode in command stream
amdgpu 0000:0f:00.0: ring comp_1.2.0 timeout, signaled seq=1700, emitted seq=1701
amdgpu 0000:0f:00.0: Starting comp_1.2.0 ring reset
amdgpu 0000:0f:00.0: Ring comp_1.2.0 reset failed
amdgpu 0000:0f:00.0: fail to wait on hqd deactivate
amdgpu 0000:0f:00.0: GPU reset begin!. Source: 1
amdgpu 0000:0f:00.0: MODE1 reset / GPU mode1 reset / GPU smu mode1 reset
amdgpu 0000:0f:00.0: GPU reset succeeded, trying to resume
amdgpu 0000:0f:00.0: VRAM is lost due to GPU reset!
amdgpu 0000:0f:00.0: [drm] device wedged, but recovered through reset
amdgpu 0000:0f:00.0: [drm] *ERROR* Failed to initialize parser -125!
```

同一故障形态在 **上一轮班次（boot -2，09-22 17:35 / 17:38）也由 `test-llama-archs` 触发过**
（卡 `0000:03:00.0`，GCVM_L2_PROTECTION_FAULT / CPG 0x6，且 17:38:43 同进程还 segfault 在
`libggml-vulkan.so.0.18.1`）。即：这是 `test-llama-archs` 的**既有已知问题**，不是本轮新增回归。

## 二、重启性质判定

- boot -2 的用户态日志在 17:42:16 仍有正常 mihomo/systemd 条目，17:42:51 即进入 kdump 内核
  （command line 含 `elfcorehdr` + `systemd.unit=kdump-tools-dump.service`）→ **硬崩溃/panic，
  不是软重启或计划内 shutdown**（无 "Shutting down" 序列、无 acpi power button 事件）。
- kdump 已进入并收集 vmcore：`/var/crash/` 下按日期留有 dump 目录。
- 当前 boot 0 的同一故障 23:24 只造成单卡 reset + 恢复，未再次整机重启。

## 三、未决事项（按优先级）

1. **`test-llama-archs` 必须从 GPU 可达的默认 ctest 轮次中隔离**：它反复在同一张卡上把
   GPUVM 打穿并拖垮整机。建议加 `RESOURCE_LOCK` / 单独 label，或用 `-E test-llama-arch`
   排除后再跑全量（GPU 专项回归走 `tests/test-vulkan-tp5-mesh.cpp` 等定向目标）。
   暂不在本次排查内擅自改测试注册，等你确认策略。
2. `00:0f:00.0` = card1 = Vulkan0。全量 ctest 与 5 卡 A/B 都用 Vulkan0 起算；在
   card2 仍锁 1200MHz 的前提下，任何同机 A/B 都要先登记该偏斜。
3. 若需深挖 page fault 的 shader 责任方：boot 0 有 devcoredump
   （`/sys/class/drm/card1/device/devcoredump/data`，已被 reset 消费后清空，需复现时立即取），
   `amdgpu` 的 `GCVM_L2_PROTECTION_FAULT_STATUS=0x00540C51 / PERMISSION_FAULTS=0x5`
   指向**越权访问**而非单纯越界，需要对照 `test-llama-archs` 生成模型的 shader。

## 四、本班次已验证未受影响的部分（重启前实跑）

- `test-vulkan-tp5-mesh --sync relay --rounds 8 --check-all`：all passed
- `--sync {timeline,star,gpuflag} --rounds 4`：all passed
- `--sync {relay,timeline} --{vary-input,delay-producer} --rounds 4`：all passed
- `ctest -R "rerot|xkv|flashprefill"`：50/50
- `ctest -R "test-tp5|test-mtp|test-predefined|test-target-capacity"`：6/6
- `cmake --build build-tp5 --target llama-server`：零 error
