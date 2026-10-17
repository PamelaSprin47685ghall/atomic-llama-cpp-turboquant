# r22 fused vs 入口还原参照数值对拍 verdict（identity/quarantine-parity-r22/）

bin：fused 臂 9b8890af…（树默认状态）、参照臂 2a525504…（+入口还原临时改动）；对拍后树已恢复 fused 状态（重编哈希回到 9b8890af，备份 diff 验证）。
两臂配置完全一致（--warmup 0 --repeats 1 --max-tokens 32、GGML_VK_CMD_REPLAY=0、GGML_VK_HC_DUMP_SITES=1）。

## 两臂运行状态

- **fused 臂**（arms/parity-fused-r22/）：不崩（断言 0——**dump 无辜性正式确认**：registry 移除后 dump 活跃下进程健康，r16「dump 破坏图执行」归因确系 registry 误判）、正常退出（teardown returncode 0——首次非 SIGABRT）、dump 采集成功（25 个文件）、SYNCCMP 交叉核对 10 行非零、dump 读数与 r16 逐字节一致（normalized 0.232353…/lo 0.132684…/mixed 0.203644…——数值稳定可复现）。
- **参照臂**（arms/parity-refentry-r22/，入口还原）：不崩、正常退出、**dumps=0**（预期——入口还原后 fused 段不激活，dump 挂 fused 路径采不到）、请求完成。

## 对拍结果——乱码的逐字分歧首次定位

| 臂 | finish | 32 token 输出 |
|---|---|---|
| fused（fused 段激活） | length | **「健康做，，，，，，，，且，，，在rata，让我，rata，FP，，rata，，rata，且，，」**（乱码） |
| 参照（入口还原，fused 段不激活） | length | **「健康减重并非靠极端节食，而是建立可持续的生活习惯。在饮食方面，优先选择高纤维的蔬菜、优质蛋白和全谷物，」**（正常中文——c2d7bdd3 的已知开头形态） |

**首个分歧 token**：前 2 个 token「健康」逐字一致，**第 3 个 token 分歧**（fused=「做」 vs 参照=「减」）。

**间接对拍与分歧环节推断（标注推断边界）**：参照臂采不到 dump（fused 段不激活），逐张量对拍不可行——分歧定位是间接的（正文分歧位置 + 两臂唯一差异=fused HC 段激活）。推断链：(1) 第 3 token 依赖前 2 token 的 KV/HC recurrent 状态——分歧起点在 prefill 尾部至第 2 个 decode 步之间的 HC 段计算；(2) fused 臂 dump 真值显示 lo/mixed 非零且量级「合理」——**量级合理 ≠ 数值正确**（浮点错误不改变量级）；(3) 唯一激活差异是 fused HC 段的 BF16 变体（qwen4_hc_down_silu_bf16 / mul_mat_vec BF16 路径）——**lo/mixed 的具体数值错误（量级对值错）是分歧的最强解释**。此为推断而非逐值证明——逐值证明需要「参照臂 fused 段等价张量」的采集设计（dump 挂 fused 路径的结构性限制）。

## 裁决

1. **乱码根因的运行时定位**：fused HC segment 的 BF16 变体路径（第 3 token 起数值分歧——与 r10 起的乱码形态、3a 轮「入口还原=数值正确」的历史证据全部自洽）。
2. **dump 无辜性正式确认**（§20.5 修正）：dump 同步读活跃下进程健康、读数可信（SYNCCMP 交叉核对 + 跨轮逐字节稳定）。
3. **kernel 级定位交 Engineer 静态审查**：分歧环节收窄到 fused 段的 down（lo 产出）或 up/gate（mixed 产出）收缩计算——qwen4_hc_down_silu_bf16 / mul_mat_vec.comp 的 BF16 路径（iter_aligned_nonquant 位展开、vec4 B 读取、HC_UP_FOLD 折叠——upfold 修复后仍乱，嫌疑重心在 down 端或折叠之外的 BF16 解码细节）。

## 树状态恢复确认

对拍后入口还原已从备份恢复（重编哈希 9b8890af = fused 开放状态，与对拍前一致）。

## 下一步（交 Manager）

Engineer 静态审查 fused 段 BF16 变体的收缩计算（重点：down 端 lo 产出与 mixed 产出的数值正确性——量级已证合理，需逐值验证；可设计 CPU 参照计算 vs dump 真值的逐元素对拍——dump 真值已可信且稳定）。
