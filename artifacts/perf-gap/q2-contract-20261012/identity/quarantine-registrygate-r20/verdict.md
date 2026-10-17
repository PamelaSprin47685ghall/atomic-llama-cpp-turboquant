# r20 registry 门控崩溃复现 verdict（identity/quarantine-registrygate-r20/）

bin：libggml-base 246012b1…（meta.cpp 所在库——本轮变化载体；libggml-vulkan 9b8890af 不变）。
CPU：__main__ runner 39 项 OK + gap-tools 46 项 OK。**测试配套缺陷两处记录在案**（交 Engineer）：(1) RegistryGateTests 6 项未挂进 __main__ 自定义 runner（仅 unittest 发现可跑）；(2) 红样本 test_red_sample_env_name_removed 失败——判别函数 _registry_gate_property_holds 对 env 名重命名 mutation 不敏感（mutation 后仍返回 True）——判别函数缺陷，不影响门控行为本身（正样本全绿、门控代码在树并编译进库）。

## 门控实验（arms/registrygate-off-r20/）——裁决 (a) registry 因果坐实

与 r18b 完全同配置 + 唯一新增 `GGML_VK_META_REGISTRY_DISABLE=1`：

| 观察 | r18b（门控关） | r20（门控开） |
|---|---|---|
| [meta-registry-bypass] 触发 | 无 | **1 次**（"registration and lookup bypassed (registry-free path)"） |
| ggml-backend.cpp:349 断言 | 1（崩溃） | **0（崩溃消失）** |
| 进程末态 | 断言即末行 | **正常 "cleaning up before exit"** |
| 请求 | 从未发出（全 NOT_COLLECTED） | **发出并完成**（乱码 500：unparsed peg 2 次、正文 hash 20c0ddaa…≠c2d7bdd3） |
| CRITICAL-FALLBACK | 5 | 10 |

**因果链完美闭合（三方对照）**：
- r9 树（registry 引入前）：乱码 500、不崩、正常退出。
- r10-r18 树（registry 活跃）：崩溃（r10-r15 乱码 500 + 崩溃共存；r16+ 崩溃且请求未发出）。
- r20 树（registry 门控旁路 = 等价 r9 该部分行为）：乱码 500、不崩、正常退出——**精确回到 r9 形态**。

## 裁决

**(a) 坐实：s_gap1_weight_simple_registry（r10 引入的 fallback 路径注册指针优先返回）是崩溃根因。** 机理方向：fallback（480 子图）路径的 ggml_backend_meta_buffer_simple_tensor 经 registry 优先返回注册指针，router 路径随后对该指针执行越界 tensor write（ggml-backend.cpp:349 断言）——registry 旁路后该路径回到 r9 行为，断言不再触发。

**不实施修复（按 charge 指示）**：registry 是隔离诊断树插桩（仓库工作树无此代码），其移除/修复交 Engineer。注意：registry 当初是为诊断「W 全零」引入的，而 r15 已证 W 全零是 dump deferred 伪影——**registry 的存在理由已被抽空，修复方向大概率是直接移除**（连同 r10/r11 的 forkfix 相关插桩一起清理）。

## 乱码线现状（不受本轮影响，但路径已通）

乱码（数值缺陷候选）在 registry 旁路下维持 r9 形态（500 + 进程活着）。崩溃线关闭后，**数值对拍路线的最大障碍（进程崩溃）已解除**：下一轮可在 registry 旁路（或移除 registry 的树）上重做 fused vs 参照的单 site 数值对拍——用 r16 的可信 dump（读数有效）+ 修好 dump 的图执行破坏问题（崩溃曾是 dump 破坏图的说法需重审——崩溃根因是 registry，dump 新版可能无辜）。

## 下一步（交 Manager）

1. Engineer：移除/修复 s_gap1_weight_simple_registry（隔离树清理——连同存在理由已失的 r10/r11 插桩）。
2. Engineer：修两处测试配套缺陷（RegistryGateTests 挂进 __main__ runner；判别函数 mutation 敏感性）。
3. 重启数值对拍（乱码根因定位——唯一存留候选）。
