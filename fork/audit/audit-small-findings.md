# 小功能审计：最终静态审计与 corpus 兼容补丁

> 整合说明：这是固定旧提交的分册审阅记录。原 coverage 文件中的逐路径、逐 hunk 和函数范围已并入 [files.json](files.json)，保留 `source_stage`；下文原始 diff / checkpoint / patch / 生成器名称是审阅资料标识，不是已迁入功能或当前构建入口。后续补审见同目录其他分册；最终实现状态以 [MIGRATION.md](../MIGRATION.md) 和 [VALIDATION.md](../VALIDATION.md) 为准。所有旧侧 / 新侧行号都指旧 fork 与固定 A 的差异，不是 clean 分支行号；记录明确标注 `audit-extra/clean` 的补充片段例外。

审计基准：旧 fork `2777543b2358c88cd1c2000f7fa342b764a0e213` 对上游 `cda0f9256665845cd9d6f3a0176880883f653a3d`。证据是本地 `audit-small.diff` 的新侧行号（即旧 fork 源码行号），以及 root 拉取的 `audit-extra/` 源码。权威 `audit-inventory.json` 有 **4,634 个唯一变更路径**；本分工的 diff 覆盖 **79 个文件，所有给定 hunk 已读完**。`audit-small-coverage.json` 为全部 4,634 个路径提供记录，对本分工外的 4,555 个路径明确标记不作处置结论，并附每个已读文件的行为、函数/行号、依赖、hunk 上下文和新增行区间。`audit-old-numstat.txt` 只有 598 个有效路径，是早先的子集，不能用它表示全仓库覆盖；其末行工具退出提示也不是路径。

本轮没有重新运行模型或测性能。唯一实现变更是独立 B corpus 的兼容补丁，已在本机通过 C++17 严格编译和格式/所有权测试，真实旧生产 writer/reader 互操作程序交由 root 在已安装 A 的环境运行。下述“保留”区分已实现修正、应单独保留的小功能、需要验证的候选，不表示所有候选已重新移植。

## 应当明确保留的细功能

| ID | 行为、证据 | 最小 A/B/C 边界 | 必要验证与限制 |
|---|---|---|---|
| S01 | canonical corpus header 的末 8 字节是 global start_id，节点首片从 1 开始，请求从 0 开始；请求携带 kind 和 timestamp_ms。`wanxiangqi/docs/request-corpus-dump.md:22–46`；`wanxiangqi/server/server-corpus-dump.cpp:75–104,360–389`。第一轮 B2 忽略 start_id 并将元数据写零，不能完整兼容原生产格式。 | 已实现独立 B 补丁：验证并写出 nodes=1/requests=0，owned metadata，默认 append 为旧 completion=1、timestamp=0 表示未记录；支持空请求；多片时明确拒绝后续 shard。 | 完整生产字节 fixture 已过；原 server loader 真代码验证程序已交 root。0 start_id 的早期 B 快照明确拒绝并提示 rebuild。词表大小相同不能证明 tokenizer 相同。 |
| S02 | BailingMoE3 接受 `ssm_f_a/ssm_g_a`，回退旧 `ssm_f/ssm_g`；metadata 存在但 nextn.eh_proj 缺失时恢复全部 trunk 层。`src/models/bailingmoe3.cpp:24–33,96–104`。这两项是模型兼容性，不该随 MTP 整体归档。 | 优先 C 离线 GGUF metadata/名称规范化；无法无损规范化时各自独立小 A seam。 | 新旧 tensor 命名 fixture；带/不带真 MTP 权重；层数与 tensor 消费完整性。不要只改 n_layer 而丢掉真实 MTP 尾部。 |
| S03 | model saver 支持 uint64 数组，避免 PLE hash 参数被截断。`src/llama-model-saver.cpp:add_kv<Container>,116–117`。 | 约 2 行小 A seam 或使用上游 gguf-py 的独立 C 导出。 | 用大于 2^32 的值往返，并验证 GGUF_TYPE_UINT64。必须是新能力的最小变更，不复制整个 saver 的重复 metadata/tensor 插入。 |
| S04 | `calib-to-imatrix.py:40–54,66–90` 将 WXQCAL03 的 exp_diag/exp_n 导出 gate/up `.in_sum2`/`.counts`，hid_sum/hid_n 导出 down；未见专家由活跃专家的全局 sums/counts 回填，全部未见则报错。 | 独立 C Python 转换器 + B WXQCAL03 parser；依赖 numpy 和上游 gguf-py，不依赖推理私有 API。 | 原 `load:30–37` 只检查 flags，未拒绝 PARTIAL；迁移应要求显式完整/部分校准策略、检查 shape、finite、计数、截断，保留完整性与上下文重置元信息。可明确保留，但此轮不新增大型实现。 |
| S05 | MoE 原始校准主体可以不改 A：`trie-calib.cpp:152–339` 通过公开 GGML_OP_MUL_MAT_ID callback 观察 gate/down 输入与路由；`904–905` 安装 params.cb_eval；`492–505` 用 public seq state get/set 做 DFS。 | B 统计与格式 + C 用 public eval callback/seq snapshots 编排；与依赖私有 pre-RoPE 的 TriAttention 工具分开。 | 真实 A 模型 callback 与按请求重放对拍、分支顺序、样本计数。callback tensor 名与布局要验证，遇到不支持模型不可静默空报告。原 `max_nodes:416–421` 未裁剪最后一个 batch，`complete:1072` 仅表示所选遍历未报错，不能当作 full-corpus 证明。 |
| S06 | 数值 token repetition recurrence 独立；`repetition-guard.cpp` 以指数遗忘的不同 token 数做检测，生成 CLI 的 `wxq-gen.cpp:92–110` 另有 8-gram repeat/distinct-token 指标。 | B token-ID 统计 + C 可选遥测或显式 abort policy。 | 不直接迁移原强制 guard：其 `observe(text):78–89` 分别 tokenize 每个流式 chunk，可能随 chunk 划分变化；checked-in 95% envelope 来自代码语料，不是所有语言任务的通用正确阈值。 |
| S07 | `llama-model.cpp:2281–2302` 对源 GGUF 分片内容作缓存 SHA256，供 state/manifest 绑定模型身份；与压缩 KV 算法没有必然联系。 | 独立 B 文件指纹 + C manifest，避免依赖 XKV 模块才能计算身份。 | 固定分片顺序与长度/边界编码；变更文件后缓存必须失效或声明文件不可变；tokenizer 身份单独处理。具体 hash 核心由别的审计分工覆盖。 |
| S08 | PLE embedding 默认放 CPU，显式 overrides 优先：`llama-model-loader.cpp:1221–1242`。 | C 通过现有公开 tensor buffer override 显式配置，避免 A 全局默认变动。 | 选定真实 PLE 模型证明参数命中、内存减少以及吞吐影响；没有实测不能声称性能收益。 |

## 应当保留为候选，而非未经验证的已完成功能

* **模型 shape/选择逻辑小 seam（S13、S21）**：Qwen4EXP 全 cache 已选时跳过 top-k 排序（`981–984,1028–1032`），且仍走相同 sparse-attention 输出；Vulkan 大 top-k 换为 argsort-top-k（`1151–1153`）。需要不带私有增量 cache 的小差分、边界值/回滚的 logits 对拍与性能证据。PLE metadata 64→32 位范围检查（`133–147`）值得保留为 malformed-model 防护。NextN metadata 不匹配 reset（`30–39`）与 Bailing 同类。新增 PLE 非零/head 数量检查的一部分与上游已有检查重复（`106–122`），不要重复搬入。HC norm 的新增代码已经证实为重复乘法，详见下节。
* **独立读取权重 slab**：`wanxiangqi/lib/gguf-slab.cpp` 使用公开 gguf/ggml metadata 与 type traits，无需运行模型；可服务专家统计 C。需补 expert index、字节偏移溢出、短读和重复 open 的资源所有权检查。通用 linalg/parallel 不因存在就整包搬运，应由实际工具消费。
* **懒行读取器**：`src/llama-lazy-reader.h` 可以分离成普通文件行读取 B；重新打开 buffered fd 避免继承 O_DIRECT（`llama-model.cpp:2192–2247`）是有用细节。但 root 全引用检索已确认 `lazy.add` 仅 `test-model-lazy-mmap.cpp:49` 调用，生产模式未真正注册 tensor。不能把测试内人工注册当作旧 fork 已落地的生产懒加载。
* **生产 corpus 录制**：原 server writer 还保留 append/restart、tail recovery、分片、重复请求、节点先于请求 flush、互斥及失败锁存等语义。独立 B snapshot 与 C scorer只解决读/写/遍历子集；HTTP JSON proxy 不能代替 chat template/tokenizer 之后的真实 token corpus。以后若保留录制，应以真实 C tokenization 或 narrow server post-tokenization seam 接入。

## 明确不能原样搬入 clean fork 的内容

* **grammar indent 没接到生成链**：root 全引用检索只找到 tests 调用 `llama_grammar_indent_lex/finish`。补充源码的 `llama_grammar_apply_impl` 和 `llama_grammar_accept_token` 继续直接 decode UTF-8，未消费 indent events。lexer 的增量状态与 clone 测试可作为实验模块保留，不能宣称约束生成支持 indentation。
* **流式保存并非此 diff 新增**：完整 `llama-model-saver.cpp:524–532` 仅调用上游 `gguf_write_to_file`/`gguf_write_to_file_ptr`。root 补查 gguf.cpp 的 +5 行只是 type 42→Q2_0 兼容映射，没有新增 streaming save。`conversion/qwen4exp.py` 的 LazyChunkedTensor 注释也不是新实现。
* **实际 merge 损坏（S10）**：`src/llama-hparams.cpp:211–212` 的 `return n_conv` 被并进注释；`src/llama-mmap.cpp:482` 的 MAP_POPULATE 分支被注释吞没；`src/llama-model.cpp:637` 的 return 也被吞没；`src/models/qwen4exp.cpp:201` 的 tensor 创建被吞没。`conversion/qwen4exp.py:4,18,289` 的 import/decorator/class 拼接造成 Python 语法错误。还存在重复 metadata/tensor 插入和重复 madvise。这些是剔除旧冲突损伤的证据，不是值得保留的新默认行为。
* **研究脚本不等于产品能力**：`fit-test.py:55–58` 使用未 import 的 struct，硬编码模型/校准路径与参数，并训练一个未部署的 student；`make-transforms.py` 是低秩/旋转 ablation，不等于可用压缩模型；TriAttention 采集依赖 `llama-ext.h`/私有 model->hparams/专用 pre-RoPE outputs，与普通 public callback 校准需要分开决策。

## 长 hunk 补读后的数学缺陷

以下均直接来自同一旧 fork 文件 `src/models/qwen4exp.cpp`，不是对算法名字的推测：

| 原函数与行号 | 实际代码行为 | 对迁移的影响 |
|---|---|---|
| `graph_mtp:665–673` | 新增 `h_norm * layer.nextn.hnorm`，随后保留上游原有的同一 gamma 乘法。 | gamma 被乘两遍；第一次还绕过后面既有的 dtype cast。不能当作 grouped RMSNorm 修复保留。 |
| `graph_mtp:713–715,754–756` | K 在新提到前面的块中已做 RoPE，普通 MTP 路径之后又执行保留的 RoPE。 | 普通 MTP 的 K 旋转两遍；KV-only 分支在 730 行返回所以不经过第二遍。两条路径的数学语义已不一致。 |
| `graph_mtp:641,644–650` | 先 `ggml_get_rows(tok_embd_w)`，之后才检查指针为空并尝试使用另一模型的 embedding。 | 原上游先选择合法权重、后 gather 的安全顺序被倒置。 |
| `graph_mtp:772–778,805–808` | 新增输出行 gather，而函数末尾保留原本的 gather。 | 部分输出场景会把原 row id 再作用于已缩减的 tensor，并提前改变 `t_h_nextn` 的行集合；需要按原语义重建，不能原样提取。 |
| `graph_mtp:668,674`；`build_layer_attn_linear:1396–1397` | capacity 版 reshape 与旧 active-token 版 reshape 同时保留。 | active 与 capacity 不等时不能满足同一元素数约束；不是已验证的动态图优化。 |
| `build_layer_attn_linear:1451` | V 的起始偏移由 `2 * head_k_dim * num_k_heads` 改成 `head_v_dim * num_v_heads`。 | Q/K/V 顺序下起点应由前置 Q+K 的宽度决定；只有两种宽度恰好相等时改动才等价。不能无模型几何证明便将它称为数学修复。 |
| `llm_graph_input_ple::set_input:1601–1624` | 宣称 fast path，但每批第 0 个 token 都缺少同批前驱，令 `need_fallback=true`，随后对整批调用 `get_prev_tokens`。 | 对合法 n-gram≥2、非空 batch，fallback 必触发；现有代码无法证明减少全 cache scan。局部填充也没有原上游的 seq-id 相等检查。 |

此外，`llama-model.cpp:2036–2044` 的 `GGML_SKIP_WEIGHT_LOAD` 整体跳过权重加载，仅适合明确标注的图构建实验。它不能成为正常 C 启动路径或性能数据来源。`delta-net-base.cpp:937–945` 的 “StateCarryFix” 主要是注释/计算，现有 cpy 本身未形成一项独立行为修复；应保留真实边界测试，避免按注释名误记为已恢复功能。

## S04：imatrix 小转换器的格式和不完整数据策略

WXQCAL03 文件具有可迁移价值，但需要按实际文件格式对待。`wanxiangqi/lib/calib-dump.h` 与 `probe/wxqcal.py:63–88` 一致描述：magic 8 字节；6 个 u32（层数量、max_ctx、act_stride、act_tokens、route_stride、flags）；3 个 u64（n_decoded、n_total、n_ctx_resets）；complete u8 与 7 字节 padding。每层有 4 个 u32 几何字段和 3 个 u64 计数字段，随后按 flags 放 counts、activation sums、per-expert first/second moments、hidden sums，以及 covariance。头中的整数约定为 little endian；当前 Python 数组读取和 C++ raw read 使用宿主字节序，需要显式固定为 `<u8`/`<f8` 和 LE 读取。

原转换器的实际映射如下：

| 输入 | GGUF tensor | Python 数组形状 |
|---|---|---|
| `exp_diag`，每专家输入平方和 | `blk.N.ffn_gate_exps.weight.in_sum2` 和 `...ffn_up_exps.weight.in_sum2` | `[n_expert,n_embd]`，float32 |
| `exp_n` | 上述两个权重名各自的 `.counts` | `[n_expert,1]`，float32 |
| `hid_sum`，每专家隐藏神经元平方和 | `blk.N.ffn_down_exps.weight.in_sum2` | `[n_expert,n_ff]`，float32 |
| `hid_n` | down 的 `.counts` | `[n_expert,1]`，float32 |

`calib-to-imatrix.py:66–70` 写 `general.type=imatrix`、`imatrix.datasets`、`imatrix.chunk_count=n_decoded`、`chunk_size=1`；`40–54` 让未见专家使用活跃专家的总 sums / 总 counts，计数置 1。它保留 raw sums，目标消费者负责归一化。现已核对 root 提供的上游 A 真代码片段：`common_imatrix_load`（`imatrix-loader.txt`）按 `.in_sum2`/`.counts` 配对，两个 tensor 都按 float32 原始数据读取；counts 经 `std::lround` 转为整数。quantize 的 `load_imatrix`（`imatrix-normalize.txt`）令 `ne0=sums.size()/counts.size()`，按每专家的连续 `ne0` 个值除以该专家 count；非正 count 的整行改为 1。因此原转换器的 expert-major raw sums 与 `[n_experts,1]` float32 counts 排布确有公开 A 消费路径，**不用修改 A**，不再只是引用旧工具的注释。

`chunk_count` 不参与 GGUF 权重归一化；`chunk_size` 参与 trace 中的 chunk 数估算，三个 metadata key 都必须存在。转换器的 `chunk_size=1` 与逐 token 累积相符，但 `chunk_count=n_decoded` 仅描述处理记录量，不能证明完整遍历 corpus。A 的这两个函数没有替输入做足验证：加载时无 tensor dtype 检查；normalize 假定 counts 非空且 sums 可被 counts 数整除；counts 经过 float32 和 lround 还存在大整数精度约束。独立 C 导出器应明确输出 F32、验证 expert-major 尺寸与有限非负数据、确保 count 的整数表示符合约定，并显式处理 partial/reset/provenance。上述结论是源代码契约核对；本轮没有运行实际 A quantize 端到端转换。

原脚本不会拒绝 `complete=false`：`load:30–37` 只要求 EXPERT_IN/HIDDEN flags，再打印 complete 状态；没有校验实际读完的层数、数据 shape、finite 值、每专家计数或数组末尾短读。`wxqcal.py` 对被跳过 covariance 直接 seek，可以跨过 EOF 而没有报错。`calib-dump.cpp:148–157` 根据 flags 声明数据却又以 vector 非空决定是否写数据，若调用者提供不一致状态，也可能写出无法正确读取的文件。迁移时应在 B 格式边界修正这些问题，而不把容错责任分散给每个 C 工具。

完整性不能简化为检查一个 complete 字节：`trie-calib.cpp:1018` 的 n_total 可以是用户选择的 max_nodes；`416–421` 的最后一批没有裁剪至剩余额度，导致 n_decoded 超过这个 total；`1072` 的 complete 只等于遍历没有返回错误。应区分完整选定范围、完整整个 corpus 和异常中止后的检查点。默认转换可要求头、范围、计数一致并完整完成所声明的范围；若允许部分校准，应有显式选项，在输出附记录 partial、decoded/total、max_ctx、reset 数和采样步长，不能伪装成完整校准。现阶段列为应保留的小工具，没有新建大型校准实现。

## S05、S25、S27：从研究目录拆出的实际工具能力

### 公开 callback 的 MoE 校准与原始样本采集

`trie-calib.cpp` 的 `collector::want/observe_gate/observe_down` 通过 `GGML_OP_MUL_MAT_ID` 和权重 tensor 名找到路由专家的输入、路由 id、SwiGLU 后隐藏值。`stage_prefix:152–160` 对 host buffer 直接读，对设备 buffer 仅拿采样需要的前缀；`params.cb_eval:904–905` 是真实的 C 消费路径。它能独立收集 routing counts、activation second moments、per-expert moments、hidden energy、optional covariance。`maybe_dump:401–412` 的定时检查点、`write_calib` 的临时文件替换、catch 后保存 partial，都是小而有用的鲁棒性行为。

这种统计加 DFS 的主体不需要私有 KV 格式：`492–505` 使用公开 seq state 数据接口，在分支处保存实际状态并恢复。迁移时仍需适配当前 A 的 common/CLI 字段和诊断 accessor，验证具体模型的 callback tensor 名、shape 和 dtype。不能用仅把若干文本请求发给 HTTP 服务来冒充同一校准人口；DFS 对共享前缀算一次，而独立请求重放会按请求重复计数，两种统计权重不同，应显式选择。

`wxq-manifold.cpp` 是另一个可保留的 C：`244–245` 安装公开 callback，`observe:120–164` 采样 gate 输入与专家 id、down slot 0 的隐藏值，`262–279` 按请求读前缀并清理上下文。它已经是独立的实际消费者，没有理由因为后续 nonlinear student 研究未成功便一并删掉。不过当前 `fetch:103–117` 假设 tensor contiguous，`write_dump:184–206` 没有保存 complete、请求/node id 或模型身份，`270` 的 decode 失败仅跳出内层循环，最后仍返回 0。迁移应保留原始样本能力并修正成功条件和来源字段。

### GGUF expert slab 和离线分析

`gguf-slab.cpp` 提供不用运行模型的专家矩阵读取和 dequantize，输出拥有数据的 float vector；`gguf-slab.h:43` 的 find 返回 reader-owned ggml_tensor 指针，新的 B 接口如暴露 metadata 应改为值返回。由它消费的 `calib-probe.cpp` 可以报告加权/未加权谱、participation ratio、专家间对 neuron permutation 不敏感的 Gram 相似度、WHT 前后分布、OBQ 指标；`qmoe-probe.cpp:60–89,163–198` 可报告三值化的 entropy 与加权失真。这些都是可运行的离线分析职责，依赖公开 gguf/ggml 和 B 数值函数，没有必要修改 A 图结构。

这些输出不能被当作已部署编码器的吞吐或压缩证明：`calib-probe.cpp:89,302–317` 用的是启发式率失真曲线；当前 Jacobi eigensolver 固定 30 次 sweep，没有返回收敛状态；校准缺失时部分路径回退无权 metric，部分输出跳过样本。保留工具时应注明采样、校准完整性、收敛/失败，并用小矩阵真值校验真正消费的数值函数。`parallel.cpp` 全局线程数与逐次 spawn 没有独立用户价值，不应为了保留目录结构而复制未使用的通用库。

`wxq-bake.cpp:178–295` 提供具体的离线 ablation：复制源 GGUF，仅把选中的 gate/up slab 变换后按原类型量化回原偏移；fitted mode 也能用零 padding 把较窄网络装进原形状。它不修改 A，可留作独立 C 实验工具；零 padding 本身不会减少文件大小或原模型的矩阵计算量。必须先验证输入/输出几何和文件，再写目标：`fitted::load:54–64` 不校验尺寸匹配，`255–266` 按模型的 expert/row 数索引文件提供的数组，可越界；写入完成和缺少 imatrix 的处理也需显式定义。`make-transforms.py` 与 `fit-test.py` 是这类实验的研究配方，不等于清洁 fork 的默认推理功能。

### 按 node 对齐的分布对比：保留能力，修正指标

`wxq-kl.cpp` 的独立价值是 stable node id、top-K log probabilities 与覆盖质量、按位置 CSV、事后 quantile/CVaR95，以及公开 seq snapshot DFS；全部可在 C 和 B 实现。其 WXQKL001 头记录 vocab、top_k、stride、record_count，但没有 corpus/tokenizer 身份、context-reset 策略、完整性标记。当前 reader 没检查 token id 边界，主程序仅检查 stride，node mismatch 只是计数，提前读完也可作为正常结束，不能直接作为新 fork 验证器。

更关键的是 `score:190–208` 的指标命名错误。令 S 为 reference top-K，m 为其质量，它实际输出：

`sum(i in S, p_i * log(p_i/q_i)) / m`。

这不是完整 KL，也不是把 p、q 都条件化到 S 后的 KL。例如 p=(0.6,0.4)、q=(0.9,0.1)、S={第一个 token}，旧输出为 log(2/3)，约 −0.405465；真正条件 KL 为 0，完整 KL 约 0.311239。reference 缺失质量小并不能单独给出完整 KL 的可靠误差上界。

可选择两种明确、可测试的指标：条件 KL 应额外包含 `log(q(S)) - log(m)`；或把所有 tail 合并为一个额外类别，输出 `sum_S p_i log(p_i/q_i) + (1-m) log((1-m)/(1-q(S)))`，这是 coarse-grained KL，通常为全 KL 的下界。若只想保留原数值，也应准确命名为截断 support 的平均 log-ratio，不能继续标成 KL。用上述两类别解析例、identical logits 和完整 support 就能验证数学契约，不需要 GPU。

该工具还存在独立的 DFS 排空问题：`visit:288–289` 在 leaf 返回时可能留着尚未 decode 的 batch；主循环 `446–448` 在下一个 root child 前先 reset memory，未先 flush 上一棵根分支的尾批。新的 C 调度必须在真实分支/root 边界排空后再清理/恢复，并对多 root、短路径和非整 batch 例子对拍。root 的新 scorer 已采用另外的实际状态验证链，应复用它而不复制这一旧驱动错误。

## 其余细项的独立处置

| ID | 具体细项 | 处置 |
|---|---|---|
| S09 | root CMake 的 pocs/wanxiangqi 注入，预定义 header 安装；wanxiangqi 强制取 tokenizer 依赖并改全局 CMake flags。 | 为实际消费者建立独立 target/package；tokenizer 和 probe 是显式可选依赖。构建集成本身不是应继承的 A 补丁。 |
| S11 | `speculative-simple.cpp:16–25,207–233` 对 recurrent/hybrid 选择 checkpoint flags；`249–292` target verify 计时。 | checkpoint 策略有价值，但 ON_DEVICE 依赖旧 fork 接口。优先 C 使用 A 真正支持的 state flags；不要用 unsupported flag 伪装 device snapshot。计时需说明仅覆盖 verification。 |
| S14 | `llama-lazy-reader.h:62–142` 排序后按 worker chunk 内去重、pread 重试 EINTR、短读报错、线程创建失败先 join、worker 异常回传。 | 这些实现细节值得抽成 B 文件行读取器，C 调用公开 dequantizer。需校验 n_threads>0、row/slot/offset 溢出、F32 row_size，默认生产 loader 仍未接通 lazy.add。 |
| S15 | `llama-model-loader.cpp:1705–1722` MIRRORED meta mmap 显式 upload；dry-run 跳过磁盘读但保留设备写。 | 前者是 TP5 对应 backend 的一致性修复，后者是 probe 行为。没有对应真实 meta backend/探针 C 消费时不单独复制。 |
| S16 | 新的 Qwen4/PLE/NextN saver metadata 与 source-path 访问器。 | 仅搬可证明缺失且被工具消费的字段；uint64 数组支持单独保留；清理重复注册。普通文件保存已由 A 提供，不能声称重建出新的 streaming save。 |
| S17 | `llama_sampler_dist_draw:4044–4051` 通过 sampler 名称判断后私有 cast，额外消费一次 uniform RNG。 | 保留为随机流兼容契约证据。B 应拥有自己的 RNG，或未来加窄 public A RNG-advance API；C 不能靠 `name==dist` 猜私有对象布局。 |
| S18 | delta-net/Qwen35/Qwen3Next 的 RERoT B+H、共用 brain/child row、capacity-tail invariants。 | 依赖其真实状态模型，不能降格成普通多次请求。大功能归档时仍留下数学/回滚边界要求；没有证明独立的小数学收益就不搬通用图补丁。 |
| S19 | DFlash device embeddings 的 3D layout 和 host interleave 一致性。 | 有用的是明确布局契约与真 DFlash 消费者；没有设备交接 API 时不复制模型 graph 分支。 |
| S20 | Nanbeige 44 logical layers 对应 22 physical blocks；TriAttention config 校验真实 head_dim、RoPE layout、full-attention 层和同构几何。 | 几何验证应随真实校准消费链保留。TriAttention 采集仍用私有 hparams/pre-RoPE outputs；不能和 S05 的公开 callback 混称为同一可移植程度。 |
| S22 | llama-bench 的 lazy-mode 参数矩阵、模式结果列；删除 zero-prompt skip；默认 load mode 由 AUTO 变 MMAP。 | C 配置/bench profile 可保留参数，但只报告真的生效模式。zero-prompt 删除需要目标行为证明，不能因删除行数少就自动采用；默认参数没有本轮性能证据。 |
| S23 | `tools/quantize/quantize.cpp:37` 注册 PQ2_0 名称。 | 依赖真实 GGML codec/type/backend 支持。单独加 CLI 名称不构成可用量化能力。 |
| S24 | UI 类型加入 reasoning effort 各级与 reasoning_budget_tokens。 | 仅在新 C/server adapter 实际消费并返回一致协议时保留，类型定义不能证明 runtime 支持。 |
| S26 | fitted-weight/transform 文件与 nonlinear student 拟合脚本。 | 留可再现研究配方与离线工具边界；修正缺 import、维度/完整性验证后才可重新运行，不把历史注释结论当当前模型性能。 |

## 已交付补丁与验证边界

`corpus-compat.patch` 修改 `fork/corpus/src/trie.cpp`、public header、独立测试及 README，没有改 A。新增 owned `request_metadata`，其 request_kind 使用旧格式 1–4 值；`append(tokens, metadata)` 保留元数据，`request_metadata_at(id)` 值返回；原 `append(tokens)` 继续可用，明确默认 completion/未知时间 0。空请求保留 root leaf 0、length 0。原 writer 仅在首次写记录时创建流，因此没有 node 文件的 empty-only corpus、尚未使用的空目录也能按 0 records 读取；缺少节点的非空请求仍失败。

补丁拒绝第一轮错误 start_id=0 的快照，不能凭全零字节恢复已经丢失的 capture metadata。它还拒绝未知 kind、非零 node reserved、duplicate edge、截断、非法 parent/leaf/depth/token，以及后续 numeric shard，包括不连续 shard 与 shard 0 的另一种拼写。暂不支持分片时明确失败，比静默把第一片当完整人口更能保护现有数据语义。它仍是独立 snapshot 包，不承诺 concurrent recording、tail repair、fsync 或 crash-atomic 发布。

本机完成 `g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic` 编译，测试输出 `corpus tests passed`。测试包含完整的 canonical headers 与 5 个 request records 的逐字节 fixture（四种 kind、两个高位非零 timestamp、duplicate request、internal endpoint、empty request），load/save 和 append/save 均比较完整字节；另测默认 metadata、ownership/copy/两实例、错误元数据不改变 tree、非法 header/shard、empty-only 缺失 node stream，以及 unused corpus。

`corpus-production-interop.cpp` 使用原封不动的 `server-corpus-dump.{h,cpp}` 和真实 A 头，先由原 writer 写 mixed 与 empty-only corpus，再由 B 读取并逐字节保存，再由原 reader 检查数量和 header/record。验证前补入真实生产 manifest，并对验证输入的 nodes、requests、manifest 做前后完整比较，证明原 reader 验证步骤不修改输入。该程序不 mock loader，不在 B 的默认测试中引入 llama/ggml 依赖；实际 A 环境执行由 root 接手。

79 个文件的静态覆盖没有剩余待读 hunk；imatrix 已按 A 的实际 loader/normalizer 核对，corpus 的原生产互操作由 root 执行。以上未测实模/性能的候选会保持“待验证”状态。后续追加的 714 个非 artifacts/reports 路径用途清点单独记录为 `audit-purpose-findings.md` / `audit-purpose-coverage.json`，明确区分内容已读与仅由文件名、提交主题推断用途，不改变本报告 79 文件的内容审计口径。
