# max-footprint-share 安全修复交付

基线：stock `f1c5fafc0449b22969ddabfcc9c361940d94d033`。按顺序应用 `01-share.patch`（原 share 补丁）和 `02-safety.patch`（原 `/home/kunweiz/work/max-footprint-share-live/wt` 到 `share-safety/src` 的六个相关源文件差异）。第二份不是单独对 stock 的补丁；两者依次构成相对于 stock 的补丁。增量只取 ggml allocator/backend 的两个头文件与两个实现文件以及 `src/llama-context.{h,cpp}`。原补丁由 `/home/kunweiz/work/max-footprint-share-live/patch.diff` 复制；原 wt 已含 share 补丁。未对补丁执行应用或检查，因此可应用性尚未验证。

本次修改：share 专用测量使用独立 gallocr，避免测量释放已有计算 buffer；最小容量 reserve 返回 bool 并向 context 传播分配失败；处理共用 allocator/buffer 的 alias；pipeline 分配失败关闭 pipeline 后重新测量 pp/tg 再按新的 max reserve。未将 union-per-window 的语义移植为 share 语义。门仅精确 `GGML_TP5_MAXFP_SHARE=1` 打开，no_alloc 和 phase arena 分支仍按原分支处理。

**限制与风险**：按 backend 的总容量 floor 并不保证另一个 graph 的 chunk placement 一定适配，不能承诺零重分配。测量 API 是 void，底层 placement/内存失败仍为 stock assert/abort，非可恢复错误传播。源树中未确认可用的真实设备分配失败注入机制；现有 `test-alloc` 使用 dummy backend，不是本任务禁止 mock 条件下的真实注入证据。尚无可运行的真实失败注入验收命令，必须先提供能按分配次数/位置令实际 backend 分配失败的机制，再验证首次 reserve、floor 增长、alias 及 pipeline 回退失败；不能以模拟器或臆造环境变量代替。也未验证两个补丁与其他工作树叠加的兼容性。

本次仅静态读取、修改与生成差异；未执行补丁应用检查、编译、测试、shell -n、mock 或 GPU。既有 evidence 是早先版本的记录，不证明本次修改。命令和依赖见 `VERIFY.txt`；这些命令均未执行。本交付不是已验证的生产安全证明。
