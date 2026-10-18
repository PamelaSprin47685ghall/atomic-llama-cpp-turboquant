# AXIOM

1. 五卡 RELAY 需要一份不依赖设备的通道所有权合同。
   Reason: 旧 fork 的 RELAY 有历史收益，但整条私有 runtime 已归档；先在 CPU 上
   独立实现、拒绝和验证 bank 生命周期，platform 与模型资格才不会被无效性能数据
   掩盖（指南 00.1、39.2 R0）。
2. 开发者需要在没有模型、没有 GPU 的机器上构建并审查这份合同。
   Reason: 所有权错误（慢读者覆盖、陈旧票据、部分提交）必须在进入 HIP 热路径
   之前变成可重复的 CPU 反例（指南 30.6、50）。
3. 计划器必须对不合法配置给出可行动的拒绝，而不是运行到一半再失败。
   Reason: 五个非空 rank、原量化块不被拆开、输出覆盖完整，是装载前就能判定的
   几何事实（指南 20.4、54.7）。

# LEMMA

1. [AXIOM 1; 父 LEMMA 1、2] bank、票据与 credit 是纯值合同，与 HIP、llama 无关。
   由 C 组合账本、transport 事实与设备提交登记；A 与 B 不互相包含或调用。
2. [AXIOM 2] 每份可变状态只有一个修改者：bank 归通道账本，尝试归提交登记，
   两者都是单线程可信调用方模型，跨线程串行化属于 C。
3. [AXIOM 3] 拒绝是值不是日志：决策、错误码、字段与详情随结果返回；内部守恒
   检查失败是编程错误，直接终止。
4. [AXIOM 1; 父 LEMMA 4] 票据与租约由签发者私有构造，副本不重复释放；唯一能够
   复用的证据是账本登记的全部完成事实。

# SPEC

1. [LEMMA 1] `channel_ledger` 实现保守合并 bank 模型：同一槽位输入与输出合在一
   起预约，复用需要输入读完、输出发布、全部消费者最后读取完成；精确票据匹配，
   陈旧/外来/跨分配代次租约被拒绝；fail() 保留首个原因并停止新工作；
   mark_drained 只接受外部已证明事实。
   Proof: `tests/ledger-test.cpp`；记录见 `../VALIDATION.md`。
2. [LEMMA 2] `run_submissions` 在 provider 调用前把尝试登记为 unknown，封口要求
   全部尝试已解决、无 unknown、全部 rank accepted 且 run 未失败；排空只由 accepted
   与 unknown 尝试的完成事实推导，no-effect 拒绝不进在飞集合，同一 rank 允许多个
   尝试。
   Proof: `tests/submissions-test.cpp`；记录见 `../VALIDATION.md`。
3. [LEMMA 3、4] `compile_glu_plan` 移植已验证的 Python 参考
   `tests/python-reference/tp5_local_plan.py`（SHA-256
   5912998214d411bda905a25f58878bf9550c8e7e7cfa9ad43733a5736b59e657）：
   覆盖、整块边界、受检字节与 MAC 记账、GATHER/SUM 两种代数与 dense 独立参考；
   所有拒绝带决策、错误码与字段。
   Proof: `tests/plan-test.cpp` 与 `tests/crosscheck-plan.py`（实时与 Python 参考
   对照）；记录见 `../VALIDATION.md`。
4. [LEMMA 1] 只依赖 C++17 标准库；安装头文件、静态库与 CMake 包足以让无关消费者
   独立构建。
   Proof: 独立 configure/build/test/install 与搬迁消费验证；记录见
   `../VALIDATION.md`。
5. [LEMMA 2] 单线程可信调用方合同写进头文件与 README；测试只证明 CPU 所有权
   模型，不推导 HIP 内存序、GPU 前进性或模型数值。
   Proof: 各测试输出的 scope 行与 `../VALIDATION.md` 的边界记录。

# Exceptions

无。没有异步、共享实例、回调或 A 依赖。跨线程使用由 C 串行化；跨进程
checkpoint 恢复由 C 重新签发 run 身份，本库不复活旧句柄。
