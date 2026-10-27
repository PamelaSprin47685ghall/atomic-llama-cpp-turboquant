# Python 参考程序

这些文件从仓库根 `AGENTS.md` 内嵌代码块按原文提取（提取脚本见指南 50.2），逐字节
SHA-256 与指南 50.9、54.15 记录一致。它们是第 50、54 章已验证的 CPU 参考实现，
不是本库的构建依赖；`crosscheck-plan.py` 在运行测试时把它们当作固定比较对象。

| 文件 | 记录 SHA-256 |
|---|---|
| `tp5_protocol_spec.py` | `4cc04d9643fa670b579de3c51860db26a52f0e6797c666f9ea1448c43b4abba2` |
| `output_credit_model.py` | `f2315f04cb78db2e957bc7fc31eecb8a962b5f85a0a731dc147fc16a99d210b4` |
| `tp5_local_plan.py` | `5912998214d411bda905a25f58878bf9550c8e7e7cfa9ad43733a5736b59e657` |
| `glu-gather.json` | `628cc99b8a669b7a509359a3e1032816e4f34ad3945fd6246ca394d0405c432c` |
| `glu-gather-replicated.json` | `18c6438e82dcc4e9ec77773886f5b51592ffbef8056f3db8d4fb0ce8f3e0eeb0` |
| `glu-middle-infeasible.json` | `41aded22db91253442018a0d77520541a9509a4e879b8d20762c8344d8055d87` |
| `glu-middle-aligned.json` | `1fb856bdcf28c34b288262712e8e7e59cc782a91d9391bf7d0d48aeaad4654da` |

若根指南的对应代码块被修订，应重新提取并更新本表与
`crosscheck-plan.py` 中的固定摘要，不得在未核对的情况下沿用旧摘要。
