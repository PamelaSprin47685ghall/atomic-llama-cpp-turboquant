# 保护集身份与哈希登记规范

- 规范版本：v1
- 制定日期：2026-10-10
- 职责划分：本文件由 Engineer 纯静态编写并完成结构落盘。内容哈希的实际计算与封存由 DevOps 在后续执行阶段于干净环境下计算并落盘至 `HASHES.json`。

---

## 1. 保护集文件完整性清单

在保护集落盘后、正式冻结生效时，保护集包含以下必需文件：

1. `guardset.md`：预注册总纲领与冻结声明
2. `rubric.md`：机器判定标准与打分细则
3. `identity.md`：本登记规范
4. `questions/manifest.json`：题目总索引与元数据
5. `questions/01_zh_science.json`：中文自然问答（科学常识）
6. `questions/02_en_general.json`：英文自然问答（能源通识）
7. `questions/03_zh_short_fact.json`：中文短回答（历史事实）
8. `questions/04_en_long_writing.json`：英文长回答（阅读习惯）
9. `questions/05_math_logic.json`：算术与多步逻辑推理
10. `questions/06_code_gen.json`：代码生成与语法逻辑
11. `questions/07_context_multi_turn.json`：多轮会话上下文引用
12. `questions/08_cross_request_isolation.json`：跨请求状态隔离验证
13. `questions/09_abort_recovery.json`：客户端中断与异常恢复验证

---

## 2. DevOps 哈希登记操作流程

当 DevOps 在无其他 GPU 跑分任务的干净宿主环境中，准备封存本保护集时，执行标准 SHA256 提取流程：

```bash
# 规范命令（供 DevOps 参考执行）
cd artifacts/perf-guardset-v1
sha256sum guardset.md rubric.md identity.md questions/*.json > HASHES.txt
python3 -c '
import json, pathlib
hashes = {}
for line in pathlib.Path("HASHES.txt").read_text().splitlines():
    if line.strip():
        h, p = line.strip().split(None, 1)
        hashes[p] = h
pathlib.Path("HASHES.json").write_text(json.dumps({
    "guardset_version": "guardset-v1-20261010",
    "algorithm": "sha256",
    "files": hashes
}, indent=2, ensure_ascii=False) + "\n")
'
rm -f HASHES.txt
```

---

## 3. 封存后的防篡改核验契约

1. 在每次候选晋级揭示施测前，测试框架必须首先计算当前文件树哈希，并逐项比对 `HASHES.json`。
2. 凡出现任何一个文件哈希不一致、文件缺失或多出未登记文件，视为**保护集受污染 (GUARDSET_CORRUPTED)**。
3. 保护集受污染时，必须终止当前候选的晋级评测，保留现场，向 Manager 报告违规，严禁在受损基准上给出任何合格判定。
