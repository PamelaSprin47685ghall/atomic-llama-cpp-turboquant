
## 后记（run-20260927T0230，n=12）：ABBA 中止，无裁决

n=12 vs n=10 ABBA 启动后被中止（daemon broker abort），仅完成 blk1–blk2 的 A 臂。
进程已清理（harness + server），GPU 空闲。n=12 无任何结论；其投影（14 cycles ×
116.5 + 242 ≈ 1873 ms，−6ms，噪声带内）不足以立项。若重启，需完整 5-block。

**用户方向修正（2026-09-27）**：计数题（"1..60"）接受率 1.0 是可预测 workload，
不能代表真实文本。n10 的收益依赖高接受率摊销；真实接受率下最优 horizon 更小。
下一轮（G7）改用**减肥建议**类自然问题作 canonical workload，token 数对齐
（prompt 31 / completion 171 / 自然 stop），先测真实接受率与最优 horizon。
