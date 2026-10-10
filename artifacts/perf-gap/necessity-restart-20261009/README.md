# 2026-10-09 必要性审计

主报告：[性能理论重建](../../../docs/performance-necessity.md)。

`audit.json` 是首次源清单与反例检查；`audit-with-observations.json` 另行复核既有
07:30轮次的24份测量响应。后者的observations字段与所有理论计算完全分离，
没有把实测成本作为下界输入，也没有重新跑GPU。

原先274→95–104→64.74–67.68的数字不能作为必要硬件上界。输出将全局必要
速度上界留为null、最优性Gap资格留为false，而不是补一个取自现状的常数。

可重现命令（使用新输出路径，旧审计拒绝覆盖）：

```bash
python3 tests/test-tp5-necessity-audit.py
python3 scripts/tp5-necessity-audit.py \
  --metadata artifacts/perf-gap/run-20261008T1845/apex-metadata.json \
  --layout tp5-manifest.json --shards 6 \
  --observations artifacts/perf-gap/run-20261009T0730 \
  --out artifacts/perf-gap/necessity-restart-20261009/recheck.json
```

源码读取范围、未证明前提、量化尺寸及模型布局哈希、逐请求原始hash都在JSON中。
本轮只改审计工具、理论文档与证据标签；不修改现有生产算法和失败现场。

定向自检：必要性审计21项、原独立预算7项、原成本模型11项，共39项通过；
`git diff --check`通过。没有运行GPU测试或全量生产回归。
