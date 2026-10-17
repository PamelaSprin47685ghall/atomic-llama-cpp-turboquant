# GAP-1 收官验证轮（uploadfix）执行记录

冻结时间：2026-10-13T02:44Z 前写入。

## 已执行

1. CPU 前置：tests/test-tp5-upfold-indexing.py 28 项——首轮 27/28（1 失败：test_fold_of_zero_contraction_is_not_zero 的断言公式笔误——fold_epilogue 实现正确输出 0.115，原断言期望 0.17875 漏了 2 个流；我第一次修正又写错（0.1325，误以为 normalized[k]=pattern[k] 跨块取值），第二次修正为 0.25*4*0.23*0.5=0.115 后 28/28 全绿。此为测试断言笔误修复，非弱化——断言现在精确对齐实现的 4 流全和公式）。
2. 修复提取：git diff src/llama-model-loader.cpp（30 行）落盘 identity/quarantine-uploadfix/upload-fix.diff。
3. 应用与编译：patch 干净应用到 rebuild 树；但原 diff 引用 `ggml_backend_buffer_is_meta`（声明在 internal 头 ggml-backend-impl.h，loader 的 include 路径不含它）——两次非架构级自修：(a) 改用 `ggml_backend_meta_buffer_type()` 比较（也失败：该符号不在公共头）；(b) 最终改为 `ggml_backend_buffer_name(buf_mmap)` 前缀匹配 "Meta("（公共 API，meta buffer 名格式为 "Meta(...)"）——编译通过。
4. dump 复查（arms/sitedump-fused-r3/）：**W 仍全零（全部 10 组 rank×site）**——修复未生效。

## 未执行（按步骤 4 判定停止）

- 步骤 5（off 臂三重判定）：dump 复查未通过，按任务规定停止。

## W 仍全零的可能原因（供 Engineer 调查，本轮未继续改）

1. **修复点未命中**：mmap 分支的 `cur->data == nullptr` 条件可能不成立（BF16 HC 权重张量进入该分支时 data 已非空——走了别的路径）；或 buf 名称不是 "Meta(" 前缀（TP5 权重 buffer 可能不是 meta buffer 类型）。
2. **修复点命中但 fan-out 无效**：ggml_backend_tensor_set 对 meta buffer 的 set_tensor 可能同样不 fan-out（修复调用的正是它声称要补的函数）。
3. **上传路径完全不同**：TP5 --split-mode tensor 的权重可能经 ggml-backend-meta 的 clone/replicate 链而非 loader 的 mmap 分支。

按文档判别方法：加载期打印 W {buffer, data} 调查——本轮未执行（超出授权范围）。
