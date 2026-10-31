【E5 Bot → lead 总工头 · 最终交接 · 2026-10-30 ~11:40 JST】
来自上一任车道执行器。Ashley 已把总工头交给你；以下请收进你的调度，不必回我。

## 车道与纪律
- 12022：Ashley 明确「先别补挂」。现有 lead-* 跑完即可，勿再大批 start 填满 20+。
- 13022 HIP 1/1：Ashley 明确「先不烧」。勿 flock / 勿挂 gpu.sh 新烧；lazy omp CPU 不占 1/1，可继续用。
- 12022 新纪律（你已 steer）：禁止编译/测试（含 CPU harness/语法检查）；只写码+静态审查；输出真实补丁/基线/依赖/可执行验证命令；不得动 GPU。
- build-slot 路径：`/home/kunweiz/.local/bin/build-slot`（勿找 hermes 槽）。
- providers：kw92.cyou/lazy/{default,task}。无 RSZ_RT。无 HIP 证明勿 promote。

## 交接全文（必读）
13022：`/home/kunweiz/dispatch/HANDOFF-MATH-VISION-20261030.md`（E 根同名；源包 `HANDOFF-20261030-sources/`）。12022 上可能缺这份——柱1 首刀已踩过坑。需要时从 13022 拷。

## 四柱现状（硬事实）
1. **varlen / n_kv·n_outputs**：未 PROVEN。`GGML_TP5_FA_MASK_KVLEN=1` 在核修 ncols2>1 越界前 **勿开 GPU/HIP**。旧 nkv-fa-true-len-prove 多轮 PARTIAL；你的 `lead-fa-boundary` 对准闭合边界。
2. **LateBind**：DEFAULT-ON SURFACE 已 PROVEN（mtp-lb-default-on-hip / promote-lctx / off0-on1）。树内 UNSET≡ON 默认化未合入。`lead-latebind-stock` 已 done：r8 与 pristine stock OFF 分流有契约风险，**不得用分流冒充同一 MTP 工作负载的 stock/OFF**。
3. **双窗异步重叠**：SERIALCUT4 = SAFE SPEED=NOISE；HIP_SPANS 直计量 KEY_V3=1 = SAFE, OVERLAP NOT PROVEN。GATE3 卡在同 PCI 双窗；`hipspans-joiner-pcimap-r8-081800` 已把 joiner 改成 MAP `provider=` 归属（fixture 51/51），但原四腿日志仍 `cross_provider_unattributed` / pairs_overlap=0。**下一刀**需带 provider= 的 MAP 新烧才能重判——等 Ashley 点头，你勿自动 flock。产物：13022 `/home/kunweiz/work/hipspans-joiner-pcimap-r8-081800/`（REPORT/REJOIN_RESULT/patch.diff）。`lead-provider-map` / `lead-union-relay` 是预备轨。
4. **Prefab / 反糊弄**：prefab-share-live 多轮 PARTIAL；真实 ON/OFF 构图下降 + OFF≡stock 未证明。`lead-prefab-fix` 在追；`lead-router-m2` 是 N3 router scratch 去浪费（别放宽别名安全）。

## 你已挂的刀（bot=lead）
running：lead-fa-boundary / lead-prefab-fix / lead-union-relay / lead-provider-map / lead-router-m2  
done：lead-fa-true-len（缺 13022 交接文件，只读未证明）/ lead-prefab（对照 PARTIAL）/ lead-latebind-stock（基线澄清，未晋升）

## 请你做到
- 收口要 PARTIAL/PROVEN 分明；无设备验收勿称 PROVEN。
- 子刀 finish 写清 SUMMARY；INDEX 别留假 running。
- 柱3 重叠证明：优先 provider= 身份链 + 真双窗同时执行证据；旧无归属日志 fail-closed。
- 额度紧：少空转、少重复谱系；能静态闭合就别烧卡。

E5 Bot 侧停自动补挂与 flock；观察权交给你。完。


## 更正 (2026-10-30 ~11:43 JST)
「12022 先不补挂」仅约束 E5 Bot，不约束 lead。Ashley：新工头可无限量派工。13022 HIP flock 新烧仍等点头。

## 再更正 (~11:44 JST)
Ashley：flock 也可新开始排队（含 provider= HIP 新烧）。E5 不代挂。

## E5 收工 (~11:45 JST)
Ashley：交代完休假。lead 全权派工+可 flock；E5 停巡视。
