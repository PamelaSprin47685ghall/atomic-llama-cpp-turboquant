#!/usr/bin/env bash
# ==============================================================================
# make-tp5-mtp-evidence-fixtures.sh
#
# 生成 check-tp5-mtp-evidence.sh 的合成日志演练夹具（只写文件，不做任何判定）。
# DevOps 重跑方式：
#   bash scripts/make-tp5-mtp-evidence-fixtures.sh /tmp/tp5-mtp-drill
#   for f in /tmp/tp5-mtp-drill/*.log; do
#       echo "=== $f"; bash scripts/check-tp5-mtp-evidence.sh "$f"; echo "rc=$?";
#   done
#
# 期望矩阵（rc = check 脚本退出码）：
#   pass.log             rc=0  全绿
#   rebuild_fail.log     rc=1  [A] MTP type=3 多次重建 + UID 分裂
#   hidden_fail.log      rc=1  [E] generation 错配
#   cycle_missing.log    rc=1  [F] 零 cycle 行（PROFILE 下缺失即 FAIL）
#   conservation_fail.log rc=1 [F] 时间恒等式背离 + Token 守恒背离 + target_us=0
#   cycle_seq_fail.log   rc=1  [F] 周期序号不连续（单调连续性断言拦截）
#   state_spin_fail.log  rc=1  [L][M][D] sidecar 超时 + CHAIN FAILED + DeviceLost
# ==============================================================================

set -euo pipefail

OUT_DIR="${1:-/tmp/tp5-mtp-drill}"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------- pass.log ---
cat > "$OUT_DIR/pass.log" <<'EOF'
[tp5-numerical-mode] mode=reference reason=disabled-by-env wire=f16 late=no direct=p1
[tp5-mtp-graph] type=3 reuse=0 definition_uid=0x510000000001 ubatch_tokens=1
[tp5-mtp-graph] type=3 reuse=1 definition_uid=0x510000000001 n_reused=1 ubatch_tokens=4
[tp5-mtp-graph] type=3 reuse=1 definition_uid=0x510000000001 n_reused=2 ubatch_tokens=1
[tp5-mtp-graph] type=3 reuse=1 definition_uid=0x510000000001 n_reused=3 ubatch_tokens=2
[tp5-mtp-graph] type=0 reuse=0 definition_uid=0x510000000010 ubatch_tokens=32
[tp5-mtp-graph] type=0 reuse=1 definition_uid=0x510000000010 n_reused=1 ubatch_tokens=1
[tp5-mtp-cycle] cycle=1 draft_us=1200 target_us=3500 catchup_us=800 handoff_us=50 total_us=5550 draft_tokens=3 accepted_tokens=2 final_tokens=3 eff=0.667 dev_hidden=1
[tp5-mtp-cycle] cycle=2 draft_us=1000 target_us=3000 catchup_us=200 handoff_us=40 total_us=4240 draft_tokens=3 accepted_tokens=0 final_tokens=1 eff=0.000 dev_hidden=1
[tp5-mtp-cycle-summary] cycles=2 avg_draft_us=1100.0 avg_target_us=3250.0 avg_catchup_us=500.0 avg_handoff_us=45.0 avg_total_us=4895.0 draft_tokens=6 accepted_tokens=2 final_tokens=4 eff=0.333
[tp5-meta] SUBMIT_EPOCH_CHAIN (PREDEFINED TRUTH): hits=1
[tp5-meta] SUBMIT_EPOCH_CHAIN (PREDEFINED TRUTH): hits=2
EOF

# ------------------------------------------------------ rebuild_fail.log ---
cat > "$OUT_DIR/rebuild_fail.log" <<'EOF'
[tp5-numerical-mode] mode=reference reason=disabled-by-env wire=f16 late=no direct=p1
[tp5-mtp-graph] type=3 reuse=0 definition_uid=0x510000000001 ubatch_tokens=1
[tp5-mtp-graph] type=3 reuse=0 definition_uid=0x510000000002 ubatch_tokens=4
[tp5-mtp-graph] type=3 reuse=0 definition_uid=0x510000000003 ubatch_tokens=1
[tp5-mtp-cycle] cycle=1 draft_us=1200 target_us=3500 catchup_us=800 handoff_us=50 total_us=5550 draft_tokens=3 accepted_tokens=2 final_tokens=3 eff=0.667 dev_hidden=1
[tp5-meta] SUBMIT_EPOCH_CHAIN (PREDEFINED TRUTH): hits=1
EOF

# -------------------------------------------------------- hidden_fail.log ---
cat > "$OUT_DIR/hidden_fail.log" <<'EOF'
[tp5-numerical-mode] mode=reference reason=disabled-by-env wire=f16 late=no direct=p1
[tp5-mtp-graph] type=3 reuse=0 definition_uid=0x510000000001 ubatch_tokens=1
[tp5-mtp-graph] type=3 reuse=1 definition_uid=0x510000000001 n_reused=1 ubatch_tokens=4
[tp5-mtp-hidden] copy gen mismatch: src_slot=0 exp_gen=7 actual_gen=6 valid_rows=4
[tp5-mtp-cycle] cycle=1 draft_us=1200 target_us=3500 catchup_us=800 handoff_us=50 total_us=5550 draft_tokens=3 accepted_tokens=2 final_tokens=3 eff=0.667 dev_hidden=1
[tp5-meta] SUBMIT_EPOCH_CHAIN (PREDEFINED TRUTH): hits=1
EOF

# ------------------------------------------------------- cycle_missing.log ---
cat > "$OUT_DIR/cycle_missing.log" <<'EOF'
[tp5-numerical-mode] mode=reference reason=disabled-by-env wire=f16 late=no direct=p1
[tp5-mtp-graph] type=3 reuse=0 definition_uid=0x510000000001 ubatch_tokens=1
[tp5-mtp-graph] type=3 reuse=1 definition_uid=0x510000000001 n_reused=1 ubatch_tokens=4
[tp5-meta] SUBMIT_EPOCH_CHAIN (PREDEFINED TRUTH): hits=1
EOF

# --------------------------------------------------- conservation_fail.log ---
cat > "$OUT_DIR/conservation_fail.log" <<'EOF'
[tp5-numerical-mode] mode=reference reason=disabled-by-env wire=f16 late=no direct=p1
[tp5-mtp-graph] type=3 reuse=0 definition_uid=0x510000000001 ubatch_tokens=1
[tp5-mtp-graph] type=3 reuse=1 definition_uid=0x510000000001 n_reused=1 ubatch_tokens=4
[tp5-mtp-cycle] cycle=1 draft_us=1200 target_us=3500 catchup_us=800 handoff_us=50 total_us=9999 draft_tokens=3 accepted_tokens=2 final_tokens=3 eff=0.667 dev_hidden=1
[tp5-mtp-cycle] cycle=2 draft_us=1000 target_us=0 catchup_us=200 handoff_us=40 total_us=1240 draft_tokens=3 accepted_tokens=1 final_tokens=5 eff=0.333 dev_hidden=0
[tp5-meta] SUBMIT_EPOCH_CHAIN (PREDEFINED TRUTH): hits=1
EOF

# ----------------------------------------------------- cycle_seq_fail.log ---
cat > "$OUT_DIR/cycle_seq_fail.log" <<'EOF'
[tp5-numerical-mode] mode=reference reason=disabled-by-env wire=f16 late=no direct=p1
[tp5-mtp-graph] type=3 reuse=0 definition_uid=0x510000000001 ubatch_tokens=1
[tp5-mtp-graph] type=3 reuse=1 definition_uid=0x510000000001 n_reused=1 ubatch_tokens=4
[tp5-mtp-cycle] cycle=1 draft_us=1200 target_us=3500 catchup_us=800 handoff_us=50 total_us=5550 draft_tokens=3 accepted_tokens=2 final_tokens=3 eff=0.667 dev_hidden=1
[tp5-mtp-cycle] cycle=3 draft_us=1000 target_us=3000 catchup_us=200 handoff_us=40 total_us=4240 draft_tokens=3 accepted_tokens=0 final_tokens=1 eff=0.000 dev_hidden=1
[tp5-meta] SUBMIT_EPOCH_CHAIN (PREDEFINED TRUTH): hits=1
EOF

# ----------------------------------------------------- state_spin_fail.log ---
cat > "$OUT_DIR/state_spin_fail.log" <<'EOF'
[tp5-numerical-mode] mode=reference reason=disabled-by-env wire=f16 late=no direct=p1
[tp5-mtp-graph] type=3 reuse=0 definition_uid=0x510000000001 ubatch_tokens=1
[tp5-mtp-graph] type=3 reuse=1 definition_uid=0x510000000001 n_reused=1 ubatch_tokens=4
[tp5-mtp-cycle] cycle=1 draft_us=1200 target_us=3500 catchup_us=800 handoff_us=50 total_us=5550 draft_tokens=3 accepted_tokens=2 final_tokens=3 eff=0.667 dev_hidden=1
[tp5-meta] SUBMIT_EPOCH_CHAIN FAILED; refusing fallback after native submission
RELAY LateBind sidecar timeout on rank 2 (epoch=9, expected=18, flag=0, error=0, done=0)
ErrorDeviceLost: vkQueueSubmit failed on device 1
EOF

echo "fixtures written to $OUT_DIR:"
ls -1 "$OUT_DIR"
