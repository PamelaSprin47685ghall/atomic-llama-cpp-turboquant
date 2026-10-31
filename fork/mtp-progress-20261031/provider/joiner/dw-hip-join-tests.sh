#!/bin/bash
# DUALWIN-HIP-JOIN self-test for dw-hip-overlap-join (parser: dw-hip-join.h).
# Self-contained: every fixture is generated below from the PRODUCER source
# forms (relay.hip meter_flush_dump, runtime.cpp worker tail) —
#   banner  : TP5_DW_HIP_SPANS enabled=1 flag=GGML_TP5_DW_HIP_SPANS join_map=1 scope=device_stream device_only=1
#   header  : TP5_DW_HIP_SPANS provider=%p rank=%u pci=%s n=%llu dropped=%llu errors=%llu shared_epoch=1
# and every expected verdict value is hand-verified. The enabled= key is
# accepted ONLY on the banner form; errors= is parsed (never ignored) and
# refused when nonzero; span body / map / aggregate lines stay strict.
J="$(cd "$(dirname "$0")/../.." && pwd)/build/dw-hip-overlap-join"
pass=0; fail=0
t() { local name="$1" want="$2"; shift 2; out=$("$J" "$@" 2>&1); rc=$?;
  if [ "$rc" = "$want" ]; then pass=$((pass+1)); echo "PASS $name (rc=$rc)";
  else fail=$((fail+1)); echo "FAIL $name want=$want got=$rc :: $out"; fi; }
want_msg() { local name="$1" pat="$2"; shift 2; out=$("$J" "$@" 2>&1); rc=$?;
  if [ "$rc" = 1 ] && grep -q "$pat" <<<"$out"; then pass=$((pass+1)); echo "PASS $name (rc=1, refused)";
  else fail=$((fail+1)); echo "FAIL $name want rc=1 + '$pat' :: rc=$rc out=$out"; fi; }
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT

# F1 — positive pointer mode on the NEW producer forms: banner with
# enabled=1, per-provider headers with errors=0. Hand-verified values:
#   a(0xaaa) spans [0,150],[200,350]           -> busy 300
#   b(0xbbb) spans [100,150],[200,250]         -> busy 100
#   pair intersection [100,150]+[200,250]      -> overlap 100
#   pci union [0,150]+[200,350] = 300, wall 350, frac 0.857143
cat > "$T/f1.log" <<'LOG'
TP5_DW_HIP_SPANS enabled=1 flag=GGML_TP5_DW_HIP_SPANS join_map=1 scope=device_stream device_only=1
TP5_DW_HIP_OVERLAP_MAP window=0 native=0xaaa pci_bdf0=0000:03:00.0
TP5_DW_HIP_OVERLAP_MAP window=1 native=0xbbb pci_bdf0=0000:03:00.0
TP5_DW_HIP_SPANS provider=0xaaa rank=0 pci=0000:03:00.0 n=2 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0xaaa rank=0 pci=0000:03:00.0 seq=0 begin_ns=0 end_ns=150
TP5_DW_HIP_SPAN provider=0xaaa rank=0 pci=0000:03:00.0 seq=1 begin_ns=200 end_ns=350
TP5_DW_HIP_SPANS provider=0xbbb rank=0 pci=0000:03:00.0 n=2 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0xbbb rank=0 pci=0000:03:00.0 seq=0 begin_ns=100 end_ns=150
TP5_DW_HIP_SPAN provider=0xbbb rank=0 pci=0000:03:00.0 seq=1 begin_ns=200 end_ns=250
TP5_DW_HIP_OVERLAP provider=0xaaa device=0 rank=0 pci=0000:03:00.0 active=1 spans=2 total_ns=300 max_ns=150 skipped_pending=0 skipped_mode=0 errors=0 coverage=enqueue_to_completion_stream_span clock=hip_event device_only=1
LOG
# A: positive pointer mode, values hand-verified (same PCI overlap)
t positive-pointer 0 "$T/f1.log"
out=$("$J" "$T/f1.log")
grep -q "TP5_DW_HIP_OVERLAP_JOINED_PAIR a=0 b=1 pci=0000:03:00.0 spans_a=2 spans_b=2 overlap_ns=100" <<<"$out" && { pass=$((pass+1)); echo "PASS pair-value"; } || { fail=$((fail+1)); echo "FAIL pair-value :: $out"; }
grep -q "TP5_DW_HIP_OVERLAP_JOINED_PCI pci=0000:03:00.0 spans=4 busy_ns=300 wall_ns=350 busy_frac=0.857143" <<<"$out" && { pass=$((pass+1)); echo "PASS pci-values"; } || { fail=$((fail+1)); echo "FAIL pci-values :: $out"; }
grep -q "TP5_DW_HIP_OVERLAP_JOINED_WINDOW_PCI window=1 pci=0000:03:00.0 spans=2 busy_ns=100 wall_ns=350 busy_frac=0.285714" <<<"$out" && { pass=$((pass+1)); echo "PASS window-values"; } || { fail=$((fail+1)); echo "FAIL window-values :: $out"; }

# B: banner without enabled= (enabled is OPTIONAL) and OLD-format headers
# without errors= (older emitters) must still parse; one aggregate present.
cat > "$T/f2.log" <<'LOG'
TP5_DW_HIP_SPANS flag=GGML_TP5_DW_HIP_SPANS join_map=1 scope=device_stream device_only=1
TP5_DW_HIP_OVERLAP_MAP window=0 native=0xaaa pci_bdf0=0000:03:00.0
TP5_DW_HIP_SPANS provider=0xaaa rank=0 pci=0000:03:00.0 n=1 dropped=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0xaaa rank=0 pci=0000:03:00.0 seq=0 begin_ns=10 end_ns=110
TP5_DW_HIP_OVERLAP provider=0xaaa device=0 rank=0 pci=0000:03:00.0 active=1 spans=1 total_ns=100 max_ns=100 skipped_pending=0 skipped_mode=0 errors=0 coverage=enqueue_to_completion_stream_span clock=hip_event device_only=1
LOG
t banner-enabled-optional 0 "$T/f2.log"

# C: multi-PCI — no cross-device claim anywhere (old-format headers, no banner)
mk3() { cat > "$1" <<'LOG'
TP5_DW_HIP_OVERLAP_MAP window=0 native=0xaaa pci_bdf0=0000:03:00.0 pci_bdf1=0000:04:00.0
TP5_DW_HIP_OVERLAP_MAP window=1 native=0xbbb pci_bdf0=0000:03:00.0 pci_bdf1=0000:05:00.0
TP5_DW_HIP_SPANS provider=0xaaa rank=0 pci=0000:03:00.0 n=1 dropped=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0xaaa rank=0 pci=0000:03:00.0 seq=0 begin_ns=100 end_ns=200
TP5_DW_HIP_SPANS provider=0xaaa rank=1 pci=0000:04:00.0 n=1 dropped=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0xaaa rank=1 pci=0000:04:00.0 seq=0 begin_ns=210 end_ns=310
TP5_DW_HIP_SPANS provider=0xbbb rank=0 pci=0000:03:00.0 n=1 dropped=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0xbbb rank=0 pci=0000:03:00.0 seq=0 begin_ns=150 end_ns=250
TP5_DW_HIP_SPANS provider=0xbbb rank=1 pci=0000:05:00.0 n=1 dropped=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0xbbb rank=1 pci=0000:05:00.0 seq=0 begin_ns=220 end_ns=320
LOG
}
mk3 "$T/f3.log"; t multipci-no-crossclaim 0 "$T/f3.log"
out=$("$J" "$T/f3.log")
grep -q "_PAIR a=0 b=1 pci=0000:04:00.0\|_PAIR a=0 b=1 pci=0000:05:00.0" <<<"$out" && { fail=$((fail+1)); echo "FAIL cross-device pair emitted"; } || { pass=$((pass+1)); echo "PASS no-cross-device-pair"; }
grep -q "_PAIR a=0 b=1 pci=0000:03:00.0 spans_a=1 spans_b=1 overlap_ns=50" <<<"$out" && { pass=$((pass+1)); echo "PASS same-pci-pair"; } || { fail=$((fail+1)); echo "FAIL same-pci-pair :: $out"; }

# D: shared mode — providers unmatched, disjoint pcis, no pairs, not_computable
cat > "$T/f4.log" <<'LOG'
TP5_DW_HIP_SPANS enabled=1 flag=GGML_TP5_DW_HIP_SPANS join_map=1 scope=device_stream device_only=1
TP5_DW_HIP_OVERLAP_MAP window=0 native=0x5555 pci_bdf0=0000:03:00.0
TP5_DW_HIP_OVERLAP_MAP window=1 native=0x6666 pci_bdf0=0000:07:00.0
TP5_DW_HIP_SPANS provider=0x1234 rank=0 pci=0000:03:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x1234 rank=0 pci=0000:03:00.0 seq=0 begin_ns=10 end_ns=110
TP5_DW_HIP_SPANS provider=0x4321 rank=0 pci=0000:07:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x4321 rank=0 pci=0000:07:00.0 seq=0 begin_ns=20 end_ns=120
LOG
t shared-mode 0 "$T/f4.log"
grep -q "pair_scope=not_computable" <("$J" "$T/f4.log") && { pass=$((pass+1)); echo "PASS shared-not-computable"; } || { fail=$((fail+1)); echo "FAIL shared-not-computable"; }

# E: multi-window-same-pci, shared mode — the REAL dual-window TP5 topology:
# both windows legitimately map the SAME physical devices (TP5 runs every
# window on every rank's GPU) and the providers (relay contexts) match no
# MAP key. Hand-verified values:
#   provider 0x111 pci03 spans [100,200],[300,400]  -> busy 200, wall 300
#   provider 0x222 pci03 spans [150,250],[350,450]  -> busy 200, wall 300
#   0x111 x 0x222 on 03: [150,200]+[350,400]        -> overlap 100
#   pci04: 0x111 [500,600] vs 0x222 [700,800]       -> overlap 0
#   pci03 union busy 300, wall 350; pci04 busy 200, wall 300
# Window attribution is UNPROVABLE (no provider= in MAP): no WINDOW_PCI
# rows anywhere, and pairs are labeled by provider index into the PROVIDER
# table — never by window.
cat > "$T/f5.log" <<'LOG'
TP5_DW_HIP_SPANS enabled=1 flag=GGML_TP5_DW_HIP_SPANS join_map=1 scope=device_stream device_only=1
TP5_DW_HIP_OVERLAP_MAP window=0 native=0x5555 pci_bdf0=0000:03:00.0 pci_bdf1=0000:04:00.0
TP5_DW_HIP_OVERLAP_MAP window=1 native=0x6666 pci_bdf0=0000:03:00.0 pci_bdf1=0000:04:00.0
TP5_DW_HIP_SPANS provider=0x111 rank=0 pci=0000:03:00.0 n=2 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x111 rank=0 pci=0000:03:00.0 seq=0 begin_ns=100 end_ns=200
TP5_DW_HIP_SPAN provider=0x111 rank=0 pci=0000:03:00.0 seq=1 begin_ns=300 end_ns=400
TP5_DW_HIP_SPANS provider=0x111 rank=1 pci=0000:04:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x111 rank=1 pci=0000:04:00.0 seq=0 begin_ns=500 end_ns=600
TP5_DW_HIP_SPANS provider=0x222 rank=0 pci=0000:03:00.0 n=2 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x222 rank=0 pci=0000:03:00.0 seq=0 begin_ns=150 end_ns=250
TP5_DW_HIP_SPAN provider=0x222 rank=0 pci=0000:03:00.0 seq=1 begin_ns=350 end_ns=450
TP5_DW_HIP_SPANS provider=0x222 rank=1 pci=0000:04:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x222 rank=1 pci=0000:04:00.0 seq=0 begin_ns=700 end_ns=800
TP5_DW_HIP_OVERLAP provider=0x111 device=0 rank=0 pci=0000:03:00.0 active=1 spans=2 total_ns=300 max_ns=100 skipped_pending=0 skipped_mode=0 errors=0 coverage=enqueue_to_completion_stream_span clock=hip_event device_only=1
TP5_DW_HIP_OVERLAP provider=0x111 device=1 rank=1 pci=0000:04:00.0 active=1 spans=1 total_ns=100 max_ns=100 skipped_pending=0 skipped_mode=0 errors=0 coverage=enqueue_to_completion_stream_span clock=hip_event device_only=1
TP5_DW_HIP_OVERLAP provider=0x222 device=0 rank=0 pci=0000:03:00.0 active=1 spans=2 total_ns=200 max_ns=100 skipped_pending=0 skipped_mode=0 errors=0 coverage=enqueue_to_completion_stream_span clock=hip_event device_only=1
TP5_DW_HIP_OVERLAP provider=0x222 device=1 rank=1 pci=0000:04:00.0 active=1 spans=1 total_ns=100 max_ns=100 skipped_pending=0 skipped_mode=0 errors=0 coverage=enqueue_to_completion_stream_span clock=hip_event device_only=1
LOG
t mwsp-shared 0 "$T/f5.log"
out=$("$J" "$T/f5.log")
grep -q "mode=shared windows=2 pcis=2 pair_scope=cross_provider_unattributed" <<<"$out" && { pass=$((pass+1)); echo "PASS mwsp-shared-mode-scope"; } || { fail=$((fail+1)); echo "FAIL mwsp-shared-mode-scope :: $out"; }
grep -q "TP5_DW_HIP_OVERLAP_JOINED_PROVIDER index=0 provider=0x111 pcis=2" <<<"$out" && grep -q "TP5_DW_HIP_OVERLAP_JOINED_PROVIDER index=1 provider=0x222 pcis=2" <<<"$out" && { pass=$((pass+1)); echo "PASS mwsp-provider-table"; } || { fail=$((fail+1)); echo "FAIL mwsp-provider-table :: $out"; }
grep -q "TP5_DW_HIP_OVERLAP_JOINED_PROVIDER_PCI provider_index=0 pci=0000:03:00.0 spans=2 busy_ns=200 wall_ns=300 busy_frac=0.666667" <<<"$out" && { pass=$((pass+1)); echo "PASS mwsp-provider-pci-values"; } || { fail=$((fail+1)); echo "FAIL mwsp-provider-pci-values :: $out"; }
grep -q "TP5_DW_HIP_OVERLAP_JOINED_PROVIDER_PAIR provider_a=0x111 provider_b=0x222 pci=0000:03:00.0 spans_a=2 spans_b=2 overlap_ns=100" <<<"$out" && { pass=$((pass+1)); echo "PASS mwsp-shared-pair-value"; } || { fail=$((fail+1)); echo "FAIL mwsp-shared-pair-value :: $out"; }
grep -q "TP5_DW_HIP_OVERLAP_JOINED_PROVIDER_PAIR provider_a=0x111 provider_b=0x222 pci=0000:04:00.0 spans_a=1 spans_b=1 overlap_ns=0" <<<"$out" && { pass=$((pass+1)); echo "PASS mwsp-shared-pair-zero-overlap"; } || { fail=$((fail+1)); echo "FAIL mwsp-shared-pair-zero-overlap :: $out"; }
[ -z "$(grep -E "TP5_DW_HIP_OVERLAP_JOINED_PAIR .*overlap_ns=" <<<"$out")" ] && { pass=$((pass+1)); echo "PASS mwsp-shared-no-original-pair-tag"; } || { fail=$((fail+1)); echo "FAIL mwsp-shared-no-original-pair-tag :: $out"; }
grep -q "TP5_DW_HIP_OVERLAP_JOINED_PCI pci=0000:03:00.0 spans=4 busy_ns=300 wall_ns=350 busy_frac=0.857143" <<<"$out" && { pass=$((pass+1)); echo "PASS mwsp-shared-pci-union"; } || { fail=$((fail+1)); echo "FAIL mwsp-shared-pci-union :: $out"; }
[ -z "$(grep "_JOINED_WINDOW_PCI" <<<"$out")" ] && { pass=$((pass+1)); echo "PASS mwsp-shared-no-window-attribution"; } || { fail=$((fail+1)); echo "FAIL mwsp-shared-no-window-attribution :: $out"; }

# F: multi-window-same-pci with PROVEN attribution — the same topology as
# E, but every MAP line carries provider= naming its window's relay
# provider. Values hand-verified as in E; a/b on PAIR rows are WINDOW ids
# again and per-window busy rows are emitted.
sed 's/native=0x5555 pci_bdf0/native=0x5555 provider=0x111 pci_bdf0/; s/native=0x6666 pci_bdf0/native=0x6666 provider=0x222 pci_bdf0/' "$T/f5.log" > "$T/f6.log"
t mwsp-provider-mode 0 "$T/f6.log"
out=$("$J" "$T/f6.log")
grep -q "mode=provider windows=2 pcis=2 pair_scope=same_pci_shared_epoch" <<<"$out" && { pass=$((pass+1)); echo "PASS mwsp-provider-mode-scope"; } || { fail=$((fail+1)); echo "FAIL mwsp-provider-mode-scope :: $out"; }
grep -q "TP5_DW_HIP_OVERLAP_JOINED_PAIR a=0 b=1 pci=0000:03:00.0 spans_a=2 spans_b=2 overlap_ns=100" <<<"$out" && { pass=$((pass+1)); echo "PASS mwsp-provider-window-pair"; } || { fail=$((fail+1)); echo "FAIL mwsp-provider-window-pair :: $out"; }
grep -q "TP5_DW_HIP_OVERLAP_JOINED_WINDOW_PCI window=0 pci=0000:03:00.0 spans=2 busy_ns=200 wall_ns=350 busy_frac=0.571429" <<<"$out" && grep -q "TP5_DW_HIP_OVERLAP_JOINED_WINDOW_PCI window=1 pci=0000:03:00.0 spans=2 busy_ns=200 wall_ns=350 busy_frac=0.571429" <<<"$out" && { pass=$((pass+1)); echo "PASS mwsp-provider-window-busy"; } || { fail=$((fail+1)); echo "FAIL mwsp-provider-window-busy :: $out"; }
grep -q "TP5_DW_HIP_OVERLAP_JOINED_WINDOW_PCI window=0 pci=0000:04:00.0 spans=1 busy_ns=100 wall_ns=300 busy_frac=0.333333" <<<"$out" && { pass=$((pass+1)); echo "PASS mwsp-provider-window-pci04"; } || { fail=$((fail+1)); echo "FAIL mwsp-provider-window-pci04 :: $out"; }
[ -z "$(grep "_JOINED_PROVIDER" <<<"$out")" ] && { pass=$((pass+1)); echo "PASS mwsp-provider-no-provider-table"; } || { fail=$((fail+1)); echo "FAIL mwsp-provider-no-provider-table :: $out"; }

# GATE guard: the ORIGINAL result.py GATE3 patterns — pair rows via
# 'TP5_DW_HIP_OVERLAP_JOINED_PAIR .*overlap_ns=' and first pair_scope ==
# same_pci_shared_epoch — must find NOTHING on the unattributed shared
# output (the gate cannot pass without proven provider->window), and the
# SAME checks must pass on the proven provider-mode output (so this
# negative is not vacuous).
sh=$($J "$T/f5.log"); pf=$($J "$T/f6.log")
sp=$(grep -oE "pair_scope=(\S+)" <<<"$sh" | head -1)
np=$(grep -cE "TP5_DW_HIP_OVERLAP_JOINED_PAIR " <<<"$sh")
if [ "$np" -eq 0 ] && [ "$sp" != "pair_scope=same_pci_shared_epoch" ]; then
  pass=$((pass+1)); echo "PASS mwsp-gate-cannot-pass-shared (scope=$sp, pairs=$np)"
else
  fail=$((fail+1)); echo "FAIL mwsp-gate-cannot-pass-shared :: scope=$sp"
fi
pp=$(grep -oE "pair_scope=(\S+)" <<<"$pf" | head -1)
if [ "$(grep -cE "TP5_DW_HIP_OVERLAP_JOINED_PAIR .*overlap_ns=[0-9]+" <<<"$pf")" -ge 1 ] \
   && [ "$pp" = "pair_scope=same_pci_shared_epoch" ]; then
  pass=$((pass+1)); echo "PASS mwsp-gate-positive-control (scope=$pp)"
else
  fail=$((fail+1)); echo "FAIL mwsp-gate-positive-control :: scope=$pp"
fi

# G: same-PCI shared topology with a SINGLE span-carrying provider (the
# failover shape): no second distinct attributable provider exists, so NO
# pair may be claimed for the pci (fail-closed, no inference).
cat > "$T/f7.log" <<'LOG'
TP5_DW_HIP_SPANS enabled=1 flag=GGML_TP5_DW_HIP_SPANS join_map=1 scope=device_stream device_only=1
TP5_DW_HIP_OVERLAP_MAP window=0 native=0x5555 pci_bdf0=0000:03:00.0
TP5_DW_HIP_OVERLAP_MAP window=1 native=0x6666 pci_bdf0=0000:03:00.0
TP5_DW_HIP_SPANS provider=0x111 rank=0 pci=0000:03:00.0 n=2 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x111 rank=0 pci=0000:03:00.0 seq=0 begin_ns=100 end_ns=200
TP5_DW_HIP_SPAN provider=0x111 rank=0 pci=0000:03:00.0 seq=1 begin_ns=300 end_ns=400
TP5_DW_HIP_OVERLAP provider=0x111 device=0 rank=0 pci=0000:03:00.0 active=1 spans=2 total_ns=200 max_ns=100 skipped_pending=0 skipped_mode=0 errors=0 coverage=enqueue_to_completion_stream_span clock=hip_event device_only=1
LOG
t mwsp-single-provider 0 "$T/f7.log"
out=$("$J" "$T/f7.log")
grep -q "mode=shared windows=2 pcis=1 pair_scope=not_computable" <<<"$out" && { pass=$((pass+1)); echo "PASS mwsp-single-provider-scope"; } || { fail=$((fail+1)); echo "FAIL mwsp-single-provider-scope :: $out"; }
[ -z "$(grep "_JOINED_PAIR" <<<"$out")" ] && { pass=$((pass+1)); echo "PASS mwsp-single-provider-no-pair"; } || { fail=$((fail+1)); echo "FAIL mwsp-single-provider-no-pair :: $out"; }

# negatives (fail-closed)
t neg-dropped 1 <(sed 's/dropped=0/dropped=2/' "$T/f1.log")
# header errors=3 must be REFUSED with the verdict refusal (not a parse
# error, not ignored) — the pass-after marker is "refusing verdict:".
want_msg neg-hdr-errors-refused "refusing verdict: errors=3" <(sed '0,/errors=0/s//errors=3/' "$T/f1.log")
t neg-hdr-errors-badint 1 <(sed 's/errors=0/errors=x/' "$T/f1.log")
# aggregate-path refusal (aggregate line is the only one with "errors=0 coverage=")
t neg-agg-errors 1 <(sed 's/errors=0 coverage/errors=3 coverage/' "$T/f1.log")
t neg-reversed 1 <(sed 's/seq=0 begin_ns=100 end_ns=150/seq=0 begin_ns=150 end_ns=100/' "$T/f1.log")
t neg-bad-int 1 <(sed 's/seq=0 begin_ns=100/seq=x begin_ns=100/' "$T/f1.log")
t neg-unknown-family 1 <(sed '1a TP5_DW_HIP_UNKNOWN x=1' "$T/f1.log")
t neg-no-map 1 <(grep -v OVERLAP_MAP "$T/f1.log")
t neg-count-mismatch 1 <(sed '0,/spans=2 total/s//spans=3 total/' "$T/f1.log")
t neg-shared-epoch 1 <(sed 's/provider=0xbbb rank=0 pci=0000:03:00.0 n=2 dropped=0 errors=0 shared_epoch=1/provider=0xbbb rank=0 pci=0000:03:00.0 n=2 dropped=0 errors=0 shared_epoch=0/' "$T/f1.log")
# enabled= is accepted ONLY on the banner form: span body, map line and
# aggregate stay strict; banner values are validated; bare tag rejected.
t neg-span-enabled 1 <(sed 's/seq=1 begin_ns=200 end_ns=250/seq=1 begin_ns=200 end_ns=250 enabled=1/' "$T/f1.log")
t neg-map-enabled 1 <(sed 's/native=0xbbb pci_bdf0/native=0xbbb enabled=1 pci_bdf0/' "$T/f1.log")
t neg-agg-enabled 1 <(sed 's/device=0 rank=0/device=0 rank=0 enabled=1/' "$T/f1.log")
t neg-banner-unknown-key 1 <(sed 's/join_map=1/join_map=1 foo=1/' "$T/f1.log")
t neg-banner-bad-enabled 1 <(sed 's/enabled=1/enabled=2/' "$T/f1.log")
t neg-banner-bad-flag 1 <(sed 's/flag=GGML_TP5_DW_HIP_SPANS/flag=GGML_TP5_DW_OVERLAP_METER/' "$T/f1.log")
t neg-banner-no-keys 1 <(sed '0,/TP5_DW_HIP_SPANS enabled=1.*/s//TP5_DW_HIP_SPANS/' "$T/f1.log")
# mwsp fail-closed: mixed MAP forms (provider= on one line only).
want_msg mwsp-mixed-map-refused "mixed TP5_DW_HIP_OVERLAP_MAP forms" <(sed '0,/native=0x5555 pci_bdf0/s//native=0x5555 provider=0x111 pci_bdf0/' "$T/f5.log")
# mwsp fail-closed: a provider matching one window's provider= AND another
# window's native= is ambiguous attribution.
cat > "$T/f8.log" <<'LOG'
TP5_DW_HIP_OVERLAP_MAP window=0 native=0x5555 provider=0x111 pci_bdf0=0000:03:00.0
TP5_DW_HIP_OVERLAP_MAP window=1 native=0x111 provider=0x222 pci_bdf0=0000:03:00.0
TP5_DW_HIP_SPANS provider=0x111 rank=0 pci=0000:03:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x111 rank=0 pci=0000:03:00.0 seq=0 begin_ns=100 end_ns=200
TP5_DW_HIP_SPANS provider=0x222 rank=0 pci=0000:03:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x222 rank=0 pci=0000:03:00.0 seq=0 begin_ns=150 end_ns=250
TP5_DW_HIP_OVERLAP provider=0x111 device=0 rank=0 pci=0000:03:00.0 active=1 spans=1 total_ns=100 max_ns=100 skipped_pending=0 skipped_mode=0 errors=0 coverage=enqueue_to_completion_stream_span clock=hip_event device_only=1
TP5_DW_HIP_OVERLAP provider=0x222 device=0 rank=0 pci=0000:03:00.0 active=1 spans=1 total_ns=100 max_ns=100 skipped_pending=0 skipped_mode=0 errors=0 coverage=enqueue_to_completion_stream_span clock=hip_event device_only=1
LOG
want_msg mwsp-ambiguous-refused "ambiguous window attribution" "$T/f8.log"

want_msg mwsp-seq-gap-refused "seq gap" <(sed 's/seq=1 begin_ns=350/seq=3 begin_ns=350/' "$T/f5.log")
want_msg mwsp-shared-epoch0-refused "shared_epoch!=1" <(sed '/provider=0x222 rank=0 pci=0000:03:00.0 n=2/s/shared_epoch=1/shared_epoch=0/' "$T/f5.log")
# H: provider mapping identity chain — missing / duplicate / conflict are
# all fail-closed, and provider= is authoritative (native= is never used
# as a fallback once the MAP carries provider=). Values hand-verified:
#   w0 provider 0xabc span [100,200]; w1 provider 0x222 span [150,250]
#   same pci 03 -> pair overlap 50; canonical 0xABC == 0xabc.
cat > "$T/f9.log" <<'LOG'
TP5_DW_HIP_OVERLAP_MAP window=0 native=0x5555 provider=0xABC pci_bdf0=0000:03:00.0
TP5_DW_HIP_OVERLAP_MAP window=1 native=0x6666 provider=0x222 pci_bdf0=0000:03:00.0
TP5_DW_HIP_SPANS provider=0xabc rank=0 pci=0000:03:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0xabc rank=0 pci=0000:03:00.0 seq=0 begin_ns=100 end_ns=200
TP5_DW_HIP_SPANS provider=0x222 rank=0 pci=0000:03:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x222 rank=0 pci=0000:03:00.0 seq=0 begin_ns=150 end_ns=250
LOG
t mwsp-provider-canonical 0 "$T/f9.log"
out=$("$J" "$T/f9.log")
grep -q "mode=provider windows=2 pcis=1 pair_scope=same_pci_shared_epoch" <<<"$out" && { pass=$((pass+1)); echo "PASS mwsp-provider-canonical-mode"; } || { fail=$((fail+1)); echo "FAIL mwsp-provider-canonical-mode :: $out"; }
grep -q "TP5_DW_HIP_OVERLAP_JOINED_PAIR a=0 b=1 pci=0000:03:00.0 spans_a=1 spans_b=1 overlap_ns=50" <<<"$out" && { pass=$((pass+1)); echo "PASS mwsp-provider-canonical-pair"; } || { fail=$((fail+1)); echo "FAIL mwsp-provider-canonical-pair :: $out"; }

# H1 missing: a span provider with no provider= entry while the MAP form
# carries provider= is an incomplete identity chain — refused, not shared.
cat > "$T/f10.log" <<'LOG'
TP5_DW_HIP_OVERLAP_MAP window=0 native=0x5555 provider=0x111 pci_bdf0=0000:03:00.0
TP5_DW_HIP_OVERLAP_MAP window=1 native=0x6666 provider=0x222 pci_bdf0=0000:03:00.0
TP5_DW_HIP_SPANS provider=0x111 rank=0 pci=0000:03:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x111 rank=0 pci=0000:03:00.0 seq=0 begin_ns=100 end_ns=200
TP5_DW_HIP_SPANS provider=0x999 rank=0 pci=0000:03:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x999 rank=0 pci=0000:03:00.0 seq=0 begin_ns=150 end_ns=250
LOG
want_msg mwsp-provider-missing-refused "incomplete provider mapping" "$T/f10.log"
[ -z "$("$J" "$T/f10.log" 2>/dev/null)" ] && { pass=$((pass+1)); echo "PASS mwsp-provider-missing-no-stdout"; } || { fail=$((fail+1)); echo "FAIL mwsp-provider-missing-no-stdout"; }

# H2 missing: provider= is authoritative — a span provider matching only a
# MAP native= pointer does NOT silently re-enable pointer attribution.
cat > "$T/f11.log" <<'LOG'
TP5_DW_HIP_OVERLAP_MAP window=0 native=0x5555 provider=0x111 pci_bdf0=0000:03:00.0
TP5_DW_HIP_OVERLAP_MAP window=1 native=0x6666 provider=0x222 pci_bdf0=0000:03:00.0
TP5_DW_HIP_SPANS provider=0x5555 rank=0 pci=0000:03:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x5555 rank=0 pci=0000:03:00.0 seq=0 begin_ns=100 end_ns=200
TP5_DW_HIP_SPANS provider=0x6666 rank=0 pci=0000:03:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x6666 rank=0 pci=0000:03:00.0 seq=0 begin_ns=150 end_ns=250
LOG
want_msg mwsp-provider-native-fallback-refused "incomplete provider mapping" "$T/f11.log"

# H3 duplicate: two windows claiming the same canonical provider pointer.
cat > "$T/f12.log" <<'LOG'
TP5_DW_HIP_OVERLAP_MAP window=0 native=0x5555 provider=0x111 pci_bdf0=0000:03:00.0
TP5_DW_HIP_OVERLAP_MAP window=1 native=0x6666 provider=0x111 pci_bdf0=0000:03:00.0
TP5_DW_HIP_SPANS provider=0x111 rank=0 pci=0000:03:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x111 rank=0 pci=0000:03:00.0 seq=0 begin_ns=100 end_ns=200
LOG
want_msg mwsp-duplicate-provider-refused "duplicate provider mapping" "$T/f12.log"
# H3b duplicate: canonical pointer text collides across 0xABC / 0xabc —
# the mapping is the same provider pointer, so two windows still refuse.
cat > "$T/f14.log" <<'LOG'
TP5_DW_HIP_OVERLAP_MAP window=0 native=0x5555 provider=0xABC pci_bdf0=0000:03:00.0
TP5_DW_HIP_OVERLAP_MAP window=1 native=0x6666 provider=0xabc pci_bdf0=0000:03:00.0
TP5_DW_HIP_SPANS provider=0xabc rank=0 pci=0000:03:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0xabc rank=0 pci=0000:03:00.0 seq=0 begin_ns=100 end_ns=200
LOG
want_msg mwsp-duplicate-provider-canonical-refused "duplicate provider mapping" "$T/f14.log"

# H4 duplicate: provider= twice on one MAP line is a parse error.
want_msg mwsp-duplicate-provider-key-refused "duplicate key in TP5_DW_HIP_OVERLAP_MAP: provider" <(sed '0,/native=0x5555 provider=0xABC pci_bdf0/s//native=0x5555 provider=0xABC provider=0xABC pci_bdf0/' "$T/f9.log")

# H5 missing value: a null provider pointer cannot prove a window link.
want_msg mwsp-null-provider-refused "null provider pointer" <(sed 's/provider=0x222/provider=(nil)/' "$T/f9.log")

# H6 conflict: provider->window and provider->pci disagree (the span's pci
# is not in the attributed window's mapped set) — refused.
cat > "$T/f13.log" <<'LOG'
TP5_DW_HIP_OVERLAP_MAP window=0 native=0x5555 provider=0x111 pci_bdf0=0000:03:00.0
TP5_DW_HIP_OVERLAP_MAP window=1 native=0x6666 provider=0x222 pci_bdf0=0000:03:00.0
TP5_DW_HIP_SPANS provider=0x111 rank=0 pci=0000:04:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x111 rank=0 pci=0000:04:00.0 seq=0 begin_ns=100 end_ns=200
TP5_DW_HIP_SPANS provider=0x222 rank=0 pci=0000:03:00.0 n=1 dropped=0 errors=0 shared_epoch=1
TP5_DW_HIP_SPAN provider=0x222 rank=0 pci=0000:03:00.0 seq=0 begin_ns=150 end_ns=250
LOG
want_msg mwsp-provider-pci-conflict-refused "not in window 0's mapped pci set" "$T/f13.log"

echo "pass=$pass fail=$fail"; [ "$fail" = 0 ]
