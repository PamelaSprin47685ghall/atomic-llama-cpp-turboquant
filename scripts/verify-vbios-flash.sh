#!/bin/bash
# Post-reboot verification for the 2026-09-28 VBIOS flash of 0000:09:00.0.
# Written and verified by hand before the reboot; run it after the reboot (or a full power cycle).
#
# Pass criteria:
#   1. ATOM BIOS for 09:00 is 113-1MS21XL203W_210810 (the retail image from 06:00), not 113-1N21XLMIN203W_210810
#   2. the 1 s ALU benchmark puts every card within 85 % of the peer median (no card at ~0.5)
#   3. pp_dpm_sclk on 09:00 tops out at 2475Mhz - INFORMATIONAL ONLY: the mining ROM advertised 1200
#      while the card was healthy, so this value alone never proves or disproves anything.
set -u
ALU=/usr/local/bin/tp5-alu1s
SPV=/usr/local/share/tp5/tp5-alu1s.spv
BDF=0000:09:00.0

echo "=== 1. ATOM BIOS as the kernel sees it"
sudo dmesg 2>/dev/null | grep -oE "amdgpu $BDF: \[drm\] ATOM BIOS: [0-9A-Za-z_-]+" | tail -1 || echo "  (need root for dmesg)"
echo "    expected: 113-1MS21XL203W_210810   (was 113-1N21XLMIN203W_210810 before the flash)"

echo "=== 2. sustained ALU benchmark per card (the only instrument that sees a half-rate card)"
if [ -x "$ALU" ] && [ -r "$SPV" ]; then
    timeout 180 "$ALU" "$SPV" 2>/dev/null | sed -nE 's/.*"bdf":"([^"]+)".*"fp32_tflops":([0-9.]+).*/\1 \2/p' | \
    awk '{printf "    %-16s %7.2f TFLOPS\n", $1, $2; v[NR]=$2}
         END{n=asort(v); med=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
             printf "    peer median = %.2f TFLOPS\n", med}'
    echo "    (any card below ~0.85x the median is still faulty)"
else
    echo "    ALU tool missing at $ALU / $SPV"
fi

echo "=== 3. pp_dpm_sclk (informational; this table has been observed to lie)"
cat /sys/bus/pci/devices/$BDF/pp_dpm_sclk 2>/dev/null | sed 's/^/    /'

echo "=== 4. what the watchdog decided by itself (it audits only while the box is idle)"
journalctl -u eagle-gpu-watchdog -n 40 --no-pager 2>/dev/null | grep -E "Capability" | tail -6 | sed 's/^/    /'

echo
echo "If (1) shows the retail part number and (2) shows no card near half the median, the flash is done."
echo "If a card is still near half, the ROM was not the cause and it is a hardware fault."
