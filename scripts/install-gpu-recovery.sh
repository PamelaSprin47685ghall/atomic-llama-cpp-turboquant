#!/usr/bin/env bash
# Install reviewed recovery scripts without changing the existing systemd unit.
# Invoked by the sudo-authorized scripts/reset-gpu-pci.sh install-watchdog path.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo 'GPU recovery installation requires root.' >&2
    exit 1
fi

repo=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
source_helper="$repo/scripts/gpu-hard-unlock.sh"
source_watchdog="$repo/scripts/eagle-gpu-watchdog.sh"
installed_helper=/usr/local/bin/gpu-hard-unlock.sh
installed_watchdog=/usr/local/bin/eagle-gpu-watchdog.sh
unit=/etc/systemd/system/eagle-gpu-watchdog.service

for source in "$source_helper" "$source_watchdog"; do
    [ -r "$source" ] || { echo "Missing recovery source: $source" >&2; exit 1; }
    bash -n "$source"
done
[ -r "$unit" ] || { echo "Missing watchdog unit: $unit" >&2; exit 1; }

backup_dir="/var/backups/eagle-gpu-recovery-$(date '+%Y%m%d-%H%M%S')"
install -d -m 0700 "$backup_dir"
for installed in "$installed_helper" "$installed_watchdog"; do
    if [ -e "$installed" ]; then
        install -m 0600 "$installed" "$backup_dir/$(basename "$installed")"
    fi
done

stage_helper="${installed_helper}.new.$$"
stage_watchdog="${installed_watchdog}.new.$$"
trap 'rm -f "$stage_helper" "$stage_watchdog"' EXIT
install -o root -g root -m 0755 "$source_helper" "$stage_helper"
install -o root -g root -m 0755 "$source_watchdog" "$stage_watchdog"
cmp -s "$source_helper" "$stage_helper"
cmp -s "$source_watchdog" "$stage_watchdog"

# Replace the helper first: any old watchdog still running now calls safer code.
mv -f "$stage_helper" "$installed_helper"
mv -f "$stage_watchdog" "$installed_watchdog"
trap - EXIT

# The existing service remains enabled. Restart only after both copies are complete.
systemctl restart eagle-gpu-watchdog.service
systemctl is-active --quiet eagle-gpu-watchdog.service
cmp -s "$source_helper" "$installed_helper"
cmp -s "$source_watchdog" "$installed_watchdog"
printf 'Eagle watchdog active; root-owned recovery scripts match source. Previous copies: %s\n' "$backup_dir"
