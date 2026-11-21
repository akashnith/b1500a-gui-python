#!/usr/bin/env sh
# Forward any CLI command, e.g.:  ./b1500.sh smu list
cd "$(dirname "$0")/.." || exit 1
exec python3 -m b1500_studio "$@"
