#!/usr/bin/env sh
cd "$(dirname "$0")/.." || exit 1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 exec python3 -m pytest "$@"
