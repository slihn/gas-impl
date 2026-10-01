#!/usr/bin/env bash
# my tests are written in from .<module> import fn1, fn2...
# this does not work out of box in gas-impl, where you would do
# pytest tests/<files>
#
# Run ./pytest_wrapper.sh -h for usage.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: pytest_wrapper.sh [TEST_FILE[::NODE] ...] [PYTEST_OPTIONS ...]

Run gas-impl's tests/ without changing the repo. The tests use relative imports
(from .gas_dist import ...), so this builds a throwaway package in a temp dir:
gas_impl/ made of symlinks to the real gas_impl/*.py plus tests/*.py. Each test is
imported as gas_impl.test_x, so the relative imports resolve to the real modules.
Bytecode lands in the temp dir, and the temp dir is removed on exit.

Examples:
  ./pytest_wrapper.sh                                  every test file in tests/
  ./pytest_wrapper.sh tests/test_gsas.py               one file (tests/ prefix optional)
  ./pytest_wrapper.sh test_gas.py::TestGSaS_PdfAtZero  one class or test
  ./pytest_wrapper.sh test_gsas.py -k PDF0 -x          other pytest options pass through
  PYTHON=~/.venv3.11/bin/python ./pytest_wrapper.sh    pick the interpreter (default: python)

Options:
  -h, --help        show this help
  --pytest-help     show pytest's own help (pytest -h)

Notes:
  gas_impl imports adp_tf, so the parent folder of gas-impl and adp_tf goes on PYTHONPATH.
  On a name clash (test_fcm_t.py is in both gas_impl/ and tests/) the tests/ copy wins.
  The exit code is pytest's; 2 also means an unknown test file was named.
EOF
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PYTHON="${PYTHON:-python}"

for a in "$@"; do
    case "$a" in
        -h|--help)     usage; exit 0 ;;
        --pytest-help) exec "$PYTHON" -m pytest -h ;;
    esac
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/gas_impl_pytest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
mkdir "$WORK/gas_impl"
for f in "$HERE"/gas_impl/*.py; do ln -s "$f" "$WORK/gas_impl/"; done
for f in "$HERE"/tests/*.py;    do ln -sf "$f" "$WORK/gas_impl/"; done

# map file arguments (tests/x.py, x.py, x.py::Node) onto the temp package; pass the rest through
args=()
have_file=0
for a in "$@"; do
    case "$a" in
        *.py|*.py::*)
            name="${a##*/}"
            [ -e "$WORK/gas_impl/${name%%::*}" ] || { echo "no such test file: $a" >&2; exit 2; }
            args+=("$WORK/gas_impl/$name"); have_file=1 ;;
        *)  args+=("$a") ;;
    esac
done
if [ "$have_file" -eq 0 ]; then
    for f in "$HERE"/tests/test_*.py; do args+=("$WORK/gas_impl/$(basename "$f")"); done
fi

cd "$WORK"
rc=0
PYTHONPATH="$WORK:$ROOT/adp_tf:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m pytest -p no:cacheprovider --rootdir="$WORK" "${args[@]}" || rc=$?
exit "$rc"
