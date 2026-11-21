#!/bin/sh
# Builds xrc (the x86 -> C translator) from src/ with Odin on Linux or macOS. Needs an Odin compiler with the i386
# fixes of patches/odin-rexcode-i386.patch applied (see README.md); ODIN: its odin (default: odin on PATH).
# The translator recurses deeply on large functions: it is linked with a 16 MB stack where the linker allows it,
# and run it with `ulimit -s 16384` (or more) if it crashes on a big binary.
cd "$(dirname "$0")" || exit 1
ODIN=${ODIN:-odin}
case "$(uname -s)" in
Darwin) STACK="-Wl,-stack_size,0x1000000" ;;
*) STACK="-Wl,-z,stacksize=16777216" ;;
esac
exec "$ODIN" build src -out:xrc -o:speed "-extra-linker-flags:$STACK" "$@"
