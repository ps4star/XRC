#!/bin/sh
# Checks xrc's integer SIMD translations against the CPU: builds simd.c as a static i386 Linux program, translates it,
# compiles the translation with the linux-i386 example's runtime, and compares the two runs' output. Needs clang with
# the i386 target, and an x86 machine that runs 32-bit programs for the comparison.
cd "$(dirname "$0")" || exit 1
CC=${CC:-cc}
XRC=${XRC:-../../xrc}
[ -x "$XRC" ] || XRC=../../xrc.exe
[ -x "$XRC" ] || { echo "run.sh: no xrc (build it first: ../../build.sh, or set XRC)"; exit 1; }
B=build
mkdir -p $B
xp() { case "$XRC" in *.exe) command -v wslpath >/dev/null && { wslpath -m "$1"; return; } ;; esac; echo "$1"; }
clang --target=i386-linux-gnu -O1 -msse2 -mssse3 -ffreestanding -nostdlib -fno-stack-protector -fno-builtin -static \
	-fno-pic simd.c -o $B/simd || exit 1
rm -rf $B/out
XRC_NO_SLOTS=1 "$XRC" --exe "$(xp $B/simd)" --all --out "$(xp $B/out)" --table "$(xp $B/out)/xrc_table.c" \
	--report "$(xp $B/out)/coverage.tsv" | grep "functions  "
$CC -O2 -fno-strict-aliasing -fwrapv -ffp-contract=off -DXRC_NO_FIXED_BASE -w -I ../../include -I ../linux-i386/runtime \
	$B/out/*.c ../linux-i386/runtime/runtime.c -o $B/simd_xrc || exit 1
./$B/simd_xrc $B/simd > $B/xrc.txt
./$B/simd > $B/native.txt || { echo "run.sh: the i386 program does not run here"; exit 1; }
echo "$(wc -l < $B/native.txt) results"
if cmp -s $B/native.txt $B/xrc.txt; then echo "-- the same as the program run natively"; exit 0; fi
echo "-- DIFFERENT from the native run:"
diff $B/native.txt $B/xrc.txt | head -40
exit 1
