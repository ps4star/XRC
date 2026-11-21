#!/bin/sh
# Builds demo.c as a Linux i386 program four ways, translates each with xrc, compiles the translation for this
# machine with runtime/, runs it, and compares its output with the program run natively (where this machine can run
# i386 code). Needs clang (with the i386 target and GNU ld or lld) and a C compiler for this machine (CC, default cc).
#
#   sh run.sh            all four: static, pie, dyn (a dynamic executable with lib.c as its shared library) and
#                        piedyn (the same as a PIE: its library calls go through the PLT with the GOT in ebx)
#   sh run.sh pie        one of them
#
# XRC: the translator (default: ../../xrc, or ../../xrc.exe -- the Windows build, from WSL). Output: build/.
cd "$(dirname "$0")" || exit 1
CC=${CC:-cc}
XRC=${XRC:-../../xrc}
[ -x "$XRC" ] || XRC=../../xrc.exe
[ -x "$XRC" ] || { echo "run.sh: no xrc (build it first: ../../build.sh, or set XRC)"; exit 1; }
B=build
mkdir -p $B
# a path for xrc: the Windows build under WSL takes Windows paths
xp() { case "$XRC" in *.exe) command -v wslpath >/dev/null && { wslpath -m "$1"; return; } ;; esac; echo "$1"; }
I386="--target=i386-linux-gnu -O2 -ffreestanding -nostdlib -fno-stack-protector -fno-builtin"
fail=0
for v in ${@:-static pie dyn piedyn}; do
	echo "== $v"
	case $v in
	static) clang $I386 -static -fno-pic demo.c -o $B/demo_$v || exit 1 ;;
	pie) clang $I386 -fPIE -pie demo.c -o $B/demo_$v || exit 1 ;;
	dyn)
		clang $I386 -fPIC -shared lib.c -o $B/libdemo.so || exit 1
		clang $I386 -fno-pic -no-pie -DWITH_LIB demo.c -L$B -ldemo -Wl,-rpath,"$PWD/$B" \
			-Wl,--dynamic-linker=/lib/ld-linux.so.2 -o $B/demo_$v 2>/dev/null || exit 1 ;;
	piedyn)
		clang $I386 -fPIC -shared lib.c -o $B/libdemo.so || exit 1
		clang $I386 -fPIE -pie -DWITH_LIB demo.c -L$B -ldemo -Wl,-rpath,"$PWD/$B" \
			-Wl,--dynamic-linker=/lib/ld-linux.so.2 -o $B/demo_$v 2>/dev/null || exit 1 ;;
	*) echo "run.sh: $v? (static, pie, dyn or piedyn)"; exit 2 ;;
	esac
	rm -rf $B/out_$v
	XRC_NO_SLOTS=1 "$XRC" --exe "$(xp $B/demo_$v)" --all --out "$(xp $B/out_$v)" --table "$(xp $B/out_$v)/xrc_table.c" \
		--report "$(xp $B/out_$v)/coverage.tsv" | grep "functions  "
	$CC -O2 -fno-strict-aliasing -fwrapv -ffp-contract=off -DXRC_NO_FIXED_BASE -w -I ../../include -I runtime \
		$B/out_$v/*.c runtime/runtime.c -o $B/demo_${v}_xrc || exit 1
	./$B/demo_${v}_xrc $B/demo_$v > $B/xrc_$v.txt
	cat $B/xrc_$v.txt
	if ./$B/demo_$v > $B/native_$v.txt 2>/dev/null; then
		if cmp -s $B/native_$v.txt $B/xrc_$v.txt; then echo "-- the same as the program run natively"; else echo "-- DIFFERENT from the native run:"; diff $B/native_$v.txt $B/xrc_$v.txt; fail=1; fi
	else
		echo "-- (this machine cannot run the i386 program itself to compare)"
	fi
done
exit $fail
