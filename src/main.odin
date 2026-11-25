package xrc

// xrc: a static recompiler from 32-bit x86 executables (PE, ELF, Mach-O, raw images) to C. See README.md for everything it does and every option.
//
//   xrc --exe FILE [--raw ADDR | --base ADDR] [--override NAME | --data DIR] (--all | --list FILE) [--out DIR] [--report FILE] [--table FILE]
//       [--matched FILE] [--llvm]
//
// main.odin reads the options that say where things are (--exe, --raw, --override, --data) and hands the rest to cmd_xrc
// (xrc.odin), which does the work.

import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:strings"

EXE_PATH := "" // --exe

usage :: proc() {
	fmt.eprintln(`usage: xrc --exe FILE [--raw ADDR | --base ADDR] [--override NAME | --data DIR] (--all | --list FILE) [--out DIR] [options]

  --exe FILE        the 32-bit x86 executable to translate: a PE (.exe, .dll), an ELF (Linux, BSD), a Mach-O
                    (macOS, also the i386 slice of a fat file), or a flat image with --raw. xrc reads it as it is on
                    disk (a packed or encrypted one must be unpacked first)
  --raw ADDR        --exe is a flat image: all of it code and data, loaded at ADDR (hex), its entry point ADDR
  --base ADDR       where a PIE or shared object (ELF, linked at 0) is placed: its addresses + ADDR (hex; default
                    56555000, where Linux puts a 32-bit PIE). The runtime loads it there too
  --override NAME   the game's folder: overrides/NAME/ beside xrc, or in the current directory, or NAME itself if it
                    is a folder. It has the tables (data/), xrc.cfg, overrides.tsv and the hand-written .c / .h files,
                    which are copied to <out>/overrides/
  --data DIR        the tables (function_map.csv, ...) somewhere else (default: the game folder's data/). The
                    function map may be left out when the executable has symbols: they are the map then
  --all             translate every function of the map (and entry points found from calls and references)
  --list FILE       translate only the functions listed (a hex address first on each line)
  --out DIR         write page_XXXX.c there (one per 64 KB page; unchanged pages are not rewritten).
                    Without it nothing is written but the report
  --report FILE     a TSV of every function: address, status, instruction count, why it was not translated
  --table FILE      the dispatch table (xrc_table.c): every translation and override by guest address
  --matched FILE    functions (hex addresses) replaced by other code: reached only through the dispatcher
  --llvm            also write page_XXXX.ll: the functions the LLVM IR backend can do, in IR (experimental)
  -h, --help        this

Environment variables (XRC_INLINE, XRC_DEVIRT, XRC_TIME, ...): see README.md.`)
}

// the folder xrc's executable is in (not os.args[0]: started from WSL that is a Linux path)
exe_dir :: proc() -> string {
	if d, err := os.get_executable_directory(context.allocator); err == nil {return d}
	return filepath.dir(os.args[0])
}

main :: proc() {
	args := os.args[1:]
	override_name, data_dir := "", ""
	rest := make([dynamic]string)
	need :: proc(args: []string, i: int) {
		if i + 1 < len(args) {return}
		fmt.eprintf("xrc: %s needs a value\n", args[i])
		os.exit(2)
	}
	for i := 0; i < len(args); i += 1 {
		switch args[i] {
		case "--exe":
			need(args, i)
			i += 1
			EXE_PATH = args[i]
		case "--override":
			need(args, i)
			i += 1
			override_name = args[i]
		case "--data":
			need(args, i)
			i += 1
			data_dir = args[i]
		case "--base":
			need(args, i)
			i += 1
			v, ok := hex_u32(args[i])
			if !ok {fmt.eprintf("xrc: --base needs a hex address, not %s\n", args[i]); os.exit(2)}
			BASE, BASE_SET = v, true
		case "--raw":
			need(args, i)
			i += 1
			v, ok := hex_u32(args[i])
			if !ok {fmt.eprintf("xrc: --raw needs a hex load address, not %s\n", args[i]); os.exit(2)}
			RAW_BASE, RAW_SET = v, true
		case "-h", "--help":
			usage()
			os.exit(0)
		case:
			if strings.has_prefix(args[i], "--override=") {
				override_name = strings.trim(args[i][len("--override="):], "\"")
				continue
			}
			if strings.has_prefix(args[i], "--data=") {
				data_dir = strings.trim(args[i][len("--data="):], "\"")
				continue
			}
			append(&rest, args[i])
		}
	}
	if len(rest) == 0 {usage(); os.exit(2)}
	if override_name != "" {
		OVERRIDE_DIR = override_find(override_name)
		if OVERRIDE_DIR == "" {
			fmt.eprintf("xrc: no overrides/%s folder beside xrc or in the current directory\n", override_name)
			os.exit(2)
		}
		cfg, _ := filepath.join({OVERRIDE_DIR, "xrc.cfg"})
		if !config_load(cfg) {os.exit(2)}
		if data_dir == "" {data_dir, _ = filepath.join({OVERRIDE_DIR, "data"})}
	}
	DATA_DIR = data_dir
	os.exit(cmd_xrc(rest[:]))
}
