package xrc

// --override=NAME: a game's folder, overrides/NAME/ (beside xrc, else in the current directory, else NAME as a
// path). Besides the tables (data/) and xrc.cfg (config.odin) it holds the hand-written override routines. Its overrides.tsv says which guest functions they cover (read by cmd_xrc: manual ones are not
// translated, fast ones are called as xo_<addr>, wrapped ones are reached only through the dispatcher), and every
// .c / .h file there is copied next to the pages, to <out>/overrides/, for the game's build to compile with the
// runtime. A file is rewritten only when its contents changed (so a build recompiles only what did), and a file that
// is no longer in overrides/NAME/ is removed from <out>/overrides/. Without --override no function is overridden.

import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:strings"

OVERRIDE_DIR := "" // overrides/NAME (main.odin)

// overrides/NAME: beside the program, else in the current directory, else NAME itself if it is a folder ("" when none)
override_find :: proc(name: string) -> string {
	for base in ([]string{exe_dir(), "."}) {
		d, _ := filepath.join({base, "overrides", name})
		if os.is_dir(d) {return d}
	}
	if os.is_dir(name) {return name}
	return ""
}

// overrides.tsv of the chosen overrides (cmd_xrc)
override_table :: proc() -> (string, bool) {
	if OVERRIDE_DIR == "" {return "", false}
	p, _ := filepath.join({OVERRIDE_DIR, "overrides.tsv"}, context.temp_allocator)
	return read_text(p)
}

// the routines to <out>/overrides/
override_copy :: proc(out: string) -> bool {
	if OVERRIDE_DIR == "" {return true}
	dst, _ := filepath.join({out, "overrides"})
	_ = os.make_directory_all(dst)
	list, err := os.read_all_directory_by_path(OVERRIDE_DIR, context.temp_allocator)
	if err != nil {
		fmt.eprintf("xrc: cannot read %s\n", OVERRIDE_DIR)
		return false
	}
	is_src :: proc(name: string) -> bool {return strings.has_suffix(name, ".c") || strings.has_suffix(name, ".h")}
	keep := make(map[string]bool, context.temp_allocator)
	n, changed := 0, 0
	for e in list {
		if e.type == .Directory || !is_src(e.name) {continue}
		keep[e.name] = true
		data, ok := read_file(e.fullpath)
		if !ok {
			fmt.eprintf("xrc: cannot read %s\n", e.fullpath)
			return false
		}
		n += 1
		to, _ := filepath.join({dst, e.name}, context.temp_allocator)
		if old, okr := read_file(to); okr && string(old) == string(data) {continue}
		if werr := os.write_entire_file(to, data); werr != nil {
			fmt.eprintf("xrc: cannot write %s\n", to)
			return false
		}
		changed += 1
	}
	if old, oerr := os.read_all_directory_by_path(dst, context.temp_allocator); oerr == nil {
		for e in old {
			if e.type != .Directory && is_src(e.name) && !(e.name in keep) {_ = os.remove(e.fullpath)}
		}
	}
	fmt.printf("xrc: %d override files from %s in %s, %d changed\n", n, OVERRIDE_DIR, dst, changed)
	return true
}
