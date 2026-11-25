package xrc

// xrc.cfg, in a game's folder (overrides/NAME/xrc.cfg): the facts about that binary which xrc cannot work out from
// the code itself. One entry a line, `#` starts a comment, addresses in hex:
//
//   frame_prolog ADDR             a compiler frame-setup helper (MSVC's __SEH_prolog4, __EH_prolog3, ...): it pushes
//                                 ebx esi edi and the caller's ebp for the function that calls it, which then writes
//                                 them freely -- taken as that function's own prologue
//   frame_prolog ADDR EBP         ... and where it leaves ebp: esp at the call + EBP (decimal; __SEH_prolog4 4,
//                                 __EH_prolog3 0) -- the stack balance then follows the function through it (esp
//                                 unknown after it, ebp known; an epilog: esp = ebp + 4, as leave)
//   frame_epilog ADDR             its counterpart (__SEH_epilog4, __EH_epilog3): restores all four
//   x87_helper ADDR TAKES GIVES   a function with an x87 interface the flow analysis cannot derive (its x87 paths are
//                                 dynamic, e.g. the CRT's __CIsin): how many x87 values it takes from the stack and how
//                                 many it leaves there
//   patched ADDR [LEN]            read-only data the runtime writes (LEN bytes, default 4): not folded into the
//                                 code as a constant (x_rdata_fold)
//   patch ADDR HEX...             the image's bytes at ADDR replaced before anything is decoded (a mod's hooks:
//                                 hex bytes, spaces between them allowed). Within a section's file bytes they are
//                                 written there; anywhere else they are new code (what a hook jumps to), a section
//                                 of its own (data_patch.odin)
//
// Without the file (or without --override) there are none of these: every call is a plain call.

import "core:fmt"
import "core:strconv"
import "core:strings"

cfg_prologs: map[u32]bool
cfg_epilogs: map[u32]bool
cfg_prolog_ebp: map[u32]i32 // (frame_prolog ADDR EBP: ebp after it, from esp at the call)
cfg_x87: [dynamic][3]u32 // address, values taken, values returned
cfg_patches: [dynamic]Cfg_Patch

Cfg_Patch :: struct {addr: u32, bytes: []u8, line: int}

config_load :: proc(path: string) -> bool {
	txt, ok := read_text(path)
	if !ok {return true} // (optional)
	for raw, n in strings.split_lines(txt, context.temp_allocator) {
		line := raw
		if k := strings.index_byte(line, '#'); k >= 0 {line = line[:k]}
		f := strings.fields(line, context.temp_allocator)
		if len(f) == 0 {continue}
		bad := false
		switch f[0] {
		case "frame_prolog", "frame_epilog":
			a, oka := hex_u32(len(f) == 2 || (len(f) == 3 && f[0] == "frame_prolog") ? f[1] : "")
			if !oka {bad = true; break}
			if f[0] == "frame_prolog" {cfg_prologs[a] = true} else {cfg_epilogs[a] = true}
			if len(f) == 3 {
				v, okv := strconv.parse_int(f[2], 10)
				if !okv {bad = true; break}
				cfg_prolog_ebp[a] = i32(v)
			}
		case "x87_helper":
			if len(f) != 4 {bad = true; break}
			a, oka := hex_u32(f[1])
			t, okt := strconv.parse_uint(f[2], 10)
			g, okg := strconv.parse_uint(f[3], 10)
			if !oka || !okt || !okg {bad = true; break}
			append(&cfg_x87, [3]u32{a, u32(t), u32(g)})
		case "patched":
			if len(f) != 2 && len(f) != 3 {bad = true; break}
			a, oka := hex_u32(f[1])
			l, okl := uint(4), true
			if len(f) == 3 {l, okl = strconv.parse_uint(f[2], 10)}
			if !oka || !okl {bad = true; break}
			append(&exe.no_fold, [2]u32{a, a + u32(l)})
		case "patch":
			if len(f) < 3 {bad = true; break}
			a, oka := hex_u32(f[1])
			hx := strings.concatenate(f[2:], context.temp_allocator)
			if !oka || len(hx) % 2 != 0 {bad = true; break}
			b := make([]u8, len(hx) / 2)
			for k in 0 ..< len(b) {
				v, okv := strconv.parse_uint(hx[2 * k:2 * k + 2], 16)
				if !okv {bad = true; break}
				b[k] = u8(v)
			}
			if !bad {append(&cfg_patches, Cfg_Patch{a, b, n + 1})}
		case:
			bad = true
		}
		if bad {
			fmt.eprintf("xrc: %s:%d: cannot read \"%s\"\n", path, n + 1, strings.trim_space(raw))
			return false
		}
	}
	return true
}
