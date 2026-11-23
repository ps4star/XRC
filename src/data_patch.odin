package xrc

// xrc.cfg's `patch` entries (cfg_patches), applied to the loaded image before anything is decoded: a mod's code
// patches, translated as if the program had always had them.
//
// A patch within a section's file bytes is written there (a hook: a jmp to new code, an instruction changed in place).
// A patch in no section's span is new code -- what the hooks jump to, typically in the padding after the code section
// -- and becomes an executable section of its own; the code range grows to take it in, so a hook's jump to it is
// followed as part of the function hooked (a function chunk, rx_function) and its jumps back are that function's labels.
// Anything else (a patch across the end of a section's file bytes, into another patch) is refused.

import "core:fmt"

apply_patches :: proc() -> bool {
	if len(cfg_patches) == 0 {return true}
	if exe.format != .PE {fmt.eprintln("xrc: xrc.cfg patch entries are for PE images only"); return false}
	data := make([dynamic]u8, 0, len(exe.data) + 4096)
	append(&data, ..exe.data)
	secs := make([dynamic]Pe_Sec, 0, len(exe.secs) + len(cfg_patches))
	append(&secs, ..exe.secs)
	nin, nnew := 0, 0
	for p in cfg_patches {
		lo, hi := p.addr, p.addr + u32(len(p.bytes))
		placed := false
		for s in secs {
			span := max(s.vsize, s.rsize)
			if hi <= s.base || lo >= s.base + span {continue}
			if lo < s.base || hi > s.base + s.rsize || int(s.raw + s.rsize) > len(exe.data) {
				fmt.eprintf("xrc: xrc.cfg:%d: patch %08x-%08x is not within one section's file bytes\n", p.line, lo, hi)
				return false
			}
			copy(data[int(s.raw + (lo - s.base)):], p.bytes)
			placed = true
			nin += 1
			break
		}
		if placed {continue}
		append(&secs, Pe_Sec{base = lo, vsize = u32(len(p.bytes)), raw = u32(len(data)), rsize = u32(len(p.bytes)), chars = Sec_Read | Sec_Exec})
		append(&data, ..p.bytes)
		if lo < exe.text_lo || hi > exe.text_hi {
			if lo < exe.text_hi && hi > exe.text_lo {
				fmt.eprintf("xrc: xrc.cfg:%d: patch %08x-%08x overlaps the code range's end\n", p.line, lo, hi)
				return false
			}
			exe.text_lo = min(exe.text_lo, lo)
			exe.text_hi = max(exe.text_hi, hi)
		}
		exe.hi = max(exe.hi, (hi + PAGE - 1) & ~u32(PAGE - 1))
		nnew += 1
	}
	exe.data = data[:]
	exe.secs = secs[:]
	fmt.printf("xrc: %d patches from xrc.cfg (%d in place, %d new code)\n", nin + nnew, nin, nnew)
	return true
}
