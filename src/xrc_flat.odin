package xrc

import "core:strings"
import x86 "core:rexcode/isa/x86"

// [reg + disp] with a small displacement (the frame's locals and arguments, a structure's fields): the address as
// (XEA_T)(u32)(registers) + disp, a 64-bit sum, and the accesses through MX*, which do not truncate it to 32 bits. The
// host instruction then takes the displacement (plus the fixed guest base) as its own, where the 32-bit wrap made
// the C compiler add it separately first (an lea per access). The two differ only for an address that would have
// wrapped past 0 or 4 GB (a register within 64 KB of either end), and no such address is valid in the guest.
// XRC_FLAT_EA=0: off (as before).

x_flat_on := false

X_FLAT_MAX :: 0x10000

x_flat_addr :: proc(m: x86.Memory) -> (regs: string, disp: i32, ok: bool) {
	if !x_flat_on || m.segment != 0 {return}
	d := m.disp
	if d == 0 || d <= -X_FLAT_MAX || d >= X_FLAT_MAX {return}
	hb, hi := x86.mem_has_base(m), x86.mem_has_index(m)
	if !hb && !hi {return}
	b := ""
	if hb {b = x_home(x86.mem_base(m))}
	if hi {
		sc := x86.mem_scale(m)
		ix := x_home(x86.mem_index(m)) if sc == 1 else strings.concatenate({x_home(x86.mem_index(m)), " * ", sc == 2 ? "2" : sc == 4 ? "4" : "8"}, context.temp_allocator)
		// (the registers' sum in 32 bits first: an index register holds a negative index as a large u32)
		b = hb ? strings.concatenate({"(u32)(", b, " + ", ix, ")"}, context.temp_allocator) : strings.concatenate({"(u32)(", ix, ")"}, context.temp_allocator)
	}
	return b, d, true
}

@(private = "file")
x_flat_pairs := [?][2]string {
	{"M8(_ea", "MX8(_ea"},
	{"M16(_ea", "MX16(_ea"},
	{"M32(_ea", "MX32(_ea"},
	{"M64(_ea", "MX64(_ea"},
	{"MF32(_ea", "MXF32(_ea"},
	{"MF64(_ea", "MXF64(_ea"},
	{"XLD(", "XLDX("},
	{"XST(_ea", "XSTX(_ea"},
}

// the instruction's accesses through _ea (emitted from start on) as MX*
x_flat_rewrite :: proc(xl: ^Xlat, start: int) {
	seg := string(xl.b.buf[start:])
	if strings.index(seg, "(_ea") < 0 && strings.index(seg, ", _ea") < 0 {return}
	out := strings.clone(seg, context.temp_allocator)
	for p in x_flat_pairs {
		if strings.index(out, p[0]) >= 0 {out, _ = strings.replace_all(out, p[0], p[1], context.temp_allocator)}
	}
	resize(&xl.b.buf, start)
	strings.write_string(&xl.b, out)
}

// a C identifier alone (a register or temp variable)
x_is_ident :: proc(v: string) -> bool {
	if len(v) == 0 {return false}
	for ch, i in transmute([]u8)v {
		if ch == '_' || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (i > 0 && ch >= '0' && ch <= '9') {continue}
		return false
	}
	return true
}
