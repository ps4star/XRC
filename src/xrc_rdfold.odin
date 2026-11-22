package xrc

import "core:fmt"
import "core:strings"

// Reads of read-only data at a fixed address (a float constant, a table of masks, a string's bytes) as the values
// themselves: `movss xmm0, [0x25a3c20]` gave X[0].u[0] = M32(_ea), a load the C compiler cannot fold; now it is
// X[0].u[0] = 0x3f800000u, and what is computed from it can be folded too. Only data the image's file holds, in a
// section neither writable nor executable, and none of exe.no_fold (the import tables the runtime fills; xrc.cfg's
// `patched`: what the runtime changes). An instruction that also writes its address (it would fault: the runtime
// maps those sections read-only) is left as it was. XRC_RDATA_FOLD=0: off.

x_rdata_on := false

// the data at [a, a + n) as constant: in one read-only section's file bytes and not exe.no_fold's
x_rdata_const :: proc(a: u32, n: u32) -> (v: u64, ok: bool) {
	if exe.format != .PE || a + n < a {return}
	for r in exe.no_fold {if a < r[1] && a + n > r[0] {return}}
	for s in exe.secs {
		if a < s.base || a + n > s.base + s.rsize {continue}
		if s.chars & (Sec_Write | Sec_Exec) != 0 {return}
		b := exe_read(a, int(n))
		if len(b) != int(n) {return}
		for i in 0 ..< int(n) {v |= u64(b[i]) << (8 * u64(i))}
		return v, true
	}
	return
}

@(private = "file")
Rd_Mac :: struct {
	name: string,
	size: u32,
}

@(private = "file")
rd_macs := [?]Rd_Mac{{"MF64(", 8}, {"MF32(", 4}, {"M64(", 8}, {"M32(", 4}, {"M16(", 2}, {"M8(", 1}}

@(private = "file")
rd_ident :: #force_inline proc(ch: u8) -> bool {
	return ch == '_' || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')
}

// the instruction emitted from start on, its address the constant a: its reads through _ea as values
x_rdata_fold :: proc(xl: ^Xlat, start: int, a: u32) {
	seg := string(xl.b.buf[start:])
	if strings.index(seg, "(_ea") < 0 {return}
	if strings.index(seg, "XST(_ea") >= 0 || strings.index(seg, "XSTX(_ea") >= 0 {return} // (a store)
	out := strings.builder_make(context.temp_allocator)
	i := 0
	folded := false
	scan: for i < len(seg) {
		for m in rd_macs {
			if !strings.has_prefix(seg[i:], m.name) || (i > 0 && rd_ident(seg[i - 1])) {continue}
			j := i + len(m.name)
			if !strings.has_prefix(seg[j:], "_ea") {continue}
			j += 3
			off := u32(0)
			if strings.has_prefix(seg[j:], " + ") {
				k := j + 3
				e := k
				for e < len(seg) && seg[e] != ')' {e += 1}
				num := seg[k:e]
				if strings.has_suffix(num, "u") {num = num[:len(num) - 1]}
				v, okv := hex_u32(strings.has_prefix(num, "0x") ? num[2:] : "")
				if !okv {
					// (a decimal offset)
					dv := u32(0)
					if len(num) == 0 {return}
					for ch in transmute([]u8)num {
						if ch < '0' || ch > '9' {return}
						dv = dv * 10 + u32(ch - '0')
					}
					v = dv
				}
				off = v
				j = e
			}
			if j >= len(seg) || seg[j] != ')' {return}
			j += 1
			// a store (M32(_ea) = ..., M32(_ea) |= ...) or its address taken: the instruction as it was
			p := j
			for p < len(seg) && seg[p] == ' ' {p += 1}
			if p < len(seg) && seg[p] == '=' && (p + 1 >= len(seg) || seg[p + 1] != '=') {return}
			if p + 1 < len(seg) && seg[p + 1] == '=' && strings.index_byte("+-*/%&|^", seg[p]) >= 0 {return}
			if p + 2 < len(seg) && seg[p + 2] == '=' && (seg[p:p + 2] == "<<" || seg[p:p + 2] == ">>") {return}
			if i > 0 && seg[i - 1] == '&' {return}
			v, ok := x_rdata_const(a + off, m.size)
			if !ok {return}
			switch m.name {
			case "MF64(": fmt.sbprintf(&out, "f64_of(0x%xull)", v)
			case "MF32(": fmt.sbprintf(&out, "f32_of(0x%xu)", u32(v))
			case "M64(": fmt.sbprintf(&out, "0x%xull", v)
			case "M32(": fmt.sbprintf(&out, "0x%xu", u32(v))
			case "M16(": fmt.sbprintf(&out, "((u16)0x%xu)", u32(v))
			case "M8(": fmt.sbprintf(&out, "((u8)0x%xu)", u32(v))
			}
			folded = true
			i = j
			continue scan
		}
		strings.write_byte(&out, seg[i])
		i += 1
	}
	if !folded {return}
	s := strings.to_string(out)
	resize(&xl.b.buf, start)
	strings.write_string(&xl.b, s)
}
