package xrc

// Flag liveness and compare-branch fusion.
//
// The translation keeps the x86 arithmetic flags lazily in C locals: a group (cc_op, cc_sz, cc_a, cc_b, cc_r) from
// which xcc() derives ZF/SF/OF/PF, and the carry `cf` on its own (inc/dec keep it). Before a function is emitted,
// x_fl_plan works out over its control flow graph:
//   - for each condition reader (jcc / setcc / cmovcc / fcmov), the flag setters whose group reaches it. When that is
//     one setter on every path (cmp / sub / test / and / or / xor / add / inc / dec), the reader becomes a C
//     comparison of that setter's operands (x_fused_cond), wherever it is: instructions that do not touch the flags
//     may come between, and labels too, as long as every way in comes from that setter;
//   - which of the group's fields and the carry each setter must store: all of them only when some reader takes the
//     flags the general way (xcond(xcc(...)) / pushfd / lahf / sahf / adc ...), only cc_a / cc_b / cc_r when its
//     readers are fused, and nothing at all when nothing reads its flags (dead flag results are dropped).
// The setters that store less are the ALU ops (add sub cmp adc sbb and or xor test) and inc / dec; every other
// instruction that touches the flags is emitted as before, and counted conservatively (it reads and writes all).
// An instruction classified as not touching the flags is checked on its output: if it names a flag local after all,
// the function is translated again without this plan (XRC_FL_DEBUG=1 prints those). XRC_FLAGS_OLD=1 turns the plan
// off (the old behaviour: only a reader right after its setter fused, every setter storing everything).

import "core:fmt"
import "core:os"
import "core:strings"
import x86 "core:rexcode/isa/x86"

// what a setter stores (Xlat.fl_need)
FL_A :: u8(1) // cc_a
FL_B :: u8(2) // cc_b
FL_R :: u8(4) // cc_r
FL_FULL :: u8(8) // the whole group (cc_op cc_sz cc_a cc_b cc_r: a reader evaluates xcc)
FL_C :: u8(16) // cf

Fl_Cls :: struct {
	known:  bool, // touches the flags (an instruction that is not known must not: checked on its output)
	g_use:  bool, // reads the group the general way (and so needs it whole from every setter that reaches it)
	c_use:  bool, // reads cf
	g_def:  bool, // writes the group (a reader past it never sees an earlier setter's)
	c_def:  bool, // always writes cf
	setter: bool, // stores only what its readers need (alu / inc / dec)
	kind:   int, // fusable setter (x_fused_cond): 1 sub/cmp/neg, 2 logic, 3 add, 4 float compare, 5 result only (inc dec shifts), else 0
	sz:     u8,
	cc:     int, // a condition reader: its tttn 0..15, else -1
}

x_fl_off_env: int = -1 // XRC_FLAGS_OLD (cached)
x_fl_dbg_env: int = -1 // XRC_FL_DEBUG (cached)

x_fl_disabled :: proc() -> bool {
	if x_fl_off_env < 0 {x_fl_off_env = os.get_env("XRC_FLAGS_OLD", context.temp_allocator) != "" ? 1 : 0}
	return x_fl_off_env == 1
}

x_fl_ver_env: int = -1 // XRC_FL_VERIFY (cached)

// XRC_FL_VERIFY=1: a checking translation. Every flag writer stores everything (the old behaviour, which the game
// runs on), and the plan's view is kept beside it in p_* locals: a setter stores there only what the plan says it
// must, the rest poisoned (complemented); every condition reader evaluates the plan's expression on p_* and the old
// one on cc_*, and xrc_flchk reports a difference (xrc_rt.c: "xrc: flag plan wrong at <addr>").
x_fl_verify :: proc() -> bool {
	if x_fl_ver_env < 0 {x_fl_ver_env = os.get_env("XRC_FL_VERIFY", context.temp_allocator) != "" ? 1 : 0}
	return x_fl_ver_env == 1
}

// verify: the setter's plan view (after its full stores to cc_*): what it needs copied, the rest complemented
x_fl_shadow :: proc(xl: ^Xlat, has_cf: bool) -> string {
	nd := xl.fl_need[xl.cur]
	full := nd & FL_FULL != 0
	f :: proc(ok: bool, name: string) -> string {return ok ? fmt.tprintf("p_%s = cc_%s; ", name, name) : fmt.tprintf("p_%s = ~cc_%s; ", name, name)}
	opsz := full ? "p_op = cc_op; p_sz = cc_sz; " : "p_op = CC_RAW; p_sz = 4; "
	c := ""
	if has_cf {c = nd & FL_C != 0 ? "p_cf = cf; " : "p_cf = !cf; "}
	return fmt.tprintf("%s%s%s%s%s", opsz, f(full || nd & FL_A != 0, "a"), f(full || nd & FL_B != 0, "b"), f(full || nd & FL_R != 0, "r"), c)
}

// a C expression with the flag locals renamed to the plan view
x_fl_to_plan :: proc(e: string) -> string {
	r := e
	for w in ([?][2]string{{"cc_fa", "p_fa"}, {"cc_fb", "p_fb"}, {"cc_op", "p_op"}, {"cc_sz", "p_sz"}, {"cc_a", "p_a"}, {"cc_b", "p_b"}, {"cc_r", "p_r"}, {", cf,", ", p_cf,"}}) {
		r, _ = strings.replace_all(r, w[0], w[1], context.temp_allocator)
	}
	return r
}

x_fl_debug :: proc() -> bool {
	if x_fl_dbg_env < 0 {x_fl_dbg_env = os.get_env("XRC_FL_DEBUG", context.temp_allocator) != "" ? 1 : 0}
	return x_fl_dbg_env == 1
}

x_fl_class :: proc(xl: ^Xlat, i: ^Rx_Ins) -> (c: Fl_Cls) {
	c.cc = -1
	m := i.inst.mnemonic
	n := int(i.inst.operand_count)
	imm_count := n >= 2 && i.inst.ops[1].kind == .IMMEDIATE && (u32(i.inst.ops[1].immediate) & 31) != 0
	opaque :: proc(c: ^Fl_Cls) {c.known = true; c.g_use = true; c.c_use = true; c.g_def = true}
	// x87 (by opcode: fcmov reads the flags, fcomi / fucomi(p) write them)
	if i.opc >= 0xd8 && i.opc <= 0xdf && i.xop == .None {
		reg := (i.modrm >> 3) & 7
		if i.modrm >= 0xc0 {
			if i.opc == 0xda && reg <= 3 {tb := [4]int{2, 4, 6, 10}; c.known = true; c.cc = tb[reg]; return}
			if i.opc == 0xdb && reg <= 3 {tb := [4]int{3, 5, 7, 11}; c.known = true; c.cc = tb[reg]; return}
			if (i.opc == 0xdb || i.opc == 0xdf) && (reg == 5 || reg == 6) {c.known, c.setter, c.g_def, c.c_def, c.kind = true, true, true, true, 4; return}
		}
		return
	}
	#partial switch m {
	case .CALL:
		// __security_check_cookie, inlined as its compare (x_call: the same conditions)
		if n > 0 && i.inst.ops[0].kind == .RELATIVE {
			t := rx_rel_target(i, 0)
			if _, is_ck := x_cookie_check(t); is_ck && t in x_direct_ok {
				e := xl.callfp[i.addr]
				if e.transparent || (e.kin == 0 && e.kout == 0) {
					c.known, c.setter, c.g_def, c.c_def, c.kind, c.sz = true, true, true, true, 1, 4
				}
			}
		}
	case .ADD, .SUB, .CMP, .AND, .OR, .XOR, .TEST:
		c.known, c.setter, c.g_def, c.c_def = true, true, true, true
		c.sz = i.inst.ops[0].size
		#partial switch m {
		case .SUB, .CMP: c.kind = 1
		case .ADD: c.kind = 3
		case: c.kind = 2
		}
	case .ADC, .SBB:
		c.known, c.setter, c.g_def, c.c_def, c.c_use = true, true, true, true, true
		c.sz = i.inst.ops[0].size
	case .INC, .DEC:
		c.known, c.setter, c.g_def = true, true, true
		c.kind, c.sz = 5, i.inst.ops[0].size
	case .NEG: // (0 - x: a sub)
		c.known, c.setter, c.g_def, c.c_def = true, true, true, true
		c.kind, c.sz = 1, i.inst.ops[0].size
	case .COMISS, .UCOMISS, .COMISD, .UCOMISD:
		c.known, c.setter, c.g_def, c.c_def, c.kind = true, true, true, true, 4
	case .XADD, .CMPXCHG, .MUL, .IMUL, .POPF, .POPFD:
		c.known, c.g_def, c.c_def = true, true, true
	case .SAHF: c.known, c.g_use, c.g_def, c.c_def = true, true, true, true
	case .LAHF, .PUSHF, .PUSHFD: c.known, c.g_use, c.c_use = true, true, true
	case .CLC, .STC: c.known, c.c_def = true, true
	case .CMC: c.known, c.c_use, c.c_def = true, true, true
	case .BT, .BTS, .BTR, .BTC: c.known, c.c_def = true, true
	case .BSF, .BSR: c.known, c.g_def = true, true
	case .SHL, .SAL, .SHR, .SAR:
		// a count of 0 leaves the flags alone: only a constant non-zero count surely writes them
		if imm_count {
			c.known, c.setter, c.g_def, c.c_def = true, true, true, true
			c.kind, c.sz = 5, i.inst.ops[0].size
		} else {opaque(&c)}
	case .ROL, .ROR:
		c.known = true
		if imm_count {c.c_def = true} else {c.c_use = true}
	case .RCL, .RCR: c.known, c.c_use = true, true
	case .SHLD, .SHRD, .LOOPE, .LOOPNE, .SCAS, .SCASB, .SCASW, .SCASD, .CMPS, .CMPSB, .CMPSW, .CMPSD:
		opaque(&c)
	case:
		if cc := x_cc(m); cc >= 0 {c.known = true; c.cc = cc}
	}
	return
}

// the fields of a fusable setter that x_fused_cond reads for this condition
x_fl_fields :: proc(kind: int, cc: int) -> u8 {
	switch kind {
	case 1: return (cc == 8 || cc == 9 || cc == 10 || cc == 11) ? FL_R : FL_A | FL_B
	case 2: return (cc <= 3) ? 0 : FL_R
	case 3: return (cc == 2 || cc == 3) ? FL_A | FL_R : FL_R
	case 4: return (cc == 0 || cc == 1 || cc == 8 || cc == 9 || cc == 12 || cc == 13) ? 0 : FL_A | FL_B
	}
	return FL_R
}

// a float compare (comiss / comisd / ucomis* / fcomi / fucomi(p)) as a setter: the operands kept as doubles in
// cc_fa / cc_fb for its fused readers, cf alone, or the whole XFCOMI when a reader takes the flags the general way
x_fl_fcomi :: proc(xl: ^Xlat, a, b: string) {
	xl.uses_f = true
	if !xl.fl_on || xl.cur >= len(xl.fl_need) || !xl.fl_cls[xl.cur].setter {
		pf(xl, "    XFCOMI(%s, %s)", a, b)
		return
	}
	nd := xl.fl_need[xl.cur]
	v := x_fl_verify()
	full := v || nd & FL_FULL != 0
	ab := v || nd & (FL_A | FL_B) != 0
	c := !full && nd & FL_C != 0
	if !full && !ab && !c {return} // (dead: nothing reads it)
	if ab {xl.uses_ffl = true}
	sh := ""
	if v {
		sh = fmt.tprintf("%s%s%s", x_fl_shadow(xl, true), nd & FL_A != 0 ? "p_fa = cc_fa; " : "p_fa = 1e300; ", nd & FL_B != 0 ? "p_fb = cc_fb; " : "p_fb = -1e300; ")
	}
	pf(xl, "    { double _fa = (%s), _fb = (%s);%s%s%s%s }", a, b, full ? " XFCOMI(_fa, _fb)" : " XFC_PROBE(_fa, _fb);",
		ab ? " cc_fa = _fa; cc_fb = _fb;" : "", c ? " cf = xfc_un(_fa, _fb) || _fa < _fb;" : "", sh == "" ? "" : fmt.tprintf(" %s", sh))
}

// the plan (see the top): xl.fl_cls, xl.fl_fuse (the fused setter of each reader, -1 none), xl.fl_need
x_fl_plan :: proc(xl: ^Xlat, ins: []Rx_Ins, entry: u32) {
	n := len(ins)
	ta := context.temp_allocator
	cls := make([]Fl_Cls, n, ta)
	idx := make(map[u32]i32, n, ta)
	any_reader := false
	for &i, k in ins {
		cls[k] = x_fl_class(xl, &i)
		idx[i.addr] = i32(k)
		if cls[k].cc >= 0 || cls[k].g_use || cls[k].c_use {any_reader = true}
	}
	xl.fl_cls = cls
	xl.fl_fuse = make([]i32, n, ta)
	xl.fl_need = make([]u8, n, ta)
	for k in 0 ..< n {xl.fl_fuse[k] = -1}
	xl.fl_on = true
	if !any_reader {return} // (no flag is ever read: every setter stores nothing)

	// edges: more than the emitted code has is only conservative; fewer would not be
	edges := make([dynamic][2]i32, 0, 2 * n + 8, ta)
	for &i, k in ins {
		m := i.inst.mnemonic
		rel := i.inst.operand_count > 0 && i.inst.ops[0].kind == .RELATIVE
		if (rx_is_jcc(m) || m == .JMP) && rel {
			t := rx_rel_target(&i, 0)
			if t in xl.labels {if j, ok := idx[t]; ok {append(&edges, [2]i32{i32(k), j})}}
		}
		if m == .JMP && !rel {
			if _, is_tbl := xl.tables[i.addr]; is_tbl {
				for st in rx_table_slots[i.addr] {
					if j, ok := idx[st[1]]; ok {append(&edges, [2]i32{i32(k), j})}
				}
			}
		}
		if m != .RET && m != .JMP {
			if j, ok := idx[i.addr + i.len]; ok {append(&edges, [2]i32{i32(k), j})}
			if k + 1 < n && ins[k + 1].addr != i.addr + i.len {append(&edges, [2]i32{i32(k), i32(k + 1)})}
		}
	}
	// predecessors (CSR)
	pstart := make([]i32, n + 1, ta)
	for e in edges {pstart[e[1] + 1] += 1}
	for k in 0 ..< n {pstart[k + 1] += pstart[k]}
	pred := make([]i32, len(edges), ta)
	fill := make([]i32, n, ta)
	for e in edges {
		pred[pstart[e[1]] + fill[e[1]]] = e[0]
		fill[e[1]] += 1
	}
	ent := i32(-1)
	if j, ok := idx[entry]; ok {ent = j}

	seen := make([]i32, n, ta) // the walk that last visited each node (walk numbers from 1)
	stack := make([dynamic]i32, 0, 64, ta)
	defs := make([dynamic]i32, 0, 8, ta)
	cuse := make([]bool, n, ta) // reads cf (after the fusion decisions)
	walk := i32(0)

	// the group: who reaches each reader
	for k in 0 ..< n {
		c := &cls[k]
		if c.cc < 0 && !c.g_use {continue}
		walk += 1
		clear(&stack)
		clear(&defs)
		at_entry := i32(k) == ent
		for p in pred[pstart[k]:pstart[k + 1]] {append(&stack, p)}
		for len(stack) > 0 {
			j := pop(&stack)
			if seen[j] == walk {continue}
			seen[j] = walk
			if cls[j].g_def {append(&defs, j); continue}
			if j == ent {at_entry = true}
			for p in pred[pstart[j]:pstart[j + 1]] {if seen[p] != walk {append(&stack, p)}}
		}
		if c.g_use { 	// read the general way (opaque instructions)
			for d in defs {xl.fl_need[d] |= FL_FULL}
			if c.c_use {cuse[k] = true}
			continue
		}
		cc := c.cc
		if len(defs) == 1 && !at_entry {
			d := defs[0]
			if cls[d].kind > 0 {
				if _, ok := x_fused_cond(cls[d].kind, cls[d].sz, cc); ok {
					xl.fl_fuse[k] = d
					xl.fl_need[d] |= x_fl_fields(cls[d].kind, cc)
					continue
				}
			}
		}
		if x_fl_debug() {
			why := ""
			if at_entry {why = "entry"} else if len(defs) != 1 {why = fmt.tprintf("%d setters", min(len(defs), 3))} else {
				d := defs[0]
				why = cls[d].kind > 0 ? fmt.tprintf("cc %d after kind %d", cc, cls[d].kind) : fmt.tprintf("after %s", rx_mn(&ins[d]))
			}
			fmt.eprintf("xrc flags generic: %s\n", why)
		}
		if cc != 2 && cc != 3 {for d in defs {xl.fl_need[d] |= FL_FULL}}
		if cc == 2 || cc == 3 || cc == 6 || cc == 7 {cuse[k] = true}
	}
	for k in 0 ..< n {if cls[k].c_use {cuse[k] = true}}

	// the carry: which writers of cf reach a reader of it
	for k in 0 ..< n {
		if !cuse[k] {continue}
		walk += 1
		clear(&stack)
		for p in pred[pstart[k]:pstart[k + 1]] {append(&stack, p)}
		for len(stack) > 0 {
			j := pop(&stack)
			if seen[j] == walk {continue}
			seen[j] = walk
			if cls[j].c_def {xl.fl_need[j] |= FL_C; continue}
			if cuse[j] {continue} // (its own walk marks what reaches it, which is what reaches here through it)
			for p in pred[pstart[j]:pstart[j + 1]] {if seen[p] != walk {append(&stack, p)}}
		}
	}
	if x_fl_debug() {
		ns, nd, nf := 0, 0, 0
		for k in 0 ..< n {
			if cls[k].setter {ns += 1; if xl.fl_need[k] == 0 {nd += 1}}
			if xl.fl_fuse[k] >= 0 {nf += 1}
		}
		fmt.eprintf("xrc flags %08x: %d setters, %d dead, %d fused readers\n", entry, ns, nd, nf)
	}
}

// a flag local named in this C as a whole word (the check of instructions classified as not touching the flags)
x_fl_mentions :: proc(t: string) -> bool {
	idc :: proc(c: u8) -> bool {return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'}
	WORDS :: [?]string{"cc_op", "cc_sz", "cc_a", "cc_b", "cc_r", "cc_fa", "cc_fb", "cf", "xcc", "xcond", "XFCOMI", "xfc_un"}
	for k := 0; k < len(t); k += 1 {
		if k > 0 && idc(t[k - 1]) {continue}
		ch := t[k]
		if ch != 'c' && ch != 'x' && ch != 'X' {continue}
		for w in WORDS {
			e := k + len(w)
			if e <= len(t) && t[k:e] == w && (e == len(t) || !idc(t[e])) {return true}
		}
	}
	return false
}

// the flag stores of a setter (fields per the plan): "" when none
x_fl_stores :: proc(xl: ^Xlat, full: string) -> string {
	if !xl.fl_on {return full}
	if x_fl_verify() {return fmt.tprintf("%s%s", full, x_fl_shadow(xl, false))}
	nd := xl.fl_need[xl.cur]
	if nd & FL_FULL != 0 {return full}
	a := nd & FL_A != 0 ? "cc_a = _a; " : ""
	b := nd & FL_B != 0 ? "cc_b = _b; " : ""
	r := nd & FL_R != 0 ? "cc_r = _r; " : ""
	return fmt.tprintf("%s%s%s", a, b, r)
}
