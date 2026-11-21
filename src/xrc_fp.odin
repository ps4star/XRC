package xrc

// The floating-point half of the translator -- x87 and SSE (see xrc.odin, include/xrc.h).
//
// x87: the register stack is translated at statically known depths. A function is entered with an empty stack
//   (the Win32 ABI), so the depth before every instruction is fixed by the instructions on the path to it; slot k
//   (from the bottom) is the C double `fk`, and st(i) at depth d is f(d-1-i). x87_flow computes the depths over the
//   control-flow graph and fails on anything that is not static (a merge with two depths, an underflow, values
//   still on the stack across a call that does not take them). Calls are where values cross functions:
//     * a float result comes back in st(0): the callee's depth at its `ret` is 1
//     * CRT intrinsics (__CIsqrt, __CIpow, _ftol2 ...) take their arguments on the stack
//   x87_effect works out (kin = values taken, kout = values returned) for a direct callee by running the same flow
//   analysis on it (memoized; recursion is assumed to be (0,0)); for an indirect call or a callee it cannot
//   analyze, it looks at the code right after the call: a read of st(0) before anything pushes means a float
//   result. Arithmetic is in double, rounded to the precision the x87 control word selects (XR87).
// SSE: xmm0-7 live in c->x (X128 X[8]; lanes .f/.u/.d/.q/...). Each instruction reads its operands into temps
//   and writes the destination register whole, so aliasing (addps xmm0, xmm0) needs no care. The 32-bit builds
//   use the real instructions for the estimates (rcpps/rsqrtps) and sqrt, so results are bit-identical.

import "core:fmt"
import "core:os"
import "core:mem/virtual"
import "core:strings"
import x86 "core:rexcode/isa/x86"

X87_Eff :: struct {
	kin, kout:   int,
	known:       bool,
	transparent: bool, // never touches the x87 stack: values on it stay live across the call (cookie check)
	flex:        bool, // no x87 instruction of its own, its result only that of calls it makes (a forwarder of a
	                   // virtual float getter): whether it returns st(0) is up to the caller's flow (x87_solve)
}

X87_Flow :: struct {
	depth:  map[u32]int,
	call:   map[u32]X87_Eff,
	ret:    int, // depth at ret (-1: no ret reached)
	fmax:   int,
	ok:     bool,
	reason: string,
	guessed: [dynamic]u32, // calls whose float result was guessed (x87_guess_result), in the order met
}

x87_memo: map[u32]X87_Eff
x87_busy: map[u32]bool
x87_named: map[u32]X87_Eff
x87_named_loaded: bool
// CRT helpers taking x87 arguments that leave the values below them alone (MSVC keeps values live across them:
// fld a; fld b; call _ftol2_sse -> a stays)
x87_deep: map[u32]bool

// CRT helpers that take x87 arguments, by name (ghidra/function_map.csv)
x87_load_named :: proc() {
	if x87_named_loaded {return}
	x87_named_loaded = true
	// the binary's helpers with an x87 interface the flow analysis cannot derive (xrc.cfg x87_helper: address, values
	// taken, values returned; e.g. the CRT's math entry points, whose x87 paths are dynamic). A wrong guess breaks
	// every caller (__CItan once came out as (0, 1)).
	for p in cfg_x87 {
		x87_named[p[0]] = {int(p[1]), int(p[2]), true, false, false}
		if p[1] > 0 {x87_deep[p[0]] = true}
	}
	// the names: function_map.csv's, or the executable's symbols when it is the map
	named := make([dynamic]struct {a: u32, n: string}, context.temp_allocator)
	if map_from_symbols {
		for a, n in exe.names {append(&named, struct {a: u32, n: string}{a, n})}
	} else {
		txt, ok := read_text(table_path("function_map.csv"))
		if !ok {return}
		for line in strings.split_lines_iterator(&txt) {
			f := strings.split(line, ",", context.temp_allocator)
			if len(f) < 2 {continue}
			a, ok1 := hex_u32(f[0])
			if !ok1 {continue}
			append(&named, struct {a: u32, n: string}{a, f[1]})
		}
	}
	for e in named {
		a, n := e.a, e.n
		switch {
		// the CRT math intrinsics are stubs `mov edx, table; jmp __cintrindisp2/1` (or __ctrandisp*): naming the
		// dispatchers lets x87_effect derive each stub's arguments from its tail jump
		case n == "__security_check_cookie": x87_named[a] = {0, 0, true, true, false}
		case n == "__cintrindisp2" || n == "__ctrandisp2": x87_named[a] = {2, 1, true, false, false}
		case n == "__cintrindisp1" || n == "__ctrandisp1": x87_named[a] = {1, 1, true, false, false}
		case n == "__CIpow" || n == "__CIatan2" || n == "__CIfmod": x87_named[a] = {2, 1, true, false, false}
		case strings.has_prefix(n, "__CI"): x87_named[a] = {1, 1, true, false, false}
		case strings.contains(n, "ftol"): x87_named[a] = {1, 0, true, false, false}
		}
	}
}

// XRC_X87_DEBUG, looked up once
x87_debug_v: int
x87_debug :: proc() -> bool {
	if x87_debug_v == 0 {x87_debug_v = os.get_env("XRC_X87_DEBUG", context.temp_allocator) != "" ? 2 : 1}
	return x87_debug_v == 2
}

x87_is :: proc(ins: ^Rx_Ins) -> bool {
	return (ins.opc >= 0xd8 && ins.opc <= 0xdf) || ins.opc == 0x9b
}

// the x87 effect of calling t
x87_effect :: proc(t: u32) -> X87_Eff {
	x87_load_named()
	if e, ok := x87_named[t]; ok {return e}
	if e, ok := x87_memo[t]; ok {return e}
	sz, okf := funcs[t]
	if !okf {return {}}
	if t in x87_busy {return {-1, 0, true, false, true}} // recursion: the stack holds its arguments (kin -1); its result is the caller's flow's to decide (flex)
	x87_busy[t] = true
	res := X87_Eff{}
	{
		// each callee is analyzed in its own arena, freed when done (the call tree would not fit in temp)
		ar: virtual.Arena
		if virtual.arena_init_growing(&ar) == nil {
			al := virtual.arena_allocator(&ar)
			old := context.temp_allocator
			context.temp_allocator = al
			ins, bad := rx_function(t, sz, al)
			// no x87 instruction of its own and only calls to such functions (pure SSE / integer code, e.g. an LTCG
			// helper taking xmm arguments): it cannot touch the x87 stack, so values on it stay live across the call
			pure := bad == 0
			callee_ok :: proc(tt, t: u32) -> bool {return tt == t || x87_effect(tt).transparent}
			for &i in ins {
				if !pure {break}
				m := i.inst.mnemonic
				k := i.inst.operand_count > 0 ? i.inst.ops[0].kind : .NONE
				switch {
				case x87_is(&i): pure = false
				case m == .CALL: pure = k == .RELATIVE && callee_ok(rx_rel_target(&i, 0), t)
				case m == .JMP && k == .RELATIVE, rx_is_jcc(m) && k == .RELATIVE:
					tt := rx_rel_target(&i, 0)
					if tt in funcs {pure = callee_ok(tt, t)} // a tail call
				case m == .JMP && k == .MEMORY: pure = x86.mem_has_index(i.inst.ops[0].mem) // a switch, not a tail call
				case m == .JMP: pure = false
				}
			}
			if pure {
				res = {0, 0, true, true, false}
			} else if bad == 0 {
				for e in 0 ..= 2 {
					fl := x87_solve(ins[:], t, sz, e)
					if fl.ok {
						res = {e, max(fl.ret, 0), true, false, false}
						// its result decided by guesses (a float returned from a virtual call it makes, maybe after
						// x87 code of its own): flexible for the caller. In the portable runtime c->st0 passes
						// through to the caller either way.
						res.flex = e == 0 && res.kout == 0 && len(fl.guessed) > 0
						break
					}
				}
			}
			context.temp_allocator = old
			virtual.arena_destroy(&ar)
		}
	}
	delete_key(&x87_busy, t)
	x87_memo[t] = res
	return res
}

// depth effect of one x87 instruction at depth d (-1 + fail reason on underflow / unsupported)
x87_step :: proc(ins: ^Rx_Ins, d: int) -> (int, string) {
	xl := Xlat{ok = true, dry = true} // (dry: pf writes nothing, so no builder)
	nd := x87_emit(&xl, ins, d, "_ea", true)
	if !xl.ok {return -1, xl.reason}
	return nd, ""
}

// a float result after a call we cannot analyze: is st(0) read before anything is pushed?
x87_guess_result :: proc(ins: []Rx_Ins, i: int) -> int {
	d := 0
	pc := ins[i].addr + ins[i].len
	for j := i + 1; j < len(ins) && j < i + 40; j += 1 {
		it := &ins[j]
		if it.addr != pc {break}
		pc += it.len
		m := it.inst.mnemonic
		if x87_is(it) {
			nd, _ := x87_step(it, d)
			if nd < 0 {return 1}
			d = nd
			continue
		}
		if m == .CALL && it.inst.ops[0].kind == .RELATIVE && d == 0 {
			if e := x87_effect(rx_rel_target(it, 0)); e.known && e.kin >= 1 {return 1} // (_ftol2 of the result)
		}
		if m == .CALL || m == .RET || m == .JMP || rx_is_jcc(m) {break}
	}
	return 0
}

// x87_flow, revising the guessed results of indirect calls (a virtual call returning a float whose st(0) is only
// consumed by a callee, or returned by this function) until the depths are consistent: the guesses as made, then each
// guess flipped, then all of them flipped. Compiled code has exactly one consistent reading.
X87_BY_TARGET :: u32(0xc0000000) // x87_flow's force map: a key with these bits flipped is a callee address

x87_solve :: proc(ins: []Rx_Ins, a, size: u32, entry: int) -> X87_Flow {
	fl := x87_flow(ins, a, size, entry, nil)
	if fl.ok || len(fl.guessed) == 0 || len(fl.guessed) > 64 {return fl}
	first := fl
	dbg := x87_debug()
	if dbg {
		fmt.eprintf("x87_solve %08x entry %d: %s; guessed:", a, entry, fl.reason)
		for x in fl.guessed {fmt.eprintf(" %08x(k%d)", x, fl.call[x].kout)}
		fmt.eprintln()
	}
	// guesses grouped by callee: calls to the same function share one result (a recursive function's own calls)
	idx := make(map[u32]int, len(ins), context.temp_allocator)
	for &i, k in ins {idx[i.addr] = k}
	group := make(map[u32][dynamic]u32, context.temp_allocator)
	order := make([dynamic]u32, context.temp_allocator)
	for x in first.guessed {
		key := x
		if k, ok := idx[x]; ok && ins[k].inst.ops[0].kind == .RELATIVE {key = rx_rel_target(&ins[k], 0)}
		if !(key in group) {group[key] = make([dynamic]u32, context.temp_allocator); append(&order, key)}
		append(&group[key], x)
	}
	force := make(map[u32]int, context.temp_allocator)
	for key in order {
		clear(&force)
		for x in group[key] {force[x] = 1 - first.call[x].kout}
		// also every call to the same callee the failed walk did not reach (keyed by target ^ X87_BY_TARGET)
		if k, ok := idx[group[key][0]]; ok && ins[k].inst.ops[0].kind == .RELATIVE {force[key ~ X87_BY_TARGET] = 1 - first.call[group[key][0]].kout}
		r := x87_flow(ins, a, size, entry, &force)
		if r.ok {return r}
		if dbg {fmt.eprintf("  flip group %08x: %s\n", key, r.reason)}
	}
	clear(&force)
	for x in first.guessed {force[x] = 1 - first.call[x].kout}
	if r := x87_flow(ins, a, size, entry, &force); r.ok {return r}
	return first
}

// the static x87 stack depth before every instruction of a function entered at depth `entry` (force: results for
// indirect calls, by address, in place of the guess)
x87_flow :: proc(ins: []Rx_Ins, a, size: u32, entry: int, force: ^map[u32]int) -> X87_Flow {
	fl := X87_Flow{ret = -1, ok = true, fmax = entry}
	fl.guessed = make([dynamic]u32, context.temp_allocator)
	fl.depth = make(map[u32]int, len(ins), context.temp_allocator)
	fl.call = make(map[u32]X87_Eff, context.temp_allocator)
	if len(ins) == 0 {return fl}
	idx := make(map[u32]int, len(ins), context.temp_allocator)
	for &i, k in ins {idx[i.addr] = k}
	W :: struct {i, d: int}
	tbl: [dynamic]u32
	pending := false // tail calls whose result is this function's (resolved after the walk)
	work := make([dynamic]W, context.temp_allocator)
	append(&work, W{0, entry})
	bad :: proc(fl: ^X87_Flow, why: string, at: u32) -> X87_Flow {
		fl.ok = false
		fl.reason = fmt.tprintf("%s @%08x", why, at)
		return fl^
	}
	for len(work) > 0 {
		w := pop(&work)
		it := &ins[w.i]
		if d0, seen := fl.depth[it.addr]; seen {
			if d0 != w.d {
				if x87_debug() {fmt.eprintf("x87 merge at %08x: %d vs %d (entry %d of %08x)\n", it.addr, d0, w.d, entry, a)}
				return bad(&fl, "x87 merge", it.addr)
			}
			continue
		}
		fl.depth[it.addr] = w.d
		d := w.d
		m := it.inst.mnemonic
		nd := d
		succ := true
		if x87_is(it) {
			r, why := x87_step(it, d)
			if r < 0 {return bad(&fl, why, it.addr)}
			nd = r
		} else if m == .JMP && rx_table_jmp(it) && x87_table(ins, w.i, a, size, &tbl) {
			// a switch: every case at this depth (the default leaves as a tail call: record it like one)
			for x in tbl {
				if k, ok := idx[x]; ok {append(&work, W{k, d})}
			}
			fl.call[it.addr] = {d, -1, true, false, false}
			pending = true
			fl.fmax = max(fl.fmax, d)
			continue
		} else if m == .CALL || (m == .JMP && (it.inst.ops[0].kind != .RELATIVE || !(rx_rel_target(it, 0) in idx))) {
			e := X87_Eff{}
			if it.inst.ops[0].kind == .RELATIVE {e = x87_effect(rx_rel_target(it, 0))}
			if e.known && e.flex && m == .CALL {
				if e.kin < 0 {e.kin = d}
				if force != nil {
					if v, ok := force[it.addr]; ok {
						e.kout = v
					} else if v2, ok2 := force[rx_rel_target(it, 0) ~ X87_BY_TARGET]; ok2 {
						e.kout = v2
					}
				}
				append(&fl.guessed, it.addr)
			}
			// a callee we cannot analyze: what is on the stack is its arguments (MSVC never keeps x87 values live
			// across an ordinary call). A computed tail call's result is whatever this function returns.
			if !e.known {
				if m == .JMP {
					e = {d, -1, true, false, false}
					pending = true
				} else if force != nil && it.addr in force {
					e = {d, force[it.addr], true, false, false}
					append(&fl.guessed, it.addr)
				} else {
					e = {d, x87_guess_result(ins, w.i), true, false, false}
					append(&fl.guessed, it.addr)
				}
			}
			fl.call[it.addr] = e
			if e.transparent {
				nd = d
			} else {
				deep := m == .CALL && it.inst.ops[0].kind == .RELATIVE && rx_rel_target(it, 0) in x87_deep && d > e.kin
				if d != e.kin && !deep {return bad(&fl, d > e.kin ? "x87 live across call" : "x87 args", it.addr)}
				if e.kin > 2 {return bad(&fl, "x87 args", it.addr)}
				nd = d - e.kin + e.kout
			}
			if m == .JMP { // tail call: its result is ours
				if nd >= 0 {
					if fl.ret >= 0 && fl.ret != nd {return bad(&fl, "x87 ret depth", it.addr)}
					fl.ret = nd
				}
				succ = false
			}
		} else if m == .RET {
			if fl.ret >= 0 && fl.ret != d {return bad(&fl, "x87 ret depth", it.addr)}
			if d > 1 {return bad(&fl, "x87 ret depth", it.addr)}
			fl.ret = d
			succ = false
		}
		if rx_is_jcc(m) && it.inst.ops[0].kind == .RELATIVE && !(rx_rel_target(it, 0) in idx) {
			// conditional tail call
			e := x87_effect(rx_rel_target(it, 0))
			if !e.known {e = {d, -1, true, false, false}; pending = true}
			if !e.transparent && d != e.kin {return bad(&fl, "x87 args", it.addr)}
			fl.call[it.addr] = e
			if e.kout >= 0 {
				if fl.ret >= 0 && fl.ret != e.kout {return bad(&fl, "x87 ret depth", it.addr)}
				fl.ret = e.kout
			}
		}
		if nd > 8 {return bad(&fl, "x87 overflow", it.addr)}
		fl.fmax = max(fl.fmax, nd, d)
		if !succ {continue}
		if (rx_is_jcc(m) || m == .JMP) && it.inst.operand_count > 0 && it.inst.ops[0].kind == .RELATIVE {
			if k, ok := idx[rx_rel_target(it, 0)]; ok {append(&work, W{k, nd})}
		}
		if !rx_is_branch_end(m) {
			if k, ok := idx[it.addr + it.len]; ok {
				append(&work, W{k, nd})
			} else {
				// falls through past the function's end: a tail call to the next function (key with the high bit)
				nx := it.addr + it.len
				e := x87_effect(nx)
				if !e.known {e = {nd, -1, true, false, false}; pending = true}
				if !e.transparent && nd != e.kin {return bad(&fl, "x87 args", it.addr)}
				fl.call[nx | 0x80000000] = e
				if e.kout >= 0 {
					if fl.ret >= 0 && fl.ret != e.kout {return bad(&fl, "x87 ret depth", it.addr)}
					fl.ret = e.kout
				}
			}
		}
	}
	if pending {
		for k, &e in fl.call {
			if e.kout < 0 {e.kout = max(fl.ret, 0)}
		}
	}
	return fl
}

x87_table :: proc(ins: []Rx_Ins, k: int, a, size: u32, out: ^[dynamic]u32) -> bool {
	t, ok := rx_jump_table(ins, k, a, size)
	out^ = t
	return ok
}

// ---------------------------------------------------------------------------------------------------------------
// x87 emission

@(private = "file")
st :: proc(xl: ^Xlat, d, i: int) -> string {
	if i >= d {fail(xl, "x87 underflow"); return "f0"}
	return fmt.tprintf("f%d", d - 1 - i)
}

@(private = "file")
x87_arith :: proc(op: int, a, b: string) -> string {
	switch op {
	case 0: return fmt.tprintf("XR87(%s + %s)", a, b)
	case 1: return fmt.tprintf("XR87(%s * %s)", a, b)
	case 4: return fmt.tprintf("XR87(%s - %s)", a, b)
	case 5: return fmt.tprintf("XR87(%s - %s)", b, a)
	case 6: return fmt.tprintf("XR87(%s / %s)", a, b)
	case 7: return fmt.tprintf("XR87(%s / %s)", b, a)
	}
	return "0"
}

@(private = "file")
x87_cmp :: proc(xl: ^Xlat, a, b: string) {
	pf(xl, "    x_fsw = (x_fsw & ~0x4500u) | fsw_cmp(%s, %s);", a, b)
}

@(private = "file")
x87_fcomi :: proc(xl: ^Xlat, a, b: string) {
	x_fl_fcomi(xl, a, b)
}

@(private = "file")
x87_push :: proc(xl: ^Xlat, d: int, v: string) -> int {
	if d >= 8 {fail(xl, "x87 overflow"); return -1}
	pf(xl, "    f%d = %s;", d, v)
	return d + 1
}

// the integer status word as fnstsw stores it (TOP = 8 - depth for a function entered with an empty stack)
@(private = "file")
x87_fsw :: proc(d: int) -> string {
	return fmt.tprintf("((x_fsw & 0x4700u) | 0x%xu)", u32((8 - d) & 7) << 11)
}

// emit one x87 instruction at stack depth d; returns the depth after it (-1 when unsupported)
x87_emit :: proc(xl: ^Xlat, ins: ^Rx_Ins, d: int, ea: string, have_ea: bool) -> int {
	xl.uses_fp = true
	o := ins.opc
	if o == 0x9b {return d} // fwait
	mod := ins.modrm >> 6
	reg := int((ins.modrm >> 3) & 7)
	rm := int(ins.modrm & 7)
	mn := rx_mn(ins)
	if mod != 3 {
		if !have_ea {fail(xl, mn); return -1}
		switch o {
		case 0xd8, 0xda, 0xdc, 0xde:
			src := ""
			switch o {
			case 0xd8: src = fmt.tprintf("(double)MF32(%s)", ea)
			case 0xda: src = fmt.tprintf("(double)(s32)M32(%s)", ea)
			case 0xdc: src = fmt.tprintf("MF64(%s)", ea)
			case 0xde: src = fmt.tprintf("(double)(s16)M16(%s)", ea)
			}
			s0 := st(xl, d, 0)
			switch reg {
			case 2: x87_cmp(xl, s0, src); return d
			case 3: x87_cmp(xl, s0, src); return d - 1
			}
			pf(xl, "    %s = %s;", s0, x87_arith(reg, s0, src))
			return d
		case 0xd9:
			switch reg {
			case 0: return x87_push(xl, d, fmt.tprintf("(double)MF32(%s)", ea))
			case 2: pf(xl, "    MF32(%s) = (float)%s;", ea, st(xl, d, 0)); return d
			case 3: pf(xl, "    MF32(%s) = (float)%s;", ea, st(xl, d, 0)); return d - 1
			case 5: pf(xl, "    XRC_FLDCW(M16(%s))", ea); return d
			case 7: pf(xl, "    M16(%s) = (u16)x_fcw;", ea); return d
			}
		case 0xdb:
			switch reg {
			case 0: return x87_push(xl, d, fmt.tprintf("(double)(s32)M32(%s)", ea))
			case 1: pf(xl, "    M32(%s) = xrc_i32(xrc_rint(%s, 3));", ea, st(xl, d, 0)); return d - 1
			case 2: pf(xl, "    M32(%s) = xrc_i32(xrc_rint(%s, x_fcw >> 10));", ea, st(xl, d, 0)); return d
			case 3: pf(xl, "    M32(%s) = xrc_i32(xrc_rint(%s, x_fcw >> 10));", ea, st(xl, d, 0)); return d - 1
			}
		case 0xdd:
			switch reg {
			case 0: return x87_push(xl, d, fmt.tprintf("MF64(%s)", ea))
			case 1: pf(xl, "    M64(%s) = xrc_i64(xrc_rint(%s, 3));", ea, st(xl, d, 0)); return d - 1
			case 2: pf(xl, "    MF64(%s) = %s;", ea, st(xl, d, 0)); return d
			case 3: pf(xl, "    MF64(%s) = %s;", ea, st(xl, d, 0)); return d - 1
			case 7: pf(xl, "    M16(%s) = (u16)%s;", ea, x87_fsw(d)); return d
			}
		case 0xdf:
			switch reg {
			case 0: return x87_push(xl, d, fmt.tprintf("(double)(s16)M16(%s)", ea))
			case 1: pf(xl, "    M16(%s) = xrc_i16(xrc_rint(%s, 3));", ea, st(xl, d, 0)); return d - 1
			case 2: pf(xl, "    M16(%s) = xrc_i16(xrc_rint(%s, x_fcw >> 10));", ea, st(xl, d, 0)); return d
			case 3: pf(xl, "    M16(%s) = xrc_i16(xrc_rint(%s, x_fcw >> 10));", ea, st(xl, d, 0)); return d - 1
			case 5: return x87_push(xl, d, fmt.tprintf("xrc_i64d(M64(%s))", ea))
			case 7: pf(xl, "    M64(%s) = xrc_i64(xrc_rint(%s, x_fcw >> 10));", ea, st(xl, d, 0)); return d - 1
			}
		}
		fail(xl, mn)
		return -1
	}
	// register forms
	i := rm
	switch o {
	case 0xd8:
		s0, si := st(xl, d, 0), st(xl, d, i)
		switch reg {
		case 2: x87_cmp(xl, s0, si); return d
		case 3: x87_cmp(xl, s0, si); return d - 1
		}
		pf(xl, "    %s = %s;", s0, x87_arith(reg, s0, si))
		return d
	case 0xd9:
		switch ins.modrm {
		case 0xc0 ..= 0xc7: return x87_push(xl, d, st(xl, d, i))
		case 0xc8 ..= 0xcf:
			s0, si := st(xl, d, 0), st(xl, d, i)
			pf(xl, "    { double _t = %s; %s = %s; %s = _t; }", s0, s0, si, si)
			return d
		case 0xd0: return d
		case 0xe0: s0 := st(xl, d, 0); pf(xl, "    %s = -%s;", s0, s0); return d
		case 0xe1: s0 := st(xl, d, 0); pf(xl, "    %s = xrc_fabs(%s);", s0, s0); return d
		case 0xe4: x87_cmp(xl, st(xl, d, 0), "0.0"); return d
		case 0xe8: return x87_push(xl, d, "1.0")
		case 0xe9: return x87_push(xl, d, "3.3219280948873622") // log2(10)
		case 0xea: return x87_push(xl, d, "1.4426950408889634") // log2(e)
		case 0xeb: return x87_push(xl, d, "3.1415926535897931")
		case 0xec: return x87_push(xl, d, "0.30102999566398120") // log10(2)
		case 0xed: return x87_push(xl, d, "0.69314718055994531") // ln(2)
		case 0xee: return x87_push(xl, d, "0.0")
		case 0xf0: s0 := st(xl, d, 0); pf(xl, "    %s = XR87(xf_f2xm1(%s));", s0, s0); return d
		case 0xf1:
			s0, s1 := st(xl, d, 0), st(xl, d, 1)
			pf(xl, "    %s = XR87(xf_fyl2x(%s, %s));", s1, s1, s0)
			return d - 1
		case 0xf2:
			s0 := st(xl, d, 0)
			pf(xl, "    %s = XR87(xf_fptan(%s));", s0, s0)
			return x87_push(xl, d, "1.0")
		case 0xf3:
			s0, s1 := st(xl, d, 0), st(xl, d, 1)
			pf(xl, "    %s = XR87(xf_fpatan(%s, %s));", s1, s1, s0)
			return d - 1
		case 0xf5, 0xf8:
			s0, s1 := st(xl, d, 0), st(xl, d, 1)
			pf(xl, "    %s = %s(%s, %s); x_fsw &= ~0x400u;", s0, ins.modrm == 0xf5 ? "xf_fprem1" : "xf_fprem", s0, s1)
			return d
		case 0xfa: s0 := st(xl, d, 0); pf(xl, "    %s = XR87(xsqrt(%s));", s0, s0); return d
		case 0xfb:
			s0 := st(xl, d, 0)
			if d >= 8 {fail(xl, "x87 overflow"); return -1}
			pf(xl, "    { double _t = %s; %s = XR87(xf_fsin(_t)); f%d = XR87(xf_fcos(_t)); }", s0, s0, d)
			return d + 1
		case 0xfc: s0 := st(xl, d, 0); pf(xl, "    %s = xrc_rint(%s, x_fcw >> 10);", s0, s0); return d
		case 0xfd:
			s0, s1 := st(xl, d, 0), st(xl, d, 1)
			pf(xl, "    %s = xf_fscale(%s, %s);", s0, s0, s1)
			return d
		case 0xfe: s0 := st(xl, d, 0); pf(xl, "    %s = XR87(xf_fsin(%s));", s0, s0); return d
		case 0xff: s0 := st(xl, d, 0); pf(xl, "    %s = XR87(xf_fcos(%s));", s0, s0); return d
		}
	case 0xda:
		if ins.modrm == 0xe9 { // fucompp
			x87_cmp(xl, st(xl, d, 0), st(xl, d, 1))
			return d - 2
		}
		if reg <= 3 { // fcmovb / fcmove / fcmovbe / fcmovu
			tb := [4]int{2, 4, 6, 10}; cc := tb[reg]
			s0, si := st(xl, d, 0), st(xl, d, i)
			pf(xl, "    if (%s) %s = %s;", x_cond_expr(xl, cc), s0, si)
			return d
		}
	case 0xdb:
		if reg <= 3 { // fcmovnb / fcmovne / fcmovnbe / fcmovnu
			tb := [4]int{3, 5, 7, 11}; cc := tb[reg]
			s0, si := st(xl, d, 0), st(xl, d, i)
			pf(xl, "    if (%s) %s = %s;", x_cond_expr(xl, cc), s0, si)
			return d
		}
		if ins.modrm == 0xe2 {return d} // fnclex
		if reg == 5 || reg == 6 { // fucomi / fcomi
			x87_fcomi(xl, st(xl, d, 0), st(xl, d, i))
			return d
		}
	case 0xdc, 0xde:
		if o == 0xde && ins.modrm == 0xd9 { // fcompp
			x87_cmp(xl, st(xl, d, 0), st(xl, d, 1))
			return d - 2
		}
		if reg == 2 || reg == 3 {break}
		// st(i) = st(i) op st(0); the sub/subr and div/divr encodings are swapped relative to D8
		op := reg
		switch reg {
		case 4: op = 5
		case 5: op = 4
		case 6: op = 7
		case 7: op = 6
		}
		s0, si := st(xl, d, 0), st(xl, d, i)
		pf(xl, "    %s = %s;", si, x87_arith(op, si, s0))
		return o == 0xde ? d - 1 : d
	case 0xdd:
		switch reg {
		case 0: return d // ffree
		case 2: s0, si := st(xl, d, 0), st(xl, d, i); pf(xl, "    %s = %s;", si, s0); return d
		case 3:
			s0, si := st(xl, d, 0), st(xl, d, i)
			if i != 0 {pf(xl, "    %s = %s;", si, s0)}
			return d - 1
		case 4: x87_cmp(xl, st(xl, d, 0), st(xl, d, i)); return d
		case 5: x87_cmp(xl, st(xl, d, 0), st(xl, d, i)); return d - 1
		}
	case 0xdf:
		if ins.modrm == 0xe0 { // fnstsw ax
			pf(xl, "    SET16(r_eax, %s);", x87_fsw(d))
			return d
		}
		if reg == 5 || reg == 6 { // fucomip / fcomip
			x87_fcomi(xl, st(xl, d, 0), st(xl, d, i))
			return d - 1
		}
	}
	fail(xl, mn)
	return -1
}

// ---------------------------------------------------------------------------------------------------------------
// SSE emission

@(private = "file")
is_x :: proc(op: x86.Operand) -> bool {return op.kind == .REGISTER && x86.reg_class(op.reg) == x86.REG_XMM}
@(private = "file")
is_mmx :: proc(op: x86.Operand) -> bool {return op.kind == .REGISTER && x86.reg_class(op.reg) == x86.REG_MM}
@(private = "file")
xn :: proc(op: x86.Operand) -> int {return int(x86.reg_hw(op.reg))}

// "X128 _s ..." -- the source operand as a 128-bit temp, reading only `bytes` of a memory operand
@(private = "file")
sse_src :: proc(op: x86.Operand, ea: string, bytes: int) -> (string, bool) {
	if is_x(op) {return fmt.tprintf("X128 _s = X[%d];", xn(op)), true}
	if op.kind != .MEMORY {return "", false}
	switch bytes {
	case 4: return fmt.tprintf("X128 _s; _s.u[0] = M32(%s); _s.u[1] = 0; _s.q[1] = 0;", ea), true
	case 8: return fmt.tprintf("X128 _s; _s.q[0] = M64(%s); _s.q[1] = 0;", ea), true
	}
	return fmt.tprintf("X128 _s; XLD(_s, %s);", ea), true
}

// lane-wise op: for lane i in 0..<lanes, _r.<rt>[i] = tmpl with $a = _d.<st>[i], $b = _s.<st>[i], $i = i.
// Lanes past `lanes` keep the destination's value (scalar ss/sd forms).
@(private = "file")
sse_lanes :: proc(xl: ^Xlat, ops: []x86.Operand, ea: string, bytes, lanes: int, rt, st_: string, tmpl: string) -> bool {
	if len(ops) < 2 || !is_x(ops[0]) {return false}
	src, ok := sse_src(ops[1], ea, bytes)
	if !ok {return false}
	d := xn(ops[0])
	sb := strings.builder_make(context.temp_allocator)
	fmt.sbprintf(&sb, "    {{ X128 _d = X[%d], _r = X[%d]; %s", d, d, src)
	for i in 0 ..< lanes {
		e, _ := strings.replace_all(tmpl, "$a", fmt.tprintf("_d.%s[%d]", st_, i), context.temp_allocator)
		e, _ = strings.replace_all(e, "$b", fmt.tprintf("_s.%s[%d]", st_, i), context.temp_allocator)
		e, _ = strings.replace_all(e, "$i", fmt.tprintf("%d", i), context.temp_allocator)
		fmt.sbprintf(&sb, " _r.%s[%d] = %s;", rt, i, e)
	}
	fmt.sbprintf(&sb, " X[%d] = _r; }}", d)
	p(xl, strings.to_string(sb), "\n")
	return true
}

// explicit lane assignments (shuffles): each item is "rt[k] = expr" over _d/_s
@(private = "file")
sse_shuffle :: proc(xl: ^Xlat, ops: []x86.Operand, ea: string, bytes: int, assigns: []string) -> bool {
	if len(ops) < 2 || !is_x(ops[0]) {return false}
	src, ok := sse_src(ops[1], ea, bytes)
	if !ok {return false}
	d := xn(ops[0])
	sb := strings.builder_make(context.temp_allocator)
	fmt.sbprintf(&sb, "    {{ X128 _d = X[%d], _r = X[%d]; %s", d, d, src)
	for a in assigns {fmt.sbprintf(&sb, " _r.%s;", a)}
	fmt.sbprintf(&sb, " X[%d] = _r; }}", d)
	p(xl, strings.to_string(sb), "\n")
	return true
}

// a packed op on whole registers: X[d] = fn(X[d], <16-byte source>), fn one of xrc.h's xv_* (the host's SSE
// instruction where it has SSE2, the lane-wise C otherwise)
@(private = "file")
sse_vec :: proc(xl: ^Xlat, ops: []x86.Operand, ea: string, fn: string) -> bool {
	if len(ops) < 2 || !is_x(ops[0]) {return false}
	d := xn(ops[0])
	if is_x(ops[1]) {
		pf(xl, "    X[%d] = %s(X[%d], X[%d]);", d, fn, d, xn(ops[1]))
		return true
	}
	src, ok := sse_src(ops[1], ea, 16)
	if !ok {return false}
	pf(xl, "    {{ %s X[%d] = %s(X[%d], _s); }}", src, d, fn, d)
	return true
}

@(private = "file")
cmp_pred :: proc(k: i64, a, b: string) -> string {
	switch k & 7 {
	case 0: return fmt.tprintf("(%s == %s)", a, b)
	case 1: return fmt.tprintf("(%s < %s)", a, b)
	case 2: return fmt.tprintf("(%s <= %s)", a, b)
	// (NaN on the bits, xrc_isnan: the unordered outcomes hold under -ffast-math too -- the launcher's Fast Math)
	case 3: return fmt.tprintf("(xrc_isnan(%s) || xrc_isnan(%s))", a, b)
	case 4: return fmt.tprintf("(xrc_isnan(%s) || xrc_isnan(%s) || !(%s == %s))", a, b, a, b)
	case 5: return fmt.tprintf("(xrc_isnan(%s) || xrc_isnan(%s) || !(%s < %s))", a, b, a, b)
	case 6: return fmt.tprintf("(xrc_isnan(%s) || xrc_isnan(%s) || !(%s <= %s))", a, b, a, b)
	}
	return fmt.tprintf("(!xrc_isnan(%s) && !xrc_isnan(%s))", a, b)
}

// a 32-bit scalar float/double source: xmm lane 0 or memory
@(private = "file")
sse_scalar :: proc(op: x86.Operand, ea: string, dbl: bool) -> (string, bool) {
	if is_x(op) {return fmt.tprintf("X[%d].%s[0]", xn(op), dbl ? "d" : "f"), true}
	if op.kind == .MEMORY {return fmt.tprintf("%s(%s)", dbl ? "MF64" : "MF32", ea), true}
	return "", false
}

// returns false when the instruction is not an SSE instruction we handle (caller fails with its mnemonic)
sse_emit :: proc(xl: ^Xlat, ins: ^Rx_Ins, ops: []x86.Operand, ea: string, have_ea: bool) -> bool {
	m := ins.inst.mnemonic
	n := len(ops)
	for o in ops {if is_mmx(o) {fail(xl, "mmx"); return true}}
	imm: i64 = 0
	if n >= 3 && ops[2].kind == .IMMEDIATE {imm = ops[2].immediate}
	// string movsd/cmpsd share mnemonics with the SSE2 ones
	sse2_sd := ins.opc == 0x0f
	xl_x :: proc(xl: ^Xlat) {xl.uses_x = true}

	#partial switch m {
	case .PREFETCHT0, .PREFETCHT1, .PREFETCHT2, .PREFETCHNTA, .SFENCE, .LFENCE, .MFENCE:
		return true
	// ---- moves
	case .MOVSS, .MOVSD:
		if m == .MOVSD && !sse2_sd {return false}
		xl_x(xl)
		dbl := m == .MOVSD
		if is_x(ops[0]) && is_x(ops[1]) {
			pf(xl, dbl ? "    X[%d].q[0] = X[%d].q[0];" : "    X[%d].u[0] = X[%d].u[0];", xn(ops[0]), xn(ops[1]))
		} else if is_x(ops[0]) && ops[1].kind == .MEMORY {
			if dbl {
				pf(xl, "    X[%d].q[0] = M64(%s); X[%d].q[1] = 0;", xn(ops[0]), ea, xn(ops[0]))
			} else {
				pf(xl, "    X[%d].u[0] = M32(%s); X[%d].u[1] = 0; X[%d].q[1] = 0;", xn(ops[0]), ea, xn(ops[0]), xn(ops[0]))
			}
		} else if ops[0].kind == .MEMORY && is_x(ops[1]) {
			pf(xl, dbl ? "    M64(%s) = X[%d].q[0];" : "    M32(%s) = X[%d].u[0];", ea, xn(ops[1]))
		} else {return false}
		return true
	case .MOVAPS, .MOVUPS, .MOVAPD, .MOVUPD, .MOVDQA, .MOVDQU, .MOVNTPS, .MOVNTPD, .MOVNTDQ, .LDDQU:
		xl_x(xl)
		if is_x(ops[0]) && is_x(ops[1]) {
			pf(xl, "    X[%d] = X[%d];", xn(ops[0]), xn(ops[1]))
		} else if is_x(ops[0]) && ops[1].kind == .MEMORY {
			pf(xl, "    XLD(X[%d], %s);", xn(ops[0]), ea)
		} else if ops[0].kind == .MEMORY && is_x(ops[1]) {
			pf(xl, "    XST(%s, X[%d]);", ea, xn(ops[1]))
		} else {return false}
		return true
	case .MOVLPS, .MOVLPD, .MOVHPS, .MOVHPD:
		xl_x(xl)
		k := (m == .MOVHPS || m == .MOVHPD) ? 1 : 0
		if is_x(ops[0]) && ops[1].kind == .MEMORY {
			pf(xl, "    X[%d].q[%d] = M64(%s);", xn(ops[0]), k, ea)
		} else if ops[0].kind == .MEMORY && is_x(ops[1]) {
			pf(xl, "    M64(%s) = X[%d].q[%d];", ea, xn(ops[1]), k)
		} else {return false}
		return true
	case .MOVHLPS:
		xl_x(xl)
		pf(xl, "    X[%d].q[0] = X[%d].q[1];", xn(ops[0]), xn(ops[1]))
		return true
	case .MOVLHPS:
		xl_x(xl)
		pf(xl, "    X[%d].q[1] = X[%d].q[0];", xn(ops[0]), xn(ops[1]))
		return true
	case .MOVD:
		xl_x(xl)
		if is_x(ops[0]) {
			v, ok := x_val(xl, ops[1], ea, have_ea)
			if !ok {return false}
			d := xn(ops[0])
			pf(xl, "    { u32 _v = %s; X[%d].u[0] = _v; X[%d].u[1] = 0; X[%d].q[1] = 0; }", v, d, d, d)
		} else if is_x(ops[1]) {
			pf(xl, "    %s", x_store(ops[0], fmt.tprintf("X[%d].u[0]", xn(ops[1])), ea))
		} else {return false}
		return true
	case .MOVQ:
		xl_x(xl)
		if is_x(ops[0]) && is_x(ops[1]) {
			pf(xl, "    X[%d].q[0] = X[%d].q[0]; X[%d].q[1] = 0;", xn(ops[0]), xn(ops[1]), xn(ops[0]))
		} else if is_x(ops[0]) && ops[1].kind == .MEMORY {
			pf(xl, "    X[%d].q[0] = M64(%s); X[%d].q[1] = 0;", xn(ops[0]), ea, xn(ops[0]))
		} else if ops[0].kind == .MEMORY && is_x(ops[1]) {
			pf(xl, "    M64(%s) = X[%d].q[0];", ea, xn(ops[1]))
		} else {return false}
		return true
	case .MOVMSKPS, .MOVMSKPD, .PMOVMSKB:
		xl_x(xl)
		if !is_x(ops[1]) {return false}
		s := xn(ops[1])
		sb := strings.builder_make(context.temp_allocator)
		#partial switch m {
		case .MOVMSKPS: for i in 0 ..< 4 {fmt.sbprintf(&sb, "%s((X[%d].u[%d] >> 31) << %d)", i > 0 ? " | " : "", s, i, i)}
		case .MOVMSKPD: for i in 0 ..< 2 {fmt.sbprintf(&sb, "%s((X[%d].u[%d] >> 31) << %d)", i > 0 ? " | " : "", s, 2 * i + 1, i)}
		case: for i in 0 ..< 16 {fmt.sbprintf(&sb, "%s(((u32)X[%d].b[%d] >> 7) << %d)", i > 0 ? " | " : "", s, i, i)}
		}
		pf(xl, "    %s", x_reg_write(ops[0].reg, strings.to_string(sb)))
		return true

	// ---- float arithmetic
	case .ADDPS: xl_x(xl); return sse_vec(xl, ops, ea, "xv_addps")
	case .SUBPS: xl_x(xl); return sse_vec(xl, ops, ea, "xv_subps")
	case .MULPS: xl_x(xl); return sse_vec(xl, ops, ea, "xv_mulps")
	case .DIVPS: xl_x(xl); return sse_vec(xl, ops, ea, "xv_divps")
	case .MINPS: xl_x(xl); return sse_vec(xl, ops, ea, "xv_minps")
	case .MAXPS: xl_x(xl); return sse_vec(xl, ops, ea, "xv_maxps")
	case .SQRTPS: xl_x(xl); return sse_vec(xl, ops, ea, "xv_sqrtps")
	case .RCPPS: xl_x(xl); return sse_vec(xl, ops, ea, "xv_rcpps")
	case .RSQRTPS: xl_x(xl); return sse_vec(xl, ops, ea, "xv_rsqrtps")
	case .ADDSS: xl_x(xl); return sse_lanes(xl, ops, ea, 4, 1, "f", "f", "$a + $b")
	case .SUBSS: xl_x(xl); return sse_lanes(xl, ops, ea, 4, 1, "f", "f", "$a - $b")
	case .MULSS: xl_x(xl); return sse_lanes(xl, ops, ea, 4, 1, "f", "f", "$a * $b")
	case .DIVSS: xl_x(xl); return sse_lanes(xl, ops, ea, 4, 1, "f", "f", "$a / $b")
	case .MINSS: xl_x(xl); return sse_lanes(xl, ops, ea, 4, 1, "f", "f", "$a < $b ? $a : $b")
	case .MAXSS: xl_x(xl); return sse_lanes(xl, ops, ea, 4, 1, "f", "f", "$a > $b ? $a : $b")
	case .SQRTSS: xl_x(xl); return sse_lanes(xl, ops, ea, 4, 1, "f", "f", "xsqrtf($b)")
	case .RCPSS: xl_x(xl); return sse_lanes(xl, ops, ea, 4, 1, "f", "f", "xrcpf($b)")
	case .RSQRTSS: xl_x(xl); return sse_lanes(xl, ops, ea, 4, 1, "f", "f", "xrsqrtf($b)")
	case .ADDPD: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 2, "d", "d", "$a + $b")
	case .SUBPD: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 2, "d", "d", "$a - $b")
	case .MULPD: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 2, "d", "d", "$a * $b")
	case .DIVPD: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 2, "d", "d", "$a / $b")
	case .MINPD: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 2, "d", "d", "$a < $b ? $a : $b")
	case .MAXPD: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 2, "d", "d", "$a > $b ? $a : $b")
	case .SQRTPD: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 2, "d", "d", "xsqrt($b)")
	case .ADDSD: xl_x(xl); return sse_lanes(xl, ops, ea, 8, 1, "d", "d", "$a + $b")
	case .SUBSD: xl_x(xl); return sse_lanes(xl, ops, ea, 8, 1, "d", "d", "$a - $b")
	case .MULSD: xl_x(xl); return sse_lanes(xl, ops, ea, 8, 1, "d", "d", "$a * $b")
	case .DIVSD: xl_x(xl); return sse_lanes(xl, ops, ea, 8, 1, "d", "d", "$a / $b")
	case .MINSD: xl_x(xl); return sse_lanes(xl, ops, ea, 8, 1, "d", "d", "$a < $b ? $a : $b")
	case .MAXSD: xl_x(xl); return sse_lanes(xl, ops, ea, 8, 1, "d", "d", "$a > $b ? $a : $b")
	case .SQRTSD: xl_x(xl); return sse_lanes(xl, ops, ea, 8, 1, "d", "d", "xsqrt($b)")

	// ---- bitwise (float and integer forms)
	case .ANDPS, .ANDPD, .PAND: xl_x(xl); return sse_vec(xl, ops, ea, "xv_andps")
	case .ANDNPS, .ANDNPD, .PANDN: xl_x(xl); return sse_vec(xl, ops, ea, "xv_andnps")
	case .ORPS, .ORPD, .POR: xl_x(xl); return sse_vec(xl, ops, ea, "xv_orps")
	case .XORPS, .XORPD, .PXOR: xl_x(xl); return sse_vec(xl, ops, ea, "xv_xorps")

	// ---- compares
	case .CMPPS: xl_x(xl); return sse_vec(xl, ops, ea, fmt.tprintf("xv_cmpps%d", imm & 7))
	case .CMPSS: xl_x(xl); return sse_lanes(xl, ops, ea, 4, 1, "u", "f", fmt.tprintf("%s ? 0xffffffffu : 0u", cmp_pred(imm, "$a", "$b")))
	case .CMPPD: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 2, "q", "d", fmt.tprintf("%s ? ~(u64)0 : (u64)0", cmp_pred(imm, "$a", "$b")))
	case .CMPSD:
		if !sse2_sd {return false}
		xl_x(xl)
		return sse_lanes(xl, ops, ea, 8, 1, "q", "d", fmt.tprintf("%s ? ~(u64)0 : (u64)0", cmp_pred(imm, "$a", "$b")))
	case .COMISS, .UCOMISS, .COMISD, .UCOMISD:
		xl_x(xl)
		dbl := m == .COMISD || m == .UCOMISD
		if !is_x(ops[0]) {return false}
		b, ok := sse_scalar(ops[1], ea, dbl)
		if !ok {return false}
		x_fl_fcomi(xl, fmt.tprintf("(double)X[%d].%s[0]", xn(ops[0]), dbl ? "d" : "f"), fmt.tprintf("(double)%s", b))
		return true

	// ---- conversions
	case .CVTSI2SS, .CVTSI2SD:
		xl_x(xl)
		if !is_x(ops[0]) {return false}
		v, ok := x_val(xl, ops[1], ea, have_ea)
		if !ok {return false}
		if m == .CVTSI2SS {
			pf(xl, "    X[%d].f[0] = (float)(s32)(%s);", xn(ops[0]), v)
		} else {
			pf(xl, "    X[%d].d[0] = (double)(s32)(%s);", xn(ops[0]), v)
		}
		return true
	case .CVTTSS2SI, .CVTSS2SI, .CVTTSD2SI, .CVTSD2SI:
		xl_x(xl)
		dbl := m == .CVTTSD2SI || m == .CVTSD2SI
		b, ok := sse_scalar(ops[1], ea, dbl)
		if !ok || ops[0].kind != .REGISTER {return false}
		fnm := (m == .CVTTSS2SI || m == .CVTTSD2SI) ? "cvt_trunc" : "cvt_round"
		pf(xl, "    %s", x_reg_write(ops[0].reg, fmt.tprintf("%s((double)%s)", fnm, b)))
		return true
	case .CVTSS2SD: xl_x(xl); return sse_lanes(xl, ops, ea, 4, 1, "d", "f", "(double)$b")
	case .CVTSD2SS: xl_x(xl); return sse_lanes(xl, ops, ea, 8, 1, "f", "d", "(float)$b")
	case .CVTPS2PD: xl_x(xl); return sse_shuffle(xl, ops, ea, 8, {"d[0] = (double)_s.f[0]", "d[1] = (double)_s.f[1]"})
	case .CVTPD2PS: xl_x(xl); return sse_shuffle(xl, ops, ea, 16, {"f[0] = (float)_s.d[0]", "f[1] = (float)_s.d[1]", "q[1] = 0"})
	case .CVTDQ2PS: xl_x(xl); return sse_vec(xl, ops, ea, "xv_cvtdq2ps")
	case .CVTTPS2DQ: xl_x(xl); return sse_vec(xl, ops, ea, "xv_cvttps2dq")
	case .CVTPS2DQ: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 4, "u", "f", "cvt_round((double)$b)")
	case .CVTDQ2PD: xl_x(xl); return sse_shuffle(xl, ops, ea, 8, {"d[0] = (double)_s.s[0]", "d[1] = (double)_s.s[1]"})
	case .CVTTPD2DQ: xl_x(xl); return sse_shuffle(xl, ops, ea, 16, {"u[0] = cvt_trunc(_s.d[0])", "u[1] = cvt_trunc(_s.d[1])", "q[1] = 0"})
	case .CVTPD2DQ: xl_x(xl); return sse_shuffle(xl, ops, ea, 16, {"u[0] = cvt_round(_s.d[0])", "u[1] = cvt_round(_s.d[1])", "q[1] = 0"})

	// ---- shuffles / unpacks
	case .SHUFPS:
		xl_x(xl)
		if len(ops) < 2 || !is_x(ops[0]) {return false}
		if is_x(ops[1]) {
			pf(xl, "    X[%d] = XV_SHUFPS(X[%d], X[%d], 0x%x);", xn(ops[0]), xn(ops[0]), xn(ops[1]), u32(imm) & 0xff)
			return true
		}
		{
			src, ok := sse_src(ops[1], ea, 16)
			if !ok {return false}
			pf(xl, "    {{ %s X[%d] = XV_SHUFPS(X[%d], _s, 0x%x); }}", src, xn(ops[0]), xn(ops[0]), u32(imm) & 0xff)
		}
		return true
	case .SHUFPD:
		xl_x(xl)
		k := u32(imm)
		return sse_shuffle(xl, ops, ea, 16, {fmt.tprintf("q[0] = _d.q[%d]", k & 1), fmt.tprintf("q[1] = _s.q[%d]", (k >> 1) & 1)})
	case .PSHUFD:
		xl_x(xl)
		k := u32(imm)
		return sse_shuffle(xl, ops, ea, 16, {
			fmt.tprintf("u[0] = _s.u[%d]", k & 3), fmt.tprintf("u[1] = _s.u[%d]", (k >> 2) & 3),
			fmt.tprintf("u[2] = _s.u[%d]", (k >> 4) & 3), fmt.tprintf("u[3] = _s.u[%d]", (k >> 6) & 3)})
	case .PSHUFLW, .PSHUFHW:
		xl_x(xl)
		k := u32(imm)
		b := m == .PSHUFHW ? 4 : 0
		as := make([dynamic]string, context.temp_allocator)
		append(&as, m == .PSHUFHW ? "q[0] = _s.q[0]" : "q[1] = _s.q[1]")
		for i in 0 ..< 4 {append(&as, fmt.tprintf("w[%d] = _s.w[%d]", b + i, b + int((k >> (2 * u32(i))) & 3)))}
		return sse_shuffle(xl, ops, ea, 16, as[:])
	case .UNPCKLPS, .PUNPCKLDQ: xl_x(xl); return sse_vec(xl, ops, ea, "xv_unpcklps")
	case .UNPCKHPS, .PUNPCKHDQ: xl_x(xl); return sse_vec(xl, ops, ea, "xv_unpckhps")
	case .UNPCKLPD, .PUNPCKLQDQ: xl_x(xl); return sse_shuffle(xl, ops, ea, 16, {"q[0] = _d.q[0]", "q[1] = _s.q[0]"})
	case .UNPCKHPD, .PUNPCKHQDQ: xl_x(xl); return sse_shuffle(xl, ops, ea, 16, {"q[0] = _d.q[1]", "q[1] = _s.q[1]"})
	case .PUNPCKLWD, .PUNPCKHWD:
		xl_x(xl)
		b := m == .PUNPCKHWD ? 4 : 0
		as := make([dynamic]string, context.temp_allocator)
		for i in 0 ..< 4 {
			append(&as, fmt.tprintf("w[%d] = _d.w[%d]", 2 * i, b + i), fmt.tprintf("w[%d] = _s.w[%d]", 2 * i + 1, b + i))
		}
		return sse_shuffle(xl, ops, ea, 16, as[:])
	case .PUNPCKLBW, .PUNPCKHBW:
		xl_x(xl)
		b := m == .PUNPCKHBW ? 8 : 0
		as := make([dynamic]string, context.temp_allocator)
		for i in 0 ..< 8 {
			append(&as, fmt.tprintf("b[%d] = _d.b[%d]", 2 * i, b + i), fmt.tprintf("b[%d] = _s.b[%d]", 2 * i + 1, b + i))
		}
		return sse_shuffle(xl, ops, ea, 16, as[:])

	// ---- integer lanes
	case .PADDD: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 4, "u", "u", "$a + $b")
	case .PSUBD: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 4, "u", "u", "$a - $b")
	case .PADDQ: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 2, "q", "q", "$a + $b")
	case .PSUBQ: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 2, "q", "q", "$a - $b")
	case .PADDW: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 8, "w", "w", "(u16)($a + $b)")
	case .PSUBW: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 8, "w", "w", "(u16)($a - $b)")
	case .PADDB: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 16, "b", "b", "(u8)($a + $b)")
	case .PSUBB: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 16, "b", "b", "(u8)($a - $b)")
	case .PADDUSW: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 8, "w", "w", "(u16)((u32)$a + $b > 0xffffu ? 0xffffu : (u32)$a + $b)")
	case .PSUBUSW: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 8, "w", "w", "(u16)($a > $b ? $a - $b : 0)")
	case .PMAXSW: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 8, "sw", "sw", "$a > $b ? $a : $b")
	case .PMINSW: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 8, "sw", "sw", "$a < $b ? $a : $b")
	case .PMULLW: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 8, "w", "w", "(u16)((u32)$a * $b)")
	case .PCMPEQD: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 4, "u", "u", "$a == $b ? 0xffffffffu : 0u")
	case .PCMPEQW: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 8, "w", "w", "$a == $b ? 0xffffu : 0u")
	case .PCMPEQB: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 16, "b", "b", "$a == $b ? 0xffu : 0u")
	case .PCMPGTD: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 4, "u", "s", "$a > $b ? 0xffffffffu : 0u")
	case .PCMPGTW: xl_x(xl); return sse_lanes(xl, ops, ea, 16, 8, "w", "sw", "$a > $b ? 0xffffu : 0u")
	case .PMULUDQ: xl_x(xl); return sse_shuffle(xl, ops, ea, 16, {"q[0] = XMULU(_d.u[0], _s.u[0])", "q[1] = XMULU(_d.u[2], _s.u[2])"})
	case .PEXTRW:
		xl_x(xl)
		if ops[0].kind != .REGISTER || !is_x(ops[1]) {return false}
		pf(xl, "    %s", x_reg_write(ops[0].reg, fmt.tprintf("(u32)X[%d].w[%d]", xn(ops[1]), imm & 7)))
		return true
	case .PINSRW:
		xl_x(xl)
		if !is_x(ops[0]) {return false}
		v := ""
		if ops[1].kind == .MEMORY {v = fmt.tprintf("M16(%s)", ea)} else if ops[1].kind == .REGISTER {v = x_home(ops[1].reg)} else {return false}
		pf(xl, "    X[%d].w[%d] = (u16)(%s);", xn(ops[0]), imm & 7, v)
		return true
	case .PSLLD, .PSRLD, .PSRAD, .PSLLQ, .PSRLQ, .PSLLW, .PSRLW, .PSRAW:
		xl_x(xl)
		if !is_x(ops[0]) {return false}
		cnt := ""
		if ops[1].kind == .IMMEDIATE {
			cnt = fmt.tprintf("%du", u32(u8(ops[1].immediate)))
		} else {
			src, ok := sse_src(ops[1], ea, 16)
			if !ok {return false}
			pf(xl, "    { %s", src)
			cnt = "(_s.u[1] || _s.u[0] > 255u ? 255u : _s.u[0])"
		}
		d := xn(ops[0])
		sb := strings.builder_make(context.temp_allocator)
		fmt.sbprintf(&sb, "    {{ u32 _n = %s; X128 _r = X[%d];", cnt, d)
		#partial switch m {
		case .PSLLD: for i in 0 ..< 4 {fmt.sbprintf(&sb, " _r.u[%d] = _n > 31 ? 0 : _r.u[%d] << _n;", i, i)}
		case .PSRLD: for i in 0 ..< 4 {fmt.sbprintf(&sb, " _r.u[%d] = _n > 31 ? 0 : _r.u[%d] >> _n;", i, i)}
		case .PSRAD: for i in 0 ..< 4 {fmt.sbprintf(&sb, " _r.s[%d] = _r.s[%d] >> (_n > 31 ? 31 : _n);", i, i)}
		case .PSLLW: for i in 0 ..< 8 {fmt.sbprintf(&sb, " _r.w[%d] = (u16)(_n > 15 ? 0 : (u32)_r.w[%d] << _n);", i, i)}
		case .PSRLW: for i in 0 ..< 8 {fmt.sbprintf(&sb, " _r.w[%d] = (u16)(_n > 15 ? 0 : (u32)_r.w[%d] >> _n);", i, i)}
		case .PSRAW: for i in 0 ..< 8 {fmt.sbprintf(&sb, " _r.sw[%d] = (s16)((s32)_r.sw[%d] >> (_n > 15 ? 15 : _n));", i, i)}
		case .PSLLQ: for i in 0 ..< 2 {fmt.sbprintf(&sb, " _r.q[%d] = xsh64l(_r.q[%d], _n);", i, i)}
		case .PSRLQ: for i in 0 ..< 2 {fmt.sbprintf(&sb, " _r.q[%d] = xsh64r(_r.q[%d], _n);", i, i)}
		}
		fmt.sbprintf(&sb, " X[%d] = _r; }}", d)
		p(xl, strings.to_string(sb), "\n")
		if ops[1].kind != .IMMEDIATE {pf(xl, "    }")}
		return true
	}
	return false
}
