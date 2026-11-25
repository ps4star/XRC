package xrc

import "base:intrinsics"
import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import x86 "core:rexcode/isa/x86"

// Trimmed register syncs around calls and returns (xrc.h XRC_SM_ / XRC_FM_).
//
// A translated function keeps the guest registers in C locals (r_eax ...). A call stores all eight to XCtx before
// (the callee reads them there) and loads all eight back after; a return stores all eight. Most of that is not
// needed: a register the function has not changed since XCtx last had it is there already, and one that is written
// (or goes into another call) before it is read again need not be loaded. x_sync_plan works that out per function,
// over its control flow graph, from the C each instruction became:
//   - dirty (forward): the registers whose local may differ from XCtx -- written since the entry or the last call.
//     A call or return stores only those (x_sm_). After a call none is: XCtx has what the callee left.
//   - live (backward): the registers whose local is read before being written. A call loads only those back (x_fm_);
//     the others' locals are stale until written, which is fine as nothing reads them -- and nothing stores them:
//     a stale local is never dirty (a call's store reads the local, so it counts as a read, and the call before it
//     then loads it).
// An instruction's C reads a register where it names it, and writes it where it assigns it (r_x = / op= / ++ / &r_x /
// SET8L(r_x ...)); a write kills (makes dead) only for instructions that overwrite the whole register whatever they
// do (mov / movzx / movsx / lea / pop to a 32-bit register). Anything else that syncs registers itself -- an
// inlined body, a tail call, a devirtualized or cookie-guarded call, XRC_INL_PREEMPT, two calls in one instruction --
// keeps the full sync (the file-scope enum): it reads every local and leaves every register dirty.
// The masks go in a block's own enum around the macro: `{ enum { x_sm_ = .., x_fm_ = .. }; XCALLDK(...); }`.
// XRC_SYNC_TRIM=0: none (every call and return syncs everything, as before).

X_SYNC_ALL :: u8(0x7f)

x_sync_on: int = -1
x_sync_lo, x_sync_hi: u32
x_sync_sites, x_sync_st_saved, x_sync_ld_saved, x_sync_st_all, x_sync_ld_all: int

// hw register number -> mask bit (esp: none, always synced)
x_sync_bit :: proc(hw: int) -> u8 {
	B := [8]u8{1, 2, 4, 8, 0, 0x10, 0x20, 0x40}
	return hw >= 0 && hw < 8 ? B[hw] : 0
}

x_sync_idc :: proc(c: u8) -> bool {return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'}

// the registers this C reads and writes
x_sync_scan :: proc(t: string) -> (rd, wr: u8) {
	NAMES := [7]string{"r_eax", "r_ecx", "r_edx", "r_ebx", "r_ebp", "r_esi", "r_edi"}
	for k := 0; k + 5 <= len(t); k += 1 {
		if t[k] != 'r' || t[k + 1] != '_' || t[k + 2] != 'e' {continue}
		if k > 0 && x_sync_idc(t[k - 1]) {continue}
		if k + 5 < len(t) && x_sync_idc(t[k + 5]) {continue}
		bit: u8 = 0
		for nm, q in NAMES {if t[k:k + 5] == nm {bit = 1 << u8(q); break}}
		if bit == 0 {continue}
		isw, isr := false, true
		j := k + 5
		for j < len(t) && t[j] == ' ' {j += 1}
		if j < len(t) && t[j] == '=' && (j + 1 >= len(t) || t[j + 1] != '=') {
			isw, isr = true, false
		} else if j + 1 < len(t) && t[j + 1] == '=' && strings.index_byte("+-*/%&|^", t[j]) >= 0 {
			isw = true
		} else if j + 2 < len(t) && t[j + 2] == '=' && (t[j:j + 2] == "<<" || t[j:j + 2] == ">>") {
			isw = true
		} else if j + 1 < len(t) && ((t[j] == '+' && t[j + 1] == '+') || (t[j] == '-' && t[j + 1] == '-')) {
			isw = true
		}
		p := k - 1
		for p >= 0 && t[p] == ' ' {p -= 1}
		if p >= 0 && t[p] == '&' && !(p > 0 && t[p - 1] == '&') {isw = true}
		if p >= 1 && ((t[p] == '+' && t[p - 1] == '+') || (t[p] == '-' && t[p - 1] == '-')) {isw = true}
		if p >= 0 && t[p] == '(' {
			for m in ([3]string{"SET8L", "SET8H", "SET16"}) {if strings.has_suffix(t[:p], m) {isw = true}}
		}
		if isw {wr |= bit}
		if isr {rd |= bit}
	}
	if strings.contains(t, "XRC_DIV(") || strings.contains(t, "XRC_IDIV(") {rd |= 1 | 4; wr |= 1 | 4}
	return
}

x_sync_count :: proc(t, w: string) -> int {return strings.count(t, w)}

// the end of the macro call starting at `at` (its name): past the ')' matching its '(' and the ';' after
x_sync_macro_end :: proc(t: string, at: int) -> (int, bool) {
	o := strings.index_byte(t[at:], '(')
	if o < 0 {return 0, false}
	d := 0
	for q := at + o; q < len(t); q += 1 {
		if t[q] == '(' {d += 1} else if t[q] == ')' {
			d -= 1
			if d == 0 {
				e := q + 1
				if e < len(t) && t[e] == ';' {e += 1}
				return e, true
			}
		}
	}
	return 0, false
}

// the callee-saved registers an XCALL*DK keeps (its last argument: the ones it may change, bits ebx esi edi ebp)
x_sync_kept :: proc(t: string, at, end: int) -> u8 {
	name_end := at + strings.index_byte(t[at:], '(')
	name := t[at:name_end]
	if !strings.has_suffix(name, "DK") {return 0}
	cl := strings.last_index_byte(t[at:end], ')')
	cm := strings.last_index_byte(t[at:at + cl], ',')
	if cl < 0 || cm < 0 {return 0}
	v, ok := strconv.parse_int(strings.trim_space(t[at + cm + 1:at + cl]))
	if !ok {return 0}
	m := u8(v)
	kept: u8 = 0
	if m & 1 == 0 {kept |= 8}     // ebx
	if m & 2 == 0 {kept |= 0x20}  // esi
	if m & 4 == 0 {kept |= 0x40}  // edi
	if m & 8 == 0 {kept |= 0x10}  // ebp
	return kept
}

X_Sync_Kind :: enum u8 {Plain, Opaque, Call, Ret}

// (the switches, looked up once: before the translation threads start)
x_sync_init :: proc() {
	if x_sync_on >= 0 {return}
	x_sync_on = os.get_env("XRC_SYNC_TRIM", context.temp_allocator) == "0" ? 0 : 1
	// XRC_SYNC_RANGE=lo-hi (hex): only the functions starting there (bisecting a wrong trim)
	x_sync_hi = 0xffffffff
	if v := os.get_env("XRC_SYNC_RANGE", context.temp_allocator); v != "" {
		if d := strings.index_byte(v, '-'); d > 0 {
			lo, ok1 := strconv.parse_uint(v[:d], 16)
			hi, ok2 := strconv.parse_uint(v[d + 1:], 16)
			if ok1 && ok2 {x_sync_lo, x_sync_hi = u32(lo), u32(hi)}
		}
	}
}

x_sync_plan :: proc(xl: ^Xlat, ins: []Rx_Ins, spans: [][2]int, entry: u32) {
	x_sync_init()
	if x_sync_on == 0 {return}
	if entry < x_sync_lo || entry > x_sync_hi {return}
	n := len(ins)
	if n == 0 || len(spans) != n {return}
	ta := context.temp_allocator
	buf := string(xl.b.buf[:])
	kind := make([]X_Sync_Kind, n, ta)
	rd := make([]u8, n, ta)     // Plain: the reads; Call / Ret: before the macro
	wr := make([]u8, n, ta)     // Plain: the writes; Call / Ret: before the macro
	rd2 := make([]u8, n, ta)    // Call: after the macro
	wr2 := make([]u8, n, ta)
	kill := make([]u8, n, ta)
	kept := make([]u8, n, ta)
	mat := make([]int, n, ta)   // Call / Ret: where the macro starts, and ends
	mend := make([]int, n, ta)
	rk := make([]bool, n, ta)    // Call: an XCALLRDK (eax is its result: in the local, not in XCtx -- dirty after it)
	idx := make(map[u32]i32, n, ta)
	for &i, k in ins {idx[i.addr] = i32(k)}

	for &i, k in ins {
		t := buf[spans[k][0]:spans[k][1]]
		// (a back edge's XRC_INL_PREEMPT is no sync point: it stores every local only on its rare path, where a stale one is
		// harmless -- a register whose local is stale is one this function does not read again -- and loads nothing)
		if strings.contains(t, "XRC_INL_PREEMPT() goto") {t, _ = strings.replace_all(t, "XRC_INL_PREEMPT() ", "", context.temp_allocator)}
		ncall := x_sync_count(t, "XCALL") + x_sync_count(t, "XC_(") + x_sync_count(t, "XCD_(") + x_sync_count(t, "XCDK_(") + x_sync_count(t, "XCRDK_(")
		nret := x_sync_count(t, "XRC_RETURN")
		if ncall + nret > 1 || strings.contains(t, "/* inlined") || strings.contains(t, "XRC_INL_") || strings.contains(t, "x_dv_") ||
		   strings.contains(t, "_a != _b") || strings.contains(t, "XRC_SPILL") || strings.contains(t, "XRC_FILL") || strings.contains(t, "c->e") {
			kind[k] = .Opaque
			continue
		}
		if ncall + nret == 1 {
			// (only an instruction's own call or ret, which are unconditional: anything else keeps the full sync)
			if (ncall == 1 && i.inst.mnemonic != .CALL) || (nret == 1 && i.inst.mnemonic != .RET) {kind[k] = .Opaque; continue}
			at := ncall == 1 ? strings.index(t, "XCALL") : strings.index(t, "XRC_RETURN")
			if at < 0 {kind[k] = .Opaque; continue}
			e, ok := x_sync_macro_end(t, at)
			if !ok {kind[k] = .Opaque; continue}
			kind[k] = ncall == 1 ? .Call : .Ret
			if ncall == 1 && strings.has_prefix(t[at:], "XCALLRDK(") {rk[k] = true}
			mat[k], mend[k] = at, e
			rd[k], wr[k] = x_sync_scan(t[:at])
			// (the macro's own arguments are read before its stores: XCALL(r_esi) -- the target)
			ra, wa := x_sync_scan(t[at:e])
			rd[k] |= ra
			wr[k] |= wa
			rd2[k], wr2[k] = x_sync_scan(t[e:])
			if nret == 1 && xl.dabi {rd[k] |= 1} // (an xd_ function returns its eax local)
			if rk[k] {rd[k] |= 6} // (XCALLRDK passes the ecx and edx locals: they must be right there)
			if ncall == 1 {kept[k] = x_sync_kept(t, at, e)}
			continue
		}
		rd[k], wr[k] = x_sync_scan(t)
		m := i.inst.mnemonic
		if (m == .MOV || m == .MOVZX || m == .MOVSX || m == .LEA || m == .POP) && i.inst.operand_count >= 1 && i.inst.ops[0].kind == .REGISTER &&
		   x86.reg_class(i.inst.ops[0].reg) == x86.REG_GPR32 {
			kill[k] = x_sync_bit(int(x86.reg_hw(i.inst.ops[0].reg))) & wr[k]
		}
	}

	// edges: each instruction's C falls into the next one's (more edges than the code has is only conservative), and
	// its gotos to the instructions' labels
	succ := make([dynamic][2]i32, 0, 2 * n, ta)
	for k in 0 ..< n {
		if k + 1 < n {append(&succ, [2]i32{i32(k), i32(k + 1)})}
		t := buf[spans[k][0]:spans[k][1]]
		for q := 0; q + 15 <= len(t); q += 1 {
			if t[q] != 'g' || !strings.has_prefix(t[q:], "goto L_") {continue}
			h := t[q + 7:q + 15]
			if q + 15 < len(t) && x_sync_idc(t[q + 15]) {continue} // (an inlined body's own label: L_xxxxxxxx_site)
			v, ok := strconv.parse_uint(h, 16)
			if !ok {continue}
			if j, okj := idx[u32(v)]; okj {append(&succ, [2]i32{i32(k), j})}
		}
	}
	ent := i32(0)
	if j, ok := idx[entry]; ok {ent = j}

	// dirty: forward
	din := make([]u8, n, ta)
	dout := make([]u8, n, ta)
	// (an xd_ function gets ecx and edx in arguments: their locals are not in XCtx at its entry -- dirty there)
	XD_P :: u8(6)
	if xl.dabi {din[ent] = XD_P}
	for changed := true; changed; {
		changed = false
		for k in 0 ..< n {
			o: u8
			switch kind[k] {
			case .Plain: o = din[k] | wr[k]
			case .Opaque: o = X_SYNC_ALL
			case .Call: o = wr2[k] | (rk[k] ? 1 : 0)
			case .Ret: o = X_SYNC_ALL
			}
			if o != dout[k] {dout[k] = o; changed = true}
		}
		for e in succ {
			nv := din[e[1]] | dout[e[0]]
			if nv != din[e[1]] {din[e[1]] = nv; changed = true}
		}
	}
	// (otherwise the entry's dirty set is empty: the locals are loaded from XCtx -- every din starts so)
	// an xd_ function's ecx / edx that its own code never writes (on any way to a return) are not stored there: XCtx
	// already has them wherever its caller reads them back (XCALLRDK stores them first then, and the wrapper passes
	// XCtx's) -- may-written, forward
	mw := make([]u8, n, ta)
	mwo := make([]u8, n, ta)
	if xl.dabi {
		for changed := true; changed; {
			changed = false
			for k in 0 ..< n {
				o: u8
				switch kind[k] {
				case .Plain: o = mw[k] | wr[k]
				case .Opaque: o = X_SYNC_ALL
				case .Call: o = mw[k] | wr[k] | wr2[k]
				case .Ret: o = mw[k] | wr[k]
				}
				if o != mwo[k] {mwo[k] = o; changed = true}
			}
			for e in succ {
				nv := mw[e[1]] | mwo[e[0]]
				if nv != mw[e[1]] {mw[e[1]] = nv; changed = true}
			}
		}
	}

	// live: backward
	lin := make([]u8, n, ta)
	lout := make([]u8, n, ta)
	sm := make([]u8, n, ta)
	fm := make([]u8, n, ta)
	for changed := true; changed; {
		changed = false
		for e in succ {
			nv := lout[e[0]] | lin[e[1]]
			if nv != lout[e[0]] {lout[e[0]] = nv; changed = true}
		}
		for k := n - 1; k >= 0; k -= 1 {
			v: u8
			switch kind[k] {
			case .Plain: v = rd[k] | (lout[k] &~ kill[k])
			case .Opaque: v = X_SYNC_ALL
			case .Call:
				sm[k] = din[k] | wr[k]
				fm[k] = rd2[k] | lout[k]
				v = rd[k] | sm[k] | (fm[k] & kept[k])
			case .Ret:
				sm[k] = din[k] | wr[k]
				// (an xd_ function stores ecx / edx at its returns like any dirty register: XCtx has them after XCALLRDK, which
				// does not store them -- the game relies on registers a callee keeps, also caller-saved ones: whole-program
				// code generation; leaving a passed register unstored crashed 11-23)
				v = rd[k] | sm[k]
			}
			if v != lin[k] {lin[k] = v; changed = true}
		}
	}

	// the masks into the C
	any := false
	for k in 0 ..< n {if (kind[k] == .Call && (sm[k] != X_SYNC_ALL || fm[k] != X_SYNC_ALL)) || (kind[k] == .Ret && sm[k] != X_SYNC_ALL) {any = true; break}}
	if !any {return}
	nb := strings.builder_make(0, len(buf) + 64 * n)
	strings.write_string(&nb, buf[:spans[0][0]])
	for k in 0 ..< n {
		t := buf[spans[k][0]:spans[k][1]]
		if kind[k] != .Call && kind[k] != .Ret {strings.write_string(&nb, t); continue}
		f := kind[k] == .Call ? fm[k] : X_SYNC_ALL
		intrinsics.atomic_add(&x_sync_sites, 1)
		intrinsics.atomic_add(&x_sync_st_all, 7)
		intrinsics.atomic_add(&x_sync_st_saved, 7 - int(intrinsics.count_ones(sm[k] & X_SYNC_ALL)))
		if kind[k] == .Call {
			intrinsics.atomic_add(&x_sync_ld_all, int(intrinsics.count_ones(X_SYNC_ALL &~ kept[k])))
			intrinsics.atomic_add(&x_sync_ld_saved, int(intrinsics.count_ones(X_SYNC_ALL &~ kept[k] &~ f)))
		}
		strings.write_string(&nb, t[:mat[k]])
		fmt.sbprintf(&nb, "{{ enum {{ x_sm_ = 0x%x, x_fm_ = 0x%x }}; ", sm[k], f)
		strings.write_string(&nb, t[mat[k]:mend[k]])
		strings.write_string(&nb, " }")
		strings.write_string(&nb, t[mend[k]:])
	}
	strings.write_string(&nb, buf[spans[n - 1][1]:])
	strings.builder_destroy(&xl.b)
	xl.b = nb
}
