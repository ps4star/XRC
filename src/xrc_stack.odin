package xrc

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import x86 "core:rexcode/isa/x86"

// The stack balance of every function: the N of its `ret N` when every way out of it is one (or a tail jump to a
// function that is) with esp back where it was at the entry -- what the callers' code assumes of it. A direct call to
// such a function then leaves esp at its value before the call + N, which the call site writes as that (a constant
// offset from the frame's value the C compiler already has) instead of reading it back from XCtx, and its stale
// return-address slot unconditionally (x_call: the callee did not move the stack down).
// A tail jump (jmp, jcc or the fall into the next function) to such a function likewise sets esp after it to the value
// it had after the return address was taken off + N (x_post_tail). Indirect calls read esp back (their callee is not known).
//
// Per function, a forward data flow over its instructions of two offsets from the entry esp: esp's own, and ebp's
// (a frame pointer set by mov ebp, esp; mov esp, ebp / leave / lea esp, [ebp - x] get esp back from it after an
// alignment (and esp, -16), a call whose stack effect is not known, or alloca). A merge of two different values, or
// any other write, is "unknown". Across a call ebp is taken to be kept (what the function's own code after the call
// assumes too); a call's esp: + the callee's N when that is known, else unknown. The callees' N come from the
// previous round: every function again until none changes (unknown -> known only, so it ends). A function whose
// way out is not one of those (a computed jump that is no switch, a ret at another offset, rets of different N,
// none at all) has no N; nor does a hand-written or byte-matched one (its code is not what runs).
//
// XRC_STACK=0: off. XRC_STACK_VERIFY=1: each such call site checks XCtx's esp against the computed one
// (xrc_stack_bad: fatal, naming the callee).

X_STK_UNK :: i32(-0x7fffffff)

X_Stk_K :: enum u8 {
	None,      // nothing to esp / ebp
	Adj,       // esp += arg
	Esp_Unk,   // esp unknown
	Ebp_Unk,   // ebp unknown
	Both_Unk,  // both
	Ebp_Esp,   // ebp = esp + arg (mov ebp, esp / lea ebp, [esp + d])
	Esp_Ebp,   // esp = ebp + arg (mov esp, ebp / lea esp, [ebp + d])
	Leave,     // esp = ebp + 4, ebp unknown
	Pro,       // ebp = esp + arg, esp unknown (a call to a frame_prolog helper with its EBP)
	Pop_Ebp,   // esp += arg, ebp unknown (pop ebp)
	Ebp_Adj,   // ebp += arg
	Call,      // esp += callee's N (tgt), else unknown
	Ret,       // a way out: esp must be at 0; N = arg
	Tail,      // (a jump out to tgt, alone or as a jcc's taken edge): esp must be at 0; N = tgt's N
	Fail,      // reached: the function has no N
}

X_Stk_Op :: struct {
	k:   X_Stk_K,
	tail: bool,    // (a jcc / call-less op that also leaves to tgt: the Tail check besides its own effect)
	arg: i32,
	tgt: u32,
}

X_Stk_Fn :: struct {
	ops:   []X_Stk_Op,
	succ:  []i32, // CSR: succ[start[k] ..< start[k+1]]
	start: []i32,
	addr:  []u32, // (each op's instruction: the audit's reports)
	ok:    bool,  // decoded
}

x_stk_on := false
x_stk_frames := false // (XRC_STACK_FRAMES=1: frame_prolog EBP / frame_epilog calls followed; 11-24 with import N: +1% battle, off)
x_stk_verify := false
// the return-address check of a direct call (or an inlined body) to a stack-balanced function: left out (ret pops the
// very slot the call wrote -- the check only catches a callee that overwrote its own return address, a fatal error
// anyway); XRC_RET_VERIFY=1 keeps it
x_ret_nocheck := false
// no GIL preemption check at direct calls (XCALLDK mask 0x200): the back edge of every loop with a call in it has one
// instead (x_spin_plan) -- XRC_LEAF_NOPREEMPT=1, off: measured no gain (11-23: battle +0.4%, field +0.35%)
x_call_nopre := false
// the direct-call entry (XRC_DIRECT_ABI=0: off): a stack-balanced translated function F is emitted as
// `u32 xd_F(XCtx *c, XEA_T esp)` -- esp comes in an argument and eax goes out as the result, and its returns store
// neither them nor eip in XCtx -- plus the usual xf_F(XCtx *c) as a wrapper (XRC_DWRAP: the dispatcher, indirect calls, the
// runtime). Direct calls to F (XCALLRDK) then skip the stores and loads of esp, eax and eip around the call.
x_dabi_on := false
x_dabi_ok :: proc(a: u32) -> bool {return x_dabi_on && !x_stk_verify && x_balanced(a) && !(a in x_fast)} // (XRC_STACK_VERIFY: off -- an xd_ function leaves XCtx's esp stale, what the check reads)
x_balanced :: proc(target: u32) -> bool {_, ok := x_stk_n[target]; return ok && x_stk_on && target != 0 && target in x_direct_ok}
x_stk_n: map[u32]i32 // the functions with an N (read by x_call; written once, before the translation)

// Imports' stack effect: an import called through its IAT slot (call [slot]) pops its return address and N bytes of
// arguments -- the runtime's handler's ret N, measured by an XRC_IMPN_PROF build (xrc_rt.c: "dll name N" lines in
// XRC_IMPN_FILE, an import whose N differed between calls left out) in data\\import_stack.txt beside xrc.exe, or
// XRC_IMPN=FILE; XRC_IMPN=0: none. Such a call is then esp += N to the stack balance (its function can have an N), and
// the site checks esp after it (xrc_imp_bad: fatal, naming the import -- a handler that pops something else).
x_imp_n: map[u32]i32 // IAT slot -> N
x_imp_site :: proc(o: x86.Operand) -> (i32, bool) {
	if o.kind != .MEMORY || o.mem.segment != 0 || x86.mem_has_base(o.mem) || x86.mem_has_index(o.mem) {return 0, false}
	n, ok := x_imp_n[u32(o.mem.disp)]
	return n, ok
}
x_imp_n_load :: proc() {
	f := os.get_env("XRC_IMPN", context.temp_allocator)
	if f == "0" {return}
	if f == "" {
		f = table_path("import_stack.txt")
		if f == "" || !os.exists(f) {return}
	}
	txt, ok := read_text(f)
	if !ok {return}
	byname: map[string]i32
	defer delete(byname)
	for line in strings.split_lines_iterator(&txt) {
		fs := strings.fields(line, context.temp_allocator)
		if len(fs) != 3 || strings.has_prefix(fs[0], "#") {continue}
		v, vok := strconv.parse_int(fs[2])
		if !vok || v < 0 || v > 0x400 || v % 4 != 0 {continue}
		byname[fmt.tprintf("%s!%s", strings.to_lower(fs[0], context.temp_allocator), fs[1])] = i32(v)
	}
	for slot, name in exe.imp_slot {
		if v, has := byname[name]; has {x_imp_n[slot] = v}
	}
	fmt.eprintfln("xrc: %d import slots with a known stack effect (%s)", len(x_imp_n), f)
}

@(private = "file") stk_addrs: []u32
@(private = "file") stk_funcs: map[u32]u32
@(private = "file") stk_fns: []X_Stk_Fn
@(private = "file") stk_prev: []i32 // this round's input: N per function (index = stk_addrs order), X_STK_UNK none
@(private = "file") stk_next: []i32
@(private = "file") stk_idx: map[u32]i32
@(private = "file") stk_elig: []bool // translated code that runs as such (x_direct_ok or a fast override's original)

x_stk_reg :: proc(r: x86.Register) -> int {
	cl := x86.reg_class(r)
	if cl != x86.REG_GPR32 && cl != x86.REG_GPR16 {return -1}
	hw := int(x86.reg_hw(r))
	if hw == 4 {return 0} // esp
	if hw == 5 {return 1} // ebp
	return -1
}

// what i does to esp / ebp (bit 0 esp written, 1 ebp written) besides the cases x_stk_class knows
@(private = "file")
stk_writes :: proc(i: ^Rx_Ins) -> (w: u8) {
	mn := i.inst.mnemonic
	er := x86.ENCODE_RUNS[u16(mn)]
	for c in x86.CLOBBER_FORMS[er.start:][:er.count] {
		for k in 0 ..< int(i.inst.operand_count) {
			if k < 4 && k in c.written && i.inst.ops[k].kind == .REGISTER {
				if r := x_stk_reg(i.inst.ops[k].reg); r >= 0 {w |= 1 << u8(r)}
			}
		}
		if .RSP in c.implicit_wr {w |= 1}
		if .RBP in c.implicit_wr {w |= 2}
	}
	if (mn == .XCHG || mn == .XADD || mn == .CMPXCHG) && i.inst.operand_count == 2 {
		for k in 0 ..< 2 {if i.inst.ops[k].kind == .REGISTER {if r := x_stk_reg(i.inst.ops[k].reg); r >= 0 {w |= 1 << u8(r)}}}
	}
	if i.xop == .MOVD_ST && i.modrm >> 6 == 3 {
		switch i.modrm & 7 {case 4: w |= 1; case 5: w |= 2}
	}
	return
}

@(private = "file")
stk_class :: proc(i: ^Rx_Ins) -> X_Stk_Op {
	mn := i.inst.mnemonic
	n := int(i.inst.operand_count)
	ops := i.inst.ops
	w := i32(i.p66 ? 2 : 4)
	is_reg :: proc(o: x86.Operand, which: int) -> bool {return o.kind == .REGISTER && x86.reg_class(o.reg) == x86.REG_GPR32 && x_stk_reg(o.reg) == which}
	#partial switch mn {
	case .PUSH: return {k = .Adj, arg = -w}
	case .POP:
		if n == 1 && ops[0].kind == .REGISTER {
			r := x_stk_reg(ops[0].reg)
			if r == 0 {return {k = .Esp_Unk}}
			if r == 1 {return {k = i.p66 ? .Both_Unk : .Pop_Ebp, arg = 4}}
		}
		if n == 1 && ops[0].kind == .MEMORY && x86.mem_has_base(ops[0].mem) && x_stk_reg(x86.mem_base(ops[0].mem)) == 0 {return {k = .Esp_Unk}} // (pop [esp+x]: its address after the pop)
		return {k = .Adj, arg = w}
	case .PUSHF, .PUSHFD: return {k = .Adj, arg = -w}
	case .POPF, .POPFD: return {k = .Adj, arg = w}
	case .LEAVE: return i.p66 ? {k = .Both_Unk} : {k = .Leave}
	case .ENTER: return {k = .Both_Unk}
	case .MOV:
		if n == 2 && is_reg(ops[0], 0) && is_reg(ops[1], 1) {return {k = .Esp_Ebp}}
		if n == 2 && is_reg(ops[0], 1) && is_reg(ops[1], 0) {return {k = .Ebp_Esp}}
	case .LEA:
		if n == 2 && ops[1].kind == .MEMORY && ops[1].mem.segment == 0 && x86.mem_has_base(ops[1].mem) && !x86.mem_has_index(ops[1].mem) {
			b := x_stk_reg(x86.mem_base(ops[1].mem))
			d := ops[1].mem.disp
			if is_reg(ops[0], 0) && b == 1 {return {k = .Esp_Ebp, arg = d}}
			if is_reg(ops[0], 0) && b == 0 {return {k = .Adj, arg = d}}
			if is_reg(ops[0], 1) && b == 0 {return {k = .Ebp_Esp, arg = d}}
			if is_reg(ops[0], 1) && b == 1 {return {k = .Ebp_Adj, arg = d}}
		}
	case .ADD, .SUB:
		if n == 2 && ops[1].kind == .IMMEDIATE && (is_reg(ops[0], 0) || is_reg(ops[0], 1)) {
			d := i32(u32(ops[1].immediate))
			if mn == .SUB {d = -d}
			return {k = is_reg(ops[0], 0) ? .Adj : .Ebp_Adj, arg = d}
		}
	case .RET:
		nb := i32(0)
		if n > 0 && ops[0].kind == .IMMEDIATE {nb = i32(u16(ops[0].immediate))}
		return {k = .Ret, arg = nb}
	case .CALL:
		if n > 0 && ops[0].kind == .RELATIVE {
			t := rx_rel_target(i, 0)
			if t == i.addr + i.len {return {k = .Adj, arg = -4}} // (call $+5: the return address pushed, no call)
			if d, ok := cfg_prolog_ebp[t]; ok && x_stk_frames {return {k = .Pro, arg = d}}
			if t in cfg_epilogs && x_stk_frames && len(cfg_prolog_ebp) > 0 {return {k = .Leave}} // (esp = ebp + 4 when it returns)
			if t in cfg_prologs || t in cfg_epilogs {return {k = .Both_Unk}}
			return {k = .Call, tgt = t}
		}
		if nn, ok := x_imp_site(ops[0]); ok {return {k = .Adj, arg = nn}} // (call [IAT slot]: the import's ret N, x_imp_n -- the site checks it)
		return {k = .Esp_Unk} // (indirect: the callee's N not known)
	}
	switch stk_writes(i) {
	case 1: return {k = .Esp_Unk}
	case 2: return {k = .Ebp_Unk}
	case 3: return {k = .Both_Unk}
	}
	return {}
}

// decode a function into its ops and edges (per thread: x_par_run)
@(private = "file")
stk_decode :: proc(a, size: u32) -> (f: X_Stk_Fn) {
	ins, bad := rx_function(a, size, context.temp_allocator)
	if bad != 0 || len(ins) == 0 {return}
	n := len(ins)
	idx := make(map[u32]i32, n, context.temp_allocator)
	for &i, k in ins {idx[i.addr] = i32(k)}
	ops := make([]X_Stk_Op, n)
	addrs := make([]u32, n)
	succ := make([dynamic]i32, 0, n * 2)
	start := make([]i32, n + 1)
	for &i, k in ins {
		start[k] = i32(len(succ))
		op := stk_class(&i)
		mn := i.inst.mnemonic
		rel := i.inst.operand_count > 0 && i.inst.ops[0].kind == .RELATIVE
		fall := true
		if mn == .RET || mn == .INT3 || mn == .UD2 || mn == .HLT {fall = false}
		if mn == .JMP {
			fall = false
			if rel {
				t := rx_rel_target(&i, 0)
				if j, ok := idx[t]; ok {append(&succ, j)} else {op = {k = .Tail, tgt = t}}
			} else if t, ok := rx_jump_table(ins[:], k, a, size); ok {
				for s in t {if j, okj := idx[s]; okj {append(&succ, j)} else {op = {k = .Fail}}}
			} else {
				op = {k = .Fail} // (a computed jump out: a tail call through a pointer, or a switch not understood)
			}
		} else if rx_is_jcc(mn) && rel {
			t := rx_rel_target(&i, 0)
			if j, ok := idx[t]; ok {append(&succ, j)} else {op.tail = true; op.tgt = t}
		}
		if fall {
			if j, ok := idx[i.addr + i.len]; ok {
				append(&succ, j)
			} else if op.k == .None || op.k == .Adj {
				// (it runs on into the next function: a tail call there; x_translate does the same)
				if op.k == .Adj {op = {k = .Fail}} else {op.tail = true; op.tgt = i.addr + i.len}
			} else {
				op = {k = .Fail}
			}
		}
		addrs[k] = i.addr
		ops[k] = op
	}
	start[n] = i32(len(succ))
	return {ops = ops, succ = succ[:], start = start, addr = addrs, ok = true}
}

// the N of function fi from the callee Ns of the last round
@(private = "file")
stk_eval :: proc(fi: int) -> i32 {
	f := &stk_fns[fi]
	if !f.ok || !stk_elig[fi] {return X_STK_UNK}
	n := len(f.ops)
	esp := make([]i32, n, context.temp_allocator)
	ebp := make([]i32, n, context.temp_allocator)
	seen := make([]bool, n, context.temp_allocator)
	work := make([dynamic]i32, 0, 64, context.temp_allocator)
	esp[0], ebp[0], seen[0] = 0, X_STK_UNK, true
	append(&work, 0)
	res := X_STK_UNK
	have := false
	callee :: proc(t: u32) -> i32 {
		if j, ok := stk_idx[t]; ok {return stk_prev[j]}
		return X_STK_UNK
	}
	leave :: proc(e: i32, nn: i32, res: ^i32, have: ^bool) -> bool {
		if e != 0 || nn == X_STK_UNK {return false}
		if have^ && res^ != nn {return false}
		res^, have^ = nn, true
		return true
	}
	add :: proc(v, d: i32) -> i32 {return v == X_STK_UNK ? X_STK_UNK : v + d}
	for len(work) > 0 {
		k := pop(&work)
		e, b := esp[k], ebp[k]
		op := f.ops[k]
		if op.tail && !leave(e, callee(op.tgt), &res, &have) {return X_STK_UNK}
		switch op.k {
		case .None:
		case .Adj: e = add(e, op.arg)
		case .Esp_Unk: e = X_STK_UNK
		case .Ebp_Unk: b = X_STK_UNK
		case .Both_Unk: e, b = X_STK_UNK, X_STK_UNK
		case .Ebp_Esp: b = add(e, op.arg)
		case .Esp_Ebp: e = add(b, op.arg)
		case .Leave: e, b = add(b, 4), X_STK_UNK
		case .Pro: e, b = X_STK_UNK, add(e, op.arg)
		case .Pop_Ebp: e, b = add(e, op.arg), X_STK_UNK
		case .Ebp_Adj: b = add(b, op.arg)
		case .Call:
			nn := callee(op.tgt)
			e = nn == X_STK_UNK ? X_STK_UNK : add(e, nn)
		case .Ret: if !leave(e, op.arg, &res, &have) {return X_STK_UNK}
		case .Tail: if !leave(e, callee(op.tgt), &res, &have) {return X_STK_UNK}
		case .Fail: return X_STK_UNK
		}
		for s in f.succ[f.start[k]:f.start[k + 1]] {
			if !seen[s] {
				esp[s], ebp[s], seen[s] = e, b, true
				append(&work, s)
				continue
			}
			ne, nb := esp[s], ebp[s]
			if ne != e {ne = X_STK_UNK}
			if nb != b {nb = X_STK_UNK}
			if ne != esp[s] || nb != ebp[s] {
				esp[s], ebp[s] = ne, nb
				append(&work, s)
			}
		}
	}
	return have ? res : X_STK_UNK
}

x_stack_plan :: proc(addrs: []u32, funcs: map[u32]u32) {
	x_stk_n = make(map[u32]i32)
	x_stk_on = os.get_env("XRC_STACK", context.temp_allocator) != "0"
	x_stk_frames = os.get_env("XRC_STACK_FRAMES", context.temp_allocator) == "1"
	if x_stk_on {x_imp_n_load()}
	x_stk_verify = os.get_env("XRC_STACK_VERIFY", context.temp_allocator) != ""
	x_ret_nocheck = os.get_env("XRC_RET_VERIFY", context.temp_allocator) == "" && !x_stk_verify
	x_call_nopre = os.get_env("XRC_LEAF_NOPREEMPT", context.temp_allocator) != ""
	if !x_stk_on {return}
	stk_addrs, stk_funcs = addrs, funcs
	stk_fns = make([]X_Stk_Fn, len(addrs))
	stk_idx = make(map[u32]i32, len(addrs))
	stk_elig = make([]bool, len(addrs))
	for a, i in addrs {
		stk_idx[a] = i32(i)
		stk_elig[i] = a in x_direct_ok || a in x_fast
	}
	x_par_run(len(addrs), proc(i: int) {stk_fns[i] = stk_decode(stk_addrs[i], stk_funcs[stk_addrs[i]])})
	stk_prev = make([]i32, len(addrs))
	stk_next = make([]i32, len(addrs))
	for &v in stk_prev {v = X_STK_UNK}
	rounds := 0
	for {
		rounds += 1
		x_par_run(len(addrs), proc(i: int) {stk_next[i] = stk_eval(i)})
		changed := 0
		for i in 0 ..< len(addrs) {if stk_next[i] != stk_prev[i] {changed += 1}}
		stk_prev, stk_next = stk_next, stk_prev
		if changed == 0 || rounds >= 64 {break}
	}
	n := 0
	for a, i in addrs {
		if stk_prev[i] != X_STK_UNK {x_stk_n[a] = stk_prev[i]; n += 1}
	}
	stk_why_all()
	stk_audit_all()
	for &f in stk_fns {delete(f.ops); delete(f.succ); delete(f.start); delete(f.addr)}
	delete(stk_fns); delete(stk_prev); delete(stk_next); delete(stk_idx); delete(stk_elig)
	fmt.printf("xrc: %d of %d functions return with the stack balanced (ret N; %d rounds): direct calls to them keep esp\n", n, len(addrs), rounds)
}

// after a call (x_s_: esp before it): esp and the stale return-address slot (see x_call). target: a direct call's
// (0: an indirect call -- its callee's N not known, esp read back)
x_post_call :: proc(xl: ^Xlat, target: u32, ra: u32) -> string {
	if nn, ok := x_stk_n[target]; ok && x_stk_on && target != 0 && target in x_direct_ok {
		chk := x_stk_verify ? fmt.tprintf(" if ((u32)c->esp != (u32)(x_s_ + 0x%xu)) xrc_stack_bad(c, 0x%xu, (u32)c->esp, (u32)(x_s_ + 0x%xu));", u32(nn), target, u32(nn)) : ""
		// (no rewrite of the return-address slot: the callee's ret popped that very slot, and it held ra)
		return fmt.tprintf("   %s r_esp = x_s_ + 0x%xu;", chk, u32(nn))
	}
	return fmt.tprintf("    if (r_esp >= x_s_) MX32(x_s_ - 4) = 0x%xu;", ra)
}

// The stack audit (every function, after the rounds above; XRC_STACK_AUDIT=FILE writes the findings there): code the
// function map calls a function that is not one. A function's esp is known to be off its entry value (data of its own
// still on the stack) where it
//   - jumps to another function's start (a jmp, or a jcc's taken edge): that target uses this frame -- a shared
//     epilogue or a later part of this function, translated as a tail call into a function of its own, which then
//     returns with the wrong return address (00c3ff18: 00c3fd40's tail, its pops and ret, "returned to ..., not to
//     its caller", 2026-11-22);
//   - returns (ret) with esp above the entry: it popped what it never pushed -- itself the frame-sharing tail of
//     another function, listed as one.
// Every finding is a map entry to fix (the size of the function it belongs to, the false entry removed). Its offsets
// are only those the data flow knows (a call with an unknown N, an alignment, alloca: unknown from there on), so
// "none found" is not a proof, but every finding is real.
@(private = "file")
stk_audit_fn :: proc(fi: int, out: ^[dynamic]string) {
	f := &stk_fns[fi]
	if !f.ok || len(f.ops) == 0 {return}
	n := len(f.ops)
	esp := make([]i32, n)
	ebp := make([]i32, n)
	seen := make([]bool, n)
	work := make([dynamic]i32, 0, 64)
	said := make(map[u32]bool)
	defer {delete(esp); delete(ebp); delete(seen); delete(work); delete(said)}
	esp[0], ebp[0], seen[0] = 0, X_STK_UNK, true
	append(&work, 0)
	add :: proc(v, d: i32) -> i32 {return v == X_STK_UNK ? X_STK_UNK : v + d}
	a := stk_addrs[fi]
	for len(work) > 0 {
		k := pop(&work)
		e, b := esp[k], ebp[k]
		op := f.ops[k]
		at := f.addr[k]
		tgt_fn := op.tgt in stk_idx
		if (op.tail || op.k == .Tail) && tgt_fn && e != X_STK_UNK && e != 0 && !(at in said) {
			said[at] = true
			append(out, fmt.aprintf("%08x: %08x jumps to %08x with esp at %+d from its entry (its own data on the stack): %08x is its code, not a function", at, a, op.tgt, e, op.tgt))
		}
		switch op.k {
		case .None:
		case .Adj: e = add(e, op.arg)
		case .Esp_Unk: e = X_STK_UNK
		case .Ebp_Unk: b = X_STK_UNK
		case .Both_Unk: e, b = X_STK_UNK, X_STK_UNK
		case .Ebp_Esp: b = add(e, op.arg)
		case .Esp_Ebp: e = add(b, op.arg)
		case .Leave: e, b = add(b, 4), X_STK_UNK
		case .Pro: e, b = X_STK_UNK, add(e, op.arg)
		case .Pop_Ebp: e, b = add(e, op.arg), X_STK_UNK
		case .Ebp_Adj: b = add(b, op.arg)
		case .Call:
			nn := X_STK_UNK
			if j, ok := stk_idx[op.tgt]; ok {nn = stk_prev[j]}
			e = nn == X_STK_UNK ? X_STK_UNK : add(e, nn)
		case .Ret:
			if e != X_STK_UNK && e > 0 && !(at in said) {
				said[at] = true
				append(out, fmt.aprintf("%08x: %08x returns with esp at %+d from its entry (it popped what it never pushed): another function's tail, not a function", at, a, e))
			}
		case .Tail, .Fail:
		}
		if op.k == .Ret || op.k == .Tail || op.k == .Fail {continue}
		for s in f.succ[f.start[k]:f.start[k + 1]] {
			if !seen[s] {
				esp[s], ebp[s], seen[s] = e, b, true
				append(&work, s)
				continue
			}
			ne, nb := esp[s], ebp[s]
			if ne != e {ne = X_STK_UNK}
			if nb != b {nb = X_STK_UNK}
			if ne != esp[s] || nb != ebp[s] {
				esp[s], ebp[s] = ne, nb
				append(&work, s)
			}
		}
	}
}

@(private = "file")
stk_audit_all :: proc() {
	found := make([dynamic]string)
	// (every function's N this time, not only the translated code's: the offsets known past more calls; x_stk_n is
	// already taken, so this changes nothing of the translation)
	for &e in stk_elig {e = true}
	for r := 0; r < 64; r += 1 {
		x_par_run(len(stk_addrs), proc(i: int) {stk_next[i] = stk_eval(i)})
		changed := 0
		for i in 0 ..< len(stk_addrs) {if stk_next[i] != stk_prev[i] {changed += 1}}
		stk_prev, stk_next = stk_next, stk_prev
		if changed == 0 {break}
	}
	for _, i in stk_addrs {
		stk_audit_fn(i, &found)
	}
	fmt.printf("xrc: stack audit: %d places where code the map calls a function uses its caller's frame%s\n", len(found), len(found) > 0 ? " (XRC_STACK_AUDIT=FILE: the list)" : "")
	if path := os.get_env("XRC_STACK_AUDIT", context.temp_allocator); path != "" {
		b: [dynamic]u8
		for s in found {append(&b, ..transmute([]u8)s); append(&b, '\n')}
		_ = os.write_entire_file(path, b[:])
	}
	for s in found {delete(s)}
	delete(found)
}

// after a tail call's call (x_s_: esp after the return address was taken off): esp, when the target is balanced
// (target: a direct jump's; nothing for a computed one)
x_post_tail :: proc(xl: ^Xlat, target: u32) -> string {
	nn, ok := x_stk_n[target]
	if !ok || !x_stk_on || target == 0 || !(target in x_direct_ok) {return ""}
	chk := x_stk_verify ? fmt.tprintf(" if ((u32)c->esp != (u32)(x_s_ + 0x%xu)) xrc_stack_bad(c, 0x%xu, (u32)c->esp, (u32)(x_s_ + 0x%xu));", u32(nn), target, u32(nn)) : ""
	return fmt.tprintf("   %s r_esp = x_s_ + 0x%xu;", chk, u32(nn))
}


// XRC_STACK_WHY=FILE: for each function without an N, why -- stk_eval again with each unknown esp's cause carried
// along (the op that made it unknown): "addr cause-kind cause-addr [callee]" lines
@(private = "file")
stk_why :: proc(fi: int) -> string {
	f := &stk_fns[fi]
	a := stk_addrs[fi]
	if !f.ok {return fmt.aprintf("%08x undecoded", a)}
	if !stk_elig[fi] {return fmt.aprintf("%08x not-translated", a)}
	n := len(f.ops)
	if n == 0 {return fmt.aprintf("%08x empty", a)}
	esp := make([]i32, n, context.temp_allocator)
	ebp := make([]i32, n, context.temp_allocator)
	ce := make([]i32, n, context.temp_allocator) // (esp's cause: an op index, -1 none, -2 a merge)
	cb := make([]i32, n, context.temp_allocator)
	seen := make([]bool, n, context.temp_allocator)
	work := make([dynamic]i32, 0, 64, context.temp_allocator)
	esp[0], ebp[0], seen[0], ce[0], cb[0] = 0, X_STK_UNK, true, -1, -3
	append(&work, 0)
	res := X_STK_UNK
	have := false
	callee :: proc(t: u32) -> i32 {
		if j, ok := stk_idx[t]; ok {return stk_prev[j]}
		return X_STK_UNK
	}
	add :: proc(v, d: i32) -> i32 {return v == X_STK_UNK ? X_STK_UNK : v + d}
	cause :: proc(f: ^X_Stk_Fn, c: i32, what: string, a: u32) -> string {
		if c == -2 {return fmt.aprintf("%08x merge-of-different-esp (%s)", a, what)}
		if c == -3 {return fmt.aprintf("%08x ebp-from-entry (%s)", a, what)}
		if c < 0 {return fmt.aprintf("%08x %s", a, what)}
		op := f.ops[c]
		if op.k == .Call || op.k == .Tail {return fmt.aprintf("%08x %s via-call %08x -> %08x", a, what, f.addr[c], op.tgt)}
		return fmt.aprintf("%08x %s via-%v %08x", a, what, op.k, f.addr[c])
	}
	for len(work) > 0 {
		k := pop(&work)
		e, b, xe, xb := esp[k], ebp[k], ce[k], cb[k]
		op := f.ops[k]
		chk :: proc(e: i32, nn: i32, res: ^i32, have: ^bool) -> int {
			if e == X_STK_UNK {return 1}
			if nn == X_STK_UNK {return 2}
			if e != 0 {return 3}
			if have^ && res^ != nn {return 4}
			res^, have^ = nn, true
			return 0
		}
		leave_why :: proc(f: ^X_Stk_Fn, a: u32, r: int, xe: i32, k: i32, tgt: u32) -> string {
			switch r {
			case 1: return cause(f, xe, "esp-unknown-at-exit", a)
			case 2: return fmt.aprintf("%08x tail-to-no-N %08x -> %08x", a, f.addr[k], tgt)
			case 3: return fmt.aprintf("%08x exit-with-esp-off %08x", a, f.addr[k])
			}
			return fmt.aprintf("%08x different-ret-N %08x", a, f.addr[k])
		}
		if op.tail {if r := chk(e, callee(op.tgt), &res, &have); r != 0 {return leave_why(f, a, r, xe, k, op.tgt)}}
		switch op.k {
		case .None:
		case .Adj: e = add(e, op.arg)
		case .Esp_Unk: e, xe = X_STK_UNK, k
		case .Ebp_Unk: b, xb = X_STK_UNK, k
		case .Both_Unk: e, b, xe, xb = X_STK_UNK, X_STK_UNK, k, k
		case .Ebp_Esp: b, xb = add(e, op.arg), xe
		case .Esp_Ebp: e, xe = add(b, op.arg), xb
		case .Leave: e, b, xe, xb = add(b, 4), X_STK_UNK, xb, k
		case .Pro: e, b, xe, xb = X_STK_UNK, add(e, op.arg), k, xe
		case .Pop_Ebp: e, b, xb = add(e, op.arg), X_STK_UNK, k
		case .Ebp_Adj: b = add(b, op.arg)
		case .Call:
			nn := callee(op.tgt)
			if nn == X_STK_UNK && e != X_STK_UNK {xe = k}
			e = nn == X_STK_UNK ? X_STK_UNK : add(e, nn)
		case .Ret: if r := chk(e, op.arg, &res, &have); r != 0 {return leave_why(f, a, r, xe, k, 0)}
		case .Tail: if r := chk(e, callee(op.tgt), &res, &have); r != 0 {return leave_why(f, a, r, xe, k, op.tgt)}
		case .Fail: return fmt.aprintf("%08x fail-op %08x", a, f.addr[k])
		}
		for s in f.succ[f.start[k]:f.start[k + 1]] {
			if !seen[s] {
				esp[s], ebp[s], ce[s], cb[s], seen[s] = e, b, xe, xb, true
				append(&work, s)
				continue
			}
			ne, nb := esp[s], ebp[s]
			if ne != e {ne = X_STK_UNK}
			if nb != b {nb = X_STK_UNK}
			if ne != esp[s] || nb != ebp[s] {
				if ne != esp[s] {ce[s] = esp[s] == X_STK_UNK ? ce[s] : (e == X_STK_UNK ? xe : -2)}
				if nb != ebp[s] {cb[s] = ebp[s] == X_STK_UNK ? cb[s] : (b == X_STK_UNK ? xb : -2)}
				esp[s], ebp[s] = ne, nb
				append(&work, s)
			}
		}
	}
	return have ? "" : fmt.aprintf("%08x no-exit", a)
}

@(private = "file")
stk_why_all :: proc() {
	file := os.get_env("XRC_STACK_WHY", context.temp_allocator)
	if file == "" {return}
	sb := strings.builder_make()
	for i in 0 ..< len(stk_addrs) {
		if stk_prev[i] != X_STK_UNK {continue}
		if w := stk_why(i); w != "" {strings.write_string(&sb, w); strings.write_byte(&sb, '\n')}
	}
	_ = os.write_entire_file(file, sb.buf[:])
	fmt.eprintfln("xrc: why functions have no stack N: %s", file)
}
