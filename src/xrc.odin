package xrc

// xrc.odin - the x86 -> C translator (cmd_xrc: the whole run; x_translate: one function).
//
// Each function of the 32-bit PE becomes
//     void xf_<addr>(XCtx *c) { ... }
// in the page file of its 64 KB page (<out>/page_XXXX.c). The x86 registers live in locals (r_eax .. r_edi, r_esp),
// the flags are computed only where something reads them (xrc_flags.odin), memory goes through M8 / M16 / M32 on the
// guest memory, and a call goes to another translation directly (XCALLD*) or through the dispatcher (XCALL) --
// include/xrc.h has those macros. x87 and SSE: xrc_fp.odin. A function that uses something not handled is reported
// as needing a hand-written override (--report) and gets no translation. Also here: the callee-saved register plan
// (x_abi_*), inlining (x_inline_*), devirtualization (x_devirt_*), stack slot promotion, the dispatch table and the
// page files (x_flush_page: only changed pages are rewritten).

import "base:intrinsics"
import "core:fmt"
import "core:os"
import "core:slice"
import "core:strconv"
import "core:strings"
import "core:time"
import x86 "core:rexcode/isa/x86"


GPR := [8]string{"eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"}


Xlat :: struct {
	b:      strings.Builder, // the function body
	addr:   u32,
	size:   u32,
	tables: map[u32][dynamic]u32, // switch jumps: jmp address -> targets
	btables: map[u32]bool, // the movzx that reads a switch's byte index table (rx_byte_tables), in this function
	ok:     bool,
	reason: string, // first blocking mnemonic (when !ok)
	nins:   int,
	labels: map[u32]bool,
	uses_f: bool, // references the lazy-flag locals (so we can omit them when unused)
	uses_ffl: bool, // the fused float compare's operands (cc_fa, cc_fb)
	uses_mm: bool, // MMX: x_mm[8]
	uses_x: bool, // SSE: keeps xmm0-7 in c->x (X) and round-trips them around native calls
	uses_fp: bool, // x87: double slots f0..f7, x_fsw, x_fcw
	fmax: int, // the deepest x87 stack depth reached (number of f slots to declare)
	fret: bool, // returns a float in st(0) (entry thunk XRC_THUNKF)
	fdepth: map[u32]int, // x87 stack depth before each instruction
	callfp: map[u32]X87_Eff, // x87 effect of each call / tail jmp
	dry: bool, // a scratch run (x87 flow analysis): output is discarded
	fl_kind: int, // the flag setter just emitted: 0 none, 1 sub/cmp, 2 logic (and/or/xor/test), 3 add/inc/dec
	fl_sz:   u8,
	fuse:    int, // fl_kind of the instruction right before this one, when its flags reach this one directly
	fuse_sz: u8,
	// flag liveness / fusion (xrc_flags.odin): per instruction index
	fl_on:   bool,
	fl_cls:  []Fl_Cls,
	fl_fuse: []i32, // a reader's fused setter (-1: none)
	fl_need: []u8, // what a setter stores (FL_*)
	fl_bad:  bool, // an instruction thought not to touch the flags did: translate again without the plan
	cur:     int, // the index of the instruction being emitted
	cur_addr: u32,
	direct:  [dynamic]u32, // the translations this one calls directly (XCALLD: declared before it)
	// stack slots kept in C locals (x_slot_plan): frame offset (negative, from ebp) -> promoted; the instruction after
	// which they are loaded; the ones an instruction used otherwise than as M32 (dropped, retranslated without)
	slots:     map[i32]bool,
	slot_init: u32,
	slot_cur:  i32, // the slot the instruction being emitted accesses (0: none)
	slot_bad:  map[i32]bool,
	spin_back: map[u32]bool, // backward jumps of call-free loops with an atomic (spin locks): XRC_SPIN_YIELD
	inl_back:  map[u32]bool, // backward jumps of loops whose every call is inlined: XRC_INL_PREEMPT (each call was a
	                         // point where a waiting thread got the GIL -- a loop polling through a getter needs one)
	// inlined leaf calls (x_inline_plan): the callees' flag locals (ic_*) are declared; the instruction being emitted is
	// an inlined call (its text is the callee's, self-contained: not the flag plan's business)
	uses_ic:   bool,
	uses_icfl: bool,
	uses_ix:   bool, // an inlined SSE body (XRC_IXOUT_ / XRC_IXIN_)
	inl_site:  bool,
	n_inlined: int,
	n_devirt:  int,
}

// Direct calls (--all): a call or tail jmp to a known function start is a C call of its translation instead of a
// search by address (xrc_dispatch). x_direct_ok: the function starts that may be called so (not hand-written or
// unreachable: those stay with the dispatcher, where a hand-written version wins); x_direct_used: the ones called.
// A target whose translation fails anyway gets a forwarding stub in the dispatch table (xf_X -> xrc_dispatch).
x_no_slots := false // XRC_NO_SLOTS=1: no stack slot promotion (to compare)
x_direct_ok: map[u32]bool
// "fast" overrides (overrides.tsv): a hand-written xo_<addr> in the runtime replaces the function; translated as
// usual (the reference it is checked against), but direct calls go to xo_<addr>
x_fast: map[u32]bool
x_direct_used: map[u32]bool
// the callee-saved registers each function keeps for its caller (x_abi_plan): bit 0 ebx, 1 esi, 2 edi, 3 ebp. A
// direct call reloads from XCtx only the registers its callee may change (XCALLDK's mask: eax ecx edx esp always,
// and the callee-saved ones it does not keep); the caller's own locals hold the rest across the call
x_abi_keep: map[u32]u8

// ---- callee-saved registers (x_abi_plan) ----
// The x86 ABI has a function keep ebx esi edi ebp; compiled code does (it pushes one before changing it and pops it
// back), the CRT's frame helpers (__SEH_prolog4, __EH_epilog3 ...) and __finally funclets called inline (which load
// their parent's registers) do not. A function keeps register r when it never writes r before a push of r -- writes
// by its own instructions (the decoder's operands written plus rexcode's implicit ones), by a direct callee that
// does not keep r, or by a direct tail jump's target that does not. Indirect calls and jumps (virtual methods,
// function pointers: compiled code) are trusted. Unknown targets (no function there) keep nothing. A greatest fixpoint
// over the call graph (recursion keeps what it keeps). XRC_ABI_KEEP=0: every direct call reloads everything.
X_Abi_Fn :: struct {
	own:   u8,             // the registers its own instructions keep
	calls: [dynamic][2]u32, // direct calls: target, the registers pushed (saved) before it
	tails: [dynamic]u32,    // direct tail jumps
}
x_abi_idx :: proc(r: x86.Register) -> int {
	cl := x86.reg_class(r)
	if cl != x86.REG_GPR32 && cl != x86.REG_GPR16 && cl != x86.REG_GPR8 && cl != x86.REG_GPR8H {return -1}
	hw := int(x86.reg_hw(r))
	if cl == x86.REG_GPR8H {hw &= 3}
	switch hw {case 3: return 0; case 6: return 1; case 7: return 2; case 5: return 3} // ebx esi edi ebp
	return -1
}
x_abi_scan :: proc(a: u32, size: u32) -> (f: X_Abi_Fn) {
	f.calls = make([dynamic][2]u32)
	f.tails = make([dynamic]u32)
	ins, bad := rx_function(a, size, context.temp_allocator)
	if bad != 0 || len(ins) == 0 {return} // (own 0: keeps nothing)
	body := make(map[u32]bool, context.temp_allocator)
	for &i in ins {body[i.addr] = true}
	pushed, written: u8
	f.own = 0xf
	// the frame slots each register is loaded back from (mov r, [ebp-x]): a save into one counts only with this
	reloads := make(map[[2]i32]bool, context.temp_allocator)
	for &i in ins {
		if i.inst.mnemonic != .MOV || i.inst.operand_count != 2 || i.inst.ops[0].kind != .REGISTER || i.inst.ops[1].kind != .MEMORY {continue}
		m := i.inst.ops[1].mem
		if k := x_abi_idx(i.inst.ops[0].reg); k >= 0 && m.segment == 0 && x86.mem_has_base(m) && x86.mem_base(m) == x86.EBP && !x86.mem_has_index(m) && m.disp < 0 {
			reloads[{i32(k), m.disp}] = true
		}
	}
	leaves := false // (a ret, or a jump out: a tail call, a computed jump; none: it never returns -- __report_gsfailure,
	                // exit paths -- and what it does to the registers reaches no caller)
	for &i in ins {
		mn := i.inst.mnemonic
		if mn == .RET {leaves = true}
		if mn == .JMP && i.inst.operand_count >= 1 && i.inst.ops[0].kind != .RELATIVE {leaves = true}
		w: u8
		if mn == .PUSH && i.inst.operand_count >= 1 && i.inst.ops[0].kind == .REGISTER {
			if k := x_abi_idx(i.inst.ops[0].reg); k >= 0 && written & (1 << u8(k)) == 0 {pushed |= 1 << u8(k)}
			continue
		}
		if mn == .CALL && i.inst.operand_count >= 1 && i.inst.ops[0].kind == .RELATIVE {
			t := rx_rel_target(&i, 0)
			// the compiler's frame helpers (xrc.cfg frame_prolog / frame_epilog): a prolog saves ebx esi edi and the
			// caller's ebp for this function (which then writes them freely), its epilog restores them -- the pushes,
			// and the pops, of a normal prologue
			if t in cfg_prologs {pushed |= 0xf &~ written; continue}
			if t in cfg_epilogs {continue}
			append(&f.calls, [2]u32{t, u32(pushed)})
			continue
		}
		if mn == .JMP && i.inst.operand_count >= 1 && i.inst.ops[0].kind == .RELATIVE {
			if t := rx_rel_target(&i, 0); !(t in body) {append(&f.tails, t); leaves = true}
			continue
		}
		// a save into its own frame (mov [ebp-x], r: the CRT's hand-written helpers, _VEC_memzero), before r changes, of
		// a slot it is loaded back from (__SEH_prolog4's mov [esp+0x10], ebp is not one: the caller's ebp changes)
		if mn == .MOV && i.inst.operand_count == 2 && i.inst.ops[0].kind == .MEMORY && i.inst.ops[1].kind == .REGISTER && x86.reg_class(i.inst.ops[1].reg) == x86.REG_GPR32 {
			m := i.inst.ops[0].mem
			if k := x_abi_idx(i.inst.ops[1].reg); k >= 0 && written & (1 << u8(k)) == 0 && m.segment == 0 && x86.mem_has_base(m) && !x86.mem_has_index(m) &&
			   x86.mem_base(m) == x86.EBP && m.disp < 0 && ([2]i32{i32(k), m.disp} in reloads) {pushed |= 1 << u8(k)}
			continue
		}
		// (no-ops that name a register: the hot-patch "mov edi, edi" every CRT function starts with, lea r, [r] padding)
		if i.inst.operand_count == 2 && i.inst.ops[0].kind == .REGISTER {
			if mn == .MOV && i.inst.ops[1].kind == .REGISTER && i.inst.ops[1].reg == i.inst.ops[0].reg {continue}
			if mn == .LEA && i.inst.ops[1].kind == .MEMORY {
				m := i.inst.ops[1].mem
				if x86.mem_has_base(m) && x86.mem_base(m) == i.inst.ops[0].reg && !x86.mem_has_index(m) && m.disp == 0 {continue}
			}
		}
		if i.xop == .MOVD_ST && i.modrm >> 6 == 3 {
			switch i.modrm & 7 {case 3: w |= 1; case 6: w |= 2; case 7: w |= 4; case 5: w |= 8}
		}
		er := x86.ENCODE_RUNS[u16(mn)] // (rexcode's clobber_forms, which it keeps private)
		for c in x86.CLOBBER_FORMS[er.start:][:er.count] {
			for k in 0 ..< int(i.inst.operand_count) {
				if k < 4 && k in c.written && i.inst.ops[k].kind == .REGISTER {
					if r := x_abi_idx(i.inst.ops[k].reg); r >= 0 {w |= 1 << u8(r)}
				}
			}
			if .RBX in c.implicit_wr {w |= 1}
			if .RSI in c.implicit_wr {w |= 2}
			if .RDI in c.implicit_wr {w |= 4}
			if .RBP in c.implicit_wr {w |= 8}
		}
		// (xchg / xadd write both operands whatever the table lists)
		if (mn == .XCHG || mn == .XADD) && i.inst.operand_count == 2 {
			for k in 0 ..< 2 {if i.inst.ops[k].kind == .REGISTER {if r := x_abi_idx(i.inst.ops[k].reg); r >= 0 {w |= 1 << u8(r)}}}
		}
		f.own &~= w &~ pushed
		written |= w
	}
	if !leaves {f.own = 0xf; clear(&f.calls)}
	return
}
x_abi_plan :: proc(addrs: []u32, funcs: map[u32]u32) {
	x_abi_keep = make(map[u32]u8)
	if os.get_env("XRC_ABI_KEEP", context.temp_allocator) == "0" {return}
	fns := make(map[u32]X_Abi_Fn)
	for a in addrs {
		fns[a] = x_abi_scan(a, funcs[a])
		free_all(context.temp_allocator)
	}
	for a, f in fns {x_abi_keep[a] = f.own}
	// the frame helpers (xrc.cfg), whatever the scan says: a prolog changes its caller's ebp (keeps ebx esi edi: it
	// pushes them unchanged), an epilog restores all four (each changes)
	for h in cfg_prologs {if h in x_abi_keep {x_abi_keep[h] = 7}}
	for h in cfg_epilogs {if h in x_abi_keep {x_abi_keep[h] = 0}}
	for changed := true; changed; {
		changed = false
		for a, f in fns {
			k := x_abi_keep[a]
			for c in f.calls {k &= x_abi_keep[c[0]] | u8(c[1])} // (a target not in the map keeps nothing)
			for t in f.tails {k &= x_abi_keep[t]}
			if k != x_abi_keep[a] {x_abi_keep[a] = k; changed = true}
		}
	}
	// XRC_ABI_WHY=addr[,addr...]: what each of them keeps and why (its own instructions, its calls' and tail jumps')
	if why := os.get_env("XRC_ABI_WHY", context.temp_allocator); why != "" {
		for ws in strings.split(why, ",", context.temp_allocator) {
			wa, okw := hex_u32(ws)
			f, okf := fns[wa]
			if !okw || !okf {fmt.printf("xrc: abi %s: no such function\n", ws); continue}
			fmt.printf("xrc: abi %08x keeps %x (own instructions %x)\n", wa, x_abi_keep[wa], f.own)
			for c in f.calls {if k, okk := x_abi_keep[c[0]]; !okk || (k | u8(c[1])) != 0xf {fmt.printf("    call %08x keeps %x (pushed before it %x)%s\n", c[0], k, c[1], okk ? "" : " -- not a function")}}
			for t in f.tails {if k, okk := x_abi_keep[t]; !okk || k != 0xf {fmt.printf("    tail jump %08x keeps %x%s\n", t, k, okk ? "" : " -- not a function")}}
		}
	}
	n := 0
	for _, k in x_abi_keep {if k == 0xf {n += 1}}
	fmt.printf("xrc: %d of %d functions keep ebx esi edi ebp (direct calls to them reload only eax ecx edx esp)\n", n, len(x_abi_keep))
}

// the 32-bit home local of a GPR of any width

// (a segment / control / debug / MMX register here -- garbage decoded as code, or an instruction the translator does
// not model -- sets x_bad_reg; x_translate then fails the function with "register")
x_bad_reg: bool

x_home :: proc(r: x86.Register) -> string {
	hw := x86.reg_hw(r)
	cl := x86.reg_class(r)
	// ah/ch/dh/bh come as class GPR8H with their raw encoding 4..7
	if cl == x86.REG_GPR8H {hw &= 3}
	if (cl != x86.REG_GPR32 && cl != x86.REG_GPR16 && cl != x86.REG_GPR8 && cl != x86.REG_GPR8H) || int(hw) >= len(GPR) {
		x_bad_reg = true
		return "r_eax"
	}
	return GPR_HOME[hw]
}
GPR_HOME := [8]string{"r_eax", "r_ecx", "r_edx", "r_ebx", "r_esp", "r_ebp", "r_esi", "r_edi"}

// "0x<hex>u" (a C unsigned literal), in temp
x_hexu :: proc(v: u32) -> string {
	buf: [24]u8
	buf[0], buf[1] = '0', 'x'
	h := strconv.write_bits(buf[2:], u64(v), 16, false, 64, strconv.digits, {})
	n := 2 + len(h)
	buf[n] = 'u'
	return strings.clone(string(buf[:n + 1]), context.temp_allocator)
}

// a C expression (u32) for the value of a register operand at its width

x_reg_read :: proc(r: x86.Register) -> string {
	// segment registers read as the fixed selectors of a 32-bit process: ES CS SS DS FS GS -- Windows (WOW64)'s for a
	// PE, Linux's for the rest (gs the TLS entry glibc sets up)
	if x86.reg_class(r) == x86.REG_SEG {
		sel := exe.format == .PE ? [6]u32{0x2b, 0x23, 0x2b, 0x2b, 0x53, 0x2b} : [6]u32{0x7b, 0x73, 0x7b, 0x7b, 0, 0x33}
		if hw := int(x86.reg_hw(r)); hw < 6 {return x_hexu(u32(sel[hw]))}
	}
	h := x_home(r)
	switch x86.reg_class(r) {
	case x86.REG_GPR32: return h
	case x86.REG_GPR16: return fmt.tprintf("(%s & 0xffffu)", h)
	case x86.REG_GPR8:  return fmt.tprintf("(%s & 0xffu)", h)
	case x86.REG_GPR8H: return fmt.tprintf("((%s >> 8) & 0xffu)", h)
	}
	return "0/*reg?*/"
}

// a C statement assigning `v` into a register operand

x_reg_write :: proc(r: x86.Register, v: string) -> string {
	// a segment register loaded (not in a PE): the runtime sets the segment's base from the selector (gs: Linux TLS)
	if x86.reg_class(r) == x86.REG_SEG && exe.format != .PE && int(x86.reg_hw(r)) < 6 {
		return fmt.tprintf("xrc_set_seg(c, %d, (%s) & 0xffffu);", int(x86.reg_hw(r)), v)
	}
	h := x_home(r)
	switch x86.reg_class(r) {
	case x86.REG_GPR32: return fmt.tprintf("%s = (%s);", h, v)
	case x86.REG_GPR16: return fmt.tprintf("SET16(%s, %s);", h, v)
	case x86.REG_GPR8:  return fmt.tprintf("SET8L(%s, %s);", h, v)
	case x86.REG_GPR8H: return fmt.tprintf("SET8H(%s, %s);", h, v)
	}
	return "/*reg?*/;"
}

// the address expression of a memory operand (base + index*scale + disp); NONE for an unsupported segment

x_addr :: proc(m: x86.Memory) -> (string, bool) {
	seg := m.segment
	// es / cs / ss / ds are flat. fs / gs: an address from the segment's base, which the runtime keeps per thread
	// (XRC_SEG_FS / XRC_SEG_GS): a Windows thread's TEB is fs:, Linux's TLS is gs:
	if seg == 5 || seg == 6 {
		m2 := m
		m2.segment = 0
		inner, ok := x_addr(m2)
		if !ok {return "", false}
		return fmt.tprintf("%s + (%s)", seg == 5 ? "XRC_SEG_FS" : "XRC_SEG_GS", inner), true
	}
	parts := make([dynamic]string, context.temp_allocator)
	if x86.mem_has_base(m) {append(&parts, x_home(x86.mem_base(m)))}
	if x86.mem_has_index(m) {
		sc := x86.mem_scale(m)
		if sc == 1 {append(&parts, x_home(x86.mem_index(m)))} else {append(&parts, fmt.tprintf("%s * %d", x_home(x86.mem_index(m)), sc))}
	}
	d: i32 = m.disp
	if d != 0 || len(parts) == 0 {
		if d < 0 && len(parts) > 0 {
			append(&parts, fmt.tprintf("- 0x%xu", u32(-d)))
			// rewrite as subtraction: join below handles '+', so patch
		} else {
			append(&parts, x_hexu(u32(u32(d))))
		}
	}
	// join with ' + ', turning the '- 0x..' marker into a real minus
	sb := strings.builder_make(context.temp_allocator)
	for p, i in parts {
		if i > 0 {
			if strings.has_prefix(p, "- ") {
				strings.write_string(&sb, " ")
			} else {
				strings.write_string(&sb, " + ")
			}
		}
		strings.write_string(&sb, p)
	}
	return strings.to_string(sb), true
}

// Mn size helpers

msz :: proc(sz: u8) -> string {
	switch sz {
	case 1: return "M8"
	case 2: return "M16"
	case 8: return "M64"
	}
	return "M32"
}

// read any operand as a u32 value expression; ok=false if a memory operand's address is unsupported

x_val :: proc(xl: ^Xlat, op: x86.Operand, ea: string, have_ea: bool) -> (string, bool) {
	#partial switch op.kind {
	case .REGISTER: return x_reg_read(op.reg), true
	case .IMMEDIATE:
		// x86 sign-extends a short immediate to the operand size (83 /n ib, 6B ib): give the 32-bit sign-extended
		// value; every consumer masks to its operand size (SET8L/SET16, M8/M16 stores, xcc by cc_sz)
		return x_hexu(u32(u32(i32(op.immediate)))), true
	case .MEMORY:
		if !have_ea {return "", false}
		return fmt.tprintf("%s(%s)", msz(op.size), ea), true
	}
	return "", false
}

// store a u32 value expression into any operand

x_store :: proc(op: x86.Operand, v: string, ea: string) -> string {
	#partial switch op.kind {
	case .REGISTER: return x_reg_write(op.reg, v)
	case .MEMORY:   return fmt.tprintf("%s(%s) = (%s);", msz(op.size), ea, v)
	}
	return "/*store?*/;"
}

// the tttn condition code 0..15 of a jcc/setcc/cmovcc mnemonic, or -1

x_cc :: proc(m: x86.Mnemonic) -> int {
	#partial switch m {
	case .JO, .SETO, .CMOVO: return 0
	case .JNO, .SETNO, .CMOVNO: return 1
	case .JB, .JC, .JNAE, .SETB, .SETC, .SETNAE, .CMOVB, .CMOVC, .CMOVNAE: return 2
	case .JAE, .JNB, .JNC, .SETAE, .SETNB, .SETNC, .CMOVAE, .CMOVNB, .CMOVNC: return 3
	case .JE, .JZ, .SETE, .SETZ, .CMOVE, .CMOVZ: return 4
	case .JNE, .JNZ, .SETNE, .SETNZ, .CMOVNE, .CMOVNZ: return 5
	case .JBE, .JNA, .SETBE, .SETNA, .CMOVBE, .CMOVNA: return 6
	case .JA, .JNBE, .SETA, .SETNBE, .CMOVA, .CMOVNBE: return 7
	case .JS, .SETS, .CMOVS: return 8
	case .JNS, .SETNS, .CMOVNS: return 9
	case .JP, .JPE, .SETP, .SETPE, .CMOVP, .CMOVPE: return 10
	case .JNP, .JPO, .SETNP, .SETPO, .CMOVNP, .CMOVPO: return 11
	case .JL, .JNGE, .SETL, .SETNGE, .CMOVL, .CMOVNGE: return 12
	case .JGE, .JNL, .SETGE, .SETNL, .CMOVGE, .CMOVNL: return 13
	case .JLE, .JNG, .SETLE, .SETNG, .CMOVLE, .CMOVNG: return 14
	case .JG, .JNLE, .SETG, .SETNLE, .CMOVG, .CMOVNLE: return 15
	}
	return -1
}


x_cond_expr :: proc(xl: ^Xlat, cc: int) -> string {
	xl.uses_f = true
	if xl.fl_on {
		old := fmt.tprintf("xcond(%d, cf, xcc(cc_op, cc_sz, cc_a, cc_b, cc_r))", cc)
		e := old
		if xl.cur < len(xl.fl_fuse) && xl.fl_fuse[xl.cur] >= 0 {
			d := xl.fl_cls[xl.fl_fuse[xl.cur]]
			if f, ok := x_fused_cond(d.kind, d.sz, cc); ok {e = f}
		}
		if x_fl_verify() {return fmt.tprintf("xrc_flchk(0x%xu, %s, %s)", xl.cur_addr, x_fl_to_plan(e), old)}
		return e
	}
	if e, ok := x_fused_cond(xl.fuse, xl.fuse_sz, cc); ok {return e}
	return fmt.tprintf("xcond(%d, cf, xcc(cc_op, cc_sz, cc_a, cc_b, cc_r))", cc)
}

// a condition straight from the operands of the flag setter right before it (cmp a, b / test / and ... / add):
// what the lazy flags would give, without evaluating them
x_fused_cond :: proc(kind: int, sz: u8, cc: int) -> (string, bool) {
	u, sg := "u32", "s32"
	switch sz {
	case 1: u, sg = "u8", "s8"
	case 2: u, sg = "u16", "s16"
	}
	ua := fmt.tprintf("(%s)cc_a", u)
	ub := fmt.tprintf("(%s)cc_b", u)
	sa := fmt.tprintf("(%s)cc_a", sg)
	sb := fmt.tprintf("(%s)cc_b", sg)
	ur := fmt.tprintf("(%s)cc_r", u)
	sr := fmt.tprintf("(%s)cc_r", sg)
	switch kind {
	case 1: // sub / cmp
		switch cc {
		case 2: return fmt.tprintf("(%s < %s)", ua, ub), true
		case 3: return fmt.tprintf("(%s >= %s)", ua, ub), true
		case 4: return fmt.tprintf("(%s == %s)", ua, ub), true
		case 5: return fmt.tprintf("(%s != %s)", ua, ub), true
		case 6: return fmt.tprintf("(%s <= %s)", ua, ub), true
		case 7: return fmt.tprintf("(%s > %s)", ua, ub), true
		case 8: return fmt.tprintf("(%s < 0)", sr), true
		case 9: return fmt.tprintf("(%s >= 0)", sr), true
		case 10: return fmt.tprintf("PAR8((u8)cc_r)"), true
		case 11: return fmt.tprintf("!PAR8((u8)cc_r)"), true
		case 12: return fmt.tprintf("(%s < %s)", sa, sb), true
		case 13: return fmt.tprintf("(%s >= %s)", sa, sb), true
		case 14: return fmt.tprintf("(%s <= %s)", sa, sb), true
		case 15: return fmt.tprintf("(%s > %s)", sa, sb), true
		}
	case 2: // logic: CF = OF = 0
		switch cc {
		case 0: return "0", true
		case 1: return "1", true
		case 10: return "PAR8((u8)cc_r)", true
		case 11: return "!PAR8((u8)cc_r)", true
		case 2: return "0", true
		case 3: return "1", true
		case 4, 6: return fmt.tprintf("(%s == 0)", ur), true
		case 5, 7: return fmt.tprintf("(%s != 0)", ur), true
		case 8, 12: return fmt.tprintf("(%s < 0)", sr), true
		case 9, 13: return fmt.tprintf("(%s >= 0)", sr), true
		case 14: return fmt.tprintf("(%s <= 0)", sr), true
		case 15: return fmt.tprintf("(%s > 0)", sr), true
		}
	case 3, 5: // add: the result's conditions and CF (r < a); 5 (inc / dec / shifts: CF is not theirs): the result's only
		switch cc {
		case 2: if kind == 3 {return fmt.tprintf("(%s < %s)", ur, ua), true}
		case 3: if kind == 3 {return fmt.tprintf("(%s >= %s)", ur, ua), true}
		case 4: return fmt.tprintf("(%s == 0)", ur), true
		case 5: return fmt.tprintf("(%s != 0)", ur), true
		case 8: return fmt.tprintf("(%s < 0)", sr), true
		case 9: return fmt.tprintf("(%s >= 0)", sr), true
		case 10: return "PAR8((u8)cc_r)", true
		case 11: return "!PAR8((u8)cc_r)", true
		}
	case 4: // float compare (XFCOMI): ZF PF CF = 111 unordered, 100 equal, 001 below, 000 above; OF = SF = 0. NaN is
		// tested explicitly (xfc_un): the pages are built with finite math
		un := "xfc_un(cc_fa, cc_fb)"
		switch cc {
		case 0, 8, 12: return "0", true
		case 1, 9, 13: return "1", true
		case 2: return fmt.tprintf("(%s || cc_fa < cc_fb)", un), true
		case 3: return fmt.tprintf("!(%s || cc_fa < cc_fb)", un), true
		case 4, 14: return fmt.tprintf("(%s || cc_fa == cc_fb)", un), true
		case 5, 15: return fmt.tprintf("!(%s || cc_fa == cc_fb)", un), true
		case 6: return fmt.tprintf("(%s || cc_fa <= cc_fb)", un), true
		case 7: return fmt.tprintf("!(%s || cc_fa <= cc_fb)", un), true
		case 10: return un, true
		case 11: return fmt.tprintf("!%s", un), true
		}
	}
	return "", false
}


p :: proc(xl: ^Xlat, args: ..any) {
	if xl.dry {return} // (a scratch run: x87_step wants only the depth)
	fmt.sbprint(&xl.b, ..args)
}

// pf: a line of C. Braces in the format are literal (the generated C is full of them; Odin fmt would read them as
// directives); %s %d %x %08x and %% as in fmt, written straight into the builder (fmt.sbprintf for anything else,
// which gives what fmt gave: this is only the fast path of fmt.sbprintf(esc_braces(f), ..args)).
pf :: proc(xl: ^Xlat, f: string, args: ..any) {
	if xl.dry {return}
	b := &xl.b
	ai, start := 0, 0
	i := 0
	for i < len(f) {
		if f[i] != '%' {i += 1; continue}
		strings.write_string(b, f[start:i])
		if i + 1 < len(f) && f[i + 1] == '%' {
			strings.write_byte(b, '%')
			i += 2
			start = i
			continue
		}
		j := i + 1
		for j < len(f) && ((f[j] >= '0' && f[j] <= '9') || f[j] == '-' || f[j] == '+' || f[j] == '#' || f[j] == ' ' || f[j] == '.') {j += 1}
		if j >= len(f) { // a lone % at the end
			fmt.sbprintf(b, f[i:])
			i = len(f)
			start = i
			break
		}
		verb := f[i:j + 1]
		if ai < len(args) {
			pf_arg(b, verb, args[ai])
		} else {
			fmt.sbprintf(b, verb) // (%!s(MISSING), as fmt)
		}
		ai += 1
		i = j + 1
		start = i
	}
	strings.write_string(b, f[start:])
	if ai < len(args) { // too many arguments: what fmt appends
		rest := make([dynamic]any, 0, len(args) - ai, context.temp_allocator)
		for k in ai ..< len(args) {append(&rest, args[k])}
		fmt.sbprintf(b, "", ..rest[:])
	}
	strings.write_byte(b, '\n')
}

// one %verb of pf
@(private = "file")
pf_arg :: proc(b: ^strings.Builder, verb: string, a: any) {
	u: u64
	neg, isint := false, true
	switch v in a {
	case string:
		if verb == "%s" {strings.write_string(b, v); return}
		isint = false
	case int: u, neg = u64(abs(v)), v < 0
	case i32: u, neg = u64(abs(i64(v))), v < 0
	case i64: u, neg = u64(abs(v)), v < 0
	case i16: u, neg = u64(abs(i64(v))), v < 0
	case i8: u, neg = u64(abs(i64(v))), v < 0
	case uint: u = u64(v)
	case u64: u = v
	case u32: u = u64(v)
	case u16: u = u64(v)
	case u8: u = u64(v)
	case: isint = false
	}
	if isint && !neg {
		buf: [24]u8
		switch verb {
		case "%d", "%x":
			s := strconv.write_bits(buf[:], u, verb == "%d" ? 10 : 16, false, 64, strconv.digits, {})
			strings.write_string(b, s)
			return
		case "%08x", "%04x", "%02x":
			w := int(verb[2] - '0')
			s := strconv.write_bits(buf[:], u, 16, false, 64, strconv.digits, {})
			for _ in len(s) ..< w {strings.write_byte(b, '0')}
			strings.write_string(b, s)
			return
		}
	}
	if isint && neg && verb == "%d" {
		buf: [24]u8
		strings.write_byte(b, '-')
		strings.write_string(b, strconv.write_bits(buf[:], u, 10, false, 64, strconv.digits, {}))
		return
	}
	fmt.sbprintf(b, verb, a)
}


fail :: proc(xl: ^Xlat, why: string) {
	if xl.ok {
		xl.ok = false
		xl.reason = why
	}
}

// emit the C for one instruction (into xl.b). Sets xl.ok=false with a reason on anything unsupported.

x_emit :: proc(xl: ^Xlat, ins: ^Rx_Ins) {
	m := ins.inst.mnemonic
	n := int(ins.inst.operand_count)
	ops := ins.inst.ops

	// an address temp if any operand is memory
	ea, have_ea := "", false
	mem_i := -1
	for i in 0 ..< n {
		if ops[i].kind == .MEMORY {mem_i = i; break}
	}
	if mem_i >= 0 {
		a, ok := x_addr(ops[mem_i].mem)
		if !ok {fail(xl, fmt.tprintf("%s(seg)", rx_mn(ins))); return}
		ea = "_ea"
		have_ea = true
		mm := ops[mem_i].mem
		if len(xl.slots) > 0 && mm.segment == 0 && x86.mem_has_base(mm) && x86.mem_base(mm) == x86.EBP && !x86.mem_has_index(mm) && mm.disp in xl.slots {
			xl.slot_cur = mm.disp // (a promoted slot: x_translate puts its local in place of M32(_ea))
			pf(xl, "    {")
		} else {
			pf(xl, "    { u32 _ea = (%s);", a)
		}
	}
	defer if have_ea {pf(xl, "    }")}

	sz := ops[0].size if n > 0 else 4

	// the SSE control / status register (see XRC_LDMXCSR in include/xrc.h)
	switch ins.xop {
	case .LDMXCSR: pf(xl, "    XRC_LDMXCSR(M32(_ea));"); return
	case .STMXCSR: pf(xl, "    M32(_ea) = XRC_STMXCSR();"); return
	// MMX: mm0-7 are the function's x_mm[] (MSVC's MMX intrinsic code keeps them within one function)
	case .CVTTPS2PI, .CVTPS2PI:
		xl.uses_mm = true
		lo, hi := "f32_of(M32(_ea))", "f32_of(M32(_ea + 4))"
		if !ins.xmem {xl.uses_x = true; lo, hi = fmt.tprintf("X[%d].f[0]", ins.xrm), fmt.tprintf("X[%d].f[1]", ins.xrm)}
		fnm := ins.xop == .CVTTPS2PI ? "cvt_trunc" : "cvt_round"
		pf(xl, "    x_mm[%d] = (u64)%s((double)%s) | xsh64l((u64)%s((double)%s), 32);", ins.xr, fnm, lo, fnm, hi)
		return
	case .CVTPI2PS:
		xl.uses_mm = true
		xl.uses_x = true
		src := ins.xmem ? "M64(_ea)" : fmt.tprintf("x_mm[%d]", ins.xrm)
		pf(xl, "    { u64 _q = %s; X[%d].f[0] = (float)(s32)(u32)_q; X[%d].f[1] = (float)(s32)(u32)xsh64r(_q, 32); }", src, ins.xr, ins.xr)
		return
	case .MOVQ_LD:
		xl.uses_mm = true
		pf(xl, "    x_mm[%d] = %s;", ins.xr, ins.xmem ? "M64(_ea)" : fmt.tprintf("x_mm[%d]", ins.xrm))
		return
	case .MOVQ_ST:
		xl.uses_mm = true
		pf(xl, "    %s = x_mm[%d];", ins.xmem ? "M64(_ea)" : fmt.tprintf("x_mm[%d]", ins.xrm), ins.xr)
		return
	case .MOVD_LD, .MOVD_ST:
		xl.uses_mm = true
		gpr := [8]x86.Register{x86.EAX, x86.ECX, x86.EDX, x86.EBX, x86.ESP, x86.EBP, x86.ESI, x86.EDI}
		if ins.xop == .MOVD_LD {
			src := ins.xmem ? "M32(_ea)" : x_reg_read(gpr[ins.xrm])
			pf(xl, "    x_mm[%d] = (u64)(u32)(%s);", ins.xr, src)
		} else {
			pf(xl, "    %s", x_store(ins.xmem ? ins.inst.ops[0] : x86.op_reg(gpr[ins.xrm]), fmt.tprintf("(u32)x_mm[%d]", ins.xr), ea))
		}
		return
	case .MMXOP:
		xl.uses_mm = true
		src := ins.xmem ? "M64(_ea)" : fmt.tprintf("x_mm[%d]", ins.xrm)
		lanes :: proc(op: string) -> string {
			return fmt.tprintf("(u64)(u32)((u32)_a %s (u32)_b) | xsh64l((u64)(u32)((u32)xsh64r(_a, 32) %s (u32)xsh64r(_b, 32)), 32)", op, op)
		}
		e := ""
		switch ins.opc2 {
		case 0xdb: e = "_a & _b"
		case 0xdf: e = "~_a & _b"
		case 0xeb: e = "_a | _b"
		case 0xef: e = "_a ^ _b"
		case 0xd4: e = "_a + _b"
		case 0xfb: e = "_a - _b"
		case 0xfe: e = lanes("+")
		case 0xfa: e = lanes("-")
		}
		pf(xl, "    { u64 _a = x_mm[%d], _b = %s; x_mm[%d] = %s; }", ins.xr, src, ins.xr, e)
		return
	case .EMMS: return
	case .UD:
		pf(xl, "    XRC_UD(0x%xu);", ins.addr)
		return
	case .None:
	}

	if x87_is(ins) {
		_ = x87_emit(xl, ins, xl.fdepth[ins.addr], ea, have_ea)
		return
	}

	// helper closures
	rd :: proc(xl: ^Xlat, op: x86.Operand, ea: string, have_ea: bool) -> (string, bool) {
		return x_val(xl, op, ea, have_ea)
	}

	// ALU with lazy flags: op_kind, result expr in terms of _a (dst) and _b (src)
	alu :: proc(xl: ^Xlat, ins: ^Rx_Ins, ops: []x86.Operand, ea: string, have_ea: bool, cc: string, rexpr: string, writes: bool, sets_cf: string) {
		a, oka := x_val(xl, ops[0], ea, have_ea)
		b, okb := x_val(xl, ops[1], ea, have_ea)
		if !oka || !okb {fail(xl, "operand"); return}
		sz := ops[0].size
		pf(xl, "    { u32 _a = (%s), _b = (%s), _r = (%s);", a, b, rexpr)
		if xl.fl_on && x_fl_verify() {
			pf(xl, "      cc_op = %s; cc_sz = %d; cc_a = _a; cc_b = _b; cc_r = _r;", cc, sz)
			if sets_cf != "" {pf(xl, "      %s", sets_cf)}
			pf(xl, "      %s", x_fl_shadow(xl, sets_cf != ""))
			xl.uses_f = true
			if writes {pf(xl, "      %s", x_store(ops[0], "_r", ea))}
			pf(xl, "    }")
			return
		}
		if xl.fl_on {
			nd := xl.fl_need[xl.cur]
			st := x_fl_stores(xl, fmt.tprintf("cc_op = %s; cc_sz = %d; cc_a = _a; cc_b = _b; cc_r = _r;", cc, sz))
			if st != "" {pf(xl, "      %s", st)}
			if nd & FL_C != 0 && sets_cf != "" {
				s, _ := strings.replace_all(sets_cf, "cc_sz", fmt.tprintf("%d", sz), context.temp_allocator)
				pf(xl, "      %s", s)
			}
			if !writes && nd & (FL_FULL | FL_R) == 0 {pf(xl, "      (void)_r;")}
			xl.uses_f = true
			if writes {pf(xl, "      %s", x_store(ops[0], "_r", ea))}
			pf(xl, "    }")
			return
		}
		pf(xl, "      cc_op = %s; cc_sz = %d; cc_a = _a; cc_b = _b; cc_r = _r;", cc, sz)
		switch cc {
		case "CC_SUB": if rexpr == "_a - _b" {xl.fl_kind = 1}
		case "CC_LOGIC": xl.fl_kind = 2
		case "CC_ADD": if rexpr == "_a + _b" {xl.fl_kind = 3}
		}
		xl.fl_sz = sz
		xl.uses_f = true
		if sets_cf != "" {pf(xl, "      %s", sets_cf)}
		if writes {pf(xl, "      %s", x_store(ops[0], "_r", ea))}
		pf(xl, "    }")
	}

	#partial switch m {
	case .MOV, .MOVABS:
		v, ok := x_val(xl, ops[1], ea, have_ea)
		if !ok {fail(xl, "mov"); return}
		pf(xl, "    %s", x_store(ops[0], v, ea))
	case .MOVZX:
		if bt, is_bt := rx_byte_tables[ins.addr]; is_bt && have_ea && ins.addr in xl.btables {
			// a switch's byte index table (in .text: see the jmp): its entries as cases, by address
			pf(xl, "    { u32 _v; switch (%s) {", ea)
			for b, i in exe_read(bt.at, int(bt.n)) {pf(xl, "    case 0x%xu: _v = %d; break;", bt.at + u32(i), b)}
			pf(xl, "    default: _v = M8(%s); }", ea)
			pf(xl, "    %s }", x_store(ops[0], "_v", ea))
			break
		}
		v, ok := x_val(xl, ops[1], ea, have_ea)
		if !ok {fail(xl, "movzx"); return}
		pf(xl, "    %s", x_store(ops[0], v, ea)) // x_val already masks to src width
	case .MOVSX:
		v, ok := x_val(xl, ops[1], ea, have_ea)
		if !ok {fail(xl, "movsx"); return}
		ss := ops[1].size
		cast_t := ss == 1 ? "s8" : "s16"
		pf(xl, "    %s", x_store(ops[0], fmt.tprintf("(u32)(s32)(%s)(%s)", cast_t, v), ea))
	case .LEA:
		if !have_ea {fail(xl, "lea"); return}
		pf(xl, "    %s", x_store(ops[0], ea, ea))
	case .PUSH:
		// push always moves a full stack slot (4 bytes, or 2 with a 0x66 prefix); an imm8/imm32 is sign-extended to
		// that width -- NOT the operand's encoded size (push 0x18 is 6a 18 with operand size 1 but pushes a dword)
		w := ins.p66 ? 2 : 4
		v := ""
		if ops[0].kind == .IMMEDIATE {
			v = x_hexu(u32(u32(i32(ops[0].immediate)))) // rexcode sign-extends imm8/imm32 into .immediate
		} else {
			vv, ok := x_val(xl, ops[0], ea, have_ea)
			if !ok {fail(xl, "push"); return}
			v = vv
		}
		// the value is read before esp moves (push esp pushes the old esp)
		if strings.contains(v, "r_esp") {
			pf(xl, "    {{ u32 _pv = (%s); r_esp -= %d; %s(r_esp) = _pv; }}", v, w, w == 2 ? "M16" : "M32")
		} else {
			pf(xl, "    r_esp -= %d; %s(r_esp) = (%s);", w, w == 2 ? "M16" : "M32", v)
		}
	case .POP:
		w := ins.p66 ? 2 : 4
		pf(xl, "    { u32 _v = %s(r_esp); r_esp += %d; %s }", w == 2 ? "M16" : "M32", w, x_store(ops[0], "_v", ea))
	case .ADD: alu(xl, ins, ops[:], ea, have_ea, "CC_ADD", "_a + _b", true, "cf = xcf_add(cc_sz, _a, _r);")
	case .SUB: alu(xl, ins, ops[:], ea, have_ea, "CC_SUB", "_a - _b", true, "cf = xcf_sub(cc_sz, _a, _b);")
	case .CMP: alu(xl, ins, ops[:], ea, have_ea, "CC_SUB", "_a - _b", false, "cf = xcf_sub(cc_sz, _a, _b);")
	case .ADC: alu(xl, ins, ops[:], ea, have_ea, "CC_ADD", "_a + _b + cf", true, "cf = xcf_adc(cc_sz, _a, _b, cf);")
	case .SBB: alu(xl, ins, ops[:], ea, have_ea, "CC_SUB", "_a - _b - cf", true, "cf = xcf_sbb(cc_sz, _a, _b, cf);")
	case .AND: alu(xl, ins, ops[:], ea, have_ea, "CC_LOGIC", "_a & _b", true, "cf = 0;")
	case .OR:  alu(xl, ins, ops[:], ea, have_ea, "CC_LOGIC", "_a | _b", true, "cf = 0;")
	case .XOR: alu(xl, ins, ops[:], ea, have_ea, "CC_LOGIC", "_a ^ _b", true, "cf = 0;")
	case .TEST: alu(xl, ins, ops[:], ea, have_ea, "CC_LOGIC", "_a & _b", false, "cf = 0;")
	case .INC:
		a, ok := x_val(xl, ops[0], ea, have_ea)
		if !ok {fail(xl, "inc"); return}
		pf(xl, "    { u32 _a = (%s), _r = _a + 1; %s%s }", a, x_fl_stores(xl, fmt.tprintf("cc_op = CC_INC; cc_sz = %d; cc_a = _a; cc_b = 1; cc_r = _r; ", ops[0].size)), x_store(ops[0], "_r", ea))
		xl.uses_f = true
		xl.fl_kind, xl.fl_sz = 5, ops[0].size
	case .DEC:
		a, ok := x_val(xl, ops[0], ea, have_ea)
		if !ok {fail(xl, "dec"); return}
		pf(xl, "    { u32 _a = (%s), _r = _a - 1; %s%s }", a, x_fl_stores(xl, fmt.tprintf("cc_op = CC_DEC; cc_sz = %d; cc_a = _a; cc_b = 1; cc_r = _r; ", ops[0].size)), x_store(ops[0], "_r", ea))
		xl.uses_f = true
		xl.fl_kind, xl.fl_sz = 5, ops[0].size
	case .NEG:
		a, ok := x_val(xl, ops[0], ea, have_ea)
		if !ok {fail(xl, "neg"); return}
		if xl.fl_on && xl.fl_cls[xl.cur].setter { // (as 0 - x: x_fl_class)
			sz := ops[0].size
			nd := xl.fl_need[xl.cur]
			full := fmt.tprintf("cc_op = CC_SUB; cc_sz = %d; cc_a = _a; cc_b = _b; cc_r = _r; ", sz)
			cfs := fmt.tprintf("cf = (_b & xcc_mask(%d)) != 0; ", sz)
			st := ""
			if x_fl_verify() {st = fmt.tprintf("%s%s%s", full, cfs, x_fl_shadow(xl, true))} else {st = fmt.tprintf("%s%s", x_fl_stores(xl, full), nd & FL_C != 0 ? cfs : "")}
			pf(xl, "    { u32 _a = 0, _b = (%s), _r = (u32)(0 - _b); %s%s }", a, st, x_store(ops[0], "_r", ea))
			xl.uses_f = true
			break
		}
		pf(xl, "    { u32 _a = (%s), _r = (u32)(0 - _a); cc_op = CC_NEG; cc_sz = %d; cc_a = _a; cc_b = 0; cc_r = _r; cf = (_a & xcc_mask(%d)) != 0; %s }", a, ops[0].size, ops[0].size, x_store(ops[0], "_r", ea))
		xl.uses_f = true
	case .NOT:
		a, ok := x_val(xl, ops[0], ea, have_ea)
		if !ok {fail(xl, "not"); return}
		pf(xl, "    %s", x_store(ops[0], fmt.tprintf("~(%s)", a), ea))
	case .SHL, .SAL: x_shift(xl, ops[:], ea, have_ea, "shl")
	case .SHR: x_shift(xl, ops[:], ea, have_ea, "shr")
	case .SAR: x_shift(xl, ops[:], ea, have_ea, "sar")
	case .ROL, .ROR, .RCL, .RCR:
		// CF from the bit rotated out / through; OF is not modelled (defined only for count 1, unused by MSVC code)
		a, oka := x_val(xl, ops[0], ea, have_ea)
		b, okb := x_val(xl, ops[1], ea, have_ea)
		if !oka || !okb {fail(xl, rx_mn(ins)); return}
		w := int(ops[0].size) * 8
		st := x_store(ops[0], "_r", ea)
		xl.uses_f = true
		#partial switch m {
		case .ROL: pf(xl, "    { u32 _a = (%s) & xcc_mask(%d), _n = (%s) & 31; if (_n) { u32 _k = _n %% %d, _r = _k ? ((_a << _k) | (_a >> (%d - _k))) & xcc_mask(%d) : _a; cf = _r & 1u; %s } }", a, sz, b, w, w, sz, st)
		case .ROR: pf(xl, "    { u32 _a = (%s) & xcc_mask(%d), _n = (%s) & 31; if (_n) { u32 _k = _n %% %d, _r = _k ? ((_a >> _k) | (_a << (%d - _k))) & xcc_mask(%d) : _a; cf = (_r >> %d) & 1u; %s } }", a, sz, b, w, w, sz, w - 1, st)
		case .RCL: pf(xl, "    { u32 _r = (%s) & xcc_mask(%d), _n = ((%s) & 31) %% %d; while (_n--) { u32 _c = (_r >> %d) & 1u; _r = ((_r << 1) | (cf ? 1u : 0u)) & xcc_mask(%d); cf = _c; } %s }", a, sz, b, w + 1, w - 1, sz, st)
		case .RCR: pf(xl, "    { u32 _r = (%s) & xcc_mask(%d), _n = ((%s) & 31) %% %d; while (_n--) { u32 _c = _r & 1u; _r = (_r >> 1) | ((cf ? 1u : 0u) << %d); cf = _c; } %s }", a, sz, b, w + 1, w - 1, st)
		}
	case .SHLD, .SHRD:
		if n != 3 || ops[0].size != 4 {fail(xl, fmt.tprintf("%s16", rx_mn(ins))); return}
		a, oka := x_val(xl, ops[0], ea, have_ea)
		b, okb := x_val(xl, ops[1], ea, have_ea)
		c, okc := x_val(xl, ops[2], ea, have_ea)
		if !oka || !okb || !okc {fail(xl, rx_mn(ins)); return}
		r := m == .SHLD ? "(_a << _n) | (_b >> (32 - _n))" : "(_a >> _n) | (_b << (32 - _n))"
		c_ := m == .SHLD ? "(_a >> (32 - _n)) & 1u" : "(_a >> (_n - 1)) & 1u"
		pf(xl, "    { u32 _a = (%s), _b = (%s), _n = (%s) & 31; if (_n) { u32 _r = %s; cf = %s; cc_op = CC_LOGIC; cc_sz = 4; cc_a = _a; cc_b = 0; cc_r = _r; %s } }", a, b, c, r, c_, x_store(ops[0], "_r", ea))
		xl.uses_f = true
	case .BSF, .BSR:
		// ZF = (source == 0); the destination is left alone then (as Intel / AMD parts do)
		v, ok := x_val(xl, ops[1], ea, have_ea)
		if !ok {fail(xl, rx_mn(ins)); return}
		pf(xl, "    { u32 _v = (%s); cc_op = CC_RAW; if (_v) { cc_r = 0; %s } else { cc_r = XF_ZF; } }", v, x_store(ops[0], m == .BSF ? "xbsf(_v)" : "xbsr(_v)", ea))
		xl.uses_f = true
	case .BT, .BTS, .BTR, .BTC:
		upd := ""
		#partial switch m {
		case .BTS: upd = "_v | (1u << _k)"
		case .BTR: upd = "_v & ~(1u << _k)"
		case .BTC: upd = "_v ^ (1u << _k)"
		}
		xl.uses_f = true
		w := u32(ops[0].size) * 8
		if ops[1].kind != .IMMEDIATE && ops[0].kind == .MEMORY {
			// a register bit offset reaches past the operand: the dword at ea + 4 * (offset >> 5) (signed)
			if ops[0].size != 4 {fail(xl, fmt.tprintf("%s16", rx_mn(ins))); return}
			o, ok := x_val(xl, ops[1], ea, have_ea)
			if !ok {fail(xl, rx_mn(ins)); return}
			pf(xl, "    { u32 _o = (%s), _p = %s + (u32)((s32)_o >> 5) * 4u, _k = _o & 31, _v = M32(_p); cf = (_v >> _k) & 1u;%s }", o, ea, upd == "" ? "" : fmt.tprintf(" M32(_p) = %s;", upd))
			break
		}
		a, oka := x_val(xl, ops[0], ea, have_ea)
		o, okb := x_val(xl, ops[1], ea, have_ea)
		if !oka || !okb {fail(xl, rx_mn(ins)); return}
		pf(xl, "    { u32 _v = (%s), _k = (%s) & %d; cf = (_v >> _k) & 1u;%s }", a, o, w - 1, upd == "" ? "" : fmt.tprintf(" %s", x_store(ops[0], upd, ea)))
	case .XADD:
		// (not atomic: the guest's threads are not run concurrently with each other)
		a, oka := x_val(xl, ops[0], ea, have_ea)
		b, okb := x_val(xl, ops[1], ea, have_ea)
		if !oka || !okb {fail(xl, "xadd"); return}
		pf(xl, "    { u32 _a = (%s), _b = (%s), _r = _a + _b; cc_op = CC_ADD; cc_sz = %d; cc_a = _a; cc_b = _b; cc_r = _r; cf = xcf_add(cc_sz, _a, _r); %s %s }", a, b, sz, x_store(ops[1], "_a", ea), x_store(ops[0], "_r", ea))
		xl.uses_f = true
	case .CMPXCHG:
		acc := x86.op_reg(sz == 1 ? x86.AL : sz == 2 ? x86.AX : x86.EAX)
		av, ok0 := x_val(xl, acc, ea, have_ea)
		d, oka := x_val(xl, ops[0], ea, have_ea)
		v, okb := x_val(xl, ops[1], ea, have_ea)
		if !ok0 || !oka || !okb {fail(xl, "cmpxchg"); return}
		pf(xl, "    { u32 _a = (%s), _b = (%s), _r = _a - _b; cc_op = CC_SUB; cc_sz = %d; cc_a = _a; cc_b = _b; cc_r = _r; cf = xcf_sub(cc_sz, _a, _b); if ((_r & xcc_mask(%d)) == 0) { %s } else { %s } }", av, d, sz, sz, x_store(ops[0], v, ea), x_store(acc, "_b", ea))
		xl.uses_f = true
	case .BSWAP:
		a, _ := x_val(xl, ops[0], ea, have_ea)
		pf(xl, "    %s", x_store(ops[0], fmt.tprintf("bswap32(%s)", a), ea))
	case .IMUL: x_imul(xl, ins, ops[:], ea, have_ea)
	case .MUL:
		a, ok := x_val(xl, ops[0], ea, have_ea)
		if !ok {fail(xl, "mul"); return}
		switch ops[0].size {
		case 4: pf(xl, "    { u64 _p = XMULU(r_eax, %s); r_eax = (u32)_p; r_edx = (u32)(_p >> 32); cf = r_edx != 0; cc_r = r_eax; }", a)
		case 2: pf(xl, "    { u32 _p = (u32)(u16)r_eax * (u32)(u16)(%s); SET16(r_eax, _p); SET16(r_edx, _p >> 16); cf = (_p >> 16) != 0; cc_r = _p; }", a)
		case 1: pf(xl, "    { u32 _p = (u32)(u8)r_eax * (u32)(u8)(%s); SET16(r_eax, _p); cf = (_p >> 8) != 0; cc_r = _p; }", a)
		case: fail(xl, "mul"); return
		}
		pf(xl, "    cc_op = CC_MUL; cc_sz = %d; cc_a = cf; cc_b = 0;", ops[0].size)
		xl.uses_f = true
	case .DIV:
		a, ok := x_val(xl, ops[0], ea, have_ea)
		if !ok || ops[0].size != 4 {fail(xl, "div"); return}
		pf(xl, "    XRC_DIV(%s);", a)
	case .IDIV:
		a, ok := x_val(xl, ops[0], ea, have_ea)
		if !ok || ops[0].size != 4 {fail(xl, "idiv"); return}
		pf(xl, "    XRC_IDIV(%s);", a)
	case .CDQ: pf(xl, "    r_edx = (u32)((s32)r_eax >> 31);")
	case .CWDE: pf(xl, "    r_eax = (u32)(s32)(s16)(u16)r_eax;")
	case .CBW: pf(xl, "    SET16(r_eax, (u32)(s16)(s8)(u8)r_eax);")
	case .XCHG:
		// (66 90, xchg ax, ax: a 2-byte nop in padding)
		if b := exe_read(ins.addr, 2); ins.len == 2 && len(b) == 2 && b[0] == 0x66 && b[1] == 0x90 {pf(xl, "    /* nop */"); break}
		x_xchg(xl, ops[:], ea, have_ea)
	case .LEAVE: pf(xl, "    r_esp = r_ebp; r_ebp = M32(r_esp); r_esp += 4;")
	case .NOP, .FNOP: // nothing
	case .CALL: x_call(xl, ins, ops[:], ea, have_ea)
	case .JMP: x_jmp(xl, ins, ops[:], ea, have_ea)
	case .RET:
		if xl.fret {pf(xl, "    XRC_FRET(f0);")}
		if n > 0 && ops[0].kind == .IMMEDIATE { // ret N (stdcall): pop N extra bytes of arguments
			pf(xl, "    XRC_RETURN(%d);", u32(ops[0].immediate))
		} else {
			pf(xl, "    XRC_RETURN(0);")
		}
	case .SETO, .SETNO, .SETB, .SETC, .SETNAE, .SETAE, .SETNB, .SETNC, .SETE, .SETZ, .SETNE, .SETNZ,
	     .SETBE, .SETNA, .SETA, .SETNBE, .SETS, .SETNS, .SETP, .SETPE, .SETNP, .SETPO, .SETL, .SETNGE,
	     .SETGE, .SETNL, .SETLE, .SETNG, .SETG, .SETNLE:
		cc := x_cc(m)
		pf(xl, "    %s", x_store(ops[0], fmt.tprintf("(u32)(%s)", x_cond_expr(xl, cc)), ea))
	case .CMOVO, .CMOVNO, .CMOVB, .CMOVC, .CMOVNAE, .CMOVAE, .CMOVNB, .CMOVNC, .CMOVE, .CMOVZ, .CMOVNE,
	     .CMOVNZ, .CMOVBE, .CMOVNA, .CMOVA, .CMOVNBE, .CMOVS, .CMOVNS, .CMOVP, .CMOVPE, .CMOVNP, .CMOVPO,
	     .CMOVL, .CMOVNGE, .CMOVGE, .CMOVNL, .CMOVLE, .CMOVNG, .CMOVG, .CMOVNLE:
		cc := x_cc(m)
		v, ok := x_val(xl, ops[1], ea, have_ea)
		if !ok {fail(xl, "cmov"); return}
		pf(xl, "    if (%s) { %s }", x_cond_expr(xl, cc), x_store(ops[0], v, ea))
	case .MOVS, .MOVSB, .MOVSW, .MOVSD:
		// SSE2 movsd (F2 0F 10/11) shares the mnemonic with the string move (A5)
		if ins.opc == 0x0f {
			if !sse_emit(xl, ins, ops[:n], ea, have_ea) {fail(xl, rx_mn(ins))}
			return
		}
		x_rep(xl, ins, "movs", sz)
	case .STOS, .STOSB, .STOSW, .STOSD: x_rep(xl, ins, "stos", sz)
	case .LODS, .LODSB, .LODSW, .LODSD: x_rep(xl, ins, "lods", sz)
	case .SCAS, .SCASB, .SCASW, .SCASD: x_rep(xl, ins, "scas", sz)
	case .CMPS, .CMPSB, .CMPSW, .CMPSD:
		// SSE2 cmpsd (F2 0F C2) shares the mnemonic with the string compare (A7)
		if ins.opc == 0x0f {
			if !sse_emit(xl, ins, ops[:n], ea, have_ea) {fail(xl, rx_mn(ins))}
			return
		}
		x_rep(xl, ins, "cmps", sz)
	case .CLD: pf(xl, "    xrc_eflags_sys &= ~XRC_EFL_DF;")
	case .STD: pf(xl, "    xrc_eflags_sys |= XRC_EFL_DF;")
	case .INT:
		// a software interrupt (int 0x80: a Linux system call; int 0x2e: an old Windows one): the runtime's, with the
		// registers published and reloaded (eip: the instruction after it)
		if ins.inst.operand_count < 1 || ins.inst.ops[0].kind != .IMMEDIATE {fail(xl, "int*"); return}
		pf(xl, "    { XRC_SPILL c->eip = 0x%xu; xrc_int(c, 0x%xu); XRC_FILL }", ins.addr + ins.len, u32(ins.inst.ops[0].immediate) & 0xff)
	case .INT3: pf(xl, "    /* int3 */")
	case .CLC: pf(xl, "    cf = 0;"); xl.uses_f = true
	case .CLI, .STI, .HLT: pf(xl, "    XRC_UD(0x%xu); /* privileged: #GP in user mode */", ins.addr)
	case .STC: pf(xl, "    cf = 1;"); xl.uses_f = true
	case .CMC: pf(xl, "    cf = !cf;"); xl.uses_f = true
	case .LAHF:
		xl.uses_f = true
		pf(xl, "    { u32 _f = xcc(cc_op, cc_sz, cc_a, cc_b, cc_r); SET8H(r_eax, ((_f & XF_SF) ? 0x80u : 0u) | ((_f & XF_ZF) ? 0x40u : 0u) | ((_f & XF_PF) ? 4u : 0u) | 2u | (cf ? 1u : 0u)); }")
	case .PUSHF, .PUSHFD:
		// EFLAGS from the lazy flags + the guest's system bits (xrc_eflags_sys: IF, DF, ID, ...)
		if ins.p66 {fail(xl, "pushf16"); return}
		xl.uses_f = true
		pf(xl, "    r_esp -= 4; M32(r_esp) = xrc_eflags_pack(xcc(cc_op, cc_sz, cc_a, cc_b, cc_r), cf);")
	case .POPF, .POPFD:
		if ins.p66 {fail(xl, "popf16"); return}
		xl.uses_f = true
		pf(xl, "    { u32 _v = M32(r_esp); r_esp += 4; cc_op = CC_RAW; cc_r = xrc_eflags_unpack(_v); cf = _v & 1u; }")
	case .CPUID:
		pf(xl, "    xrc_cpuid(&r_eax, &r_ebx, &r_ecx, &r_edx);")
	case .SAHF:
		xl.uses_f = true
		pf(xl, "    { u32 _h = (r_eax >> 8) & 0xffu, _o = xcc(cc_op, cc_sz, cc_a, cc_b, cc_r) & XF_OF; cc_op = CC_RAW; cc_r = _o | ((_h & 0x80u) ? XF_SF : 0u) | ((_h & 0x40u) ? XF_ZF : 0u) | ((_h & 4u) ? XF_PF : 0u); cf = _h & 1u; }")
	case:
		if rx_is_jcc(m) {
			x_jcc(xl, ins, ops[:])
			return
		}
		if sse_emit(xl, ins, ops[:n], ea, have_ea) {return}
		fail(xl, rx_mn(ins))
	}
}


x_shift :: proc(xl: ^Xlat, ops: []x86.Operand, ea: string, have_ea: bool, kind: string) {
	a, oka := x_val(xl, ops[0], ea, have_ea)
	b, okb := x_val(xl, ops[1], ea, have_ea)
	if !oka || !okb {fail(xl, kind); return}
	sz := ops[0].size
	// count masked to 5 bits (32-bit operand) as on x86
	expr := ""
	switch kind {
	case "shl": expr = fmt.tprintf("(_a << _n)")
	case "shr": expr = fmt.tprintf("((_a & xcc_mask(%d)) >> _n)", sz)
	case "sar": expr = fmt.tprintf("(u32)(((s32)(_a << %d)) >> (_n + %d))", (4 - sz) * 8, (4 - sz) * 8)
	}
	if xl.fl_on && xl.cur < len(xl.fl_cls) && xl.fl_cls[xl.cur].setter { // (a constant count: x_fl_class)
		nd := xl.fl_need[xl.cur]
		full := fmt.tprintf("cc_op = CC_LOGIC; cc_sz = %d; cc_a = _a; cc_b = 0; cc_r = _r; ", sz)
		st := ""
		if x_fl_verify() {st = fmt.tprintf("%s%s %s", full, x_shift_cf(kind, sz), x_fl_shadow(xl, true))} else {st = fmt.tprintf("%s%s", x_fl_stores(xl, full), nd & FL_C != 0 ? x_shift_cf(kind, sz) : "")}
		pf(xl, "    { u32 _a = (%s), _n = (%s) & 31; if (_n) { u32 _r = %s; %s %s } }", a, b, expr, st, x_store(ops[0], "_r", ea))
		xl.uses_f = true
		return
	}
	pf(xl, "    { u32 _a = (%s), _n = (%s) & 31; if (_n) { u32 _r = %s; cc_op = CC_LOGIC; cc_sz = %d; cc_a = _a; cc_b = 0; cc_r = _r; %s %s } }",
		a, b, expr, sz, x_shift_cf(kind, sz), x_store(ops[0], "_r", ea))
	xl.uses_f = true
}


x_shift_cf :: proc(kind: string, sz: u8) -> string {
	switch kind {
	case "shl": return fmt.tprintf("cf = (_a >> (%d - _n)) & 1;", sz * 8)
	case "shr", "sar": return "cf = (_a >> (_n - 1)) & 1;"
	}
	return ""
}


x_imul :: proc(xl: ^Xlat, ins: ^Rx_Ins, ops: []x86.Operand, ea: string, have_ea: bool) {
	n := int(ins.inst.operand_count)
	if n == 1 {
		a, ok := x_val(xl, ops[0], ea, have_ea)
		if !ok {fail(xl, "imul1"); return}
		switch ops[0].size {
		case 4: pf(xl, "    { s64 _p = XMULS(r_eax, %s); r_eax = (u32)_p; r_edx = (u32)((u64)_p >> 32); cf = (_p != (s64)(s32)_p); cc_r = r_eax; }", a)
		case 2: pf(xl, "    { s32 _p = (s32)(s16)r_eax * (s32)(s16)(%s); SET16(r_eax, _p); SET16(r_edx, (u32)_p >> 16); cf = (_p != (s32)(s16)_p); cc_r = (u32)_p; }", a)
		case 1: pf(xl, "    { s32 _p = (s32)(s8)r_eax * (s32)(s8)(%s); SET16(r_eax, _p); cf = (_p != (s32)(s8)_p); cc_r = (u32)_p; }", a)
		case: fail(xl, "imul1"); return
		}
		pf(xl, "    cc_op = CC_MUL; cc_sz = %d; cc_a = cf; cc_b = 0;", ops[0].size)
		xl.uses_f = true
		return
	}
	// 2-op (dst *= src) or 3-op (dst = src * imm)
	src1 := ops[0] if n == 2 else ops[1]
	src2 := ops[1] if n == 2 else ops[2]
	a, oka := x_val(xl, src1, ea, have_ea)
	b, okb := x_val(xl, src2, ea, have_ea)
	if !oka || !okb {fail(xl, "imul"); return}
	if ops[0].size == 2 {
		pf(xl, "    { s32 _p = (s32)(s16)(%s) * (s32)(s16)(%s); cf = (_p != (s32)(s16)_p); cc_r = (u32)_p; %s }", a, b, x_store(ops[0], "(u32)_p", ea))
	} else {
		pf(xl, "    { s64 _p = XMULS(%s, %s); cf = (_p != (s64)(s32)_p); cc_r = (u32)_p; %s }", a, b, x_store(ops[0], "(u32)_p", ea))
	}
	pf(xl, "    cc_op = CC_MUL; cc_sz = %d; cc_a = cf; cc_b = 0;", ops[0].size)
	xl.uses_f = true
}


x_xchg :: proc(xl: ^Xlat, ops: []x86.Operand, ea: string, have_ea: bool) {
	a, oka := x_val(xl, ops[0], ea, have_ea)
	b, okb := x_val(xl, ops[1], ea, have_ea)
	if !oka || !okb {fail(xl, "xchg"); return}
	pf(xl, "    { u32 _t = (%s); %s %s }", a, x_store(ops[0], b, ea), x_store(ops[1], "_t", ea))
}


x_rep :: proc(xl: ^Xlat, ins: ^Rx_Ins, kind: string, sz: u8) {
	// the direction from DF (xrc_eflags_sys; cld / std / popfd); rep / repe / repne drive ecx. The element size from
	// the mnemonic suffix / operand size.
	es := sz
	m := ins.inst.mnemonic
	#partial switch m {
	case .MOVSB, .STOSB, .LODSB, .SCASB, .CMPSB: es = 1
	case .MOVSW, .STOSW, .LODSW, .SCASW, .CMPSW: es = 2
	case .MOVSD, .STOSD, .LODSD, .SCASD, .CMPSD: es = 4
	}
	acc := msz(es)
	ax := es == 1 ? "(u8)r_eax" : es == 2 ? "(u16)r_eax" : "r_eax"
	areg := x86.op_reg(es == 1 ? x86.AL : es == 2 ? x86.AX : x86.EAX)
	cmp :: proc(a, b: string, es: u8) -> string {
		return fmt.tprintf("{{ u32 _a = %s, _b = %s, _r = _a - _b; cc_op = CC_SUB; cc_sz = %d; cc_a = _a; cc_b = _b; cc_r = _r; cf = xcf_sub(cc_sz, _a, _b); }}", a, b, es)
	}
	body := ""
	switch kind {
	case "movs": body = fmt.tprintf("%s(r_edi) = %s(r_esi); r_esi += _d; r_edi += _d;", acc, acc)
	case "stos": body = fmt.tprintf("%s(r_edi) = %s; r_edi += _d;", acc, ax)
	case "lods": body = fmt.tprintf("%s r_esi += _d;", x_store(areg, fmt.tprintf("%s(r_esi)", acc), ""))
	case "scas": body = fmt.tprintf("%s r_edi += _d;", cmp(ax, fmt.tprintf("%s(r_edi)", acc), es))
	case "cmps": body = fmt.tprintf("%s r_esi += _d; r_edi += _d;", cmp(fmt.tprintf("%s(r_esi)", acc), fmt.tprintf("%s(r_edi)", acc), es))
	}
	stop := ""
	if kind == "scas" || kind == "cmps" {
		xl.uses_f = true
		stop = ins.inst.flags.rep == .REPNE ? fmt.tprintf(" if ((cc_r & xcc_mask(%d)) == 0) break;", es) : fmt.tprintf(" if ((cc_r & xcc_mask(%d)) != 0) break;", es)
	}
	d := fmt.tprintf("u32 _d = (xrc_eflags_sys & XRC_EFL_DF) ? (u32)-%d : %du;", es, es)
	if ins.inst.flags.rep != .NONE {
		pf(xl, "    { %s while (r_ecx) { %s r_ecx--;%s } }", d, body, stop)
	} else {
		pf(xl, "    { %s %s }", d, body)
	}
}


// __security_check_cookie (VC9: cmp ecx, [__security_cookie]; jne __report_gsfailure; rep ret): the cookie it compares
// with, when the function at a is it. Called at the end of every /GS function (~100k calls a frame), it is inlined:
// the compare (its flags too), and the call only when the cookie differs.
x_cookie_check :: proc(a: u32) -> (u32, bool) {
	b := exe_read(a, 10)
	if len(b) < 10 || b[0] != 0x3b || b[1] != 0x0d || b[6] != 0x75 || b[7] != 0x02 {return 0, false}
	if !((b[8] == 0xf3 && b[9] == 0xc3) || b[8] == 0xc3) {return 0, false}
	return u32(b[2]) | u32(b[3]) << 8 | u32(b[4]) << 16 | u32(b[5]) << 24, true
}

// ---- inlining small leaf functions (x_inline_plan) ----
// A direct call to a small function that calls nothing (a getter, a setter, a compare, a small vector op: over half of
// the calls the game makes) costs more than the callee: the return address pushed, every register stored to XCtx, the
// GIL's preemption count, a C call, the callee loading its registers and storing them back, the return address
// checked, the registers reloaded. Such a callee's translation is put in its callers instead, run on their locals:
// the call's push of the original's return address, the callee's instructions (its labels made the site's own, its
// lazy flags in locals of its own -- ic_*, as a called function's are its own), and each ret as a pop of that address
// (checked: a callee that changed its return address is fatal, as xrc_bad_return is for a call) and a jump past the
// body. Everything the original does to guest memory -- the return address in the slot below esp included -- is the
// same. Not inlined: callees that call (or tail-jump) anything, x87 or MMX code (per-function stacks / state), the
// float compares that set the flags through XFCOMI, hand-written (fast / wrapped) functions, and anything over
// XRC_INLINE_MAX instructions (default 24). XRC_INLINE=0: none (to compare); XRC_INLINE_HOT=FILE: only the
// functions listed there (hex addresses, one a line: e.g. the most called ones of an XRC_COUNT_CALLS run).
X_Inline :: struct {
	body:     string, // the translation, flag locals renamed (and its xmm registers XI[k]: the inlined body's own)
	nins:     int,
	uses_x:   bool,
	xmask:    u8, // the xmm registers it names
	uses_f:   bool,
	uses_ffl: bool,
	direct:   []u32, // the functions it calls directly (declared before its callers)
}
x_inline: map[u32]X_Inline
x_inline_off := false // (planning: the candidates translated without inlining in them -- one level only)

// ---- speculative devirtualization (x_devirt_load) ----
// An indirect call (a virtual method, a function pointer) goes through the call site's inline cache: an indirect C call
// with every register stored and reloaded. A site that, in a profile of the game (an XRC_COUNT_CALLS build's
// XRC_SITE_FILE: "calls return-address target" lines), nearly always calls one function gets that function as a
// guarded fast path: if the target is it, its translation inlined (x_inline) or called directly (XCALLDK: the
// callee-saved registers kept); else the cache's call as before. Exact either way -- the guard compares the target the
// original computed. The profile: data\devirt_sites.txt beside xrc.exe, or XRC_DEVIRT=FILE; XRC_DEVIRT=0: none.
// A site qualifies with at least XRC_DEVIRT_MIN calls (default 1000) of which 90% go to one target.
// A site with two targets (a base class and one derived class, say) that together take 95% of its calls, each at
// least 5%, gets both (x_devirt2: the second).
x_devirt: map[u32]u32 // return address (the site) -> its target
x_devirt2: map[u32]u32

x_devirt_load :: proc() {
	x_devirt = make(map[u32]u32)
	x_devirt2 = make(map[u32]u32)
	f := os.get_env("XRC_DEVIRT", context.temp_allocator)
	if f == "0" || x_fl_verify() {return}
	if f == "" {
		f = table_path("devirt_sites.txt") // (the game's data folder; none there: no devirtualization)
		if !os.exists(f) {return}
	}
	minn := 1000
	if v := os.get_env("XRC_DEVIRT_MIN", context.temp_allocator); v != "" {minn, _ = strconv.parse_int(v)}
	txt, ok := read_text(f)
	if !ok {fmt.printf("xrc: devirtualization profile %s not readable\n", f); return}
	// (profiles of several scenes concatenated: a site's target counts summed first)
	Site :: struct {total, best, best2: int, t, t2: u32}
	pairs := make(map[[2]u32]int, context.temp_allocator)
	for line in strings.split_lines_iterator(&txt) {
		w := strings.fields(line, context.temp_allocator)
		if len(w) != 3 {continue}
		n, ok1 := strconv.parse_int(w[0])
		ra, ok2 := hex_u32(w[1])
		t, ok3 := hex_u32(w[2])
		if !ok1 || !ok2 || !ok3 {continue}
		pairs[{ra, t}] += n
	}
	sites := make(map[u32]Site, context.temp_allocator)
	for k, n in pairs {
		st := sites[k[0]]
		st.total += n
		// (ties to the lower address: the same output every run -- the map's order is not)
		if n > st.best || (n == st.best && k[1] < st.t) {
			st.best2, st.t2 = st.best, st.t
			st.best, st.t = n, k[1]
		} else if n > st.best2 || (n == st.best2 && k[1] < st.t2) {st.best2, st.t2 = n, k[1]}
		sites[k[0]] = st
	}
	n2 := 0
	for ra, st in sites {
		if st.total < minn || !(st.t in x_direct_ok) {continue}
		if st.best * 10 >= st.total * 9 {
			x_devirt[ra] = st.t
		} else if (st.best + st.best2) * 20 >= st.total * 19 && st.best2 * 20 >= st.total && st.t2 in x_direct_ok {
			x_devirt[ra] = st.t
			x_devirt2[ra] = st.t2
			n2 += 1
		}
	}
	fmt.printf("xrc: %d indirect call sites devirtualized, %d of them with two targets (profile %s: %d sites)\n", len(x_devirt), n2, f, len(sites))
}

// the callee's flag locals as the caller's ic_* (whole identifiers only)
x_inline_rename :: proc(t: string) -> string {
	idc :: proc(c: u8) -> bool {return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'}
	REN :: [?][2]string{{"cc_op", "ic_op"}, {"cc_sz", "ic_sz"}, {"cc_a", "ic_a"}, {"cc_b", "ic_b"}, {"cc_r", "ic_r"}, {"cc_fa", "ic_fa"}, {"cc_fb", "ic_fb"}, {"cf", "icf"}}
	b := strings.builder_make()
	from := 0
	for k := 0; k < len(t); k += 1 {
		if (k > 0 && idc(t[k - 1])) || (t[k] != 'c') {continue}
		for r in REN {
			e := k + len(r[0])
			if e <= len(t) && t[k:e] == r[0] && (e == len(t) || !idc(t[e])) {
				strings.write_string(&b, t[from:k])
				strings.write_string(&b, r[1])
				from = e
				k = e - 1
				break
			}
		}
	}
	strings.write_string(&b, t[from:])
	return strings.to_string(b)
}

x_inline_plan :: proc(addrs: []u32, funcs: map[u32]u32) {
	x_inline = make(map[u32]X_Inline)
	if os.get_env("XRC_INLINE", context.temp_allocator) == "0" || x_fl_verify() {return}
	maxn := 24
	if v := os.get_env("XRC_INLINE_MAX", context.temp_allocator); v != "" {maxn, _ = strconv.parse_int(v)}
	hot: map[u32]bool
	if hf := os.get_env("XRC_INLINE_HOT", context.temp_allocator); hf != "" {
		hot = make(map[u32]bool)
		if txt, okh := read_text(hf); okh {
			for line in strings.split_lines_iterator(&txt) {if ha, okx := hex_u32(strings.trim_space(line)); okx {hot[ha] = true}}
		}
		fmt.printf("xrc: inlining only the %d functions of %s\n", len(hot), hf)
	}
	// a hot tier: the functions a call profile found called most (data\inline_hot.txt beside xrc.exe or
	// XRC_INLINE_HOTLIST=FILE, an address per line) inlined up to XRC_INLINE_HOTMAX instructions (default 64)
	hotl := make(map[u32]bool)
	hotmax := 64
	if v := os.get_env("XRC_INLINE_HOTMAX", context.temp_allocator); v != "" {hotmax, _ = strconv.parse_int(v)}
	{
		hf := os.get_env("XRC_INLINE_HOTLIST", context.temp_allocator)
		if hf == "" {hf = table_path("inline_hot.txt")} // (the game's data folder)
		if hf != "0" && os.exists(hf) {
			if txt, okh := read_text(hf); okh {
				for line in strings.split_lines_iterator(&txt) {if ha, okx := hex_u32(strings.trim_space(line)); okx {hotl[ha] = true}}
			}
		}
	}
	// (x87 calls, the float compare macro, the flag check's shadows)
	// (XRC_UD: a fatal error, reported as in the function it is in)
	BAD :: [?]string{"XRC_FRET", "XFCOMI", "x_fa", "x_fret", "p_op", "XRC_THUNK", "LX_", "ic_", "XRC_RETURNX", "XRC_UD"}
	x_inline_off = true
	nl, nc := 0, 0
	why: u32 = 0 // XRC_INLINE_WHY=addr: what keeps that function from being inlined (a diagnostic)
	if v := os.get_env("XRC_INLINE_WHY", context.temp_allocator); v != "" {why, _ = hex_u32(v)}
	if why != 0 {fmt.printf("xrc: inline %08x: direct_ok %v fast %v size %d\n", why, why in x_direct_ok, why in x_fast, funcs[why])}
	for a in addrs {
		if !(a in x_direct_ok) || a in x_fast || (hot != nil && !(a in hot)) {continue}
		// (bytes: no use translating a long one -- loosely: the map's size runs to the next function it knows, padding
		// and code it never saw included; the instruction count decides)
		if funcs[a] == 0 || funcs[a] > 4096 {continue}
		xl := x_translate(a, funcs[a])
		ok := xl.ok && (xl.nins <= maxn || (a in hotl && xl.nins <= hotmax)) && !xl.uses_fp && !xl.uses_mm && len(xl.slots) == 0 && !xl.fret
		body := strings.to_string(xl.b)
		if ok {for w in BAD {if strings.contains(body, w) {ok = false; break}}}
		calls := strings.contains(body, "XCALL")
		if ok && !strings.contains(body, "XRC_RETURN(") {ok = false} // (it never returns: nothing to gain)
		if a == why {
			bw := ""
			for w in BAD {if strings.contains(body, w) {bw = w}}
			fmt.printf("xrc: inline %08x: ok %v nins %d fp %v mm %v slots %d fret %v x %v calls %v bad %q -> %v\n", a, xl.ok, xl.nins, xl.uses_fp, xl.uses_mm, len(xl.slots), xl.fret, xl.uses_x, calls, bw, ok)
		}
		if ok {
			il := X_Inline{nins = xl.nins, uses_x = xl.uses_x, uses_f = xl.uses_f, uses_ffl = xl.uses_ffl}
			t := x_inline_rename(body)
			if xl.uses_x { // X[k] -> XI[k], and which k
				bx := strings.builder_make()
				from := 0
				for q := 0; q + 4 <= len(t); q += 1 {
					if t[q] == 'X' && t[q + 1] == '[' && t[q + 2] >= '0' && t[q + 2] <= '7' && t[q + 3] == ']' && (q == 0 || !(t[q - 1] == '_' || (t[q - 1] >= 'A' && t[q - 1] <= 'Z') || (t[q - 1] >= 'a' && t[q - 1] <= 'z') || (t[q - 1] >= '0' && t[q - 1] <= '9'))) {
						il.xmask |= 1 << (t[q + 2] - '0')
						strings.write_string(&bx, t[from:q])
						strings.write_string(&bx, "XI")
						from = q + 1
					}
				}
				strings.write_string(&bx, t[from:])
				t = strings.to_string(bx)
				if calls {
					// its calls: its xmm registers to XCtx before each and back after (the callee's own, x86's
					// volatile ones); the call macros under names x_append_function's SSE renaming leaves alone
					// (it would store the caller's X over them: xrc.h XCDK_ / XCD_ / XC_)
					sv := strings.builder_make()
					ld := strings.builder_make()
					for k in 0 ..< 8 {
						if il.xmask & (1 << u8(k)) == 0 {continue}
						fmt.sbprintf(&sv, " c->x[%d] = XI[%d];", k, k)
						fmt.sbprintf(&ld, " XI[%d] = c->x[%d];", k, k)
					}
					bl := strings.builder_make()
					rest := t
					for line in strings.split_lines_iterator(&rest) {
						if strings.contains(line, "XCALL") {
							l2, _ := strings.replace_all(line, "XCALLDK(", "XCDK_(")
							l2, _ = strings.replace_all(l2, "XCALLD(", "XCD_(")
							l2, _ = strings.replace_all(l2, "XCALL(", "XC_(")
							fmt.sbprintf(&bl, "    {{%s\n%s\n   %s }}\n", strings.to_string(sv), l2, strings.to_string(ld))
						} else {
							strings.write_string(&bl, line)
							strings.write_byte(&bl, '\n')
						}
					}
					t = strings.to_string(bl)
					if strings.contains(t, "XCALL") {ok = false} // (another call macro: not handled)
				}
			}
			if ok {
				il.body = t
				il.direct = slice.clone(xl.direct[:])
				x_inline[a] = il
				if calls {nc += 1} else {nl += 1}
			}
		}
		strings.builder_destroy(&xl.b)
		delete(xl.labels)
		delete(xl.direct)
		free_all(context.temp_allocator)
	}
	x_inline_off = false
	fmt.printf("xrc: %d functions of up to %d instructions inlined at their direct calls (%d leaves, %d that call)\n", len(x_inline), maxn, nl, nc)
}

// the inlined call at ins (a direct call to target): the callee's body in place, its labels L_X as L_X_<site>, its
// returns to the site's end
x_inline_site :: proc(xl: ^Xlat, ins: ^Rx_Ins, target: u32, il: X_Inline, alt := 0) {
	// (alt: a second body at the same call -- devirtualization's second target: labels of its own)
	site := alt == 0 ? fmt.tprintf("%x", ins.addr) : fmt.tprintf("%x_%d", ins.addr, alt)
	ra := ins.addr + ins.len
	pf(xl, "    /* inlined %08x */", target)
	pf(xl, "    {{ r_esp -= 4; M32(r_esp) = 0x%xu;", ra)
	// (SSE: the callee's xmm registers its own locals XI, from XCtx as at a call -- the caller's own ones there first,
	// XRC_IXOUT_ -- and back at its returns, then the caller's again, XRC_IXIN_: x_append_function defines those)
	xst := ""
	if il.uses_x {
		xl.uses_ix = true
		ld := strings.builder_make(context.temp_allocator)
		st := strings.builder_make(context.temp_allocator)
		for k in 0 ..< 8 {
			if il.xmask & (1 << u8(k)) == 0 {continue}
			fmt.sbprintf(&ld, " XI[%d] = c->x[%d];", k, k)
			fmt.sbprintf(&st, " c->x[%d] = XI[%d];", k, k)
		}
		pf(xl, "    X128 XI[8]; XRC_IXOUT_%s", strings.to_string(ld))
		xst = fmt.tprintf("{{%s XRC_IXIN_ ", strings.to_string(st))
	}
	t := il.body
	b := strings.builder_make(context.temp_allocator)
	from := 0
	hexd :: proc(c: u8) -> bool {return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')}
	for k := 0; k + 10 <= len(t); k += 1 {
		if t[k] == 'L' && t[k + 1] == '_' && (k == 0 || !(t[k - 1] == '_' || (t[k - 1] >= 'a' && t[k - 1] <= 'z') || (t[k - 1] >= 'A' && t[k - 1] <= 'Z') || (t[k - 1] >= '0' && t[k - 1] <= '9'))) {
			all := true
			for q in 2 ..< 10 {if !hexd(t[k + q]) {all = false; break}}
			if all {
				strings.write_string(&b, t[from:k + 10])
				fmt.sbprintf(&b, "_%s", site)
				from = k + 10
				k += 9
				continue
			}
		}
		if t[k] == 'X' && strings.has_prefix(t[k:], "XRC_RETURN(") {
			e := strings.index_byte(t[k:], ')')
			if e > 0 {
				n := t[k + len("XRC_RETURN("):k + e]
				strings.write_string(&b, t[from:k])
				if xst != "" {
					fmt.sbprintf(&b, "%sXRC_INL_RET(%s, 0x%xu, 0x%xu, LX_%s) }}", xst, n, ra, target, site)
					if k + e + 1 < len(t) && t[k + e + 1] == ';' {e += 1} // (the block needs none)
				} else {
					fmt.sbprintf(&b, "XRC_INL_RET(%s, 0x%xu, 0x%xu, LX_%s)", n, ra, target, site)
				}
				from = k + e + 1
				k += e
			}
		}
	}
	strings.write_string(&b, t[from:])
	strings.write_string(&xl.b, strings.to_string(b))
	pf(xl, "    LX_%s: ; }}", site)
	if !xl.dry {for d in il.direct {append(&xl.direct, d)}}
	if il.uses_f {xl.uses_ic = true}
	if il.uses_ffl {xl.uses_icfl = true}
	xl.inl_site = true
	xl.n_inlined += 1
}

x_call :: proc(xl: ^Xlat, ins: ^Rx_Ins, ops: []x86.Operand, ea: string, have_ea: bool) {
	// `call $+5; pop reg` (PIC code reading its own address): only the push
	if ops[0].kind == .RELATIVE && rx_rel_target(ins, 0) == ins.addr + ins.len {
		pf(xl, "    r_esp -= 4; M32(r_esp) = 0x%xu; /* call $+5 */", ins.addr + ins.len)
		return
	}
	t := ""
	direct: u32 = 0
	if ops[0].kind == .RELATIVE {
		direct = rx_rel_target(ins, 0)
		t = x_hexu(u32(direct))
		if il, is_il := x_inline[direct]; is_il && direct != xl.addr && !x_inline_off {
			e := xl.callfp[ins.addr]
			if e.transparent || (e.kin == 0 && e.kout == 0) {x_inline_site(xl, ins, direct, il); return}
		}
		if ck, is_ck := x_cookie_check(direct); is_ck && direct in x_direct_ok {
			e := xl.callfp[ins.addr]
			if e.transparent || (e.kin == 0 && e.kout == 0) {
				xl.uses_f = true
				if xl.fl_on && x_fl_verify() {
					pf(xl, "    { u32 _a = r_ecx, _b = M32(0x%xu); cc_op = CC_SUB; cc_sz = 4; cc_a = _a; cc_b = _b; cc_r = _a - _b; cf = _a < _b; %s", ck, x_fl_shadow(xl, true))
				} else if xl.fl_on { // (a setter like cmp: x_fl_class)
					nd := xl.fl_need[xl.cur]
					pf(xl, "    { u32 _a = r_ecx, _b = M32(0x%xu), _r = _a - _b; %s%s(void)_r;", ck, x_fl_stores(xl, "cc_op = CC_SUB; cc_sz = 4; cc_a = _a; cc_b = _b; cc_r = _r; "), nd & FL_C != 0 ? "cf = _a < _b; " : "")
				} else {
					pf(xl, "    { u32 _a = r_ecx, _b = M32(0x%xu); cc_op = CC_SUB; cc_sz = 4; cc_a = _a; cc_b = _b; cc_r = _a - _b; cf = _a < _b;", ck)
				}
				pf(xl, "    if (_a != _b) {{ u32 x_s_ = r_esp;")
				x_call_fp(xl, ins, t, 0, direct)
				pf(xl, "    if (r_esp >= x_s_) M32(x_s_ - 4) = 0x%xu; }} else M32(r_esp - 4) = 0x%xu; }", ins.addr + ins.len, ins.addr + ins.len)
				return
			}
		}
	} else {
		v, ok := x_val(xl, ops[0], ea, have_ea)
		if !ok {fail(xl, "call*"); return}
		t = v
		ra := ins.addr + ins.len
		if dt, is_dv := x_devirt[ra]; is_dv && !x_inline_off {
			// (the profile's target: inlined or called directly when it is the one, else as before)
			pf(xl, "    {{ u32 x_dv_ = %s;", v)
			e := xl.callfp[ins.addr]
			dts := [2]u32{dt, x_devirt2[ra]}
			for d, di in dts {
				if d == 0 {continue}
				pf(xl, di == 0 ? "    if (x_dv_ == 0x%xu) {{" : "    }} else if (x_dv_ == 0x%xu) {{", d)
				if il, is_il := x_inline[d]; is_il && d != xl.addr && (e.transparent || (e.kin == 0 && e.kout == 0)) {
					x_inline_site(xl, ins, d, il, di)
				} else {
					pf(xl, "    {{ u32 x_s_ = r_esp;")
					x_call_fp(xl, ins, x_hexu(d), 0, d)
					pf(xl, "    if (r_esp >= x_s_) M32(x_s_ - 4) = 0x%xu; }}", ra)
				}
			}
			pf(xl, "    }} else {{ u32 x_s_ = r_esp;")
			x_call_fp(xl, ins, "x_dv_", 0, 0)
			pf(xl, "    if (r_esp >= x_s_) M32(x_s_ - 4) = 0x%xu; }} }}", ra)
			xl.n_devirt += 1
			return
		}
	}
	// the call's push of the return address leaves the host return address in the dead slot below the guest esp;
	// put the original's there (code reads such stale slots as uninitialized locals of the next call) -- unless the
	// callee moved the stack down (__SEH_prolog4, __chkstk): the slot is then inside the new frame, and live
	pf(xl, "    { u32 x_s_ = r_esp;")
	x_call_fp(xl, ins, t, 0, direct)
	pf(xl, "    if (r_esp >= x_s_) M32(x_s_ - 4) = 0x%xu; }", ins.addr + ins.len)
}

// XCALL / XCALLX with the x87 arguments and result of the call site (from the x87 flow pass). ra: the return
// address the call pushes (x_ra_, read by the call macros): the original's (the next instruction), or for a tail
// call its own caller's (x_ip_) -- so the guest stack holds real return addresses, which the runtime's tools walk
// (the frame-pointer chain: stack dumps, the profiler's callers) instead of recording every call
// the callee-saved registers a direct call to a reloads from XCtx (XCALLDK's mask: bit 0 ebx, 1 esi, 2 edi, 3 ebp)
x_abi_reload :: proc(a: u32) -> int {
	k, ok := x_abi_keep[a]
	return ok ? int(~k & 0xf) : 0xf
}

x_call_fp :: proc(xl: ^Xlat, ins: ^Rx_Ins, t: string, key: u32 = 0, direct: u32 = 0, ra: string = "") {
	e := xl.callfp[key != 0 ? key : ins.addr]
	pf(xl, "    x_ra_ = %s;", ra != "" ? ra : x_hexu(u32(ins.addr + ins.len)))
	d := direct != 0 && direct in x_direct_ok
	if d && !xl.dry {
		x_direct_used[direct] = true
		append(&xl.direct, direct)
	}
	if e.transparent || (e.kin == 0 && e.kout == 0) {
		if d {pf(xl, "    XCALLDK(%s, %s_%08x, %d);", t, direct in x_fast ? "xo" : "xf", direct, x_abi_reload(direct))} else {pf(xl, "    XCALL(%s);", t)}
		return
	}
	xl.uses_fp = true
	// the arguments are the top kin values; a CRT helper (x87_deep) leaves the ones below them alone
	base := 0
	if key == 0 {base = max(xl.fdepth[ins.addr] - e.kin, 0)}
	if e.kin >= 1 {pf(xl, "    x_fa[0] = f%d;", base + e.kin - 1)}
	if e.kin >= 2 {pf(xl, "    x_fa[1] = f%d;", base + e.kin - 2)}
	if d {pf(xl, "    XCALLXDK(%s, %d, %d, %s_%08x, %d);", t, e.kin, e.kout, direct in x_fast ? "xo" : "xf", direct, x_abi_reload(direct))} else {pf(xl, "    XCALLX(%s, %d, %d);", t, e.kin, e.kout)}
	if e.kout >= 1 {pf(xl, "    f%d = x_fret;", base)}
}


x_jmp :: proc(xl: ^Xlat, ins: ^Rx_Ins, ops: []x86.Operand, ea: string, have_ea: bool) {
	if ops[0].kind == .RELATIVE {
		t := rx_rel_target(ins, 0)
		if t in xl.labels {
			if ins.addr in xl.spin_back {pf(xl, "    XRC_SPIN_YIELD()")}
			if ins.addr in xl.inl_back {pf(xl, "    XRC_INL_PREEMPT()")}
			pf(xl, "    goto L_%08x;", t)
		} else {
			// jmp outside the function body = a tail call to another function; its result is ours
			x_tail(xl, ins, x_hexu(u32(t)), 0, t)
		}
		return
	}
	v, ok := x_val(xl, ops[0], ea, have_ea)
	if !ok {fail(xl, "jmp*"); return}
	if _, is_tbl := xl.tables[ins.addr]; is_tbl && have_ea {
		// a switch: the table slot (its address) selects the case, nothing is read (the table is in .text, which the
		// retail exe has encrypted: the walker's slots and their targets are the cases); a slot it did not see is read
		// and left through as a tail call
		pf(xl, "    switch (%s) {", ea)
		seen := make(map[u32]bool, context.temp_allocator)
		for st in rx_table_slots[ins.addr] {
			if st[0] in seen || !(st[1] in xl.labels) {continue}
			seen[st[0]] = true
			pf(xl, "    case 0x%xu: goto L_%08x;", st[0], st[1])
		}
		pf(xl, "    }")
		pf(xl, "    { u32 _t = %s;", v)
		x_tail(xl, ins, "_t")
		pf(xl, "    }")
		return
	}
	if _, is_tbl := xl.tables[ins.addr]; is_tbl && ops[0].kind == .REGISTER {
		// a GOT-relative switch (PIC code): the computed target selects the case; any other leaves as a tail call
		pf(xl, "    { u32 _t = %s;", v)
		pf(xl, "    switch (_t) {")
		seen := make(map[u32]bool, context.temp_allocator)
		for st in rx_table_slots[ins.addr] {
			if st[1] in seen || !(st[1] in xl.labels) {continue}
			seen[st[1]] = true
			pf(xl, "    case 0x%xu: goto L_%08x;", st[1], st[1])
		}
		pf(xl, "    }")
		x_tail(xl, ins, "_t")
		pf(xl, "    }")
		return
	}
	// jmp reg / jmp [mem] without a table: a computed tail call (thunks, vtable forwarding)
	pf(xl, "    { u32 _t = %s;", v)
	x_tail(xl, ins, "_t")
	pf(xl, "    }")
}

// a tail call: the return address on top of the stack is ours, so pop it, run the target (which pushes its own and
// pops its arguments), put it back and return through it
x_tail :: proc(xl: ^Xlat, ins: ^Rx_Ins, t: string, key: u32 = 0, direct: u32 = 0) {
	k := key != 0 ? key : ins.addr
	pf(xl, "    { u32 x_ip_ = M32(r_esp); r_esp += 4;")
	x_call_fp(xl, ins, t, k, direct, "x_ip_")
	pf(xl, "    r_esp -= 4; M32(r_esp) = x_ip_; }")
	if xl.fret {
		e := xl.callfp[k]
		pf(xl, e.kout >= 1 ? "    XRC_FRET(f0);" : "    XRC_FRET(0.0);")
	}
	pf(xl, "    XRC_RETURN(0);")
}

x_jcc :: proc(xl: ^Xlat, ins: ^Rx_Ins, ops: []x86.Operand) {
	m := ins.inst.mnemonic
	if ops[0].kind != .RELATIVE {fail(xl, "jcc*"); return}
	t := rx_rel_target(ins, 0)
	if !(t in xl.labels) {
		// a conditional tail call (jcc to another function)
		cc := x_cc(m)
		if cc < 0 {fail(xl, "jcc->outside"); return}
		pf(xl, "    if (%s) {", x_cond_expr(xl, cc))
		x_tail(xl, ins, x_hexu(u32(t)))
		pf(xl, "    }")
		return
	}
	#partial switch m {
	case .JECXZ: pf(xl, "    if (r_ecx == 0) goto L_%08x;", t); return
	case .LOOP:  pf(xl, "    if (--r_ecx) goto L_%08x;", t); return
	case .LOOPE: pf(xl, "    if (--r_ecx && (xcc(cc_op,cc_sz,cc_a,cc_b,cc_r) & XF_ZF)) goto L_%08x;", t); xl.uses_f = true; return
	case .LOOPNE: pf(xl, "    if (--r_ecx && !(xcc(cc_op,cc_sz,cc_a,cc_b,cc_r) & XF_ZF)) goto L_%08x;", t); xl.uses_f = true; return
	}
	cc := x_cc(m)
	if cc < 0 {fail(xl, rx_mn(ins)); return}
	if ins.addr in xl.spin_back {
		pf(xl, "    if (%s) { XRC_SPIN_YIELD() goto L_%08x; }", x_cond_expr(xl, cc), t)
		return
	}
	if ins.addr in xl.inl_back {
		pf(xl, "    if (%s) { XRC_INL_PREEMPT() goto L_%08x; }", x_cond_expr(xl, cc), t)
		return
	}
	pf(xl, "    if (%s) goto L_%08x;", x_cond_expr(xl, cc), t)
}

// translate one function; returns the Xlat (ok, body, reason, counts)

// spin loops: a backward jump whose loop (target .. jump, in address order) has no call but an xchg with memory, a
// lock-prefixed instruction or a pause gives the GIL to a waiting thread before jumping (XRC_SPIN_YIELD)
x_spin_plan :: proc(xl: ^Xlat, ins: []Rx_Ins) {
	xl.spin_back = make(map[u32]bool, context.temp_allocator)
	xl.inl_back = make(map[u32]bool, context.temp_allocator)
	// (ins is in address order: a loop target .. jump is a run of it; whether each instruction is a call / an atomic,
	// as running counts, so a backward jump costs a binary search, not a pass over the whole function)
	have_back := false
	for &i in ins {
		im := i.inst.mnemonic
		if (rx_is_jcc(im) || im == .JMP) && i.inst.operand_count > 0 && i.inst.ops[0].kind == .RELATIVE && rx_rel_target(&i, 0) <= i.addr {have_back = true; break}
	}
	if !have_back {return}
	ncall := make([]i32, len(ins) + 1, context.temp_allocator)
	ninl := make([]i32, len(ins) + 1, context.temp_allocator) // (the calls x_call inlines: direct, or devirtualized)
	natom := make([]i32, len(ins) + 1, context.temp_allocator)
	for &i, k in ins {
		im := i.inst.mnemonic
		atomic := false
		if im != .CALL {
			b := exe_read(i.addr, int(min(i.len, 4)))
			for q in 0 ..< len(b) {
				if b[q] == 0xf0 {atomic = true} // lock
				if b[q] == 0xf3 && q + 1 < len(b) && b[q + 1] == 0x90 {atomic = true} // pause
				if b[q] != 0xf0 && b[q] != 0xf2 && b[q] != 0xf3 && b[q] != 0x66 && b[q] != 0x2e && b[q] != 0x3e && b[q] != 0x26 && b[q] != 0x36 && b[q] != 0x64 && b[q] != 0x65 {break}
			}
			if im == .XCHG {
				for o in 0 ..< int(i.inst.operand_count) {if i.inst.ops[o].kind == .MEMORY {atomic = true}}
			}
		}
		ncall[k + 1] = ncall[k] + (im == .CALL ? 1 : 0)
		inl := false
		if im == .CALL && !x_inline_off && i.inst.operand_count > 0 {
			if i.inst.ops[0].kind == .RELATIVE {
				t := rx_rel_target(&i, 0)
				inl = t in x_inline && t != xl.addr
			} else if dt, is_dv := x_devirt[i.addr + i.len]; is_dv {inl = dt in x_inline}
		}
		ninl[k + 1] = ninl[k] + (inl ? 1 : 0)
		natom[k + 1] = natom[k] + (atomic ? 1 : 0)
	}
	for &j, kj in ins {
		mm := j.inst.mnemonic
		if !((rx_is_jcc(mm) || mm == .JMP) && j.inst.operand_count > 0 && j.inst.ops[0].kind == .RELATIVE) {continue}
		t := rx_rel_target(&j, 0)
		if t > j.addr || !(t in xl.labels) {continue}
		// the first instruction at or after t: the loop is ins[lo ..= kj]
		lo, hi := 0, kj
		for lo < hi {
			mid := (lo + hi) / 2
			if ins[mid].addr < t {lo = mid + 1} else {hi = mid}
		}
		if natom[kj + 1] - natom[lo] > 0 && ncall[kj + 1] - ncall[lo] == 0 {xl.spin_back[j.addr] = true}
		if nc, ni := ncall[kj + 1] - ncall[lo], ninl[kj + 1] - ninl[lo]; ni > 0 && nc == ni {xl.inl_back[j.addr] = true}
	}
}

x_translate :: proc(a: u32, size: u32, exclude: map[i32]bool = nil, nofl := false) -> Xlat {
	xl := Xlat{addr = a, size = size, ok = true, labels = make(map[u32]bool)}
	xl.slots = make(map[i32]bool, context.temp_allocator)
	xl.slot_bad = make(map[i32]bool, context.temp_allocator)
	xl.tables = make(map[u32][dynamic]u32, context.temp_allocator)
	xl.btables = make(map[u32]bool, context.temp_allocator)
	strings.builder_init(&xl.b)
	ins, bad := rx_function(a, size, context.temp_allocator)
	if bad != 0 {
		xl.ok = false
		bb := exe_read(bad, 6)
		xl.reason = fmt.aprintf("decode %08x: % x", bad, bb)
		return xl
	}
	xl.nins = len(ins)
	// collect branch-target labels: every target decoded as part of the function (its range, and the chunks
	// rx_function follows beyond it)
	body := make(map[u32]bool, len(ins), context.temp_allocator)
	for &i in ins {body[i.addr] = true}
	for &i in ins {
		mm := i.inst.mnemonic
		if (rx_is_jcc(mm) || mm == .JMP) && i.inst.operand_count > 0 && i.inst.ops[0].kind == .RELATIVE {
			t := rx_rel_target(&i, 0)
			if t in body {xl.labels[t] = true}
		}
	}
	x_spin_plan(&xl, ins[:])
	for &i, k in ins {
		if i.inst.mnemonic == .JMP && rx_table_jmp(&i) {
			if t, ok := rx_jump_table(ins[:], k, a, size); ok {
				xl.tables[i.addr] = t
				// (its byte table: the movzx just before the jmp)
				for q := k - 1; q >= 0 && q >= k - 8; q -= 1 {
					if _, is_bt := rx_byte_tables[ins[q].addr]; is_bt {xl.btables[ins[q].addr] = true}
				}
				for x in t {xl.labels[x] = true}
			}
		}
	}
	// x87 stack depths (static), float arguments/results of calls, and whether we return a float
	// (marked busy like x87_effect does: its own recursive calls are then decided by this flow)
	x87_busy[a] = true
	fl := x87_solve(ins[:], a, size, 0)
	delete_key(&x87_busy, a)
	if !fl.ok {
		xl.ok = false
		xl.reason = fl.reason
		return xl
	}
	xl.fdepth = fl.depth
	xl.callfp = fl.call
	xl.fret = fl.ret == 1
	xl.fmax = fl.fmax
	if xl.fret || fl.fmax > 0 {xl.uses_fp = true}
	decoded := body // (the same set: every decoded address)
	if !x_no_slots {x_slot_plan(&xl, ins[:], exclude)}
	if !nofl && !x_fl_disabled() {x_fl_plan(&xl, ins[:], a)}
	prev_end: u32 = 0
	// the instructions are in address order: a chunk below the entry (a shared tail the function jumps back to, as
	// the CRT's strchr has) comes first, so execution must start with a jump to the entry
	if len(ins) > 0 && ins[0].addr != a {
		xl.labels[a] = true
		pf(&xl, "    goto L_%08x;", a)
	}
	spans := make([dynamic][2]int, 0, len(ins), context.temp_allocator) // (each instruction's C: x_sync_plan)
	for &i, k in ins {
		xl.cur, xl.cur_addr = k, i.addr
		span0 := len(xl.b.buf)
		if i.addr in xl.labels {pf(&xl, "L_%08x:", i.addr)}
		pf(&xl, "    /* %08x %s %s */", i.addr, rx_mn(&i), inst_op_str(&i))
		// the flags reach this instruction straight from the one before only when it follows it and is no branch target
		xl.fuse, xl.fuse_sz = 0, 0
		if prev_end == i.addr && !(i.addr in xl.labels) {xl.fuse, xl.fuse_sz = xl.fl_kind, xl.fl_sz}
		xl.fl_kind = 0
		x_bad_reg = false
		// the frame ends: the promoted slots back to the stack
		if len(xl.slots) > 0 && ((i.inst.mnemonic == .POP && i.inst.operand_count == 1 && i.inst.ops[0].kind == .REGISTER && i.inst.ops[0].reg == x86.EBP) || i.inst.mnemonic == .LEAVE) {
			for d in xl.slots {pf(&xl, "    M32(r_ebp - 0x%xu) = s_%x;", u32(-d), u32(-d))}
		}
		start := len(xl.b.buf)
		xl.slot_cur = 0
		xl.inl_site = false
		x_emit(&xl, &i)
		if x_bad_reg && xl.ok {fail(&xl, "register")}
		if xl.fl_on && x_fl_verify() && xl.fl_cls[k].known && !xl.fl_cls[k].setter && xl.fl_cls[k].cc < 0 {
			pf(&xl, "    p_op = cc_op; p_sz = cc_sz; p_a = cc_a; p_b = cc_b; p_r = cc_r; p_cf = cf;")
		}
		if xl.fl_on && !xl.fl_bad && !xl.fl_cls[k].known && !xl.inl_site && x_fl_mentions(string(xl.b.buf[start:])) {
			xl.fl_bad = true
			if x_fl_debug() {fmt.eprintf("xrc flags %08x: %08x %s touches the flags: again without the plan\n", a, i.addr, rx_mn(&i))}
		}
		if xl.slot_cur != 0 {
			seg := strings.clone(string(xl.b.buf[start:]), context.temp_allocator)
			name := fmt.tprintf("s_%x", u32(-xl.slot_cur))
			seg, _ = strings.replace_all(seg, "M32(_ea)", name, context.temp_allocator)
			if x_mentions_ea(seg) {
				if os.get_env("XRC_SLOT_DEBUG", context.temp_allocator) != "" {fmt.eprintf("slot %x at %08x: %s\n", -xl.slot_cur, i.addr, seg)}
				xl.slot_bad[xl.slot_cur] = true
			} else {
				resize(&xl.b.buf, start)
				strings.write_string(&xl.b, seg)
			}
		}
		// the frame exists: the promoted slots from the stack
		if len(xl.slots) > 0 && i.addr == xl.slot_init {
			for d in xl.slots {pf(&xl, "    s_%x = M32(r_ebp - 0x%xu);", u32(-d), u32(-d))}
		}
		prev_end = i.addr + i.len
		if !xl.ok {break}
		// execution continues past the function's end (it falls into the next function): a tail call there
		nx := i.addr + i.len
		if !rx_is_branch_end(i.inst.mnemonic) && !(nx in decoded) {
			pf(&xl, "    /* falls through to %08x */", nx)
			x_tail(&xl, &i, x_hexu(u32(nx)), nx | 0x80000000)
		}
		append(&spans, [2]int{span0, len(xl.b.buf)})
	}
	// a promoted slot an instruction used some other way: again without it (and the ones found before)
	if xl.ok && len(xl.slot_bad) > 0 && len(exclude) < 64 {
		ex := make(map[i32]bool, context.temp_allocator)
		for d in exclude {ex[d] = true}
		for d in xl.slot_bad {ex[d] = true}
		strings.builder_destroy(&xl.b)
		delete(xl.labels)
		delete(xl.direct)
		return x_translate(a, size, ex, nofl || xl.fl_bad)
	}
	if xl.ok && xl.fl_bad {
		strings.builder_destroy(&xl.b)
		delete(xl.labels)
		delete(xl.direct)
		return x_translate(a, size, exclude, true)
	}
	// (not in an inlining candidate: its body runs on its callers' locals, whose syncs are theirs)
	if xl.ok && !x_inline_off && len(xl.slots) == 0 && len(spans) == len(ins) {x_sync_plan(&xl, ins[:], spans[:], a)}
	return xl
}

// Stack slot promotion: in a function with the frame `push ebp; mov ebp, esp; sub esp, N`, the dword locals
// [ebp-k] become C locals (s_<k>), which the compiler keeps in registers -- guest memory accesses that any store
// would otherwise order (MSVC /Od code keeps every variable in its slot). Only when nothing can reach the frame but
// these accesses: ebp is used only as their base (and by the epilogue, the /GS cookie's xor), never taken as an
// address (lea, mov from ebp, an indexed [ebp + reg]), and no instruction names [esp + N]; a slot is promoted only
// if every access to its bytes is that same dword, by an integer instruction. Exactness: the slots are loaded from
// memory once the frame exists, and written back before the frame ends (pop ebp / leave), so the stack holds what
// the original left there (stale slots the game reads as uninitialized locals stay the same).
// the address temp _ea as a whole word (not the _ea of r_eax)
x_mentions_ea :: proc(t: string) -> bool {
	for k := 0; k + 3 <= len(t); k += 1 {
		if t[k:k + 3] != "_ea" {continue}
		if k > 0 {
			p := t[k - 1]
			if (p >= 'a' && p <= 'z') || (p >= 'A' && p <= 'Z') || (p >= '0' && p <= '9') || p == '_' {continue}
		}
		if k + 3 < len(t) {
			n := t[k + 3]
			if (n >= 'a' && n <= 'z') || (n >= 'A' && n <= 'Z') || (n >= '0' && n <= '9') || n == '_' {continue}
		}
		return true
	}
	return false
}

x_slot_plan :: proc(xl: ^Xlat, ins: []Rx_Ins, exclude: map[i32]bool) {
	if len(ins) < 4 || ins[0].addr != xl.addr {return}
	is_reg :: proc(o: x86.Operand, r: x86.Register) -> bool {return o.kind == .REGISTER && o.reg == r}
	i0, i1, i2 := &ins[0], &ins[1], &ins[2]
	if i0.inst.mnemonic != .PUSH || i0.inst.operand_count != 1 || !is_reg(i0.inst.ops[0], x86.EBP) {return}
	if i1.inst.mnemonic != .MOV || i1.inst.operand_count != 2 || !is_reg(i1.inst.ops[0], x86.EBP) || !is_reg(i1.inst.ops[1], x86.ESP) {return}
	if i2.inst.mnemonic != .SUB || i2.inst.operand_count != 2 || !is_reg(i2.inst.ops[0], x86.ESP) || i2.inst.ops[1].kind != .IMMEDIATE {return}
	frame := i32(i2.inst.ops[1].immediate)
	if frame <= 0 || frame > 0x10000 {return}
	Acc :: struct {d: i32, sz: i32, good: bool}
	accs := make([dynamic]Acc, context.temp_allocator)
	for &i, k in ins {
		if k < 3 {continue}
		m := i.inst.mnemonic
		n := int(i.inst.operand_count)
		if i.xop != .None {
			// (its memory operand, if any, is ops[0]: checked below as a non-integer access)
		}
		mn := rx_mn(&i)
		intop := i.xop == .None && !(len(mn) > 0 && (mn[0] == 'f' || mn[0] == 'F'))
		for q in 0 ..< n {
			o := i.inst.ops[q]
			if o.kind == .REGISTER {
				cl := x86.reg_class(o.reg)
				if cl != x86.REG_GPR32 && cl != x86.REG_GPR16 && cl != x86.REG_GPR8 && cl != x86.REG_GPR8H {intop = false}
			}
		}
		for q in 0 ..< n {
			o := i.inst.ops[q]
			if o.kind == .REGISTER && (o.reg == x86.EBP || o.reg == x86.BP) {
				switch {
				case m == .POP && q == 0:                                       // the epilogue
				case m == .MOV && q == 1 && n == 2 && is_reg(i.inst.ops[0], x86.ESP): // mov esp, ebp
				case m == .XOR && q == 1 && !is_reg(i.inst.ops[0], x86.EBP):     // the /GS cookie
				case: return
				}
			}
			if o.kind != .MEMORY {continue}
			mm := o.mem
			if mm.segment != 0 {continue}
			if x86.mem_has_base(mm) && x86.mem_base(mm) == x86.ESP {return}
			if x86.mem_has_index(mm) && (x86.mem_index(mm) == x86.EBP || x86.mem_index(mm) == x86.ESP) {return}
			if !x86.mem_has_base(mm) || x86.mem_base(mm) != x86.EBP {continue}
			if x86.mem_has_index(mm) || m == .LEA {return}
			if mm.disp >= 0 {continue} // the arguments: not promoted
			sz := i32(o.size)
			if sz <= 0 || i.xop != .None {sz = 16} // (an SSE / MMX operand: up to 16 bytes)
			append(&accs, Acc{mm.disp, sz, intop && sz == 4})
		}
		if m == .ENTER {return}
	}
	for a in accs {
		if !a.good || a.d < -frame || a.d % 4 != 0 || a.d in exclude {continue}
		clean := true
		for b in accs {
			if b.d < a.d + 4 && a.d < b.d + b.sz && (b.d != a.d || !b.good) {clean = false; break}
		}
		if clean {xl.slots[a.d] = true}
	}
	if len(xl.slots) > 0 {xl.slot_init = i2.addr}
}

// op_str via rexcode printer, best-effort for the comment

inst_op_str :: proc(i: ^Rx_Ins) -> string {
	one := [1]x86.Instruction{i.inst}
	info := [1]x86.Instruction_Info{i.info}
	return strings.trim_space(x86.tprint(one[:], info[:], nil))
}

cmd_xrc :: proc(args: []string) -> int {
	list, out, report, table, matched := "", "", "", "", ""
	all := false
	for i := 0; i < len(args); i += 1 {
		switch args[i] {
		case "--list", "--out", "--report", "--table", "--matched":
			if i + 1 >= len(args) {fmt.eprintf("xrc: %s needs a value\n", args[i]); return 2}
		}
		switch args[i] {
		case "--list": i += 1; list = args[i]
		case "--all": all = true
		case "--out": i += 1; out = args[i]
		case "--report": i += 1; report = args[i]
		case "--table": i += 1; table = args[i]
		case "--matched": i += 1; matched = args[i]
		case "--llvm": ir_mode = true // (xrc_ir.odin: the functions it can do as page_XXXX.ll, the rest C)
		case: fmt.eprintf("xrc: unknown option %s (xrc --help)\n", args[i]); return 2
		}
	}
	// XRC_TIME=1: how long each phase took (stderr)
	x_timing := os.get_env("XRC_TIME", context.temp_allocator) != ""
	t_last := time.tick_now()
	lap :: proc(on: bool, last: ^time.Tick, what: string) {
		if !on {return}
		now := time.tick_now()
		fmt.eprintf("xrc time: %-26s %8.2f s\n", what, time.duration_seconds(time.tick_diff(last^, now)))
		last^ = now
	}
	if EXE_PATH == "" {fmt.eprintln("xrc: --exe FILE is required"); return 2}
	if !load_exe(EXE_PATH) {return 2}
	if !load_tables() {return 2}
	lap(x_timing, &t_last, "load exe + tables")
	addrs := make([dynamic]u32)
	if all {
		extra := x_discover_entries()
		lap(x_timing, &t_last, "discover entries")
		if len(extra) > 0 {fmt.printf("xrc: %d entry points found from direct calls and address references (not in the function map)\n", len(extra))}
		append(&addrs, ..fstarts[:])
	} else if list != "" {
		mm := make(map[u32]bool)
		if !addr_list(list, &mm) {return 2}
		for a in mm {append(&addrs, a)}
		slice.sort(addrs[:])
	} else {
		fmt.eprintln("xrc: --list or --all required"); return 2
	}

	rpt := strings.builder_make()
	strings.write_string(&rpt, "address\tstatus\tnins\treason\n")
	// the portable runtime's dispatch table: every translated function by guest address (sorted), and the
	// functions that need a hand-written override with why
	Ovr :: struct {a: u32, why: string}
	ok_list := make([dynamic]u32)
	ovr_list := make([dynamic]Ovr)
	n_ok, n_bad := 0, 0
	reasons := make(map[string]int)
	// the overrides covered without a translation (--override: overrides/NAME/overrides.tsv): hand-written, or unreachable
	covered := make(map[u32]string)
	wrapped := make(map[u32]bool)
	x_fast = make(map[u32]bool)
	if txt, okc := override_table(); okc {
		for line in strings.split_lines_iterator(&txt) {
			if len(line) == 0 || line[0] == '#' {continue}
			f := strings.split(line, "\t", context.temp_allocator)
			if len(f) < 2 {continue}
			if ca, okh := hex_u32(f[0]); okh {
				// wrapped: translated as usual, but only ever reached through the dispatcher (a hand-written function
				// there falls back to the translation)
				if f[1] == "wrapped" {wrapped[ca] = true} else if f[1] == "fast" {x_fast[ca] = true} else {covered[ca] = strings.clone(f[1])}
			}
		}
	}
	n_manual, n_unreach := 0, 0
	x_no_slots = os.get_env("XRC_NO_SLOTS", context.temp_allocator) != ""
	x_direct_ok = make(map[u32]bool)
	x_direct_used = make(map[u32]bool)
	if all {for a in addrs {if !(a in covered) && !(a in wrapped) {x_direct_ok[a] = true}}}
	// --matched: functions replaced in the dispatcher by other code (e.g. byte-matched recompiled C): no direct calls to them
	if matched != "" {
		if txt, okm := read_text(matched); okm {
			nm := 0
			for line in strings.split_lines_iterator(&txt) {if ma, okh := hex_u32(line); okh {delete_key(&x_direct_ok, ma); nm += 1}}
			fmt.printf("xrc: %d byte-matched functions called through the dispatcher\n", nm)
		}
	}
	x_abi_plan(addrs[:], funcs)
	lap(x_timing, &t_last, "callee-saved registers")
	if all {x_inline_plan(addrs[:], funcs); x_devirt_load()}
	lap(x_timing, &t_last, "inlining plan")
	n_inl_sites, n_dv_sites := 0, 0
	t_flush: time.Duration
	irm: Ir_Mod
	irm.decls = make(map[string]bool)
	irm.copies = make(map[u32]bool)
	ir_funcs = funcs
	irm.defs = make(map[string]bool)
	ir_reasons = make(map[string]int)
	pg := strings.builder_make()
	cur_page: u32 = 0xffffffff
	n_pages, n_written := 0, 0
	for a in addrs {
		if out != "" && a >> 16 != cur_page {
			if cur_page != 0xffffffff {
				n_pages += 1
				tf := time.tick_now()
				if x_flush_page(out, cur_page, &pg) {n_written += 1}
				ir_flush_page(out, cur_page, &irm)
				t_flush += time.tick_since(tf)
			}
			cur_page = a >> 16
		}
		xl := x_translate(a, funcs[a])
		if xl.ok {
			n_ok += 1
			n_inl_sites += xl.n_inlined
			n_dv_sites += xl.n_devirt
			if out != "" && !(ir_mode && ir_try(&irm, &xl, a, funcs[a])) {x_append_function(&pg, &xl)}
			fmt.sbprintf(&rpt, "%08x\tok\t%d\t\n", a, xl.nins)
			append(&ok_list, a)
		} else if k, okk := covered[a]; okk {
			if k == "manual" {n_manual += 1} else {n_unreach += 1}
			fmt.sbprintf(&rpt, "%08x\t%s\t%d\t%s\n", a, k, xl.nins, xl.reason)
			append(&ovr_list, Ovr{a, strings.clone(xl.reason)})
		} else {
			n_bad += 1
			reasons[strings.clone(xl.reason)] += 1
			fmt.sbprintf(&rpt, "%08x\toverride\t%d\t%s\n", a, xl.nins, xl.reason)
			append(&ovr_list, Ovr{a, strings.clone(xl.reason)})
		}
		strings.builder_destroy(&xl.b)
		delete(xl.labels)
		delete(xl.direct)
		free_all(context.temp_allocator)
	}
	if out != "" && cur_page != 0xffffffff {
		n_pages += 1
		tf := time.tick_now()
		if x_flush_page(out, cur_page, &pg) {n_written += 1}
		ir_flush_page(out, cur_page, &irm)
		x_pages_save()
		t_flush += time.tick_since(tf)
		fmt.printf("xrc: %d page files in %s, %d changed; %d calls inlined, %d indirect calls devirtualized\n", n_pages, out, n_written, n_inl_sites, n_dv_sites)
		if x_sync_sites > 0 {
			fmt.printf("xrc: register syncs trimmed at %d calls / returns: %d of %d stores, %d of %d loads left out\n", x_sync_sites, x_sync_st_saved, x_sync_st_all, x_sync_ld_saved, x_sync_ld_all)
		}
		if ir_mode {ir_report()}
		if !override_copy(out) {return 2}
	}
	if x_timing {
		fmt.eprintf("xrc time: %-26s %8.2f s\n", "page files (in translate)", time.duration_seconds(t_flush))
		lap(x_timing, &t_last, "translate + page files")
	}
	if report != "" {
		_ = os.write_entire_file(report, transmute([]u8)strings.to_string(rpt))
	}
	if table != "" {
		tb := strings.builder_make()
		fmt.sbprintf(&tb, "/* xrc dispatch table: %d translated functions, %d overrides. Generated; do not edit. */\n", len(ok_list), len(ovr_list))
		fmt.sbprintf(&tb, "#include \"xrc.h\"\n#include \"xrc_rt.h\"\n\n")
		for a in ok_list {fmt.sbprintf(&tb, "void XRC_CC xf_%08x(XCtx *);\n", a)}
		// directly called functions without a translation after all: forwarded to the dispatcher
		okset := make(map[u32]bool)
		for a in ok_list {okset[a] = true}
		for a in x_direct_used {
			if !(a in okset) {fmt.sbprintf(&tb, "void XRC_CC xf_%08x(XCtx *c) {{ xrc_dispatch(c, 0x%08xu); }}\n", a, a)}
		}
		fmt.sbprintf(&tb, "\nconst XrcFn xrc_fns[] = {{\n")
		for a in ok_list {fmt.sbprintf(&tb, "\t{{0x%08xu, xf_%08x}},\n", a, a)}
		fmt.sbprintf(&tb, "}};\nconst unsigned xrc_nfns = %d;\n\nconst XrcOverride xrc_overrides[] = {{\n", len(ok_list))
		for o in ovr_list {
			why, _ := strings.replace_all(o.why, "\"", "'")
			fmt.sbprintf(&tb, "\t{{0x%08xu, \"%s\"}},\n", o.a, why)
		}
		if len(ovr_list) == 0 {fmt.sbprintf(&tb, "\t{{0, \"\"}},\n")}
		fmt.sbprintf(&tb, "}};\nconst unsigned xrc_noverrides = %d;\n", len(ovr_list))
		x_write_image_facts(&tb)
		_ = os.write_entire_file(table, transmute([]u8)strings.to_string(tb))
	}
	lap(x_timing, &t_last, "report + table")
	Kv :: struct {k: string, v: int}
	kv := make([dynamic]Kv)
	for k, v in reasons {append(&kv, Kv{k, v})}
	slice.sort_by(kv[:], proc(x, y: Kv) -> bool {return x.v > y.v})
	fmt.printf("xrc: %d functions  %d ok (%.1f%%)  %d hand-written  %d unreachable  %d need an override (covered %.2f%%)\n", len(addrs), n_ok,
		100.0 * f64(n_ok) / f64(max(1, len(addrs))), n_manual, n_unreach, n_bad, 100.0 * f64(n_ok + n_manual + n_unreach) / f64(max(1, len(addrs))))
	fmt.println("top override reasons (first blocking mnemonic):")
	for e, i in kv {
		if i >= 40 {break}
		fmt.printf("  %6d  %s\n", e.v, e.k)
	}
	return 0
}


// Entry points the function map misses: code reached only through a stored address (a function pointer in a data
// table, `push offset f`, a CRT fallback like __crtTlsAlloc after a function the map sized past it). An address
// into .text found in a data-section dword or an instruction immediate, that is no known function's start and no
// instruction any known function reaches, becomes a function (sized to the next start); repeated until none are new
// (the new functions' code has references too). They are added to funcs / fstarts.
x_discover_entries :: proc() -> [dynamic]u32 {
	found := make([dynamic]u32)
	covered := make(map[u32]bool)
	refs := make(map[u32]bool)
	for s in exe.secs {
		if s.chars & 0x20000000 != 0 {continue} // (code: its immediates are taken from decoded instructions)
		b := exe_read(s.base, int(min(s.vsize, s.rsize)))
		for o := 0; o + 4 <= len(b); o += 4 {refs[u32le(b, o)] = true}
	}
	// where a known function's code ends (after a ret / jmp, or after its switch tables): a function laid out right
	// behind it starts there
	ends := make(map[u32]bool)
	// targets of direct calls: entry points by construction (a __finally / local unwind block the compiler emits as a
	// called funclet inside or after its function), even where another function's code also reaches them
	calls := make(map[u32]bool)
	todo := make([dynamic]u32)
	append(&todo, ..fstarts[:])
	for len(todo) > 0 {
		for a in todo {
			ins, _ := rx_function(a, funcs[a], context.temp_allocator)
			for &i in ins {
				covered[i.addr] = true
				if rx_is_branch_end(i.inst.mnemonic) {ends[i.addr + i.len] = true}
				// (not `call $+5`: PIC code's way of reading its own address, a push and no function)
				if i.inst.mnemonic == .CALL && i.inst.operand_count > 0 && i.inst.ops[0].kind == .RELATIVE && rx_rel_target(&i, 0) != i.addr + i.len {
					calls[rx_rel_target(&i, 0)] = true
				}
				for k in 0 ..< int(i.inst.operand_count) {
					if i.inst.ops[k].kind == .IMMEDIATE {refs[u32(i32(i.inst.ops[k].immediate))] = true}
				}
			}
			free_all(context.temp_allocator)
		}
		clear(&todo)
		// a function laid out right behind another's switch tables (jump table / byte index table) starts at their end
		// (not where the next dword is another code address: the switch read only the start of a longer table)
		for _, slots in rx_table_slots {
			for st in slots {
				nb := exe_read(st[0] + 4, 4)
				if len(nb) == 4 && u32le(nb, 0) >= exe.text_lo && u32le(nb, 0) < exe.text_hi {continue}
				ends[st[0] + 4] = true
			}
		}
		for _, bt in rx_byte_tables {ends[bt.at + bt.n] = true}
		for v in calls {
			if v < exe.text_lo || v >= exe.text_hi || v in funcs {continue}
			append(&todo, v)
		}
		for v in refs {
			if v in calls {continue}
			if v < exe.text_lo || v >= exe.text_hi || v in funcs || v in covered {continue}
			b := exe_read(v, 1)
			if len(b) == 0 || b[0] == 0xcc || b[0] == 0x00 {continue} // padding / zero fill
			// a function start is aligned behind padding, or right behind the previous function's last instruction
			// (a value that only happens to point into code is neither)
			pb := exe_read(v - 1, 1)
			if !(v in ends) && !(len(pb) == 1 && pb[0] == 0xcc) {continue}
			append(&todo, v)
		}
		if len(todo) == 0 {break}
		slice.sort(todo[:])
		// size: up to the next known start (or the next new one)
		all := make([dynamic]u32, 0, len(fstarts) + len(todo))
		append(&all, ..fstarts[:])
		append(&all, ..todo[:])
		slice.sort(all[:])
		for v in todo {
			k, _ := slice.binary_search(all[:], v)
			nx := k + 1 < len(all) ? all[k + 1] : exe.text_hi
			funcs[v] = nx - v
			append(&found, v)
		}
		fstarts = all[:]
	}
	delete(covered)
	delete(refs)
	delete(ends)
	delete(calls)
	return found
}

// Output: one C file per 64 KB page, DIR/page_XXXX.c, holding that page's translated functions in address order
// (52k single-function files cost ~250x the file operations, and the build compiles pages anyway). A page file is
// rewritten only when its contents change, so its object stays up to date.
// A function that uses SSE calls out with XCALLXM / XCALLM, which also pass the guest xmm registers (c->x) through
// the real ones (the other functions have no X).
x_append_function :: proc(sb: ^strings.Builder, xl: ^Xlat) {
	fmt.sbprintf(sb, "\n/* ---- %08x ---- */\n", xl.addr)
	for d in xl.direct {if d != xl.addr || d in x_fast {fmt.sbprintf(sb, "void XRC_CC %s_%08x(XCtx *c);\n", d in x_fast ? "xo" : "xf", d)}}
	fmt.sbprintf(sb, "void XRC_CC xf_%08x(XCtx *c);\n", xl.addr)
	fmt.sbprintf(sb, "%s(%08x)\n", xl.fret ? "XRC_THUNKF" : "XRC_THUNK", xl.addr)
	// SSE: the xmm registers it uses are locals (XRC_XLOCAL), from c->x at the entry and back to it around calls and
	// at returns (XRC_XIN_ / XRC_XOUT_: only those it names)
	xbody := strings.to_string(xl.b)
	if xl.uses_x {
		// the X[k] it names (one scan for all eight)
		named: [8]bool
		for q := 0; q + 4 <= len(xbody); q += 1 {
			if xbody[q] == 'X' && xbody[q + 1] == '[' && xbody[q + 2] >= '0' && xbody[q + 2] <= '7' && xbody[q + 3] == ']' {named[xbody[q + 2] - '0'] = true}
		}
		strings.write_string(sb, "#define XRC_XIN_ {")
		for k in 0 ..< 8 {if named[k] {fmt.sbprintf(sb, " X[%d] = c->x[%d];", k, k)}}
		strings.write_string(sb, " }\n#define XRC_XOUT_ {")
		for k in 0 ..< 8 {if named[k] {fmt.sbprintf(sb, " c->x[%d] = X[%d];", k, k)}}
		strings.write_string(sb, " }\n")
	}
	if xl.uses_ix {strings.write_string(sb, xl.uses_x ? "#define XRC_IXOUT_ XRC_XOUT_\n#define XRC_IXIN_ XRC_XIN_\n" : "#define XRC_IXOUT_\n#define XRC_IXIN_\n")}
	fmt.sbprintf(sb, "void XRC_CC xf_%08x(XCtx *c) {{\n", xl.addr)
	fmt.sbprintf(sb, "    u32 r_eax = c->eax, r_ecx = c->ecx, r_edx = c->edx, r_ebx = c->ebx;\n")
	fmt.sbprintf(sb, "    u32 r_esp = c->esp, r_ebp = c->ebp, r_esi = c->esi, r_edi = c->edi;\n")
	if xl.uses_f {
		fmt.sbprintf(sb, "    u32 cc_a = 0, cc_b = 0, cc_r = 0, cf = 0; u8 cc_op = 0, cc_sz = 4;\n")
		if xl.fl_on && x_fl_verify() {fmt.sbprintf(sb, "    u32 p_a = 0, p_b = 0, p_r = 0, p_cf = 0; u8 p_op = 0, p_sz = 4;\n")}
		if xl.uses_ffl {fmt.sbprintf(sb, xl.fl_on && x_fl_verify() ? "    double cc_fa = 0, cc_fb = 0, p_fa = 0, p_fb = 0;\n" : "    double cc_fa = 0, cc_fb = 0;\n")}
	}
	if xl.uses_ic {fmt.sbprintf(sb, "    u32 ic_a = 0, ic_b = 0, ic_r = 0, icf = 0; u8 ic_op = 0, ic_sz = 4;\n")}
	if xl.uses_icfl {fmt.sbprintf(sb, "    double ic_fa = 0, ic_fb = 0;\n")}
	if xl.uses_x {fmt.sbprintf(sb, "    XRC_XDECL\n")}
	if xl.uses_mm {fmt.sbprintf(sb, "    u64 x_mm[8] = {{0}};\n")}
	if xl.uses_fp {
		fmt.sbprintf(sb, "    double f0 = 0")
		for k in 1 ..< max(xl.fmax, 1) {fmt.sbprintf(sb, ", f%d = 0", k)}
		fmt.sbprintf(sb, ";\n    u32 x_fsw = 0, x_fcw = 0x27f;\n")
	}
	if len(xl.slots) > 0 {
		fmt.sbprintf(sb, "    u32")
		k := 0
		for d in xl.slots {fmt.sbprintf(sb, "%s s_%x = 0", k > 0 ? "," : "", u32(-d)); k += 1}
		fmt.sbprintf(sb, ";\n")
	}
	fmt.sbprintf(sb, "    XRC_LOCALS\n")
	if xl.uses_fp {fmt.sbprintf(sb, "    XRC_FCW_INIT\n")}
	fmt.sbprintf(sb, "    XRC_PROLOGUE\n")
	fmt.sbprintf(sb, "    XRC_FN_ENTER(0x%xu)\n", xl.addr)
	if xl.uses_x {fmt.sbprintf(sb, "    XRC_XENTRY\n")}
	body := strings.to_string(xl.b)
	if xl.uses_x {
		// the call / return macros that also move the xmm registers: one pass (the names cannot overlap, and no
		// replacement makes another match)
		REN :: [?][2]string{{"XRC_RETURN(", "XRC_RETURNX("}, {"XCALLX(", "XCALLXM("}, {"XCALL(", "XCALLM("}, {"XCALLXD(", "XCALLXMD("},
			{"XCALLD(", "XCALLMD("}, {"XCALLXDK(", "XCALLXMDK("}, {"XCALLDK(", "XCALLMDK("}}
		from := 0
		for q := 0; q < len(body); q += 1 {
			if body[q] != 'X' {continue}
			for r in REN {
				if strings.has_prefix(body[q:], r[0]) {
					strings.write_string(sb, body[from:q])
					strings.write_string(sb, r[1])
					q += len(r[0]) - 1
					from = q + 1
					break
				}
			}
		}
		strings.write_string(sb, body[from:])
	} else {
		strings.write_string(sb, body)
	}
	fmt.sbprintf(sb, "}}\n")
	if xl.uses_x {strings.write_string(sb, "#undef XRC_XIN_\n#undef XRC_XOUT_\n")}
	if xl.uses_ix {strings.write_string(sb, "#undef XRC_IXIN_\n#undef XRC_IXOUT_\n")}
}

// The page manifest, DIR/.xrc_pages: each page file's hash, size and modification time as this run left it. A page
// whose new text has the recorded hash, while its file still has the recorded size and time, is not read back to be
// compared (the 400 MB of pages are on the WSL disk: reading them through \\wsl.localhost cost as long as the
// translation). Anything else -- no manifest, another hash, a file changed since -- reads and compares as before.
X_Page_Rec :: struct {hash: u64, size: i64, mtime: i64}
x_pages_dir: string
x_pages_old: map[string]X_Page_Rec
x_pages_new: strings.Builder

x_pages_load :: proc(dir: string) {
	if x_pages_dir == dir {return}
	x_pages_dir = strings.clone(dir)
	x_pages_old = make(map[string]X_Page_Rec)
	strings.builder_init(&x_pages_new)
	txt, ok := read_text(fmt.tprintf("%s/.xrc_pages", dir))
	if !ok {return}
	for line in strings.split_lines_iterator(&txt) {
		f := strings.fields(line, context.temp_allocator)
		if len(f) != 4 {continue}
		h, okh := strconv.parse_u64_of_base(f[1], 16)
		sz, oks := strconv.parse_i64_of_base(f[2], 10)
		mt, okm := strconv.parse_i64_of_base(f[3], 10)
		if okh && oks && okm {x_pages_old[strings.clone(f[0])] = {h, sz, mt}}
	}
}

x_pages_save :: proc() {
	if x_pages_dir == "" {return}
	_ = os.write_entire_file(fmt.tprintf("%s/.xrc_pages", x_pages_dir), transmute([]u8)strings.to_string(x_pages_new))
}

// a 64-bit hash of the bytes (8 at a time), continuing from h
x_hash :: proc(h: u64, s: string) -> u64 {
	h := h
	i := 0
	for ; i + 8 <= len(s); i += 8 {
		w := intrinsics.unaligned_load((^u64)(raw_data(s[i:])))
		h = (h ~ w) * 0x9e3779b97f4a7c15
		h ~= h >> 29
	}
	for ; i < len(s); i += 1 {
		h = (h ~ u64(s[i])) * 0x100000001b3
	}
	return h ~ (h >> 32)
}

x_file_mtime :: proc(path: string) -> (size, mtime: i64, ok: bool) {
	fi, err := os.stat(path, context.temp_allocator)
	if err != nil {return}
	return fi.size, time.to_unix_nanoseconds(fi.modification_time), true
}

x_flush_page :: proc(dir: string, page: u32, sb: ^strings.Builder) -> bool {
	if strings.builder_len(sb^) == 0 {
		// (--llvm: every function of the page is in its .ll -- an older .c of it must go, or both define them)
		if ir_mode {if p := fmt.tprintf("%s/page_%04x.c", dir, page); os.exists(p) {os.remove(p); return true}}
		return false
	}
	os.make_directory(dir)
	x_pages_load(dir)
	hdr := fmt.tprintf("/* page %04x0000: x86->C (xrc). Generated; do not edit. */\n#define XRC_PAGE 1\n#include \"xrc.h\"\n", page)
	body := strings.to_string(sb^)
	name := fmt.tprintf("page_%04x.c", page)
	path := fmt.tprintf("%s/%s", dir, name)
	h := x_hash(x_hash(0x243f6a8885a308d3, hdr), body)
	size := i64(len(hdr) + len(body))
	same := false
	mt: i64 = 0
	if rec, okr := x_pages_old[name]; okr && rec.hash == h && rec.size == size {
		if fs, fm, okf := x_file_mtime(path); okf && fs == size && fm == rec.mtime {same = true; mt = rec.mtime}
	}
	if !same {
		text := strings.concatenate({hdr, body}, context.temp_allocator)
		old, had := read_file(path)
		same = had && string(old) == text
		if had {delete(old)}
		if !same {_ = os.write_entire_file(path, transmute([]u8)text)}
		if _, fm, okf := x_file_mtime(path); okf {mt = fm}
	}
	fmt.sbprintf(&x_pages_new, "%s %016x %d %d\n", name, h, size, mt)
	strings.builder_reset(sb)
	return !same
}

// What the runtime needs from the exe besides its data sections, for loading the player's own copy (the retail exe:
// .text encrypted by SteamStub, the entry point in its .bind stub; the other sections the same bytes):
//   xrc_entry      the real entry point
//   xrc_data_hash  FNV-1a 64 of every section but .text as the file has them (raw size): the runtime refuses another
//                  version of the game
x_write_image_facts :: proc(tb: ^strings.Builder) {
	d := exe.data
	fmt.sbprintf(tb, "\nconst unsigned xrc_entry = 0x%08xu;\n", exe.entry)
	if exe.bias != 0 {fmt.sbprintf(tb, "const unsigned xrc_load_base = 0x%08xu; /* (a PIE / shared object: its link addresses + this) */\n", exe.bias)}
	// the data sections' hash: every section but code, over the bytes the file has (the runtime hashes the same
	// ranges of its loaded image, before anything writes there)
	h: u64 = 0xcbf29ce484222325
	fmt.sbprintf(tb, "const XrcRange xrc_hash_ranges[] = {{\n")
	nh := 0
	for sec in exe.secs {
		if sec.chars & 0x20000000 != 0 {continue}
		n := sec.vsize == 0 ? sec.rsize : min(sec.rsize, sec.vsize)
		raw := d[sec.raw:min(int(sec.raw + n), len(d))]
		for b in raw {h = (h ~ u64(b)) * 0x100000001b3}
		fmt.sbprintf(tb, "\t{{0x%08xu, %d}},\n", sec.base, len(raw))
		nh += 1
	}
	fmt.sbprintf(tb, "}};\nconst unsigned xrc_nhash_ranges = %d;\nconst unsigned long long xrc_data_hash = 0x%016xull;\n", nh, h)
}
