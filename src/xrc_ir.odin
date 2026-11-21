package xrc

import "core:fmt"
import "core:os"
import "core:slice"
import "core:strconv"
import "core:strings"
import x86 "core:rexcode/isa/x86"

// ---- the LLVM IR backend (xrc --llvm) ----
// The same functions as the C pages, as LLVM IR instead: page_XXXX.ll beside page_XXXX.c, each function in one or
// the other, the same symbol (xf_XXXXXXXX(XCtx *c)) and the same calling protocol as the C translation (xrc.h): the
// dispatch table, the C pages and the runtime call it unchanged. Not every function: one this backend does not handle
// yet (MMX, jump tables, string instructions, x87 transcendentals ... -- see ir_ins) stays C, so the build is always
// complete and the coverage grows instruction by instruction (xrc prints the commonest reasons a function stayed C).
//
// How: the function's control-flow graph (leaders: the entry, every branch target, every instruction after a
// branch, return or call), translated block by block into SSA:
//   - the machine state is SSA values, never memory: the eight registers, the five arithmetic flags (ZF SF CF OF PF,
//     i1), the xmm registers the function names (<4 x i32>, bitcast to the lane type each instruction works in), the
//     x87 stack slots (double; the stack depth before every instruction is static -- x87_solve, the C translation's
//     flow analysis -- so st(i) is a fixed slot) and the x87 status / control words. Each block starts with a phi of
//     each from every predecessor's end state (the blocks are generated first, the phis filled in after); the entry
//     block loads the registers from XCtx;
//   - the flags are computed eagerly by every instruction that sets them: what nothing reads is dead code LLVM
//     removes, and what a branch reads folds (zf of a cmp is an icmp eq) -- the C path's lazy flags and
//     compare-branch fusion, for free;
//   - guest memory is xrc_mem + the 32-bit address, unaligned loads and stores;
//   - calls as XCALLDK / XCALL / XCALLX do it: the x87 arguments to XCtx.fa, the return address pushed, the registers
//     and the xmm registers to XCtx, the preemption check, the callee (directly, or through the site's inline cache
//     in C: xrc_ir_icall, which the runtime provides), its return checked, the registers back (the callee-saved ones
//     only when the callee does not keep them: x_abi_plan), a float result from XCtx.st0;
//   - LLVM's own passes (clang -O2 on the .ll) then remove the phis nobody needs and allocate the registers. The
//     slow paths (a waiting thread's preemption, a bad return) are marked cold and unlikely, so they are laid out of
//     the way.
// The floating point is the C path's, operation by operation: x87 arithmetic in double rounded to the control
// word's precision (XR87), the x86 results for NaN / out-of-range conversions (the "integer indefinite"), SSE
// min/max/compares with x86's NaN rules, rcp/rsqrt exact (as the portable C).

IR_ZF :: 0
IR_SF :: 1
IR_CF :: 2
IR_OF :: 3
IR_PF :: 4
IR_OFF_EIP :: 32
IR_OFF_ST0 :: 40
IR_OFF_FA :: 48
IR_OFF_X :: 64
IR_OFF_FCW :: 192
IR_OFF_MXCSR :: 196
IR_OFF_EFLSYS :: 200
IR_OFF_PREN :: 204

V4I :: "<4 x i32>"
V4F :: "<4 x float>"
V2D :: "<2 x double>"
V2Q :: "<2 x i64>"
V8W :: "<8 x i16>"
V16B :: "<16 x i8>"

// the machine state at a point: SSA value names
Ir_St :: struct {
	r:   [8]string,
	f:   [5]string,
	x:   [8]string,
	st:  [8]string,
	fsw: string,
	fcw: string,
	mm:  [8]string, // MMX (the function's own: x_mm, zero at its entry, never across calls -- MSVC's intrinsics)
}
IR_NCOMP :: 8 + 5 + 8 + 8 + 2 + 8

Ir_Edge :: struct {
	from:     string, // the LLVM block the branch is in
	from_blk: u32,    // its CFG block (0xffffffff: the entry)
	s:        Ir_St,
}

Ir_Mod :: struct {
	decls:   map[string]bool, // functions called
	defs:    map[string]bool, // functions defined here
	globals: strings.Builder, // the inline caches
	body:    strings.Builder,
	n_fn:    int,
	copies:  map[u32]bool, // the inline copies (xi_) its functions call
}

// ---- inlining (the C path's x_inline set: the same callees, the same sites) ----
// A call to a function of x_inline calls an internal alwaysinline copy of it (xi_XXXXXXXX: its translation as IR, put
// in the caller's module) without the preemption check, as the C pages inline it; LLVM then inlines it, and the
// registers through XCtx between caller and callee become plain values (XCtx is noalias). The copy's own calls are
// ordinary ones (one level, as the C).
Ir_Copy :: struct {
	text, globals: string,
	decls:         [dynamic]string,
	ok:            bool,
}
ir_copies: map[u32]Ir_Copy
ir_funcs: map[u32]u32
ir_copy :: proc(t: u32) -> bool {
	if c, ok := ir_copies[t]; ok {return c.ok}
	if ir_copies == nil {ir_copies = make(map[u32]Ir_Copy)}
	ir_copies[t] = {} // (recursion: not while its own copy is made)
	cp: Ir_Copy
	if sz, ok := ir_funcs[t]; ok && sz > 0 {
		xl2 := x_translate(t, sz)
		if xl2.ok {
			mod2: Ir_Mod
			mod2.decls = make(map[string]bool, context.temp_allocator)
			mod2.defs = make(map[string]bool, context.temp_allocator)
			mod2.copies = make(map[u32]bool, context.temp_allocator)
			mod2.globals = strings.builder_make(context.temp_allocator)
			text, ok2, _ := ir_translate(&mod2, &xl2, t, sz, true)
			if ok2 {
				cp.ok = true
				cp.text = strings.clone(text)
				cp.globals = strings.clone(strings.to_string(mod2.globals))
				cp.decls = make([dynamic]string)
				for d in mod2.decls {append(&cp.decls, strings.clone(d))}
			}
		}
	}
	ir_copies[t] = cp
	return cp.ok
}

Ir :: struct {
	a:      u32,
	nv:     int,
	nl:     int,
	s:      Ir_St,
	b:      ^strings.Builder,
	cur:    string, // the current LLVM block's label
	blk:    u32,    // the current CFG block
	ok:     bool,
	why:    string,
	at:     u32, // the instruction being translated (XRC_IR_WHY: where a function failed)
	mod:    ^Ir_Mod,
	xl:     ^Xlat,
	named:  [8]bool, // the xmm registers the function names
	nst:    int,     // x87 slots (the deepest stack)
	fp:     bool,    // x87 state (slots, fsw, fcw)
	mmx:    bool,    // MMX state
	copy:   bool,    // translating an inline copy (xi_: no inlining in it, its globals named apart)
	edges:  map[u32][dynamic]Ir_Edge,
	succ:   map[u32][dynamic]u32,
}

ir_mode := false
ir_n_fns, ir_n_c := 0, 0
ir_reasons: map[string]int

// ---- text helpers (no fmt format strings around IR text: its % and { are fmt's syntax) ----

ir_w :: proc(ir: ^Ir, parts: ..string) {
	strings.write_string(ir.b, "  ")
	for p in parts {strings.write_string(ir.b, p)}
	strings.write_byte(ir.b, '\n')
}
ir_v :: proc(ir: ^Ir) -> string {ir.nv += 1; return strings.concatenate({"%v", ir_num(i64(ir.nv))}, context.temp_allocator)}
ir_num :: proc(v: i64) -> string {
	buf: [32]u8
	return strings.clone(strconv.write_int(buf[:], v, 10), context.temp_allocator)
}
ir_hex8 :: proc(v: u32) -> string {return fmt.tprintf("%08x", v)}
// a double (or a float: LLVM writes float constants as the double of the same value) constant, exactly
ir_dc :: proc(v: f64) -> string {return fmt.tprintf("0x%016X", transmute(u64)v)}
ir_newlabel :: proc(ir: ^Ir, p: string) -> string {ir.nl += 1; return strings.concatenate({p, "_", ir_num(i64(ir.nl))}, context.temp_allocator)}
ir_label :: proc(ir: ^Ir, l: string) {
	strings.write_string(ir.b, l)
	strings.write_string(ir.b, ":\n")
	ir.cur = l
}
ir_fail :: proc(ir: ^Ir, why: string) {
	if ir.ok {
		ir.ok = false
		ir.why = why
		if os.get_env("XRC_IR_WHY", context.temp_allocator) != "" {fmt.printf("xrc: --llvm: %08x (in %08x): %s\n", ir.at, ir.a, why)}
	}
}
ir_ty :: proc(bits: int) -> string {return bits == 8 ? "i8" : bits == 16 ? "i16" : bits == 64 ? "i64" : "i32"}

ir_op :: proc(ir: ^Ir, op, ty, a, b: string) -> string {
	d := ir_v(ir)
	ir_w(ir, d, " = ", op, " ", ty, " ", a, ", ", b)
	return d
}
ir_cmp :: proc(ir: ^Ir, pred, ty, a, b: string) -> string {
	d := ir_v(ir)
	ir_w(ir, d, " = icmp ", pred, " ", ty, " ", a, ", ", b)
	return d
}
ir_fcmp :: proc(ir: ^Ir, pred, ty, a, b: string) -> string {
	d := ir_v(ir)
	ir_w(ir, d, " = fcmp ", pred, " ", ty, " ", a, ", ", b)
	return d
}
ir_cast :: proc(ir: ^Ir, op, from, v, to: string) -> string {
	if from == to && op == "bitcast" {return v}
	d := ir_v(ir)
	ir_w(ir, d, " = ", op, " ", from, " ", v, " to ", to)
	return d
}
ir_sel :: proc(ir: ^Ir, ty, c, a, b: string) -> string {
	d := ir_v(ir)
	ir_w(ir, d, " = select i1 ", c, ", ", ty, " ", a, ", ", ty, " ", b)
	return d
}
// select with a vector condition (ct: <N x i1>)
ir_selv :: proc(ir: ^Ir, ct, c, ty, a, b: string) -> string {
	d := ir_v(ir)
	ir_w(ir, d, " = select ", ct, " ", c, ", ", ty, " ", a, ", ", ty, " ", b)
	return d
}
ir_call1 :: proc(ir: ^Ir, rt, fn, ty, a: string) -> string {
	d := ir_v(ir)
	ir_w(ir, d, " = call ", rt, " @", fn, "(", ty, " ", a, ")")
	return d
}
ir_call2 :: proc(ir: ^Ir, rt, fn, ty, a, b: string) -> string {
	d := ir_v(ir)
	ir_w(ir, d, " = call ", rt, " @", fn, "(", ty, " ", a, ", ", ty, " ", b, ")")
	return d
}
ir_not1 :: proc(ir: ^Ir, a: string) -> string {return ir_op(ir, "xor", "i1", a, "true")}
// an i32 value to iN (its low bits) and back (zero-extended)
ir_tr :: proc(ir: ^Ir, v: string, bits: int) -> string {return bits == 32 ? v : ir_cast(ir, "trunc", "i32", v, ir_ty(bits))}
ir_zx :: proc(ir: ^Ir, v: string, bits: int) -> string {return bits == 32 ? v : ir_cast(ir, "zext", ir_ty(bits), v, "i32")}
ir_sx :: proc(ir: ^Ir, v: string, bits: int) -> string {return bits == 32 ? v : ir_cast(ir, "sext", ir_ty(bits), v, "i32")}
ir_ext :: proc(ir: ^Ir, ty, v: string, k: int) -> string {
	d := ir_v(ir)
	ir_w(ir, d, " = extractelement ", ty, " ", v, ", i32 ", ir_num(i64(k)))
	return d
}
ir_ins_el :: proc(ir: ^Ir, ty, v, ety, e: string, k: int) -> string {
	d := ir_v(ir)
	ir_w(ir, d, " = insertelement ", ty, " ", v, ", ", ety, " ", e, ", i32 ", ir_num(i64(k)))
	return d
}
ir_shuf :: proc(ir: ^Ir, ty, a, b: string, mask: []int, rty: string) -> string {
	sb := strings.builder_make(context.temp_allocator)
	strings.write_string(&sb, "<")
	strings.write_string(&sb, ir_num(i64(len(mask))))
	strings.write_string(&sb, " x i32> <")
	for k, i in mask {
		if i > 0 {strings.write_string(&sb, ", ")}
		strings.write_string(&sb, "i32 ")
		strings.write_string(&sb, ir_num(i64(k)))
	}
	strings.write_string(&sb, ">")
	d := ir_v(ir)
	ir_w(ir, d, " = shufflevector ", ty, " ", a, ", ", ty, " ", b, ", ", strings.to_string(sb))
	_ = rty
	return d
}

// a pointer into XCtx / guest memory
ir_cfield :: proc(ir: ^Ir, off: int) -> string {
	if off == 0 {return "%c"}
	d := ir_v(ir)
	ir_w(ir, d, " = getelementptr i8, ptr %c, i64 ", ir_num(i64(off)))
	return d
}
ir_cload :: proc(ir: ^Ir, off: int, ty: string) -> string {
	p := ir_cfield(ir, off)
	d := ir_v(ir)
	ir_w(ir, d, " = load ", ty, ", ptr ", p, ", align 4")
	return d
}
ir_cstore :: proc(ir: ^Ir, off: int, ty, v: string) {
	p := ir_cfield(ir, off)
	ir_w(ir, "store ", ty, " ", v, ", ptr ", p, ", align 4")
}
ir_gptr :: proc(ir: ^Ir, ea: string) -> string {
	z := ir_cast(ir, "zext", "i32", ea, "i64")
	d := ir_v(ir)
	ir_w(ir, d, " = getelementptr i8, ptr %xm, i64 ", z)
	return d
}
ir_loadt :: proc(ir: ^Ir, ea: string, ty: string) -> string {
	p := ir_gptr(ir, ea)
	d := ir_v(ir)
	ir_w(ir, d, " = load ", ty, ", ptr ", p, ", align 1")
	return d
}
ir_storet :: proc(ir: ^Ir, ea: string, ty: string, v: string) {
	p := ir_gptr(ir, ea)
	ir_w(ir, "store ", ty, " ", v, ", ptr ", p, ", align 1")
}
ir_load :: proc(ir: ^Ir, ea: string, bits: int) -> string {return ir_zx(ir, ir_loadt(ir, ea, ir_ty(bits)), bits)}
ir_store :: proc(ir: ^Ir, ea: string, bits: int, v: string) {ir_storet(ir, ea, ir_ty(bits), ir_tr(ir, v, bits))}

// ---- registers and operands (i32 values; a narrower operand zero-extended) ----

ir_reg :: proc(ir: ^Ir, r: x86.Register) -> (hw: int, bits: int, high: bool) {
	h := int(x86.reg_hw(r))
	switch x86.reg_class(r) {
	case x86.REG_GPR32: if h < 8 {return h, 32, false}
	case x86.REG_GPR16: if h < 8 {return h, 16, false}
	case x86.REG_GPR8:  if h < 8 {return h, 8, false}
	case x86.REG_GPR8H: return h & 3, 8, true
	}
	ir_fail(ir, "register")
	return 0, 32, false
}
ir_reg_read :: proc(ir: ^Ir, r: x86.Register) -> string {
	// segment registers read as the fixed selectors of a 32-bit Windows process (WOW64): ES CS SS DS FS GS
	if x86.reg_class(r) == x86.REG_SEG {
		SEL := [6]i64{0x2b, 0x23, 0x2b, 0x2b, 0x53, 0x2b}
		if hw := int(x86.reg_hw(r)); hw < 6 {return ir_num(SEL[hw])}
	}
	hw, bits, high := ir_reg(ir, r)
	v := ir.s.r[hw]
	if high {return ir_op(ir, "and", "i32", ir_op(ir, "lshr", "i32", v, "8"), "255")}
	if bits == 16 {return ir_op(ir, "and", "i32", v, "65535")}
	if bits == 8 {return ir_op(ir, "and", "i32", v, "255")}
	return v
}
ir_reg_write :: proc(ir: ^Ir, r: x86.Register, v: string) {
	hw, bits, high := ir_reg(ir, r)
	old := ir.s.r[hw]
	if high {
		k := ir_op(ir, "and", "i32", old, "-65281")
		s := ir_op(ir, "shl", "i32", ir_op(ir, "and", "i32", v, "255"), "8")
		ir.s.r[hw] = ir_op(ir, "or", "i32", k, s)
	} else if bits == 16 {
		ir.s.r[hw] = ir_op(ir, "or", "i32", ir_op(ir, "and", "i32", old, "-65536"), ir_op(ir, "and", "i32", v, "65535"))
	} else if bits == 8 {
		ir.s.r[hw] = ir_op(ir, "or", "i32", ir_op(ir, "and", "i32", old, "-256"), ir_op(ir, "and", "i32", v, "255"))
	} else {
		ir.s.r[hw] = v
	}
}
// the address of a memory operand (flat segments only)
ir_addr :: proc(ir: ^Ir, m: x86.Memory) -> string {
	if m.segment == 5 || m.segment == 6 {ir_fail(ir, "fs/gs"); return "0"}
	v := ""
	if x86.mem_has_base(m) {
		hw, bits, _ := ir_reg(ir, x86.mem_base(m))
		if bits != 32 {ir_fail(ir, "addr16")}
		v = ir.s.r[hw]
	}
	if x86.mem_has_index(m) {
		hw, bits, _ := ir_reg(ir, x86.mem_index(m))
		if bits != 32 {ir_fail(ir, "addr16")}
		x := ir.s.r[hw]
		switch x86.mem_scale(m) {
		case 2: x = ir_op(ir, "shl", "i32", x, "1")
		case 4: x = ir_op(ir, "shl", "i32", x, "2")
		case 8: x = ir_op(ir, "shl", "i32", x, "3")
		}
		v = v == "" ? x : ir_op(ir, "add", "i32", v, x)
	}
	if v == "" {return ir_num(i64(i32(u32(m.disp))))}
	if m.disp != 0 {v = ir_op(ir, "add", "i32", v, ir_num(i64(m.disp)))}
	return v
}
ir_read :: proc(ir: ^Ir, op: x86.Operand, ea: string) -> string {
	#partial switch op.kind {
	case .REGISTER:
		if ir_is_x(op) {return ir_ext(ir, V4I, ir.s.x[ir_xn(op)], 0)}
		return ir_reg_read(ir, op.reg)
	case .IMMEDIATE:
		// (sign-extended to 32 bits, as x86 extends a short immediate to the operand size: users truncate)
		return ir_num(i64(i32(op.immediate)))
	case .MEMORY:
		if op.size != 1 && op.size != 2 && op.size != 4 {ir_fail(ir, "msize"); return "0"}
		return ir_load(ir, ea, int(op.size) * 8)
	}
	ir_fail(ir, "operand")
	return "0"
}
ir_write :: proc(ir: ^Ir, op: x86.Operand, ea: string, v: string) {
	#partial switch op.kind {
	case .REGISTER:
		if ir_is_x(op) {ir_fail(ir, "write xmm"); return}
		ir_reg_write(ir, op.reg, v)
		return
	case .MEMORY:
		if op.size != 1 && op.size != 2 && op.size != 4 {ir_fail(ir, "msize"); return}
		ir_store(ir, ea, int(op.size) * 8, v)
		return
	}
	ir_fail(ir, "operand")
}

// ---- flags ----

ir_zsp :: proc(ir: ^Ir, rn: string, bits: int) {
	ty := ir_ty(bits)
	ir.s.f[IR_ZF] = ir_cmp(ir, "eq", ty, rn, "0")
	ir.s.f[IR_SF] = ir_cmp(ir, "slt", ty, rn, "0")
	b8 := bits == 8 ? rn : ir_cast(ir, "trunc", ty, rn, "i8")
	p := ir_call1(ir, "i8", "llvm.ctpop.i8", "i8", b8)
	ir.s.f[IR_PF] = ir_cmp(ir, "eq", "i8", ir_op(ir, "and", "i8", p, "1"), "0")
}
// the condition (tttn 0..15) as an i1
ir_cond :: proc(ir: ^Ir, cc: int) -> string {
	f := &ir.s.f
	base := ""
	switch cc >> 1 {
	case 0: base = f[IR_OF]
	case 1: base = f[IR_CF]
	case 2: base = f[IR_ZF]
	case 3: base = ir_op(ir, "or", "i1", f[IR_CF], f[IR_ZF])
	case 4: base = f[IR_SF]
	case 5: base = f[IR_PF]
	case 6: base = ir_op(ir, "xor", "i1", f[IR_SF], f[IR_OF])
	case 7: base = ir_op(ir, "or", "i1", f[IR_ZF], ir_op(ir, "xor", "i1", f[IR_SF], f[IR_OF]))
	}
	return cc & 1 != 0 ? ir_not1(ir, base) : base
}
// fcomi / comiss: ZF PF CF from an ordered compare (all three when unordered), OF SF clear
ir_fcomi :: proc(ir: ^Ir, ty, a, b: string) {
	un := ir_fcmp(ir, "uno", ty, a, b)
	ir.s.f[IR_ZF] = ir_fcmp(ir, "ueq", ty, a, b)
	ir.s.f[IR_CF] = ir_fcmp(ir, "ult", ty, a, b)
	ir.s.f[IR_PF] = un
	ir.s.f[IR_OF], ir.s.f[IR_SF] = "false", "false"
}

// add / sub / adc / sbb / and / or / xor / cmp / test (r: the iN result)
ir_alu :: proc(ir: ^Ir, m: x86.Mnemonic, a32, b32: string, bits: int) -> string {
	ty := ir_ty(bits)
	a := ir_tr(ir, a32, bits)
	b := ir_tr(ir, b32, bits)
	r := ""
	f := &ir.s.f
	#partial switch m {
	case .ADD, .ADC:
		r = ir_op(ir, "add", ty, a, b)
		cf := ir_cmp(ir, "ult", ty, r, a)
		if m == .ADC {
			ci := ir_cast(ir, "zext", "i1", f[IR_CF], ty)
			r = ir_op(ir, "add", ty, r, ci)
			cf2 := ir_op(ir, "and", "i1", f[IR_CF], ir_cmp(ir, "eq", ty, r, "0"))
			cf = ir_op(ir, "or", "i1", cf, cf2)
		}
		f[IR_CF] = cf
		ox := ir_op(ir, "and", ty, ir_op(ir, "xor", ty, a, r), ir_op(ir, "xor", ty, b, r))
		f[IR_OF] = ir_cmp(ir, "slt", ty, ox, "0")
	case .SUB, .SBB, .CMP:
		r = ir_op(ir, "sub", ty, a, b)
		cf := ir_cmp(ir, "ult", ty, a, b)
		if m == .SBB {
			ci := ir_cast(ir, "zext", "i1", f[IR_CF], ty)
			r = ir_op(ir, "sub", ty, r, ci)
			cf2 := ir_op(ir, "and", "i1", f[IR_CF], ir_cmp(ir, "eq", ty, a, b))
			cf = ir_op(ir, "or", "i1", cf, cf2)
		}
		f[IR_CF] = cf
		ox := ir_op(ir, "and", ty, ir_op(ir, "xor", ty, a, b), ir_op(ir, "xor", ty, a, r))
		f[IR_OF] = ir_cmp(ir, "slt", ty, ox, "0")
	case .AND, .TEST: r = ir_op(ir, "and", ty, a, b)
	case .OR:         r = ir_op(ir, "or", ty, a, b)
	case .XOR:        r = ir_op(ir, "xor", ty, a, b)
	}
	#partial switch m {
	case .AND, .TEST, .OR, .XOR: f[IR_CF], f[IR_OF] = "false", "false"
	}
	ir_zsp(ir, r, bits)
	return r
}

// shl / shr / sar / rol / ror: the count masked to 5 bits; a zero count leaves the flags (and for rotates the value)
ir_shift :: proc(ir: ^Ir, m: x86.Mnemonic, a32, cnt32: string, bits: int) -> string {
	ty := ir_ty(bits)
	n := ir_op(ir, "and", "i32", cnt32, "31")
	z := ir_cmp(ir, "eq", "i32", n, "0")
	old := ir.s.f
	sb := ir_num(i64(bits))
	r32 := ""
	cf, of := "", ""
	#partial switch m {
	case .SHL, .SAL:
		r32 = ir_op(ir, "shl", "i32", a32, n)
		k := ir_op(ir, "and", "i32", ir_op(ir, "sub", "i32", sb, n), "31")
		cf = ir_cast(ir, "trunc", "i32", ir_op(ir, "lshr", "i32", a32, k), "i1")
	case .SHR:
		r32 = ir_op(ir, "lshr", "i32", a32, n)
		k := ir_op(ir, "and", "i32", ir_op(ir, "sub", "i32", n, "1"), "31")
		cf = ir_cast(ir, "trunc", "i32", ir_op(ir, "lshr", "i32", a32, k), "i1")
	case .SAR:
		s := ir_sx(ir, ir_tr(ir, a32, bits), bits)
		r32 = ir_op(ir, "ashr", "i32", s, n)
		k := ir_op(ir, "and", "i32", ir_op(ir, "sub", "i32", n, "1"), "31")
		cf = ir_cast(ir, "trunc", "i32", ir_op(ir, "lshr", "i32", s, k), "i1")
	case .ROL, .ROR:
		// (8 / 16 bits: the count modulo the width, as the funnel shift takes it)
		if bits != 32 && !ir_feat(ir, "rot816") {return a32}
		an := ir_tr(ir, a32, bits)
		nn := ir_tr(ir, n, bits)
		dn := ir_v(ir)
		ir_w(ir, dn, " = call ", ty, " @llvm.", m == .ROL ? "fshl" : "fshr", ".", ty, "(", ty, " ", an, ", ", ty, " ", an, ", ", ty, " ", nn, ")")
		msb := ir_cmp(ir, "slt", ty, dn, "0")
		if m == .ROL {
			cf = ir_cast(ir, "trunc", ty, dn, "i1")
			of = ir_op(ir, "xor", "i1", msb, cf)
		} else {
			cf = msb
			of = ir_op(ir, "xor", "i1", msb, ir_cmp(ir, "slt", ty, ir_op(ir, "shl", ty, dn, "1"), "0"))
		}
		ir.s.f[IR_CF] = ir_sel(ir, "i1", z, old[IR_CF], cf)
		ir.s.f[IR_OF] = ir_sel(ir, "i1", z, old[IR_OF], of)
		return ir_zx(ir, dn, bits)
	}
	rn := ir_tr(ir, r32, bits)
	ir_zsp(ir, rn, bits)
	msb := ir_cmp(ir, "slt", ty, rn, "0")
	#partial switch m {
	case .SHL, .SAL: of = ir_op(ir, "xor", "i1", msb, cf)
	case .SHR: of = ir_cmp(ir, "slt", ty, ir_tr(ir, a32, bits), "0")
	case .SAR: of = "false"
	}
	nf := ir.s.f
	nf[IR_CF], nf[IR_OF] = cf, of
	for k in 0 ..< 5 {ir.s.f[k] = ir_sel(ir, "i1", z, old[k], nf[k])}
	return ir_zx(ir, rn, bits)
}

// ---- floating point: conversions as x86 does them ----

// round a double per mode (0 nearest-even, 1 down, 2 up, 3 toward zero; cm >= 0: that mode, else the i32 m)
ir_rint :: proc(ir: ^Ir, ty, sfx, v: string, cm: int, m: string = "") -> string {
	NAMES := [4]string{"roundeven", "floor", "ceil", "trunc"}
	if cm >= 0 {return ir_call1(ir, ty, strings.concatenate({"llvm.", NAMES[cm], ".", sfx}, context.temp_allocator), ty, v)}
	r: [4]string
	for k in 0 ..< 4 {r[k] = ir_call1(ir, ty, strings.concatenate({"llvm.", NAMES[k], ".", sfx}, context.temp_allocator), ty, v)}
	x := ir_sel(ir, ty, ir_cmp(ir, "eq", "i32", m, "1"), r[1], r[0])
	x = ir_sel(ir, ty, ir_cmp(ir, "eq", "i32", m, "2"), r[2], x)
	return ir_sel(ir, ty, ir_cmp(ir, "eq", "i32", m, "3"), r[3], x)
}
// an integral double to i16/i32/i64, out of range or NaN the "integer indefinite" (xrc_i16 / xrc_i32 / xrc_i64)
ir_f2i :: proc(ir: ^Ir, v: string, bits: int) -> string {
	lim := bits == 16 ? 32768.0 : bits == 32 ? 2147483648.0 : 9223372036854775808.0
	ok := ir_op(ir, "and", "i1", ir_fcmp(ir, "oge", "double", v, ir_dc(-lim)), ir_fcmp(ir, "olt", "double", v, ir_dc(lim)))
	ty := ir_ty(bits)
	i := ir_cast(ir, "fptosi", "double", v, ty)
	fr := ir_v(ir)
	ir_w(ir, fr, " = freeze ", ty, " ", i)
	ind := bits == 16 ? "-32768" : bits == 32 ? "-2147483648" : "-9223372036854775808"
	return ir_sel(ir, ty, ok, fr, ind)
}
// the SSE rounding mode (MXCSR.RC)
ir_sse_rc :: proc(ir: ^Ir) -> string {
	return ir_op(ir, "and", "i32", ir_op(ir, "lshr", "i32", ir_cload(ir, IR_OFF_MXCSR, "i32"), "13"), "3")
}

// ---- x87 ----

ir_st :: proc(ir: ^Ir, d, i: int) -> int {
	if i >= d || d > ir.nst {ir_fail(ir, "x87 underflow"); return 0}
	return d - 1 - i
}
// XR87: rounded to float when the control word selects 24-bit precision
ir_xr87 :: proc(ir: ^Ir, v: string) -> string {
	z := ir_cmp(ir, "eq", "i32", ir_op(ir, "and", "i32", ir.s.fcw, "768"), "0")
	f := ir_cast(ir, "fpext", "float", ir_cast(ir, "fptrunc", "double", v, "float"), "double")
	return ir_sel(ir, "double", z, f, v)
}
ir_x87_arith :: proc(ir: ^Ir, op: int, a, b: string) -> string {
	r := ""
	switch op {
	case 0: r = ir_op(ir, "fadd", "double", a, b)
	case 1: r = ir_op(ir, "fmul", "double", a, b)
	case 4: r = ir_op(ir, "fsub", "double", a, b)
	case 5: r = ir_op(ir, "fsub", "double", b, a)
	case 6: r = ir_op(ir, "fdiv", "double", a, b)
	case 7: r = ir_op(ir, "fdiv", "double", b, a)
	case: ir_fail(ir, "x87 op"); return a
	}
	return ir_xr87(ir, r)
}
// fcom: C0 C2 C3 of the status word (fsw_cmp)
ir_x87_cmp :: proc(ir: ^Ir, a, b: string) {
	un := ir_fcmp(ir, "uno", "double", a, b)
	gt := ir_fcmp(ir, "ogt", "double", a, b)
	lt := ir_fcmp(ir, "olt", "double", a, b)
	v := ir_sel(ir, "i32", lt, "256", "16384")
	v = ir_sel(ir, "i32", gt, "0", v)
	v = ir_sel(ir, "i32", un, "17664", v)
	ir.s.fsw = ir_op(ir, "or", "i32", ir_op(ir, "and", "i32", ir.s.fsw, "-17665"), v)
}
ir_x87_push :: proc(ir: ^Ir, d: int, v: string) -> int {
	if d >= ir.nst {ir_fail(ir, "x87 overflow"); return -1}
	ir.s.st[d] = v
	return d + 1
}
// the status word as fnstsw stores it (TOP = 8 - depth)
ir_x87_fsw :: proc(ir: ^Ir, d: int) -> string {
	return ir_op(ir, "or", "i32", ir_op(ir, "and", "i32", ir.s.fsw, "18176"), ir_num(i64(u32((8 - d) & 7) << 11)))
}
ir_fcw_rc :: proc(ir: ^Ir) -> string {return ir_op(ir, "and", "i32", ir_op(ir, "lshr", "i32", ir.s.fcw, "10"), "3")}
// fist / fistp / fisttp to memory (mode 3: fisttp; else the control word's rounding)
ir_x87_fist :: proc(ir: ^Ir, ea: string, v: string, bits: int, trunc: bool) {
	r := trunc ? ir_rint(ir, "double", "f64", v, 3) : ir_rint(ir, "double", "f64", v, -1, ir_fcw_rc(ir))
	ir_storet(ir, ea, ir_ty(bits), ir_f2i(ir, r, bits))
}

// one x87 instruction at stack depth d; the depth after it (-1: not handled)
ir_x87 :: proc(ir: ^Ir, ins: ^Rx_Ins, d: int, ea: string) -> int {
	o := ins.opc
	if o == 0x9b {return d}
	mod := ins.modrm >> 6
	reg := int((ins.modrm >> 3) & 7)
	i := int(ins.modrm & 7)
	st := &ir.s.st
	if mod != 3 {
		if ea == "" {ir_fail(ir, "x87 ea"); return -1}
		switch o {
		case 0xd8, 0xda, 0xdc, 0xde:
			src := ""
			switch o {
			case 0xd8: src = ir_cast(ir, "fpext", "float", ir_loadt(ir, ea, "float"), "double")
			case 0xda: src = ir_cast(ir, "sitofp", "i32", ir_loadt(ir, ea, "i32"), "double")
			case 0xdc: src = ir_loadt(ir, ea, "double")
			case 0xde: src = ir_cast(ir, "sitofp", "i16", ir_loadt(ir, ea, "i16"), "double")
			}
			s0 := ir_st(ir, d, 0)
			switch reg {
			case 2: ir_x87_cmp(ir, st[s0], src); return d
			case 3: ir_x87_cmp(ir, st[s0], src); return d - 1
			}
			st[s0] = ir_x87_arith(ir, reg, st[s0], src)
			return d
		case 0xd9:
			switch reg {
			case 0: return ir_x87_push(ir, d, ir_cast(ir, "fpext", "float", ir_loadt(ir, ea, "float"), "double"))
			case 2, 3:
				s0 := ir_st(ir, d, 0)
				ir_storet(ir, ea, "float", ir_cast(ir, "fptrunc", "double", st[s0], "float"))
				return reg == 3 ? d - 1 : d
			case 5:
				v := ir_cast(ir, "zext", "i16", ir_loadt(ir, ea, "i16"), "i32")
				ir.s.fcw = v
				ir_cstore(ir, IR_OFF_FCW, "i32", v)
				return d
			case 7: ir_storet(ir, ea, "i16", ir_cast(ir, "trunc", "i32", ir.s.fcw, "i16")); return d
			}
		case 0xdb:
			switch reg {
			case 0: return ir_x87_push(ir, d, ir_cast(ir, "sitofp", "i32", ir_loadt(ir, ea, "i32"), "double"))
			case 1: ir_x87_fist(ir, ea, st[ir_st(ir, d, 0)], 32, true); return d - 1
			case 2: ir_x87_fist(ir, ea, st[ir_st(ir, d, 0)], 32, false); return d
			case 3: ir_x87_fist(ir, ea, st[ir_st(ir, d, 0)], 32, false); return d - 1
			}
		case 0xdd:
			switch reg {
			case 0: return ir_x87_push(ir, d, ir_loadt(ir, ea, "double"))
			case 1: ir_x87_fist(ir, ea, st[ir_st(ir, d, 0)], 64, true); return d - 1
			case 2: ir_storet(ir, ea, "double", st[ir_st(ir, d, 0)]); return d
			case 3: ir_storet(ir, ea, "double", st[ir_st(ir, d, 0)]); return d - 1
			case 7: ir_storet(ir, ea, "i16", ir_cast(ir, "trunc", "i32", ir_x87_fsw(ir, d), "i16")); return d
			}
		case 0xdf:
			switch reg {
			case 0: return ir_x87_push(ir, d, ir_cast(ir, "sitofp", "i16", ir_loadt(ir, ea, "i16"), "double"))
			case 1: ir_x87_fist(ir, ea, st[ir_st(ir, d, 0)], 16, true); return d - 1
			case 2: ir_x87_fist(ir, ea, st[ir_st(ir, d, 0)], 16, false); return d
			case 3: ir_x87_fist(ir, ea, st[ir_st(ir, d, 0)], 16, false); return d - 1
			case 5: return ir_x87_push(ir, d, ir_cast(ir, "sitofp", "i64", ir_loadt(ir, ea, "i64"), "double"))
			case 7: ir_x87_fist(ir, ea, st[ir_st(ir, d, 0)], 64, false); return d - 1
			}
		}
		ir_fail(ir, rx_mn(ins))
		return -1
	}
	// register forms
	switch o {
	case 0xd8:
		s0, si := ir_st(ir, d, 0), ir_st(ir, d, i)
		switch reg {
		case 2: ir_x87_cmp(ir, st[s0], st[si]); return d
		case 3: ir_x87_cmp(ir, st[s0], st[si]); return d - 1
		}
		st[s0] = ir_x87_arith(ir, reg, st[s0], st[si])
		return d
	case 0xd9:
		switch ins.modrm {
		case 0xc0 ..= 0xc7: return ir_x87_push(ir, d, st[ir_st(ir, d, i)])
		case 0xc8 ..= 0xcf:
			s0, si := ir_st(ir, d, 0), ir_st(ir, d, i)
			st[s0], st[si] = st[si], st[s0]
			return d
		case 0xd0: return d
		case 0xe0:
			s0 := ir_st(ir, d, 0)
			v := ir_v(ir)
			ir_w(ir, v, " = fneg double ", st[s0])
			st[s0] = v
			return d
		case 0xe1: s0 := ir_st(ir, d, 0); st[s0] = ir_call1(ir, "double", "llvm.fabs.f64", "double", st[s0]); return d
		case 0xe4: ir_x87_cmp(ir, st[ir_st(ir, d, 0)], "0.0"); return d
		case 0xe8: return ir_x87_push(ir, d, "1.0")
		case 0xe9: return ir_x87_push(ir, d, ir_dc(3.3219280948873622))
		case 0xea: return ir_x87_push(ir, d, ir_dc(1.4426950408889634))
		case 0xeb: return ir_x87_push(ir, d, ir_dc(3.1415926535897931))
		case 0xec: return ir_x87_push(ir, d, ir_dc(0.30102999566398120))
		case 0xed: return ir_x87_push(ir, d, ir_dc(0.69314718055994531))
		case 0xee: return ir_x87_push(ir, d, "0.0")
		case 0xfa:
			s0 := ir_st(ir, d, 0)
			st[s0] = ir_xr87(ir, ir_call1(ir, "double", "llvm.sqrt.f64", "double", st[s0]))
			return d
		case 0xf0, 0xf1, 0xf2, 0xf3, 0xf5, 0xf8, 0xfb, 0xfd, 0xfe, 0xff:
			if !ir_feat(ir, "x87t") {return -1}
		}
		switch ins.modrm {
		case 0xf0: // f2xm1
			s0 := ir_st(ir, d, 0)
			st[s0] = ir_xr87(ir, ir_op(ir, "fsub", "double", ir_call1(ir, "double", "exp2", "double", st[s0]), "1.0"))
			return d
		case 0xf1: // fyl2x: st1 = st1 * log2(st0)
			s0, s1 := ir_st(ir, d, 0), ir_st(ir, d, 1)
			st[s1] = ir_xr87(ir, ir_op(ir, "fmul", "double", st[s1], ir_call1(ir, "double", "log2", "double", st[s0])))
			return d - 1
		case 0xf2: // fptan: st0 = tan, push 1
			s0 := ir_st(ir, d, 0)
			st[s0] = ir_xr87(ir, ir_call1(ir, "double", "tan", "double", st[s0]))
			return ir_x87_push(ir, d, "1.0")
		case 0xf3: // fpatan: st1 = atan2(st1, st0)
			s0, s1 := ir_st(ir, d, 0), ir_st(ir, d, 1)
			st[s1] = ir_xr87(ir, ir_call2(ir, "double", "atan2", "double", st[s1], st[s0]))
			return d - 1
		case 0xf5, 0xf8: // fprem1 / fprem
			s0, s1 := ir_st(ir, d, 0), ir_st(ir, d, 1)
			st[s0] = ir_call2(ir, "double", ins.modrm == 0xf5 ? "remainder" : "fmod", "double", st[s0], st[s1])
			ir.s.fsw = ir_op(ir, "and", "i32", ir.s.fsw, "-1025")
			return d
		case 0xfb: // fsincos
			s0 := ir_st(ir, d, 0)
			x := st[s0]
			st[s0] = ir_xr87(ir, ir_call1(ir, "double", "sin", "double", x))
			return ir_x87_push(ir, d, ir_xr87(ir, ir_call1(ir, "double", "cos", "double", x)))
		case 0xfd: // fscale: st0 * 2^trunc(st1)
			s0, s1 := ir_st(ir, d, 0), ir_st(ir, d, 1)
			k := ir_cast(ir, "fptosi", "double", ir_rint(ir, "double", "f64", st[s1], 3), "i32")
			kf := ir_v(ir)
			ir_w(ir, kf, " = freeze i32 ", k)
			r := ir_v(ir)
			ir_w(ir, r, " = call double @ldexp(double ", st[s0], ", i32 ", kf, ")")
			st[s0] = r
			return d
		case 0xfe: s0 := ir_st(ir, d, 0); st[s0] = ir_xr87(ir, ir_call1(ir, "double", "sin", "double", st[s0])); return d
		case 0xff: s0 := ir_st(ir, d, 0); st[s0] = ir_xr87(ir, ir_call1(ir, "double", "cos", "double", st[s0])); return d
		case 0xfc:
			s0 := ir_st(ir, d, 0)
			st[s0] = ir_rint(ir, "double", "f64", st[s0], -1, ir_fcw_rc(ir))
			return d
		}
	case 0xda:
		if ins.modrm == 0xe9 { // fucompp
			ir_x87_cmp(ir, st[ir_st(ir, d, 0)], st[ir_st(ir, d, 1)])
			return d - 2
		}
		if reg <= 3 { // fcmovb / fcmove / fcmovbe / fcmovu
			tb := [4]int{2, 4, 6, 10}
			s0, si := ir_st(ir, d, 0), ir_st(ir, d, i)
			st[s0] = ir_sel(ir, "double", ir_cond(ir, tb[reg]), st[si], st[s0])
			return d
		}
	case 0xdb:
		if reg <= 3 { // fcmovnb / fcmovne / fcmovnbe / fcmovnu
			tb := [4]int{3, 5, 7, 11}
			s0, si := ir_st(ir, d, 0), ir_st(ir, d, i)
			st[s0] = ir_sel(ir, "double", ir_cond(ir, tb[reg]), st[si], st[s0])
			return d
		}
		if ins.modrm == 0xe2 {return d} // fnclex
		if reg == 5 || reg == 6 { // fucomi / fcomi
			ir_fcomi(ir, "double", st[ir_st(ir, d, 0)], st[ir_st(ir, d, i)])
			return d
		}
	case 0xdc, 0xde:
		if o == 0xde && ins.modrm == 0xd9 { // fcompp
			ir_x87_cmp(ir, st[ir_st(ir, d, 0)], st[ir_st(ir, d, 1)])
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
		s0, si := ir_st(ir, d, 0), ir_st(ir, d, i)
		st[si] = ir_x87_arith(ir, op, st[si], st[s0])
		return o == 0xde ? d - 1 : d
	case 0xdd:
		switch reg {
		case 0: return d // ffree
		case 2: s0, si := ir_st(ir, d, 0), ir_st(ir, d, i); st[si] = st[s0]; return d
		case 3:
			s0, si := ir_st(ir, d, 0), ir_st(ir, d, i)
			if i != 0 {st[si] = st[s0]}
			return d - 1
		case 4: ir_x87_cmp(ir, st[ir_st(ir, d, 0)], st[ir_st(ir, d, i)]); return d
		case 5: ir_x87_cmp(ir, st[ir_st(ir, d, 0)], st[ir_st(ir, d, i)]); return d - 1
		}
	case 0xdf:
		if ins.modrm == 0xe0 { // fnstsw ax
			ir_reg_write(ir, x86.AX, ir_x87_fsw(ir, d))
			return d
		}
		if reg == 5 || reg == 6 { // fucomip / fcomip
			ir_fcomi(ir, "double", st[ir_st(ir, d, 0)], st[ir_st(ir, d, i)])
			return d - 1
		}
	}
	ir_fail(ir, strings.concatenate({"x87 ", rx_mn(ins)}, context.temp_allocator))
	return -1
}

// ---- SSE ----

ir_is_x :: proc(op: x86.Operand) -> bool {return op.kind == .REGISTER && x86.reg_class(op.reg) == x86.REG_XMM}
ir_is_mm :: proc(op: x86.Operand) -> bool {return op.kind == .REGISTER && x86.reg_class(op.reg) == x86.REG_MM}
ir_xn :: proc(op: x86.Operand) -> int {return int(x86.reg_hw(op.reg)) & 7}
ir_xget :: proc(ir: ^Ir, k: int, ty: string) -> string {return ir_cast(ir, "bitcast", V4I, ir.s.x[k], ty)}
ir_xput :: proc(ir: ^Ir, k: int, v, ty: string) {ir.s.x[k] = ir_cast(ir, "bitcast", ty, v, V4I)}
// the source operand as 128 bits (<4 x i32>), reading only `bytes` of a memory operand (the rest zero)
ir_xsrc :: proc(ir: ^Ir, op: x86.Operand, ea: string, bytes: int) -> string {
	if ir_is_x(op) {return ir.s.x[ir_xn(op)]}
	if op.kind != .MEMORY {ir_fail(ir, "sse src"); return "zeroinitializer"}
	switch bytes {
	case 4: return ir_ins_el(ir, V4I, "zeroinitializer", "i32", ir_loadt(ir, ea, "i32"), 0)
	case 8: return ir_cast(ir, "bitcast", V2Q, ir_ins_el(ir, V2Q, "zeroinitializer", "i64", ir_loadt(ir, ea, "i64"), 0), V4I)
	}
	return ir_loadt(ir, ea, V4I)
}
// a scalar float (lane 0 of an xmm register, or memory)
ir_xscalar :: proc(ir: ^Ir, op: x86.Operand, ea: string, dbl: bool) -> string {
	ety := dbl ? "double" : "float"
	if ir_is_x(op) {return ir_ext(ir, dbl ? V2D : V4F, ir_xget(ir, ir_xn(op), dbl ? V2D : V4F), 0)}
	if op.kind == .MEMORY {return ir_loadt(ir, ea, ety)}
	ir_fail(ir, "sse scalar")
	return "0.0"
}
// a float binary op (op: an LLVM instruction, or min / max with x86's rule: the second operand unless a <op> b)
ir_fbin :: proc(ir: ^Ir, op, ty, a, b: string) -> string {
	ct := ir_cmpty(ty)
	switch op {
	case "min": return ir_selv(ir, ct, ir_fcmp(ir, "olt", ty, a, b), ty, a, b)
	case "max": return ir_selv(ir, ct, ir_fcmp(ir, "ogt", ty, a, b), ty, a, b)
	case "sqrt": return ir_call1(ir, ty, strings.concatenate({"llvm.sqrt.", ir_sfx(ty)}, context.temp_allocator), ty, b)
	case "rcp": return ir_op(ir, "fdiv", ty, ir_one(ty), b)
	case "rsqrt": return ir_op(ir, "fdiv", ty, ir_one(ty), ir_call1(ir, ty, strings.concatenate({"llvm.sqrt.", ir_sfx(ty)}, context.temp_allocator), ty, b))
	}
	return ir_op(ir, op, ty, a, b)
}
ir_cmpty :: proc(ty: string) -> string {
	switch ty {
	case V4F, V4I: return "<4 x i1>"
	case V2D, V2Q: return "<2 x i1>"
	case V8W: return "<8 x i1>"
	case V16B: return "<16 x i1>"
	}
	return "i1"
}
ir_sfx :: proc(ty: string) -> string {
	switch ty {
	case V4F: return "v4f32"
	case V2D: return "v2f64"
	case "float": return "f32"
	case V8W: return "v8i16"
	}
	return "f64"
}
ir_one :: proc(ty: string) -> string {
	switch ty {
	case V4F: return "splat (float 1.0)"
	case V2D: return "splat (double 1.0)"
	}
	return "1.0"
}
// packed: X[d] = X[d] <op> src (ty: the lane type)
ir_xpacked :: proc(ir: ^Ir, ops: []x86.Operand, ea, ty, op: string) -> bool {
	if len(ops) < 2 || !ir_is_x(ops[0]) {return false}
	d := ir_xn(ops[0])
	b := ir_cast(ir, "bitcast", V4I, ir_xsrc(ir, ops[1], ea, 16), ty)
	a := ir_xget(ir, d, ty)
	r := ""
	switch op {
	case "fadd", "fsub", "fmul", "fdiv", "min", "max", "sqrt", "rcp", "rsqrt": r = ir_fbin(ir, op, ty, a, b)
	case "andn": r = ir_op(ir, "and", ty, ir_op(ir, "xor", ty, a, "splat (i32 -1)"), b)
	case "uadd.sat", "usub.sat", "smax", "smin":
		r = ir_call2(ir, ty, strings.concatenate({"llvm.", op, ".", ir_sfx(ty)}, context.temp_allocator), ty, a, b)
	case "eq", "sgt":
		r = ir_cast(ir, "sext", ir_cmpty(ty), ir_cmp(ir, op, ty, a, b), ty)
	case: r = ir_op(ir, op, ty, a, b)
	}
	ir_xput(ir, d, r, ty)
	return true
}
// scalar (lane 0): X[d].lane0 = X[d].lane0 <op> src
ir_xscal :: proc(ir: ^Ir, ops: []x86.Operand, ea: string, dbl: bool, op: string) -> bool {
	if len(ops) < 2 || !ir_is_x(ops[0]) {return false}
	d := ir_xn(ops[0])
	vt := dbl ? V2D : V4F
	ety := dbl ? "double" : "float"
	b := ir_xscalar(ir, ops[1], ea, dbl)
	v := ir_xget(ir, d, vt)
	a := ir_ext(ir, vt, v, 0)
	ir_xput(ir, d, ir_ins_el(ir, vt, v, ety, ir_fbin(ir, op, ety, a, b), 0), vt)
	return true
}
CMP_PRED := [8]string{"oeq", "olt", "ole", "uno", "une", "uge", "ugt", "ord"}

// one SSE instruction (false: not one; it fails the function itself when it is one it cannot do)
ir_sse :: proc(ir: ^Ir, ins: ^Rx_Ins, ea: string) -> bool {
	m := ins.inst.mnemonic
	n := int(ins.inst.operand_count)
	ops := ins.inst.ops[:n]
	for o in ops {if ir_is_mm(o) {ir_fail(ir, "mmx"); return true}}
	imm: i64 = 0
	if n >= 3 && ops[2].kind == .IMMEDIATE {imm = ops[2].immediate}
	sse2_sd := ins.opc == 0x0f
	X := len(ops) > 0 && ir_is_x(ops[0])
	d := X ? ir_xn(ops[0]) : 0
	#partial switch m {
	case .PREFETCHT0, .PREFETCHT1, .PREFETCHT2, .PREFETCHNTA, .SFENCE, .LFENCE, .MFENCE: return true
	// ---- moves
	case .MOVSS, .MOVSD:
		if m == .MOVSD && !sse2_sd {return false}
		dbl := m == .MOVSD
		if X && ir_is_x(ops[1]) {
			if dbl {
				ir_xput(ir, d, ir_ins_el(ir, V2Q, ir_xget(ir, d, V2Q), "i64", ir_ext(ir, V2Q, ir_xget(ir, ir_xn(ops[1]), V2Q), 0), 0), V2Q)
			} else {
				ir.s.x[d] = ir_ins_el(ir, V4I, ir.s.x[d], "i32", ir_ext(ir, V4I, ir.s.x[ir_xn(ops[1])], 0), 0)
			}
		} else if X && ops[1].kind == .MEMORY {
			ir.s.x[d] = ir_xsrc(ir, ops[1], ea, dbl ? 8 : 4)
		} else if ops[0].kind == .MEMORY && ir_is_x(ops[1]) {
			s := ir_xn(ops[1])
			if dbl {ir_storet(ir, ea, "i64", ir_ext(ir, V2Q, ir_xget(ir, s, V2Q), 0))} else {ir_storet(ir, ea, "i32", ir_ext(ir, V4I, ir.s.x[s], 0))}
		} else {ir_fail(ir, "movss"); }
		return true
	case .MOVAPS, .MOVUPS, .MOVAPD, .MOVUPD, .MOVDQA, .MOVDQU, .MOVNTPS, .MOVNTPD, .MOVNTDQ, .LDDQU:
		if X && ir_is_x(ops[1]) {
			ir.s.x[d] = ir.s.x[ir_xn(ops[1])]
		} else if X && ops[1].kind == .MEMORY {
			ir.s.x[d] = ir_loadt(ir, ea, V4I)
		} else if ops[0].kind == .MEMORY && ir_is_x(ops[1]) {
			ir_storet(ir, ea, V4I, ir.s.x[ir_xn(ops[1])])
		} else {ir_fail(ir, "movaps")}
		return true
	case .MOVLPS, .MOVLPD, .MOVHPS, .MOVHPD:
		k := (m == .MOVHPS || m == .MOVHPD) ? 1 : 0
		if X && ops[1].kind == .MEMORY {
			ir_xput(ir, d, ir_ins_el(ir, V2Q, ir_xget(ir, d, V2Q), "i64", ir_loadt(ir, ea, "i64"), k), V2Q)
		} else if ops[0].kind == .MEMORY && ir_is_x(ops[1]) {
			ir_storet(ir, ea, "i64", ir_ext(ir, V2Q, ir_xget(ir, ir_xn(ops[1]), V2Q), k))
		} else {ir_fail(ir, "movlps")}
		return true
	case .MOVHLPS, .MOVLHPS:
		a := ir_xget(ir, d, V2Q)
		b := ir_xget(ir, ir_xn(ops[1]), V2Q)
		mk := m == .MOVHLPS ? [2]int{3, 1} : [2]int{0, 2}
		ir_xput(ir, d, ir_shuf(ir, V2Q, a, b, mk[:], V2Q), V2Q)
		return true
	case .MOVD:
		if X {
			v := ""
			if ops[1].kind == .MEMORY {v = ir_loadt(ir, ea, "i32")} else {v = ir_read(ir, ops[1], ea)}
			ir.s.x[d] = ir_ins_el(ir, V4I, "zeroinitializer", "i32", v, 0)
		} else if ir_is_x(ops[1]) {
			v := ir_ext(ir, V4I, ir.s.x[ir_xn(ops[1])], 0)
			if ops[0].kind == .MEMORY {ir_storet(ir, ea, "i32", v)} else {ir_write(ir, ops[0], ea, v)}
		} else {ir_fail(ir, "movd")}
		return true
	case .MOVQ:
		if X && ir_is_x(ops[1]) {
			ir_xput(ir, d, ir_ins_el(ir, V2Q, "zeroinitializer", "i64", ir_ext(ir, V2Q, ir_xget(ir, ir_xn(ops[1]), V2Q), 0), 0), V2Q)
		} else if X && ops[1].kind == .MEMORY {
			ir.s.x[d] = ir_xsrc(ir, ops[1], ea, 8)
		} else if ops[0].kind == .MEMORY && ir_is_x(ops[1]) {
			ir_storet(ir, ea, "i64", ir_ext(ir, V2Q, ir_xget(ir, ir_xn(ops[1]), V2Q), 0))
		} else {ir_fail(ir, "movq")}
		return true
	case .MOVMSKPS, .MOVMSKPD, .PMOVMSKB:
		if !ir_is_x(ops[1]) || ops[0].kind != .REGISTER {ir_fail(ir, "movmsk"); return true}
		ty, it := V4I, "i4"
		#partial switch m {
		case .MOVMSKPD: ty, it = V2Q, "i2"
		case .PMOVMSKB: ty, it = V16B, "i16"
		}
		neg := ir_cmp(ir, "slt", ty, ir_xget(ir, ir_xn(ops[1]), ty), "zeroinitializer")
		ir_write(ir, ops[0], "", ir_cast(ir, "zext", it, ir_cast(ir, "bitcast", ir_cmpty(ty), neg, it), "i32"))
		return true

	// ---- float arithmetic
	case .ADDPS: return ir_xpacked(ir, ops, ea, V4F, "fadd") || ir_bad(ir, m)
	case .SUBPS: return ir_xpacked(ir, ops, ea, V4F, "fsub") || ir_bad(ir, m)
	case .MULPS: return ir_xpacked(ir, ops, ea, V4F, "fmul") || ir_bad(ir, m)
	case .DIVPS: return ir_xpacked(ir, ops, ea, V4F, "fdiv") || ir_bad(ir, m)
	case .MINPS: return ir_xpacked(ir, ops, ea, V4F, "min") || ir_bad(ir, m)
	case .MAXPS: return ir_xpacked(ir, ops, ea, V4F, "max") || ir_bad(ir, m)
	case .SQRTPS: return ir_xpacked(ir, ops, ea, V4F, "sqrt") || ir_bad(ir, m)
	case .RCPPS: return ir_xpacked(ir, ops, ea, V4F, "rcp") || ir_bad(ir, m)
	case .RSQRTPS: return ir_xpacked(ir, ops, ea, V4F, "rsqrt") || ir_bad(ir, m)
	case .ADDPD: return ir_xpacked(ir, ops, ea, V2D, "fadd") || ir_bad(ir, m)
	case .SUBPD: return ir_xpacked(ir, ops, ea, V2D, "fsub") || ir_bad(ir, m)
	case .MULPD: return ir_xpacked(ir, ops, ea, V2D, "fmul") || ir_bad(ir, m)
	case .DIVPD: return ir_xpacked(ir, ops, ea, V2D, "fdiv") || ir_bad(ir, m)
	case .MINPD: return ir_xpacked(ir, ops, ea, V2D, "min") || ir_bad(ir, m)
	case .MAXPD: return ir_xpacked(ir, ops, ea, V2D, "max") || ir_bad(ir, m)
	case .SQRTPD: return ir_xpacked(ir, ops, ea, V2D, "sqrt") || ir_bad(ir, m)
	case .ADDSS: return ir_xscal(ir, ops, ea, false, "fadd") || ir_bad(ir, m)
	case .SUBSS: return ir_xscal(ir, ops, ea, false, "fsub") || ir_bad(ir, m)
	case .MULSS: return ir_xscal(ir, ops, ea, false, "fmul") || ir_bad(ir, m)
	case .DIVSS: return ir_xscal(ir, ops, ea, false, "fdiv") || ir_bad(ir, m)
	case .MINSS: return ir_xscal(ir, ops, ea, false, "min") || ir_bad(ir, m)
	case .MAXSS: return ir_xscal(ir, ops, ea, false, "max") || ir_bad(ir, m)
	case .SQRTSS: return ir_xscal(ir, ops, ea, false, "sqrt") || ir_bad(ir, m)
	case .RCPSS: return ir_xscal(ir, ops, ea, false, "rcp") || ir_bad(ir, m)
	case .RSQRTSS: return ir_xscal(ir, ops, ea, false, "rsqrt") || ir_bad(ir, m)
	case .ADDSD: return ir_xscal(ir, ops, ea, true, "fadd") || ir_bad(ir, m)
	case .SUBSD: return ir_xscal(ir, ops, ea, true, "fsub") || ir_bad(ir, m)
	case .MULSD: return ir_xscal(ir, ops, ea, true, "fmul") || ir_bad(ir, m)
	case .DIVSD: return ir_xscal(ir, ops, ea, true, "fdiv") || ir_bad(ir, m)
	case .MINSD: return ir_xscal(ir, ops, ea, true, "min") || ir_bad(ir, m)
	case .MAXSD: return ir_xscal(ir, ops, ea, true, "max") || ir_bad(ir, m)
	case .SQRTSD: return ir_xscal(ir, ops, ea, true, "sqrt") || ir_bad(ir, m)

	// ---- bitwise
	case .ANDPS, .ANDPD, .PAND: return ir_xpacked(ir, ops, ea, V4I, "and") || ir_bad(ir, m)
	case .ANDNPS, .ANDNPD, .PANDN: return ir_xpacked(ir, ops, ea, V4I, "andn") || ir_bad(ir, m)
	case .ORPS, .ORPD, .POR: return ir_xpacked(ir, ops, ea, V4I, "or") || ir_bad(ir, m)
	case .XORPS, .XORPD, .PXOR:
		// xorps x, x: zero (an idiom)
		if X && ir_is_x(ops[1]) && ir_xn(ops[1]) == d {ir.s.x[d] = "zeroinitializer"; return true}
		return ir_xpacked(ir, ops, ea, V4I, "xor") || ir_bad(ir, m)

	// ---- compares
	case .CMPPS, .CMPPD:
		if !X {return ir_bad(ir, m)}
		ty := m == .CMPPS ? V4F : V2D
		it := m == .CMPPS ? V4I : V2Q
		a := ir_xget(ir, d, ty)
		b := ir_cast(ir, "bitcast", V4I, ir_xsrc(ir, ops[1], ea, 16), ty)
		c := ir_fcmp(ir, CMP_PRED[imm & 7], ty, a, b)
		ir_xput(ir, d, ir_cast(ir, "sext", ir_cmpty(ty), c, it), it)
		return true
	case .CMPSS, .CMPSD:
		if m == .CMPSD && !sse2_sd {return false}
		if !X {return ir_bad(ir, m)}
		dbl := m == .CMPSD
		vt := dbl ? V2D : V4F
		it := dbl ? V2Q : V4I
		ety := dbl ? "double" : "float"
		ity := dbl ? "i64" : "i32"
		b := ir_xscalar(ir, ops[1], ea, dbl)
		a := ir_ext(ir, vt, ir_xget(ir, d, vt), 0)
		c := ir_fcmp(ir, CMP_PRED[imm & 7], ety, a, b)
		ir_xput(ir, d, ir_ins_el(ir, it, ir_xget(ir, d, it), ity, ir_cast(ir, "sext", "i1", c, ity), 0), it)
		return true
	case .COMISS, .UCOMISS, .COMISD, .UCOMISD:
		if !X {return ir_bad(ir, m)}
		dbl := m == .COMISD || m == .UCOMISD
		vt := dbl ? V2D : V4F
		b := ir_xscalar(ir, ops[1], ea, dbl)
		a := ir_ext(ir, vt, ir_xget(ir, d, vt), 0)
		ir_fcomi(ir, dbl ? "double" : "float", a, b)
		return true

	// ---- conversions
	case .CVTSI2SS, .CVTSI2SD:
		if !X {return ir_bad(ir, m)}
		v := ""
		if ops[1].kind == .MEMORY {v = ir_loadt(ir, ea, "i32")} else {v = ir_read(ir, ops[1], ea)}
		if m == .CVTSI2SS {
			ir_xput(ir, d, ir_ins_el(ir, V4F, ir_xget(ir, d, V4F), "float", ir_cast(ir, "sitofp", "i32", v, "float"), 0), V4F)
		} else {
			ir_xput(ir, d, ir_ins_el(ir, V2D, ir_xget(ir, d, V2D), "double", ir_cast(ir, "sitofp", "i32", v, "double"), 0), V2D)
		}
		return true
	case .CVTTSS2SI, .CVTSS2SI, .CVTTSD2SI, .CVTSD2SI:
		if ops[0].kind != .REGISTER {return ir_bad(ir, m)}
		dbl := m == .CVTTSD2SI || m == .CVTSD2SI
		b := ir_xscalar(ir, ops[1], ea, dbl)
		if !dbl {b = ir_cast(ir, "fpext", "float", b, "double")}
		trunc := m == .CVTTSS2SI || m == .CVTTSD2SI
		r := trunc ? ir_rint(ir, "double", "f64", b, 3) : ir_rint(ir, "double", "f64", b, -1, ir_sse_rc(ir))
		ir_write(ir, ops[0], "", ir_f2i(ir, r, 32))
		return true
	case .CVTSS2SD:
		if !X {return ir_bad(ir, m)}
		b := ir_cast(ir, "fpext", "float", ir_xscalar(ir, ops[1], ea, false), "double")
		ir_xput(ir, d, ir_ins_el(ir, V2D, ir_xget(ir, d, V2D), "double", b, 0), V2D)
		return true
	case .CVTSD2SS:
		if !X {return ir_bad(ir, m)}
		b := ir_cast(ir, "fptrunc", "double", ir_xscalar(ir, ops[1], ea, true), "float")
		ir_xput(ir, d, ir_ins_el(ir, V4F, ir_xget(ir, d, V4F), "float", b, 0), V4F)
		return true
	case .CVTPS2PD:
		if !X {return ir_bad(ir, m)}
		s := ir_cast(ir, "bitcast", V4I, ir_xsrc(ir, ops[1], ea, 8), V4F)
		lo := ir_shuf(ir, V4F, s, s, {0, 1}, "<2 x float>")
		ir_xput(ir, d, ir_cast(ir, "fpext", "<2 x float>", lo, V2D), V2D)
		return true
	case .CVTPD2PS:
		if !X {return ir_bad(ir, m)}
		s := ir_cast(ir, "bitcast", V4I, ir_xsrc(ir, ops[1], ea, 16), V2D)
		f := ir_cast(ir, "fptrunc", V2D, s, "<2 x float>")
		ir_xput(ir, d, ir_shuf(ir, "<2 x float>", f, "zeroinitializer", {0, 1, 2, 3}, V4F), V4F)
		return true
	case .CVTDQ2PS:
		if !X {return ir_bad(ir, m)}
		ir_xput(ir, d, ir_cast(ir, "sitofp", V4I, ir_xsrc(ir, ops[1], ea, 16), V4F), V4F)
		return true
	case .CVTDQ2PD:
		if !X {return ir_bad(ir, m)}
		s := ir_xsrc(ir, ops[1], ea, 8)
		lo := ir_shuf(ir, V4I, s, s, {0, 1}, "<2 x i32>")
		ir_xput(ir, d, ir_cast(ir, "sitofp", "<2 x i32>", lo, V2D), V2D)
		return true
	case .CVTTPS2DQ, .CVTPS2DQ:
		if !X {return ir_bad(ir, m)}
		s := ir_cast(ir, "bitcast", V4I, ir_xsrc(ir, ops[1], ea, 16), V4F)
		s = ir_cast(ir, "fpext", V4F, s, "<4 x double>")
		r := m == .CVTTPS2DQ ? ir_rint(ir, "<4 x double>", "v4f64", s, 3) : ir_rint(ir, "<4 x double>", "v4f64", s, -1, ir_sse_rc(ir))
		ir.s.x[d] = ir_vf2i(ir, r, 4)
		return true
	case .CVTTPD2DQ, .CVTPD2DQ:
		if !X {return ir_bad(ir, m)}
		s := ir_cast(ir, "bitcast", V4I, ir_xsrc(ir, ops[1], ea, 16), V2D)
		r := m == .CVTTPD2DQ ? ir_rint(ir, V2D, "v2f64", s, 3) : ir_rint(ir, V2D, "v2f64", s, -1, ir_sse_rc(ir))
		v := ir_vf2i(ir, r, 2)
		ir.s.x[d] = ir_shuf(ir, "<2 x i32>", v, "zeroinitializer", {0, 1, 2, 3}, V4I)
		return true

	// ---- shuffles / unpacks
	case .SHUFPS:
		if !X {return ir_bad(ir, m)}
		k := int(imm)
		a := ir.s.x[d]
		b := ir_xsrc(ir, ops[1], ea, 16)
		ir.s.x[d] = ir_shuf(ir, V4I, a, b, {k & 3, (k >> 2) & 3, 4 + ((k >> 4) & 3), 4 + ((k >> 6) & 3)}, V4I)
		return true
	case .SHUFPD:
		if !X {return ir_bad(ir, m)}
		k := int(imm)
		a := ir_xget(ir, d, V2Q)
		b := ir_cast(ir, "bitcast", V4I, ir_xsrc(ir, ops[1], ea, 16), V2Q)
		ir_xput(ir, d, ir_shuf(ir, V2Q, a, b, {k & 1, 2 + ((k >> 1) & 1)}, V2Q), V2Q)
		return true
	case .PSHUFD:
		if !X {return ir_bad(ir, m)}
		k := int(imm)
		s := ir_xsrc(ir, ops[1], ea, 16)
		ir.s.x[d] = ir_shuf(ir, V4I, s, s, {k & 3, (k >> 2) & 3, (k >> 4) & 3, (k >> 6) & 3}, V4I)
		return true
	case .PSHUFLW, .PSHUFHW:
		if !X {return ir_bad(ir, m)}
		k := int(imm)
		s := ir_cast(ir, "bitcast", V4I, ir_xsrc(ir, ops[1], ea, 16), V8W)
		mk: [8]int
		for i in 0 ..< 8 {mk[i] = i}
		b := m == .PSHUFHW ? 4 : 0
		for i in 0 ..< 4 {mk[b + i] = b + ((k >> (2 * uint(i))) & 3)}
		ir_xput(ir, d, ir_shuf(ir, V8W, s, s, mk[:], V8W), V8W)
		return true
	case .UNPCKLPS, .PUNPCKLDQ, .UNPCKHPS, .PUNPCKHDQ:
		if !X {return ir_bad(ir, m)}
		b := ir_xsrc(ir, ops[1], ea, 16)
		hi := m == .UNPCKHPS || m == .PUNPCKHDQ
		mk := hi ? [4]int{2, 6, 3, 7} : [4]int{0, 4, 1, 5}
		ir.s.x[d] = ir_shuf(ir, V4I, ir.s.x[d], b, mk[:], V4I)
		return true
	case .UNPCKLPD, .PUNPCKLQDQ, .UNPCKHPD, .PUNPCKHQDQ:
		if !X {return ir_bad(ir, m)}
		b := ir_cast(ir, "bitcast", V4I, ir_xsrc(ir, ops[1], ea, 16), V2Q)
		hi := m == .UNPCKHPD || m == .PUNPCKHQDQ
		mk := hi ? [2]int{1, 3} : [2]int{0, 2}
		ir_xput(ir, d, ir_shuf(ir, V2Q, ir_xget(ir, d, V2Q), b, mk[:], V2Q), V2Q)
		return true
	case .PUNPCKLWD, .PUNPCKHWD:
		if !X {return ir_bad(ir, m)}
		b := ir_cast(ir, "bitcast", V4I, ir_xsrc(ir, ops[1], ea, 16), V8W)
		o := m == .PUNPCKHWD ? 4 : 0
		mk: [8]int
		for i in 0 ..< 4 {mk[2 * i], mk[2 * i + 1] = o + i, 8 + o + i}
		ir_xput(ir, d, ir_shuf(ir, V8W, ir_xget(ir, d, V8W), b, mk[:], V8W), V8W)
		return true
	case .PUNPCKLBW, .PUNPCKHBW:
		if !X {return ir_bad(ir, m)}
		b := ir_cast(ir, "bitcast", V4I, ir_xsrc(ir, ops[1], ea, 16), V16B)
		o := m == .PUNPCKHBW ? 8 : 0
		mk: [16]int
		for i in 0 ..< 8 {mk[2 * i], mk[2 * i + 1] = o + i, 16 + o + i}
		ir_xput(ir, d, ir_shuf(ir, V16B, ir_xget(ir, d, V16B), b, mk[:], V16B), V16B)
		return true

	// ---- integer lanes
	case .PADDD: return ir_xpacked(ir, ops, ea, V4I, "add") || ir_bad(ir, m)
	case .PSUBD: return ir_xpacked(ir, ops, ea, V4I, "sub") || ir_bad(ir, m)
	case .PADDQ: return ir_xpacked(ir, ops, ea, V2Q, "add") || ir_bad(ir, m)
	case .PSUBQ: return ir_xpacked(ir, ops, ea, V2Q, "sub") || ir_bad(ir, m)
	case .PADDW: return ir_xpacked(ir, ops, ea, V8W, "add") || ir_bad(ir, m)
	case .PSUBW: return ir_xpacked(ir, ops, ea, V8W, "sub") || ir_bad(ir, m)
	case .PADDB: return ir_xpacked(ir, ops, ea, V16B, "add") || ir_bad(ir, m)
	case .PSUBB: return ir_xpacked(ir, ops, ea, V16B, "sub") || ir_bad(ir, m)
	case .PADDUSW: return ir_xpacked(ir, ops, ea, V8W, "uadd.sat") || ir_bad(ir, m)
	case .PSUBUSW: return ir_xpacked(ir, ops, ea, V8W, "usub.sat") || ir_bad(ir, m)
	case .PMAXSW: return ir_xpacked(ir, ops, ea, V8W, "smax") || ir_bad(ir, m)
	case .PMINSW: return ir_xpacked(ir, ops, ea, V8W, "smin") || ir_bad(ir, m)
	case .PMULLW: return ir_xpacked(ir, ops, ea, V8W, "mul") || ir_bad(ir, m)
	case .PCMPEQD: return ir_xpacked(ir, ops, ea, V4I, "eq") || ir_bad(ir, m)
	case .PCMPEQW: return ir_xpacked(ir, ops, ea, V8W, "eq") || ir_bad(ir, m)
	case .PCMPEQB: return ir_xpacked(ir, ops, ea, V16B, "eq") || ir_bad(ir, m)
	case .PCMPGTD: return ir_xpacked(ir, ops, ea, V4I, "sgt") || ir_bad(ir, m)
	case .PCMPGTW: return ir_xpacked(ir, ops, ea, V8W, "sgt") || ir_bad(ir, m)
	case .PMULUDQ:
		if !X {return ir_bad(ir, m)}
		mask := "splat (i64 4294967295)"
		a := ir_op(ir, "and", V2Q, ir_xget(ir, d, V2Q), mask)
		b := ir_op(ir, "and", V2Q, ir_cast(ir, "bitcast", V4I, ir_xsrc(ir, ops[1], ea, 16), V2Q), mask)
		ir_xput(ir, d, ir_op(ir, "mul", V2Q, a, b), V2Q)
		return true
	case .PEXTRW:
		if ops[0].kind != .REGISTER || !ir_is_x(ops[1]) {return ir_bad(ir, m)}
		w := ir_ext(ir, V8W, ir_xget(ir, ir_xn(ops[1]), V8W), int(imm & 7))
		ir_write(ir, ops[0], "", ir_cast(ir, "zext", "i16", w, "i32"))
		return true
	case .PINSRW:
		if !X {return ir_bad(ir, m)}
		v := ""
		if ops[1].kind == .MEMORY {v = ir_loadt(ir, ea, "i16")} else if ops[1].kind == .REGISTER {
			hw, _, _ := ir_reg(ir, ops[1].reg)
			v = ir_cast(ir, "trunc", "i32", ir.s.r[hw], "i16")
		} else {return ir_bad(ir, m)}
		ir_xput(ir, d, ir_ins_el(ir, V8W, ir_xget(ir, d, V8W), "i16", v, int(imm & 7)), V8W)
		return true
	case .PSLLD, .PSRLD, .PSRAD, .PSLLQ, .PSRLQ, .PSLLW, .PSRLW, .PSRAW:
		if !X {return ir_bad(ir, m)}
		ty, lb := V4I, 32
		#partial switch m {
		case .PSLLQ, .PSRLQ: ty, lb = V2Q, 64
		case .PSLLW, .PSRLW, .PSRAW: ty, lb = V8W, 16
		}
		ety := ir_ty(lb)
		arith := m == .PSRAD || m == .PSRAW
		if ops[1].kind != .IMMEDIATE {
			if !ir_feat(ir, "psr") {return true}
			// the count: the source's low 64 bits, past the lane width all zero (arithmetic: the sign)
			c64 := ir_ext(ir, V2Q, ir_cast(ir, "bitcast", V4I, ir_xsrc(ir, ops[1], ea, 16), V2Q), 0)
			big := ir_cmp(ir, "ugt", "i64", c64, ir_num(i64(lb - 1)))
			cl := ir_sel(ir, "i64", big, ir_num(i64(lb - 1)), c64)
			ce := lb == 64 ? cl : ir_cast(ir, "trunc", "i64", cl, ety)
			sp := ir_v(ir)
			ir_w(ir, sp, " = insertelement ", ty, " poison, ", ety, " ", ce, ", i32 0")
			spl := ir_v(ir)
			nl := lb == 64 ? 2 : lb == 32 ? 4 : 8
			ir_w(ir, spl, " = shufflevector ", ty, " ", sp, ", ", ty, " poison, <", ir_num(i64(nl)), " x i32> zeroinitializer")
			op := "shl"
			#partial switch m {
			case .PSRLD, .PSRLQ, .PSRLW: op = "lshr"
			case .PSRAD, .PSRAW: op = "ashr"
			}
			v := ir_op(ir, op, ty, ir_xget(ir, d, ty), spl)
			if !arith {v = ir_sel(ir, ty, big, "zeroinitializer", v)}
			ir_xput(ir, d, v, ty)
			return true
		}
		cnt := int(u8(ops[1].immediate))
		if cnt >= lb && !arith {ir.s.x[d] = "zeroinitializer"; return true}
		if arith {cnt = min(cnt, lb - 1)}
		op := "shl"
		#partial switch m {
		case .PSRLD, .PSRLQ, .PSRLW: op = "lshr"
		case .PSRAD, .PSRAW: op = "ashr"
		}
		sp := strings.concatenate({"splat (", ety, " ", ir_num(i64(cnt)), ")"}, context.temp_allocator)
		ir_xput(ir, d, ir_op(ir, op, ty, ir_xget(ir, d, ty), sp), ty)
		return true
	}
	return false
}
ir_bad :: proc(ir: ^Ir, m: x86.Mnemonic) -> bool {ir_fail(ir, fmt.tprintf("sse %v", m)); return true}
// <n x double> (integral) -> <n x i32>, out of range or NaN 0x80000000 (cvt_trunc / cvt_round)
ir_vf2i :: proc(ir: ^Ir, v: string, n: int) -> string {
	dt := n == 4 ? "<4 x double>" : V2D
	it := n == 4 ? V4I : "<2 x i32>"
	ct := n == 4 ? "<4 x i1>" : "<2 x i1>"
	lo := strings.concatenate({"splat (double ", ir_dc(-2147483648.0), ")"}, context.temp_allocator)
	hi := strings.concatenate({"splat (double ", ir_dc(2147483648.0), ")"}, context.temp_allocator)
	ok := ir_op(ir, "and", ct, ir_fcmp(ir, "oge", dt, v, lo), ir_fcmp(ir, "olt", dt, v, hi))
	i := ir_cast(ir, "fptosi", dt, v, it)
	fr := ir_v(ir)
	ir_w(ir, fr, " = freeze ", it, " ", i)
	return ir_selv(ir, ct, ok, it, fr, "splat (i32 -2147483648)")
}

// ---- calls and returns ----

IR_UNLIKELY :: ", !prof !0"

// XRC_PREEMPT: a waiting guest thread may get the GIL here
ir_preempt :: proc(ir: ^Ir) {
	w := ir_v(ir)
	ir_w(ir, w, " = load volatile i32, ptr @xrc_gil_waiters, align 4")
	lw := ir_newlabel(ir, "pw")
	lc := ir_newlabel(ir, "pc")
	lp := ir_newlabel(ir, "pp")
	ir_w(ir, "br i1 ", ir_cmp(ir, "ne", "i32", w, "0"), ", label %", lw, ", label %", lc, IR_UNLIKELY)
	ir_label(ir, lw)
	pn := ir_cfield(ir, IR_OFF_PREN)
	x := ir_v(ir)
	ir_w(ir, x, " = load i32, ptr ", pn, ", align 4")
	x1 := ir_op(ir, "add", "i32", x, "1")
	ir_w(ir, "store i32 ", x1, ", ptr ", pn, ", align 4")
	ir_w(ir, "br i1 ", ir_cmp(ir, "uge", "i32", x1, "64"), ", label %", lp, ", label %", lc)
	ir_label(ir, lp)
	ir_w(ir, "call void @xrc_preempt_n(ptr %c)")
	ir_w(ir, "br label %", lc)
	ir_label(ir, lc)
}
ir_spill :: proc(ir: ^Ir) {
	for k in 0 ..< 8 {ir_cstore(ir, k * 4, "i32", ir.s.r[k])}
}
ir_xspill :: proc(ir: ^Ir) {
	for k in 0 ..< 8 {if ir.named[k] {ir_cstore(ir, IR_OFF_X + 16 * k, V4I, ir.s.x[k])}}
}
ir_xfill :: proc(ir: ^Ir) {
	for k in 0 ..< 8 {if ir.named[k] {ir.s.x[k] = ir_cload(ir, IR_OFF_X + 16 * k, V4I)}}
}
ir_fill :: proc(ir: ^Ir, mask: int) {
	// eax ecx edx esp always; ebx esi edi ebp per mask bits 0..3 (x_abi_reload)
	ir.s.r[0] = ir_cload(ir, 0, "i32")
	ir.s.r[1] = ir_cload(ir, 4, "i32")
	ir.s.r[2] = ir_cload(ir, 8, "i32")
	ir.s.r[4] = ir_cload(ir, 16, "i32")
	if mask & 1 != 0 {ir.s.r[3] = ir_cload(ir, 12, "i32")}
	if mask & 2 != 0 {ir.s.r[6] = ir_cload(ir, 24, "i32")}
	if mask & 4 != 0 {ir.s.r[7] = ir_cload(ir, 28, "i32")}
	if mask & 8 != 0 {ir.s.r[5] = ir_cload(ir, 20, "i32")}
}
ir_declare :: proc(ir: ^Ir, sym: string) {ir.mod.decls[strings.clone(sym)] = true}

// a guest call to target t (direct: a translated function's address, else 0 and tv its value), returning to ra
// (an i32 value); the return address is pushed here. key: the call's entry in the x87 flow (xl.callfp); tail: a
// tail call (its x87 arguments are the whole stack)
ir_call_to :: proc(ir: ^Ir, ins: ^Rx_Ins, direct: u32, tv: string, ra: string, key: u32, tail: bool, inl := false) {
	e := ir.xl.callfp[key]
	fpc := !(e.transparent || (e.kin == 0 && e.kout == 0))
	base := 0
	if fpc {
		if !tail {base = max(ir.xl.fdepth[ins.addr] - e.kin, 0)}
		if e.kin >= 1 {ir_cstore(ir, IR_OFF_FA, "double", ir.s.st[base + e.kin - 1])}
		if e.kin >= 2 {ir_cstore(ir, IR_OFF_FA + 8, "double", ir.s.st[base + e.kin - 2])}
	}
	ir.s.r[4] = ir_op(ir, "sub", "i32", ir.s.r[4], "4")
	ir_store(ir, ir.s.r[4], 32, ra)
	ir_spill(ir)
	ir_xspill(ir)
	if direct != 0 && direct in x_direct_ok {
		sym := ""
		if inl {
			// (inlined: no preemption check here -- loops of inlined calls get one on their back edge)
			sym = strings.concatenate({"xi_", ir_hex8(direct)}, context.temp_allocator)
			ir.mod.copies[direct] = true
		} else {
			ir_preempt(ir)
			sym = strings.concatenate({direct in x_fast ? "xo_" : "xf_", ir_hex8(direct)}, context.temp_allocator)
			ir_declare(ir, sym)
		}
		ir_w(ir, "call void @", sym, "(ptr %c)")
		ev := ir_cload(ir, IR_OFF_EIP, "i32")
		lb := ir_newlabel(ir, "badret")
		lo := ir_newlabel(ir, "ret")
		ir_w(ir, "br i1 ", ir_cmp(ir, "ne", "i32", ev, ra), ", label %", lb, ", label %", lo, IR_UNLIKELY)
		ir_label(ir, lb)
		ir_w(ir, "call void @xrc_bad_return(ptr %c, i32 ", ir_num(i64(i32(direct))), ")")
		ir_w(ir, "br label %", lo)
		ir_label(ir, lo)
		ir_fill(ir, x_abi_reload(direct))
	} else {
		// the site's inline cache (XIc: addr imp fn addr2 imp2 fn2)
		ic := fmt.tprintf("@%s_%08x_%d", ir.copy ? "xic" : "ic", ins.addr, ir.nl)
		ir.nl += 1
		strings.write_string(&ir.mod.globals, ic)
		strings.write_string(&ir.mod.globals, " = internal global { i32, i32, ptr, i32, i32, ptr } { i32 -1, i32 0, ptr null, i32 0, i32 0, ptr null }\n")
		t := direct != 0 ? ir_num(i64(i32(direct))) : tv
		// XCALL's hit path inline (the cached target: called through its pointer), the rest in the runtime
		ca := ir_v(ir)
		ir_w(ir, ca, " = load i32, ptr ", ic, ", align 8")
		lh := ir_newlabel(ir, "ichit")
		lm := ir_newlabel(ir, "icmiss")
		lj := ir_newlabel(ir, "icdone")
		ir_w(ir, "br i1 ", ir_cmp(ir, "eq", "i32", ca, t), ", label %", lh, ", label %", lm)
		ir_label(ir, lh)
		ir_preempt(ir)
		ip := ir_v(ir)
		ir_w(ir, ip, " = getelementptr i8, ptr ", ic, ", i64 4")
		imp := ir_v(ir)
		ir_w(ir, imp, " = load i32, ptr ", ip, ", align 4")
		li := ir_newlabel(ir, "icimp")
		lc := ir_newlabel(ir, "iccall")
		ir_w(ir, "br i1 ", ir_cmp(ir, "ne", "i32", imp, "0"), ", label %", li, ", label %", lc, IR_UNLIKELY)
		ir_label(ir, li)
		ir_w(ir, "call void @xrc_imp_enter(i32 ", ir_op(ir, "sub", "i32", imp, "1"), ")")
		ir_w(ir, "br label %", lc)
		ir_label(ir, lc)
		fp := ir_v(ir)
		ir_w(ir, fp, " = getelementptr i8, ptr ", ic, ", i64 8")
		fn := ir_v(ir)
		ir_w(ir, fn, " = load ptr, ptr ", fp, ", align 8")
		ir_w(ir, "call void ", fn, "(ptr %c)")
		ev := ir_cload(ir, IR_OFF_EIP, "i32")
		lb := ir_newlabel(ir, "badret")
		ir_w(ir, "br i1 ", ir_cmp(ir, "ne", "i32", ev, ra), ", label %", lb, ", label %", lj, IR_UNLIKELY)
		ir_label(ir, lb)
		ir_w(ir, "call void @xrc_bad_return(ptr %c, i32 ", t, ")")
		ir_w(ir, "br label %", lj)
		ir_label(ir, lm)
		ir_w(ir, "call void @xrc_dispatch_ic(ptr %c, i32 ", t, ", ptr ", ic, ")")
		ir_w(ir, "br label %", lj)
		ir_label(ir, lj)
		ir_fill(ir, 0xf)
	}
	ir_xfill(ir)
	if fpc && e.kout >= 1 {ir.s.st[base] = ir_cload(ir, IR_OFF_ST0, "double")}
}
// call (not a terminator): XCALL / XCALLDK with the stale-slot fix of x_call
ir_call :: proc(ir: ^Ir, ins: ^Rx_Ins, ea: string) {
	op := ins.inst.ops[0]
	ra := ir_num(i64(i32(ins.addr + ins.len)))
	direct: u32 = 0
	tv := ""
	if op.kind == .RELATIVE {direct = rx_rel_target(ins, 0)} else {tv = ir_read(ir, op, ea)}
	e := ir.xl.callfp[ins.addr]
	plain := e.transparent || (e.kin == 0 && e.kout == 0)
	if direct != 0 {
		if ir_inlinable(ir, direct, plain) {ir_call_site(ir, ins, direct, "", ra, true); return}
		if ck, is_ck := x_cookie_check(direct); is_ck && direct in x_direct_ok && plain {
			// __security_check_cookie: the compare inline (its flags too), the call only when the cookie differs;
			// else the slot below esp gets the return address, as the call would have left it
			cv := ir_load(ir, ir_num(i64(i32(ck))), 32)
			a := ir.s.r[1]
			_ = ir_alu(ir, .CMP, a, cv, 32)
			lc := ir_newlabel(ir, "ckcall")
			lk := ir_newlabel(ir, "ckok")
			lj := ir_newlabel(ir, "ckj")
			ir_w(ir, "br i1 ", ir_cmp(ir, "ne", "i32", a, cv), ", label %", lc, ", label %", lk, IR_UNLIKELY)
			s0 := ir.s
			preds := make([dynamic]Ir_Pred, context.temp_allocator)
			ir_label(ir, lc)
			ir_call_site(ir, ins, direct, "", ra, false)
			append(&preds, Ir_Pred{ir.cur, ir.s})
			ir_w(ir, "br label %", lj)
			ir.s = s0
			ir_label(ir, lk)
			ir_store(ir, ir_op(ir, "sub", "i32", ir.s.r[4], "4"), 32, ra)
			append(&preds, Ir_Pred{ir.cur, ir.s})
			ir_w(ir, "br label %", lj)
			ir_label(ir, lj)
			ir_merge(ir, preds[:])
			return
		}
		ir_call_site(ir, ins, direct, "", ra, false)
		return
	}
	if dt, is_dv := x_devirt[ins.addr + ins.len]; is_dv && !ir.copy {
		// a profiled site's target(s) called directly (inlined when it can be) when the computed target is it
		dts := [2]u32{dt, x_devirt2[ins.addr + ins.len]}
		s0 := ir.s
		lj := ir_newlabel(ir, "dvj")
		preds := make([dynamic]Ir_Pred, context.temp_allocator)
		for d in dts {
			if d == 0 {continue}
			ly := ir_newlabel(ir, "dv")
			ln := ir_newlabel(ir, "dvn")
			ir_w(ir, "br i1 ", ir_cmp(ir, "eq", "i32", tv, ir_num(i64(i32(d)))), ", label %", ly, ", label %", ln)
			ir_label(ir, ly)
			ir.s = s0
			ir_call_site(ir, ins, d, "", ra, ir_inlinable(ir, d, plain))
			append(&preds, Ir_Pred{ir.cur, ir.s})
			ir_w(ir, "br label %", lj)
			ir_label(ir, ln)
			ir.s = s0
		}
		ir_call_site(ir, ins, 0, tv, ra, false)
		append(&preds, Ir_Pred{ir.cur, ir.s})
		ir_w(ir, "br label %", lj)
		ir_label(ir, lj)
		ir_merge(ir, preds[:])
		return
	}
	ir_call_site(ir, ins, 0, tv, ra, false)
}
ir_inlinable :: proc(ir: ^Ir, t: u32, plain: bool) -> bool {
	if ir.copy || !plain || t == ir.a || t not_in x_inline || t not_in x_direct_ok || t in x_fast {return false}
	if os.get_env("XRC_IR_INLINE", context.temp_allocator) == "0" {return false}
	return ir_copy(t)
}
Ir_Pred :: struct {lab: string, s: Ir_St}
// the state at a join of several paths (at the start of its block): a phi of each component that differs
ir_merge :: proc(ir: ^Ir, preds: []Ir_Pred) {
	ns := preds[0].s
	for k in 0 ..< IR_NCOMP {
		if !ir_comp_on(ir, k) {continue}
		p0, ty, _ := ir_comp(&ns, k)
		same := true
		for &pr in preds[1:] {
			pk, _, _ := ir_comp(&pr.s, k)
			if pk^ != p0^ {same = false; break}
		}
		if same {continue}
		d := ir_v(ir)
		sb := strings.builder_make(context.temp_allocator)
		for &pr, q in preds {
			pk, _, _ := ir_comp(&pr.s, k)
			if q > 0 {strings.write_string(&sb, ", ")}
			strings.write_string(&sb, "[ ")
			strings.write_string(&sb, pk^)
			strings.write_string(&sb, ", %")
			strings.write_string(&sb, pr.lab)
			strings.write_string(&sb, " ]")
		}
		ir_w(ir, d, " = phi ", ty, " ", strings.to_string(sb))
		p0^ = d
	}
	ir.s = ns
}
// one call (the return address pushed by the call protocol) with x_call's stale-slot fix after it
ir_call_site :: proc(ir: ^Ir, ins: ^Rx_Ins, direct: u32, tv: string, ra: string, inl: bool) {
	xs := ir.s.r[4]
	ir_call_to(ir, ins, direct, tv, ra, ins.addr, false, inl)
	// the dead slot below the guest esp gets the original's return address (x_call's comment)
	ls := ir_newlabel(ir, "slot")
	ld := ir_newlabel(ir, "slotd")
	ir_w(ir, "br i1 ", ir_cmp(ir, "uge", "i32", ir.s.r[4], xs), ", label %", ls, ", label %", ld)
	ir_label(ir, ls)
	ir_store(ir, ir_op(ir, "sub", "i32", xs, "4"), 32, ra)
	ir_w(ir, "br label %", ld)
	ir_label(ir, ld)
}
// XRC_RETURN(n) (XRC_RETURNX: the xmm registers back; XRC_FRET: a float result)
ir_ret :: proc(ir: ^Ir, n: u32) {
	if ir.xl.fret {ir_cstore(ir, IR_OFF_ST0, "double", ir.s.st[0])}
	ip := ir_load(ir, ir.s.r[4], 32)
	ir.s.r[4] = ir_op(ir, "add", "i32", ir.s.r[4], ir_num(i64(4 + n)))
	ir_spill(ir)
	ir_xspill(ir)
	ir_cstore(ir, IR_OFF_EIP, "i32", ip)
	ir_w(ir, "ret void")
}
// a tail call (jmp / jcc out of the function, or falling off its end): x_tail
ir_tail :: proc(ir: ^Ir, ins: ^Rx_Ins, direct: u32, tv: string, key: u32) {
	ip := ir_load(ir, ir.s.r[4], 32)
	ir.s.r[4] = ir_op(ir, "add", "i32", ir.s.r[4], "4")
	ir_call_to(ir, ins, direct, tv, ip, key, true)
	ir.s.r[4] = ir_op(ir, "sub", "i32", ir.s.r[4], "4")
	ir_store(ir, ir.s.r[4], 32, ip)
	if ir.xl.fret {
		e := ir.xl.callfp[key]
		if e.kout < 1 {ir.s.st[0] = "0.0"}
	}
	ir_ret(ir, 0)
}
ir_edge :: proc(ir: ^Ir, t: u32) {
	if t not_in ir.edges {ir.edges[t] = make([dynamic]Ir_Edge, context.temp_allocator)}
	append(&ir.edges[t], Ir_Edge{ir.cur, ir.blk, ir.s})
	if ir.blk not_in ir.succ {ir.succ[ir.blk] = make([dynamic]u32, context.temp_allocator)}
	append(&ir.succ[ir.blk], t)
}
ir_lab :: proc(a: u32) -> string {return strings.concatenate({"L_", ir_hex8(a)}, context.temp_allocator)}

// XRC_IR_OFF=div,rep,...: features left to C (bisecting a wrong translation): false = off (the function stays C)
ir_off_list: string
ir_off_read := false
ir_feat :: proc(ir: ^Ir, name: string) -> bool {
	if !ir_off_read {ir_off_read = true; ir_off_list = os.get_env("XRC_IR_OFF", context.allocator)}
	if ir_off_list == "" {return true}
	for f in strings.split(ir_off_list, ",", context.temp_allocator) {
		if f == name {ir_fail(ir, strings.concatenate({"off:", name}, context.temp_allocator)); return false}
	}
	return true
}
// XRC_INL_PREEMPT: a loop whose calls are all inlined has no preemption point but this, on its back edge
ir_inl_preempt :: proc(ir: ^Ir) {
	w := ir_v(ir)
	ir_w(ir, w, " = load volatile i32, ptr @xrc_gil_waiters, align 4")
	lw := ir_newlabel(ir, "ipw")
	lp := ir_newlabel(ir, "ipp")
	lc := ir_newlabel(ir, "ipc")
	ir_w(ir, "br i1 ", ir_cmp(ir, "ne", "i32", w, "0"), ", label %", lw, ", label %", lc, IR_UNLIKELY)
	ir_label(ir, lw)
	pn := ir_cfield(ir, IR_OFF_PREN)
	x := ir_v(ir)
	ir_w(ir, x, " = load i32, ptr ", pn, ", align 4")
	x1 := ir_op(ir, "add", "i32", x, "1")
	ir_w(ir, "store i32 ", x1, ", ptr ", pn, ", align 4")
	ir_w(ir, "br i1 ", ir_cmp(ir, "uge", "i32", x1, "64"), ", label %", lp, ", label %", lc)
	ir_label(ir, lp)
	ir_spill(ir)
	ir_w(ir, "call void @xrc_preempt_n(ptr %c)")
	ir_w(ir, "br label %", lc)
	ir_label(ir, lc)
}
// XRC_SPIN_YIELD
ir_spin_yield :: proc(ir: ^Ir) {
	w := ir_v(ir)
	ir_w(ir, w, " = load volatile i32, ptr @xrc_gil_waiters, align 4")
	ly := ir_newlabel(ir, "sy")
	lc := ir_newlabel(ir, "syc")
	ir_w(ir, "br i1 ", ir_cmp(ir, "ne", "i32", w, "0"), ", label %", ly, ", label %", lc, IR_UNLIKELY)
	ir_label(ir, ly)
	ir_w(ir, "call void @xrc_spin_yield()")
	ir_w(ir, "br label %", lc)
	ir_label(ir, lc)
}
// XRC_UD: an invalid / privileged instruction (the runtime reports it and stops)
ir_ud :: proc(ir: ^Ir, addr: u32) {
	ir_spill(ir)
	ir_w(ir, "call void @xrc_ud(ptr %c, i32 ", ir_num(i64(i32(addr))), ")")
}
// EFLAGS as pushfd stores it (xrc_eflags_pack: the arithmetic flags over the guest's system bits)
ir_eflags :: proc(ir: ^Ir) -> string {
	f := &ir.s.f
	sys := ir_cload(ir, IR_OFF_EFLSYS, "i32")
	v := ir_op(ir, "or", "i32", ir_op(ir, "and", "i32", sys, "-2262"), "2")
	BITS := [5]int{6, 7, 0, 11, 2} // ZF SF CF OF PF
	for k in 0 ..< 5 {
		v = ir_op(ir, "or", "i32", v, ir_op(ir, "shl", "i32", ir_cast(ir, "zext", "i1", f[k], "i32"), ir_num(i64(BITS[k]))))
	}
	return v
}
// mul / imul r/m8, r/m16 (ax = al * src; dx:ax = ax * src)
ir_mul816 :: proc(ir: ^Ir, op: x86.Operand, ea: string, bits: int, signed: bool) {
	ty := ir_ty(bits)
	w := bits == 8 ? "i16" : "i32"
	ext := signed ? "sext" : "zext"
	a := ir_cast(ir, ext, ty, ir_tr(ir, ir.s.r[0], bits), w)
	b := ir_cast(ir, ext, ty, ir_tr(ir, ir_read(ir, op, ea), bits), w)
	p := ir_op(ir, "mul", w, a, b)
	lo := ir_cast(ir, "trunc", w, p, ty)
	ov := signed ? ir_cmp(ir, "ne", w, p, ir_cast(ir, "sext", ty, lo, w)) : ir_cmp(ir, "ne", w, ir_op(ir, "lshr", w, p, ir_num(i64(bits))), "0")
	p32 := bits == 8 ? ir_cast(ir, "zext", "i16", p, "i32") : p
	if bits == 8 {
		ir_reg_write(ir, x86.AX, p32)
	} else {
		ir_reg_write(ir, x86.AX, p32)
		ir_reg_write(ir, x86.DX, ir_op(ir, "lshr", "i32", p32, "16"))
	}
	ir.s.f[IR_CF], ir.s.f[IR_OF] = ov, ov
	ir_zsp(ir, lo, bits)
}
// MMX (xop): mm0-7 the function's own i64 values (x_mm in the C)
ir_mmx :: proc(ir: ^Ir, ins: ^Rx_Ins, ea: string) {
	mm := &ir.s.mm
	xr, xrm := int(ins.xr & 7), int(ins.xrm & 7)
	src64 :: proc(ir: ^Ir, ins: ^Rx_Ins, ea: string, xrm: int) -> string {
		if ins.xmem {if ea == "" {ir_fail(ir, "mmx ea"); return "0"}; return ir_loadt(ir, ea, "i64")}
		return ir.s.mm[xrm]
	}
	#partial switch ins.xop {
	case .CVTTPS2PI, .CVTPS2PI:
		lo, hi := "", ""
		if ins.xmem {
			lo = ir_loadt(ir, ea, "float")
			hi = ir_loadt(ir, ir_op(ir, "add", "i32", ea, "4"), "float")
		} else {
			v := ir_xget(ir, xrm, V4F)
			lo, hi = ir_ext(ir, V4F, v, 0), ir_ext(ir, V4F, v, 1)
		}
		cv :: proc(ir: ^Ir, f: string, trunc: bool) -> string {
			d := ir_cast(ir, "fpext", "float", f, "double")
			r := trunc ? ir_rint(ir, "double", "f64", d, 3) : ir_rint(ir, "double", "f64", d, -1, ir_sse_rc(ir))
			return ir_cast(ir, "zext", "i32", ir_f2i(ir, r, 32), "i64")
		}
		t := ins.xop == .CVTTPS2PI
		mm[xr] = ir_op(ir, "or", "i64", cv(ir, lo, t), ir_op(ir, "shl", "i64", cv(ir, hi, t), "32"))
	case .CVTPI2PS:
		q := src64(ir, ins, ea, xrm)
		l := ir_cast(ir, "sitofp", "i32", ir_cast(ir, "trunc", "i64", q, "i32"), "float")
		h := ir_cast(ir, "sitofp", "i32", ir_cast(ir, "trunc", "i64", ir_op(ir, "lshr", "i64", q, "32"), "i32"), "float")
		v := ir_ins_el(ir, V4F, ir_xget(ir, xr, V4F), "float", l, 0)
		ir_xput(ir, xr, ir_ins_el(ir, V4F, v, "float", h, 1), V4F)
	case .MOVQ_LD: mm[xr] = src64(ir, ins, ea, xrm)
	case .MOVQ_ST:
		if ins.xmem {ir_storet(ir, ea, "i64", mm[xr])} else {mm[xrm] = mm[xr]}
	case .MOVD_LD:
		v := ""
		if ins.xmem {v = ir_loadt(ir, ea, "i32")} else {v = ir.s.r[xrm]}
		mm[xr] = ir_cast(ir, "zext", "i32", v, "i64")
	case .MOVD_ST:
		v := ir_cast(ir, "trunc", "i64", mm[xr], "i32")
		if ins.xmem {ir_storet(ir, ea, "i32", v)} else {ir.s.r[xrm] = v}
	case .MMXOP:
		a, b := mm[xr], src64(ir, ins, ea, xrm)
		r := ""
		switch ins.opc2 {
		case 0xdb: r = ir_op(ir, "and", "i64", a, b)
		case 0xdf: r = ir_op(ir, "and", "i64", ir_op(ir, "xor", "i64", a, "-1"), b)
		case 0xeb: r = ir_op(ir, "or", "i64", a, b)
		case 0xef: r = ir_op(ir, "xor", "i64", a, b)
		case 0xd4: r = ir_op(ir, "add", "i64", a, b)
		case 0xfb: r = ir_op(ir, "sub", "i64", a, b)
		case 0xfe, 0xfa:
			av := ir_cast(ir, "bitcast", "i64", a, "<2 x i32>")
			bv := ir_cast(ir, "bitcast", "i64", b, "<2 x i32>")
			r = ir_cast(ir, "bitcast", "<2 x i32>", ir_op(ir, ins.opc2 == 0xfe ? "add" : "sub", "<2 x i32>", av, bv), "i64")
		case: ir_fail(ir, "mmx op"); return
		}
		mm[xr] = r
	case: ir_fail(ir, fmt.tprintf("xop %v", ins.xop))
	}
}

// the string instructions (x_rep): the direction from DF (XCtx.eflags_sys: cld / std / popfd); rep / repe / repne
// drive ecx -- a loop of its own LLVM blocks, its state (registers, flags) through phis. rep movs / rep stos forward
// without the overlap that makes x86 repeat a pattern: one memmove / memset.
ir_rep :: proc(ir: ^Ir, ins: ^Rx_Ins, bits0: int) {
	m := ins.inst.mnemonic
	kind := ""
	es := bits0 / 8
	#partial switch m {
	case .MOVS, .MOVSB, .MOVSW, .MOVSD: kind = "movs"
	case .STOS, .STOSB, .STOSW, .STOSD: kind = "stos"
	case .LODS, .LODSB, .LODSW, .LODSD: kind = "lods"
	case .SCAS, .SCASB, .SCASW, .SCASD: kind = "scas"
	case .CMPS, .CMPSB, .CMPSW, .CMPSD: kind = "cmps"
	}
	#partial switch m {
	case .MOVSB, .STOSB, .LODSB, .SCASB, .CMPSB: es = 1
	case .MOVSW, .STOSW, .LODSW, .SCASW, .CMPSW: es = 2
	case .MOVSD, .STOSD, .LODSD, .SCASD, .CMPSD: es = 4
	}
	if es != 1 && es != 2 && es != 4 {ir_fail(ir, "string size"); return}
	eb := es * 8
	sys := ir_cload(ir, IR_OFF_EFLSYS, "i32")
	df := ir_cmp(ir, "ne", "i32", ir_op(ir, "and", "i32", sys, "1024"), "0")
	dl := ir_sel(ir, "i32", df, ir_num(i64(-es)), ir_num(i64(es)))
	body :: proc(ir: ^Ir, kind: string, es, eb: int, dl: string) {
		r := &ir.s.r
		switch kind {
		case "movs":
			ir_store(ir, r[7], eb, ir_load(ir, r[6], eb))
			r[6] = ir_op(ir, "add", "i32", r[6], dl)
			r[7] = ir_op(ir, "add", "i32", r[7], dl)
		case "stos":
			ir_store(ir, r[7], eb, r[0])
			r[7] = ir_op(ir, "add", "i32", r[7], dl)
		case "lods":
			v := ir_load(ir, r[6], eb)
			ir_reg_write(ir, eb == 8 ? x86.AL : eb == 16 ? x86.AX : x86.EAX, v)
			r[6] = ir_op(ir, "add", "i32", r[6], dl)
		case "scas":
			_ = ir_alu(ir, .CMP, r[0], ir_load(ir, r[7], eb), eb)
			r[7] = ir_op(ir, "add", "i32", r[7], dl)
		case "cmps":
			_ = ir_alu(ir, .CMP, ir_load(ir, r[6], eb), ir_load(ir, r[7], eb), eb)
			r[6] = ir_op(ir, "add", "i32", r[6], dl)
			r[7] = ir_op(ir, "add", "i32", r[7], dl)
		}
	}
	rp := ins.inst.flags.rep
	if rp == .NONE {
		body(ir, kind, es, eb, dl)
		return
	}
	// pre -> [fast] -> loop -> done
	s0 := ir.s
	lpre := ir.cur
	lh := ir_newlabel(ir, "rep")
	ld := ir_newlabel(ir, "repd")
	lt := ir_newlabel(ir, "reptest")
	ir_w(ir, "br i1 ", ir_cmp(ir, "ne", "i32", ir.s.r[1], "0"), ", label %", lt, ", label %", ld)
	ir_label(ir, lt)
	// the fast path: forward, and (movs) the destination not inside the source run / (stos) bytes or zero
	fast := (kind == "movs" || kind == "stos")
	lf := ir_newlabel(ir, "repfast")
	sf := ir.s
	if fast {
		nb := ir_op(ir, "mul", "i32", ir.s.r[1], ir_num(i64(es)))
		ok := ir_not1(ir, df)
		if kind == "movs" {
			ok = ir_op(ir, "and", "i1", ok, ir_cmp(ir, "uge", "i32", ir_op(ir, "sub", "i32", ir.s.r[7], ir.s.r[6]), nb))
		} else if es != 1 {
			ok = ir_op(ir, "and", "i1", ok, ir_cmp(ir, "eq", "i32", ir.s.r[0], "0"))
		}
		ir_w(ir, "br i1 ", ok, ", label %", lf, ", label %", lh)
		ir_label(ir, lf)
		dp := ir_gptr(ir, ir.s.r[7])
		n64 := ir_cast(ir, "zext", "i32", nb, "i64")
		if kind == "movs" {
			sp := ir_gptr(ir, ir.s.r[6])
			ir_w(ir, "call void @llvm.memmove.p0.p0.i64(ptr align 1 ", dp, ", ptr align 1 ", sp, ", i64 ", n64, ", i1 false)")
			sf.r[6] = ir_op(ir, "add", "i32", ir.s.r[6], nb)
		} else {
			ir_w(ir, "call void @llvm.memset.p0.i64(ptr align 1 ", dp, ", i8 ", ir_cast(ir, "trunc", "i32", ir.s.r[0], "i8"), ", i64 ", n64, ", i1 false)")
		}
		sf.r[7] = ir_op(ir, "add", "i32", ir.s.r[7], nb)
		sf.r[1] = "0"
		ir_w(ir, "br label %", ld)
	} else {
		ir_w(ir, "br label %", lh)
	}
	// the loop body into its own text: its phis (from the test block and from itself) are known after it
	main := ir.b
	tb := strings.builder_make(context.temp_allocator)
	ir.b = &tb
	ir.cur = lh
	ph: [13]string
	for k in 0 ..< 8 {ph[k] = ir_v(ir); ir.s.r[k] = ph[k]}
	for k in 0 ..< 5 {ph[8 + k] = ir_v(ir); ir.s.f[k] = ph[8 + k]}
	body(ir, kind, es, eb, dl)
	ir.s.r[1] = ir_op(ir, "sub", "i32", ir.s.r[1], "1")
	cont := ir_cmp(ir, "ne", "i32", ir.s.r[1], "0")
	if kind == "scas" || kind == "cmps" {
		zc := rp == .REPNE ? ir_not1(ir, ir.s.f[IR_ZF]) : ir.s.f[IR_ZF]
		cont = ir_op(ir, "and", "i1", cont, zc)
	}
	ir_w(ir, "br i1 ", cont, ", label %", lh, ", label %", ld)
	s1 := ir.s
	ir.b = main
	ir_label(ir, lh)
	for k in 0 ..< 13 {
		a, b := k < 8 ? sf.r[k] : sf.f[k - 8], k < 8 ? s1.r[k] : s1.f[k - 8]
		if fast {a = k < 8 ? s0.r[k] : s0.f[k - 8]}
		ir_w(ir, ph[k], " = phi ", k < 8 ? "i32" : "i1", " [ ", a, ", %", lt, " ], [ ", b, ", %", lh, " ]")
	}
	strings.write_string(ir.b, strings.to_string(tb))
	// done: from the pre block (ecx 0), the fast path, the loop
	ir_label(ir, ld)
	for k in 0 ..< 13 {
		d := ir_v(ir)
		ty := k < 8 ? "i32" : "i1"
		a := k < 8 ? s0.r[k] : s0.f[k - 8]
		b := k < 8 ? s1.r[k] : s1.f[k - 8]
		if fast {
			c := k < 8 ? sf.r[k] : sf.f[k - 8]
			ir_w(ir, d, " = phi ", ty, " [ ", a, ", %", lpre, " ], [ ", c, ", %", lf, " ], [ ", b, ", %", lh, " ]")
		} else {
			ir_w(ir, d, " = phi ", ty, " [ ", a, ", %", lpre, " ], [ ", b, ", %", lh, " ]")
		}
		if k < 8 {ir.s.r[k] = d} else {ir.s.f[k - 8] = d}
	}
}

// ---- one instruction (not a branch) ----

ir_ins :: proc(ir: ^Ir, ins: ^Rx_Ins) {
	m := ins.inst.mnemonic
	n := int(ins.inst.operand_count)
	ops := ins.inst.ops
	ea := ""
	for k in 0 ..< n {
		if ops[k].kind == .MEMORY {
			if m == .LEA || (m == .NOP && ins.xop == .None) {break}
			ea = ir_addr(ir, ops[k].mem)
			break
		}
	}
	// the SSE control / status register (XCtx.mxcsr, as the portable C: XRC_LDMXCSR)
	#partial switch ins.xop {
	case .None:
	case .LDMXCSR: if !ir_feat(ir, "mxcsr") {return}; if ea == "" {ir_fail(ir, "ldmxcsr")} else {ir_cstore(ir, IR_OFF_MXCSR, "i32", ir_loadt(ir, ea, "i32"))}; return
	case .STMXCSR: if !ir_feat(ir, "mxcsr") {return}; if ea == "" {ir_fail(ir, "stmxcsr")} else {ir_storet(ir, ea, "i32", ir_cload(ir, IR_OFF_MXCSR, "i32"))}; return
	case .EMMS: return
	case .UD: ir_ud(ir, ins.addr); return
	case: if ir_feat(ir, "mmx") {ir_mmx(ir, ins, ea)}; return
	}
	if x87_is(ins) {
		d, ok := ir.xl.fdepth[ins.addr]
		if !ok {ir_fail(ir, "x87 depth"); return}
		_ = ir_x87(ir, ins, d, ea)
		return
	}
	bits := n > 0 ? int(ops[0].size) * 8 : 32
	#partial switch m {
	case .NOP:
	case .MOV:
		if n != 2 {ir_fail(ir, "mov"); return}
		if ir_is_x(ops[0]) || ir_is_x(ops[1]) {ir_fail(ir, "mov xmm"); return}
		ir_write(ir, ops[0], ea, ir_read(ir, ops[1], ea))
	case .MOVZX:
		if bt, is_bt := rx_byte_tables[ins.addr]; is_bt && ea != "" && ins.addr in ir.xl.btables {
			// a switch's byte index table: in .text, which the retail exe has encrypted -- its entries as a constant
			// array here, by address (an address outside it read from memory, as the C's default)
			g := fmt.tprintf("@%s_%08x", ir.copy ? "xbt" : "bt", ins.addr)
			strings.write_string(&ir.mod.globals, g)
			strings.write_string(&ir.mod.globals, " = private unnamed_addr constant [")
			strings.write_string(&ir.mod.globals, ir_num(i64(bt.n)))
			strings.write_string(&ir.mod.globals, " x i8] [")
			for b, k in exe_read(bt.at, int(bt.n)) {
				if k > 0 {strings.write_string(&ir.mod.globals, ", ")}
				strings.write_string(&ir.mod.globals, "i8 ")
				strings.write_string(&ir.mod.globals, ir_num(i64(i8(b))))
			}
			strings.write_string(&ir.mod.globals, "]\n")
			off := ir_op(ir, "sub", "i32", ea, ir_num(i64(i32(bt.at))))
			inr := ir_cmp(ir, "ult", "i32", off, ir_num(i64(bt.n)))
			lin := ir_newlabel(ir, "btin")
			lout := ir_newlabel(ir, "btout")
			lj := ir_newlabel(ir, "btj")
			ir_w(ir, "br i1 ", inr, ", label %", lin, ", label %", lout)
			ir_label(ir, lin)
			p := ir_v(ir)
			ir_w(ir, p, " = getelementptr [", ir_num(i64(bt.n)), " x i8], ptr ", g, ", i32 0, i32 ", off)
			v1 := ir_v(ir)
			ir_w(ir, v1, " = load i8, ptr ", p, ", align 1")
			ir_w(ir, "br label %", lj)
			ir_label(ir, lout)
			v2 := ir_loadt(ir, ea, "i8")
			ir_w(ir, "br label %", lj)
			ir_label(ir, lj)
			v := ir_v(ir)
			ir_w(ir, v, " = phi i8 [ ", v1, ", %", lin, " ], [ ", v2, ", %", lout, " ]")
			ir_write(ir, ops[0], ea, ir_cast(ir, "zext", "i8", v, "i32"))
			return
		}
		ir_write(ir, ops[0], ea, ir_read(ir, ops[1], ea))
	case .MOVSX:
		sb := int(ops[1].size) * 8
		ir_write(ir, ops[0], ea, ir_sx(ir, ir_tr(ir, ir_read(ir, ops[1], ea), sb), sb))
	case .LEA:
		if ops[1].kind != .MEMORY {ir_fail(ir, "lea"); return}
		ir_write(ir, ops[0], "", ir_addr(ir, ops[1].mem))
	case .ADD, .SUB, .ADC, .SBB, .AND, .OR, .XOR, .CMP, .TEST:
		if n != 2 || (bits != 8 && bits != 16 && bits != 32) {ir_fail(ir, "alu"); return}
		a := ir_read(ir, ops[0], ea)
		b := ir_read(ir, ops[1], ea)
		r := ""
		if (m == .XOR || m == .SUB) && ops[0].kind == .REGISTER && ops[1].kind == .REGISTER && ops[0].reg == ops[1].reg {
			r = ir_alu(ir, m, "0", "0", bits)
		} else {
			r = ir_alu(ir, m, a, b, bits)
		}
		if m != .CMP && m != .TEST {ir_write(ir, ops[0], ea, ir_zx(ir, r, bits))}
	case .INC, .DEC:
		ty := ir_ty(bits)
		a := ir_tr(ir, ir_read(ir, ops[0], ea), bits)
		r := ir_op(ir, m == .INC ? "add" : "sub", ty, a, "1")
		cf := ir.s.f[IR_CF]
		ox := m == .INC ? ir_op(ir, "and", ty, ir_op(ir, "xor", ty, r, a), ir_op(ir, "xor", ty, r, "1")) : ir_op(ir, "and", ty, ir_op(ir, "xor", ty, a, "1"), ir_op(ir, "xor", ty, a, r))
		ir.s.f[IR_OF] = ir_cmp(ir, "slt", ty, ox, "0")
		ir_zsp(ir, r, bits)
		ir.s.f[IR_CF] = cf
		ir_write(ir, ops[0], ea, ir_zx(ir, r, bits))
	case .NEG:
		ty := ir_ty(bits)
		a := ir_tr(ir, ir_read(ir, ops[0], ea), bits)
		r := ir_op(ir, "sub", ty, "0", a)
		ir.s.f[IR_CF] = ir_cmp(ir, "ne", ty, a, "0")
		ir.s.f[IR_OF] = ir_cmp(ir, "slt", ty, ir_op(ir, "and", ty, a, r), "0")
		ir_zsp(ir, r, bits)
		ir_write(ir, ops[0], ea, ir_zx(ir, r, bits))
	case .NOT:
		a := ir_read(ir, ops[0], ea)
		ir_write(ir, ops[0], ea, ir_op(ir, "xor", "i32", a, "-1"))
	case .RCL, .RCR:
		if !ir_feat(ir, "rcr") {return}
		// through the carry: a (w+1)-bit rotate of CF:value, the count modulo w+1; only CF changes (as the C)
		cnt := "1"
		if n >= 2 {
			if ops[1].kind == .IMMEDIATE {cnt = ir_num(i64(ops[1].immediate & 31))} else {cnt = ir_read(ir, ops[1], ea)}
		}
		W := i64(bits + 1)
		a := ir_read(ir, ops[0], ea)
		x := ir_op(ir, "or", "i64", ir_cast(ir, "zext", "i32", a, "i64"), ir_op(ir, "shl", "i64", ir_cast(ir, "zext", "i1", ir.s.f[IR_CF], "i64"), ir_num(i64(bits))))
		k := ir_cast(ir, "zext", "i32", ir_op(ir, "urem", "i32", ir_op(ir, "and", "i32", cnt, "31"), ir_num(W)), "i64")
		ik := ir_op(ir, "sub", "i64", ir_num(W), k)
		r := m == .RCR ? ir_op(ir, "or", "i64", ir_op(ir, "lshr", "i64", x, k), ir_op(ir, "shl", "i64", x, ik)) : ir_op(ir, "or", "i64", ir_op(ir, "shl", "i64", x, k), ir_op(ir, "lshr", "i64", x, ik))
		z := ir_cmp(ir, "eq", "i64", k, "0")
		r = ir_sel(ir, "i64", z, x, r)
		ir.s.f[IR_CF] = ir_cast(ir, "trunc", "i64", ir_op(ir, "lshr", "i64", r, ir_num(i64(bits))), "i1")
		ir_write(ir, ops[0], ea, ir_op(ir, "and", "i32", ir_cast(ir, "trunc", "i64", r, "i32"), ir_num(i64(i32(u32((u64(1) << uint(bits)) - 1))))))
	case .SHL, .SAL, .SHR, .SAR, .ROL, .ROR:
		cnt := "1"
		if n >= 2 {
			if ops[1].kind == .IMMEDIATE {cnt = ir_num(i64(ops[1].immediate & 31))} else {cnt = ir_read(ir, ops[1], ea)}
		}
		a := ir_read(ir, ops[0], ea)
		r := ir_shift(ir, m, a, cnt, bits)
		ir_write(ir, ops[0], ea, r)
	case .IMUL:
		if n == 1 && bits != 32 {if ir_feat(ir, "mul816") {ir_mul816(ir, ops[0], ea, bits, true)}; return}
		if n == 1 {
			s := ir_cast(ir, "sext", "i32", ir_read(ir, ops[0], ea), "i64")
			p := ir_op(ir, "mul", "i64", ir_cast(ir, "sext", "i32", ir.s.r[0], "i64"), s)
			lo := ir_cast(ir, "trunc", "i64", p, "i32")
			hi := ir_cast(ir, "trunc", "i64", ir_op(ir, "lshr", "i64", p, "32"), "i32")
			ov := ir_cmp(ir, "ne", "i64", p, ir_cast(ir, "sext", "i32", lo, "i64"))
			ir.s.r[0], ir.s.r[2] = lo, hi
			ir.s.f[IR_CF], ir.s.f[IR_OF] = ov, ov
			ir_zsp(ir, lo, 32)
			return
		}
		if bits != 32 && bits != 16 {ir_fail(ir, "imul"); return}
		a := ir_read(ir, ops[n == 3 ? 1 : 0], ea)
		b := ir_read(ir, ops[n == 3 ? 2 : 1], ea)
		ty := ir_ty(bits)
		an := ir_tr(ir, a, bits)
		bn := ir_tr(ir, b, bits)
		r := ir_op(ir, "mul", ty, an, bn)
		w := bits == 32 ? "i64" : "i32"
		wide := ir_op(ir, "mul", w, ir_cast(ir, "sext", ty, an, w), ir_cast(ir, "sext", ty, bn, w))
		ov := ir_cmp(ir, "ne", w, wide, ir_cast(ir, "sext", ty, r, w))
		ir.s.f[IR_CF], ir.s.f[IR_OF] = ov, ov
		ir_zsp(ir, r, bits)
		ir_write(ir, ops[0], ea, ir_zx(ir, r, bits))
	case .MUL:
		if n == 1 && bits != 32 {if ir_feat(ir, "mul816") {ir_mul816(ir, ops[0], ea, bits, false)}; return}
		if n != 1 {ir_fail(ir, "mul"); return}
		s := ir_cast(ir, "zext", "i32", ir_read(ir, ops[0], ea), "i64")
		p := ir_op(ir, "mul", "i64", ir_cast(ir, "zext", "i32", ir.s.r[0], "i64"), s)
		lo := ir_cast(ir, "trunc", "i64", p, "i32")
		hi := ir_cast(ir, "trunc", "i64", ir_op(ir, "lshr", "i64", p, "32"), "i32")
		ov := ir_cmp(ir, "ne", "i32", hi, "0")
		ir.s.r[0], ir.s.r[2] = lo, hi
		ir.s.f[IR_CF], ir.s.f[IR_OF] = ov, ov
		ir_zsp(ir, lo, 32)
	case .CDQ: ir.s.r[2] = ir_op(ir, "ashr", "i32", ir.s.r[0], "31")
	case .CWDE: ir.s.r[0] = ir_sx(ir, ir_tr(ir, ir.s.r[0], 16), 16)
	case .CBW: ir_reg_write(ir, x86.AX, ir_sx(ir, ir_tr(ir, ir.s.r[0], 8), 8))
	case .PUSH:
		if n != 1 {ir_fail(ir, "push"); return}
		if ops[0].kind == .REGISTER && x86.reg_class(ops[0].reg) != x86.REG_GPR32 {ir_fail(ir, "push reg"); return}
		if ops[0].kind == .MEMORY && ops[0].size != 4 {ir_fail(ir, "push m"); return}
		v := ""
		if ops[0].kind == .IMMEDIATE {v = ir_num(i64(i32(op_imm32(ops[0]))))} else {v = ir_read(ir, ops[0], ea)}
		ir.s.r[4] = ir_op(ir, "sub", "i32", ir.s.r[4], "4")
		ir_store(ir, ir.s.r[4], 32, v)
	case .POP:
		if n != 1 {ir_fail(ir, "pop"); return}
		if ops[0].kind == .REGISTER && x86.reg_class(ops[0].reg) != x86.REG_GPR32 {ir_fail(ir, "pop reg"); return}
		if ops[0].kind == .MEMORY && ops[0].size != 4 {ir_fail(ir, "pop m16"); return}
		v := ir_load(ir, ir.s.r[4], 32)
		ir.s.r[4] = ir_op(ir, "add", "i32", ir.s.r[4], "4")
		// (pop [mem]: the address with the esp after the pop)
		ir_write(ir, ops[0], ops[0].kind == .MEMORY ? ir_addr(ir, ops[0].mem) : "", v)
	case .LEAVE:
		ir.s.r[4] = ir.s.r[5]
		v := ir_load(ir, ir.s.r[4], 32)
		ir.s.r[4] = ir_op(ir, "add", "i32", ir.s.r[4], "4")
		ir.s.r[5] = v
	case .XCHG:
		if ea != "" && !ir_feat(ir, "xchgm") {return}
		a := ir_read(ir, ops[0], ea)
		b := ir_read(ir, ops[1], ea)
		ir_write(ir, ops[0], ea, b)
		ir_write(ir, ops[1], ea, a)
	case .BT, .BTS, .BTR, .BTC:
		if ops[0].kind == .MEMORY && ops[1].kind != .IMMEDIATE {
			if !ir_feat(ir, "btm") {return}
			// a register bit offset reaches past the operand: the dword at ea + 4 * (offset >> 5) (signed)
			if bits != 32 {ir_fail(ir, "bt16 m,r"); return}
			o := ir_read(ir, ops[1], ea)
			pa := ir_op(ir, "add", "i32", ea, ir_op(ir, "shl", "i32", ir_op(ir, "ashr", "i32", o, "5"), "2"))
			k := ir_op(ir, "and", "i32", o, "31")
			v := ir_load(ir, pa, 32)
			ir.s.f[IR_CF] = ir_cast(ir, "trunc", "i32", ir_op(ir, "lshr", "i32", v, k), "i1")
			if m != .BT {
				bit := ir_op(ir, "shl", "i32", "1", k)
				r := ""
				#partial switch m {
				case .BTS: r = ir_op(ir, "or", "i32", v, bit)
				case .BTR: r = ir_op(ir, "and", "i32", v, ir_op(ir, "xor", "i32", bit, "-1"))
				case .BTC: r = ir_op(ir, "xor", "i32", v, bit)
				}
				ir_store(ir, pa, 32, r)
			}
			return
		}
		a := ir_read(ir, ops[0], ea)
		k := ops[1].kind == .IMMEDIATE ? ir_num(i64(ops[1].immediate & i64(bits - 1))) : ir_op(ir, "and", "i32", ir_read(ir, ops[1], ea), ir_num(i64(bits - 1)))
		ir.s.f[IR_CF] = ir_cast(ir, "trunc", "i32", ir_op(ir, "lshr", "i32", a, k), "i1")
		if m != .BT {
			bit := ir_op(ir, "shl", "i32", "1", k)
			r := ""
			#partial switch m {
			case .BTS: r = ir_op(ir, "or", "i32", a, bit)
			case .BTR: r = ir_op(ir, "and", "i32", a, ir_op(ir, "xor", "i32", bit, "-1"))
			case .BTC: r = ir_op(ir, "xor", "i32", a, bit)
			}
			ir_write(ir, ops[0], ea, r)
		}
	case .BSF, .BSR:
		if bits != 32 {ir_fail(ir, "bsf16"); return}
		s := ir_read(ir, ops[1], ea)
		z := ir_cmp(ir, "eq", "i32", s, "0")
		d := ""
		if m == .BSF {
			d = ir_v(ir)
			ir_w(ir, d, " = call i32 @llvm.cttz.i32(i32 ", s, ", i1 false)")
		} else {
			lz := ir_v(ir)
			ir_w(ir, lz, " = call i32 @llvm.ctlz.i32(i32 ", s, ", i1 false)")
			d = ir_op(ir, "sub", "i32", "31", lz)
		}
		old := ir_read(ir, ops[0], "")
		ir_write(ir, ops[0], "", ir_sel(ir, "i32", z, old, d))
		ir.s.f[IR_ZF] = z
	case .BSWAP:
		if ops[0].kind != .REGISTER || bits != 32 {ir_fail(ir, "bswap"); return}
		ir_write(ir, ops[0], "", ir_call1(ir, "i32", "llvm.bswap.i32", "i32", ir_read(ir, ops[0], "")))
	case .DIV, .IDIV:
		if !ir_feat(ir, "div") {return}
		// edx:eax / v (xdiv32 / xidiv32: a zero divisor, or idiv's INT_MIN / -1, leaves both alone -- x86 raises #DE)
		if n != 1 || bits != 32 {ir_fail(ir, "div16"); return}
		v := ir_read(ir, ops[0], ea)
		sg := m == .IDIV
		hi := ir_cast(ir, "zext", "i32", ir.s.r[2], "i64")
		dd := ir_op(ir, "or", "i64", ir_op(ir, "shl", "i64", hi, "32"), ir_cast(ir, "zext", "i32", ir.s.r[0], "i64"))
		z := ir_cmp(ir, "eq", "i32", v, "0")
		if sg {
			mn1 := ir_cmp(ir, "eq", "i32", v, "-1")
			z = ir_op(ir, "or", "i1", z, ir_op(ir, "and", "i1", mn1, ir_cmp(ir, "eq", "i64", dd, "-9223372036854775808")))
			// (the 32-bit INT_MIN / -1 too: xidiv32 leaves it)
			small := ir_cmp(ir, "eq", "i32", ir.s.r[2], ir_op(ir, "ashr", "i32", ir.s.r[0], "31"))
			z = ir_op(ir, "or", "i1", z, ir_op(ir, "and", "i1", ir_op(ir, "and", "i1", small, mn1), ir_cmp(ir, "eq", "i32", ir.s.r[0], "-2147483648")))
		}
		dv := ir_sel(ir, "i64", z, "1", ir_cast(ir, sg ? "sext" : "zext", "i32", v, "i64"))
		q := ir_cast(ir, "trunc", "i64", ir_op(ir, sg ? "sdiv" : "udiv", "i64", dd, dv), "i32")
		r := ir_cast(ir, "trunc", "i64", ir_op(ir, sg ? "srem" : "urem", "i64", dd, dv), "i32")
		ir.s.r[0] = ir_sel(ir, "i32", z, ir.s.r[0], q)
		ir.s.r[2] = ir_sel(ir, "i32", z, ir.s.r[2], r)
	case .XADD:
		if !ir_feat(ir, "xadd") {return}
		// (not atomic: guest threads never run at once)
		if n != 2 {ir_fail(ir, "xadd"); return}
		a := ir_read(ir, ops[0], ea)
		b := ir_read(ir, ops[1], ea)
		r := ir_alu(ir, .ADD, a, b, bits)
		ir_write(ir, ops[1], ea, a)
		ir_write(ir, ops[0], ea, ir_zx(ir, r, bits))
	case .CMPXCHG:
		if !ir_feat(ir, "cmpxchg") {return}
		acc := x86.op_reg(bits == 8 ? x86.AL : bits == 16 ? x86.AX : x86.EAX)
		av := ir_read(ir, acc, "")
		dv := ir_read(ir, ops[0], ea)
		sv := ir_read(ir, ops[1], ea)
		r := ir_alu(ir, .CMP, av, dv, bits)
		_ = r
		eq := ir.s.f[IR_ZF]
		// (the destination is written either way, as x86 does: its old value when unequal)
		ir_write(ir, ops[0], ea, ir_sel(ir, "i32", eq, sv, dv))
		ir_write(ir, acc, "", ir_sel(ir, "i32", eq, av, dv))
	case .SHLD, .SHRD:
		if !ir_feat(ir, "shld") {return}
		if n != 3 || bits != 32 {ir_fail(ir, "shld16"); return}
		a := ir_read(ir, ops[0], ea)
		b := ir_read(ir, ops[1], ea)
		c := ir_read(ir, ops[2], ea)
		k := ir_op(ir, "and", "i32", c, "31")
		z := ir_cmp(ir, "eq", "i32", k, "0")
		rr, cf := "", ""
		if m == .SHLD {
			rr = ir_v(ir)
			ir_w(ir, rr, " = call i32 @llvm.fshl.i32(i32 ", a, ", i32 ", b, ", i32 ", k, ")")
			cf = ir_cast(ir, "trunc", "i32", ir_op(ir, "lshr", "i32", a, ir_op(ir, "and", "i32", ir_op(ir, "sub", "i32", "32", k), "31")), "i1")
		} else {
			rr = ir_v(ir)
			ir_w(ir, rr, " = call i32 @llvm.fshr.i32(i32 ", b, ", i32 ", a, ", i32 ", k, ")")
			cf = ir_cast(ir, "trunc", "i32", ir_op(ir, "lshr", "i32", a, ir_op(ir, "and", "i32", ir_op(ir, "sub", "i32", k, "1"), "31")), "i1")
		}
		old := ir.s.f
		ir_zsp(ir, rr, 32)
		nf := ir.s.f
		nf[IR_CF], nf[IR_OF] = cf, "false"
		for q in 0 ..< 5 {ir.s.f[q] = ir_sel(ir, "i1", z, old[q], nf[q])}
		ir_write(ir, ops[0], ea, ir_sel(ir, "i32", z, a, rr))
	case .LAHF:
		if !ir_feat(ir, "flags") {return}
		f := &ir.s.f
		v := ir_op(ir, "or", "i32", ir_op(ir, "shl", "i32", ir_cast(ir, "zext", "i1", f[IR_SF], "i32"), "7"), ir_op(ir, "shl", "i32", ir_cast(ir, "zext", "i1", f[IR_ZF], "i32"), "6"))
		v = ir_op(ir, "or", "i32", v, ir_op(ir, "shl", "i32", ir_cast(ir, "zext", "i1", f[IR_PF], "i32"), "2"))
		v = ir_op(ir, "or", "i32", ir_op(ir, "or", "i32", v, "2"), ir_cast(ir, "zext", "i1", f[IR_CF], "i32"))
		ir_reg_write(ir, x86.AH, v)
	case .SAHF:
		if !ir_feat(ir, "flags") {return}
		h := ir_op(ir, "lshr", "i32", ir.s.r[0], "8")
		bit :: proc(ir: ^Ir, h: string, k: int) -> string {return ir_cast(ir, "trunc", "i32", ir_op(ir, "lshr", "i32", h, ir_num(i64(k))), "i1")}
		ir.s.f[IR_SF] = bit(ir, h, 7)
		ir.s.f[IR_ZF] = bit(ir, h, 6)
		ir.s.f[IR_PF] = bit(ir, h, 2)
		ir.s.f[IR_CF] = bit(ir, h, 0)
	case .PUSHF, .PUSHFD:
		if !ir_feat(ir, "flags") {return}
		if ins.p66 {ir_fail(ir, "pushf16"); return}
		ir.s.r[4] = ir_op(ir, "sub", "i32", ir.s.r[4], "4")
		ir_store(ir, ir.s.r[4], 32, ir_eflags(ir))
	case .POPF, .POPFD:
		if !ir_feat(ir, "flags") {return}
		if ins.p66 {ir_fail(ir, "popf16"); return}
		v := ir_load(ir, ir.s.r[4], 32)
		ir.s.r[4] = ir_op(ir, "add", "i32", ir.s.r[4], "4")
		ir_cstore(ir, IR_OFF_EFLSYS, "i32", ir_op(ir, "or", "i32", ir_op(ir, "and", "i32", v, "-2262"), "2"))
		bit :: proc(ir: ^Ir, h: string, k: int) -> string {return ir_cast(ir, "trunc", "i32", ir_op(ir, "lshr", "i32", h, ir_num(i64(k))), "i1")}
		ir.s.f[IR_CF] = bit(ir, v, 0)
		ir.s.f[IR_PF] = bit(ir, v, 2)
		ir.s.f[IR_ZF] = bit(ir, v, 6)
		ir.s.f[IR_SF] = bit(ir, v, 7)
		ir.s.f[IR_OF] = bit(ir, v, 11)
	case .CLD, .STD:
		if !ir_feat(ir, "flags") {return}
		sys := ir_cload(ir, IR_OFF_EFLSYS, "i32")
		ir_cstore(ir, IR_OFF_EFLSYS, "i32", m == .CLD ? ir_op(ir, "and", "i32", sys, "-1025") : ir_op(ir, "or", "i32", sys, "1024"))
	case .CLC: ir.s.f[IR_CF] = "false"
	case .STC: ir.s.f[IR_CF] = "true"
	case .CMC: ir.s.f[IR_CF] = ir_not1(ir, ir.s.f[IR_CF])
	case .CLI, .STI, .HLT: ir_ud(ir, ins.addr) // (privileged: #GP in user mode)
	case .INT3:
	case .CPUID:
		if !ir_feat(ir, "cpuid") {return}
		for k in ([4]int{0, 1, 2, 3}) {ir_cstore(ir, k * 4, "i32", ir.s.r[k])}
		ir_w(ir, "call void @xrc_ir_cpuid(ptr %c)")
		for k in ([4]int{0, 1, 2, 3}) {ir.s.r[k] = ir_cload(ir, k * 4, "i32")}
	case .MOVS, .MOVSB, .MOVSW, .MOVSD, .STOS, .STOSB, .STOSW, .STOSD, .LODS, .LODSB, .LODSW, .LODSD,
	     .SCAS, .SCASB, .SCASW, .SCASD, .CMPS, .CMPSB, .CMPSW, .CMPSD:
		// (movsd / cmpsd with 0F: the SSE2 instructions of the same mnemonic)
		if ins.opc == 0x0f {
			if !ir_sse(ir, ins, ea) {ir_fail(ir, rx_mn(ins))}
			return
		}
		if !ir_feat(ir, "rep") {return}
		ir_rep(ir, ins, bits)
	case:
		if cc := x_cc(m); cc >= 0 {
			if rx_is_jcc(m) {ir_fail(ir, "jcc"); return}
			c := ir_cond(ir, cc)
			if is_setcc(ins) {
				ir_write(ir, ops[0], ea, ir_cast(ir, "zext", "i1", c, "i32"))
				return
			}
			// cmovcc (the source read even when false, as x86 does)
			v := ir_read(ir, ops[1], ea)
			old := ir_read(ir, ops[0], "")
			ir_write(ir, ops[0], "", ir_sel(ir, "i32", c, v, old))
			return
		}
		if ir_sse(ir, ins, ea) {return}
		ir_fail(ir, rx_mn(ins))
	}
}
op_imm32 :: proc(op: x86.Operand) -> u32 {return u32(i32(op.immediate))}
is_setcc :: proc(ins: ^Rx_Ins) -> bool {return strings.has_prefix(rx_mn(ins), "set")}

// ---- one function ----

// whether the C translation shows something this backend leaves to C
ir_eligible :: proc(xl: ^Xlat) -> (bool, string) {
	if !xl.ok {return false, "not translated"}
	if len(xl.slots) > 0 {return false, "slots"}
	return true, ""
}

// the state component k (registers, flags, named xmm, x87 slots, fsw, fcw): its value, type, phi name prefix
ir_comp :: proc(s: ^Ir_St, k: int) -> (^string, string, string) {
	switch {
	case k < 8: return &s.r[k], "i32", "%r"
	case k < 13: return &s.f[k - 8], "i1", "%f"
	case k < 21: return &s.x[k - 13], V4I, "%x"
	case k < 29: return &s.st[k - 21], "double", "%s"
	case k == 29: return &s.fsw, "i32", "%w"
	case k == 30: return &s.fcw, "i32", "%k"
	}
	return &s.mm[k - 31], "i64", "%m"
}
ir_comp_on :: proc(ir: ^Ir, k: int) -> bool {
	switch {
	case k < 13: return true
	case k < 21: return ir.named[k - 13]
	case k < 29: return k - 21 < ir.nst
	case k < 31: return ir.fp
	}
	return ir.mmx
}
ir_comp_idx :: proc(k: int) -> int {
	switch {
	case k < 8: return k
	case k < 13: return k - 8
	case k < 21: return k - 13
	case k < 29: return k - 21
	case k < 31: return 0
	}
	return k - 31
}

ir_translate :: proc(mod: ^Ir_Mod, xl: ^Xlat, a: u32, size: u32, copy := false) -> (string, bool, string) {
	ins, bad := rx_function(a, size, context.temp_allocator)
	if bad != 0 {return "", false, "decode"}
	body := make(map[u32]bool, len(ins), context.temp_allocator)
	for &i in ins {body[i.addr] = true}
	if a not_in body {return "", false, "entry"}
	ir := Ir{a = a, ok = true, mod = mod, xl = xl}
	ir.nst = xl.fmax
	ir.fp = xl.uses_fp || xl.fret || xl.fmax > 0
	if xl.fret && ir.nst < 1 {ir.nst = 1}
	leaders := make(map[u32]bool, context.temp_allocator)
	leaders[a] = true
	for &i, k in ins {
		mm := i.inst.mnemonic
		for o in i.inst.ops[:i.inst.operand_count] {if ir_is_x(o) {ir.named[ir_xn(o)] = true}}
		#partial switch i.xop {
		case .CVTPI2PS: ir.named[i.xr & 7] = true; ir.mmx = true
		case .CVTTPS2PI, .CVTPS2PI: if !i.xmem {ir.named[i.xrm & 7] = true}; ir.mmx = true
		case .MOVQ_LD, .MOVQ_ST, .MOVD_LD, .MOVD_ST, .MMXOP: ir.mmx = true
		}
		if tb, okt := xl.tables[i.addr]; okt {for t in tb {if t in body {leaders[t] = true}}}
		if k > 0 && ins[k - 1].addr + ins[k - 1].len != i.addr {leaders[i.addr] = true}
		if rx_is_jcc(mm) || mm == .JMP || mm == .RET || mm == .CALL {
			// (after a call too: not needed for the SSA, but a call's internal blocks then never precede a phi)
			if (rx_is_jcc(mm) || mm == .JMP) && i.inst.operand_count > 0 && i.inst.ops[0].kind == .RELATIVE {
				if t := rx_rel_target(&i, 0); t in body {leaders[t] = true}
			}
			if nx := i.addr + i.len; nx in body {leaders[nx] = true}
		}
	}

	Blk :: struct {addr: u32, text: strings.Builder}
	blks := make([dynamic]Blk, context.temp_allocator)
	ir.edges = make(map[u32][dynamic]Ir_Edge, context.temp_allocator)
	ir.succ = make(map[u32][dynamic]u32, context.temp_allocator)
	ir.copy = copy

	// the entry: the state from XCtx
	entry := strings.builder_make(context.temp_allocator)
	ir.b = &entry
	ir.cur = "entry"
	ir.blk = 0xffffffff
	ir_w(&ir, "%xm = load ptr, ptr @xrc_mem, align 8")
	for k in 0 ..< 8 {ir.s.r[k] = ir_cload(&ir, k * 4, "i32")}
	for k in 0 ..< 5 {ir.s.f[k] = "false"}
	for k in 0 ..< 8 {ir.s.x[k] = "zeroinitializer"; ir.s.st[k] = "0.0"; ir.s.mm[k] = "0"}
	ir_xfill(&ir)
	ir.s.fsw = "0"
	ir.s.fcw = ir.fp ? ir_cload(&ir, IR_OFF_FCW, "i32") : "0"
	ir_edge(&ir, a)
	ir_w(&ir, "br label %", ir_lab(a))

	open := false // a block is open (not ended by a terminator)
	start_blk :: proc(ir: ^Ir, blks: ^[dynamic]Blk, addr: u32) {
		append(blks, Blk{addr = addr, text = strings.builder_make(context.temp_allocator)})
		ir.b = &blks[len(blks) - 1].text
		ir.blk = addr
		ir.cur = ir_lab(addr)
		hx := ir_hex8(addr)
		for k in 0 ..< IR_NCOMP {
			if !ir_comp_on(ir, k) {continue}
			p, _, pre := ir_comp(&ir.s, k)
			p^ = strings.concatenate({pre, ir_num(i64(ir_comp_idx(k))), "_", hx}, context.temp_allocator)
		}
	}
	for &i in ins {
		if i.addr in leaders {
			if open {ir_edge(&ir, i.addr); ir_w(&ir, "br label %", ir_lab(i.addr))}
			start_blk(&ir, &blks, i.addr)
			open = true
		}
		ir.at = i.addr
		if !open {continue} // (not reached by falling through, and no branch target: dead bytes between chunks)
		mm := i.inst.mnemonic
		nx := i.addr + i.len
		ops := i.inst.ops
		if rx_is_jcc(mm) {
			if i.inst.operand_count == 0 || ops[0].kind != .RELATIVE {ir_fail(&ir, "jcc*"); break}
			cc := x_cc(mm)
			if cc < 0 {ir_fail(&ir, rx_mn(&i)); break}
			t := rx_rel_target(&i, 0)
			c := ir_cond(&ir, cc)
			lt := t in body ? ir_lab(t) : ir_newlabel(&ir, "tct")
			lf := nx in body ? ir_lab(nx) : ir_newlabel(&ir, "tcf")
			spin := t in body && i.addr in xl.spin_back
			if spin && !ir_feat(&ir, "spin") {break}
			ipre := t in body && !spin && i.addr in xl.inl_back
			if spin || ipre {
				// a spin loop's back edge: a waiting thread gets the GIL on the taken path (XRC_SPIN_YIELD)
				ls := ir_newlabel(&ir, "spin")
				if nx in body {ir_edge(&ir, nx)}
				ir_w(&ir, "br i1 ", c, ", label %", ls, ", label %", lf)
				saved := ir.s
				ir_label(&ir, ls)
				if spin {ir_spin_yield(&ir)} else {ir_inl_preempt(&ir)}
				ir_edge(&ir, t)
				ir_w(&ir, "br label %", lt)
				ir.s = saved
				if nx not_in body {ir_label(&ir, lf); ir_tail(&ir, &i, nx, "", nx | 0x80000000)}
			} else if t in body && nx in body && t == nx {
				ir_edge(&ir, t)
				ir_w(&ir, "br label %", lt)
			} else {
				if t in body {ir_edge(&ir, t)}
				if nx in body {ir_edge(&ir, nx)}
				ir_w(&ir, "br i1 ", c, ", label %", lt, ", label %", lf)
				saved := ir.s
				if t not_in body {ir_label(&ir, lt); ir_tail(&ir, &i, t, "", i.addr); ir.s = saved}
				if nx not_in body {ir_label(&ir, lf); ir_tail(&ir, &i, nx, "", nx | 0x80000000)}
			}
			open = false
		} else if mm == .JMP {
			if i.inst.operand_count > 0 && ops[0].kind == .RELATIVE {
				t := rx_rel_target(&i, 0)
				if t in body {
					if i.addr in xl.spin_back {if !ir_feat(&ir, "spin") {break}; ir_spin_yield(&ir)} else if i.addr in xl.inl_back {ir_inl_preempt(&ir)}
					ir_edge(&ir, t)
					ir_w(&ir, "br label %", ir_lab(t))
				} else {
					ir_tail(&ir, &i, t, "", i.addr)
				}
			} else if _, is_tbl := xl.tables[i.addr]; is_tbl && ops[0].kind == .MEMORY && ir_feat(&ir, "table") {
				// a switch: the table slot (its address) selects the case -- the table is in .text, which the retail
				// exe has encrypted, so the walker's slots are the cases; a slot it did not see is read and left
				// through as a tail call
				ea := ir_addr(&ir, ops[0].mem)
				ldef := ir_newlabel(&ir, "swdef")
				sw := strings.builder_make(context.temp_allocator)
				seen := make(map[u32]bool, context.temp_allocator)
				for sl in rx_table_slots[i.addr] {
					if sl[0] in seen || sl[1] not_in body {continue}
					seen[sl[0]] = true
					strings.write_string(&sw, " i32 ")
					strings.write_string(&sw, ir_num(i64(i32(sl[0]))))
					strings.write_string(&sw, ", label %")
					strings.write_string(&sw, ir_lab(sl[1]))
					ir_edge(&ir, sl[1]) // (one per case: each is an edge of its own to LLVM's phis)
				}
				ir_w(&ir, "switch i32 ", ea, ", label %", ldef, " [", strings.to_string(sw), " ]")
				ir_label(&ir, ldef)
				ir_tail(&ir, &i, 0, ir_load(&ir, ea, 32), i.addr)
			} else {
				ea := ""
				if ops[0].kind == .MEMORY {ea = ir_addr(&ir, ops[0].mem)}
				ir_tail(&ir, &i, 0, ir_read(&ir, ops[0], ea), i.addr)
			}
			open = false
		} else if mm == .RET {
			nb: u32 = 0
			if i.inst.operand_count > 0 && ops[0].kind == .IMMEDIATE {nb = u32(ops[0].immediate) & 0xffff}
			ir_ret(&ir, nb)
			open = false
		} else if mm == .CALL {
			ea := ""
			if ops[0].kind == .MEMORY {ea = ir_addr(&ir, ops[0].mem)}
			ir_call(&ir, &i, ea)
		} else {
			ir_ins(&ir, &i)
		}
		if !ir.ok {break}
		// falls off the decoded code: a tail call to what follows (x_translate's "falls through")
		if open && nx not_in body {
			ir_tail(&ir, &i, nx, "", nx | 0x80000000)
			open = false
		}
	}
	if !ir.ok {return "", false, ir.why}
	if open {return "", false, "open end"}

	// reachable blocks (from the entry), and the phis from their reachable predecessors
	reach := make(map[u32]bool, context.temp_allocator)
	stack := make([dynamic]u32, context.temp_allocator)
	append(&stack, a)
	reach[0xffffffff] = true
	for len(stack) > 0 {
		x := pop(&stack)
		if x in reach {continue}
		reach[x] = true
		if s, oks := ir.succ[x]; oks {for y in s {if y not_in reach {append(&stack, y)}}}
	}
	out := strings.builder_make(context.temp_allocator)
	strings.write_string(&out, copy ? "define internal void @xi_" : "define void @xf_")
	strings.write_string(&out, ir_hex8(a))
	// (noalias: guest memory never holds an XCtx -- the main thread's is in the executable's data, the others' on host
	// stacks, outside the reserved guest window -- so LLVM may keep XCtx fields in registers across guest stores, and
	// forward a caller's register stores into a callee it inlines: something the C pages, -fno-strict-aliasing,
	// cannot say)
	strings.write_string(&out, copy ? "(ptr noalias %c) alwaysinline nounwind {\nentry:\n" : "(ptr noalias %c) nounwind uwtable {\nentry:\n")
	strings.write_string(&out, strings.to_string(entry))
	for &bk in blks {
		if bk.addr not_in reach {continue}
		hx := ir_hex8(bk.addr)
		strings.write_string(&out, ir_lab(bk.addr))
		strings.write_string(&out, ":\n")
		es := ir.edges[bk.addr]
		for k in 0 ..< IR_NCOMP {
			if !ir_comp_on(&ir, k) {continue}
			_, ty, pre := ir_comp(&ir.s, k)
			strings.write_string(&out, "  ")
			strings.write_string(&out, pre)
			strings.write_string(&out, ir_num(i64(ir_comp_idx(k))))
			strings.write_string(&out, "_")
			strings.write_string(&out, hx)
			strings.write_string(&out, " = phi ")
			strings.write_string(&out, ty)
			strings.write_string(&out, " ")
			first := true
			for &e in es {
				if e.from_blk not_in reach {continue}
				if !first {strings.write_string(&out, ", ")}
				first = false
				v, _, _ := ir_comp(&e.s, k)
				strings.write_string(&out, "[ ")
				strings.write_string(&out, v^)
				strings.write_string(&out, ", %")
				strings.write_string(&out, e.from)
				strings.write_string(&out, " ]")
			}
			strings.write_string(&out, "\n")
		}
		strings.write_string(&out, strings.to_string(bk.text))
	}
	strings.write_string(&out, "}\n\n")
	return strings.to_string(out), true, ""
}

IR_DECLS :: `@xrc_mem = external global ptr
@xrc_gil_waiters = external global i32
declare void @xrc_preempt_n(ptr) cold nounwind
declare void @xrc_bad_return(ptr, i32) cold nounwind
declare void @xrc_ir_icall(ptr, i32, ptr) nounwind
declare void @xrc_dispatch_ic(ptr, i32, ptr) nounwind
declare void @xrc_imp_enter(i32) nounwind
declare void @xrc_ud(ptr, i32) cold nounwind
declare void @xrc_ir_cpuid(ptr) nounwind
declare void @xrc_spin_yield() nounwind
declare double @exp2(double) nounwind
declare double @log2(double) nounwind
declare double @tan(double) nounwind
declare double @sin(double) nounwind
declare double @cos(double) nounwind
declare double @atan2(double, double) nounwind
declare double @fmod(double, double) nounwind
declare double @remainder(double, double) nounwind
declare double @ldexp(double, i32) nounwind
declare i8 @llvm.fshl.i8(i8, i8, i8)
declare i8 @llvm.fshr.i8(i8, i8, i8)
declare i16 @llvm.fshl.i16(i16, i16, i16)
declare i16 @llvm.fshr.i16(i16, i16, i16)
declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare i8 @llvm.ctpop.i8(i8)
declare i32 @llvm.cttz.i32(i32, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i32 @llvm.bswap.i32(i32)
declare i32 @llvm.fshl.i32(i32, i32, i32)
declare i32 @llvm.fshr.i32(i32, i32, i32)
declare double @llvm.fabs.f64(double)
declare double @llvm.sqrt.f64(double)
declare float @llvm.sqrt.f32(float)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)
declare <2 x double> @llvm.sqrt.v2f64(<2 x double>)
declare double @llvm.roundeven.f64(double)
declare double @llvm.floor.f64(double)
declare double @llvm.ceil.f64(double)
declare double @llvm.trunc.f64(double)
declare <2 x double> @llvm.roundeven.v2f64(<2 x double>)
declare <2 x double> @llvm.floor.v2f64(<2 x double>)
declare <2 x double> @llvm.ceil.v2f64(<2 x double>)
declare <2 x double> @llvm.trunc.v2f64(<2 x double>)
declare <4 x double> @llvm.roundeven.v4f64(<4 x double>)
declare <4 x double> @llvm.floor.v4f64(<4 x double>)
declare <4 x double> @llvm.ceil.v4f64(<4 x double>)
declare <4 x double> @llvm.trunc.v4f64(<4 x double>)
declare <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.usub.sat.v8i16(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.smax.v8i16(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.smin.v8i16(<8 x i16>, <8 x i16>)
`

// the module around a page's functions
ir_flush_page :: proc(dir: string, page: u32, mod: ^Ir_Mod) {
	name := fmt.tprintf("%s/page_%04x.ll", dir, page)
	if mod.n_fn == 0 {
		if os.exists(name) {os.remove(name)}
	} else {
		sb := strings.builder_make(context.temp_allocator)
		fmt.sbprintf(&sb, "; page %04x0000: x86 -> LLVM IR (xrc --llvm). Generated; do not edit.\n", page)
		strings.write_string(&sb, IR_DECLS)
		for t in mod.copies {
			cp := ir_copies[t]
			for d in cp.decls {mod.decls[d] = true}
			strings.write_string(&mod.globals, cp.globals)
			strings.write_string(&mod.body, cp.text)
		}
		for d in mod.decls {
			if d in mod.defs {continue}
			strings.write_string(&sb, "declare void @")
			strings.write_string(&sb, d)
			strings.write_string(&sb, "(ptr)\n")
		}
		strings.write_string(&sb, strings.to_string(mod.globals))
		strings.write_string(&sb, "\n")
		strings.write_string(&sb, strings.to_string(mod.body))
		// (the slow paths: a waiting thread's preemption, a bad return)
		strings.write_string(&sb, "!0 = !{!\"branch_weights\", i32 1, i32 4000}\n")
		text := strings.to_string(sb)
		old, had := read_file(name)
		same := had && string(old) == text
		if had {delete(old)}
		if !same {_ = os.write_entire_file(name, transmute([]u8)text)}
	}
	clear(&mod.decls)
	clear(&mod.defs)
	clear(&mod.copies)
	strings.builder_reset(&mod.globals)
	strings.builder_reset(&mod.body)
	mod.n_fn = 0
}

// in cmd_xrc's loop: the function as IR when it can be (true: the C page leaves it out)
ir_try :: proc(mod: ^Ir_Mod, xl: ^Xlat, a: u32, size: u32) -> bool {
	el, why := ir_eligible(xl)
	text, ok := "", false
	if el {text, ok, why = ir_translate(mod, xl, a, size)}
	if !ok {
		ir_n_c += 1
		ir_reasons[strings.clone(why)] += 1
		return false
	}
	ir_n_fns += 1
	mod.n_fn += 1
	mod.defs[strings.clone(fmt.tprintf("xf_%08x", a))] = true
	strings.write_string(&mod.body, text)
	return true
}

ir_report :: proc() {
	fmt.printf("xrc: --llvm: %d functions as LLVM IR, %d stay C\n", ir_n_fns, ir_n_c)
	Rc :: struct {why: string, n: int}
	v := make([dynamic]Rc, context.temp_allocator)
	for k, n in ir_reasons {append(&v, Rc{k, n})}
	slice.sort_by(v[:], proc(x, y: Rc) -> bool {return x.n > y.n})
	for r, k in v {
		if k >= 80 {break}
		fmt.printf("xrc:   %6d %s\n", r.n, r.why)
	}
}
