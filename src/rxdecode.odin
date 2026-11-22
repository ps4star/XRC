package xrc

// x86 (i386) decoding with core:rexcode (with the i386 fixes of patches/odin-rexcode-i386.patch).
//
//   rx_decode_at(addr)  one instruction at an exe address, normalized for a translator:
//     * implicit accumulators are explicit operands (rexcode leaves AL/AX/EAX of the short forms implied so they
//       re-encode short): `add eax, imm32` (05 id) has ops [eax, imm]; `xchg eax, r` (90+r) has [eax, r]
//     * moffs moves (A0-A3) are `mov al/ax/eax, [abs]` / `mov [abs], al/ax/eax` with a memory operand
//     * x87 register forms (D8-DF with mod=11) have `sti` = the ST(i) of the ModR/M byte (rexcode prints the
//       st(1) forms like `fxch` / `fmulp` without operands); -1 otherwise
//     * `seg` = the segment override (FS for fs:[...]), NONE if none
//   ff rx-probe [--list ADDRS] [--vs-capstone]  decodes the reachable instructions of functions and reports counts,
//     mnemonics and decode failures; --vs-capstone decodes every function linearly with rexcode and capstone and
//     reports where the instruction lengths or mnemonics disagree.

import "base:runtime"
import "core:fmt"
import "core:os"
import "core:slice"
import "core:strings"
import x86 "core:rexcode/isa/x86"

Rx_Ins :: struct {
	addr:   u32,
	len:    u32,
	inst:   x86.Instruction,
	info:   x86.Instruction_Info,
	opc:    u8,           // the first opcode byte after the legacy prefixes (0x0f for two-byte opcodes)
	opc2:   u8,           // the second opcode byte when opc == 0x0f
	modrm:  u8,           // the byte after the opcode (ModR/M when the form has one)
	p66:    bool,
	sti:    i8,
	seg:    x86.Register,
	xop:    Rx_Xop, // an instruction rexcode has no row for, decoded by substitution (inst.mnemonic is .NOP)
	xr:     u8,     // xop: ModR/M reg field
	xrm:    u8,     // xop: ModR/M rm field (register forms)
	xmem:   bool,   // xop: the rm operand is memory (inst.ops[0], size meaningless)
	ximm:   u8,     // xop SHIFT_IMM: the count
}

Rx_Xop :: enum u8 {
	None,
	LDMXCSR, // 0F AE /2 m32
	STMXCSR, // 0F AE /3 m32
	// MMX (no 66 / F2 / F3 prefix): the conversions and moves MSVC emits for the _mm_cvt(t)ps_pi32 intrinsics
	CVTPI2PS,  // 0F 2A /r  xmm, mm/m64
	CVTTPS2PI, // 0F 2C /r  mm, xmm/m64
	CVTPS2PI,  // 0F 2D /r  mm, xmm/m64
	MOVQ_LD,   // 0F 6F /r  mm, mm/m64
	MOVQ_ST,   // 0F 7F /r  mm/m64, mm
	EMMS,      // 0F 77
	MMXOP,     // 0F <opc2> /r  mm, mm/m64: pand pandn por pxor paddd psubd paddq psubq (opc2 says which)
	MOVD_LD,   // 0F 6E /r  mm, r/m32
	MOVD_ST,   // 0F 7E /r  r/m32, mm
	SHIFT_IMM, // 0F 71 / 72 / 73 /r ib  mm, imm8 (psrlw psraw psllw psrld psrad pslld psrlq psllq); with 66 only 73 /3 /7 (xmm:
	           // psrldq pslldq, which rexcode has no rows for). The count is in ximm, the register in xrm.
	UD,        // an invalid encoding (ud2, FE /2-7, FF /7, FF /3 or /5 with a register): raises #UD
}

@(thread_local)
rx_insts: [dynamic]x86.Instruction
@(thread_local)
rx_infos: [dynamic]x86.Instruction_Info
@(thread_local)
rx_labels: [dynamic]x86.Label_Definition
@(thread_local)
rx_errs: [dynamic]x86.Error

rx_is_legacy_prefix :: proc(b: u8) -> bool {
	switch b {
	case 0x66, 0x67, 0xf0, 0xf2, 0xf3, 0x26, 0x2e, 0x36, 0x3e, 0x64, 0x65: return true
	}
	return false
}

// The first instruction of `data` the way x86.decode's first pass decodes it (decoder.odin, `PASS 1`), without the
// rest of the buffer: x86.decode goes on through all of a 15-byte window (~4 instructions, kept for nothing) and
// allocates its branch list on every call. Its second pass (labels) changes no instruction. XRC_RX_SLOW=1: x86.decode.
rx_decode_one :: proc(data: []u8) -> (inst: x86.Instruction, info: x86.Instruction_Info, ok: bool) {
	if len(data) == 0 {return}
	state := x86.Decoder_State{data = data, position = 0, mode = ._32, segment = x86.NONE}
	err := x86.decode_prefixes(&state)
	if state.has_lock && state.position >= len(state.data) { // a lone lock prefix
		inst.mnemonic = .LOCK
		inst.length = u8(state.position)
		info.rep = .NONE
		info.has_lock = true
		return inst, info, true
	}
	if err != .NONE {return}
	if state.position < len(state.data) { // i386 inc / dec r (0x40-0x4f)
		b := state.data[state.position]
		if b >= 0x40 && b <= 0x4F {
			state.position += 1
			reg_num := b & 0x07
			is_dec := (b & 0x08) != 0
			reg: x86.Register = state.prefix_66 ? x86.gpr16_from_num(reg_num) : x86.gpr32_from_num(reg_num)
			inst.mnemonic = is_dec ? .DEC : .INC
			inst.operand_count = 1
			inst.ops[0] = x86.op_reg(reg)
			inst.length = u8(state.position)
			inst.flags.lock = state.has_lock
			inst.flags.rep = state.prefix_f2 ? .REPNE : (state.prefix_f3 ? .REP : .NONE)
			inst.flags.addr32 = state.prefix_67
			inst.flags.segment = state.segment != x86.NONE ? u8(x86.reg_hw(state.segment)) + 1 : 0
			info.has_lock = state.has_lock
			info.rep = inst.flags.rep
			info.segment = state.segment
			return inst, info, true
		}
	}
	entry, vex_entry, err2 := x86.decode_opcode(&state)
	if err2 != .NONE {return}
	if vex_entry != nil {
		inst, err = x86.decode_operands_vex(&state, vex_entry)
	} else if entry != nil {
		inst, err = x86.decode_operands(&state, entry)
	} else {
		return
	}
	if err != .NONE {return}
	inst.length = u8(state.position)
	inst.flags.lock = state.has_lock
	mand_pfx: u8 = entry != nil ? entry.flags.prefix : 0
	rep_f2 := state.prefix_f2 && mand_pfx != 3
	rep_f3 := state.prefix_f3 && mand_pfx != 2
	inst.flags.rep = rep_f2 ? .REPNE : (rep_f3 ? .REP : .NONE)
	inst.flags.addr32 = state.prefix_67
	inst.flags.segment = state.segment != x86.NONE ? u8(x86.reg_hw(state.segment)) + 1 : 0
	info.rex = state.rex
	info.has_lock = state.has_lock
	info.rep = inst.flags.rep
	info.segment = state.segment
	info.vex_type = state.vex_type
	if state.vex_type != nil && vex_entry != nil {
		if vex_entry.vex_l == .LIG {
			info.vex_l = .LIG
		} else {
			info.vex_l = state.vex_l == 0 ? .L0 : (state.vex_l == 1 ? .L1 : .L2)
		}
		if vex_entry.vex_w == .WIG {
			info.vex_w = .WIG
		} else {
			info.vex_w = state.vex_w ? .W1 : .W0
		}
		info.evex_b = state.evex_b
		info.evex_z = state.evex_z
		info.opmask = state.evex_aaa
	} else if state.vex_type != nil {
		info.vex_l = state.vex_l == 0 ? .L0 : (state.vex_l == 1 ? .L1 : .L2)
		info.vex_w = state.vex_w ? .W1 : .W0
		info.evex_b = state.evex_b
		info.evex_z = state.evex_z
		info.opmask = state.evex_aaa
	}
	return inst, info, true
}

@(private = "file")
rx_slow: int // XRC_RX_SLOW: 0 not looked up yet, 1 rx_decode_one, 2 x86.decode

// decodes the bytes of one instruction (code may continue past it)
rx_decode_bytes :: proc(code: []u8, addr: u32) -> (Rx_Ins, bool) {
	if len(code) == 0 {return {}, false}
	if r, ok, hit := rx_decode_subst(code, addr); hit {return r, ok}
	// lock xchg r/m, r (F0 86 / 87): rexcode drops the memory operand with the lock prefix. xchg with memory is atomic
	// without it, so the prefix changes nothing: decode the rest, one byte longer
	if len(code) >= 3 && code[0] == 0xf0 && (code[1] == 0x86 || code[1] == 0x87) {
		r, ok := rx_decode_bytes(code[1:], addr)
		if ok {r.len += 1; r.inst.length += 1}
		return r, ok
	}
	if rx_slow == 0 {rx_slow = os.get_env("XRC_RX_SLOW", context.temp_allocator) != "" ? 2 : 1}
	r: Rx_Ins
	if rx_slow == 1 {
		// one instruction is at most 15 bytes
		inst, info, ok := rx_decode_one(code[:min(len(code), 15)])
		if !ok {return {}, false}
		r = Rx_Ins{addr = addr, len = u32(inst.length), inst = inst, info = info, sti = -1}
	} else {
		clear(&rx_insts); clear(&rx_infos); clear(&rx_labels); clear(&rx_errs)
		// rexcode decodes a whole buffer; one instruction is at most 15 bytes, so the first decoded one is ours
		_, _ = x86.decode(code[:min(len(code), 15)], nil, &rx_insts, &rx_infos, &rx_labels, &rx_errs, ._32)
		if len(rx_insts) == 0 {return {}, false}
		if len(rx_errs) > 0 && rx_errs[0].inst_idx == 0 {return {}, false}
		r = Rx_Ins{addr = addr, len = u32(rx_insts[0].length), inst = rx_insts[0], info = rx_infos[0], sti = -1}
	}
	if r.inst.mnemonic == .INVALID || r.len == 0 {return {}, false}
	r.seg = r.info.segment
	p := 0
	for p < int(r.len) && rx_is_legacy_prefix(code[p]) {
		if code[p] == 0x66 {r.p66 = true}
		p += 1
	}
	if p >= int(r.len) {return r, true} // a lone prefix (lock)
	r.opc = code[p]
	if r.opc == 0x0f && p + 1 < int(r.len) {r.opc2 = code[p + 1]; p += 1}
	if p + 1 < len(code) {r.modrm = code[p + 1]}
	rx_normalize(&r)
	// an fs: / gs: prefix onto the memory operand (rexcode keeps it in info.segment only): fs:[...] is the TEB in Windows
	// code (the SEH chain at fs:[0], the TLS array at fs:[0x2c]), gs:[...] TLS in Linux code
	if r.seg == x86.FS || r.seg == x86.GS {
		s := x86.mem_make(x86.NONE, x86.NONE, 1, 0, r.seg).segment
		for k in 0 ..< int(r.inst.operand_count) {
			if r.inst.ops[k].kind == .MEMORY && r.inst.ops[k].mem.segment == 0 {r.inst.ops[k].mem.segment = s}
		}
	}
	return r, true
}

// Instructions rexcode has no rows for, decoded by substitution: the same ModR/M (+SIB +disp) after 8B
// (mov r32, m32) decodes to the same memory operand, one byte shorter (two for 0F xx), and the result keeps just that
// operand with inst.mnemonic = .NOP and the instruction in xop:
//   ldmxcsr / stmxcsr (0F AE /2, /3 m32), the MMX forms of Rx_Xop, and invalid encodings (.UD; what the CPU raises
//   #UD for: code that is not reached, or reached only to fault). D0-D3 / C0 / C1 /6 (the undocumented alias of
//   shl, /4) decodes as /4.
rx_decode_subst :: proc(code: []u8, addr: u32) -> (r: Rx_Ins, ok: bool, hit: bool) {
	p := 0
	sse_pfx := false
	for p < len(code) && p < 4 && rx_is_legacy_prefix(code[p]) {
		if code[p] == 0x66 || code[p] == 0xf2 || code[p] == 0xf3 {sse_pfx = true}
		p += 1
	}
	if p + 1 >= len(code) {return}
	op := code[p]
	ud :: proc(addr: u32, n: int) -> (Rx_Ins, bool, bool) {
		return Rx_Ins{addr = addr, len = u32(n), sti = -1, xop = .UD, inst = {mnemonic = .NOP, length = u8(n)}}, true, true
	}
	// a memory form via the 8B substitution (ModR/M at code[mp]); prefix bytes kept (segment overrides)
	memform :: proc(code: []u8, p, mp: int, addr: u32) -> (r: Rx_Ins, ok: bool) {
		tmp: [16]u8
		n := copy(tmp[:], code[:p])
		tmp[n] = 0x8b
		n += 1
		n += copy(tmp[n:], code[mp:min(len(code), 15)])
		r, ok = rx_decode_bytes(tmp[:n], addr)
		if !ok || r.inst.operand_count != 2 || r.inst.ops[1].kind != .MEMORY {return {}, false}
		r.len += u32(mp - p - 1)
		r.inst.ops[0] = r.inst.ops[1]
		r.inst.operand_count = 1
		r.inst.mnemonic = .NOP
		return r, true
	}
	switch op {
	case 0xc0, 0xc1, 0xd0, 0xd1, 0xd2, 0xd3:
		if (code[p + 1] >> 3) & 7 != 6 {return}
		tmp: [16]u8
		n := copy(tmp[:], code[:min(len(code), 15)])
		tmp[p + 1] = (tmp[p + 1] & 0xc7) | (4 << 3)
		hit = true
		r, ok = rx_decode_bytes(tmp[:n], addr)
		return
	case 0xfe:
		if (code[p + 1] >> 3) & 7 >= 2 {return ud(addr, p + 2)}
		return
	case 0xff:
		g := (code[p + 1] >> 3) & 7
		if g == 7 || ((g == 3 || g == 5) && code[p + 1] >= 0xc0) {return ud(addr, p + 2)}
		return
	case 0x0f:
	case: return
	}
	o2 := code[p + 1]
	if o2 == 0x0b {return ud(addr, p + 2)}
	if o2 == 0x77 && !sse_pfx {
		return Rx_Ins{addr = addr, len = u32(p + 2), sti = -1, xop = .EMMS, opc = 0x0f, opc2 = 0x77, inst = {mnemonic = .NOP, length = u8(p + 2)}}, true, true
	}
	if p + 2 >= len(code) {return}
	m := code[p + 2]
	reg := (m >> 3) & 7
	x := Rx_Xop.None
	switch o2 {
	case 0xae: if m < 0xc0 && (reg == 2 || reg == 3) {x = reg == 2 ? .LDMXCSR : .STMXCSR}
	case 0x2a: if !sse_pfx {x = .CVTPI2PS}
	case 0x2c: if !sse_pfx {x = .CVTTPS2PI}
	case 0x2d: if !sse_pfx {x = .CVTPS2PI}
	case 0x6f: if !sse_pfx {x = .MOVQ_LD}
	case 0x7f: if !sse_pfx {x = .MOVQ_ST}
	case 0x6e: if !sse_pfx {x = .MOVD_LD}
	case 0x7e: if !sse_pfx {x = .MOVD_ST}
	case 0xdb, 0xdf, 0xeb, 0xef, 0xfe, 0xfa, 0xd4, 0xfb: if !sse_pfx {x = .MMXOP}
	case 0x71, 0x72, 0x73:
		// (a register form with its count byte; the 66 forms other than psrldq / pslldq are rexcode's)
		if m >= 0xc0 && p + 3 < len(code) && (!sse_pfx || (code[p - 1] == 0x66 && o2 == 0x73 && (reg == 3 || reg == 7))) {
			if _, ok2 := simd_shift_imm_assigns(o2, reg, 0, sse_pfx ? 16 : 8); ok2 {
				hit = true
				r = Rx_Ins{addr = addr, len = u32(p + 4), sti = -1, inst = {mnemonic = .NOP, length = u8(p + 4)}, p66 = sse_pfx}
				r.opc, r.opc2, r.modrm = 0x0f, o2, m
				r.xop, r.xr, r.xrm, r.ximm = .SHIFT_IMM, reg, m & 7, code[p + 3]
				return r, true, true
			}
		}
	case:
		// the other MMX integer ops (unpacks, packs, compares, saturating and averaging arithmetic, multiplies, shifts by a register)
		if !sse_pfx {if _, ok2 := simd_int_assigns(o2, 8); ok2 {x = .MMXOP}}
	}
	if x == .None {return}
	hit = true
	if m >= 0xc0 {
		r = Rx_Ins{addr = addr, len = u32(p + 3), sti = -1, inst = {mnemonic = .NOP, length = u8(p + 3)}}
	} else {
		r, ok = memform(code, p, p + 2, addr)
		if !ok {return {}, false, true}
		r.xmem = true
	}
	r.opc, r.opc2, r.modrm = 0x0f, o2, m
	r.xop, r.xr, r.xrm = x, reg, m & 7
	return r, true, true
}

rx_decode_at :: proc(addr: u32) -> (Rx_Ins, bool) {
	return rx_decode_bytes(exe_read(addr, 15), addr)
}

@(private = "file")
rx_insert_op :: proc(r: ^Rx_Ins, k: int, op: x86.Operand) {
	n := int(r.inst.operand_count)
	if n >= 4 {return}
	for j := n; j > k; j -= 1 {r.inst.ops[j] = r.inst.ops[j - 1]}
	r.inst.ops[k] = op
	r.inst.operand_count += 1
}

@(private = "file")
rx_acc :: proc(r: ^Rx_Ins, byte_form: bool) -> x86.Operand {
	if byte_form {return x86.op_reg(x86.AL)}
	return x86.op_reg(r.p66 ? x86.AX : x86.EAX)
}

@(private = "file")
rx_normalize :: proc(r: ^Rx_Ins) {
	if r.opc == 0x0f {return}
	o := r.opc
	switch {
	case o < 0x40 && (o & 7) == 4 && (o & 0xc0) == 0: // ALU al, ib
		rx_insert_op(r, 0, rx_acc(r, true))
	case o < 0x40 && (o & 7) == 5 && (o & 0xc0) == 0: // ALU eax, iz
		rx_insert_op(r, 0, rx_acc(r, false))
	case o == 0xa8: rx_insert_op(r, 0, rx_acc(r, true))
	case o == 0xa9: rx_insert_op(r, 0, rx_acc(r, false))
	case o >= 0x91 && o <= 0x97: // xchg eax, r
		if r.inst.operand_count == 1 {rx_insert_op(r, 0, rx_acc(r, false))}
	case o >= 0xa0 && o <= 0xa3: // mov acc <-> [moffs]
		disp := i32(u32(r.inst.ops[0].immediate))
		sz: u8 = (o & 1) == 0 ? 1 : (r.p66 ? 2 : 4)
		// (an fs: / gs: prefix is put on it with the other memory operands', in rx_decode_at)
		mem := x86.op_mem(x86.mem_make(x86.NONE, x86.NONE, 1, disp, x86.NONE), sz)
		acc := rx_acc(r, (o & 1) == 0)
		r.inst.mnemonic = .MOV
		r.inst.operand_count = 2
		if o <= 0xa1 {r.inst.ops[0] = acc; r.inst.ops[1] = mem} else {r.inst.ops[0] = mem; r.inst.ops[1] = acc}
	case o >= 0xd8 && o <= 0xdf && r.modrm >= 0xc0:
		r.sti = i8(r.modrm & 7)
	}
	if r.inst.mnemonic == .MOVABS {r.inst.mnemonic = .MOV}
}

// x86.mnemonic_to_string(m, true) looks the name up by reflection and lowercases a copy each time: once per mnemonic
@(thread_local, private = "file")
rx_mn_memo: map[x86.Mnemonic]string

rx_mn :: proc(i: ^Rx_Ins) -> string {
	switch i.xop {
	case .LDMXCSR: return "ldmxcsr"
	case .STMXCSR: return "stmxcsr"
	case .CVTPI2PS: return "cvtpi2ps"
	case .CVTTPS2PI: return "cvttps2pi"
	case .CVTPS2PI: return "cvtps2pi"
	case .MOVQ_LD, .MOVQ_ST: return "movq"
	case .MOVD_LD, .MOVD_ST: return "movd"
	case .MMXOP: return "mmx"
	case .SHIFT_IMM: return "psxx imm"
	case .EMMS: return "emms"
	case .UD: return "(invalid)"
	case .None:
	}
	if s, ok := rx_mn_memo[i.inst.mnemonic]; ok {return s}
	if rx_mn_memo == nil {rx_mn_memo = make(map[x86.Mnemonic]string, 256, runtime.heap_allocator())}
	s := strings.clone(x86.mnemonic_to_string(i.inst.mnemonic, true), runtime.heap_allocator())
	rx_mn_memo[i.inst.mnemonic] = s
	return s
}

// the absolute target of a relative operand (jmp/jcc/call)
rx_rel_target :: proc(i: ^Rx_Ins, k: int) -> u32 {return u32(i64(i.addr + i.len) + i.inst.ops[k].relative)}

rx_is_branch_end :: proc(m: x86.Mnemonic) -> bool {
	#partial switch m {
	case .RET, .JMP, .INT3, .UD2, .HLT, .IRETD: return true
	}
	return false
}

rx_is_jcc :: proc(m: x86.Mnemonic) -> bool {
	#partial switch m {
	case .JO, .JNO, .JB, .JC, .JNAE, .JAE, .JNB, .JNC, .JE, .JZ, .JNE, .JNZ, .JBE, .JNA, .JA, .JNBE, .JS, .JNS, .JP,
	     .JPE, .JNP, .JPO, .JL, .JNGE, .JGE, .JNL, .JLE, .JNG, .JG, .JNLE, .JECXZ, .LOOP, .LOOPE, .LOOPNE:
		return true
	}
	return false
}

// the reachable instructions of a function from its entry (recursive descent inside [a, a+size)); jump tables are
// followed by the caller (xrc), so an indirect jmp ends a path here
rx_function :: proc(a: u32, size: u32, alloc := context.allocator) -> (out: [dynamic]Rx_Ins, bad: u32) {
	out = make([dynamic]Rx_Ins, 0, int(min(size, 1 << 20)) / 3 + 16, alloc) // (~3.4 bytes an instruction)
	// the instructions seen: a bit per byte of [a, a+size), a map for the chunks beyond it
	seen_in := make([]u64, (int(size) + 63) / 64, context.temp_allocator)
	seen_out := make(map[u32]bool, context.temp_allocator)
	is_seen :: #force_inline proc(seen_in: []u64, seen_out: ^map[u32]bool, pc, a, size: u32) -> bool {
		if pc >= a && pc < a + size {o := pc - a; return seen_in[o >> 6] & (1 << (o & 63)) != 0}
		return pc in seen_out
	}
	work := make([dynamic]u32, context.temp_allocator)
	append(&work, a)
	// Function chunks: MSVC lays some functions out non-contiguously (a __finally funclet emitted in the middle,
	// the function's tail after it), so a jmp / jcc to code beyond [a, a+size) that is no known function's start is
	// followed as part of this function, up to its own ret / jmp or the next function's start. (Were it really a
	// tail call into an unlisted function, translating that code here is still correct, only duplicated.)
	CHUNK_MAX :: 4096 // instructions decoded outside [a, a+size), against walking into garbage
	nchunk := 0
	in_body :: proc(pc, a, size: u32, chunk: bool) -> bool {
		if pc >= a && pc < a + size {return true}
		if !chunk || pc < exe.text_lo || pc >= exe.text_hi {return false}
		return !(pc in funcs)
	}
	for len(work) > 0 {
		pc := pop(&work)
		chunk := !(pc >= a && pc < a + size)
		for in_body(pc, a, size, chunk) && !is_seen(seen_in, &seen_out, pc, a, size) && (!chunk || nchunk < CHUNK_MAX) {
			if chunk {nchunk += 1}
			i, ok := rx_decode_at(pc)
			if !ok {bad = pc; break}
			if pc >= a && pc < a + size {o := pc - a; seen_in[o >> 6] |= 1 << (o & 63)} else {seen_out[pc] = true}
			append(&out, i)
			if i.xop == .UD {break} // faults: no successor
			m := i.inst.mnemonic
			if (rx_is_jcc(m) || m == .JMP) && i.inst.operand_count > 0 && i.inst.ops[0].kind == .RELATIVE {
				append(&work, rx_rel_target(&i, 0))
			}
			if m == .JMP && rx_table_jmp(&i) {
				if t, ok := rx_jump_table(out[:], len(out) - 1, a, size); ok {
					for x in t {append(&work, x)}
				}
			}
			if rx_is_branch_end(m) {break}
			pc += i.len
		}
	}
	// in address order (each address once): sorting the 8-byte keys, not the ~100-byte instructions, then one copy
	sorted := true
	for k in 1 ..< len(out) {if out[k - 1].addr > out[k].addr {sorted = false; break}}
	if !sorted {
		keys := make([]u64, len(out), context.temp_allocator)
		for &i, k in out {keys[k] = u64(i.addr) << 32 | u64(k)}
		slice.sort(keys)
		res := make([dynamic]Rx_Ins, len(out), alloc)
		for key, k in keys {res[k] = out[int(key & 0xffffffff)]}
		delete(out)
		out = res
	}
	return
}

// The targets of an MSVC switch jump `jmp [idx*4 + table]` (ins[k]), all inside the function [a, a+size):
//     cmp idx, N-1 ; ja default ; [movzx idx, byte [idx + btable]] ; jmp [idx*4 + table]
// The bound check before it gives the entry count (with a byte index table: the bytes, whose largest value + 1 is the
// dword count). Without a recognizable bound, entries are taken while they point into the function (max 512).
// ins must hold the instructions before ins[k] (in address order where they are contiguous).
// What the walker learned of each switch, for the translator to turn into code (no byte of .text is read at run time: the
// retail exe's .text is encrypted, and the bundle carries no bytes of the exe):
//   rx_table_slots   jmp address -> each table slot read: (slot address, target)  -> switch (_ea) { case slot: goto }
//   rx_byte_tables   address of the movzx that reads the byte index table -> (table address, entries)
//                                                                              -> switch (_ea) { case at: v = byte }
Rx_Byte_Table :: struct {at, n: u32}
// (per thread: each translation thread fills its own while translating a function, and reads only those)
@(thread_local)
rx_table_slots: map[u32][dynamic][2]u32
@(thread_local)
rx_byte_tables: map[u32]Rx_Byte_Table
@(private = "file", thread_local)
rx_cur_slots: ^[dynamic][2]u32
@(private = "file", thread_local)
rx_cur_bias: u32 // (a GOT-relative table: the GOT address, added to every entry)

// a jmp that may be a switch: through memory, or (PIC code: the executable has a GOT) through a register
rx_table_jmp :: proc(i: ^Rx_Ins) -> bool {
	if i.inst.operand_count == 0 {return false}
	return i.inst.ops[0].kind == .MEMORY || (i.inst.ops[0].kind == .REGISTER && exe.got != 0)
}

// PIC code (ELF, Mach-O) switches through a GOT-relative table instead:
//     cmp idx, N-1 ; ja default ; mov r, [got + idx*4 + table@GOTOFF] ; add r, got ; jmp r
// (got: the register that holds the GOT's address, ebx in most code). The same walk, with every entry + the GOT.
rx_jump_table :: proc(ins: []Rx_Ins, k: int, a, size: u32) -> (out: [dynamic]u32, ok: bool) {
	j := &ins[k]
	if !rx_table_jmp(j) {return}
	if rx_table_slots == nil {rx_table_slots = make(map[u32][dynamic][2]u32); rx_byte_tables = make(map[u32]Rx_Byte_Table)} // (the heap: callers use temp allocators)
	slots := make([dynamic][2]u32)
	rx_cur_slots = &slots
	defer {
		rx_cur_slots = nil
		if ok {
			if old, had := rx_table_slots[j.addr]; had {delete(old)}
			rx_table_slots[j.addr] = slots
		} else {
			delete(slots)
		}
	}
	rx_cur_bias = 0
	tbl: u32
	if j.inst.ops[0].kind == .REGISTER {
		// (right before the jmp: GCC's `mov r, [got + idx*4 + t]; add r, got`, or clang's `add got, [got + idx*4 + t]`
		// with r = got)
		if k < 1 {return}
		ad := &ins[k - 1]
		r := j.inst.ops[0].reg
		if ad.addr + ad.len != j.addr || ad.inst.mnemonic != .ADD || ad.inst.operand_count != 2 || ad.inst.ops[0].kind != .REGISTER ||
		   ad.inst.ops[0].reg != r {return}
		m: x86.Memory
		g: x86.Register
		if ad.inst.ops[1].kind == .MEMORY {
			m, g = ad.inst.ops[1].mem, r
		} else if ad.inst.ops[1].kind == .REGISTER && k >= 2 {
			mv := &ins[k - 2]
			g = ad.inst.ops[1].reg
			if mv.addr + mv.len != ad.addr || mv.inst.mnemonic != .MOV || mv.inst.operand_count != 2 || mv.inst.ops[0].kind != .REGISTER ||
			   mv.inst.ops[0].reg != r || mv.inst.ops[1].kind != .MEMORY {return}
			m = mv.inst.ops[1].mem
		} else {
			return
		}
		if !x86.mem_has_base(m) || x86.mem_base(m) != g || !x86.mem_has_index(m) || x86.mem_scale(m) != 4 || m.segment != 0 {return}
		rx_cur_bias = exe.got
		tbl = exe.got + u32(m.disp)
	} else {
		mm := j.inst.ops[0].mem
		if !x86.mem_has_index(mm) || x86.mem_has_base(mm) || x86.mem_scale(mm) != 4 || mm.segment != 0 {return}
		tbl = u32(mm.disp)
	}
	if tbl < exe.lo || tbl >= exe.hi {return}
	// walk back over the contiguous instructions before the jmp
	bound := -1
	btbl: u32 = 0
	bload: u32 = 0
	pc := j.addr
	for q := k - 1; q >= 0 && q >= k - 8; q -= 1 {
		p := &ins[q]
		if p.addr + p.len != pc {break}
		pc = p.addr
		pm := p.inst.mnemonic
		if pm == .MOVZX && p.inst.ops[1].kind == .MEMORY && p.inst.ops[1].size == 1 {
			bm := p.inst.ops[1].mem
			if x86.mem_has_base(bm) && !x86.mem_has_index(bm) {btbl = u32(bm.disp); bload = p.addr}
		}
		if (pm == .JA || pm == .JNBE) && q > 0 {
			c := &ins[q - 1]
			if c.addr + c.len == p.addr && c.inst.mnemonic == .CMP && c.inst.operand_count == 2 && c.inst.ops[1].kind == .IMMEDIATE {
				bound = int(c.inst.ops[1].immediate) + 1
			}
			break
		}
	}
	n := 512
	if bound > 0 && bound <= 4096 {
		if btbl != 0 {
			if btbl < exe.lo || btbl + u32(bound) > exe.hi {return}
			bytes := exe_read(btbl, bound)
			rx_byte_tables[bload] = {btbl, u32(bound)}
			mx := 0
			for b in bytes {mx = max(mx, int(b))}
			n = mx + 1
		} else {
			n = bound
		}
	}
	out = make([dynamic]u32, context.temp_allocator)
	entry :: proc(at: u32) -> u32 {
		b := exe_read(at, 4)
		if len(b) < 4 {return 0}
		t := (u32(b[0]) | u32(b[1]) << 8 | u32(b[2]) << 16 | u32(b[3]) << 24) + rx_cur_bias
		if rx_cur_slots != nil {append(rx_cur_slots, [2]u32{at, t})}
		return t
	}
	if bound > 0 {
		for e in 0 ..< n {
			if tbl + u32(4 * e) + 4 > exe.hi {break}
			t := entry(tbl + u32(4 * e))
			if t < a || t >= a + size {return out, false} // a bounded table must be all in range
			append(&out, t)
		}
		return out, len(out) > 0
	}
	// No recognizable bound (hand-written code, the CRT's memcpy): entries while they point into the function, one
	// stray entry tolerated (an index the code never uses, overlapping code), and the in-range entries right before
	// the base (a negative index into the table before it). Extra cases are harmless: the switch matches the loaded
	// target, and anything else still leaves through the dispatcher.
	miss := 0
	for e in 0 ..< n {
		if tbl + u32(4 * e) + 4 > exe.hi {break}
		t := entry(tbl + u32(4 * e))
		if t < a || t >= a + size {
			miss += 1
			if miss > 1 {break}
			continue
		}
		miss = 0
		append(&out, t)
	}
	for e in 1 ..= 8 {
		at := tbl - u32(4 * e)
		if at < exe.lo {break}
		t := entry(at)
		if t < a || t >= a + size {break}
		append(&out, t)
	}
	return out, len(out) > 0
}
