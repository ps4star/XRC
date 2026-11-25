package xrc

// Static inputs: the executable (PE, ELF, Mach-O or a raw image) and the game's tables (its function map).

import "core:encoding/csv"
import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:slice"
import "core:strconv"
import "core:strings"

M32 :: 0xffffffff
PAGE :: 0x1000

read_file :: proc(p: string) -> ([]u8, bool) {
	d, err := os.read_entire_file(p, context.allocator)
	return d, err == nil
}

read_text :: proc(p: string) -> (string, bool) {
	d, ok := read_file(p)
	if !ok {
		return "", false
	}
	s := string(d)
	if strings.has_prefix(s, "\xef\xbb\xbf") {
		s = s[3:]
	}
	return s, true
}

u32le :: #force_inline proc(b: []u8, o: int) -> u32 {
	return u32(b[o]) | u32(b[o + 1]) << 8 | u32(b[o + 2]) << 16 | u32(b[o + 3]) << 24
}

u16le :: #force_inline proc(b: []u8, o: int) -> u16 {
	return u16(b[o]) | u16(b[o + 1]) << 8
}

put32 :: #force_inline proc(b: []u8, o: int, v: u32) {
	b[o] = u8(v); b[o + 1] = u8(v >> 8); b[o + 2] = u8(v >> 16); b[o + 3] = u8(v >> 24)
}

hex_u32 :: proc(s: string) -> (u32, bool) {
	t := strings.trim_space(s)
	if strings.has_prefix(t, "0x") || strings.has_prefix(t, "0X") {
		t = t[2:]
	}
	v, ok := strconv.parse_u64_of_base(t, 16)
	return u32(v), ok && len(t) > 0
}

// ---------------------------------------------------------------------------------------------- the exe
// The executable as sections: PE sections, ELF loadable segments (PT_LOAD), Mach-O segments, or one raw image. Each
// keeps its flags in PE terms (Sec_Exec / Sec_Read / Sec_Write), so the rest of xrc does not care which format it was.
Exe_Format :: enum {PE, ELF, MachO, Raw}

Sec_Exec :: 0x20000000
Sec_Read :: 0x40000000
Sec_Write :: 0x80000000

Pe_Sec :: struct {
	base, vsize, raw, rsize, chars: u32,
}

Exe :: struct {
	format:   Exe_Format,
	data:     []u8,
	secs:     []Pe_Sec,
	lo, hi:   u32,
	text_lo:  u32,
	text_hi:  u32,
	writable: [dynamic][2]u32,
	no_fold:  [dynamic][2]u32, // read-only data never taken as constants: the import tables, xrc.cfg's `patched`
	entry:    u32,
	got:      u32,          // ELF / Mach-O PIC code's table base (_GLOBAL_OFFSET_TABLE_), for GOT-relative switch tables
	syms:     map[u32]u32,  // functions from the file's own symbols (and its import stubs): address -> size
	names:    map[u32]string,
	what:     string,       // "ELF i386 executable" ... (for the summary)
	bias:     u32,          // a PIE / shared object: where it was placed (its link addresses + bias)
	imp_slot: map[u32]string, // PE: each IAT slot's import, "dll!name" (dll lowercase; "dll!#N" by ordinal) -- x_imp_n
}

exe: Exe
RAW_BASE: u32 = 0      // --raw ADDR
RAW_SET := false
BASE: u32 = 0          // --base ADDR (a PIE / shared object)
BASE_SET := false

load_exe :: proc(p: string) -> bool {
	d, ok := read_file(p)
	if !ok {
		fmt.eprintf("xrc: cannot read %s\n", p)
		return false
	}
	exe.data = d
	exe.syms = make(map[u32]u32)
	exe.names = make(map[u32]string)
	if RAW_SET {return load_raw(d)}
	if len(d) >= 0x40 && d[0] == 'M' && d[1] == 'Z' {return load_pe(d)}
	if len(d) >= 52 && d[0] == 0x7f && d[1] == 'E' && d[2] == 'L' && d[3] == 'F' {return load_elf(d)}
	if len(d) >= 28 && (u32le(d, 0) == 0xfeedface || u32be(d, 0) == 0xcafebabe) {return load_macho(d)}
	fmt.eprintf("xrc: %s is not a PE, ELF or Mach-O file (a flat image needs --raw ADDR)\n", p)
	return false
}

u32be :: #force_inline proc(b: []u8, o: int) -> u32 {
	return u32(b[o]) << 24 | u32(b[o + 1]) << 16 | u32(b[o + 2]) << 8 | u32(b[o + 3])
}

load_pe :: proc(d: []u8) -> bool {
	exe.format = .PE
	pe := int(u32le(d, 0x3c))
	if pe + 0xf8 > len(d) || u32le(d, pe) != 0x4550 {fmt.eprintln("xrc: not a PE image (no PE header)"); return false}
	if u16le(d, pe + 4) != 0x14c {fmt.eprintln("xrc: the PE image is not 32-bit x86 (machine 0x14c)"); return false}
	nsec := int(u16le(d, pe + 6))
	optsz := int(u16le(d, pe + 20))
	base := u32le(d, pe + 24 + 28)
	exe.entry = base + u32le(d, pe + 24 + 16)
	exe.what = u16le(d, pe + 22) & 0x2000 != 0 ? "PE i386 DLL" : "PE i386 executable"
	exe.secs = make([]Pe_Sec, nsec)
	exe.lo = base // (the image base)
	sh := pe + 24 + optsz
	for i in 0 ..< nsec {
		o := sh + 40 * i
		s := &exe.secs[i]
		s.vsize = u32le(d, o + 8)
		va := u32le(d, o + 12)
		s.rsize = u32le(d, o + 16)
		s.raw = u32le(d, o + 20)
		s.chars = u32le(d, o + 36)
		s.base = base + va
		// the code: every executable section (FF13-2 has two, .text and MSSMIXER: the last one alone left discovery
		// out of .text)
		if s.chars & Sec_Exec != 0 {
			if exe.text_hi == 0 {exe.text_lo = exe.lo + va}
			exe.text_lo = min(exe.text_lo, exe.lo + va)
			exe.text_hi = max(exe.text_hi, exe.lo + va + s.vsize)
		}
		if s.chars & Sec_Write != 0 {
			append(&exe.writable, [2]u32{(exe.lo + va) & ~u32(PAGE - 1), (exe.lo + va + s.vsize + PAGE - 1) & ~u32(PAGE - 1)})
		}
		exe.hi = max(exe.hi, s.base + max(s.vsize, s.rsize))
	}
	exe.hi = (exe.hi + PAGE - 1) & ~u32(PAGE - 1)
	// the import tables (the loader -- the runtime -- writes the IAT): never folded as constants (x_rdata_fold). The
	// data directories' imports (1), IAT (12) and delayed imports (13), and each import descriptor's thunk array
	ndir := int(u32le(d, pe + 24 + 92))
	for k in ([3]int{1, 12, 13}) {
		if k >= ndir {continue}
		rva, size := u32le(d, pe + 24 + 96 + 8 * k), u32le(d, pe + 24 + 96 + 8 * k + 4)
		if rva != 0 && size != 0 {append(&exe.no_fold, [2]u32{base + rva, base + rva + size})}
	}
	if ndir > 1 {
		if irva := u32le(d, pe + 24 + 96 + 8); irva != 0 {
			for desc := base + irva; ; desc += 20 {
				b := exe_read(desc, 20)
				if len(b) < 20 || u32le(b, 12) == 0 {break}
				dll := strings.to_lower(cstr(exe_read(base + u32le(b, 12), 256), 0))
				look := u32le(b, 0) != 0 ? base + u32le(b, 0) : base + u32le(b, 16) // (the lookup table: the file's IAT holds the same before binding)
				iat := base + u32le(b, 16)
				n := u32(0)
				for {
					t := exe_read(iat + 4 * n, 4)
					if len(t) < 4 || u32le(t, 0) == 0 {break}
					if l := exe_read(look + 4 * n, 4); len(l) == 4 {
						v := u32le(l, 0)
						exe.imp_slot[iat + 4 * n] = v & 0x80000000 != 0 ? fmt.aprintf("%s!#%d", dll, v & 0xffff) : fmt.aprintf("%s!%s", dll, cstr(exe_read(base + v + 2, 256), 0))
					}
					n += 1
				}
				append(&exe.no_fold, [2]u32{iat, iat + 4 * n + 4})
			}
		}
	}
	return true
}

// a segment / section into exe.secs (non-PE: a file span past its size reads as zeros, as the loader maps it)
@(private = "file")
add_sec :: proc(secs: ^[dynamic]Pe_Sec, base, vsize, raw, rsize: u32, x, w: bool) {
	chars := u32(Sec_Read)
	if x {chars |= Sec_Exec}
	if w {chars |= Sec_Write}
	append(secs, Pe_Sec{base, vsize, raw, min(rsize, vsize), chars})
	if w {append(&exe.writable, [2]u32{base & ~u32(PAGE - 1), (base + vsize + PAGE - 1) & ~u32(PAGE - 1)})}
}

@(private = "file")
finish_secs :: proc(secs: [dynamic]Pe_Sec) {
	exe.secs = secs[:]
	exe.lo = M32
	for s in exe.secs {
		exe.lo = min(exe.lo, s.base & ~u32(PAGE - 1))
		exe.hi = max(exe.hi, s.base + s.vsize)
		// (the code range, unless the section table gave one)
		if s.chars & Sec_Exec != 0 && !code_from_sections {
			if exe.text_hi == 0 {exe.text_lo = s.base}
			exe.text_lo = min(exe.text_lo, s.base)
			exe.text_hi = max(exe.text_hi, s.base + s.vsize)
		}
	}
	exe.hi = (exe.hi + PAGE - 1) & ~u32(PAGE - 1)
}

@(private = "file")
code_from_sections := false

@(private = "file")
code_range :: proc(lo, hi: u32) {
	if !code_from_sections {exe.text_lo, exe.text_hi = lo, hi; code_from_sections = true; return}
	exe.text_lo = min(exe.text_lo, lo)
	exe.text_hi = max(exe.text_hi, hi)
}

@(private = "file")
cstr :: proc(d: []u8, o: int) -> string {
	if o < 0 || o >= len(d) {return ""}
	e := o
	for e < len(d) && d[e] != 0 {e += 1}
	return string(d[o:e])
}

// ELF32 i386 (EM_386): the PT_LOAD segments; the code range, PLT and GOT from the section headers when it has them;
// functions from .symtab / .dynsym
load_elf :: proc(d: []u8) -> bool {
	exe.format = .ELF
	if d[4] != 1 || d[5] != 1 {fmt.eprintln("xrc: the ELF file is not 32-bit little-endian"); return false}
	if u16le(d, 18) != 3 {fmt.eprintln("xrc: the ELF file is not i386 (EM_386)"); return false}
	et := u16le(d, 16)
	exe.what = et == 3 ? "ELF i386 shared object / PIE" : "ELF i386 executable"
	exe.entry = u32le(d, 24)
	phoff, shoff := int(u32le(d, 28)), int(u32le(d, 32))
	phent, phnum := int(u16le(d, 42)), int(u16le(d, 44))
	shent, shnum, shstr := int(u16le(d, 46)), int(u16le(d, 48)), int(u16le(d, 50))
	secs := make([dynamic]Pe_Sec)
	for i in 0 ..< phnum {
		o := phoff + phent * i
		if o + 32 > len(d) || u32le(d, o) != 1 {continue} // PT_LOAD
		fl := u32le(d, o + 24)
		add_sec(&secs, u32le(d, o + 8), u32le(d, o + 20), u32le(d, o + 4), u32le(d, o + 16), fl & 1 != 0, fl & 2 != 0)
	}
	if len(secs) == 0 {fmt.eprintln("xrc: the ELF file has no loadable segments"); return false}
	if shoff != 0 && shnum > 0 && shoff + shent * shnum <= len(d) {
		sh :: proc(d: []u8, shoff, shent, i, f: int) -> u32 {return u32le(d, shoff + shent * i + f)}
		strs := int(sh(d, shoff, shent, shstr, 16))
		for i in 0 ..< shnum {
			name := cstr(d, strs + int(sh(d, shoff, shent, i, 0)))
			typ, flags, addr := sh(d, shoff, shent, i, 4), sh(d, shoff, shent, i, 8), sh(d, shoff, shent, i, 12)
			off, size := int(sh(d, shoff, shent, i, 16)), sh(d, shoff, shent, i, 20)
			if flags & 4 != 0 && flags & 2 != 0 && size > 0 {code_range(addr, addr + size)} // SHF_EXECINSTR, SHF_ALLOC
			switch name {
			case ".got.plt": exe.got = addr
			case ".got": if exe.got == 0 {exe.got = addr}
			}
			// the PLT: one function per entry (.plt: 16 bytes, the first is the resolver's; .plt.got / .plt.sec: 8 / 16)
			if name == ".plt" || name == ".plt.sec" || name == ".plt.got" {
				step := u32(name == ".plt.got" ? 8 : 16)
				for a := addr; a + step <= addr + size; a += step {exe.syms[a] = step}
			}
			if typ == 2 || typ == 11 { // SHT_SYMTAB, SHT_DYNSYM
				link := int(sh(d, shoff, shent, i, 24))
				names := int(sh(d, shoff, shent, link, 16))
				for k := 0; k + 16 <= int(size) && off + k + 16 <= len(d); k += 16 {
					v, sz, info, shndx := u32le(d, off + k + 4), u32le(d, off + k + 8), d[off + k + 12], u16le(d, off + k + 14)
					t := info & 0xf
					if (t != 2 && t != 10) || shndx == 0 || v == 0 || sz == 0 {continue} // STT_FUNC / STT_GNU_IFUNC, defined
					exe.syms[v] = sz
					if v not_in exe.names {exe.names[v] = strings.clone(cstr(d, names + int(u32le(d, off + k))))}
				}
			}
		}
	}
	// A PIE / shared object (linked at 0): placed where Linux puts a 32-bit PIE (or at --base), and its dynamic
	// relocations applied as the loader would, so the data xrc reads (pointer tables) holds run-time addresses. (At
	// 0, small integers would pass for code addresses.) The runtime loads it at the same address.
	if et == 3 {
		minv: u32 = M32
		for s in secs {minv = min(minv, s.base & ~u32(PAGE - 1))}
		base := BASE_SET ? BASE : (minv == 0 ? 0x56555000 : minv)
		bias := base - minv
		// the file offset of an address the file has
		file_off :: proc(secs: []Pe_Sec, va: u32) -> (int, bool) {
			for s in secs {if va >= s.base && va + 4 <= s.base + s.rsize {return int(s.raw + (va - s.base)), true}}
			return 0, false
		}
		if shoff != 0 && shnum > 0 && shoff + shent * shnum <= len(d) {
			sh :: proc(d: []u8, shoff, shent, i, f: int) -> u32 {return u32le(d, shoff + shent * i + f)}
			for i in 0 ..< shnum {
				if sh(d, shoff, shent, i, 4) != 9 {continue} // SHT_REL
				off, size := int(sh(d, shoff, shent, i, 16)), int(sh(d, shoff, shent, i, 20))
				symoff := int(sh(d, shoff, shent, int(sh(d, shoff, shent, i, 24)), 16))
				for k := 0; k + 8 <= size && off + k + 8 <= len(d); k += 8 {
					at, info := u32le(d, off + k), u32le(d, off + k + 4)
					fo, okf := file_off(secs[:], at)
					if !okf {continue}
					switch info & 0xff {
					case 8, 7: put32(d, fo, u32le(d, fo) + bias) // R_386_RELATIVE, R_386_JMP_SLOT (its lazy stub)
					case 1, 6: // R_386_32, R_386_GLOB_DAT: of a symbol this file defines (imports: the runtime's)
						so := symoff + 16 * int(info >> 8)
						if so + 16 > len(d) || u16le(d, so + 14) == 0 {continue}
						sv := u32le(d, so + 4) + bias
						put32(d, fo, (info & 0xff) == 1 ? u32le(d, fo) + sv : sv)
					}
				}
			}
		}
		if bias != 0 {
			for &s in secs {s.base += bias}
			for &w in exe.writable {w[0] += bias; w[1] += bias}
			exe.entry += bias
			if exe.got != 0 {exe.got += bias}
			if code_from_sections {exe.text_lo += bias; exe.text_hi += bias}
			syms := make(map[u32]u32)
			names := make(map[u32]string)
			for a, n in exe.syms {syms[a + bias] = n}
			for a, n in exe.names {names[a + bias] = n}
			exe.syms, exe.names = syms, names
			exe.bias = bias
		}
	}
	finish_secs(secs)
	return true
}

// Mach-O i386 (or the i386 slice of a fat file): the segments, the code range and symbol stubs from their sections,
// functions from LC_SYMTAB (no sizes there: each runs to the next symbol or its section's end)
load_macho :: proc(full: []u8) -> bool {
	exe.format = .MachO
	d := full
	if u32be(d, 0) == 0xcafebabe {
		n := int(u32be(d, 4))
		found := false
		for i in 0 ..< n {
			o := 8 + 20 * i
			if o + 20 > len(d) {break}
			if u32be(d, o) == 7 {
				off, size := int(u32be(d, o + 8)), int(u32be(d, o + 12))
				if off + size > len(d) {break}
				d = d[off:off + size]
				found = true
				break
			}
		}
		if !found {fmt.eprintln("xrc: the fat Mach-O file has no i386 slice"); return false}
		exe.data = d
	}
	if u32le(d, 0) != 0xfeedface || u32le(d, 4) != 7 {fmt.eprintln("xrc: the Mach-O file is not i386"); return false}
	switch u32le(d, 12) {
	case 1: exe.what = "Mach-O i386 object"
	case 6: exe.what = "Mach-O i386 dylib"
	case 8: exe.what = "Mach-O i386 bundle"
	case: exe.what = "Mach-O i386 executable"
	}
	ncmds := int(u32le(d, 16))
	secs := make([dynamic]Pe_Sec)
	text_vm: u32 = 0
	Msec :: struct {lo, hi: u32}
	code_secs := make([dynamic]Msec, context.temp_allocator)
	o := 28
	symoff, nsyms, stroff := 0, 0, 0
	main_off := -1
	for _ in 0 ..< ncmds {
		if o + 8 > len(d) {break}
		cmd, csz := u32le(d, o), int(u32le(d, o + 4))
		switch cmd {
		case 1: // LC_SEGMENT
			segname := cstr(d[:o + 24], o + 8)
			vm, vsz, foff, fsz, prot := u32le(d, o + 24), u32le(d, o + 28), u32le(d, o + 32), u32le(d, o + 36), u32le(d, o + 44)
			if segname == "__TEXT" {text_vm = vm - foff}
			if prot != 0 && vsz != 0 {add_sec(&secs, vm, vsz, foff, fsz, prot & 4 != 0, prot & 2 != 0)}
			ns := int(u32le(d, o + 48))
			for k in 0 ..< ns {
				so := o + 56 + 68 * k
				sname := cstr(d[:so + 16], so)
				addr, size, flags, res2 := u32le(d, so + 32), u32le(d, so + 36), u32le(d, so + 56), u32le(d, so + 64)
				if sname == "__nl_symbol_ptr" && exe.got == 0 {exe.got = addr}
				// (code: pure / some instructions; not the self-modifying __jump_table, which dyld fills in at load)
				if flags & 0x80000400 != 0 && flags & 0x04000000 == 0 && size > 0 {
					code_range(addr, addr + size)
					append(&code_secs, Msec{addr, addr + size})
					if flags & 0xff == 8 && res2 > 0 { // S_SYMBOL_STUBS: one function per stub
						for a := addr; a + res2 <= addr + size; a += res2 {exe.syms[a] = res2}
					}
				}
			}
		case 2: // LC_SYMTAB
			symoff, nsyms, stroff = int(u32le(d, o + 8)), int(u32le(d, o + 12)), int(u32le(d, o + 16))
		case 5: // LC_UNIXTHREAD: i386_thread_state, eip the 11th register
			if u32le(d, o + 8) == 1 {exe.entry = u32le(d, o + 16 + 40)}
		case 0x80000028: // LC_MAIN: the entry as a file offset into __TEXT
			main_off = int(u32le(d, o + 8))
		}
		o += csz
	}
	if main_off >= 0 {exe.entry = text_vm + u32(main_off)} // (LC_MAIN: a file offset into __TEXT)
	if len(secs) == 0 {fmt.eprintln("xrc: the Mach-O file has no segments"); return false}
	// symbols defined in a code section (N_SECT, not a stab); a function runs to the next symbol or its section's end
	starts := make([dynamic]u32, context.temp_allocator)
	for k in 0 ..< nsyms {
		so := symoff + 12 * k
		if so + 12 > len(d) {break}
		typ, v := d[so + 4], u32le(d, so + 8)
		if typ & 0xe0 != 0 || typ & 0x0e != 0x0e {continue}
		in_code := false
		for c in code_secs {if v >= c.lo && v < c.hi {in_code = true}}
		if !in_code {continue}
		append(&starts, v)
		if v not_in exe.names {
			nm := cstr(d, stroff + int(u32le(d, so)))
			if strings.has_prefix(nm, "_") {nm = nm[1:]} // (C names carry a leading _)
			exe.names[v] = strings.clone(nm)
		}
	}
	for c in code_secs {append(&starts, c.hi)}
	for s in exe.syms {append(&starts, s)}
	slice.sort(starts[:])
	for v, k in starts {
		if v in exe.syms || v not_in exe.names {continue}
		nx := v
		for j := k + 1; j < len(starts); j += 1 {if starts[j] > v {nx = starts[j]; break}}
		end := v
		for c in code_secs {if v >= c.lo && v < c.hi {end = min(nx, c.hi)}}
		if end > v {exe.syms[v] = end - v}
	}
	finish_secs(secs)
	return true
}

// --raw ADDR: the whole file as one image at ADDR, code and data, its entry point ADDR
load_raw :: proc(d: []u8) -> bool {
	exe.format = .Raw
	exe.what = "raw image"
	exe.entry = RAW_BASE
	secs := make([dynamic]Pe_Sec)
	add_sec(&secs, RAW_BASE, u32(len(d)), 0, u32(len(d)), true, true)
	finish_secs(secs)
	return true
}

// bytes at a virtual address (a PE section spans max(vsize, rsize) of the file; past an ELF / Mach-O segment's file
// size, its memory is zeros)
exe_read :: proc(va: u32, n: int) -> []u8 {
	for s in exe.secs {
		if exe.format == .PE {
			span := max(s.vsize, s.rsize)
			if s.base <= va && va < s.base + span {
				o := int(s.raw + (va - s.base))
				o = min(o, len(exe.data))
				e := min(o + n, len(exe.data))
				return exe.data[o:e]
			}
			continue
		}
		if s.base <= va && va < s.base + s.vsize {
			// (the bytes the file has, then zeros: n bytes, or to the end of the segment)
			n := min(n, int(s.base + s.vsize - va))
			in_file := va - s.base < s.rsize ? min(n, int(s.rsize - (va - s.base))) : 0
			o := min(int(s.raw + (va - s.base)), len(exe.data))
			in_file = min(in_file, len(exe.data) - o)
			if in_file == n {return exe.data[o:o + n]}
			out := make([]u8, n, context.temp_allocator)
			if in_file > 0 {copy(out, exe.data[o:o + in_file])}
			return out
		}
	}
	return nil
}

// ---------------------------------------------------------------------------------------------- tables
// The game's tables, in its data folder (--data DIR, else overrides/NAME/data with --override):
//   function_map.csv     every function: address (hex), name, size (bytes), ... -- a header line first (e.g. a
//                        Ghidra export). The names matter only for the CRT's x87 helpers
//                        (__CI*, *ftol*, __cintrindisp*, __security_check_cookie: xrc_fp.odin). Optional when the
//                        executable has symbols (ELF .symtab / .dynsym, Mach-O LC_SYMTAB): those are the map then.
//   size_overrides.txt   (optional) "ADDR SIZE" lines that correct the map's sizes
// Besides the map: the import stubs the loader found (ELF PLT entries, Mach-O symbol stubs) and the entry point.
DATA_DIR := ""

table_path :: proc(name: string) -> string {
	if DATA_DIR == "" {return ""}
	p, _ := filepath.join({DATA_DIR, name}, context.temp_allocator)
	return p
}

funcs: map[u32]u32          // address -> size (function map, size overrides applied)
fstarts: []u32
map_from_symbols := false   // no function_map.csv: the executable's own symbols

fsize :: proc(a: u32) -> u32 {
	return funcs[a] or_else 0
}

load_tables :: proc() -> bool {
	if txt, ok := read_text(table_path("function_map.csv")); ok {
		r: csv.Reader
		csv.reader_init_with_string(&r, txt)
		r.reuse_record = true
		r.fields_per_record = -1
		first := true
		for rec, _, err in csv.iterator_next(&r) {
			if err != nil {
				break
			}
			if first {
				first = false
				continue
			}
			if len(rec) >= 3 {
				a, ok1 := hex_u32(rec[0])
				n, ok2 := strconv.parse_uint(rec[2], 10)
				if ok1 && ok2 {
					funcs[a] = u32(n)
				}
			}
		}
	} else if len(exe.syms) > 0 {
		map_from_symbols = true
	} else {
		if DATA_DIR == "" {
			fmt.eprintln("xrc: the executable has no symbols: give it a function map (--override NAME or --data DIR, with function_map.csv)")
		} else {
			fmt.eprintf("xrc: no %s, and the executable has no symbols to use instead\n", table_path("function_map.csv"))
		}
		return false
	}
	// the import stubs and (with no map) the symbols' functions, where the map does not have them
	for a, n in exe.syms {
		if a not_in funcs && (map_from_symbols || exe.format != .PE) {funcs[a] = n}
	}
	if txt, ok := read_text(table_path("size_overrides.txt")); ok {
		for l in strings.split_lines(txt, context.temp_allocator) {
			if len(l) == 0 || l[0] == '#' {
				continue
			}
			f := strings.fields(l, context.temp_allocator)
			if len(f) >= 2 {
				a, _ := hex_u32(f[0])
				n, _ := strconv.parse_uint(f[1], 10)
				funcs[a] = u32(n)
			}
		}
	}
	{
		ks := make([dynamic]u32, 0, len(funcs))
		for k in funcs {
			append(&ks, k)
		}
		slice.sort(ks[:])
		fstarts = ks[:]
	}
	// the entry point, if the map does not have it: a function to the next one (or the end of the code)
	if e := exe.entry; e != 0 && e not_in funcs && e >= exe.text_lo && e < exe.text_hi {
		nx := exe.text_hi
		for s in fstarts {if s > e {nx = s; break}}
		funcs[e] = nx - e
		ks := make([dynamic]u32, 0, len(funcs))
		for k in funcs {append(&ks, k)}
		slice.sort(ks[:])
		fstarts = ks[:]
	}
	fmt.printf("xrc: %s%s, %d sections, entry %08x; %d functions (%s)\n", exe.what, exe.bias != 0 ? fmt.tprintf(" placed at %08x", exe.bias) : "", len(exe.secs), exe.entry, len(funcs),
		map_from_symbols ? "from its symbols" : "function_map.csv")
	return true
}

// an address list file (hex address first on each line; # comments): into out, or out of it with remove
addr_list :: proc(p: string, out: ^map[u32]bool, remove := false) -> bool {
	txt, ok := read_text(p)
	if !ok {return false}
	for l in strings.split_lines(txt, context.temp_allocator) {
		if len(strings.trim_space(l)) == 0 || strings.has_prefix(l, "#") {continue}
		f := strings.fields(l, context.temp_allocator)
		a, ok2 := hex_u32(f[0])
		if !ok2 {continue}
		if remove {delete_key(out, a)} else {out[a] = true}
	}
	return true
}
