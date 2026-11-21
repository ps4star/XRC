package xrc

// Static inputs: the PE image (the unpacked exe) and the game's tables (its function map and import slots).

import "core:encoding/csv"
import "core:encoding/json"
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
Pe_Sec :: struct {
	base, vsize, raw, rsize, chars: u32,
}

Exe :: struct {
	data:     []u8,
	secs:     []Pe_Sec,
	lo, hi:   u32,
	text_lo:  u32,
	text_hi:  u32,
	writable: [dynamic][2]u32,
}

exe: Exe

load_exe :: proc(p: string) -> bool {
	d, ok := read_file(p)
	if !ok {
		return false
	}
	exe.data = d
	pe := int(u32le(d, 0x3c))
	nsec := int(u16le(d, pe + 6))
	optsz := int(u16le(d, pe + 20))
	base := u32le(d, pe + 24 + 28)
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
		if s.chars & 0x20000000 != 0 {
			exe.text_lo = exe.lo + va
			exe.text_hi = exe.lo + va + s.vsize
		}
		if s.chars & 0x80000000 != 0 {
			append(&exe.writable, [2]u32{(exe.lo + va) & ~u32(PAGE - 1), (exe.lo + va + s.vsize + PAGE - 1) & ~u32(PAGE - 1)})
		}
		exe.hi = max(exe.hi, s.base + max(s.vsize, s.rsize))
	}
	exe.hi = (exe.hi + PAGE - 1) & ~u32(PAGE - 1)
	return true
}

// bytes at a virtual address (like tools/exe_util.py: a section spans max(vsize, rsize))
exe_read :: proc(va: u32, n: int) -> []u8 {
	for s in exe.secs {
		span := max(s.vsize, s.rsize)
		if s.base <= va && va < s.base + span {
			o := int(s.raw + (va - s.base))
			o = min(o, len(exe.data))
			e := min(o + n, len(exe.data))
			return exe.data[o:e]
		}
	}
	return nil
}

// ---------------------------------------------------------------------------------------------- tables
// The game's tables, in its data folder (--data DIR, else overrides/NAME/data with --override):
//   function_map.csv     every function: address (hex), name, size (bytes), ... -- a header line first (e.g. a
//                        Ghidra export). The names matter only for the CRT's x87 helpers
//                        (__CI*, *ftol*, __cintrindisp*, __security_check_cookie: xrc_fp.odin).
//   iat.json             the import slots: {"0xSLOT": ["dll", "function"], ...}
//   size_overrides.txt   (optional) "ADDR SIZE" lines that correct the map's sizes
DATA_DIR := ""

table_path :: proc(name: string) -> string {
	p, _ := filepath.join({DATA_DIR, name}, context.temp_allocator)
	return p
}

funcs: map[u32]u32          // address -> size (function map, size overrides applied)
fstarts: []u32
iat: map[u32]string          // slot -> import name

fsize :: proc(a: u32) -> u32 {
	return funcs[a] or_else 0
}

load_tables :: proc() -> bool {
	{
		txt, ok := read_text(table_path("function_map.csv"))
		if !ok {
			fmt.eprintf("xrc: no %s\n", table_path("function_map.csv"))
			return false
		}
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
	{
		txt, ok := read_text(table_path("iat.json"))
		if !ok {
			fmt.eprintf("xrc: no %s\n", table_path("iat.json"))
			return false
		}
		v, err := json.parse_string(txt)
		if err != nil {
			fmt.eprintf("xrc: %s is not valid JSON\n", table_path("iat.json"))
			return false
		}
		for k, e in v.(json.Object) {
			a, _ := hex_u32(k)
			arr := e.(json.Array)
			iat[a] = strings.clone(arr[1].(json.String))
		}
	}
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
