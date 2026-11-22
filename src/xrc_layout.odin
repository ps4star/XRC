package xrc

import "core:fmt"
import "core:mem/virtual"
import "core:os"
import "core:slice"
import "core:strings"

// Code layout for the instruction cache: the hot functions of a profile out of their address pages, into hot pages of
// their own (page ids X_HOT_PAGE + n: page_10000.c, page_10001.c, ...), clustered so a function sits next to the hot
// ones it calls. The C compilers keep a file's functions in their order and the partial link keeps the objects' order,
// so the hot code ends up contiguous -- a few MB of a game's tens, in far fewer cache lines and pages -- on every
// target, with no linker support needed.
//
// The profile: data\layout_hot.txt beside the tables (the game's data folder), or XRC_LAYOUT=FILE: lines
// "address weight" (any positive weights: the share of the time in each function, from a sampling profiler).
// XRC_LAYOUT=0: off (every function in its address page, as before).
//
// The clusters: C3 (Ottoni & Maher, "Optimizing function placement for large-scale data-center applications") on the
// static direct calls between the hot functions, an edge weighted by the smaller of its two ends' weights: edges by
// weight, heaviest first; a caller's cluster and its callee's merge (the callee's after the caller's) while the two
// together stay within X_HOT_CLUSTER guest bytes. The clusters then by density (weight per byte), the hot pages filled
// in that order up to X_HOT_PAGE_BYTES guest bytes each.

X_HOT_PAGE :: u32(0x10000)
X_HOT_CLUSTER :: u32(16 * 1024)
X_HOT_PAGE_BYTES :: u32(48 * 1024)
X_HOT_SHARE :: f64(0.995) // (the functions that make up this share of the weight, heaviest first)

x_hot_set: map[u32]bool

@(private = "file") hot_addrs: []u32
@(private = "file") hot_calls: [][dynamic]u32

// the hot pages, each its functions in layout order; the set of them. Empty when there is no profile.
x_layout_plan :: proc(addrs: []u32, funcs: map[u32]u32) -> (pages: [][]u32) {
	// (heap, not temp: x_par_run frees this thread's temp memory after each of its jobs)
	arena: virtual.Arena
	_ = virtual.arena_init_growing(&arena)
	defer virtual.arena_destroy(&arena)
	ta := virtual.arena_allocator(&arena)
	x_hot_set = make(map[u32]bool)
	path := os.get_env("XRC_LAYOUT", ta)
	if path == "0" {return}
	if path == "" {path = table_path("layout_hot.txt")}
	path = strings.clone(path, ta)
	txt, ok := read_text(path)
	if !ok {return}
	known := make(map[u32]bool, len(addrs), ta)
	for a in addrs {known[a] = true}
	W :: struct {a: u32, w: f64}
	ws := make([dynamic]W, ta)
	total := 0.0
	for line in strings.split_lines_iterator(&txt) {
		f := strings.fields(strings.trim_space(line), ta)
		if len(f) < 2 {continue}
		a, oka := hex_u32(f[0])
		w, okw := strconv_f64(f[1])
		if !oka || !okw || w <= 0 || !(a in known) || !(a in x_direct_ok) {continue}
		append(&ws, W{a, w})
		total += w
	}
	if len(ws) == 0 {return}
	slice.sort_by(ws[:], proc(x, y: W) -> bool {return x.w > y.w || (x.w == y.w && x.a < y.a)})
	// the hot ones: the heaviest that make up X_HOT_SHARE of it
	acc := 0.0
	n := 0
	for w in ws {
		n += 1
		acc += w.w
		if acc >= total * X_HOT_SHARE {break}
	}
	hot := ws[:n]
	weight := make(map[u32]f64, n, ta)
	for w in hot {weight[w.a] = w.w}

	// their direct calls (decoded on every core)
	hot_addrs = make([]u32, n, ta)
	for w, i in hot {hot_addrs[i] = w.a}
	hot_calls = make([][dynamic]u32, n)
	x_par_run(n, proc(i: int) {
		a := hot_addrs[i]
		hot_calls[i] = make([dynamic]u32)
		ins, bad := rx_function(a, funcs[a], context.temp_allocator)
		if bad != 0 {return}
		for &k in ins {
			mn := k.inst.mnemonic
			if (mn == .CALL || mn == .JMP) && k.inst.operand_count > 0 && k.inst.ops[0].kind == .RELATIVE {
				append(&hot_calls[i], rx_rel_target(&k, 0))
			}
		}
	})
	E :: struct {from, to: u32, w: f64}
	edges := make([dynamic]E, ta)
	for a, i in hot_addrs {
		seen := make(map[u32]bool, ta)
		for t in hot_calls[i] {
			if t == a || t in seen {continue}
			seen[t] = true
			if wt, okt := weight[t]; okt {append(&edges, E{a, t, min(weight[a], wt)})}
		}
		delete(hot_calls[i])
	}
	delete(hot_calls)
	slice.sort_by(edges[:], proc(x, y: E) -> bool {
		if x.w != y.w {return x.w > y.w}
		if x.from != y.from {return x.from < y.from}
		return x.to < y.to
	})

	// C3
	cls := make([dynamic]X_Lay_Cl, 0, n, ta)
	of := make(map[u32]int, n, ta)
	for a in hot_addrs {
		c := X_Lay_Cl{fns = make([dynamic]u32, ta), size = max(funcs[a], 1), w = weight[a]}
		append(&c.fns, a)
		of[a] = len(cls)
		append(&cls, c)
	}
	for e in edges {
		ca, cb := of[e.from], of[e.to]
		if ca == cb || cls[ca].size + cls[cb].size > X_HOT_CLUSTER {continue}
		for f in cls[cb].fns {append(&cls[ca].fns, f); of[f] = ca}
		cls[ca].size += cls[cb].size
		cls[ca].w += cls[cb].w
		cls[cb].dead = true
	}
	order := make([dynamic]int, ta)
	for c, i in cls {if !c.dead {append(&order, i)}}
	x_lay_cls = cls[:]
	defer x_lay_cls = nil
	slice.sort_by(order[:], proc(x, y: int) -> bool {
		dx, dy := x_lay_cls[x].w / f64(x_lay_cls[x].size), x_lay_cls[y].w / f64(x_lay_cls[y].size)
		if dx != dy {return dx > dy}
		return x_lay_cls[x].fns[0] < x_lay_cls[y].fns[0]
	})

	// the pages
	out := make([dynamic][]u32)
	cur := make([dynamic]u32)
	cur_size := u32(0)
	hot_bytes := u32(0)
	for ci in order {
		for f in cls[ci].fns {
			if cur_size >= X_HOT_PAGE_BYTES {
				append(&out, cur[:])
				cur = make([dynamic]u32)
				cur_size = 0
			}
			append(&cur, f)
			x_hot_set[f] = true
			cur_size += funcs[f]
			hot_bytes += funcs[f]
		}
	}
	if len(cur) > 0 {append(&out, cur[:])}
	fmt.printf("xrc: layout: %d hot functions (%.1f%% of the profile, %d KB of code) in %d pages, %d clusters (%s)\n",
		n, 100 * acc / total, hot_bytes / 1024, len(out), len(order), path)
	return out[:]
}

X_Lay_Cl :: struct {fns: [dynamic]u32, size: u32, w: f64, dead: bool}
@(private = "file") x_lay_cls: []X_Lay_Cl

// the hot pages of the last run that this one does not write (fewer pages, or no profile now): their files go, or
// their functions would be defined twice
x_layout_clean :: proc(dir: string, n_pages: int) {
	for k in n_pages ..< 4096 {
		p := fmt.tprintf("%s/page_%x.c", dir, X_HOT_PAGE + u32(k))
		if !os.exists(p) {break}
		os.remove(p)
	}
}

@(private = "file")
strconv_f64 :: proc(s: string) -> (f64, bool) {
	v := 0.0
	frac := 0.0
	seen := false
	for ch in transmute([]u8)s {
		switch {
		case ch >= '0' && ch <= '9':
			seen = true
			if frac > 0 {v += f64(ch - '0') * frac; frac /= 10} else {v = v * 10 + f64(ch - '0')}
		case ch == '.' && frac == 0: frac = 0.1
		case: return 0, false
		}
	}
	return v, seen
}

