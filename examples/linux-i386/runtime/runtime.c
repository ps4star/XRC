/* runtime.c - a minimal runtime for xrc's translation of a small Linux i386 program (demo.c), on any 64-bit Linux
 * (x86-64, ARM64, ...). It is an example of what a runtime does, kept as short as it can be:
 *
 *   - guest memory: one 4 GB block (xrc_mem), the program's segments loaded at their addresses in it
 *   - a PIE is placed where xrc placed it (xrc_load_base) and its relocations applied, as Linux's loader would
 *   - the dispatcher: a guest address -> its translation (a binary search of xrc_fns)
 *   - imports: each one the program takes from a shared library gets an address of its own, bound into its GOT slot;
 *     calls there run a native C function instead (here lib_scale and lib_name, the demo's library)
 *   - system calls (int 0x80): exit, write and set_thread_area; gs: thread-local storage (xrc_set_seg)
 *
 * A real runtime does the same at scale: all of libc (or Win32, ...) natively, threads, signals, files, graphics.
 * Usage: demo-xrc PROGRAM (the same i386 ELF file xrc translated). */
#define _GNU_SOURCE
#include <elf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "xrc_rt.h"

u8 *xrc_mem;
volatile int xrc_gil_waiters; /* (one guest thread: nobody waits) */
XRC_THREAD u32 xrc_seg_fs, xrc_seg_gs;
extern const unsigned xrc_load_base __attribute__((weak)); /* (only a PIE's table has it) */

static void die(const char *fmt, unsigned a) { fprintf(stderr, "runtime: "); fprintf(stderr, fmt, a); fprintf(stderr, "\n"); exit(127); }

/* ---- imports: native functions at addresses of their own (the top of guest memory, which the program never uses) */
#define IMP_BASE 0xfff00000u
static u32 guest_strings = 0xffe00000u; /* (strings a native function returns: guest memory for the program) */
static u32 guest_str(const char *s) { u32 a = guest_strings; strcpy((char *)xrc_mem + a, s); guest_strings += (u32)strlen(s) + 1; return a; }
static u32 arg(XCtx *c, int i) { return *(u32 *)(xrc_mem + c->esp + 4 + 4 * i); }   /* the i-th stack argument */
static void ret_cdecl(XCtx *c) { c->eip = *(u32 *)(xrc_mem + c->esp); c->esp += 4; } /* pop the return address */

static void imp_lib_scale(XCtx *c) { c->eax = arg(c, 0) * arg(c, 1); ret_cdecl(c); }
static void imp_lib_name(XCtx *c) { static u32 s; if (!s) s = guest_str("native lib"); c->eax = s; ret_cdecl(c); }
static const struct { const char *name; void (*fn)(XCtx *); } imports[] = {
	{"lib_scale", imp_lib_scale},
	{"lib_name", imp_lib_name},
};
#define NIMPORTS (sizeof imports / sizeof imports[0])

/* ---- the dispatcher */
static XFn translation_of(u32 a) {
	unsigned lo = 0, hi = xrc_nfns;
	while (lo < hi) {
		unsigned m = (lo + hi) / 2;
		if (xrc_fns[m].addr == a) return xrc_fns[m].fn;
		if (xrc_fns[m].addr < a) lo = m + 1; else hi = m;
	}
	return 0;
}
void xrc_dispatch(XCtx *c, u32 t) {
	XFn f = translation_of(t);
	if (f) { f(c); return; }
	if (t >= IMP_BASE && t < IMP_BASE + 16 * NIMPORTS) { imports[(t - IMP_BASE) / 16].fn(c); return; }
	die("a call to %08x: no function there", t);
}
/* an indirect call site's cache miss: run it, and cache a translation (imports are not cached) */
void xrc_dispatch_ic(XCtx *c, u32 t, XIc *ic) {
	u32 ra = *(u32 *)(xrc_mem + c->esp);
	XFn f = translation_of(t);
	if (f) { ic->addr = t; ic->fn = f; ic->imp = 0; }
	xrc_dispatch(c, t);
	if (c->eip != ra) xrc_bad_return(c, t);
}
void xrc_imp_enter(u32 slot) { (void)slot; }
void xrc_preempt_n(XCtx *c) { c->pre_n = 0; }
void xrc_bad_return(XCtx *c, u32 t) { die("the function at %08x returned to an unexpected address", t); (void)c; }
void xrc_stack_bad(XCtx *c, u32 t, u32 esp, u32 want) { fprintf(stderr, "runtime: esp %08x, not %08x: ", esp, want); die("after a call to %08x (XRC_STACK_VERIFY)", t); (void)c; }
void xrc_ud(XCtx *c, u32 a) { (void)c; die("an undefined or privileged instruction at %08x", a); }

/* ---- system calls and thread-local storage: Linux i386 */
struct user_desc32 { u32 entry, base, limit, flags; };
static u32 gdt_base[16]; /* (set_thread_area's descriptors: the TLS entries 6..8) */
void xrc_set_seg(XCtx *c, u32 seg, u32 sel) {
	(void)c;
	u32 base = (sel >> 3) < 16 ? gdt_base[sel >> 3] : 0;
	if (seg == 4) xrc_seg_fs = base;
	if (seg == 5) xrc_seg_gs = base;
}
void xrc_int(XCtx *c, u32 n) {
	if (n != 0x80) die("int %02x", n);
	switch (c->eax) {
	case 1: case 252: exit((int)c->ebx); /* exit, exit_group */
	case 4: c->eax = (u32)write((int)c->ebx, xrc_mem + c->ecx, c->edx); break;
	case 243: { /* set_thread_area */
		struct user_desc32 *d = (struct user_desc32 *)(xrc_mem + c->ebx);
		if (d->entry == 0xffffffffu) d->entry = 6;
		if (d->entry < 6 || d->entry > 8) { c->eax = (u32)-22; break; }
		gdt_base[d->entry] = d->base;
		c->eax = 0;
		break;
	}
	default: fprintf(stderr, "runtime: system call %u not implemented\n", c->eax); c->eax = (u32)-38; /* ENOSYS */
	}
}

/* ---- loading: the PT_LOAD segments; a PIE's relocations; imports bound into their slots */
static void load(const char *path) {
	FILE *f = fopen(path, "rb");
	if (!f) die("cannot open the program (%u)", 0);
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	u8 *d = malloc((size_t)n);
	fseek(f, 0, SEEK_SET);
	if (fread(d, 1, (size_t)n, f) != (size_t)n) die("cannot read the program (%u)", 0);
	fclose(f);
	Elf32_Ehdr *eh = (Elf32_Ehdr *)d;
	u32 bias = &xrc_load_base ? xrc_load_base : 0;
	Elf32_Phdr *ph = (Elf32_Phdr *)(d + eh->e_phoff);
	for (int i = 0; i < eh->e_phnum; i++)
		if (ph[i].p_type == PT_LOAD) memcpy(xrc_mem + bias + ph[i].p_vaddr, d + ph[i].p_offset, ph[i].p_filesz);
	/* the relocations (from the section headers, as xrc reads them) */
	Elf32_Shdr *sh = (Elf32_Shdr *)(d + eh->e_shoff);
	unsigned nimp = 0;
	for (int i = 0; i < eh->e_shnum; i++) {
		if (sh[i].sh_type != SHT_REL) continue;
		Elf32_Rel *r = (Elf32_Rel *)(d + sh[i].sh_offset);
		Elf32_Sym *syms = (Elf32_Sym *)(d + sh[sh[i].sh_link].sh_offset);
		const char *names = (const char *)(d + sh[sh[sh[i].sh_link].sh_link].sh_offset);
		for (u32 k = 0; k < sh[i].sh_size / sizeof *r; k++) {
			u32 *p = (u32 *)(xrc_mem + bias + r[k].r_offset), type = ELF32_R_TYPE(r[k].r_info);
			Elf32_Sym *s = &syms[ELF32_R_SYM(r[k].r_info)];
			if (type == R_386_RELATIVE) { *p += bias; continue; }
			if (type != R_386_JMP_SLOT && type != R_386_GLOB_DAT && type != R_386_32) continue;
			if (s->st_shndx != SHN_UNDEF) { *p = (type == R_386_32 ? *p : 0) + s->st_value + bias; continue; }
			/* an import: the native function of that name */
			unsigned j;
			for (j = 0; j < NIMPORTS; j++) if (!strcmp(imports[j].name, names + s->st_name)) break;
			if (j == NIMPORTS) { fprintf(stderr, "runtime: no native %s\n", names + s->st_name); continue; }
			*p = IMP_BASE + 16 * j;
			nimp++;
		}
	}
	(void)nimp;
	free(d);
}

int main(int argc, char **argv) {
	if (argc < 2) { fprintf(stderr, "usage: %s PROGRAM (the i386 ELF file xrc translated)\n", argv[0]); return 2; }
	xrc_mem = mmap(0, (1ull << 32) + 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
	if (xrc_mem == MAP_FAILED) die("no 4 GB of address space for guest memory (%u)", 0);
	load(argv[1]);
	/* the guest stack: 1 MB below 0xf0000000; argc = 0, and a return address nothing has (the entry never returns) */
	static XCtx c;
	c.esp = 0xf0000000u - 64;
	*(u32 *)(xrc_mem + c.esp) = 0;
	c.esp -= 4;
	*(u32 *)(xrc_mem + c.esp) = 0xfffffff0u;
	c.fcw = 0x37f;
	c.mxcsr = 0x1f80;
	XFn entry = translation_of(xrc_entry);
	if (!entry) die("no translation of the entry point %08x", xrc_entry);
	entry(&c);
	die("the program returned to %08x", c.eip);
}
