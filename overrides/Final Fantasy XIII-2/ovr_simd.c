/* ovr_simd.c (XIII-2, from FF13's) - "fast" overrides (overrides.tsv kind fast): hand-written versions of hot vector-math functions of the
 * game. The originals are unoptimized compiler output that stores every intermediate __m128 to its stack frame and
 * reads it back (~100 instructions for a matrix * vector); these compute the same values with the same operations in
 * the same order (so bit-identical, NaNs included), and write the same results. The translator calls xo_<addr>
 * directly; the dispatcher maps the address to it too (xrc_manual_simd). The stack below the caller's esp (the
 * original's dead locals) is not written.
 *
 * XRC_FAST_CHECK=1: each call also runs the translation and compares what it wrote with ours (a mismatch is
 * reported, the first few in detail). XRC_NO_FAST=1: the translations only (to compare speed); XRC_FAST_SKIP=addr,..
 * (hex) only those. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "xrc_rt.h"
#include "hle_d3d9.h" /* (the device state: 00eac800) */
#include "xrc_mem.h"

static int fast_mode = -1; /* 0 ours, 1 ours checked against the translation, 2 the translation */
static int mode_init(void) {
	fast_mode = getenv("XRC_NO_FAST") ? 2 : xrc_tool_env("XRC_FAST_CHECK") ? 1 : 0;
	return fast_mode;
}
/* (inline, as the returns below: these run tens of thousands of times a frame, mostly for a few instructions of work) */
static inline __attribute__((always_inline)) int mode(void) { return __builtin_expect(fast_mode >= 0, 1) ? fast_mode : mode_init(); }
static inline __attribute__((always_inline)) void ovr_stdcall_ret(XCtx *c, int nargs) { c->eip = M32(c->esp); c->esp += 4u + 4u * (u32)nargs; }
static inline __attribute__((always_inline)) void ovr_cdecl_ret(XCtx *c) { c->eip = M32(c->esp); c->esp += 4u; }
#define xrc_stdcall_ret ovr_stdcall_ret
#define xrc_cdecl_ret ovr_cdecl_ret
static unsigned long long n_checked, n_bad;
/* per function: the mismatches and the largest difference of a float lane in ulps (its bits as an ordered integer) */
static struct { u32 addr; const char *what; unsigned long long bad; u32 ulp; } tally[32];
static int ntally;
static s32 ord_(u32 u) { return (u & 0x80000000u) ? (s32)(0x80000000u - (u & 0x7fffffffu)) : (s32)u; }
static void tally_bad(const char *what, u32 addr, const void *ours, u32 at, unsigned n) {
	int i;
	unsigned k;
	for (i = 0; i < ntally && tally[i].addr != addr; i++) {}
	if (i == ntally) { if (ntally == 32) return; tally[i].addr = addr; tally[i].what = what; ntally++; }
	tally[i].bad++;
	for (k = 0; k + 4 <= n; k += 4) {
		long long d = (long long)ord_(*(const u32 *)((const u8 *)ours + k)) - ord_(M32(at + k));
		u32 a = (u32)(d < 0 ? (d < -0x7fffffffll ? 0x7fffffff : -d) : (d > 0x7fffffff ? 0x7fffffff : d));
		if (a > tally[i].ulp) tally[i].ulp = a;
	}
}
static void tally_print(void) {
	int i;
	for (i = 0; i < ntally; i++) fprintf(stderr, "xrc:   %08x %s: %llu mismatches, up to %u ulp\n", tally[i].addr, tally[i].what, tally[i].bad, tally[i].ulp);
}
static void check(const char *what, u32 addr, const void *ours, u32 at, unsigned n) {
	static struct { u32 addr; unsigned long long n; } per[64];
	static int nper;
	int pi;
	n_checked++;
	for (pi = 0; pi < nper && per[pi].addr != addr; pi++) {}
	if (pi == nper && nper < 64) { per[pi].addr = addr; per[pi].n = 0; nper++; }
	if (pi < 64) per[pi].n++;
	if (pi < 64 && (per[pi].n & (per[pi].n - 1)) == 0 && per[pi].n >= 1024) /* (each function's count, at powers of two) */
		fprintf(stderr, "xrc: fast check: %08x %s: %llu calls checked (%llu mismatches in all)\n", addr, what, per[pi].n, n_bad);
	if (memcmp(ours, xrc_mem + at, n)) {
		tally_bad(what, addr, ours, at, n);
		if (++n_bad <= 10) {
			unsigned k;
			fprintf(stderr, "xrc: FAST MISMATCH %08x (%s) at %08x: ours", addr, what, at);
			for (k = 0; k < n; k += 4) fprintf(stderr, " %08x", *(const u32 *)((const u8 *)ours + k));
			fprintf(stderr, ", original");
			for (k = 0; k < n; k += 4) fprintf(stderr, " %08x", M32(at + k));
			fprintf(stderr, "\n");
		}
	}
	if ((n_checked & 0xffffff) == 0) { fprintf(stderr, "xrc: fast check: %llu calls checked, %llu mismatches\n", n_checked, n_bad); tally_print(); }
}
/* XRC_FAST_SKIP=addr,addr (hex): those functions use their translations (A/B, bisecting) */
static int fast_skip(u32 a) {
	const char *e = xrc_tool_env("XRC_FAST_SKIP");
	char *q;
	while (e && *e) {
		if ((u32)strtoul(e, &q, 16) == a) return 1;
		if (q == e) break;
		e = *q ? q + 1 : q;
	}
	return 0;
}
#define ORIG(addr_) static XFn orig; if (!orig) orig = xrc_translation_of(addr_)
#define ORIGS(addr_) ORIG(addr_); { static signed char skip_ = -1; if (skip_ < 0) skip_ = (signed char)fast_skip(addr_); if (skip_) md = 2; }
/* x86's own mulps / addps / minps / maxps, operands as the original has them. With -ffast-math (its finite-math-only)
 * the compiler may swap the operands of _mm_max_ps & co: on x86 a NaN makes maxps / minps return the second operand,
 * and of two NaNs mulps / addps return the first -- swapped, the result differs. Aabb::Extended came out
 * max(p, this.max) for the original's max(this.max, p) (906 mismatches in a battle, Matrix44::Mul 12). */
#if defined(XRC_VEC) && defined(__x86_64__) && defined(__AVX__)
#define EXOP(name_, ins_) static inline X128 name_(X128 a, X128 b) { X128 r; __asm__(ins_ " %2, %1, %0" : "=x"(r.v) : "x"(a.v), "x"(b.v)); return r; }
EXOP(ex_mulps, "vmulps") EXOP(ex_addps, "vaddps") EXOP(ex_subps, "vsubps") EXOP(ex_minps, "vminps") EXOP(ex_maxps, "vmaxps")
#elif defined(XRC_VEC) && defined(__x86_64__)
#define EXOP(name_, ins_) static inline X128 name_(X128 a, X128 b) { __asm__(ins_ " %1, %0" : "+x"(a.v) : "x"(b.v)); return a; }
EXOP(ex_mulps, "mulps") EXOP(ex_addps, "addps") EXOP(ex_subps, "subps") EXOP(ex_minps, "minps") EXOP(ex_maxps, "maxps")
#else
#define ex_mulps xv_mulps
#define ex_addps xv_addps
#define ex_subps xv_subps
#define ex_minps xv_minps
#define ex_maxps xv_maxps
#endif
/* 0052d240: thiscall Vec4 *Matrix44::Transform(Vec4 *out, const Vec4 *v) -- XIII-2's (optimized: 850 call sites,
 * the hottest guest function in battle) of FF13's 004559b0: out = ((row0 * x + row1 * y) + row2 * z) + row3 * w
 * (this = 4 rows of 4 floats), mulps / addps in that order; returns out */
void XRC_CC xo_0052d240(XCtx *c) {
	u32 m = c->ecx, out = M32(c->esp + 4), v = M32(c->esp + 8);
	X128 vv, r0, r1, r2, r3, acc;
	int md = mode();
	ORIGS(0x0052d240u);
	if (md == 2) { orig(c); return; }
	XLD(vv, v);
	XLD(r0, m); XLD(r1, m + 16); XLD(r2, m + 32); XLD(r3, m + 48);
	acc = ex_mulps(r0, XV_SHUFPS(vv, vv, 0x00));
	acc = ex_addps(acc, ex_mulps(r1, XV_SHUFPS(vv, vv, 0x55)));
	acc = ex_addps(acc, ex_mulps(r2, XV_SHUFPS(vv, vv, 0xaa)));
	acc = ex_addps(acc, ex_mulps(r3, XV_SHUFPS(vv, vv, 0xff)));
	if (md == 1) { orig(c); check("Matrix44::Transform", 0x0052d240u, &acc, out, 16); return; }
	XST(out, acc);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 0048cea0: thiscall Vec4 *Matrix44::TransformPoint(Vec4 *out, const Vec4 *v) -- out = ((row1 * y + row0 * x) + row2 * z)
 * + row3 (w taken as 1), the operands in the original's order; returns out */
void XRC_CC xo_0048cea0(XCtx *c) {
	u32 m = c->ecx, out = M32(c->esp + 4), v = M32(c->esp + 8);
	X128 vv, r0, r1, r2, r3, acc;
	int md = mode();
	ORIGS(0x0048cea0u);
	if (md == 2) { orig(c); return; }
	XLD(vv, v);
	XLD(r0, m); XLD(r1, m + 16); XLD(r2, m + 32); XLD(r3, m + 48);
	acc = ex_mulps(r1, XV_SHUFPS(vv, vv, 0x55));
	acc = ex_addps(acc, ex_mulps(r0, XV_SHUFPS(vv, vv, 0x00)));
	acc = ex_addps(acc, ex_mulps(r2, XV_SHUFPS(vv, vv, 0xaa)));
	acc = ex_addps(acc, r3);
	if (md == 1) { orig(c); check("Matrix44::TransformPoint", 0x0048cea0u, &acc, out, 16); return; }
	XST(out, acc);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 00a45f70: thiscall Aabb *Aabb::Extended(Aabb *out, const Vec4 *p) -- out.min = minps(this.min, p), out.max =
 * maxps(this.max, p) (operands as the original: a NaN lane gives p's); every read before the writes; returns out */
void XRC_CC xo_00a45f70(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4), p = M32(c->esp + 8);
	X128 pv, mn, mx, r[2];
	int md = mode();
	ORIGS(0x00a45f70u);
	if (md == 2) { orig(c); return; }
	XLD(pv, p); XLD(mn, t); XLD(mx, t + 16);
	r[1] = ex_maxps(mx, pv);
	r[0] = ex_minps(mn, pv);
	if (md == 1) { orig(c); check("Aabb::Extended", 0x00a45f70u, r, out, 32); return; }
	XST(out, r[0]); XST(out + 16, r[1]);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}


/* 00525f90: thiscall Quat *Quat::Mul(Quat *out, const Quat *b) -- this (a) * b, the original's operations in its
 * order (shuffles c9 / d2 / 93 / 4e, unpckhps, shufps c4), so bit-identical; reads before the writes; returns out */
#define SH_(x_, y_, k_) XV_SHUFPS(x_, y_, k_)
void XRC_CC xo_00525f90(XCtx *c) {
	u32 q = c->ecx, out = M32(c->esp + 4), bp = M32(c->esp + 8);
	X128 a, b, x0, x1, x2, x3, x4;
	int md = mode();
	ORIGS(0x00525f90u);
	if (md == 2) { orig(c); return; }
	XLD(a, q); XLD(b, bp);
	x4 = SH_(a, a, 0xc9);
	x3 = ex_mulps(SH_(a, a, 0xff), b);
	x2 = ex_mulps(SH_(b, b, 0xff), a);
	x2 = ex_addps(x2, x3);
	x3 = ex_mulps(SH_(b, b, 0xd2), x4);
	x4 = SH_(a, a, 0xd2);
	x2 = ex_addps(x2, x3);
	x3 = ex_mulps(SH_(b, b, 0xc9), x4);
	x2 = ex_subps(x2, x3);
	x3 = ex_mulps(b, a);
	x0 = SH_(a, a, 0x93);
	x1 = ex_mulps(SH_(b, b, 0x93), x0);
	x0 = ex_addps(x1, x3);
	x0 = SH_(x0, x0, 0x4e);
	x4 = x2;
	x3 = ex_subps(x3, x1);
	x3 = ex_subps(x3, x0);
	x2 = xv_unpckhps(x2, x3);
	x4 = SH_(x4, x2, 0xc4);
	if (md == 1) { orig(c); check("Quat::Mul", 0x00525f90u, &x4, out, 16); return; }
	XST(out, x4);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}



/* ============================================================================================ the game's D3D9 proxy
 * XIII-2's (FF13's 00ec2xxx: ovr_simd.c there). The game wraps the device (flag 0x23a8644): its proxy's methods
 * (00c2fca0..00c32xxx) unwrap and forward to the real device's vtable -- ~45k calls a frame in the first battle (~800
 * draws, each ~17 SetStreamSource, 17 SetStreamSourceFreq, 16 SetSamplerState, 14 SetVertexShaderConstantF, 13
 * SetPixelShaderConstantF, 7 SetRenderState). Translated, each is a guest function between the caller's vtable call and
 * the HLE method; these do the same natively and call the device's method as a translated call would (an inline cache
 * per site, XCALL in xrc.h). Same stack writes, same registers. */
#if !defined(XRC_INGAME) && !defined(XRC_NO_IC)
static inline void px_call(XCtx *c, u32 t, XIc *ic, u32 ra) {
	if (t == ic->addr) {
		XRC_PREEMPT()
		if (ic->imp) xrc_imp_enter(ic->imp - 1u);
		XRC_DS_PUSH(t) ic->fn(c); XRC_DS_POP()
		if (c->eip != ra) xrc_bad_return(c, t);
	} else xrc_dispatch_ic(c, t, ic);
}
#else
static inline void px_call(XCtx *c, u32 t, void *ic, u32 ra) { (void)ic; (void)ra; xrc_dispatch(c, t); }
#define XIc int
#endif
/* a state call the HLE device already has (the real device behind the proxy is ours: its vtable; XRC_NO_STATE_SKIP=1
 * turns it off): its HLE method would store the very values it holds (hle_d3d9.c: v_SetRenderState & co, bind()), so
 * it returns D3D_OK at once, as that method would */
static int px_noskip = -1;
static inline int px_hle(u32 dv) {
	if (px_noskip < 0) px_noskip = getenv("XRC_NO_STATE_SKIP") != 0;
	return !px_noskip && dv == the_dev && dv && M32(dv) == d3_ifaces[1]->vtable;
}
/* XRC_STATE_SKIP_STATS=1: calls and skips per proxy method (k: its vtable offset % 64 / 4; 14 BindStreams' streams,
 * 15 its stream 7 pair), every 2^20 calls */
static int px_stats = -1;
static unsigned long long px_n[16], px_sk[16];
static void px_stat_(int k, int skip);
static inline __attribute__((always_inline)) void px_stat(int k, int skip) { if (px_stats) px_stat_(k, skip); } /* (off: one test) */
static void px_stat_(int k, int skip) {
	static unsigned long long tot;
	if (px_stats < 0) px_stats = getenv("XRC_STATE_SKIP_STATS") != 0;
	if (!px_stats) return;
	px_n[k]++; px_sk[k] += (unsigned)skip;
	if ((++tot & 0xfffff) == 0) {
		int i;
		fprintf(stderr, "xrc: state skips:");
		for (i = 0; i < 16; i++) if (px_n[i]) fprintf(stderr, " %d:%llu/%llu", i, px_sk[i], px_n[i]);
		fprintf(stderr, "\n");
	}
}
static inline u32 px_sampler(u32 s) { return s >= 257 && s <= 260 ? 16 + (s - 257) : s; }
/* the plain forwarders: [esp+4] = this->real (this+8); jmp real->vtbl[off] (a tail jump: the method returns to the
 * caller). eax = the method, ecx = the vtable, as the original leaves them. SKIP: the call is a no-op (PA(k): its
 * arguments) */
#define PA(k) M32(c->esp + 4u + 4u * (k))
#define PX_FWD(addr_, off_, nargs_, SKIP) PX_FWDK(addr_, off_, nargs_, SKIP, (off_ % 64) / 4)
#define PX_FWDK(addr_, off_, nargs_, SKIP, k_)                                                                   \
	void XRC_CC xo_##addr_(XCtx *c) {                                                                     \
		static XIc ic = {0xffffffffu, 0, 0};                                                              \
		if (mode() == 2) { ORIG(0x##addr_##u); orig(c); return; }                                         \
		u32 dv = M32(M32(c->esp + 4) + 8), vt = M32(dv), t = M32(vt + (off_));                           \
		M32(c->esp + 4) = dv;                                                                             \
		c->eax = t; c->ecx = vt;                                                                          \
		{ int sk_ = (nargs_) && px_hle(dv) && (SKIP); px_stat((k_) & 15, sk_);                           \
		if (sk_) { c->eax = 0; xrc_stdcall_ret(c, (nargs_)); return; } }                                \
		px_call(c, t, &ic, M32(c->esp));                                                                  \
	}
#define PX_CONSTS(arr_, max_, w_) (PA(1) + PA(3) <= (max_) && xrc_memeq(&dev.arr_[PA(1) * (w_)], xrc_mem + PA(2), (size_t)PA(3) * (w_) * 4u))
PX_FWD(00c30500, 0x178, 4, PX_CONSTS(vsf, 256, 4)) /* SetVertexShaderConstantF(reg, data, n) */
PX_FWD(00c301f0, 0x114, 4, (px_sampler(PA(1)) < 32 && PA(2) < 16 && dev.ss[px_sampler(PA(1))][PA(2)] == PA(3))) /* SetSamplerState */
PX_FWD(00c305c0, 0x198, 3, (PA(1) < 16 && dev.freq[PA(1)] == PA(2))) /* SetStreamSourceFreq */
PX_FWD(00c30670, 0x1b4, 4, PX_CONSTS(psf, 224, 4)) /* SetPixelShaderConstantF */
PX_FWD(00c30580, 0x188, 4, PX_CONSTS(vsb, 16, 1)) /* SetVertexShaderConstantB */
PX_FWD(00c30450, 0x15c, 2, (dev.decl == PA(1))) /* SetVertexDeclaration */
PX_FWD(00c30470, 0x164, 2, (dev.fvf == PA(1))) /* SetFVF */

/* 00c300b0: SetRenderState(this, state, value): state 0xae (D3DRS_SCISSORTESTENABLE) also noted at 0x51732a5;
 * push esi; real->vtbl[0xe4](real, state, value); pop esi; ret 0xc */
void XRC_CC xo_00c300b0(XCtx *c) {
	static XIc ic = {0xffffffffu, 0, 0};
	if (mode() == 2) { ORIG(0x00c300b0u); orig(c); return; }
	u32 sp = c->esp, st = M32(sp + 8), val = M32(sp + 12), dv = M32(M32(sp + 4) + 8), vt = M32(dv), t = M32(vt + 0xe4);
	u32 esi = c->esi;
	if (st == 0xaeu) M8(0x051732a5u) = val != 0;
	M32(sp - 4) = esi; M32(sp - 8) = val; M32(sp - 12) = st; M32(sp - 16) = dv; M32(sp - 20) = 0x00c300dfu;
	{ int sk = st < 256 && px_hle(dv) && dev.rs[st] == val; px_stat((0xe4 % 64) / 4, sk);
	if (sk) { c->ecx = t; c->edx = val; c->eax = 0; c->eip = M32(sp); c->esp = sp + 16; return; } } /* (as after the call: D3D_OK) */
	c->esp = sp - 20; c->ecx = t; c->edx = val; c->eax = dv; c->esi = vt;
	px_call(c, t, &ic, 0x00c300dfu);
	c->esi = esi;
	c->eip = M32(sp); /* (pop esi; ret 0xc) */
	c->esp = sp + 16;
}
/* the wrapped resources: a proxy object holds the marker 0xb00beed0 among its first 32 dwords, the real object
 * right after it (with the proxy off -- flag 0x23a8644 clear -- the argument is the real object already) */
static inline u32 px_real(u32 o) {
	u32 k;
	if (!M8(0x023a8644u)) return o;
	if (!o) return 0;
	for (k = 0; k < 0x20; k++) if (M32(o + 4 * k) == 0xb00beed0u) return M32(o + 4 * k + 4);
	return 0;
}
/* 00c310e0: SetStreamSource(this, stream, vb, offset, stride): the game's own record first -- [0x51932c0 + 4 stream] =
 * vb, [0x51932b0] |= 4 --, then push esi; push edi; real->vtbl[0x190](real, stream, real vb, offset, stride); pop edi;
 * pop esi; ret 0x14. A call the device already has skipped: the record is still made. */
static inline void px_ss_note(u32 strm, u32 vb) { M32(strm * 4u + 0x051932c0u) = vb; M32(0x051932b0u) |= 4u; }
void XRC_CC xo_00c310e0(XCtx *c) {
	static XIc ic = {0xffffffffu, 0, 0};
	if (mode() == 2) { ORIG(0x00c310e0u); orig(c); return; }
	u32 sp = c->esp, strm = M32(sp + 8), vb = M32(sp + 12), off = M32(sp + 16), stride = M32(sp + 20), rv, dv, vt, t;
	u32 esi = c->esi, edi = c->edi;
	px_ss_note(strm, vb);
	rv = px_real(vb);
	dv = M32(M32(sp + 4) + 8); vt = M32(dv); t = M32(vt + 0x190);
	M32(sp - 4) = esi; M32(sp - 8) = edi; M32(sp - 12) = stride; M32(sp - 16) = off; M32(sp - 20) = rv; M32(sp - 24) = strm;
	M32(sp - 28) = dv; M32(sp - 32) = 0x00c31141u;
	{ int sk = strm < 16 && px_hle(dv) && dev.stream[strm][0] == rv && dev.stream[strm][1] == off && dev.stream[strm][2] == stride;
	px_stat((0x190 % 64) / 4, sk);
	if (sk) { c->ecx = dv; c->edx = t; c->eax = 0; c->eip = M32(sp); c->esp = sp + 24; return; } }
	c->esp = sp - 32; c->eax = rv; c->ecx = dv; c->edx = t; c->esi = vt; c->edi = off;
	px_call(c, t, &ic, 0x00c31141u);
	c->esi = esi; c->edi = edi;
	c->eip = M32(sp); /* (pop edi; pop esi; ret 0x14) */
	c->esp = sp + 24;
}

/* 00c1d710: cdecl Render::BindStreams(const Decl *d) -- every draw (~800 a frame in the first battle; FF13's 00eac800,
 * ovr_simd.c there): for the 16 streams of the table at 0x516b41c (16 bytes each: +0 the buffer object -- its D3D
 * buffers at +0x10, one per [0x516ba9c] --, +4 the offset, +8 the stride), dev->SetStreamSource(i, buffer, offset,
 * stride) and dev->SetStreamSourceFreq(i, 1) (dev = *0x516ba88, the game's proxy device); then stream 7 again:
 * SetStreamSource(7, [[0x516baa0] + 0x184], 0, 4), SetStreamSourceFreq(7, 0x10000); then SetVertexDeclaration(d->+4),
 * or SetFVF(0) without one; returns what that returns. Native, with one difference: when the proxy forwards to the HLE
 * device, a stream whose binding the device already has (the buffer the proxy would pass, the offset, the stride,
 * frequency 1) is not set again -- both calls would leave every state as it is (bind() keeps the same references) --
 * and neither is stream 7 when the device already has what the two calls after the loop set it to (the loop's pair
 * and those two would leave it as it is: nothing between them reads the device). The proxy's own record of each
 * SetStreamSource (px_ss_note) is still made, in the same order. XRC_NO_STREAM_SKIP=1: every call made. */
static XIc ic_ss, ic_ssf, ic_ss7, ic_ssf7, ic_dcl;
void XRC_CC xo_00c1d710(XCtx *c) {
	u32 sp = c->esp, arg = M32(sp + 4), ebx = c->ebx, esi = c->esi, edi = c->edi, i, d, vt, hle = 0;
	u32 vb7 = 0, rv7 = 0, skip7 = 0;
	static int noskip = -1;
	if (mode() == 2) { ORIG(0x00c1d710u); orig(c); return; }
	if (noskip < 0) noskip = getenv("XRC_NO_STREAM_SKIP") != 0;
	/* push ebx; push esi; push edi */
	M32(sp - 4) = ebx; M32(sp - 8) = esi; M32(sp - 12) = edi;
	d = M32(0x0516ba88u); vt = M32(d);
	if (!noskip && M32(vt + 0x190) == 0x00c310e0u && M32(vt + 0x198) == 0x00c305c0u) {
		u32 real = M32(d + 8);
		hle = real && real == the_dev && M32(real) == d3_ifaces[1]->vtable;
	}
	if (hle) { /* (stream 7 after the loop: what the device has now already?) */
		vb7 = M32(M32(0x0516baa0u) + 0x184); rv7 = px_real(vb7);
		skip7 = dev.stream[7][0] == rv7 && dev.stream[7][1] == 0 && dev.stream[7][2] == 4 && dev.freq[7] == 0x10000u;
	}
	for (i = 0; i < 16; i++) {
		u32 tb = 0x0516b41cu + 16u * i, o = M32(tb), vb = o ? M32(o + 4u * M32(0x0516ba9cu) + 0x10) : 0, off = M32(tb + 4), stride = M32(tb + 8);
		if (hle) {
			int sk = i == 7 ? (int)skip7 : (dev.stream[i][0] == px_real(vb) && dev.stream[i][1] == off && dev.stream[i][2] == stride && dev.freq[i] == 1);
			px_stat(14, sk);
			if (sk) { px_ss_note(i, vb); continue; }
		}
		d = M32(0x0516ba88u); vt = M32(d);
		/* push stride; push offset; push vb; push i; push dev; call [vt+0x190] */
		M32(sp - 16) = stride; M32(sp - 20) = off; M32(sp - 24) = vb; M32(sp - 28) = i; M32(sp - 32) = d; M32(sp - 36) = 0x00c1d74eu;
		c->esp = sp - 36; c->eax = M32(vt + 0x190); c->ecx = vb; c->edx = vt; c->ebx = off; c->esi = i; c->edi = tb;
		px_call(c, M32(vt + 0x190), &ic_ss, 0x00c1d74eu);
		d = M32(0x0516ba88u); vt = M32(d);
		/* push 1; push i; push dev; call [vt+0x198] */
		M32(sp - 16) = 1; M32(sp - 20) = i; M32(sp - 24) = d; M32(sp - 28) = 0x00c1d761u;
		c->esp = sp - 28; c->eax = d; c->ecx = vt; c->edx = M32(vt + 0x198); c->esi = i; c->edi = tb;
		px_call(c, M32(vt + 0x198), &ic_ssf, 0x00c1d761u);
	}
	px_stat(15, (int)skip7);
	if (skip7) px_ss_note(7, vb7);
	else {
		u32 x = M32(M32(0x0516baa0u) + 0x184);
		d = M32(0x0516ba88u); vt = M32(d);
		/* push 4; push 0; push x; push 7; push dev; call [vt+0x190] */
		M32(sp - 16) = 4; M32(sp - 20) = 0; M32(sp - 24) = x; M32(sp - 28) = 7; M32(sp - 32) = d; M32(sp - 36) = 0x00c1d78du;
		c->esp = sp - 36; c->eax = M32(vt + 0x190); c->ecx = vt; c->edx = x; c->esi = 16; c->edi = 0x0516b41cu + 256u;
		px_call(c, M32(vt + 0x190), &ic_ss7, 0x00c1d78du);
		d = M32(0x0516ba88u); vt = M32(d);
		/* push 0x10000; push 7; push dev; call [vt+0x198] */
		M32(sp - 16) = 0x10000u; M32(sp - 20) = 7; M32(sp - 24) = d; M32(sp - 28) = 0x00c1d7a4u;
		c->esp = sp - 28; c->eax = d; c->ecx = vt; c->edx = M32(vt + 0x198);
		px_call(c, M32(vt + 0x198), &ic_ssf7, 0x00c1d7a4u);
	}
	c->ebx = ebx; c->esi = esi; c->edi = edi; /* (pop edi; pop esi; pop ebx) */
	d = M32(0x0516ba88u); vt = M32(d);
	if (arg) { /* push d->+4; push dev; call [vt+0x15c] */
		M32(sp - 4) = M32(arg + 4); M32(sp - 8) = d; M32(sp - 12) = 0x00c1d7c4u;
		c->esp = sp - 12; c->eax = M32(arg + 4); c->ecx = M32(vt + 0x15c); c->edx = vt;
		px_call(c, M32(vt + 0x15c), &ic_dcl, 0x00c1d7c4u);
	} else { /* push 0; push dev; call [vt+0x164] */
		M32(sp - 4) = 0; M32(sp - 8) = d; M32(sp - 12) = 0x00c1d7d7u;
		c->esp = sp - 12; c->eax = M32(vt + 0x164); c->edx = vt;
		px_call(c, M32(vt + 0x164), &ic_dcl, 0x00c1d7d7u);
	}
	c->esp = sp; /* (ret: cdecl) */
	xrc_cdecl_ret(c);
}

const XrcManual xrc_manual_simd[] = {
	{0x00525f90u, xo_00525f90, "Quat::Mul (fast)"},
	{0x00c30500u, xo_00c30500, "D3D proxy SetVertexShaderConstantF (fast)"},
	{0x00c301f0u, xo_00c301f0, "D3D proxy SetSamplerState (fast)"},
	{0x00c305c0u, xo_00c305c0, "D3D proxy SetStreamSourceFreq (fast)"},
	{0x00c30670u, xo_00c30670, "D3D proxy SetPixelShaderConstantF (fast)"},
	{0x00c30580u, xo_00c30580, "D3D proxy SetVertexShaderConstantB (fast)"},
	{0x00c30450u, xo_00c30450, "D3D proxy SetVertexDeclaration (fast)"},
	{0x00c30470u, xo_00c30470, "D3D proxy SetFVF (fast)"},
	{0x00c300b0u, xo_00c300b0, "D3D proxy SetRenderState (fast)"},
	{0x00c310e0u, xo_00c310e0, "D3D proxy SetStreamSource (fast)"},
	{0x00c1d710u, xo_00c1d710, "Render::BindStreams (fast)"},
	{0x00a45f70u, xo_00a45f70, "Aabb::Extended (fast)"},
	{0x0048cea0u, xo_0048cea0, "Matrix44::TransformPoint (fast)"},
	{0x0052d240u, xo_0052d240, "Matrix44::Transform (fast)"},
};
const unsigned xrc_manual_simd_n = sizeof xrc_manual_simd / sizeof xrc_manual_simd[0];
