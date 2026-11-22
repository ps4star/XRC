/* ovr_simd.c - "fast" overrides (overrides.tsv kind fast): hand-written versions of hot vector-math functions of the
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
/* (Matrix44::Transform / Mul, Aabb::Extended and Quat::FromMatrix33 were off for a while: XRC_FAST_CHECK found them
 * off the translation -- the operand swaps above, and FromMatrix33 by 1-2 ulp: fast math had made the translation's
 * exact 1 / sqrt for rsqrtps an rsqrtps estimate, differently inlined in each (-mrecip=none since, tools/xrc_build.sh)
 * -- and with them on a character flickered out of sight in Chapter 1. Since: 0 mismatches in 470M checked calls.) */

/* 004559b0: thiscall Vec4 *Matrix44::Transform(Vec4 *out, const Vec4 *v) -- out = v.x * row0 + v.y * row1 +
 * v.z * row2 + v.w * row3 (this = 4 rows of 4 floats); ((row0 * x + row1 * y) + row2 * z) + row3 * w; returns out */
void XRC_CC xo_004559b0(XCtx *c) {
	u32 m = c->ecx, out = M32(c->esp + 4), v = M32(c->esp + 8);
	X128 vv, r0, r1, r2, r3, acc;
	int md = mode();
	ORIGS(0x004559b0u);
	if (md == 2) { orig(c); return; }
	XLD(vv, v);
	XLD(r0, m); XLD(r1, m + 16); XLD(r2, m + 32); XLD(r3, m + 48);
	acc = ex_mulps(r0, XV_SHUFPS(vv, vv, 0x00));
	acc = ex_addps(acc, ex_mulps(r1, XV_SHUFPS(vv, vv, 0x55)));
	acc = ex_addps(acc, ex_mulps(r2, XV_SHUFPS(vv, vv, 0xaa)));
	acc = ex_addps(acc, ex_mulps(r3, XV_SHUFPS(vv, vv, 0xff)));
	if (md == 1) { orig(c); check("Matrix44::Transform", 0x004559b0u, &acc, out, 16); return; }
	XST(out, acc);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 00455840: thiscall Vec4 *Vec4::operator=(const Vec4 *v) -- 4 floats copied through the x87 (fld / fstp: a
 * signaling NaN comes out quiet); returns this */
static u32 x87_copy(u32 u) { return (u & 0x7f800000u) == 0x7f800000u && (u & 0x7fffffu) && !(u & 0x400000u) ? u | 0x400000u : u; }
void XRC_CC xo_00455840(XCtx *c) {
	u32 t = c->ecx, v = M32(c->esp + 4), r[4];
	int md = mode(), k;
	ORIGS(0x00455840u);
	if (md == 2) { orig(c); return; }
	if (md == 1) {
		for (k = 0; k < 4; k++) r[k] = x87_copy(M32(v + 4 * k));
		orig(c); check("Vec4::operator=", 0x00455840u, r, t, 16); return;
	}
	for (k = 0; k < 4; k++) M32(t + 4 * k) = x87_copy(M32(v + 4 * k)); /* (each read just before its store, as the original) */
	c->eax = t;
	xrc_stdcall_ret(c, 1);
}

/* 00455880: thiscall Vec4 *Vec4::Get(Vec4 *out) -- out = this (16 bytes); returns out */
void XRC_CC xo_00455880(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4);
	X128 x;
	int md = mode();
	ORIGS(0x00455880u);
	if (md == 2) { orig(c); return; }
	XLD(x, t);
	if (md == 1) { orig(c); check("Vec4::Get", 0x00455880u, &x, out, 16); return; }
	XST(out, x);
	c->eax = out;
	xrc_stdcall_ret(c, 1);
}

/* 004586c0: thiscall Aabb *Aabb::Extended(Aabb *out, const Vec4 *p) -- out.min = minps(this.min, p), out.max =
 * maxps(this.max, p) (this.min at +0, max at +16); returns out */
void XRC_CC xo_004586c0(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4), p = M32(c->esp + 8);
	X128 pv, r[2], a;
	int md = mode();
	ORIGS(0x004586c0u);
	if (md == 2) { orig(c); return; }
	XLD(pv, p); XLD(a, t + 16); r[1] = ex_maxps(a, pv);
	XLD(a, t); r[0] = ex_minps(a, pv);
	if (md == 1) { orig(c); check("Aabb::Extended", 0x004586c0u, r, out, 32); return; }
	XST(out, r[0]); XST(out + 16, r[1]);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 00710860: thiscall Vec4 *Vec4::Dot4(Vec4 *out, const Vec4 *v) -- p = this * v; out = ((p + p.yyyy) + p.zzzz) +
 * p.wwww (lane 0: the dot product); returns out */
void XRC_CC xo_00710860(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4), v = M32(c->esp + 8);
	X128 a, b, pr, acc;
	int md = mode();
	ORIGS(0x00710860u);
	if (md == 2) { orig(c); return; }
	XLD(b, v); XLD(a, t);
	pr = xv_mulps(a, b);
	acc = xv_addps(pr, XV_SHUFPS(pr, pr, 0x55));
	acc = xv_addps(acc, XV_SHUFPS(pr, pr, 0xaa));
	acc = xv_addps(acc, XV_SHUFPS(pr, pr, 0xff));
	if (md == 1) { orig(c); check("Vec4::Dot4", 0x00710860u, &acc, out, 16); return; }
	XST(out, acc);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 00711f40: thiscall Vec4 *Matrix33::Transform(Vec4 *out, const Vec4 *v) -- out = (row0 * v.x + row1 * v.y) +
 * row2 * v.z (this = 3 rows of 4 floats); returns out */
void XRC_CC xo_00711f40(XCtx *c) {
	u32 m = c->ecx, out = M32(c->esp + 4), v = M32(c->esp + 8);
	X128 vv, r0, r1, r2, acc;
	int md = mode();
	ORIGS(0x00711f40u);
	if (md == 2) { orig(c); return; }
	XLD(vv, v); XLD(r0, m); XLD(r1, m + 16); XLD(r2, m + 32);
	acc = xv_mulps(r0, XV_SHUFPS(vv, vv, 0x00));
	acc = xv_addps(acc, xv_mulps(r1, XV_SHUFPS(vv, vv, 0x55)));
	acc = xv_addps(acc, xv_mulps(r2, XV_SHUFPS(vv, vv, 0xaa)));
	if (md == 1) { orig(c); check("Matrix33::Transform", 0x00711f40u, &acc, out, 16); return; }
	XST(out, acc);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 00455f00: thiscall Vec4 *Matrix44::Transform(Vec4 *out, const __m128 *v) -- as 004559b0 (v read with movaps) */
void XRC_CC xo_00455f00(XCtx *c) {
	u32 m = c->ecx, out = M32(c->esp + 4), v = M32(c->esp + 8);
	X128 vv, r0, r1, r2, r3, acc;
	int md = mode();
	ORIGS(0x00455f00u);
	if (md == 2) { orig(c); return; }
	XLD(vv, v);
	XLD(r0, m); XLD(r1, m + 16); XLD(r2, m + 32); XLD(r3, m + 48);
	acc = ex_mulps(r0, XV_SHUFPS(vv, vv, 0x00));
	acc = ex_addps(acc, ex_mulps(r1, XV_SHUFPS(vv, vv, 0x55)));
	acc = ex_addps(acc, ex_mulps(r2, XV_SHUFPS(vv, vv, 0xaa)));
	acc = ex_addps(acc, ex_mulps(r3, XV_SHUFPS(vv, vv, 0xff)));
	if (md == 1) { orig(c); check("Matrix44::Transform(m128)", 0x00455f00u, &acc, out, 16); return; }
	XST(out, acc);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 004584c0: thiscall Vec4 *Matrix44::TransformPoint(Vec4 *out, const __m128 *v) -- ((row0 * v.x + row1 * v.y) +
 * row2 * v.z) + row3; returns out */
void XRC_CC xo_004584c0(XCtx *c) {
	u32 m = c->ecx, out = M32(c->esp + 4), v = M32(c->esp + 8);
	X128 vv, r0, r1, r2, r3, acc;
	int md = mode();
	ORIGS(0x004584c0u);
	if (md == 2) { orig(c); return; }
	XLD(vv, v);
	XLD(r0, m); XLD(r1, m + 16); XLD(r2, m + 32); XLD(r3, m + 48);
	acc = xv_mulps(r0, XV_SHUFPS(vv, vv, 0x00));
	acc = xv_addps(acc, xv_mulps(r1, XV_SHUFPS(vv, vv, 0x55)));
	acc = xv_addps(acc, xv_mulps(r2, XV_SHUFPS(vv, vv, 0xaa)));
	acc = xv_addps(acc, r3);
	if (md == 1) { orig(c); check("Matrix44::TransformPoint", 0x004584c0u, &acc, out, 16); return; }
	XST(out, acc);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 00455d80: thiscall Matrix44 *Matrix44::Get(Matrix44 *out) -- out = this (all 64 bytes read before any is
 * written); returns out */
void XRC_CC xo_00455d80(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4);
	X128 r[4];
	int md = mode(), k;
	ORIGS(0x00455d80u);
	if (md == 2) { orig(c); return; }
	for (k = 0; k < 4; k++) XLD(r[k], t + 16 * k);
	if (md == 1) { orig(c); check("Matrix44::Get", 0x00455d80u, r, out, 64); return; }
	for (k = 0; k < 4; k++) XST(out + 16 * k, r[k]);
	c->eax = out;
	xrc_stdcall_ret(c, 1);
}

/* 00462f80: thiscall Matrix44 *Matrix44::operator=(const Matrix44 *m) -- row by row (each read, then written);
 * returns this */
void XRC_CC xo_00462f80(XCtx *c) {
	u32 t = c->ecx, m = M32(c->esp + 4);
	X128 r[4];
	int md = mode(), k;
	ORIGS(0x00462f80u);
	if (md == 2) { orig(c); return; }
	if (md == 1) {
		u8 sim[64];
		/* the result of the row-by-row copy, overlap included */
		memcpy(sim, xrc_mem + t, 64);
		for (k = 0; k < 4; k++) {
			X128 x; u32 sa = m + 16 * k; unsigned j;
			for (j = 0; j < 16; j++) {
				u32 b = sa + j; ((u8 *)&x)[j] = b - t < 64 ? sim[b - t] : M8(b);
			}
			memcpy(sim + 16 * k, &x, 16);
		}
		(void)r;
		orig(c); check("Matrix44::operator=", 0x00462f80u, sim, t, 64); return;
	}
	for (k = 0; k < 4; k++) { XLD(r[k], m + 16 * k); XST(t + 16 * k, r[k]); }
	c->eax = t;
	xrc_stdcall_ret(c, 1);
}

/* 00457e50: thiscall Aabb *Aabb::Transformed(Aabb *out, const Matrix44 *m) -- the box of the 8 transformed corners:
 * d = max - min; A = row0 * d.x, B = row1 * d.y, C = row2 * d.z; P = m->TransformPoint(min) (004584c0); a box at P
 * extended (004586c0: minps / maxps) by P+A, P+B, (P+A)+B, P+C, (P+A)+C, (P+B)+C, ((P+A)+B)+C in that order (the
 * sums by 00458460, this + v); returns out. The original's temporaries are below the caller's esp (not written). */
static void ext(X128 *b, X128 p) { b[0] = xv_minps(b[0], p); b[1] = xv_maxps(b[1], p); }
void XRC_CC xo_00457e50(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4), m = M32(c->esp + 8);
	X128 mn, mx, d, r0, r1, r2, r3, A, B, C, P, PA, PB, PAB, b[2];
	int md = mode();
	ORIGS(0x00457e50u);
	if (md == 2) { orig(c); return; }
	XLD(mn, t); XLD(mx, t + 16);
	XLD(r0, m); XLD(r1, m + 16); XLD(r2, m + 32); XLD(r3, m + 48);
	d = xv_subps(mx, mn);
	A = xv_mulps(r0, XV_SHUFPS(d, d, 0x00));
	B = xv_mulps(r1, XV_SHUFPS(d, d, 0x55));
	C = xv_mulps(r2, XV_SHUFPS(d, d, 0xaa));
	P = xv_mulps(r0, XV_SHUFPS(mn, mn, 0x00)); /* (TransformPoint, as xo_004584c0) */
	P = xv_addps(P, xv_mulps(r1, XV_SHUFPS(mn, mn, 0x55)));
	P = xv_addps(P, xv_mulps(r2, XV_SHUFPS(mn, mn, 0xaa)));
	P = xv_addps(P, r3);
	PA = xv_addps(P, A); PB = xv_addps(P, B); PAB = xv_addps(PA, B);
	b[0] = P; b[1] = P;
	ext(b, PA); ext(b, PB); ext(b, PAB); ext(b, xv_addps(P, C));
	ext(b, xv_addps(PA, C)); ext(b, xv_addps(PB, C)); ext(b, xv_addps(PAB, C));
	if (md == 1) { orig(c); check("Aabb::Transformed", 0x00457e50u, b, out, 32); return; }
	XST(out, b[0]); XST(out + 16, b[1]);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 00456970: thiscall Matrix44 *Matrix44::Mul(Matrix44 *out, const Matrix44 *n) -- out row i = this->Transform(n row i)
 * (004559b0), rows 3, 2, 1, 0 computed into temporaries before any is written (out may be this or n); returns out */
static X128 m44_transform(const X128 *r, X128 v) {
	X128 acc = ex_mulps(r[0], XV_SHUFPS(v, v, 0x00));
	acc = ex_addps(acc, ex_mulps(r[1], XV_SHUFPS(v, v, 0x55)));
	acc = ex_addps(acc, ex_mulps(r[2], XV_SHUFPS(v, v, 0xaa)));
	return ex_addps(acc, ex_mulps(r[3], XV_SHUFPS(v, v, 0xff)));
}
void XRC_CC xo_00456970(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4), n = M32(c->esp + 8);
	X128 r[4], v, o[4];
	int md = mode(), k;
	ORIGS(0x00456970u);
	if (md == 2) { orig(c); return; }
	XLD(r[0], t); XLD(r[1], t + 16); XLD(r[2], t + 32); XLD(r[3], t + 48);
	for (k = 3; k >= 0; k--) { XLD(v, n + 16u * (u32)k); o[k] = m44_transform(r, v); }
	if (md == 1) { orig(c); check("Matrix44::Mul", 0x00456970u, o, out, 64); return; }
	for (k = 0; k < 4; k++) XST(out + 16u * (u32)k, o[k]);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 00457750: thiscall Matrix44 *(Matrix44 *out, const Matrix44 *n) -- the same as 00456970 (a second copy of it:
 * four calls of 00455f00 for rows 3, 2, 1, 0 into temporaries, then the four copied to out); returns out */
void XRC_CC xo_00457750(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4), n = M32(c->esp + 8);
	X128 r[4], v, o[4];
	int md = mode(), k;
	ORIGS(0x00457750u);
	if (md == 2) { orig(c); return; }
	XLD(r[0], t); XLD(r[1], t + 16); XLD(r[2], t + 32); XLD(r[3], t + 48);
	for (k = 3; k >= 0; k--) { XLD(v, n + 16u * (u32)k); o[k] = m44_transform(r, v); }
	if (md == 1) { orig(c); check("Matrix44::Mul2", 0x00457750u, o, out, 64); return; }
	for (k = 0; k < 4; k++) XST(out + 16u * (u32)k, o[k]);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 0074dbf0: thiscall Vec4 *(Vec4 *out, const Vec4 *v) -- the same as 004559b0, Matrix44::Transform (out written with
 * four movss: the same bits); returns out */
void XRC_CC xo_0074dbf0(XCtx *c) {
	u32 m = c->ecx, out = M32(c->esp + 4), v = M32(c->esp + 8);
	X128 r[4], vv, acc;
	int md = mode();
	ORIGS(0x0074dbf0u);
	if (md == 2) { orig(c); return; }
	XLD(vv, v);
	XLD(r[0], m); XLD(r[1], m + 16); XLD(r[2], m + 32); XLD(r[3], m + 48);
	acc = m44_transform(r, vv);
	if (md == 1) { orig(c); check("Matrix44::Transform2", 0x0074dbf0u, &acc, out, 16); return; }
	XST(out, acc);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 00456f30: thiscall Vec4 *Vec4::Dot3(Vec4 *out, const Vec4 *v) -- p = this * v; out = (p + p.zzzz) + p.yyyy (lane 0:
 * the 3-component dot product; out written with movaps); returns out */
void XRC_CC xo_00456f30(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4), v = M32(c->esp + 8);
	X128 a, b, pr, acc;
	int md = mode();
	ORIGS(0x00456f30u);
	if (md == 2) { orig(c); return; }
	XLD(b, v); XLD(a, t);
	pr = ex_mulps(a, b);
	acc = ex_addps(pr, XV_SHUFPS(pr, pr, 0xaa));
	acc = ex_addps(acc, XV_SHUFPS(pr, pr, 0x55));
	if (md == 1) { orig(c); check("Vec4::Dot3", 0x00456f30u, &acc, out, 16); return; }
	XST(out, acc);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 00536990: thiscall Matrix44 *Matrix44::Inverse(Matrix44 *out) -- the SSE adjugate inverse (Intel's AP-928 form:
 * cofactors from row pairs, determinant = dot of row 0 and its cofactors, 1/det by 00536830, rcpps and one Newton
 * step); the original's operations in its order, generated from its code (xmm and stack slots followed as values);
 * all of this is read before out is written (out may be this); returns out */
static X128 xv_movlhps_(X128 a, X128 b) { a.q[1] = b.q[0]; return a; }
static X128 xv_movhlps_(X128 a, X128 b) { a.q[0] = b.q[1]; return a; }
/* 00536830: r = rcpps(x) (the translation's exact 1 / x); r + r * (1 - x * r), its 1 the float at 010aa950 */
static X128 rcp_nr(X128 x) {
	X128 r = xv_rcpps(x, x), one;
	u32 k = M32(0x010aa950u);
	one.u[0] = one.u[1] = one.u[2] = one.u[3] = k;
	return ex_addps(ex_mulps(ex_subps(one, ex_mulps(x, r)), r), r);
}
void XRC_CC xo_00536990(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4);
	X128 v0, v11, v22, v33, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91, v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107, v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121, v122, v123, v124, v125, v126, v127, v128, v129, v130, v131, v132, v133, v134, v135, v136, v137, v138;
	int md = mode();
	ORIGS(0x00536990u);
	if (md == 2) { orig(c); return; }
	v0.u[0] = v0.u[1] = v0.u[2] = v0.u[3] = 0;
	XLD(v33, t + 32u);
	v47 = XV_SHUFPS(v33, v33, 0xd8);
	XLD(v44, t + 48u);
	v48 = XV_SHUFPS(v44, v44, 0xd8);
	v50 = xv_unpcklps(v47, v48);
	XLD(v11, t + 0u);
	v45 = XV_SHUFPS(v11, v11, 0xd8);
	XLD(v22, t + 16u);
	v46 = XV_SHUFPS(v22, v22, 0xd8);
	v49 = xv_unpcklps(v45, v46);
	v55 = xv_movhlps_(v50, v49);
	v81 = XV_SHUFPS(v55, v55, 0x4e);
	v52 = xv_unpckhps(v47, v48);
	v51 = xv_unpckhps(v45, v46);
	v54 = xv_movlhps_(v52, v51);
	v80 = XV_SHUFPS(v54, v54, 0x4e);
	v56 = xv_movhlps_(v51, v52);
	v82 = ex_mulps(v80, v56);
	v83 = XV_SHUFPS(v82, v82, 0xb1);
	v84 = ex_mulps(v81, v83);
	v69 = ex_mulps(v54, v55);
	v70 = XV_SHUFPS(v69, v69, 0xb1);
	v71 = ex_mulps(v56, v70);
	v57 = ex_mulps(v55, v56);
	v58 = XV_SHUFPS(v57, v57, 0xb1);
	v63 = XV_SHUFPS(v58, v58, 0x4e);
	v64 = ex_mulps(v54, v63);
	v59 = ex_mulps(v54, v58);
	v60 = ex_subps(v0, v59);
	v65 = ex_addps(v64, v60);
	v72 = ex_addps(v71, v65);
	v74 = XV_SHUFPS(v70, v70, 0x4e);
	v75 = ex_mulps(v56, v74);
	v76 = ex_subps(v72, v75);
	v85 = ex_addps(v84, v76);
	v87 = XV_SHUFPS(v83, v83, 0x4e);
	v88 = ex_mulps(v81, v87);
	v89 = ex_subps(v85, v88);
	v53 = xv_movlhps_(v49, v50);
	v126 = ex_mulps(v53, v89);
	v127 = XV_SHUFPS(v126, v126, 0x00);
	v128 = XV_SHUFPS(v126, v126, 0x55);
	v131 = ex_addps(v127, v128);
	v129 = XV_SHUFPS(v126, v126, 0xaa);
	v130 = XV_SHUFPS(v126, v126, 0xff);
	v132 = ex_addps(v129, v130);
	v133 = ex_addps(v131, v132);
	v134 = rcp_nr(v133);
	v138 = ex_mulps(v89, v134);
	v115 = ex_mulps(v53, v81);
	v116 = XV_SHUFPS(v115, v115, 0xb1);
	v117 = ex_mulps(v56, v116);
	v104 = ex_mulps(v53, v56);
	v105 = XV_SHUFPS(v104, v104, 0xb1);
	v110 = XV_SHUFPS(v105, v105, 0x4e);
	v111 = ex_mulps(v81, v110);
	v66 = ex_mulps(v53, v63);
	v61 = ex_mulps(v53, v58);
	v62 = ex_subps(v0, v61);
	v67 = ex_addps(v66, v62);
	v68 = XV_SHUFPS(v67, v67, 0x4e);
	v106 = ex_mulps(v81, v105);
	v107 = ex_subps(v68, v106);
	v112 = ex_addps(v111, v107);
	v118 = ex_addps(v117, v112);
	v121 = XV_SHUFPS(v116, v116, 0x4e);
	v122 = ex_mulps(v56, v121);
	v123 = ex_subps(v118, v122);
	v137 = ex_mulps(v123, v134);
	v108 = ex_mulps(v54, v105);
	v93 = ex_mulps(v53, v54);
	v94 = XV_SHUFPS(v93, v93, 0xb1);
	v99 = XV_SHUFPS(v94, v94, 0x4e);
	v100 = ex_mulps(v56, v99);
	v86 = ex_mulps(v53, v83);
	v90 = ex_mulps(v53, v87);
	v91 = ex_subps(v86, v90);
	v92 = XV_SHUFPS(v91, v91, 0x4e);
	v95 = ex_mulps(v56, v94);
	v96 = ex_subps(v92, v95);
	v101 = ex_addps(v100, v96);
	v109 = ex_addps(v108, v101);
	v113 = ex_mulps(v54, v110);
	v114 = ex_subps(v109, v113);
	v136 = ex_mulps(v114, v134);
	v124 = ex_mulps(v54, v121);
	v97 = ex_mulps(v81, v94);
	v73 = ex_mulps(v53, v70);
	v77 = ex_mulps(v53, v74);
	v78 = ex_subps(v73, v77);
	v79 = XV_SHUFPS(v78, v78, 0x4e);
	v98 = ex_addps(v97, v79);
	v102 = ex_mulps(v81, v99);
	v103 = ex_subps(v98, v102);
	v119 = ex_mulps(v54, v116);
	v120 = ex_subps(v103, v119);
	v125 = ex_addps(v124, v120);
	v135 = ex_mulps(v125, v134);
	{
		X128 o[4];
		o[0] = v138;
		o[1] = v137;
		o[2] = v136;
		o[3] = v135;
		if (md == 1) { orig(c); check("Matrix44::Inverse", 0x00536990u, o, out, 64); return; }
		XST(out, o[0]); XST(out + 16, o[1]); XST(out + 32, o[2]); XST(out + 48, o[3]);
	}
	c->eax = out;
	xrc_stdcall_ret(c, 1);
}

/* 005b9d70: thiscall Vec4 *Vec4::Mul(Vec4 *out, const Vec4 *v) -- out = this * v (per lane); 0045a960: Vec4::Sub,
 * out = this - v; returns out */
#define VEC4_OP(addr_, name_, op_) \
void XRC_CC xo_##addr_(XCtx *c) { \
	u32 t = c->ecx, out = M32(c->esp + 4), v = M32(c->esp + 8); \
	X128 a, b, r; \
	int md = mode(); \
	ORIGS(0x##addr_##u); \
	if (md == 2) { orig(c); return; } \
	XLD(b, v); XLD(a, t); \
	r = op_(a, b); \
	if (md == 1) { orig(c); check(name_, 0x##addr_##u, &r, out, 16); return; } \
	XST(out, r); \
	c->eax = out; \
	xrc_stdcall_ret(c, 2); \
}
VEC4_OP(005b9d70, "Vec4::Mul", ex_mulps)
VEC4_OP(0045a960, "Vec4::Sub", ex_subps)

/* 00743c70: thiscall Quat *Quat::Mul(Quat *out, const Quat *b) -- a = this; r = ((Bw * a + Aw * b) + a.yzxw * b.zxyw)
 * - a.zxyw * b.yzxw (xyz: the cross-product form); w from s2 = (a * b - a.wxyz * b.wxyz) - (a.wxyz * b.wxyz +
 * a * b).zwxy; out = (r.x, r.y, r.z, s2.w); returns out */
static X128 quat_mul(X128 a, X128 b) {
	X128 r, t5, t6, s, u, h;
	r = xv_addps(xv_mulps(XV_SHUFPS(b, b, 0xff), a), xv_mulps(XV_SHUFPS(a, a, 0xff), b));
	r = xv_addps(xv_mulps(XV_SHUFPS(a, a, 0xc9), XV_SHUFPS(b, b, 0xd2)), r);
	r = xv_subps(r, xv_mulps(XV_SHUFPS(a, a, 0xd2), XV_SHUFPS(b, b, 0xc9)));
	t5 = xv_mulps(a, b);
	t6 = xv_mulps(XV_SHUFPS(a, a, 0x93), XV_SHUFPS(b, b, 0x93));
	s = xv_subps(t5, t6);
	u = xv_addps(t6, t5);
	s = xv_subps(s, XV_SHUFPS(u, u, 0x4e));
	h = xv_unpckhps(r, s);
	return XV_SHUFPS(r, h, 0xc4);
}
void XRC_CC xo_00743c70(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4), bp = M32(c->esp + 8);
	X128 a, b, r;
	int md = mode();
	ORIGS(0x00743c70u);
	if (md == 2) { orig(c); return; }
	XLD(a, t); XLD(b, bp);
	r = quat_mul(a, b);
	if (md == 1) { orig(c); check("Quat::Mul", 0x00743c70u, &r, out, 16); return; }
	XST(out, r);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}

/* 00744160: cdecl Matrix44 *QuatToMatrix(Matrix44 *out, const Quat *q) -- Q2 = q + q; t = (Q2.yzxw * q.wwww +
 * q.zxyw * Q2) ; a = (1 - q.yzxw * Q2.yzxw) - q.zxyw * Q2.zxyw ; b = q.yzxw * Q2 - Q2.zxyw * q.wwww; the rows mixed
 * from them (movss / unpckhps / shufps 0x94), w zeroed (00637d40), stored in reverse order; row 3 = (0, 0, 0, 1)
 * (0109f810); returns out (cdecl: no arguments popped) */
static X128 w0(X128 v) { X128 z; z.u[0] = z.u[1] = z.u[2] = z.u[3] = 0; return XV_SHUFPS(v, xv_unpckhps(v, z), 0xc4); }
static X128 movss_(X128 a, X128 b) { a.u[0] = b.u[0]; return a; }
/* the rows of 00744160 / 00517130 (before their w lanes are set) */
static void quat_rows(u32 qp, X128 *r) {
	X128 q, q2, qw, qc9, qd2, q2c9, q2d2, one, t, a, b, m180, m150, m120;
	float onef = 1.0f;
	XLD(q, qp);
	q2 = xv_addps(q, q);
	qw = XV_SHUFPS(q, q, 0xff); qc9 = XV_SHUFPS(q, q, 0xc9); qd2 = XV_SHUFPS(q, q, 0xd2);
	q2c9 = XV_SHUFPS(q2, q2, 0xc9); q2d2 = XV_SHUFPS(q2, q2, 0xd2);
	one.f[0] = one.f[1] = one.f[2] = one.f[3] = onef;
	t = xv_mulps(q2c9, qw);
	a = xv_subps(one, xv_mulps(qc9, q2c9));
	b = xv_mulps(qc9, q2);
	t = xv_addps(xv_mulps(qd2, q2), t);
	a = xv_subps(a, xv_mulps(qd2, q2d2));
	b = xv_subps(b, xv_mulps(q2d2, qw));
	m180 = movss_(t, a); m150 = movss_(a, b); m120 = movss_(b, t);
	r[0] = XV_SHUFPS(m180, xv_unpckhps(m180, b), 0x94);
	r[1] = XV_SHUFPS(m150, xv_unpckhps(m150, t), 0x94);
	r[2] = XV_SHUFPS(m120, xv_unpckhps(m120, a), 0x94);
}
void XRC_CC xo_00744160(XCtx *c) {
	u32 out = M32(c->esp + 4), qp = M32(c->esp + 8);
	X128 r[4];
	int md = mode(), k;
	ORIGS(0x00744160u);
	if (md == 2) { orig(c); return; }
	quat_rows(qp, r);
	for (k = 0; k < 3; k++) r[k] = w0(r[k]);
	XLD(r[3], 0x0109f810u);
	if (md == 1) { orig(c); check("QuatToMatrix", 0x00744160u, r, out, 64); return; }
	XST(out, r[0]); XST(out + 16, r[1]); XST(out + 32, r[2]); XST(out + 48, r[3]);
	c->eax = out;
	xrc_cdecl_ret(c);
}

/* 00517130: cdecl Matrix44 *QuatToMatrix2(Matrix44 *out, const Quat *q) -- 00744160's rows with w = the .rdata
 * float at 025a3c20 (lanes copied one by one: exact), row 3 = (0, 0, 0, 1) through Vec4::operator= (00455840: the
 * x87 copy, the constant has no NaN); returns out */
void XRC_CC xo_00517130(XCtx *c) {
	u32 out = M32(c->esp + 4), qp = M32(c->esp + 8), w = M32(0x025a3c20u);
	X128 r[4];
	int md = mode(), k;
	ORIGS(0x00517130u);
	if (md == 2) { orig(c); return; }
	quat_rows(qp, r);
	for (k = 0; k < 3; k++) r[k].u[3] = w;
	XLD(r[3], 0x0109f810u);
	if (md == 1) { orig(c); check("QuatToMatrix2", 0x00517130u, r, out, 64); return; }
	XST(out, r[0]); XST(out + 16, r[1]); XST(out + 32, r[2]); XST(out + 48, r[3]);
	c->eax = out;
	xrc_cdecl_ret(c);
}

/* 00515c00: thiscall Quat *Quat::FromMatrix33(const Matrix33 *m) -- the branch-free (select by compares) matrix to
 * quaternion of the game's math library: the four candidate quaternions (by the largest diagonal term), each scaled
 * by 0.5 * rsqrt'(1 + trace-like sums) (005148b0: rsqrtps and one Newton step), chosen with cmpltps masks; stored
 * into this; returns this (ret 4). Transcribed operation by operation from the original (same order). */
static X128 rsqrt_nr(X128 x) { /* 005148b0 */
	X128 half, one, y, a, b;
	half.f[0] = half.f[1] = half.f[2] = half.f[3] = 0.5f;
	one.f[0] = one.f[1] = one.f[2] = one.f[3] = 1.0f;
	y = xv_rsqrtps(x, x);
	a = xv_mulps(y, x);
	b = xv_mulps(y, half);
	return xv_addps(xv_mulps(xv_subps(one, xv_mulps(y, a)), b), y);
}
static X128 sel(X128 m, X128 t, X128 f) { return xv_orps(xv_andnps(m, f), xv_andps(m, t)); }
static X128 quat_from_m33(u32 mp) {
	X128 r0, r1, r2, v25, v26, v27, v29, v31, v36, v37, half, one, zero, v41, v44, v45, v46, v49, v51, v53, v54, v55, v57, v59, v61,
	     v63, v65, v67, v69, v70, v71, v72, m, q;
	half.f[0] = half.f[1] = half.f[2] = half.f[3] = 0.5f;
	one.f[0] = one.f[1] = one.f[2] = one.f[3] = 1.0f;
	zero.u[0] = zero.u[1] = zero.u[2] = zero.u[3] = 0;
	XLD(r0, mp); XLD(r1, mp + 16); XLD(r2, mp + 32);
	v25 = XV_SHUFPS(XV_SHUFPS(XV_SHUFPS(xv_unpcklps(r0, r1), r0, 0xec), r2, 0xe4), XV_SHUFPS(XV_SHUFPS(xv_unpcklps(r0, r1), r0, 0xec), r2, 0xe4), 0x24);
	v26 = XV_SHUFPS(v25, v25, 0x49);
	v27 = XV_SHUFPS(v25, v25, 0x92);
	v29 = xv_addps(xv_addps(v25, v26), v27);
	v31 = xv_subps(xv_subps(v25, v26), v27);
	v36 = xv_addps(XV_SHUFPS(v31, xv_unpckhps(v31, v29), 0xc4), one);
	v37 = rsqrt_nr(v36);
	v41 = XV_SHUFPS(XV_SHUFPS(XV_SHUFPS(r0, xv_unpckhps(r0, r1), 0x94), r2, 0xc6), XV_SHUFPS(XV_SHUFPS(r0, xv_unpckhps(r0, r1), 0x94), r2, 0xc6), 0xd8);
	v44 = XV_SHUFPS(movss_(r0, r1), r2, 0xd2);
	v44 = XV_SHUFPS(v44, v44, 0xd2);
	v45 = xv_addps(v41, v44);
	v46 = xv_subps(v41, v44);
	v49 = xv_mulps(v37, half);
	v51 = XV_SHUFPS(v45, v46, 0xc6); v51 = XV_SHUFPS(v51, v51, 0x93);
	v53 = XV_SHUFPS(v45, v46, 0xd2); v53 = XV_SHUFPS(v53, v53, 0x9c);
	v54 = XV_SHUFPS(v45, v46, 0xb1);
	v55 = movss_(v51, v36);
	v57 = XV_SHUFPS(xv_unpcklps(v53, v36), v53, 0xec);
	v59 = XV_SHUFPS(v54, xv_unpckhps(v54, v36), 0x94);
	v61 = XV_SHUFPS(v46, xv_unpckhps(v46, v36), 0xc4);
	v63 = xv_mulps(v55, XV_SHUFPS(v49, v49, 0x00));
	v65 = xv_mulps(v57, XV_SHUFPS(v49, v49, 0x55));
	v67 = xv_mulps(v59, XV_SHUFPS(v49, v49, 0xaa));
	v69 = xv_mulps(v61, XV_SHUFPS(v49, v49, 0xff));
	v70 = XV_SHUFPS(r0, r0, 0x00); v71 = XV_SHUFPS(r1, r1, 0x55); v72 = XV_SHUFPS(r2, r2, 0xaa);
	q = sel(xv_cmpps1(v70, v71), v65, v63);
	m = xv_andps(xv_cmpps1(v70, v72), xv_cmpps1(v71, v72));
	q = sel(m, v67, q);
	return sel(xv_cmpps1(zero, XV_SHUFPS(v29, v29, 0x00)), v69, q);
}
void XRC_CC xo_00515c00(XCtx *c) {
	u32 t = c->ecx, mp = M32(c->esp + 4);
	X128 q;
	int md = mode();
	ORIGS(0x00515c00u);
	if (md == 2) { orig(c); return; }
	q = quat_from_m33(mp);
	if (md == 1) { orig(c); check("Quat::FromMatrix33", 0x00515c00u, &q, t, 16); return; }
	XST(t, q);
	c->eax = t;
	xrc_stdcall_ret(c, 1);
}

/* 00c4d840: thiscall Matrix44 *Matrix44::Transposed(Matrix44 *out) -- out = the transpose of this (all 16 read before
 * any is written: lane moves only, exact); returns out */
static void m44_transpose(const X128 *r, X128 *o) {
	int i, j;
	for (i = 0; i < 4; i++)
		for (j = 0; j < 4; j++) o[i].u[j] = r[j].u[i];
}
void XRC_CC xo_00c4d840(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4);
	X128 r[4], o[4];
	int md = mode(), k;
	ORIGS(0x00c4d840u);
	if (md == 2) { orig(c); return; }
	for (k = 0; k < 4; k++) XLD(r[k], t + 16u * (u32)k);
	m44_transpose(r, o);
	if (md == 1) { orig(c); check("Matrix44::Transposed", 0x00c4d840u, o, out, 64); return; }
	for (k = 0; k < 4; k++) XST(out + 16u * (u32)k, o[k]);
	c->eax = out;
	xrc_stdcall_ret(c, 1);
}

/* 00ea2bc0: stdcall void Bone::UpdateWorld(Node *node, BoneOut *out, const BoneOut *parent) -- one bone of a
 * skeleton, every frame for every animated character (the hottest guest function of a field scene, called from
 * 00e53bb0's skeleton pass). Layout (out and parent alike): +0x00 the world matrix, +0x40 the skinning matrix
 * (transposed), +0x80 the world scale, +0x90 the world rotation (a quaternion). The node: +0x40 its bind matrix
 * (the inverse bind pose), +0xc0 its scale; its local matrix from 00e64d50 (called in the guest).
 *   L = local(node); qW = Quat::Mul(parent.q, FromMatrix33(L)); R = QuatToMatrix2(qW); s = parent.s * node.s
 *   P = ((L.t.y * pW1 + L.t.x * pW0) + L.t.z * pW2) + L.t.w * pW3  (pW: parent's world rows)
 *   out.s = s; out.q = qW; out.W = (R0 * s.x, s.y * R1, s.z * R2, P)  (mulps operands as the original)
 *   out.skin = Transposed(Matrix44::Mul(W, node.bind))  (read after out.W is written, as the original)
 * The same operations in the same order as the original (00ea2bc0 inlines its matrix code and calls the math library
 * for the rest: 00455d80, 00515c00, 00743c70, 00517130, 00456970, 00c4d840 -- the fast versions above). Returns, as
 * it, the address of its temporary for the transpose in eax (callers ignore it). */
void XRC_CC xo_00ea2bc0(XCtx *c) {
	u32 node = M32(c->esp + 4), out = M32(c->esp + 8), par = M32(c->esp + 12);
	u32 sp = c->esp, buf = (sp - 0x60u) & ~15u, arg[1];
	u32 frame = (((sp - 4u) & ~15u) - 0x1a4u - 12u); /* (the original's esp after its prologue) */
	X128 L[4], qL, qW, R[3], s, ns, ps, t, pw[4], P, W[4], B[4], M[4], skin[4], res[10];
	int md = mode(), k;
	ORIGS(0x00ea2bc0u);
	if (md == 2) { orig(c); return; }
	/* the local matrix, into a temporary below the stack (as the original's frame) */
	arg[0] = buf;
	c->esp = buf;
	c->ecx = node;
	{ static XFn local; if (!local) local = xrc_translation_of(0x00e64d50u); xrc_call_fn(c, local, 0x00e64d50u, 1, arg); }
	c->esp = sp;
	for (k = 0; k < 4; k++) XLD(L[k], buf + 16u * (u32)k);
	/* (FromMatrix33 reads the local matrix's rows 0-2) */
	{
		X128 r0 = L[0], r1 = L[1], r2 = L[2];
		XST(buf, r0); XST(buf + 16, r1); XST(buf + 32, r2);
		qL = quat_from_m33(buf);
	}
	{ X128 a; XLD(a, par + 0x90); qW = quat_mul(a, qL); }
	XST(buf, qW); /* (quat_rows reads the quaternion from guest memory) */
	{ X128 r4[3]; quat_rows(buf, r4); for (k = 0; k < 3; k++) { R[k] = r4[k]; R[k].u[3] = M32(0x025a3c20u); } }
	XLD(ns, node + 0xc0); XLD(ps, par + 0x80);
	s = ex_mulps(ps, ns);
	t = L[3];
	for (k = 0; k < 4; k++) XLD(pw[k], par + 16u * (u32)k);
	P = ex_mulps(XV_SHUFPS(t, t, 0x55), pw[1]);
	P = ex_addps(P, ex_mulps(XV_SHUFPS(t, t, 0x00), pw[0]));
	P = ex_addps(P, ex_mulps(XV_SHUFPS(t, t, 0xaa), pw[2]));
	P = ex_addps(P, ex_mulps(XV_SHUFPS(t, t, 0xff), pw[3]));
	W[2] = ex_mulps(XV_SHUFPS(s, s, 0xaa), R[2]);
	W[1] = ex_mulps(XV_SHUFPS(s, s, 0x55), R[1]);
	W[0] = ex_mulps(R[0], XV_SHUFPS(s, s, 0x00));
	W[3] = P;
	if (md == 1) { /* ours into res (out's 0xa0 bytes, the bind matrix read from the node as it is now) */
		for (k = 0; k < 4; k++) { XLD(B[k], node + 0x40 + 16u * (u32)k); M[k] = m44_transform(W, B[k]); }
		m44_transpose(M, skin);
		for (k = 0; k < 4; k++) { res[k] = W[k]; res[4 + k] = skin[k]; }
		res[8] = s; res[9] = qW;
		orig(c);
		check("Bone::UpdateWorld", 0x00ea2bc0u, res, out, 0xa0);
		return;
	}
	XST(out + 0x80, s);
	XST(out + 0x90, qW);
	for (k = 0; k < 4; k++) XST(out + 16u * (u32)k, W[k]);
	for (k = 0; k < 4; k++) XLD(B[k], node + 0x40 + 16u * (u32)k);
	for (k = 3; k >= 0; k--) M[k] = m44_transform(W, B[k]);
	m44_transpose(M, skin);
	for (k = 0; k < 4; k++) XST(out + 0x40 + 16u * (u32)k, skin[k]);
	c->eax = frame + 0x70u;
	xrc_stdcall_ret(c, 3);
}

/* 00534b70: thiscall Matrix44 *Matrix44::Matrix44(const Matrix44 *m) -- the copy constructor: 16 dwords, each read
 * and then written, in order (integer moves: exact); returns this */
void XRC_CC xo_00534b70(XCtx *c) {
	u32 t = c->ecx, m = M32(c->esp + 4), r[16];
	int md = mode(), k;
	ORIGS(0x00534b70u);
	if (md == 2) { orig(c); return; }
	if (md == 1) {
		for (k = 0; k < 16; k++) r[k] = M32(m + 4u * (u32)k);
		orig(c); check("Matrix44::Matrix44(copy)", 0x00534b70u, r, t, 64); return;
	}
	for (k = 0; k < 16; k++) M32(t + 4u * (u32)k) = M32(m + 4u * (u32)k);
	c->eax = t;
	xrc_stdcall_ret(c, 1);
}

/* 00462cd0: thiscall Matrix44 *Matrix44::Matrix44(const Vec4 *r0, *r1, *r2, *r3) -- the rows, 4 dwords each, each
 * read and then written, in order; returns this */
void XRC_CC xo_00462cd0(XCtx *c) {
	u32 t = c->ecx, row[4], r[16];
	int md = mode(), k, j;
	ORIGS(0x00462cd0u);
	if (md == 2) { orig(c); return; }
	for (k = 0; k < 4; k++) row[k] = M32(c->esp + 4u + 4u * (u32)k);
	if (md == 1) {
		for (k = 0; k < 4; k++) for (j = 0; j < 4; j++) r[4 * k + j] = M32(row[k] + 4u * (u32)j);
		orig(c); check("Matrix44::Matrix44(rows)", 0x00462cd0u, r, t, 64); return;
	}
	for (k = 0; k < 4; k++) for (j = 0; j < 4; j++) M32(t + 16u * (u32)k + 4u * (u32)j) = M32(row[k] + 4u * (u32)j);
	c->eax = t;
	xrc_stdcall_ret(c, 4);
}


/* small Vec4 / Matrix helpers of the collision tests (00f14110 & co): thiscall, out = the first argument, returned.
 * V2(addr, name, expr): out = expr of a = this and b = *arg1; operands in the original's order (dst first) */
#define VFAST_BODY(addr_, name_, nargs_, LOADS, EXPR)                                                       \
	u32 t = c->ecx, out = M32(c->esp + 4);                                                                \
	X128 a, b, r;                                                                                         \
	int md = mode();                                                                                      \
	ORIG(addr_);                                                                                          \
	if (md == 2) { orig(c); return; }                                                                     \
	LOADS;                                                                                                \
	r = EXPR;                                                                                             \
	if (md == 1) { orig(c); check(name_, addr_, &r, out, 16); return; }                                   \
	XST(out, r);                                                                                          \
	c->eax = out;                                                                                         \
	xrc_stdcall_ret(c, nargs_);
/* 00458460: Vec4::Add(out, v) -- this + v */
void XRC_CC xo_00458460(XCtx *c) { VFAST_BODY(0x00458460u, "Vec4::Add", 2, (XLD(a, t), XLD(b, M32(c->esp + 8))), ex_addps(a, b)) }
/* 00710f50: Vec4::Sub(out, v) -- this - v (movaps loads) */
void XRC_CC xo_00710f50(XCtx *c) { VFAST_BODY(0x00710f50u, "Vec4::Sub(m128)", 2, (XLD(a, t), XLD(b, M32(c->esp + 8))), ex_subps(a, b)) }
/* 00cae180: Vec4::Max(out, v) -- maxps(this, v) */
void XRC_CC xo_00cae180(XCtx *c) { VFAST_BODY(0x00cae180u, "Vec4::Max", 2, (XLD(a, t), XLD(b, M32(c->esp + 8))), ex_maxps(a, b)) }
/* 00727710: Vec4::Scale(out, float s) -- this * ssss */
void XRC_CC xo_00727710(XCtx *c) { VFAST_BODY(0x00727710u, "Vec4::Scale", 2, (XLD(a, t), b.u[0] = b.u[1] = b.u[2] = b.u[3] = M32(c->esp + 8)), ex_mulps(a, b)) }
/* 00517c80: Vec4::Neg(out) -- 0 - this */
void XRC_CC xo_00517c80(XCtx *c) { VFAST_BODY(0x00517c80u, "Vec4::Neg", 1, (XLD(a, t), memset(&b, 0, sizeof b)), ex_subps(b, a)) }
/* 00727690: Vec4::Vec4(float x, y, z, w) -- this = (x, y, z, w) (movss: the bits as they are); returns this */
void XRC_CC xo_00727690(XCtx *c) {
	u32 t = c->ecx, r[4];
	int md = mode(), k;
	ORIGS(0x00727690u);
	if (md == 2) { orig(c); return; }
	for (k = 0; k < 4; k++) r[k] = M32(c->esp + 4u + 4u * (u32)k);
	if (md == 1) { orig(c); check("Vec4::Vec4(xyzw)", 0x00727690u, r, t, 16); return; }
	for (k = 0; k < 4; k++) M32(t + 4u * (u32)k) = r[k];
	c->eax = t;
	xrc_stdcall_ret(c, 4);
}
/* 00c28db0: Matrix44::Matrix44(const Vec4 *r0, *r1, *r2, *r3) (movaps rows; each row read, then written); returns this */
void XRC_CC xo_00c28db0(XCtx *c) {
	u32 t = c->ecx;
	X128 r[4];
	int md = mode(), k;
	ORIGS(0x00c28db0u);
	if (md == 2) { orig(c); return; }
	if (md == 1) {
		for (k = 0; k < 4; k++) XLD(r[k], M32(c->esp + 4u + 4u * (u32)k));
		orig(c); check("Matrix44::Matrix44(m128 rows)", 0x00c28db0u, r, t, 64); return;
	}
	for (k = 0; k < 4; k++) { X128 v; XLD(v, M32(c->esp + 4u + 4u * (u32)k)); XST(t + 16u * (u32)k, v); }
	c->eax = t;
	xrc_stdcall_ret(c, 4);
}
/* 00db4dd0: Matrix44::Abs(out) -- each row maxps(0 - r, r); all four rows read first; returns out */
void XRC_CC xo_00db4dd0(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4);
	X128 m[4], r[4], z;
	memset(&z, 0, sizeof z);
	int md = mode(), k;
	ORIGS(0x00db4dd0u);
	if (md == 2) { orig(c); return; }
	for (k = 0; k < 4; k++) XLD(m[k], t + 16u * (u32)k);
	for (k = 0; k < 4; k++) r[k] = ex_maxps(ex_subps(z, m[k]), m[k]);
	if (md == 1) { orig(c); check("Matrix44::Abs", 0x00db4dd0u, r, out, 64); return; }
	for (k = 0; k < 4; k++) XST(out + 16u * (u32)k, r[k]);
	c->eax = out;
	xrc_stdcall_ret(c, 1);
}


/* ============================================================================================ the game's D3D9 proxy
 * The game wraps the device (00ec3970: flag 0x26e5160): its proxy's methods (00ec2xxx) unwrap and forward to the real
 * device's vtable -- ~30k calls a frame in battle (SetVertexShaderConstantF alone ~7000). Translated, each is a guest
 * function between the caller's vtable call and the HLE method; these do the same natively and call the device's
 * method as a translated call would (an inline cache per site, XCALL in xrc.h). Same stack writes, same registers. */
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
 * it returns D3D_OK at once, as that method would -- most of the ~30k state calls of a battle frame */
static int px_noskip = -1;
static inline int px_hle(u32 dv) {
	if (px_noskip < 0) px_noskip = getenv("XRC_NO_STATE_SKIP") != 0;
	return !px_noskip && dv == the_dev && dv && M32(dv) == d3_ifaces[1]->vtable;
}
/* XRC_STATE_SKIP_STATS=1: calls and skips per proxy method, every 2^20 calls */
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
 * caller). eax = the method, ecx = the vtable, as the original leaves them. SKIP: the call is a no-op (A(k): its
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
PX_FWD(00ec2890, 0x178, 4, PX_CONSTS(vsf, 256, 4)) /* SetVertexShaderConstantF(reg, data, n) */
PX_FWD(00ec25f0, 0x114, 4, (px_sampler(PA(1)) < 32 && PA(2) < 16 && dev.ss[px_sampler(PA(1))][PA(2)] == PA(3))) /* SetSamplerState */
PX_FWD(00ec2950, 0x198, 3, (PA(1) < 16 && dev.freq[PA(1)] == PA(2))) /* SetStreamSourceFreq */
PX_FWD(00ec29b0, 0x1b4, 4, PX_CONSTS(psf, 224, 4)) /* SetPixelShaderConstantF */
PX_FWD(00ec2910, 0x188, 4, PX_CONSTS(vsb, 16, 1)) /* SetVertexShaderConstantB */
PX_FWD(00ec2810, 0x15c, 2, (dev.decl == PA(1))) /* SetVertexDeclaration */
PX_FWD(00ec2870, 0x170, 2, (dev.vs == PA(1))) /* SetVertexShader */
PX_FWD(00ec2990, 0x1ac, 2, (dev.ps == PA(1))) /* SetPixelShader */
PX_FWD(00ec2830, 0x164, 2, (dev.fvf == PA(1))) /* SetFVF */
/* 00ec24d0: SetRenderState(this, state, value): state 0xae (D3DRS_SCISSORTESTENABLE) also noted at 0x2846a85;
 * push esi; real->vtbl[0xe4](real, state, value); pop esi; ret 0xc */
void XRC_CC xo_00ec24d0(XCtx *c) {
	static XIc ic = {0xffffffffu, 0, 0};
	if (mode() == 2) { ORIG(0x00ec24d0u); orig(c); return; }
	u32 sp = c->esp, st = M32(sp + 8), val = M32(sp + 12), dv = M32(M32(sp + 4) + 8), vt = M32(dv), t = M32(vt + 0xe4);
	u32 esi = c->esi;
	if (st == 0xaeu) M8(0x02846a85u) = val != 0;
	if (st < 256 && px_hle(dv) && dev.rs[st] == val) { /* (a no-op: as after the call, with D3D_OK) */
		M32(sp - 4) = esi; M32(sp - 8) = val; M32(sp - 12) = st; M32(sp - 16) = dv; M32(sp - 20) = 0x00ec24ffu;
		c->ecx = t; c->edx = val; c->eax = 0; c->eip = M32(sp); c->esp = sp + 16;
		return;
	}
	M32(sp - 4) = esi; M32(sp - 8) = val; M32(sp - 12) = st; M32(sp - 16) = dv; M32(sp - 20) = 0x00ec24ffu;
	c->esp = sp - 20; c->ecx = t; c->edx = val; c->eax = dv;
	px_call(c, t, &ic, 0x00ec24ffu);
	c->esi = esi;
	c->esp = sp; /* (pop esi; ret 0xc) */
	c->eip = M32(sp);
	c->esp = sp + 16;
}
/* the wrapped resources: a proxy object holds the marker 0xb00beed0 among its first 32 dwords, the real object
 * right after it (with the proxy off -- flag 0x26e5160 clear -- the argument is the real object already) */
static inline u32 px_real(u32 o, u32 *ecx) {
	u32 k;
	if (!M8(0x026e5160u)) return o;
	*ecx = o;
	if (!o) return 0;
	for (k = 0; k < 0x20; k++) if (M32(o + 4 * k) == 0xb00beed0u) return M32(o + 4 * k + 4);
	return 0;
}
/* 00ec3370: SetStreamSource(this, stream, vb, offset, stride): push esi; real->vtbl[0x190](real, stream, real vb,
 * offset, stride); pop esi; ret 0x14 */
void XRC_CC xo_00ec3370(XCtx *c) {
	static XIc ic = {0xffffffffu, 0, 0};
	if (mode() == 2) { ORIG(0x00ec3370u); orig(c); return; }
	u32 sp = c->esp, ecx = c->ecx, vb = px_real(M32(sp + 12), &ecx), dv = M32(M32(sp + 4) + 8), vt = M32(dv), t = M32(vt + 0x190);
	u32 esi = c->esi, off = M32(sp + 16), stride = M32(sp + 20), strm = M32(sp + 8);
	M32(sp - 4) = esi; M32(sp - 8) = stride; M32(sp - 12) = off; M32(sp - 16) = vb; M32(sp - 20) = strm; M32(sp - 24) = dv;
	M32(sp - 28) = 0x00ec33c4u;
	if (strm < 16 && px_hle(dv) && dev.stream[strm][0] == vb && dev.stream[strm][1] == off && dev.stream[strm][2] == stride) {
		c->ecx = t; c->edx = vt; c->eax = 0; c->eip = M32(sp); c->esp = sp + 24;
		return;
	}
	c->esp = sp - 28; c->ecx = t; c->edx = vt; c->eax = strm; c->esi = off;
	px_call(c, t, &ic, 0x00ec33c4u);
	c->esi = esi;
	c->eip = M32(sp);
	c->esp = sp + 24;
}
/* 00ec3230: SetTexture(this, stage, tex): [esp+0xc] = the real texture, [esp+4] = real; jmp real->vtbl[0x104] */
void XRC_CC xo_00ec3230(XCtx *c) {
	static XIc ic = {0xffffffffu, 0, 0};
	if (mode() == 2) { ORIG(0x00ec3230u); orig(c); return; }
	u32 sp = c->esp, ecx = c->ecx, tx = px_real(M32(sp + 12), &ecx), dv = M32(M32(sp + 4) + 8), vt = M32(dv), t = M32(vt + 0x104);
	M32(sp + 12) = tx; M32(sp + 4) = dv;
	c->eax = tx; c->ecx = t; c->edx = vt;
	if (px_sampler(M32(sp + 8)) < 32 && px_hle(dv) && dev.tex[px_sampler(M32(sp + 8))] == tx) { c->eax = 0; xrc_stdcall_ret(c, 3); return; }
	px_call(c, t, &ic, M32(sp));
}


/* the math of the overrides above, as values (for whole functions built from them) */
static inline X128 m33_xf(u32 m, X128 vv) { /* Matrix33::Transform (00711f40) */
	X128 r0, r1, r2, acc;
	XLD(r0, m); XLD(r1, m + 16); XLD(r2, m + 32);
	acc = xv_mulps(r0, XV_SHUFPS(vv, vv, 0x00));
	acc = xv_addps(acc, xv_mulps(r1, XV_SHUFPS(vv, vv, 0x55)));
	return xv_addps(acc, xv_mulps(r2, XV_SHUFPS(vv, vv, 0xaa)));
}
static inline X128 m33_xfr(const X128 *r, X128 vv) {
	X128 acc = xv_mulps(r[0], XV_SHUFPS(vv, vv, 0x00));
	acc = xv_addps(acc, xv_mulps(r[1], XV_SHUFPS(vv, vv, 0x55)));
	return xv_addps(acc, xv_mulps(r[2], XV_SHUFPS(vv, vv, 0xaa)));
}
static inline u32 xv_movmsk(X128 a) { return a.u[0] >> 31 | (a.u[1] >> 31) << 1 | (a.u[2] >> 31) << 2 | (a.u[3] >> 31) << 3; }
static inline X128 dot4v(X128 a, X128 b) { /* Vec4::Dot4 (00710860) */
	X128 pr = xv_mulps(a, b), acc;
	acc = xv_addps(pr, XV_SHUFPS(pr, pr, 0x55));
	acc = xv_addps(acc, XV_SHUFPS(pr, pr, 0xaa));
	return xv_addps(acc, XV_SHUFPS(pr, pr, 0xff));
}
/* 00f14110: cdecl bool Shapes::Overlap(const Aabb *box, const Shape *s, u32 n, const Vec4 *at) -- does the box
 * (moved by -at) overlap any of n shapes (0xa0 bytes each; +0x90 the kind: 0 a box -- rows +0..+0x20 one basis,
 * +0x30..+0x50 another, +0x60 its half size, +0x70 its centre; 1 an oriented box -- +0x00..+0x80 three bases,
 * +0x94..+0x9c its radii; others skipped). Every step as the original's calls (Vec4::Add / Scale / Sub / Neg / Max /
 * Dot4, Matrix33::Transform, Matrix44::Abs -- the fast versions above), in the same order. Returns al = 1 at the
 * first overlap; 0 otherwise, with the rest of eax the loop count's (n & ~0xff), as the original leaves it. */
static int nb_f14110;
void XRC_CC xo_00f14110(XCtx *c) {
	u32 box = M32(c->esp + 4), sh = M32(c->esp + 8), n = M32(c->esp + 12), at = M32(c->esp + 16), i, res = 0;
	X128 mn, mx, k, ctr, ext, p, rel, z;
	int md = mode(), hit = 0;
	ORIGS(0x00f14110u);
	if (md == 2) { orig(c); return; }
	memset(&z, 0, sizeof z);
	XLD(mn, box); XLD(mx, box + 16);
	k.u[0] = k.u[1] = k.u[2] = k.u[3] = x87_copy(M32(0x0109b130u)); /* (pushed through the x87) */
	ctr = ex_mulps(ex_addps(mn, mx), k);
	ext = ex_subps(mx, ctr);
	XLD(p, at);
	rel = ex_subps(ctr, p);
	for (i = 0; i < n && !hit; i++) {
		u32 s = sh + i * 0xa0u, kind = M32(s + 0x90);
		if (kind == 0) {
			X128 h, hc, nh, ah, d, d1, d2, x, q;
			XLD(h, s + 0x60); XLD(hc, s + 0x70);
			nh = ex_subps(z, h);
			ah = ex_maxps(nh, h);
			d = ex_subps(rel, hc);
			d1 = dot4v(ah, ext);
			d2 = dot4v(h, d);
			x = ex_addps(d2, d1);
			q = ex_addps(m33_xf(s, rel), m33_xf(s + 0x30, ext));
			if ((xv_movmsk(q) & 0xf) == 0xf) {
				float cf, xf;
				u32 cb = M32(0x025a3c20u);
				memcpy(&cf, &cb, 4); xf = x.f[0];
				if (cf > xf) hit = 1;
			}
		} else if (kind == 1) {
			X128 rad, A[4], B[4], C[4], aA[4], aB[4], aC[4], t1, t2, t3, m1, m2;
			int j;
			for (j = 0; j < 3; j++) rad.u[j] = x87_copy(M32(s + 0x94 + 4u * (u32)j));
			rad.u[3] = 0; /* (fldz) */
			for (j = 0; j < 3; j++) { XLD(A[j], s + 16u * (u32)j); XLD(B[j], s + 0x30 + 16u * (u32)j); XLD(C[j], s + 0x60 + 16u * (u32)j); }
			for (j = 0; j < 3; j++) { aA[j] = ex_maxps(ex_subps(z, A[j]), A[j]); aB[j] = ex_maxps(ex_subps(z, B[j]), B[j]); aC[j] = ex_maxps(ex_subps(z, C[j]), C[j]); }
			t1 = ex_addps(m33_xfr(A, rel), m33_xfr(aA, ext));
			t2 = ex_addps(m33_xfr(B, rel), m33_xfr(aB, ext));
			{ X128 ce = m33_xfr(aC, ext); t3 = ex_addps(m33_xfr(C, ex_subps(rel, rad)), ce); }
			m1 = ex_maxps(t1, t2);
			m2 = ex_maxps(m1, t3);
			if ((xv_movmsk(m2) & 7) == 7) hit = 1;
		}
	}
	res = hit ? 1u : (n & ~0xffu);
	if (md == 1) {
		orig(c);
		if ((c->eax & (hit ? 0xffu : 0xffffffffu)) != res) {
			if (nb_f14110++ < 10) fprintf(stderr, "xrc: FAST MISMATCH 00f14110: ours %08x, original %08x\n", res, c->eax);
			n_bad++;
		}
		n_checked++;
		{ static unsigned long long nc; if (++nc, (nc & (nc - 1)) == 0 && nc >= 1024) fprintf(stderr, "xrc: fast check: 00f14110 Shapes::Overlap: %llu calls checked (%d mismatches)\n", nc, nb_f14110); }
		return;
	}
	c->eax = res;
	c->ecx = M32(0x026e5a84u); /* (the security cookie check leaves the cookie in ecx) */
	xrc_cdecl_ret(c);
}


/* 00c43bf0: thiscall Aabb *Aabb::Extended(Aabb *out, const Vec4 *p) -- as 004586c0 with unaligned loads: out.max =
 * maxps(this.max, p), out.min = minps(this.min, p), both computed before either is written; returns out */
void XRC_CC xo_00c43bf0(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4), p = M32(c->esp + 8);
	X128 pv, r[2], a;
	int md = mode();
	ORIGS(0x00c43bf0u);
	if (md == 2) { orig(c); return; }
	XLD(pv, p); XLD(a, t + 16); r[1] = ex_maxps(a, pv);
	XLD(a, t); r[0] = ex_minps(a, pv);
	if (md == 1) { orig(c); check("Aabb::Extended(unaligned)", 0x00c43bf0u, r, out, 32); return; }
	XST(out, r[0]); XST(out + 16, r[1]);
	c->eax = out;
	xrc_stdcall_ret(c, 2);
}


/* 00eac800: cdecl Render::BindStreams(const Decl *d) -- every draw (~320 a frame): for the 16 streams of the table at
 * 0x283e424 (16 bytes each: +0 the buffer object -- its D3D buffer at +0x10 --, +4 the offset, +8 the stride),
 * dev->SetStreamSource(i, buffer, offset, stride) and dev->SetStreamSourceFreq(i, 1) (dev = *0x283eac0, the game's
 * proxy device); then SetVertexDeclaration(d->+4), or SetFVF(2) without one; returns what that returns. Native, with
 * one difference: when the proxy forwards to the HLE device, a stream whose binding the device already has (the
 * buffer the proxy would pass, the offset, the stride, frequency 1) is not set again -- both calls would leave every
 * state as it is (bind() keeps the same references). That is most of them: ~10k proxy + device calls a frame. */
static XIc ic_ss, ic_ssf, ic_dcl;
void XRC_CC xo_00eac800(XCtx *c) {
	u32 sp = c->esp, arg = M32(sp + 4), ebx = c->ebx, esi = c->esi, edi = c->edi, i, d, vt, real = 0, hle = 0, e = 0;
	static int noskip = -1;
	if (mode() == 2) { ORIG(0x00eac800u); orig(c); return; }
	if (noskip < 0) noskip = getenv("XRC_NO_STREAM_SKIP") != 0;
	/* push ebx; push esi; push edi */
	M32(sp - 4) = ebx; M32(sp - 8) = esi; M32(sp - 12) = edi;
	d = M32(0x0283eac0u); vt = M32(d);
	if (!noskip && M32(vt + 0x190) == 0x00ec3370u && M32(vt + 0x198) == 0x00ec2950u) {
		real = M32(d + 8);
		hle = real && real == the_dev && M32(real) == d3_ifaces[1]->vtable;
	}
	for (i = 0; i < 16; i++) {
		u32 tb = 0x0283e424u + 16u * i, o = M32(tb), vb = o ? M32(o + 0x10) : 0, off = M32(tb + 4), stride = M32(tb + 8);
		if (hle) {
			u32 rv = vb, k;
			if (M8(0x026e5160u) && vb) { /* (the proxy's unwrap: px_real) */
				rv = 0;
				for (k = 0; k < 0x20; k++) if (M32(vb + 4 * k) == 0xb00beed0u) { rv = M32(vb + 4 * k + 4); break; }
			}
			if (dev.stream[i][0] == rv && dev.stream[i][1] == off && dev.stream[i][2] == stride && dev.freq[i] == 1) continue;
		}
		d = M32(0x0283eac0u); vt = M32(d);
		/* push stride; push offset; push vb; push i; push dev; call [vt+0x190] */
		M32(sp - 16) = stride; M32(sp - 20) = off; M32(sp - 24) = vb; M32(sp - 28) = i; M32(sp - 32) = d; M32(sp - 36) = 0x00eac837u;
		c->esp = sp - 36; c->eax = d; c->ecx = vb; c->edx = vt; c->ebx = off; c->esi = i; c->edi = tb;
		px_call(c, M32(vt + 0x190), &ic_ss, 0x00eac837u);
		d = M32(0x0283eac0u); vt = M32(d);
		/* push 1; push i; push dev; call [vt+0x198] */
		M32(sp - 16) = 1; M32(sp - 20) = i; M32(sp - 24) = d; M32(sp - 28) = 0x00eac84au;
		c->esp = sp - 28; c->eax = d; c->ecx = vt; c->esi = i; c->edi = tb;
		px_call(c, M32(vt + 0x198), &ic_ssf, 0x00eac84au);
		e++;
	}
	c->ebx = ebx; c->esi = esi; c->edi = edi;
	d = M32(0x0283eac0u); vt = M32(d);
	if (arg) { /* push d->+4; push dev; call [vt+0x15c] */
		M32(sp - 4) = M32(arg + 4); M32(sp - 8) = d; M32(sp - 12) = 0x00eac872u;
		c->esp = sp - 12; c->eax = d; c->ecx = M32(arg + 4); c->edx = M32(vt + 0x15c);
		px_call(c, M32(vt + 0x15c), &ic_dcl, 0x00eac872u);
	} else { /* push 2; push dev; call [vt+0x164] */
		M32(sp - 4) = 2; M32(sp - 8) = d; M32(sp - 12) = 0x00eac880u;
		c->esp = sp - 12; c->eax = d; c->ecx = vt; c->edx = M32(vt + 0x164);
		px_call(c, M32(vt + 0x164), &ic_dcl, 0x00eac880u);
	}
	c->esp = sp; /* (ret: cdecl) */
	xrc_cdecl_ret(c);
	(void)e;
}


/* 00ec1ff0: Render::Forget(ecx = p) -- the 32768-entry pointer table at 0x2846a88 (sorted every frame: 00ec4560):
 * the last entry equal to p, searched from the top, is cleared, the count at 0x2866a88 decremented and the dirty flag
 * 0x2866a8c set; eax = its index (-1: none), edx = 1. Twice a frame, a scan of most of the table: here 4 at a time */
void XRC_CC xo_00ec1ff0(XCtx *c) {
	const u32 *tab = (const u32 *)(xrc_mem + 0x02846a88u);
	u32 p = c->ecx;
	int i = 0x7fff;
	if (mode() == 2) { ORIG(0x00ec1ff0u); orig(c); return; }
#if defined(XRC_VEC) && defined(__SSE2__)
	{
		__m128i pv = _mm_set1_epi32((int)p);
		for (; i >= 3; i -= 4) { /* (entries i-3 .. i) */
			__m128i v = _mm_loadu_si128((const __m128i *)(tab + i - 3));
			if (_mm_movemask_epi8(_mm_cmpeq_epi32(v, pv))) break;
		}
	}
#endif
	for (; i >= 0 && tab[i] != p; i--) {}
	c->edx = 1;
	c->eax = (u32)i;
	if (i >= 0) {
		M32(0x02866a88u) -= 1;
		M32(0x02846a88u + 4u * (u32)i) = 0;
		M8(0x02866a8cu) = 1;
	}
	xrc_cdecl_ret(c);
}


/* register-passing vector helpers (00ad5550 & co: cdecl, a = xmm0, b = xmm1, the result in xmm0); eax / ecx left as
 * the original's security cookie code leaves them (eax = cookie ^ its esp, ecx = the cookie) */
static unsigned long long n_xbad;
#define XRFAST(addr_, name_, frame_, EXPR)                                                                  \
	void XRC_CC xo_##addr_(XCtx *c) {                                                                     \
		X128 a = c->x[0], b = c->x[1], r;                                                                 \
		u32 ck = M32(0x026e5a84u), e = ((c->esp - 4u) & ~15u) - (frame_);                                \
		int md = mode();                                                                                  \
		ORIGS(0x##addr_##u);                                                                               \
		if (md == 2) { orig(c); return; }                                                                 \
		(void)b;                                                                                          \
		r = EXPR;                                                                                         \
		if (md == 1) {                                                                                    \
			orig(c); n_checked++;                                                                         \
			if (memcmp(&r, &c->x[0], 16) && ++n_xbad <= 10)                                              \
				fprintf(stderr, "xrc: FAST MISMATCH %s (%s): ours %08x %08x %08x %08x, original %08x %08x %08x %08x\n", #addr_, name_, \
				        r.u[0], r.u[1], r.u[2], r.u[3], c->x[0].u[0], c->x[0].u[1], c->x[0].u[2], c->x[0].u[3]); \
			{ static unsigned long long nc; if (++nc, (nc & (nc - 1)) == 0 && nc >= 1024) fprintf(stderr, "xrc: fast check: %s %s: %llu calls checked (%llu register mismatches in all)\n", #addr_, name_, nc, n_xbad); } \
			return;                                                                                       \
		}                                                                                                 \
		c->x[0] = r;                                                                                      \
		c->eax = ck ^ e; c->ecx = ck;                                                                     \
		xrc_cdecl_ret(c);                                                                                 \
	}
static inline X128 setw(X128 a, X128 b) { X128 r = a; r.u[3] = b.u[3]; return r; }
static inline X128 cross3(X128 a, X128 b) {
	X128 t1 = XV_SHUFPS(a, a, 0xc9), t2 = XV_SHUFPS(b, b, 0xd2), t3 = XV_SHUFPS(a, a, 0xd2), t4 = XV_SHUFPS(b, b, 0xc9);
	return ex_subps(ex_mulps(t1, t2), ex_mulps(t3, t4));
}
static inline X128 dot3s(X128 a, X128 b) {
	X128 p = ex_mulps(a, b);
	X128 m = ex_mulps(XV_SHUFPS(a, a, 0x09), XV_SHUFPS(b, b, 0x09));
	X128 q = ex_addps(m, p);
	return ex_addps(ex_mulps(XV_SHUFPS(a, a, 0x52), XV_SHUFPS(b, b, 0x52)), q);
}
XRFAST(00ad4ef0, "Vec4::SetW (xmm)", 0x60, setw(a, b))      /* (a.x, a.y, a.z, b.w): unpckhps + shufps 0xc4 */
XRFAST(00ad5090, "Vec4::Cross (xmm)", 0xf0, cross3(a, b))   /* a.yzx * b.zxy - a.zxy * b.yzx */
XRFAST(00ad4f50, "Vec4::Dot3 (xmm)", 0xd0, dot3s(a, b))     /* lane 0: x*x' + y*y' + z*z' (as the original adds) */
XRFAST(00ad51c0, "Vec4::Rcp (xmm)", 0x90, rcp_nr(a))        /* rcpps + a Newton step */

/* quaternion slerp 005168e0 and its register-passing helpers (cdecl, xmm0.. in, xmm0 out), generated from the
 * originals' straight-line SSE / MMX by symbolic evaluation (scratch symsse3.py), the operations in their order */
static inline X128 xv_zero(void) { X128 r; memset(&r, 0, sizeof r); return r; }
static inline X128 xv_ss(u32 u) { X128 r = xv_zero(); r.u[0] = u; return r; }
static inline X128 K_(u32 u) { return xv_ss(u); }
static inline X128 xv_movhlps(X128 d, X128 s) { d.q[0] = s.q[1]; return d; }
static inline X128 xv_movlhps(X128 d, X128 s) { d.q[1] = s.q[0]; return d; }
static inline X128 mm_cvtt(X128 s) { X128 r = xv_zero(); r.u[0] = cvt_trunc((double)s.f[0]); r.u[1] = cvt_trunc((double)s.f[1]); return r; } /* cvttps2pi */
static inline X128 mm_cvtpi(X128 d, X128 s) { d.f[0] = (float)(s32)s.u[0]; d.f[1] = (float)(s32)s.u[1]; return d; }      /* cvtpi2ps */
/* 00514a60: dot4, the sum in every lane */
static inline X128 qs_dot4(X128 a, X128 b) {
	X128 v4, v5, v6, v7, v8;
	v4 = ex_mulps(a, b);
	v5 = XV_SHUFPS(v4, v4, 0x39);
	v6 = ex_addps(v5, v4);
	v7 = XV_SHUFPS(v6, v6, 0x4e);
	v8 = ex_addps(v7, v6);
	return v8;
}
/* 005149f0: select: c ? b : a */
static inline X128 qs_sel(X128 a, X128 b, X128 c) {
	X128 v4, v5, v6;
	v5 = xv_andnps(c, a);
	v4 = xv_andps(c, b);
	v6 = xv_orps(v5, v4);
	return v6;
}
/* 00514b20: acos (sqrt(1 - |x|) * a polynomial, reflected for x < 0) */
static inline X128 qs_acos(X128 a) {
	X128 v0, v4, v5, v6, v7, v9, v10, v11, v12, v13, v14, v16, v17, v19, v20, v21, v22, v24, v25, v26, v27, v29, v30, v31, v32, v34, v35, v37, v38, v39, v40, v42, v43, v44, v45, v47, v48, v49, v50, v51, v52, v53, v55, v56, v57, v58, v59;
	v0 = xv_zero();
	v6 = xv_cmpps1(a, v0);
	v4 = ex_subps(v0, a);
	v5 = ex_maxps(a, v4);
	v7 = xv_ss(M32(0 + 0x10aa950u));
	v9 = XV_SHUFPS(v7, v7, 0x00);
	v10 = ex_subps(v9, v5);
	v11 = xv_sqrtps(v5, v10);
	v17 = xv_ss(M32(0 + 0x10a717cu));
	v19 = XV_SHUFPS(v17, v17, 0x00);
	v20 = ex_mulps(v19, v5);
	v14 = xv_ss(M32(0 + 0x10a7180u));
	v16 = XV_SHUFPS(v14, v14, 0x00);
	v21 = ex_addps(v20, v16);
	v25 = ex_mulps(v21, v5);
	v22 = xv_ss(M32(0 + 0x10a7178u));
	v24 = XV_SHUFPS(v22, v22, 0x00);
	v26 = ex_addps(v25, v24);
	v30 = ex_mulps(v26, v5);
	v27 = xv_ss(M32(0 + 0x10a7174u));
	v29 = XV_SHUFPS(v27, v27, 0x00);
	v31 = ex_addps(v30, v29);
	v12 = ex_mulps(v5, v5);
	v13 = ex_mulps(v12, v12);
	v50 = ex_mulps(v31, v13);
	v35 = xv_ss(M32(0 + 0x10a716cu));
	v37 = XV_SHUFPS(v35, v35, 0x00);
	v38 = ex_mulps(v37, v5);
	v32 = xv_ss(M32(0 + 0x10a7170u));
	v34 = XV_SHUFPS(v32, v32, 0x00);
	v39 = ex_addps(v38, v34);
	v43 = ex_mulps(v39, v5);
	v40 = xv_ss(M32(0 + 0x10a7168u));
	v42 = XV_SHUFPS(v40, v40, 0x00);
	v44 = ex_addps(v43, v42);
	v48 = ex_mulps(v44, v5);
	v45 = xv_ss(M32(0 + 0x10a7164u));
	v47 = XV_SHUFPS(v45, v45, 0x00);
	v49 = ex_addps(v48, v47);
	v51 = ex_addps(v50, v49);
	v52 = ex_mulps(v11, v51);
	v58 = xv_andnps(v6, v52);
	v53 = xv_ss(M32(0 + 0x10a6f2cu));
	v55 = XV_SHUFPS(v53, v53, 0x00);
	v56 = ex_subps(v55, v52);
	v57 = xv_andps(v6, v56);
	v59 = xv_orps(v58, v57);
	return v59;
}
/* 00514fe0: sin (quadrant by cvttps2pi, two polynomials) */
static inline X128 qs_sincos(X128 a) {
	X128 v0, v4, v6, v7, v8, v9, v11, v12, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v69, v70, v71, v72, v73, v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88;
	v0 = xv_zero();
	v4 = xv_ss(M32(0 + 0x10a7184u));
	v6 = XV_SHUFPS(v4, v4, 0x00);
	v7 = ex_mulps(a, v6);
	v8 = xv_cmpps1(v7, v0);
	v12 = xv_ss(M32(0 + 0x109b130u));
	v14 = XV_SHUFPS(v12, v12, 0x00);
	v16 = xv_andnps(v8, v14);
	v9 = xv_ss(M32(0 + 0x10a1458u));
	v11 = XV_SHUFPS(v9, v9, 0x00);
	v15 = xv_andps(v8, v11);
	v17 = xv_orps(v16, v15);
	v18 = ex_addps(v7, v17);
	v19 = mm_cvtt(v18);
	v20 = __extension__({ X128 r_; r_.u[0] = v19.u[0]; r_.u[1] = v19.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v26 = __extension__({ X128 r_; r_.u[0] = v20.u[0]; r_.u[1] = v20.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v69 = __extension__({ X128 r_; r_.u[0] = K_(0x2u).u[0]; r_.u[1] = K_(0x2u).u[0]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v70 = __extension__({ X128 r_; r_.u[0] = v69.u[0]; r_.u[1] = v69.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v71 = xv_andps(v26, v70);
	v72 = __extension__({ X128 r_; r_.u[0] = v71.u[0]; r_.u[1] = v71.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v77 = __extension__({ X128 r_; r_.u[0] = v72.u[0]; r_.u[1] = v72.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v78 = mm_cvtpi(v0, v77);
	v21 = xv_movhlps(v18, v18);
	v22 = mm_cvtt(v21);
	v23 = __extension__({ X128 r_; r_.u[0] = v22.u[0]; r_.u[1] = v22.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v24 = __extension__({ X128 r_; r_.u[0] = v23.u[0]; r_.u[1] = v23.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v73 = xv_andps(v24, v70);
	v74 = __extension__({ X128 r_; r_.u[0] = v73.u[0]; r_.u[1] = v73.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v75 = __extension__({ X128 r_; r_.u[0] = v74.u[0]; r_.u[1] = v74.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v76 = mm_cvtpi(v0, v75);
	v79 = xv_movlhps(v78, v76);
	v84 = xv_cmpps0(v79, v0);
	v57 = __extension__({ X128 r_; r_.u[0] = K_(0x1u).u[0]; r_.u[1] = K_(0x1u).u[0]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v58 = __extension__({ X128 r_; r_.u[0] = v57.u[0]; r_.u[1] = v57.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v59 = xv_andps(v26, v58);
	v60 = __extension__({ X128 r_; r_.u[0] = v59.u[0]; r_.u[1] = v59.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v65 = __extension__({ X128 r_; r_.u[0] = v60.u[0]; r_.u[1] = v60.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v66 = mm_cvtpi(v0, v65);
	v61 = xv_andps(v24, v58);
	v62 = __extension__({ X128 r_; r_.u[0] = v61.u[0]; r_.u[1] = v61.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v63 = __extension__({ X128 r_; r_.u[0] = v62.u[0]; r_.u[1] = v62.u[1]; r_.u[2] = 0; r_.u[3] = 0; r_; });
	v64 = mm_cvtpi(v0, v63);
	v67 = xv_movlhps(v66, v64);
	v80 = xv_cmpps0(v67, v0);
	XLD(v37, 0 + 39600912u);
	v27 = mm_cvtpi(v0, v26);
	v25 = mm_cvtpi(v0, v24);
	v28 = xv_movlhps(v27, v25);
	XLD(v29, 0 + 40267504u);
	v30 = ex_mulps(v28, v29);
	v31 = ex_subps(a, v30);
	XLD(v32, 0 + 17601888u);
	v33 = ex_mulps(v28, v32);
	v34 = ex_subps(v31, v33);
	v35 = ex_mulps(v34, v34);
	v38 = ex_mulps(v37, v35);
	XLD(v39, 0 + 40245616u);
	v40 = ex_addps(v38, v39);
	v45 = ex_mulps(v40, v35);
	XLD(v46, 0 + 39924960u);
	v47 = ex_addps(v45, v46);
	v51 = ex_mulps(v47, v35);
	XLD(v52, 0 + 17591776u);
	v53 = ex_addps(v51, v52);
	v82 = xv_andnps(v80, v53);
	XLD(v41, 0 + 39602320u);
	v42 = ex_mulps(v41, v35);
	XLD(v43, 0 + 40266704u);
	v44 = ex_addps(v42, v43);
	v48 = ex_mulps(v44, v35);
	XLD(v49, 0 + 17585280u);
	v50 = ex_addps(v48, v49);
	v36 = ex_mulps(v35, v34);
	v54 = ex_mulps(v50, v36);
	v55 = ex_addps(v54, v34);
	v81 = xv_andps(v80, v55);
	v83 = xv_orps(v82, v81);
	v85 = ex_subps(v0, v83);
	v87 = xv_andnps(v84, v85);
	v86 = xv_andps(v84, v83);
	v88 = xv_orps(v87, v86);
	return v88;
}
XRFAST(00514a60, "Quat::Dot4 (xmm)", 0x90, qs_dot4(a, b))
XRFAST(00514b20, "Quat::Acos (xmm)", 0x340, qs_acos(a))
XRFAST(00514fe0, "Quat::Sin (xmm)", 0x580, qs_sincos(a))
/* 005168e0: thiscall Quat *Quat::Slerp(Quat *out, const Quat *q1, float t) const -- out = q0' w0 + q1 w1 (q0' = q0
 * negated when q0.q1 < 0; sin weights past the cos threshold 0x10a6f30, else 1 - t, t), written through the x87
 * (Vec4::operator= 00455840: a signaling NaN comes out quiet) */
void XRC_CC xo_005168e0(XCtx *c) {
	u32 t = c->ecx, a0 = M32(c->esp + 4), a1 = M32(c->esp + 8), r[4];
	int md = mode(), k;
	X128 p1, p2, o;
	X128 v0, v11, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v38, v40, v41, v42, v43, v45, v46, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61;
	ORIGS(0x005168e0u);
	if (md == 2) { orig(c); return; }
	v43 = xv_ss(M32(c->esp + 0xcu));
	v45 = XV_SHUFPS(v43, v43, 0x00);
	v46 = xv_ss(M32(0 + 0x10aa950u));
	v48 = XV_SHUFPS(v46, v46, 0x00);
	v50 = xv_unpcklps(v48, v45);
	v49 = ex_subps(v48, v45);
	v51 = xv_unpcklps(v50, v49);
	XLD(v22, t + 0u);
	XLD(v11, a1 + 0u);
	v23 = qs_dot4(v22, v11);
	v24 = XV_SHUFPS(v23, v23, 0x00);
	v0 = xv_zero();
	v25 = xv_cmpps1(v24, v0);
	v28 = xv_andnps(v25, v24);
	v26 = ex_subps(v0, v24);
	v27 = xv_andps(v25, v26);
	v29 = xv_orps(v28, v27);
	v42 = qs_acos(v29);
	v52 = ex_mulps(v51, v42);
	v53 = qs_sincos(v52);
	v54 = XV_SHUFPS(v53, v53, 0x00);
	v55 = xv_divps(v53, v54);
	v60 = XV_SHUFPS(v55, v55, 0xaa);
	v38 = xv_ss(M32(0 + 0x10a6f30u));
	v40 = XV_SHUFPS(v38, v38, 0x00);
	v41 = xv_cmpps1(v29, v40);
	v61 = qs_sel(v45, v60, v41);
	v58 = xv_andnps(v41, v49);
	v56 = XV_SHUFPS(v55, v55, 0x55);
	v57 = xv_andps(v41, v56);
	v59 = xv_orps(v58, v57);
	v32 = xv_andnps(v25, v22);
	v30 = ex_subps(v0, v22);
	v31 = xv_andps(v25, v30);
	v33 = xv_orps(v32, v31);
	p1 = ex_mulps(v11, v61);
	p2 = ex_mulps(v33, v59);
	o = ex_addps(p2, p1);
	for (k = 0; k < 4; k++) r[k] = x87_copy(o.u[k]);
	if (md == 1) { orig(c); check("Quat::Slerp", 0x005168e0u, r, a0, 16); return; }
	for (k = 0; k < 4; k++) M32(a0 + 4u * (u32)k) = r[k];
	c->eax = a0;
	xrc_stdcall_ret(c, 3);
}


/* 00eac130: cdecl Render::SetStream(u32 i, u32 buffer, u32 offset, u32 stride) -- the stream table entry
 * 0x283e420 + 16 i: (byte 1, buffer, offset, stride), or all 0 without a buffer; eax = the entry */
static inline void set_stream(XCtx *c, u32 i, u32 buf, u32 off, u32 stride) {
	u32 e = (i << 4) + 0x0283e420u;
	if (buf) { M32(e + 4) = buf; M8(e) = 1; M32(e + 12) = stride; M32(e + 8) = off; c->ecx = stride; c->edx = off; }
	else { M8(e) = 0; M32(e + 4) = 0; M32(e + 12) = 0; M32(e + 8) = 0; c->ecx = 0; c->edx = 0; }
	c->eax = e;
}
void XRC_CC xo_00eac130(XCtx *c) {
	if (mode() == 2) { ORIG(0x00eac130u); orig(c); return; }
	set_stream(c, M32(c->esp + 4), M32(c->esp + 8), M32(c->esp + 12), M32(c->esp + 16));
	xrc_cdecl_ret(c);
}
/* 00f49840: thiscall void VertexLayout::Bind(const u32 *buffers) -- for each 0x24-byte element of the vector at
 * this+8..+0xc: SetStream(e.+0, buffers ? buffers[e.+0x20] : e.+0x1c, 0, e.+8). eax as the original leaves it (the
 * count, or the last entry, or what its count computation left) */
void XRC_CC xo_00f49840(XCtx *c) {
	u32 t = c->ecx, arg = M32(c->esp + 4), beg = M32(t + 8), n, i, eax;
	if (mode() == 2) { ORIG(0x00f49840u); orig(c); return; }
	if (!beg) { n = 0; eax = t + 4; }
	else { s32 d = (s32)(M32(t + 12) - beg); n = (u32)(d / 0x24); eax = n; c->edx = (u32)(d % 0x24); }
	for (i = 0; (s32)i < (s32)n; i++) {
		u32 e = M32(t + 8) + i * 0x24u;
		set_stream(c, M32(e), arg ? M32(arg + 4u * M32(e + 0x20)) : M32(e + 0x1c), 0, M32(e + 8));
		eax = arg ? i + 1 : c->eax;
	}
	c->eax = eax;
	xrc_stdcall_ret(c, 1);
}


/* the CRT's floor / ceil (VC9, SSE2 entries: cdecl double on the stack, the result in st0): exact in any
 * implementation. ~200 calls a frame each (field / battle). And 0100f08c, __libm_sse2_pow or atan2 (xmm0, xmm1 ->
 * xmm0): which one, XRC_FAST_CHECK says (fn_probe). */
static unsigned long long n_fbad;
static void crt_check(XCtx *c, const char *name, double r) {
	static unsigned long long nc;
	double o = c->st0;
	n_checked++;
	if (memcmp(&o, &r, 8) && !(o != o && r != r) && ++n_fbad <= 10)
		fprintf(stderr, "xrc: FAST MISMATCH %s: ours %.17g, original %.17g\n", name, r, o);
	if (++nc, (nc & (nc - 1)) == 0 && nc >= 1024) fprintf(stderr, "xrc: fast check: floor/ceil: %llu calls checked (%llu mismatches)\n", nc, n_fbad);
}
#define CRT_ROUND(addr_, name_, fn_)                                                        \
	void XRC_CC xo_##addr_(XCtx *c) {                                                     \
		double x, r;                                                                      \
		int md = mode();                                                                  \
		ORIGS(0x##addr_##u);                                                               \
		if (md == 2) { orig(c); return; }                                                 \
		x = f64_of(M64(c->esp + 4));                                                      \
		r = fn_(x);                                                                       \
		if (md == 1) { orig(c); crt_check(c, name_, r); return; }                         \
		c->st0 = r;                                                                       \
		xrc_cdecl_ret(c);                                                                 \
	}
CRT_ROUND(01010960, "floor", floor)
CRT_ROUND(01011050, "ceil", ceil)
void XRC_CC xo_0100f08c(XCtx *c) {
	/* __libm_sse2_atan2(y = xmm0, x = xmm1) -> xmm0 (probed: host atan2 equal on 99.97% of the field's calls, the rest
	 * an ulp of a double off; every caller seen rounds to float at once). Check mode: mismatches as floats */
	double y = c->x[0].d[0], x = c->x[1].d[0], r = atan2(y, x);
	int md = mode();
	ORIGS(0x0100f08cu);
	if (md == 2) { orig(c); return; }
	if (md == 1) {
		static unsigned long long n, nbad;
		float a, b;
		orig(c);
		a = (float)r; b = (float)c->x[0].d[0];
		n_checked++; n++;
		if (memcmp(&a, &b, 4) && !(a != a && b != b) && ++nbad <= 10)
			fprintf(stderr, "xrc: FAST MISMATCH 0100f08c atan2(%.9g, %.9g): ours %.9g, original %.9g (as floats)\n", y, x, a, b);
		if ((n & (n - 1)) == 0 && n >= 1024) fprintf(stderr, "xrc: fast check: 0100f08c atan2: %llu calls checked (%llu float mismatches)\n", n, nbad);
		return;
	}
	c->x[0].d[0] = r;
	c->x[0].d[1] = 0; /* (the original's high half: not read by any caller seen; zeroed) */
	xrc_cdecl_ret(c);
}


/* 00ad5550: thiscall Matrix44 *Matrix44::InverseAffine(Matrix44 *out) const -- the rows' cross products over their
 * determinant (dot3, rcpps + a Newton step: the helpers above), w 0, the translation -(t) through the new rows, w 1;
 * written by 00462cd0 from four Vec4 copies (00455840: through the x87, a signaling NaN comes out quiet). Generated
 * from the original's straight-line SSE by symbolic evaluation (scratch symsse.py), the operations in its order */
void XRC_CC xo_00ad5550(XCtx *c) {
	u32 t = c->ecx, out = M32(c->esp + 4), r[16];
	int md = mode(), k;
	X128 v0, v11, v22, v23, v34, v35, v36, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v72, v73, v74, v75, v76;
	XLD(v11, t + 16u);
	XLD(v34, t + 32u);
	v35 = cross3(v11, v34);
	XLD(v22, t + 0u);
	v23 = cross3(v22, v11);
	v52 = xv_unpcklps(v35, v23);
	v36 = cross3(v34, v22);
	v54 = xv_unpcklps(v52, v36);
	v49 = dot3s(v23, v34);
	v50 = XV_SHUFPS(v49, v49, 0x00);
	v51 = rcp_nr(v50);
	v66 = ex_mulps(v54, v51);
	memset(&v0, 0, sizeof v0);
	v76 = setw(v66, v0);
	v56 = XV_SHUFPS(v36, v36, 0x1e);
	v57 = xv_unpckhps(v52, v56);
	v67 = ex_mulps(v57, v51);
	v75 = setw(v67, v0);
	v53 = xv_unpckhps(v35, v23);
	v58 = xv_unpcklps(v53, v56);
	v68 = ex_mulps(v58, v51);
	v74 = setw(v68, v0);
	XLD(v47, t + 48u);
	v48 = ex_subps(v0, v47);
	v60 = XV_SHUFPS(v48, v48, 0xaa);
	v64 = ex_mulps(v58, v60);
	v59 = XV_SHUFPS(v48, v48, 0x55);
	v62 = ex_mulps(v57, v59);
	v55 = XV_SHUFPS(v48, v48, 0x00);
	v61 = ex_mulps(v54, v55);
	v63 = ex_addps(v62, v61);
	v65 = ex_addps(v64, v63);
	v69 = ex_mulps(v65, v51);
	v70 = v0; v70.u[0] = M32(0x010aa950u);
	v72 = XV_SHUFPS(v70, v70, 0x00);
	v73 = setw(v69, v72);
	ORIGS(0x00ad5550u);
	if (md == 2) { orig(c); return; }
	for (k = 0; k < 4; k++) { r[k] = x87_copy(v76.u[k]); r[4 + k] = x87_copy(v75.u[k]); r[8 + k] = x87_copy(v74.u[k]); r[12 + k] = x87_copy(v73.u[k]); }
	if (md == 1) { orig(c); check("Matrix44::InverseAffine", 0x00ad5550u, r, out, 64); return; }
	for (k = 0; k < 16; k++) M32(out + 4u * (u32)k) = r[k];
	c->eax = out;
	xrc_stdcall_ret(c, 1);
}

void XRC_CC xo_00d785d0(XCtx *c); /* ovr_bvh.c */
void XRC_CC xo_00c77820(XCtx *c); /* ovr_unlocked.c */


/* ============================================================================ small hot functions (battle and field)
 * Short functions high in both profiles (2026-11-21): the render layer's wrappers around the proxy device (*0x283eac0)
 * and a few loops. Each does natively what the original does -- the same stack writes (pushed arguments, return
 * addresses, saved registers), the same registers at each call and at the return -- and calls the device method (or
 * the guest function) as a translated call would: an inline cache per site, its return checked. XRC_FAST_SKIP=addr
 * (or XRC_NO_FAST=1) runs a translation instead. No XRC_FAST_CHECK comparison: their effects are calls. */
#if !defined(XRC_INGAME) && !defined(XRC_NO_IC)
static inline void g_call(XCtx *c, XFn f, u32 t, u32 ra) {
	XRC_PREEMPT()
	XRC_DS_PUSH(t) f(c); XRC_DS_POP()
	if (c->eip != ra) xrc_bad_return(c, t);
}
#else
static inline void g_call(XCtx *c, XFn f, u32 t, u32 ra) { (void)f; (void)ra; xrc_dispatch(c, t); }
#endif
/* a call: the return address at `at`, esp = at */
static inline void w_call(XCtx *c, u32 at, u32 t, XIc *ic, u32 ra) { M32(at) = ra; c->esp = at; px_call(c, t, ic, ra); }
#define W_HEAD(addr_) int md = mode(); ORIGS(0x##addr_##u); if (md == 2) { orig(c); return; }
#define W_DEV u32 d = M32(0x0283eac0u), vt = M32(d)

/* 00eaba90 / 00eabab0 / 00eaba70: cdecl (reg, data[, n]): dev->SetVertexShaderConstantF(reg, data, n) (n = 4 / 1 for
 * the two-argument ones); eax = the method, ecx = the vtable, edx = reg at the call */
#define W_VSC(addr_, n_, ra_)                                                                                    \
	void XRC_CC xo_##addr_(XCtx *c) {                                                                     \
		static XIc ic = {0xffffffffu, 0, 0};                                                              \
		u32 sp = c->esp, a1 = M32(sp + 4), a2 = M32(sp + 8);                                              \
		W_HEAD(addr_)                                                                                     \
		W_DEV;                                                                                            \
		M32(sp - 4) = (n_) ? (u32)(n_) : M32(sp + 12); M32(sp - 8) = a2; M32(sp - 12) = a1; M32(sp - 16) = d; \
		c->eax = M32(vt + 0x178); c->ecx = vt; c->edx = a1;                                               \
		w_call(c, sp - 20, c->eax, &ic, ra_);                                                             \
		xrc_cdecl_ret(c);                                                                                 \
	}
W_VSC(00eaba90, 0, 0x00eabaafu)
W_VSC(00eabab0, 4, 0x00eabaccu)
W_VSC(00eaba70, 1, 0x00eaba8cu)

/* 00eabc60: cdecl (stage, tex): the bound texture table 0x283e718 + stage * 48 = tex; dev->SetTexture(stage,
 * (tex ? tex : *0x283eac8)->+0x2c); push esi (esi = the vtable, eax = dev, ecx = the method, edx = stage) */
void XRC_CC xo_00eabc60(XCtx *c) {
	static XIc ic = {0xffffffffu, 0, 0};
	u32 sp = c->esp, st = M32(sp + 4), tx = M32(sp + 8), r;
	W_HEAD(00eabc60)
	M32(st * 48u + 0x0283e718u) = tx;
	r = M32((tx ? tx : M32(0x0283eac8u)) + 0x2c);
	{
		W_DEV;
		M32(sp - 4) = c->esi; M32(sp - 8) = r; M32(sp - 12) = st; M32(sp - 16) = d;
		c->eax = d; c->ecx = M32(vt + 0x104); c->edx = st; c->esi = vt;
		w_call(c, sp - 20, c->ecx, &ic, 0x00eabc93u);
	}
	c->esi = M32(c->esp); c->esp += 4; /* (pop esi) */
	xrc_cdecl_ret(c);
}

/* 00eac1c0 / 00eac1f0: cdecl (shader): the current one at slot_ = shader; dev->SetVertexShader / SetPixelShader
 * (shader->+4) (eax = dev, ecx = the argument, edx = the method); no shader: the original (a jump elsewhere) */
#define W_SHADER(addr_, slot_, off_, ra_)                                                                        \
	void XRC_CC xo_##addr_(XCtx *c) {                                                                     \
		static XIc ic = {0xffffffffu, 0, 0};                                                              \
		u32 sp = c->esp, p = M32(sp + 4);                                                                 \
		W_HEAD(addr_)                                                                                     \
		if (!p) { orig(c); return; }                                                                      \
		M32(slot_) = p;                                                                                   \
		{                                                                                                 \
			W_DEV;                                                                                        \
			M32(sp - 4) = M32(p + 4); M32(sp - 8) = d;                                                    \
			c->eax = d; c->ecx = M32(p + 4); c->edx = M32(vt + (off_));                                   \
			w_call(c, sp - 12, c->edx, &ic, ra_);                                                         \
		}                                                                                                 \
		xrc_cdecl_ret(c);                                                                                 \
	}
W_SHADER(00eac1c0, 0x0283e648u, 0x170, 0x00eac1e2u)
W_SHADER(00eac1f0, 0x0283e64cu, 0x1ac, 0x00eac212u)

/* 00eacf50: cdecl (ib): the current at 0x283e520 = ib; dev->SetIndices(ib ? ib->+0x14 : 0) (eax = the method, ecx =
 * dev, edx = the vtable) */
void XRC_CC xo_00eacf50(XCtx *c) {
	static XIc ic = {0xffffffffu, 0, 0};
	u32 sp = c->esp, p = M32(sp + 4);
	W_HEAD(00eacf50)
	M32(0x0283e520u) = p;
	{
		W_DEV;
		M32(sp - 4) = p ? M32(p + 0x14) : 0; M32(sp - 8) = d;
		c->eax = M32(vt + 0x1a0); c->ecx = d; c->edx = vt;
		w_call(c, sp - 12, c->eax, &ic, p ? 0x00eacf72u : 0x00eacf87u);
	}
	xrc_cdecl_ret(c);
}

/* 00eaf1b0: dev->EndScene() (eax = dev, ecx = the vtable, edx = the method) */
void XRC_CC xo_00eaf1b0(XCtx *c) {
	static XIc ic = {0xffffffffu, 0, 0};
	u32 sp = c->esp;
	W_HEAD(00eaf1b0)
	{
		W_DEV;
		M32(sp - 4) = d;
		c->eax = d; c->ecx = vt; c->edx = M32(vt + 0xa8);
		w_call(c, sp - 8, c->edx, &ic, 0x00eaf1c0u);
	}
	xrc_cdecl_ret(c);
}

/* 00c27cf0: thiscall (i, reg): dev->SetPixelShaderConstantF(reg, (*this)[i * 2 - 1], 1); push esi (esi = i, eax =
 * dev, ecx = the data, edx = the method); ret 8 */
void XRC_CC xo_00c27cf0(XCtx *c) {
	static XIc ic = {0xffffffffu, 0, 0};
	u32 sp = c->esp, i = M32(sp + 4), reg = M32(sp + 8), data = M32(M32(c->ecx) + i * 8u - 4u);
	W_HEAD(00c27cf0)
	{
		W_DEV;
		M32(sp - 4) = c->esi; M32(sp - 8) = 1; M32(sp - 12) = reg; M32(sp - 16) = data; M32(sp - 20) = d;
		c->eax = d; c->ecx = data; c->edx = M32(vt + 0x1b4); c->esi = i;
		w_call(c, sp - 24, c->edx, &ic, 0x00c27d13u);
	}
	c->esi = M32(c->esp); c->esp += 4;
	xrc_stdcall_ret(c, 2);
}

/* 00f32f40: thiscall (tab, reg, data): reg < 0x200 (else the original: an assertion's one-time set-up and call): reg
 * != 0: dev->SetVertexShaderConstantF(reg, data, 1)... the first argument tab->+8's [reg * 2 - 1], the constants'
 * pointer; an ebp frame with this at ebp - 8, the bytes ebp - 1 and ebp - 0xc 0 (reg >= 0x200); ret 0xc. (eax = dev,
 * ecx = the method, edx = the vtable at the call; reg 0: al = cl = 0) */
void XRC_CC xo_00f32f40(XCtx *c) {
	static XIc ic = {0xffffffffu, 0, 0};
	u32 sp = c->esp, a1 = M32(sp + 4), reg = M32(sp + 8), a3 = M32(sp + 12), bp = sp - 4;
	W_HEAD(00f32f40)
	if (reg >= 0x200u) { orig(c); return; }
	M32(bp) = c->ebp; M32(bp - 8) = c->ecx; M8(bp - 1) = 0; M8(bp - 12) = 0;
	if (!reg) {
		c->eax &= ~0xffu; c->ecx &= ~0xffu;
		c->eip = M32(sp); c->esp = sp + 16;
		return;
	}
	{
		W_DEV;
		M32(sp - 20) = 1; M32(sp - 24) = a3; M32(sp - 28) = M32(M32(a1 + 8) + reg * 8u - 4u); M32(sp - 32) = d;
		c->eax = d; c->ecx = M32(vt + 0x178); c->edx = vt; c->ebp = bp;
		w_call(c, sp - 36, c->ecx, &ic, 0x00f32fe4u);
	}
	c->esp = c->ebp; c->ebp = M32(c->esp); c->esp += 4; /* (mov esp, ebp; pop ebp) */
	xrc_stdcall_ret(c, 3);
}

/* 00eac520 / 00eac590: cdecl (sampler, a, b, c[, lod]): dev->SetSamplerState(sampler, s0_ + k, tab_[arg k]) for
 * k = 0..2 (the address modes / the filters); 00eac590 also SetSamplerState(sampler, 8 -- MIPMAPLODBIAS --,
 * (float)(char)lod * *0x10c7fb0), the float stored over its argument (and left in xmm0's low lane). push esi (esi =
 * the sampler; eax = the method, ecx = the vtable, edx = the value at each call; ecx = the bias, edx = the method at
 * the last) */
static inline void w_samp3(XCtx *c, u32 sp, u32 tab, u32 s0, const u32 *ra, XIc *ic) {
	u32 s = M32(sp + 4), k;
	M32(sp - 4) = c->esi;
	for (k = 0; k < 3; k++) {
		u32 v = M32(M32(sp + 8u + 4u * k) * 4u + tab);
		W_DEV;
		M32(sp - 8) = v; M32(sp - 12) = s0 + k; M32(sp - 16) = s; M32(sp - 20) = d;
		c->eax = M32(vt + 0x114); c->ecx = vt; c->edx = v; c->esi = s;
		w_call(c, sp - 24, c->eax, &ic[k], ra[k]);
	}
}
void XRC_CC xo_00eac520(XCtx *c) {
	static XIc ic[3] = {{0xffffffffu, 0, 0}, {0xffffffffu, 0, 0}, {0xffffffffu, 0, 0}};
	static const u32 ra[3] = {0x00eac544u, 0x00eac563u, 0x00eac582u};
	u32 sp = c->esp;
	W_HEAD(00eac520)
	w_samp3(c, sp, 0x02624858u, 1, ra, ic);
	c->esi = M32(c->esp); c->esp += 4;
	xrc_cdecl_ret(c);
}
void XRC_CC xo_00eac590(XCtx *c) {
	static XIc ic[4] = {{0xffffffffu, 0, 0}, {0xffffffffu, 0, 0}, {0xffffffffu, 0, 0}, {0xffffffffu, 0, 0}};
	static const u32 ra[3] = {0x00eac5b4u, 0x00eac5d3u, 0x00eac5f2u};
	u32 sp = c->esp, bits;
	float f;
	W_HEAD(00eac590)
	w_samp3(c, sp, 0x0262486cu, 5, ra, ic);
	f = (float)(int)(signed char)M8(sp + 0x14) * MF32(0x010c7fb0u);
	memcpy(&bits, &f, 4);
	M32(sp + 0x14) = bits;
	c->x[0].u[0] = bits;
	{
		u32 s = c->esi;
		W_DEV;
		M32(sp - 8) = bits; M32(sp - 12) = 8; M32(sp - 16) = s; M32(sp - 20) = d;
		c->eax = d; c->ecx = bits; c->edx = M32(vt + 0x114);
		w_call(c, sp - 24, c->edx, &ic[3], 0x00eac621u);
	}
	c->esi = M32(c->esp); c->esp += 4;
	xrc_cdecl_ret(c);
}

/* 00eac780: cdecl (mode, a, b, c): dev->SetRenderState(0x26248e4[mode * 4 + 2k]... the state table's three entries
 * for mode, 0x2624838[arg k]) for k = 0..2 (eax = the method, ecx = the vtable, edx = the state) */
void XRC_CC xo_00eac780(XCtx *c) {
	static XIc ic[3] = {{0xffffffffu, 0, 0}, {0xffffffffu, 0, 0}, {0xffffffffu, 0, 0}};
	static const u32 ra[3] = {0x00eac7a8u, 0x00eac7d0u, 0x00eac7f8u};
	u32 sp = c->esp, k;
	W_HEAD(00eac780)
	for (k = 0; k < 3; k++) {
		u32 v = M32(M32(sp + 8u + 4u * k) * 4u + 0x02624838u), st = M32(M32(sp + 4) * 4u + 0x026248e4u + 8u * k);
		W_DEV;
		M32(sp - 4) = v; M32(sp - 8) = st; M32(sp - 12) = d;
		c->eax = M32(vt + 0xe4); c->ecx = vt; c->edx = st;
		w_call(c, sp - 16, c->eax, &ic[k], ra[k]);
	}
	xrc_cdecl_ret(c);
}

/* 00eac720: cdecl (mode, a, b, c): dev->SetRenderState(0x26248dc[mode], 0x26247c8[a]), (0x39, b), (0x3a, c) (eax =
 * the method, ecx = the vtable, edx = the state, then the values) */
void XRC_CC xo_00eac720(XCtx *c) {
	static XIc ic[3] = {{0xffffffffu, 0, 0}, {0xffffffffu, 0, 0}, {0xffffffffu, 0, 0}};
	static const u32 ra[3] = {0x00eac748u, 0x00eac75fu, 0x00eac776u};
	u32 sp = c->esp, k;
	W_HEAD(00eac720)
	for (k = 0; k < 3; k++) {
		u32 v = k ? M32(sp + 8u + 4u * k) : M32(M32(sp + 8) * 4u + 0x026247c8u);
		u32 st = k ? 0x38u + k : M32(M32(sp + 4) * 4u + 0x026248dcu);
		W_DEV;
		M32(sp - 4) = v; M32(sp - 8) = st; M32(sp - 12) = d;
		c->eax = M32(vt + 0xe4); c->ecx = vt; c->edx = k ? v : st;
		w_call(c, sp - 16, c->eax, &ic[k], ra[k]);
	}
	xrc_cdecl_ret(c);
}

/* the bone update (00ea6f30 & co): a bone whose world matrix is new copies its rows 1..3 (+0x90 / 0xb0 / 0xd0) over
 * the previous ones (+0x80 / 0xa0 / 0xc0), sets bit 0 of +0xf3, clears +0xf2; v = the last row copied (xmm0) */
static inline void w_bone(u32 e, X128 *v, int fl) {
	X128 a;
	XLD(a, e + 0x90);
	if (fl) { M8(e + 0xf3) |= 1; M8(e + 0xf2) = 0; }
	XST(e + 0x80, a);
	XLD(a, e + 0xb0); XST(e + 0xa0, a);
	XLD(a, e + 0xd0); XST(e + 0xc0, a);
	*v = a;
}

/* 00ea26b0: thiscall (unused): for each of the u16 count at +0x50 entries of +0x5c (28 bytes each), 00ea6f30 -- the
 * bone of its +0xc updated; push esi; push edi (and ebx with entries); eax = the last bone, ecx = the count; ret 4 */
void XRC_CC xo_00ea26b0(XCtx *c) {
	u32 sp = c->esp, th = c->ecx, n = M16(th + 0x50), i = 0, off = 0, e = 0;
	X128 v;
	W_HEAD(00ea26b0)
	M32(sp - 4) = c->esi; M32(sp - 8) = c->edi;
	if (!n) { c->eax = 0; c->eip = M32(sp); c->esp = sp + 8; return; }
	M32(sp - 12) = c->ebx; M32(sp - 16) = 0x00ea26cbu;
	do {
		e = M32(M32(th + 0x5c) + off + 0xc);
		w_bone(e, &v, 1);
		n = M16(th + 0x50); i++; off += 0x1c;
	} while ((int)i < (int)n);
	c->x[0] = v;
	c->eax = e; c->ecx = n;
	c->eip = M32(sp); c->esp = sp + 8;
}

/* 00ea3b50: thiscall (unused): the skeleton's bones (u16 count +0x50; +0x40, 256 bytes each; their transforms +0x44,
 * 160 bytes each): a bone with bit 1 of +0xf3 moves its matrix (w_bone above), then 00ea2bc0 -- or 00ea31b0, bit 1
 * of +0x52 clear -- (bone, transform, its parent's: +0x44 + (s16)bone->+0xf0 * 160); an aligned ebp frame, the index
 * at [esp + 0xc]; eax = edx = the count; ret 4 */
void XRC_CC xo_00ea3b50(XCtx *c) {
	static XFn f1, f2;
	u32 sp = c->esp, th = c->ecx, a, n = M16(th + 0x50), esi, ebx, edi, t, ra, i, e;
	XFn f;
	W_HEAD(00ea3b50)
	if (!f1) { f1 = xrc_translation_of(0x00ea2bc0u); f2 = xrc_translation_of(0x00ea31b0u); }
	M32(sp - 4) = c->ebp;
	a = (sp - 4) & ~15u;
	M32(a - 4) = th; M32(a - 8) = c->ebx; M32(a - 12) = c->esi; M32(a - 16) = c->edi;
	if (!n) { c->eax = 0; c->ecx = 0; c->eip = M32(sp); c->esp = sp + 8; return; }
	if (M8(th + 0x52) & 2) { f = f1; t = 0x00ea2bc0u; ra = 0x00ea3befu; } else { f = f2; t = 0x00ea31b0u; ra = 0x00ea3c7fu; }
	esi = M32(th + 0x40); ebx = M32(th + 0x44); edi = th;
	M32(a - 4) = 0;
	c->ebp = sp - 4;
	for (;;) {
		u32 cl = M8(esi + 0xf3), p = (u32)(int)(short)M16(esi + 0xf0) * 160u + M32(edi + 0x44);
		if (cl & 2) w_bone(esi, &c->x[0], 1);
		M32(a - 20) = p; M32(a - 24) = ebx; M32(a - 28) = esi; M32(a - 32) = ra;
		c->eax = p; c->ecx = edi; c->ebx = ebx; c->esi = esi; c->edi = edi; c->esp = a - 32;
		g_call(c, f, t, ra);
		e = c->esp; edi = c->edi;
		i = M32(e + 0xc) + 1; n = M16(edi + 0x50); c->edx = n;
		esi = c->esi + 0x100; ebx = c->ebx + 0xa0;
		M32(e + 0xc) = i;
		if ((int)i >= (int)n) break;
	}
	c->eax = i; c->edx = n;
	c->edi = M32(e); c->esi = M32(e + 4); c->ebx = M32(e + 8);
	c->esp = c->ebp; c->ebp = M32(c->esp); c->esp += 4;
	xrc_stdcall_ret(c, 1);
}

/* 00f08f70: thiscall (arg): for i < this->+0x44: o = this->+0x48[i], o->vtbl[2](o; arg, this) -- an ebp frame, i at
 * ebp - 4, this at ebp - 8 (eax = the method, ecx = o, edx = its vtable at each call); ret 4 */
void XRC_CC xo_00f08f70(XCtx *c) {
	static XIc ic = {0xffffffffu, 0, 0};
	u32 sp = c->esp, bp = sp - 4, ecx, edx, eax = c->eax;
	W_HEAD(00f08f70)
	M32(bp) = c->ebp; M32(bp - 8) = c->ecx; M32(bp - 4) = 0;
	c->ebp = bp;
	for (;;) {
		u32 th, o, ov;
		bp = c->ebp;
		ecx = M32(bp - 8); edx = M32(bp - 4);
		if ((int)edx >= (int)M32(ecx + 0x44)) break;
		th = M32(bp - 8);
		M32(bp - 12) = th; M32(bp - 16) = M32(bp + 8);
		o = M32(M32(th + 0x48) + M32(bp - 4) * 4u); ov = M32(o);
		c->eax = M32(ov + 8); c->ecx = o; c->edx = ov;
		w_call(c, bp - 20, c->eax, &ic, 0x00f08fbdu);
		bp = c->ebp;
		eax = M32(bp - 4) + 1; M32(bp - 4) = eax;
	}
	c->eax = eax; c->ecx = ecx; c->edx = edx;
	c->esp = bp; c->ebp = M32(bp); c->esp += 4;
	xrc_stdcall_ret(c, 1);
}

/* 008d86d0: thiscall (float t): 008d6990(t), then each child (the list from +8, linked by +0x14) child->vtbl[1](t);
 * bit 3 of +0xcc cleared -- the float passed through the x87 (fld / fstp: a signaling NaN comes out quiet); an ebp
 * frame, this at ebp - 8, the child at ebp - 4; eax = ecx = this, edx = the new +0xcc; ret 4 */
void XRC_CC xo_008d86d0(XCtx *c) {
	static XFn f0;
	static XIc ic = {0xffffffffu, 0, 0};
	u32 sp = c->esp, bp = sp - 4, th = c->ecx, fv = M32(sp + 4), ch;
	W_HEAD(008d86d0)
	if (!f0) f0 = xrc_translation_of(0x008d6990u);
	if ((fv & 0x7f800000u) == 0x7f800000u && (fv & 0x007fffffu) && !(fv & 0x00400000u)) fv |= 0x00400000u;
	M32(bp) = c->ebp; M32(bp - 8) = th;
	M32(bp - 12) = fv; M32(bp - 16) = 0x008d86e8u;
	c->ebp = bp; c->esp = bp - 16;
	g_call(c, f0, 0x008d6990u, 0x008d86e8u);
	bp = c->ebp;
	ch = M32(M32(bp - 8) + 8); M32(bp - 4) = ch;
	while (M32(bp - 4)) {
		u32 o = M32(bp - 4), ov = M32(o);
		M32(bp - 12) = fv;
		c->eax = M32(ov + 4); c->ecx = o; c->edx = ov;
		w_call(c, bp - 16, c->eax, &ic, 0x008d8716u);
		bp = c->ebp;
		M32(bp - 4) = M32(M32(bp - 4) + 0x14);
	}
	th = M32(bp - 8);
	c->edx = M32(th + 0xcc) & ~8u; M32(th + 0xcc) = c->edx;
	c->eax = th; c->ecx = th;
	c->esp = bp; c->ebp = M32(bp); c->esp += 4;
	xrc_stdcall_ret(c, 1);
}


const XrcManual xrc_manual_simd[] = {
	{0x00c77820u, xo_00c77820, "zlib inflate_fast, run without the GIL (ovr_unlocked.c)"},
	{0x00d785d0u, xo_00d785d0, "the collision BVH build step (fast, ovr_bvh.c)"},
	{0x00515c00u, xo_00515c00, "Quat::FromMatrix33 (fast)"},
	{0x00517130u, xo_00517130, "QuatToMatrix2 (fast)"},
	{0x00744160u, xo_00744160, "QuatToMatrix (fast)"},
	{0x00743c70u, xo_00743c70, "Quat::Mul (fast)"},
	{0x00456970u, xo_00456970, "Matrix44::Mul (fast)"},
	{0x00457750u, xo_00457750, "Matrix44::Mul, a second copy (fast)"},
	{0x0074dbf0u, xo_0074dbf0, "Matrix44::Transform, a second copy (fast)"},
	{0x00456f30u, xo_00456f30, "Vec4::Dot3 (fast)"},
	{0x00536990u, xo_00536990, "Matrix44::Inverse (fast)"},
	{0x005b9d70u, xo_005b9d70, "Vec4::Mul (fast)"},
	{0x0045a960u, xo_0045a960, "Vec4::Sub (fast)"},
	{0x00457e50u, xo_00457e50, "Aabb::Transformed (fast)"},
	{0x004559b0u, xo_004559b0, "Matrix44::Transform (fast)"},
	{0x00455840u, xo_00455840, "Vec4::operator= (fast)"},
	{0x00455880u, xo_00455880, "Vec4::Get (fast)"},
	{0x004586c0u, xo_004586c0, "Aabb::Extended (fast)"},
	{0x00710860u, xo_00710860, "Vec4::Dot4 (fast)"},
	{0x00711f40u, xo_00711f40, "Matrix33::Transform (fast)"},
	{0x00455f00u, xo_00455f00, "Matrix44::Transform(m128) (fast)"},
	{0x004584c0u, xo_004584c0, "Matrix44::TransformPoint (fast)"},
	{0x00455d80u, xo_00455d80, "Matrix44::Get (fast)"},
	{0x00462f80u, xo_00462f80, "Matrix44::operator= (fast)"},
	{0x00c4d840u, xo_00c4d840, "Matrix44::Transposed (fast)"},
	{0x00534b70u, xo_00534b70, "Matrix44 copy (fast)"},
	{0x00462cd0u, xo_00462cd0, "Matrix44 from rows (fast)"},
	{0x00ea2bc0u, xo_00ea2bc0, "Bone::UpdateWorld (fast)"},
	{0x00ad5550u, xo_00ad5550, "Matrix44::InverseAffine (fast)"},
	{0x00514a60u, xo_00514a60, "Quat::Dot4 (fast)"},
	{0x00514b20u, xo_00514b20, "Quat::Acos (fast)"},
	{0x00514fe0u, xo_00514fe0, "Quat::Sin (fast)"},
	{0x005168e0u, xo_005168e0, "Quat::Slerp (fast)"},
	{0x01010960u, xo_01010960, "floor (fast)"},
	{0x01011050u, xo_01011050, "ceil (fast)"},
	{0x0100f08cu, xo_0100f08c, "__libm_sse2_atan2 (fast)"},
	{0x00eac130u, xo_00eac130, "Render::SetStream (fast)"},
	{0x00f49840u, xo_00f49840, "VertexLayout::Bind (fast)"},
	{0x00ad4ef0u, xo_00ad4ef0, "Vec4::SetW (xmm) (fast)"},
	{0x00ad5090u, xo_00ad5090, "Vec4::Cross (xmm) (fast)"},
	{0x00ad4f50u, xo_00ad4f50, "Vec4::Dot3 (xmm) (fast)"},
	{0x00ad51c0u, xo_00ad51c0, "Vec4::Rcp (xmm) (fast)"},
	{0x00ec1ff0u, xo_00ec1ff0, "Render::Forget: the pointer table scan (fast)"},
	{0x00eac800u, xo_00eac800, "Render::BindStreams (fast)"},
	{0x00c43bf0u, xo_00c43bf0, "Aabb::Extended, unaligned (fast)"},
	{0x00f14110u, xo_00f14110, "Shapes::Overlap (fast)"},
	{0x00ec2890u, xo_00ec2890, "D3D proxy SetVertexShaderConstantF (fast)"},
	{0x00ec25f0u, xo_00ec25f0, "D3D proxy SetSamplerState (fast)"},
	{0x00ec2950u, xo_00ec2950, "D3D proxy SetStreamSourceFreq (fast)"},
	{0x00ec29b0u, xo_00ec29b0, "D3D proxy SetPixelShaderConstantF (fast)"},
	{0x00ec2910u, xo_00ec2910, "D3D proxy SetVertexShaderConstantB (fast)"},
	{0x00ec2810u, xo_00ec2810, "D3D proxy forwarder 0x15c (fast)"},
	{0x00ec2870u, xo_00ec2870, "D3D proxy forwarder 0x170 (fast)"},
	{0x00ec2990u, xo_00ec2990, "D3D proxy forwarder 0x1ac (fast)"},
	{0x00ec2830u, xo_00ec2830, "D3D proxy forwarder 0x164 (fast)"},
	{0x00ec24d0u, xo_00ec24d0, "D3D proxy SetRenderState (fast)"},
	{0x00ec3370u, xo_00ec3370, "D3D proxy SetStreamSource (fast)"},
	{0x00ec3230u, xo_00ec3230, "D3D proxy SetTexture (fast)"},
	{0x00458460u, xo_00458460, "Vec4::Add (fast)"},
	{0x00710f50u, xo_00710f50, "Vec4::Sub(m128) (fast)"},
	{0x00cae180u, xo_00cae180, "Vec4::Max (fast)"},
	{0x00727710u, xo_00727710, "Vec4::Scale (fast)"},
	{0x00517c80u, xo_00517c80, "Vec4::Neg (fast)"},
	{0x00727690u, xo_00727690, "Vec4::Vec4(xyzw) (fast)"},
	{0x00c28db0u, xo_00c28db0, "Matrix44 from m128 rows (fast)"},
	{0x00db4dd0u, xo_00db4dd0, "Matrix44::Abs (fast)"},
	{0x00eaba90u, xo_00eaba90, "Render SetVertexShaderConstantF(reg, data, n) (fast)"},
	{0x00eabab0u, xo_00eabab0, "Render SetVertexShaderConstantF(reg, data, 4) (fast)"},
	{0x00eaba70u, xo_00eaba70, "Render SetVertexShaderConstantF(reg, data, 1) (fast)"},
	{0x00eabc60u, xo_00eabc60, "Render SetTexture(stage, tex) (fast)"},
	{0x00eac1c0u, xo_00eac1c0, "Render SetVertexShader (fast)"},
	{0x00eac1f0u, xo_00eac1f0, "Render SetPixelShader (fast)"},
	{0x00eacf50u, xo_00eacf50, "Render SetIndices (fast)"},
	{0x00eaf1b0u, xo_00eaf1b0, "Render EndScene (fast)"},
	{0x00c27cf0u, xo_00c27cf0, "SetPixelShaderConstantF(i, reg) wrapper (fast)"},
	{0x00f32f40u, xo_00f32f40, "SetVertexShaderConstantF(tab, reg, data) wrapper (fast)"},
	{0x00eac520u, xo_00eac520, "Render sampler address modes (fast)"},
	{0x00eac590u, xo_00eac590, "Render sampler filters + LOD bias (fast)"},
	{0x00eac780u, xo_00eac780, "Render render states (3 from tables) (fast)"},
	{0x00eac720u, xo_00eac720, "Render render states (table, 0x39, 0x3a) (fast)"},
	{0x00ea26b0u, xo_00ea26b0, "bone matrix moves (00ea6f30 loop) (fast)"},
	{0x00ea3b50u, xo_00ea3b50, "skeleton bone update loop (fast)"},
	{0x00f08f70u, xo_00f08f70, "child list vtbl[2] loop (fast)"},
	{0x008d86d0u, xo_008d86d0, "node update + children vtbl[1] loop (fast)"},
};
const unsigned xrc_manual_simd_n = sizeof xrc_manual_simd / sizeof xrc_manual_simd[0];
