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
static void quat_rows_q(X128 q, X128 *r) {
	X128 q2, qw, qc9, qd2, q2c9, q2d2, one, t, a, b, m180, m150, m120;
	float onef = 1.0f;
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
static void quat_rows(u32 qp, X128 *r) { X128 q; XLD(q, qp); quat_rows_q(q, r); }
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

/* 00e64d50: thiscall Matrix44 *Bone::LocalMatrix(Matrix44 *out) -- a skeleton node's local matrix (3000 calls a frame
 * in battle, 10000 in the field; Bone::UpdateWorld's first step). b = this:
 *   b[0xf2] != 0: out = the 64 bytes at b, row by row (007a8b50: movaps)
 *   else rows 0-2 a rotation matrix's, row 3 = Vec4::operator= of b+0x80 (00455840: the x87 copy):
 *     b[0xf3] & 1: QuatToMatrix (00744160) of Quat::Mul(b+0xe0, Vec4::operator= of b+0xa0) (00743c70)
 *     else 00456420(m, b+0xa0) (angles, with sincos: its translation, called in the guest with m the original's
 *     temporary in its frame below the stack, the same esp)
 *   then Matrix44::Get, the copy constructor, Matrix44(rows), operator= (00455d80, 00534b70, 00462cd0, 00462f80):
 *   integer copies. Returns out (ret 4). sp: esp at its entry (the return address there); o: the 64 bytes. */
static void bone_local_rows(XCtx *c, u32 b, u32 sp, X128 *o) {
	int k;
	if (M8(b + 0xf2)) { for (k = 0; k < 4; k++) XLD(o[k], b + 16u * (u32)k); return; }
	if (M8(b + 0xf3) & 1) {
		X128 t1, a, q;
		for (k = 0; k < 4; k++) t1.u[k] = x87_copy(M32(b + 0xa0u + 4u * (u32)k));
		XLD(a, b + 0xe0u);
		q = quat_mul(a, t1);
		quat_rows_q(q, o);
		for (k = 0; k < 3; k++) o[k] = w0(o[k]);
	} else {
		static XFn euler;
		u32 save = c->esp, frame = ((sp - 4u) & ~15u) - 0x1d0u, args[2], p;
		if (!euler) euler = xrc_translation_of(0x00456420u);
		args[0] = frame + 0xe0u; args[1] = b + 0xa0u;
		c->esp = frame;
		p = xrc_call_fn(c, euler, 0x00456420u, 2, args);
		c->esp = save;
		for (k = 0; k < 3; k++) XLD(o[k], p + 16u * (u32)k);
	}
	for (k = 0; k < 4; k++) o[3].u[k] = x87_copy(M32(b + 0x80u + 4u * (u32)k));
}
/* (its effect: out written; the f2 copy row by row, each read just before its write, as the original) */
static void bone_local_to(XCtx *c, u32 b, u32 sp, u32 out) {
	X128 o[4];
	int k;
	if (M8(b + 0xf2)) { for (k = 0; k < 4; k++) { X128 r; XLD(r, b + 16u * (u32)k); XST(out + 16u * (u32)k, r); } return; }
	bone_local_rows(c, b, sp, o);
	for (k = 0; k < 4; k++) XST(out + 16u * (u32)k, o[k]);
}
void XRC_CC xo_00e64d50(XCtx *c) {
	u32 b = c->ecx, out = M32(c->esp + 4);
	int md = mode();
	ORIGS(0x00e64d50u);
	if (md == 2) { orig(c); return; }
	if (md == 1) {
		X128 o[4];
		bone_local_rows(c, b, c->esp, o);
		c->ecx = b;
		orig(c); check("Bone::LocalMatrix", 0x00e64d50u, o, out, 64); return;
	}
	bone_local_to(c, b, c->esp, out);
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
	{
		/* (Bone::LocalMatrix native -- its entry esp as the call would have it: the argument, then the return address,
		 * below buf; in the checked mode, or with 00e64d50 in XRC_FAST_SKIP, its translation called in the guest) */
		static signed char nat = -1;
		if (nat < 0) nat = (signed char)!fast_skip(0x00e64d50u);
		if (md == 0 && nat) bone_local_to(c, node, buf - 8u, buf);
		else { static XFn local; if (!local) local = xrc_translation_of(0x00e64d50u); xrc_call_fn(c, local, 0x00e64d50u, 1, arg); }
	}
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
	if (!f1) {
		/* (Bone::UpdateWorld: its fast version (which does checked mode itself) unless skipped -- exact since 11-24: 0 mismatches with the
		 * pages built without FMA contraction; before, its translation here) */
		f1 = !fast_skip(0x00ea2bc0u) ? xo_00ea2bc0 : xrc_translation_of(0x00ea2bc0u);
		f2 = xrc_translation_of(0x00ea31b0u); /* (a fast 00ea31b0 was exact but neutral, 11-24: removed) */
	}
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


/* 0101a95d: cdecl void *__VEC_memzero(void *dst, int val, size_t size) -- the CRT's SSE2 zero fill: memset (0100df80)
 * jumps here for val == 0 and size >= 256 (and it calls itself once). A head up to the next 16 bytes by rep stosb (its
 * length from the SIGNED dst % 16: 1..31 bytes), 128-byte movdqa blocks (0101a906), the tail by rep stosb; returns dst.
 * For size >= 32 that is exactly [dst, dst + size) zeroed, done here with one memset; the registers left as the original
 * leaves them (eax dst, ecx 0, edx the length of the last tail; edi kept). Smaller sizes (never from memset): the
 * translation. */
void XRC_CC xo_0101a95d(XCtx *c) {
	u32 d = M32(c->esp + 4), n = M32(c->esp + 12), r = (u32)((s32)d % 16), m = n;
	int md = mode();
	ORIGS(0x0101a95du);
	if (md == 2 || n < 32u) { orig(c); return; }
	if (r) m = n - (16u - r); /* (the second pass's size: the head done) */
	if (md == 1) {
		static const u8 zero[4096];
		u32 k;
		orig(c);
		if (c->eax != d || c->ecx != 0 || c->edx != (m & 0x7fu)) {
			n_bad++;
			fprintf(stderr, "xrc: FAST MISMATCH 0101a95d (__VEC_memzero) dst %08x size %u: eax %08x ecx %08x edx %08x (ours edx %08x)\n",
			        d, n, c->eax, c->ecx, c->edx, m & 0x7fu);
		}
		for (k = 0; k < n; k += 4096u) check("__VEC_memzero", 0x0101a95du, zero, d + k, n - k < 4096u ? n - k : 4096u);
		return;
	}
	memset(XM + d, 0, n);
	c->eax = d;
	c->ecx = 0;
	c->edx = m & 0x7fu;
	xrc_cdecl_ret(c);
}

#if defined(__x86_64__) && defined(XRC_VEC) /* (battle batches: x86 inline asm; elsewhere the translations, below) */


/* ======================================================================= battle batch 2 framework (2026-11-25, ovr_simd.c; from batch 1)
 * Hot in a Chapter 1 battle (headless XRC_BENCH, XRC_PROFILE_WALL + XRC_PROFILE_GIL: the main thread, and the thread
 * it waits on): mostly the renderer's material / shader selection and state code, compiled without optimization (/Od:
 * every local lives in the stack frame and is read back after each statement). These keep the locals in registers.
 * Same effects: the same writes outside the stack, the same calls (arguments, return addresses, the registers a
 * callee reads), eax (where it is a result), esp, eip and ebp at the return. The original's dead locals are not
 * written; where calls are made the frame's saved ebp is (the ebp chain).
 *
 * XRC_FAST_CHECK=1 checks every call: ours runs first, its writes outside the stack journaled (BW8 / BW16 / BW32),
 * which are then undone; the translation runs from the same registers; each byte ours wrote must then hold the same
 * value, and esp, eip, ebp, ebx, esi, edi (and eax where it is a result) must match. Their calls run twice then: the
 * callees here are getters, setters of the same values, or rebuild what they write from scratch. A batch function
 * called inside another's check just runs (it is checked at its own outer calls). */
static XCtx *bj_jctx, *bj_owner; /* (the thread whose call is being checked: its context; bj_jctx while ours runs) */
static unsigned bj_n, bj_over;
#define BJ_MAX (1u << 18)
static struct BJ { u32 a; u8 old, val; } *bj;
static XRC_COLDFN void bj_note(u32 a, unsigned n) {
	unsigned k;
	if (!bj) bj = (struct BJ *)calloc(BJ_MAX, sizeof *bj);
	if (!bj || bj_n + n > BJ_MAX) { bj_over = 1; return; }
	for (k = 0; k < n; k++) { bj[bj_n].a = a + k; bj[bj_n].old = M8(a + k); bj_n++; }
}
#define BW8(a_, v_) do { u32 a8_ = (a_); u8 v8_ = (u8)(v_); if (XRC_UNL(bj_jctx == c)) bj_note(a8_, 1); M8(a8_) = v8_; } while (0)
#define BW16(a_, v_) do { u32 a16_ = (a_); u16 v16_ = (u16)(v_); if (XRC_UNL(bj_jctx == c)) bj_note(a16_, 2); M16(a16_) = v16_; } while (0)
#define BW32(a_, v_) do { u32 a32_ = (a_); u32 v32_ = (u32)(v_); if (XRC_UNL(bj_jctx == c)) bj_note(a32_, 4); M32(a32_) = v32_; } while (0)
#define BR_EAX 1u
#define BR_ECX 2u
#define BR_EDX 4u
#define BR_XMM0 64u /* xmm0 (all 4 lanes) is a result */
#define BR_ST0 128u /* x87 st(0) (c->st0) is a result */
#define BR_XMM0L 256u /* xmm0 lane 0 is a result */
typedef void (*BFn)(XCtx *, XFn);
static struct { u32 addr; const char *what; unsigned long long n, bad, nocmp; } btal[64];
/* set by ours during a checked call whose result depends on a callee that differs between the two runs (rand): that call
 * is not compared (counted as not compared) */
static int b_nocmp;
/* a value ours changed through a callee whose effect the translation's run must start from too (the CRT rand seed): put back
 * before the translation runs */
static u32 b_rs_at, b_rs_val;
static int b_rs;
static int nbtal;
static XRC_COLDFN void b_check(XCtx *c, u32 addr, const char *what, BFn ours, XFn orig, unsigned regs) {
	XCtx in = *c, mine;
	unsigned k, n, i, bad = 0, bk = 0;
	bj_owner = c; bj_jctx = c; bj_n = 0; bj_over = 0; b_nocmp = 0; b_rs = 0;
	ours(c, orig);
	bj_jctx = 0;
	mine = *c;
	n = bj_n;
	for (k = 0; k < n; k++) bj[k].val = M8(bj[k].a);
	for (k = n; k-- > 0;) M8(bj[k].a) = bj[k].old;
	*c = in;
	if (b_rs) { M32(b_rs_at) = b_rs_val; b_rs = 0; }
	orig(c);
	bj_owner = 0;
	if (b_nocmp) { for (i = 0; i < (unsigned)nbtal && btal[i].addr != addr; i++) {} if (i < (unsigned)nbtal) btal[i].nocmp++; return; }
	/* (bytes below the returned esp -- the dead frames of the call and its callees, 256 KB -- are not compared: nothing
	 * reads them after the return; both runs must end with the same esp) */
	for (k = 0; k < n; k++) if ((u32)(mine.esp - 1u - bj[k].a) >= 0x40000u && M8(bj[k].a) != bj[k].val) { bad |= 1; bk = k; break; }
	if (c->esp != mine.esp || c->eip != mine.eip || c->ebp != mine.ebp || c->ebx != mine.ebx || c->esi != mine.esi || c->edi != mine.edi) bad |= 2;
	if ((regs & BR_EAX) && c->eax != mine.eax) bad |= 4;
	if ((regs & BR_ECX) && c->ecx != mine.ecx) bad |= 8;
	if ((regs & BR_EDX) && c->edx != mine.edx) bad |= 16;
	if ((regs & BR_XMM0) && memcmp(&c->x[0], &mine.x[0], 16)) bad |= 64;
	if ((regs & BR_XMM0L) && c->x[0].u[0] != mine.x[0].u[0]) bad |= 64;
	if ((regs & BR_ST0) && memcmp(&c->st0, &mine.st0, sizeof c->st0)) bad |= 128;
	if (bj_over) bad |= 32;
	for (i = 0; i < (unsigned)nbtal && btal[i].addr != addr; i++) {}
	if (i == (unsigned)nbtal && nbtal < 64) { btal[i].addr = addr; btal[i].what = what; nbtal++; }
	n_checked++;
	if (bad) n_bad++;
	if (i < 64) {
		btal[i].n++;
		if (bad) btal[i].bad++;
		if ((btal[i].n & (btal[i].n - 1)) == 0 && btal[i].n >= 64)
			fprintf(stderr, "xrc: fast check: %08x %s: %llu calls checked, %llu mismatches (%llu not compared)\n", addr, what, btal[i].n, btal[i].bad, btal[i].nocmp);
	}
	if (bad && (i >= 64 || btal[i].bad <= 4)) {
		fprintf(stderr, "xrc: FAST MISMATCH %08x (%s):%s%s%s%s%s%s", addr, what, bad & 1 ? " memory" : "", bad & 2 ? " esp/eip/ebp/ebx/esi/edi" : "",
		        bad & 4 ? " eax" : "", bad & 8 ? " ecx" : "", bad & 16 ? " edx" : "", bad & 32 ? " (journal full)" : "");
		if (bad & 64) fprintf(stderr, " xmm0 ours %08x %08x %08x %08x, original %08x %08x %08x %08x", mine.x[0].u[0], mine.x[0].u[1], mine.x[0].u[2], mine.x[0].u[3], c->x[0].u[0], c->x[0].u[1], c->x[0].u[2], c->x[0].u[3]);
		if (bad & 128) fprintf(stderr, " st0 ours %.9g, original %.9g", mine.st0, c->st0);
		if (bad & 1) fprintf(stderr, " -- at %08x ours %02x, original %02x (%u bytes written)", bj[bk].a, bj[bk].val, M8(bj[bk].a), n);
		fprintf(stderr, "; ours eax %08x esp %08x eip %08x ebp %08x, original eax %08x esp %08x eip %08x ebp %08x\n", mine.eax, mine.esp, mine.eip, mine.ebp, c->eax,
		        c->esp, c->eip, c->ebp);
	}
}
/* a batch function: xo_ (mode, XRC_FAST_SKIP, the check) around bo_ (ours) */
#define B_FN(addr_, what_, regs_)                                                                                         \
	static void bo_##addr_(XCtx *c, XFn orig);                                                                           \
	void XRC_CC xo_##addr_(XCtx *c) {                                                                                      \
		int md = mode();                                                                                                   \
		ORIGS(0x##addr_##u);                                                                                               \
		if (md == 2) { orig(c); return; }                                                                                  \
		if (md == 1 && !bj_owner) { b_check(c, 0x##addr_##u, what_, bo_##addr_, orig, (regs_)); return; }                  \
		bo_##addr_(c, orig);                                                                                               \
	}                                                                                                                      \
	static void bo_##addr_(XCtx *c, XFn orig __attribute__((unused)))
/* a call as the original makes it: the return address at at_, esp = at_, through the dispatcher's mapping (an inline
 * cache per site: a direct callee hits it every time) */
#define BCALL(at_, t_, ra_) do { static XIc ic_ = {0xffffffffu, 0, 0}; w_call(c, (at_), (t_), &ic_, (ra_)); } while (0)
/* the frame: push ebp / mov ebp, esp (bp = sp - 4) ... mov esp, ebp / pop ebp / ret 4 * nargs */
#define B_ENTER(bp_) do { M32(bp_) = c->ebp; c->ebp = (bp_); } while (0)
#define B_RET(sp_, nargs_) do { c->ebp = M32((sp_) - 4); c->esp = (sp_); xrc_stdcall_ret(c, (nargs_)); } while (0)

/* a virtual getter (o->vtbl[slot / 4]() with ecx = o): when the target's code is "mov eax, [ecx + d]; ret" or "lea
 * eax, [ecx + d]; ret" (d 0, 8 or 32 bits) it is done here (its call wrote nothing but the return address), else
 * called. Registers as at the original's call: eax the method, ecx = o, edx the vtable. Returns eax. */
static struct { u32 t; s32 d; int kind; } bgk[128];
static int b_gk_off = -1;
static int b_getter_kind(u32 t, s32 *d) {
	unsigned h = (t >> 4) & 127u;
	int k = 0;
	if (bgk[h].t == t && t) { *d = bgk[h].d; return bgk[h].kind; }
	if (b_gk_off < 0) b_gk_off = M8(0x00efec60u) != 0x55 || getenv("XRC_NO_GETTER_INLINE") != 0; /* (the guest image's code is readable) */
	*d = 0;
	if (!b_gk_off && t >= 0x00401000u && t < 0x01100000u) {
		u8 b0 = M8(t), b1 = M8(t + 1);
		if ((b0 == 0x8b || b0 == 0x8d) && b1 == 0x41 && M8(t + 3) == 0xc3) { k = b0 == 0x8b ? 1 : 2; *d = (s32)(s8)M8(t + 2); }
		else if ((b0 == 0x8b || b0 == 0x8d) && b1 == 0x81 && M8(t + 6) == 0xc3) { k = b0 == 0x8b ? 1 : 2; *d = (s32)M32(t + 2); }
		else if (b0 == 0x8b && b1 == 0x01 && M8(t + 2) == 0xc3) k = 1;
		else if (b0 == 0x8b && b1 == 0xc1 && M8(t + 2) == 0xc3) k = 2;
	}
	bgk[h].t = t; bgk[h].d = *d; bgk[h].kind = k;
	return k;
}
static inline u32 b_vget(XCtx *c, u32 o, u32 slot, u32 at, u32 ra, XIc *ic) {
	u32 vt = M32(o), m = M32(vt + slot);
	s32 d;
	int k = b_getter_kind(m, &d);
	c->ecx = o; c->edx = vt; c->eax = m;
	if (k == 1) return c->eax = M32(o + (u32)d);
	if (k == 2) return c->eax = o + (u32)d;
	w_call(c, at, m, ic, ra);
	return c->eax;
}
#define BVGET(o_, slot_, at_, ra_) ({ static XIc icg_ = {0xffffffffu, 0, 0}; b_vget(c, (o_), (slot_), (at_), (ra_), &icg_); })

/* the engine's assertion: the handler pointer at 026e80c0 (set to deflt on first use, flag 026e80c4 bit 0), called
 * cdecl with 5 dwords (pushed a4 .. a0) at at_ */
static void b_assert(XCtx *c, u32 at, u32 deflt, u32 a0, u32 a1, u32 a2, u32 a3, u32 a4, u32 ra) {
	if (!(M32(0x026e80c4u) & 1)) { BW32(0x026e80c4u, M32(0x026e80c4u) | 1); BW32(0x026e80c0u, deflt); }
	M32(at + 4) = a0; M32(at + 8) = a1; M32(at + 12) = a2; M32(at + 16) = a3; M32(at + 20) = a4;
	BCALL(at, M32(0x026e80c0u), ra);
}

/* exact float ops for batch 2: inline asm, so the compiler's fast math cannot reorder, fold, swap operands (NaN
 * propagation) or estimate. Packed (X128 in, X128 out) and scalar (lane 0 computed, lanes 1..3 of a kept, as the SSE
 * scalar instructions do). ex_mulps / ex_addps / ex_subps / ex_minps / ex_maxps are defined above (ovr_simd.c). */
#if defined(__x86_64__)
#define EXOP2(name_, ins_) static inline X128 name_(X128 a, X128 b) { __asm__(ins_ " %1, %0" : "+x"(a.v) : "x"(b.v)); return a; }
EXOP2(ex_divps, "divps") EXOP2(ex_mulss, "mulss") EXOP2(ex_addss, "addss") EXOP2(ex_subss, "subss") EXOP2(ex_divss, "divss")
EXOP2(ex_minss, "minss") EXOP2(ex_maxss, "maxss") EXOP2(ex_andps, "andps") EXOP2(ex_orps, "orps") EXOP2(ex_xorps, "xorps")
EXOP2(ex_andnps, "andnps") EXOP2(ex_unpcklps, "unpcklps") EXOP2(ex_unpckhps, "unpckhps") EXOP2(ex_movlhps, "movlhps")
EXOP2(ex_movhlps, "movhlps") EXOP2(ex_sqrtss2, "sqrtss") EXOP2(ex_rsqrtss2, "rsqrtss") EXOP2(ex_rcpss2, "rcpss")
#define EXOP1(name_, ins_) static inline X128 name_(X128 a) { X128 r; __asm__(ins_ " %1, %0" : "=x"(r.v) : "x"(a.v)); return r; }
EXOP1(ex_sqrtps, "sqrtps")
/* packed rcpps / rsqrtps as the translation computes them (xrc.h xv_rcpps / xv_rsqrtps): exact divides 1 / x and
 * 1 / sqrt(x), not the hardware estimates (the scalar rcpss / rsqrtss are the hardware's, as xrcpf / xrsqrtf) */
static inline X128 ex_rcpps(X128 a) { X128 one; one.u[0] = one.u[1] = one.u[2] = one.u[3] = 0x3f800000u; return ex_divps(one, a); }
static inline X128 ex_rsqrtps(X128 a) { X128 one; one.u[0] = one.u[1] = one.u[2] = one.u[3] = 0x3f800000u; return ex_divps(one, ex_sqrtps(a)); }
/* sqrtss / rsqrtss / rcpss: lane 0 from b's lane 0, lanes 1..3 of a (sqrtss xmm_a, xmm_b) */
#define ex_sqrtss(a_, b_) ex_sqrtss2((a_), (b_))
#define ex_rsqrtss(a_, b_) ex_rsqrtss2((a_), (b_))
#define ex_rcpss(a_, b_) ex_rcpss2((a_), (b_))
/* cmpps / cmpss with predicate imm (0 eq, 1 lt, 2 le, 3 unord, 4 neq, 5 nlt, 6 nle, 7 ord) */
#define ex_cmpps(a_, b_, imm_) ({ X128 ca_ = (a_), cb_ = (b_); __asm__("cmpps %2, %1, %0" : "+x"(ca_.v) : "x"(cb_.v), "i"(imm_)); ca_; })
#define ex_cmpss(a_, b_, imm_) ({ X128 ca_ = (a_), cb_ = (b_); __asm__("cmpss %2, %1, %0" : "+x"(ca_.v) : "x"(cb_.v), "i"(imm_)); ca_; })
/* shufps a, b, imm (lanes 0,1 from a, 2,3 from b) */
#define ex_shufps(a_, b_, imm_) ({ X128 sa_ = (a_), sb_ = (b_); __asm__("shufps %2, %1, %0" : "+x"(sa_.v) : "x"(sb_.v), "i"(imm_)); sa_; })
/* comiss / ucomiss a, b as the flags' outcomes: ZF, PF, CF (unordered: all three set) */
static inline unsigned ex_ucomiss(X128 a, X128 b) { unsigned char z, p, cf; __asm__("ucomiss %3, %4\n\tsetz %0\n\tsetp %1\n\tsetc %2" : "=r"(z), "=r"(p), "=r"(cf) : "x"(b.v), "x"(a.v) : "cc"); return (unsigned)z | (unsigned)p << 1 | (unsigned)cf << 2; }
static inline unsigned ex_comiss(X128 a, X128 b) { unsigned char z, p, cf; __asm__("comiss %3, %4\n\tsetz %0\n\tsetp %1\n\tsetc %2" : "=r"(z), "=r"(p), "=r"(cf) : "x"(b.v), "x"(a.v) : "cc"); return (unsigned)z | (unsigned)p << 1 | (unsigned)cf << 2; }
/* cvtsi2ss (lane 0 of a replaced), cvttss2si, cvtss2si (the MXCSR rounding: the guest's, as the translation uses) */
static inline X128 ex_cvtsi2ss(X128 a, s32 i) { __asm__("cvtsi2ss %1, %0" : "+x"(a.v) : "r"(i)); return a; }
static inline s32 ex_cvttss2si(X128 a) { s32 r; __asm__("cvttss2si %1, %0" : "=r"(r) : "x"(a.v)); return r; }
static inline s32 ex_cvtss2si(X128 a) { s32 r; __asm__("cvtss2si %1, %0" : "=r"(r) : "x"(a.v)); return r; }
#endif
/* a float lane as X128 (lanes 1..3 zero, as movss from memory) and back */
static inline X128 ex_ss(u32 bits) { X128 r; r.u[0] = bits; r.u[1] = r.u[2] = r.u[3] = 0; return r; }
static inline X128 ex_ld(u32 a) { X128 r; XLD(r, a); return r; }
static inline X128 ex_ldss(u32 a) { return ex_ss(M32(a)); }
/* journaled stores (outside the stack): 16 bytes / one lane */
#define BW128(a_, x_) do { X128 bw_ = (x_); u32 ba_ = (a_); BW32(ba_, bw_.u[0]); BW32(ba_ + 4, bw_.u[1]); BW32(ba_ + 8, bw_.u[2]); BW32(ba_ + 12, bw_.u[3]); } while (0)
#define BWSS(a_, x_) BW32((a_), (x_).u[0])
/* a float compare as ucomiss's "equal" (ZF set, PF clear): neither a NaN, +0 == -0 */
static inline int b_feq(u32 a, u32 b) {
	if ((a & 0x7fffffffu) > 0x7f800000u || (b & 0x7fffffffu) > 0x7f800000u) return 0;
	if (!((a | b) & 0x7fffffffu)) return 1;
	return a == b;
}
/* fld (float) + fistp (64-bit, chop): the low dword (NaN, infinities and |x| >= 2^63: the indefinite integer, 0) */
static inline u32 b_fist_chop(u32 f) {
	float x;
	if (((f >> 23) & 0xffu) >= 127u + 63u) return 0;
	memcpy(&x, &f, 4);
	return (u32)(s64)x;
}
/* ---- battle batch 2 */
/* ---- battle batch 2, part A: 00f99040, 00f98690, 00fa0700, 00cd8240, 00a39b20 (optimized code, not /Od: they keep
 * their values in registers already; these save the translation's flag / stack-slot traffic and the CRT calls)
 *
 * PA_CRT_CALLS 1: the CRT integer / x87 helpers are called as the original does (BCALL: _allshr 01012100, _allmul
 * 0100e890, _ftol2 0100fd76 with st0 in c->fa[0]). 0 (default): computed here -- _allshr / _allmul are pure register /
 * argument arithmetic (the translation inlines them itself), _ftol2 is the runtime's manual m_ftol2 (ovr_crt_math.c:
 * xrc_i64(xrc_rint(st0, 3))), so nothing a caller or the checker sees differs. */
#ifndef PA_CRT_CALLS
#define PA_CRT_CALLS 0
#endif

/* the x87 as the translation computes it (xrc.h: double arithmetic, XR87 = rounded to float when the guest FCW's
 * precision control is 24-bit, fild = xrc_i64d, fstp float = cvtsd2ss): SSE2 double instructions, exact like ex_* */
#define PA_DOP2(name_, ins_) static inline double name_(double a, double b) { __asm__(ins_ " %1, %0" : "+x"(a) : "x"(b)); return a; }
PA_DOP2(pa_mulsd, "mulsd") PA_DOP2(pa_subsd, "subsd") PA_DOP2(pa_divsd, "divsd")
static inline double pa_sqrtsd(double a) { double r = 0; __asm__("sqrtsd %1, %0" : "+x"(r) : "x"(a)); return r; }
static inline u32 pa_d2f(double v) { X128 r = ex_ss(0); __asm__("cvtsd2ss %1, %0" : "+x"(r.v) : "x"(v)); return r.u[0]; }
static inline double pa_f2d(u32 bits) { X128 a = ex_ss(bits); double r; __asm__("cvtss2sd %1, %0" : "=x"(r) : "x"(a.v)); return r; }
static inline double pa_r87(double v, int pc24) { return pc24 ? pa_f2d(pa_d2f(v)) : v; }
#define PA_SG 0x80000000u

/* 00f99040: cdecl (out, src, flag) ret: a quaternion unpacked from 64 bits (src: lo = [src], hi = [src + 4] & 0x7ff):
 * w = (lo & 0x1ffff) / 131071 (x87, float; flag: w = 1 - w * w, SSE); m = (hi:lo >> 17) & 0x3fffff (_allshr),
 * q = (s64)sqrt(m) (x87 fsqrt, _ftol2), phi = (1 - q / 2047) * pi/2, th = q ? (m - q * q) / (2 q) * pi/2 : 0 (x87,
 * _allmul); s = sqrt(1 - w * w); Y = sin-series of (-th, pi/2 - th, phi, phi - pi/2) (x + V1 x^3 + V2 x^5 + V3 x^7 +
 * V4 x^9, V1..V4 the function-static vectors at 0286d8e0 / d0 / c0 / b0, set on the first call: guard bits 0..3 of
 * 0286d8f0); out = (-Y1 Y3 s, Y2 s, Y3 Y0 s, w) with signs from hi bits 0x200 / 0x100 / 0x80 / 0x400 (see below);
 * eax = hi & 0x80, ecx = hi & 0x100, edx = out. The x87 part rounds as the guest FCW says (c->fcw, as XR87). */
B_FN(00f99040, "quaternion: unpack from 64 bits", BR_EAX) {
	u32 sp = c->esp, out = M32(sp + 4), src = M32(sp + 8), bp = sp - 4, fa = (bp & ~15u) - 0x44u; /* (fa: esp after the sub) */
	u32 lo = M32(src), hi = M32(src + 4) & 0x7ffu, m, g, r0, r1, r2, tw;
	int pc24 = ((c->fcw >> 8) & 3u) == 0;
	double d0, d1, pi2d = pa_f2d(0x3fc90fdbu); /* (010a1f9c, pi/2) */
	u64 q;
	X128 t, s, th, ph, pi2 = ex_ss(0x3fc90fdbu), sg, a0, a1, a2, xx, s2, x3, x5, x7, x9, y, nw, aa, bb, cc, rr;
	(void)fa;
#if PA_CRT_CALLS
	B_ENTER(bp);
#endif
	/* w: fild (lo & 0x1ffff) * [0265e5f4] (1/131071), fstp float */
	d0 = pa_r87(pa_mulsd((double)(lo & 0x1ffffu), pa_f2d(0x37000040u)), pc24);
	t = ex_ss(pa_d2f(d0));
	if (M8(sp + 12)) t = ex_ss(ex_subss(ex_ss(0x3f800000u), ex_mulss(t, t)).u[0]);
	/* m, q */
#if PA_CRT_CALLS
	c->eax = lo; c->edx = hi; c->ecx = (lo & 0x1ff00u) | 0x11u;
	BCALL(fa - 16, 0x01012100u, 0x00f990b1u);
	m = c->eax & 0x3fffffu;
	d0 = pa_r87(pa_sqrtsd((double)m), pc24);
	c->fa[0] = d0;
	BCALL(fa - 16, 0x0100fd76u, 0x00f990d0u);
	q = (u64)c->edx << 32 | c->eax;
	c->ecx = c->eax | c->edx;
#else
	m = ((lo >> 17) | (hi << 15)) & 0x3fffffu;
	d0 = pa_r87(pa_sqrtsd((double)m), pc24);
	c->fa[0] = d0;
	q = xrc_i64(xrc_rint(d0, 3));
#endif
	/* phi: fild q * [0265e5f0] (1/2047), 1 - that, * pi/2, fstp float */
	d1 = pa_r87(pa_mulsd(xrc_i64d(q), pa_f2d(0x3a001002u)), pc24);
	d1 = pa_r87(pa_subsd(1.0, d1), pc24);
	d1 = pa_r87(pa_mulsd(d1, pi2d), pc24);
	ph = ex_ss(pa_d2f(d1));
	if (!q) th = ex_ss(0);
	else {
		u64 qq, e;
#if PA_CRT_CALLS
		u32 at = fa - 32;
		M32(at + 4) = (u32)q; M32(at + 8) = (u32)(q >> 32); M32(at + 12) = (u32)q; M32(at + 16) = (u32)(q >> 32);
		BCALL(at, 0x0100e890u, 0x00f9910au);
		qq = (u64)c->edx << 32 | c->eax;
		M32(at + 4) = (u32)q; M32(at + 8) = (u32)(q >> 32); M32(at + 12) = 2; M32(at + 16) = 0;
		c->eax = (u32)q; c->edx = (u32)(q >> 32);
		BCALL(at, 0x0100e890u, 0x00f9912du);
		e = (u64)c->edx << 32 | c->eax;
#else
		qq = q * q;
		e = q * 2u;
#endif
		/* fild (m - q * q) / fild (2 q), * pi/2 (the fld'd constant still on the stack), fstp float */
		d1 = pa_r87(pa_divsd(xrc_i64d((u64)m - qq), xrc_i64d(e)), pc24);
		d1 = pa_r87(pa_mulsd(pi2d, d1), pc24);
		th = ex_ss(pa_d2f(d1));
	}
	/* s = sqrt(1 - w * w) */
	s = ex_subss(ex_ss(0x3f800000u), ex_mulss(t, t));
	s = ex_sqrtss(s, s);
	/* the static vectors (-1/6, 1/120, -1/5040, 1/362880 from 0265e5ec / e8 / e4 / e0) */
	g = M32(0x0286d8f0u);
	if (!(g & 1)) { g |= 1; BW32(0x0286d8f0u, g); BW32(0x0286d8e0u, 0xbe2aaaa4u); BW32(0x0286d8e4u, 0xbe2aaaa4u); BW32(0x0286d8e8u, 0xbe2aaaa4u); BW32(0x0286d8ecu, 0xbe2aaaa4u); }
	if (!(g & 2)) { g |= 2; BW32(0x0286d8f0u, g); BW32(0x0286d8d0u, 0x3c08873eu); BW32(0x0286d8d4u, 0x3c08873eu); BW32(0x0286d8d8u, 0x3c08873eu); BW32(0x0286d8dcu, 0x3c08873eu); }
	if (!(g & 4)) { g |= 4; BW32(0x0286d8f0u, g); BW32(0x0286d8c0u, 0xb94fb21fu); BW32(0x0286d8c4u, 0xb94fb21fu); BW32(0x0286d8c8u, 0xb94fb21fu); BW32(0x0286d8ccu, 0xb94fb21fu); }
	if (!(g & 8)) { g |= 8; BW32(0x0286d8f0u, g); BW32(0x0286d8b0u, 0x362e9c14u); BW32(0x0286d8b4u, 0x362e9c14u); BW32(0x0286d8b8u, 0x362e9c14u); BW32(0x0286d8bcu, 0x362e9c14u); }
	/* x = (-th, pi/2 - th, phi, phi - pi/2) (th's lanes 1..3 zero: -0 after the xorps) */
	sg.u[0] = sg.u[1] = sg.u[2] = sg.u[3] = PA_SG;
	a0 = ex_xorps(th, sg);
	a1 = ex_subss(pi2, th);
	a2 = ex_subss(ph, pi2);
	a1 = ex_unpcklps(a1, a2);
	a0 = ex_unpcklps(a0, ph);
	xx = ex_unpcklps(a0, a1);
	s2 = ex_mulps(xx, xx);
	x3 = ex_mulps(s2, xx);
	x5 = ex_mulps(x3, s2);
	x7 = ex_mulps(x5, s2);
	x9 = ex_mulps(x7, s2);
	y = ex_addps(ex_mulps(ex_ld(0x0286d8e0u), x3), xx);
	y = ex_addps(ex_mulps(ex_ld(0x0286d8d0u), x5), y);
	y = ex_addps(ex_mulps(ex_ld(0x0286d8c0u), x7), y);
	y = ex_addps(ex_mulps(ex_ld(0x0286d8b0u), x9), y);
	/* (Y1 * -Y3, Y2, -(-Y3 * Y0), 0) * s (-Y3: Y3 * [0109f7c0] -1.0) */
	nw = ex_mulss(ex_ss(y.u[3]), ex_ss(0xbf800000u));
	aa = ex_mulss(ex_ss(y.u[1]), nw);
	bb = ex_xorps(ex_mulss(nw, ex_ss(y.u[0])), ex_ss(PA_SG));
	cc = ex_unpcklps(ex_ss(y.u[2]), ex_ss(0));
	aa = ex_unpcklps(aa, bb);
	aa = ex_unpcklps(aa, cc);
	rr = ex_mulps(aa, ex_shufps(s, s, 0x00));
	/* the signs (xorps 0x80000000): hi & 0x400 set: lane i negated when its bit (0x200 / 0x100 / 0x80) is clear, w
	 * negated; clear: lane i negated when its bit is set, w as it is */
	r0 = rr.u[0]; r1 = rr.u[1]; r2 = rr.u[2]; tw = t.u[0];
	if (hi & 0x400u) {
		if (!(hi & 0x200u)) r0 ^= PA_SG;
		if (!(hi & 0x100u)) r1 ^= PA_SG;
		if (!(hi & 0x80u)) r2 ^= PA_SG;
		tw ^= PA_SG;
	} else {
		if (hi & 0x200u) r0 ^= PA_SG;
		if (hi & 0x100u) r1 ^= PA_SG;
		if (hi & 0x80u) r2 ^= PA_SG;
	}
	BW32(out, r0); BW32(out + 4, r1); BW32(out + 8, r2); BW32(out + 12, tw);
	c->eax = hi & 0x80u; c->ecx = hi & 0x100u; c->edx = out;
#if PA_CRT_CALLS
	B_RET(sp, 0);
#else
	c->esp = sp; xrc_stdcall_ret(c, 0);
#endif
}

/* 00f98690: cdecl (x, n, cnt, base, o_prev, o_cur, o_t0, o_t1) ret: the key pair around time x in groups of 6-byte keys
 * (cnt: u8 counts per group of 32 time units, keys from base; a key's time in its group = bits 43..47 of its 8 bytes
 * (_allshr 0x2b, & 0x1f)). r = x - 32 n; tot = cnt[0] + .. + cnt[n - 1] (pairwise, as the original); from group g = n:
 * slot args x = base + 6 tot, n = tot; each key k of group g: *o_t1 = (float)k, stop at k >= r (comiss, jae; else
 * *o_t0 = k); a group without a stop: tot += cnt[g], g++, r = 0. At key j: *o_cur = key j, *o_t1 = 32 g + *o_t1; j > 0:
 * *o_prev = key j - 1, *o_t0 = 32 g + *o_t0 (eax = o_t0, edx = o_prev); j = 0: *o_prev = the last key of group g - 1,
 * *o_t0 = its time + 32 (g - 1) (eax = that time, edx = its sign word). ecx = r (popped). The x and n argument slots
 * hold the last group's start / tot (written as the original does). */
B_FN(00f98690, "keyframe groups: the keys around a time", 0) {
	u32 sp = c->esp, n = M32(sp + 8), b = M32(sp + 0xc), base = M32(sp + 0x10), op = M32(sp + 0x14), oc = M32(sp + 0x18);
	u32 o0 = M32(sp + 0x1c), o1 = M32(sp + 0x20), i = 0, s0 = 0, s1 = 0, tl = 0, tot, gi = n, j = 0, rb, p0;
	X128 x0;
	x0 = ex_subss(ex_ldss(sp + 4), ex_cvtsi2ss(c->x[1], (s32)(n << 5)));
	rb = x0.u[0];
	if ((s32)n >= 2) do { s0 += M8(b + i); s1 += M8(b + i + 1); i += 2; } while ((s32)i < (s32)(n - 1));
	if ((s32)i < (s32)n) tl = M8(b + i);
	tot = tl + (s0 + s1);
	for (;;) {
		p0 = base + tot * 6u;
		if (M8(b + gi)) {
			u32 p = p0, ecx = tot;
			for (;;) {
				u32 k;
				X128 kx;
#if PA_CRT_CALLS
				c->eax = M32(p); c->edx = M32(p + 4); c->ecx = (ecx & ~0xffu) | 0x2bu;
				BCALL(sp - 24, 0x01012100u, 0x00f98713u);
				k = c->eax & 0x1fu;
#else
				k = (M32(p + 4) >> 11) & 0x1fu;
				(void)ecx;
#endif
				ecx = o1;
				kx = ex_cvtsi2ss(x0, (s32)k);
				BW32(o1, kx.u[0]);
				if (!(ex_comiss(kx, ex_ss(rb)) & 4)) break; /* jae */
				BW32(o0, kx.u[0]);
				j++; p += 6;
				if ((s32)j >= (s32)M8(b + gi)) break;
			}
		}
		if (j != M8(b + gi)) break;
		x0 = ex_ss(0); tot += M8(b + gi); gi++; rb = 0; j = 0;
	}
	BW32(sp + 4, p0); BW32(sp + 8, tot); /* (the last group's: the original writes them each group) */
	{
		u32 pf = p0 + j * 6u;
		X128 gx = ex_cvtsi2ss(x0, (s32)(gi << 5)), t1;
		t1 = ex_addss(gx, ex_ldss(o1));
		BW32(oc, pf);
		BW32(o1, t1.u[0]);
		if (j) {
			X128 t0;
			BW32(op, pf - 6);
			t0 = ex_addss(gx, ex_ldss(o0));
			BW32(o0, t0.u[0]);
			c->eax = o0; c->edx = op;
		} else {
			u32 e = gi - 1, ii = 0, u0 = 0, u1 = 0, u2 = 0, st, last, k;
			X128 ef, kf;
			if ((s32)e >= 2) do { u0 += M8(b + ii); u1 += M8(b + ii + 1); ii += 2; } while ((s32)ii < (s32)(e - 1));
			if ((s32)ii < (s32)e) u2 = M8(b + ii);
			st = base + (u0 + u1 + u2) * 6u;
			BW32(op, st + M8(b + e) * 6u - 6);
			last = st + (M8(b + e) - 1) * 6u;
#if PA_CRT_CALLS
			c->eax = M32(last); c->edx = M32(last + 4); c->ecx = (st & ~0xffu) | 0x2bu;
			BCALL(sp - 24, 0x01012100u, 0x00f987f4u);
			k = c->eax & 0x1fu;
#else
			k = (M32(last + 4) >> 11) & 0x1fu;
			c->edx = (u32)((s32)M32(last + 4) >> 31);
#endif
			ef = ex_cvtsi2ss(t1, (s32)(e << 5));
			kf = ex_addss(ex_cvtsi2ss(gx, (s32)k), ef);
			BW32(o0, kf.u[0]);
			c->eax = k;
		}
	}
	c->ecx = rb;
	c->esp = sp; xrc_stdcall_ret(c, 0);
}

/* 00fa0700: thiscall (out, t) ret 8: a quaternion track at time t -- this->+0xc the key times (floats, (u16)
 * *this->+8 of them), this->+0x10 / 14 / 18 / 1c the x / y / z / w arrays. t below the first time: key 0 (eax = out,
 * ecx = +0x1c, edx = +0x14); not below the last: key n - 1 (eax = out, ecx = +0x1c, edx = +0x18); else the segment
 * [i, i + 1] by bisection (i = n / 2, step n / 4 halving, while >= 4; then a linear scan up or down, nothing found:
 * keys 0, 0), u = (t - t_i) / (t_i+1 - t_i): 005168e0 (thiscall key_lo->Slerp(&res, &key_hi, u), ret 0xc, /Od) on
 * stack copies of the two keys (entry esp - 0x20 / - 0x30: bisection lo / hi, scan hi / lo), res = entry esp - 0x10;
 * out = the 16 bytes at its eax (8 + 8, as the original); eax = 005168e0's eax (res), ecx = out, edx = its edx. At the
 * call ebx / esi / edi / ebp as the original has them (ebp = the times: no frame pointer); restored after. */
B_FN(00fa0700, "quaternion track: slerped key at t", BR_EAX) {
	u32 sp = c->esp, th = c->ecx, out = M32(sp + 4), tm = M32(th + 0xc), cp, n, i, h, lo = 0, hi = 0, k0, k1, r, v0, v1, v2, v3;
	u32 a0, a1, a2, a3, eb = c->ebx, es = c->esi, ed = c->edi, ep = c->ebp;
	int fnd = 0;
	X128 t = ex_ldss(sp + 8), x1, x2, u;
	/* (t before the first key, t at or after the last: the translation -- these paths were not run by the equivalence
	 * checks (11-25 coverage), so they stay the verified original) */
	if (!(ex_comiss(ex_ldss(tm), t) & 5)) { orig(c); return; } /* times[0] > t */
	cp = M32(th + 8);
	n = M16(cp);
	if (!(ex_comiss(t, ex_ldss(tm + n * 4u - 4)) & 4)) { orig(c); return; } /* t >= times[n - 1] */
	i = n >> 1; h = i;
	for (;;) {
		x1 = ex_ldss(tm + i * 4u); x2 = ex_ldss(tm + i * 4u + 4);
		h >>= 1;
		if (!(ex_comiss(t, x1) & 4) && !(ex_comiss(x2, t) & 5)) { fnd = 1; break; } /* t_i <= t < t_i+1 */
		if (h < 4) break;
		if (!(ex_comiss(x1, t) & 5)) i -= h; else i += h;
	}
	a0 = M32(th + 0x10); a1 = M32(th + 0x14); a2 = M32(th + 0x18); a3 = M32(th + 0x1c);
	if (fnd) {
		k0 = sp - 0x20; k1 = sp - 0x30;
		M32(k0) = M32(a0 + i * 4u); M32(k0 + 4) = M32(a1 + i * 4u); M32(k0 + 8) = M32(a2 + i * 4u); M32(k0 + 12) = M32(a3 + i * 4u);
		M32(k1) = M32(a0 + i * 4u + 4); M32(k1 + 4) = M32(a1 + i * 4u + 4); M32(k1 + 8) = M32(a2 + i * 4u + 4); M32(k1 + 12) = M32(a3 + i * 4u + 4);
		u = ex_divss(ex_subss(t, x1), ex_subss(x2, x1));
		c->eax = sp - 0x10; c->ecx = k0; c->edx = k1;
		c->ebx = n; c->esi = a1 + i * 4u; c->edi = a2 + i * 4u;
	} else {
		/* linear scan from i: up when t_i <= t (or unordered), down when t_i > t */
		X128 xl, xh;
		if (ex_comiss(x1, t) & 5) {
			u32 lim = n - 1;
			for (; i < lim; i++)
				if (!(ex_comiss(t, ex_ldss(tm + i * 4u)) & 4) && !(ex_comiss(ex_ldss(tm + i * 4u + 4), t) & 5)) { lo = i; hi = i + 1; break; }
		} else {
			for (; i >= 1; i--)
				if (!(ex_comiss(t, ex_ldss(tm + i * 4u - 4)) & 4) && !(ex_comiss(ex_ldss(tm + i * 4u), t) & 5)) { lo = i - 1; hi = i; break; }
		}
		xl = ex_ldss(tm + lo * 4u); xh = ex_ldss(tm + hi * 4u);
		k0 = sp - 0x30; k1 = sp - 0x20;
		M32(k0) = M32(a0 + lo * 4u); M32(k0 + 4) = M32(a1 + lo * 4u); M32(k0 + 8) = M32(a2 + lo * 4u); M32(k0 + 12) = M32(a3 + lo * 4u);
		M32(k1) = M32(a0 + hi * 4u); M32(k1 + 4) = M32(a1 + hi * 4u); M32(k1 + 8) = M32(a2 + hi * 4u); M32(k1 + 12) = M32(a3 + hi * 4u);
		u = ex_divss(ex_subss(t, xl), ex_subss(xh, xl));
		c->eax = a0; c->ecx = k0; c->edx = sp - 0x10;
		c->ebx = a2; c->esi = hi; c->edi = a1;
	}
	M32(sp - 0x44) = u.u[0]; M32(sp - 0x48) = k1; M32(sp - 0x4c) = sp - 0x10;
	c->ebp = tm;
	BCALL(sp - 0x50, 0x005168e0u, 0x00fa09b3u);
	r = c->eax;
	v0 = M32(r); v1 = M32(r + 4);
	BW32(out, v0); BW32(out + 4, v1);
	v2 = M32(r + 8); v3 = M32(r + 12);
	BW32(out + 8, v2); BW32(out + 12, v3);
	c->ecx = out;
	c->ebx = eb; c->esi = es; c->edi = ed; c->ebp = ep;
	c->esp = sp; xrc_stdcall_ret(c, 2);
}

/* 00cd8240: cdecl (out, p, x) ret: polynomial curves at x -- p: u8 degree d, u8 count m, then m rows of d + 1 floats;
 * d and m not 0: out[j] = (c1 * x + c0) + out[m + 1 + j], then for k = 2 .. d: out[j] = c_k * x^k (x * x * .. left to
 * right) * 0.5 ([0109b130]) + out[j] (stored each step, as the original); the p argument slot = 4 m + 4 (its spill);
 * eax = 4 m + 4, ecx = p[0], edx = p[1]. d = 0 / m = 0: nothing (al = d, ecx = m, edx = out + 4 m + 4). */
B_FN(00cd8240, "curve: polynomial rows at x", 0) {
	u32 sp = c->esp, out = M32(sp + 4), p = M32(sp + 8), d = M8(p), m = M8(p + 1), off = m * 4u + 4, j, dc, mc;
	X128 x, hk = ex_ss(0x3f000000u);
	if (!d || !m) {
		c->eax = (c->eax & ~0xffu) | d; c->ecx = m; c->edx = out + off;
		c->esp = sp; xrc_stdcall_ret(c, 0);
		return;
	}
	x = ex_ldss(sp + 0xc);
	BW32(sp + 8, off);
	dc = d;
	for (j = 0;; ) {
		u32 row = p + (dc + 1) * j * 4u, oj = out + j * 4u;
		X128 v = ex_mulss(ex_ldss(row + 8), x);
		v = ex_addss(v, ex_ldss(row + 4));
		v = ex_addss(v, ex_ldss(oj + off));
		BW32(oj, v.u[0]);
		dc = M8(p);
		if ((s32)dc >= 2) {
			u32 e = 1, cf = row + 0xc, k;
			do {
				X128 pw = x, w;
				for (k = 0; k < e; k++) pw = ex_mulss(pw, x);
				w = ex_mulss(ex_ldss(cf), pw);
				w = ex_mulss(w, hk);
				w = ex_addss(w, ex_ldss(oj));
				BW32(oj, w.u[0]);
				dc = M8(p);
				e++; cf += 4;
			} while ((s32)(e + 1) <= (s32)dc);
		}
		mc = M8(p + 1);
		if ((s32)++j >= (s32)mc) break;
	}
	c->eax = off; c->ecx = dc; c->edx = mc;
	c->esp = sp; xrc_stdcall_ret(c, 0);
}

/* 00a39b20: cdecl (out, v) ret: 4 floats to bytes -- each: 0 > f: 0, f > 255: 0xff, else (u8)cvttss2si(f) (NaN: 0);
 * out[0] = bswap(b(v[3]) << 24 | b(v[0]) << 16 | b(v[1]) << 8 | b(v[2])), out[1..3] = the 12 bytes its frame leaves
 * unwritten (entry esp - 0x14 .. - 9, whatever is there); eax = ecx = out, edx = out[2] */
static inline u32 pa_u8f(X128 f) {
	if (!(ex_comiss(ex_ss(0), f) & 5)) return 0;               /* [025a3c20] 0.0 > f */
	if (!(ex_comiss(f, ex_ss(0x437f0000u)) & 5)) return 0xffu; /* f > [0109f7a4] 255.0 */
	return (u32)ex_cvttss2si(f) & 0xffu;
}
B_FN(00a39b20, "color: 4 floats to a byte-swapped dword", BR_EAX) {
	u32 sp = c->esp, out = M32(sp + 4), v = M32(sp + 8), b3, b0, b1, b2, g0, g1, g2;
	b3 = pa_u8f(ex_ldss(v + 0xc)); b0 = pa_u8f(ex_ldss(v)); b1 = pa_u8f(ex_ldss(v + 4)); b2 = pa_u8f(ex_ldss(v + 8));
	BW32(out, bswap32(b3 << 24 | b0 << 16 | b1 << 8 | b2));
	g0 = M32(sp - 0x14); BW32(out + 4, g0);
	g1 = M32(sp - 0x10); BW32(out + 8, g1);
	g2 = M32(sp - 0xc); BW32(out + 12, g2);
	c->eax = out; c->ecx = out; c->edx = g1;
	c->esp = sp; xrc_stdcall_ret(c, 0);
}
/* ---------------------------------------------------------------- battle batch 2, part B (00f99680, 00d7e170, 00db7310, 00462d70) */
/* part-local helpers: the SSE2 integer <-> float conversions (inline asm: the compiler cannot fold cvtdq2ps(cvttps2dq x)
 * into a truncation), a lane broadcast, the bit select of the andps / andnps / orps idiom (bitwise: plain C) */
static inline X128 pb_cvtdq2ps(X128 a) { X128 r; __asm__("cvtdq2ps %1, %0" : "=x"(r.v) : "x"(a.v)); return r; }
static inline X128 pb_cvttps2dq(X128 a) { X128 r; __asm__("cvttps2dq %1, %0" : "=x"(r.v) : "x"(a.v)); return r; }
/* (bit operations as SSE2 integer instructions -- exact; lane-by-lane C goes through the stack at -O1: store-forwarding
 * stalls, 11-25: 00db7310 5 x slower than its translation) */
static inline X128 pb_set1(u32 b) { X128 r; r.vi = _mm_set1_epi32((int)b); return r; }
static inline X128 pb_lane(X128 a, int k) { return pb_set1(a.u[k]); }
static inline X128 pb_sel(X128 m, X128 a, X128 b) { /* (m & a) | (~m & b) */
	X128 r;
	r.vi = _mm_or_si128(_mm_and_si128(m.vi, a.vi), _mm_andnot_si128(m.vi, b.vi));
	return r;
}
static inline X128 pb_andn(X128 m, X128 b) { X128 r; r.vi = _mm_andnot_si128(m.vi, b.vi); return r; }
static inline X128 pb_and(X128 a, X128 b) { X128 r; r.vi = _mm_and_si128(a.vi, b.vi); return r; }
static inline X128 pb_or(X128 a, X128 b) { X128 r; r.vi = _mm_or_si128(a.vi, b.vi); return r; }

/* 00f99680 (the odd-power polynomial): x2 = v * v, x3 = x2 * v; ((((x3 * k630 + v) + x5 * k9c0) + x7 * k9b0) + x9 * k620)
 * with x5 = x3 * x2, x7 = x5 * x2, x9 = x7 * x2 (each block's order) */
static inline X128 pb_poly(X128 v, X128 k630, X128 k9c0, X128 k9b0, X128 k620) {
	X128 x2 = ex_mulps(v, v), x3 = ex_mulps(x2, v), acc, x5, x7, x9;
	acc = ex_mulps(x3, k630);
	acc = ex_addps(acc, v);
	x5 = ex_mulps(x3, x2);
	acc = ex_addps(acc, ex_mulps(x5, k9c0));
	x7 = ex_mulps(x5, x2);
	acc = ex_addps(acc, ex_mulps(x7, k9b0));
	x9 = ex_mulps(x7, x2);
	return ex_addps(acc, ex_mulps(x9, k620));
}
/* 00f99680: the sign mask factor of a key's flags word e: ((cvtdq2ps((e & k9a0) == 0 ? -1 : 0) * k9d0) + k6) * k600 */
static inline X128 pb_sgn(u32 e, X128 k9a0, X128 k9d0, X128 k6, X128 k600) {
	X128 m;
	m.vi = _mm_cmpeq_epi32(_mm_and_si128(_mm_set1_epi32((int)e), k9a0.vi), _mm_setzero_si128());
	return ex_mulps(ex_addps(ex_mulps(pb_cvtdq2ps(m), k9d0), k6), k600);
}

/* 00f99680: cdecl (out0, out1, key0, key1, (a 5th, unused)), ret: two packed 6-byte rotation keys (dword +0, e = the
 * low word of +4) decoded to quaternions (Vec4). As qwords {key1, key0}: a = (q >> 17) & 0x3fffff, b = q & 0x1ffff to
 * float, x5 = {a1, a0, b1, b0} * {1, 1, 2^-17', 2^-17'} (0265e650), its lanes 2, 3 then 1 - b^2 (010c6de0: 1);
 * sqrtps, T = its integer part (cvttps2dq, cvtdq2ps); X2 = 1 - T * [0265e9e0], X1 = ((x5 - T T) / max(2 T, FLT_MIN))
 * * (T bits 0 ? 0 : 1). Per key (key0: lanes 1, key1: lanes 0): {X1, X2, X1, X2} * [0286d910] + [0286d900] (+-pi/2)
 * through the sine polynomial (pb_poly), lane 3 * -1 (0109f7c0); {s3 s0, s1, s3 s2, s3} * {1, 1, -1, 1} * sqrt lane
 * 3 / 2 * the signs of e's bits 0x200 / 0x100 / 0x80 (pb_sgn); out = {its x, y, z, x5 lane 3 / 2} (16 bytes, out0 then
 * out1); then e bit 0x400: out *= -1 (re-read: out0 then out1, so out0 == out1 works). The statics 0286d920 / 910 /
 * 900 made on first use (flag 0286d930 bits 0..2). Constants read from guest memory as the original (the translation
 * folds the .rdata ones). eax = ecx = key1's e & 0x400, edx = key0's (the one caller, 00f9a595, uses none) */
B_FN(00f99680, "packed rotation keys: decode two", 0) {
	u32 sp = c->esp, o0 = M32(sp + 4), o1 = M32(sp + 8), s0 = M32(sp + 12), s1 = M32(sp + 16);
	u32 lo0 = M32(s0), e0 = M32(s0 + 4) & 0xffffu, lo1 = M32(s1), e1 = M32(s1 + 4) & 0xffffu, fl = M32(0x0286d930u);
	u32 k9c = M32(0x010a1f9cu), k640 = M32(0x0265e640u), k7c0 = M32(0x0109f7c0u);
	u64 q0 = (u64)e1 << 32 | lo1, q1 = (u64)e0 << 32 | lo0;
	X128 k6 = ex_ld(0x010c6de0u), ka00 = ex_ld(0x0265ea00u), k660 = ex_ld(0x0265e660u), k650 = ex_ld(0x0265e650u);
	X128 k9f0 = ex_ld(0x0265e9f0u), k9e0 = ex_ld(0x0265e9e0u), k9d0 = ex_ld(0x0265e9d0u), k9a0 = ex_ld(0x0265e9a0u);
	X128 k630 = ex_ld(0x0265e630u), k9c0 = ex_ld(0x0265e9c0u), k9b0 = ex_ld(0x0265e9b0u), k620 = ex_ld(0x0265e620u);
	X128 k610 = ex_ld(0x0265e610u), k600 = ex_ld(0x0265e600u), kk = pb_set1(k7c0);
	X128 a, b, x0, x1, x2, x3, x4, x5, T, m, X1, X2, v, P, R;
	if (!(fl & 1)) {
		fl |= 1; BW32(0x0286d930u, fl);
		BW32(0x0286d920u, k9c); BW32(0x0286d924u, k9c); BW32(0x0286d928u, k9c); BW32(0x0286d92cu, k9c);
	}
	a.vi = _mm_and_si128(_mm_set_epi32((int)(u32)(q1 >> 49), (int)(u32)(q1 >> 17), (int)(u32)(q0 >> 49), (int)(u32)(q0 >> 17)), ka00.vi);
	b.vi = _mm_and_si128(_mm_set_epi32((int)e0, (int)lo0, (int)e1, (int)lo1), k660.vi);
	x5 = ex_shufps(pb_cvtdq2ps(a), pb_cvtdq2ps(b), 0x88);
	x5 = ex_mulps(x5, k650);
	x0 = ex_shufps(x5, x5, 0xee);
	x0 = ex_mulps(x0, x0);
	x1 = ex_subps(k6, x0);
	x5 = ex_shufps(x5, x1, 0x44);
	x1 = ex_mulps(x5, x5);
	x2 = ex_subps(k6, x1);
	x0 = ex_shufps(x5, x2, 0xe4);
	x4 = ex_sqrtps(x0);
	T = pb_cvtdq2ps(pb_cvttps2dq(x4));
	X2 = ex_subps(k6, ex_mulps(T, k9e0));
	x1 = ex_subps(x5, ex_mulps(T, T));
	x3 = ex_maxps(ex_mulps(T, k9d0), k9f0);
	m.vi = _mm_cmpeq_epi32(T.vi, _mm_setzero_si128()); /* (pcmpeqd: the bits) */
	m = ex_addps(pb_cvtdq2ps(m), k6);
	X1 = ex_mulps(ex_divps(x1, x3), m);
	if (!(fl & 2)) {
		fl |= 2; BW32(0x0286d930u, fl);
		BW32(0x0286d910u, k640); BW32(0x0286d914u, k9c); BW32(0x0286d918u, k640); BW32(0x0286d91cu, k9c);
	}
	if (!(fl & 4)) {
		fl |= 4; BW32(0x0286d900u, k9c); BW32(0x0286d930u, fl);
		BW32(0x0286d904u, 0); BW32(0x0286d908u, 0); BW32(0x0286d90cu, k640);
	}
	/* key0 -> out0: lanes 1 */
	v.u[0] = X1.u[1]; v.u[1] = X2.u[1]; v.u[2] = X1.u[1]; v.u[3] = X2.u[1];
	v = ex_addps(ex_mulps(v, ex_ld(0x0286d910u)), ex_ld(0x0286d900u));
	P = pb_poly(v, k630, k9c0, k9b0, k620);
	P.u[3] = ex_mulss(ex_ss(P.u[3]), ex_ss(k7c0)).u[0];
	x2.u[0] = P.u[3]; x2.u[1] = k6.u[2]; x2.u[2] = P.u[3]; x2.u[3] = k6.u[3];
	x2 = ex_mulps(ex_mulps(x2, P), k610);
	x0 = pb_sgn(e0, k9a0, k9d0, k6, k600);
	x2 = ex_mulps(x2, pb_lane(x4, 3));
	x0 = ex_mulps(x0, x2);
	R = x0; R.u[3] = x5.u[3];
	BW128(o0, R);
	/* key1 -> out1: lanes 0 */
	v.u[0] = X1.u[0]; v.u[1] = X2.u[0]; v.u[2] = X1.u[0]; v.u[3] = X2.u[0];
	v = ex_addps(ex_mulps(v, ex_ld(0x0286d910u)), ex_ld(0x0286d900u));
	P = pb_poly(v, k630, k9c0, k9b0, k620);
	P.u[3] = ex_mulss(ex_ss(P.u[3]), ex_ss(k7c0)).u[0];
	x0 = pb_sgn(e1, k9a0, k9d0, k6, k600);
	x3.u[0] = P.u[3]; x3.u[1] = k6.u[2]; x3.u[2] = P.u[3]; x3.u[3] = k6.u[3];
	x3 = ex_mulps(ex_mulps(x3, P), k610);
	x3 = ex_mulps(x3, pb_lane(x4, 2));
	x0 = ex_mulps(x0, x3);
	R = x0; R.u[3] = x5.u[2];
	BW128(o1, R);
	if (e0 & 0x400u) BW128(o0, ex_mulps(ex_ld(o0), kk));
	if (e1 & 0x400u) BW128(o1, ex_mulps(ex_ld(o1), kk));
	c->eax = c->ecx = e1 & 0x400u; c->edx = e0 & 0x400u;
	c->esp = sp;
	xrc_stdcall_ret(c, 0);
}

/* 00d7e170: thiscall (out, pos, stride, tris, v), ret 0x14 (this -> a u16 triangle index): the triangle tris + 8 *
 * this->+0 (three u16 vertex indices i0..i2) as 3 Vec4 rows R_j = {pos[i_j * stride + 0..2], k} (k the float at
 * 010aa950; pos floats, stride the low word of the 3rd arg); its winding flipped by the signs of v.x, v.y, v.z (v
 * aligned 16 bytes): s = (0 <= v.x) ? k : 0 - k, then s = (0 <= v.y) ? s : 0 - s, (0 <= v.z) likewise (cmpleps and
 * andps / andnps / orps selects, all 4 lanes); with 0 <= s: out = {R0, R1, R2}, else {R2, R1, R0} (48 bytes, +0, +0x10,
 * +0x20); eax = out, ecx = pos, edx = i0 * stride */
B_FN(00d7e170, "triangle rows, winding by sign", BR_EAX) {
	u32 sp = c->esp, th = c->ecx, out = M32(sp + 4), pos = M32(sp + 8), st = M16(sp + 12), tris = M32(sp + 16), vp = M32(sp + 20);
	u32 kb = M32(0x010aa950u), e, i0, i1, i2;
	X128 z = pb_set1(0), K4 = pb_set1(kb), V = ex_ld(vp), N4, S, NS, M, R0, R1, R2;
	N4 = ex_subps(z, K4);
	M = ex_cmpps(z, pb_lane(V, 0), 2);
	S = pb_sel(M, K4, N4);
	NS = ex_subps(z, S);
	M = ex_cmpps(z, pb_lane(V, 1), 2);
	S = pb_sel(M, S, NS);
	NS = ex_subps(z, S);
	M = ex_cmpps(z, pb_lane(V, 2), 2);
	S = pb_sel(M, S, NS);
	e = tris + (u32)M16(th) * 8u;
	i0 = (u32)M16(e) * st; i1 = (u32)M16(e + 2) * st; i2 = (u32)M16(e + 4) * st;
	R0.u[0] = M32(pos + i0 * 4u); R0.u[1] = M32(pos + i0 * 4u + 4); R0.u[2] = M32(pos + i0 * 4u + 8); R0.u[3] = kb;
	R1.u[0] = M32(pos + i1 * 4u); R1.u[1] = M32(pos + i1 * 4u + 4); R1.u[2] = M32(pos + i1 * 4u + 8); R1.u[3] = kb;
	R2.u[0] = M32(pos + i2 * 4u); R2.u[1] = M32(pos + i2 * 4u + 4); R2.u[2] = M32(pos + i2 * 4u + 8); R2.u[3] = kb;
	M = ex_cmpps(z, S, 2);
	BW128(out, pb_sel(M, R0, R2));
	BW128(out + 0x10, R1);
	BW128(out + 0x20, pb_sel(M, R2, R0));
	c->eax = out; c->ecx = pos; c->edx = i0;
	c->esp = sp;
	xrc_stdcall_ret(c, 5);
}

/* 00db7310: cdecl (out, A, B), ret: the closest points of the segments A = {a0, a1} and B = {b0, b1} (2 aligned Vec4
 * each): dA = a1 - a0, dB = b1 - b0, r = a0 - b0; D = {a, e, c, f} = {dA.dA, dB.dB, r.dA, r.dB} (xyz, (y + x) + z),
 * b = dA.dB (4 lanes, each its own sum order); 1 / x = rcpps (exact, as the translation) + a Newton step r + r (H - r x)
 * (H 010aa950); s = (b f - c e) / (a e - b b), 0 when |a e - b b| <= [010a523c], clamped to [0, C] (C 010a6fa0);
 * tn = s b + f, t = tn / e; then {s, t} = tn < 0 ? {-c / a, 0} : tn > e ? {(b - c) / a, C} : {s, t} (cmpleps masks),
 * s = -c / a where e < E, t = f / e where a < E, s = 0 where a < E, t = 0 where e < E (E 02613d58: degenerate
 * segments), clamped again; out = {a0 + s dA, b0 + t dB} (32 bytes). The /GS cookie store and check skipped. eax =
 * out, ecx = the cookie value (as the check returns), edx = B */
B_FN(00db7310, "segment-segment closest points", BR_EAX) {
	u32 sp = c->esp, out = M32(sp + 4), pa = M32(sp + 8), pb = M32(sp + 12);
	X128 z = pb_set1(0), a0 = ex_ld(pa), b0 = ex_ld(pb), a1 = ex_ld(pa + 16), b1 = ex_ld(pb + 16);
	X128 dA, dB, r, dA2, dB2, rdA, rdB, p, x0, x1, x2, x3, x4, x5, x6, x7, D, B4, AE, rAE, DmE, F4, H4, E4, C4, G4, Q4;
	X128 den, rD, Z, S4, tn, rE, ST, TC0, TC1, M5, M6;
	dB = ex_subps(b1, b0);
	dA = ex_subps(a1, a0);
	dB2 = ex_mulps(dB, dB);
	r = ex_subps(a0, b0);
	dA2 = ex_mulps(dA, dA);
	rdB = ex_mulps(r, dB);
	p = ex_mulps(dB, dA);
	rdA = ex_mulps(r, dA);
	x5 = ex_shufps(dA2, dB2, 0x44);
	x3 = ex_shufps(dA2, dB2, 0xee);
	x1 = ex_addps(ex_shufps(p, p, 0x4e), p);
	x4 = ex_shufps(rdA, rdB, 0xee);
	x6 = ex_shufps(rdA, rdB, 0x44);
	x0 = ex_shufps(x5, x6, 0x88);
	x5 = ex_shufps(x5, x6, 0xdd);
	C4 = pb_set1(M32(0x010a6fa0u));
	x3 = ex_shufps(x3, x4, 0x88);
	E4 = pb_set1(M32(0x02613d58u));
	G4 = pb_set1(M32(0x010a523cu));
	D = ex_addps(ex_addps(x5, x0), x3);
	B4 = ex_addps(ex_shufps(x1, x1, 0x39), x1);
	AE = ex_shufps(D, D, 0x44);
	rAE = ex_rcpps(AE);
	DmE = ex_subps(D, E4);
	x5 = ex_subps(ex_shufps(z, B4, 0xe4), ex_shufps(D, D, 0xaa));
	F4 = ex_shufps(D, D, 0xff);
	x5 = ex_shufps(x5, F4, 0xd8);
	H4 = pb_set1(M32(0x010aa950u));
	x7 = ex_mulps(rAE, AE);
	x3 = ex_shufps(D, D, 0x55);
	x4 = ex_mulps(ex_subps(H4, x7), rAE);
	x5 = ex_shufps(x5, x5, 0xd8);
	x4 = ex_addps(x4, rAE);
	Q4 = ex_mulps(x4, x5);
	den = ex_subps(ex_mulps(ex_shufps(D, D, 0x00), x3), ex_mulps(B4, B4));
	x7 = ex_shufps(ex_subps(z, x3), B4, 0xd8);
	x6 = ex_shufps(D, D, 0xee);
	rD = ex_rcpps(den);
	x7 = ex_shufps(x7, x7, 0xd8);
	x6 = ex_mulps(x6, x7);
	x0 = ex_maxps(ex_subps(z, den), den);
	Z = ex_cmpps(z, ex_subps(G4, x0), 2);
	x7 = ex_mulps(ex_subps(H4, ex_mulps(rD, den)), rD);
	x7 = ex_addps(x7, rD);
	x7 = ex_mulps(x7, ex_addps(ex_shufps(x6, x6, 0x39), x6));
	x0 = pb_or(pb_andn(Z, x7), pb_and(Z, z));
	S4 = ex_maxps(ex_minps(x0, C4), z);
	tn = ex_addps(ex_mulps(pb_lane(S4, 0), B4), F4);
	rE = ex_rcpps(x3);
	x2 = ex_mulps(ex_subps(H4, ex_mulps(rE, x3)), rE);
	x2 = ex_addps(x2, rE);
	x2 = ex_mulps(x2, tn);
	ST = ex_shufps(S4, x2, 0xd8);
	TC1 = ex_cmpps(z, ex_subps(x3, tn), 2);
	TC0 = ex_cmpps(z, tn, 2);
	M5 = ex_cmpps(z, ex_shufps(DmE, DmE, 0xb1), 2);
	x2 = ex_shufps(ex_shufps(Q4, Q4, 0x4e), C4, 0xd8);
	M6 = ex_cmpps(z, DmE, 2);
	x3 = ex_shufps(Q4, z, 0xd8);
	x2 = ex_shufps(x2, x2, 0xd8);
	x0 = pb_andn(TC1, x2);
	x7 = pb_and(ex_shufps(ST, ST, 0xd8), TC1);
	x0 = pb_or(x0, x7);
	x0 = pb_and(x0, TC0);
	x3 = ex_shufps(x3, x3, 0xd8);
	x0 = pb_or(x0, pb_andn(TC0, x3));
	x0 = pb_and(x0, M5);
	x0 = pb_or(x0, pb_andn(M5, Q4));
	x0 = pb_and(x0, M6);
	x0 = pb_or(x0, pb_andn(M6, z));
	x0 = ex_maxps(ex_minps(x0, C4), z);
	BW128(out, ex_addps(ex_mulps(pb_lane(x0, 0), dA), a0));
	BW128(out + 0x10, ex_addps(ex_mulps(pb_lane(x0, 1), dB), b0));
	c->eax = out; c->ecx = M32(0x026e5a84u); c->edx = pb;
	c->esp = sp;
	xrc_stdcall_ret(c, 0);
}

/* 00462d70: cdecl Matrix44 *(out), ret: out = the identity from its 4 constant rows (010a6fa0, 010a6fb0, 010a6fc0,
 * 0109f810; bit copies through movss / mov); eax = out, ecx = out + 0x30, edx = row 3's z, xmm0 = {010a6fac's float,
 * 0, 0, 0} (the last movss load) */
B_FN(00462d70, "Matrix44 identity", BR_EAX) {
	u32 sp = c->esp, out = M32(sp + 4), r[16];
	int k;
	for (k = 0; k < 4; k++) {
		r[k] = M32(0x010a6fa0u + 4u * (u32)k); r[4 + k] = M32(0x010a6fb0u + 4u * (u32)k);
		r[8 + k] = M32(0x010a6fc0u + 4u * (u32)k); r[12 + k] = M32(0x0109f810u + 4u * (u32)k);
	}
	for (k = 0; k < 16; k++) BW32(out + 4u * (u32)k, r[k]);
	c->x[0] = ex_ss(r[3]);
	c->eax = out; c->ecx = out + 0x30u; c->edx = r[14];
	c->esp = sp;
	xrc_stdcall_ret(c, 0);
}

/* ---- battle batch 2, part C ---- */
/* SSE2 integer ops of 0045e580 (cvtps2dq: the MXCSR rounding, as the translation's), and lane helpers */
#if defined(__x86_64__)
static inline X128 pc_cvtps2dq(X128 a) { X128 r; __asm__("cvtps2dq %1, %0" : "=x"(r.v) : "x"(a.v)); return r; }
static inline X128 pc_cvtdq2ps(X128 a) { X128 r; __asm__("cvtdq2ps %1, %0" : "=x"(r.v) : "x"(a.v)); return r; }
#endif
static inline X128 pc_pand(X128 a, X128 b) { X128 r; r.vi = _mm_and_si128(a.vi, b.vi); return r; } /* (SSE2: exact, no stack round trip) */
static inline X128 pc_paddd(X128 a, X128 b) { X128 r; r.vi = _mm_add_epi32(a.vi, b.vi); return r; }
/* movss x, y + shufps x, x, 0: y's lane 0 in all four */
static inline X128 pc_bc(u32 bits) { X128 r; r.u[0] = r.u[1] = r.u[2] = r.u[3] = bits; return r; }
static inline unsigned pc_movmskps(X128 a) { return (a.u[0] >> 31) | (a.u[1] >> 31) << 1 | (a.u[2] >> 31) << 2 | (a.u[3] >> 31) << 3; }
static inline X128 pc_zero(void) { return ex_ss(0); }

/* 0045e580: cdecl (__m128 x in xmm0, out_a, out_b), ret: the SSE sincos (Cephes style) -- j = cvtps2dq(x * 2/pi),
 * z = (x - y * pi/2 hi) - y * pi/2 lo (y = float(j)); p = the cos polynomial of z*z, s = the sin one * z^3 + z; per
 * lane out_a / out_b = s or p by j & 1 / (j & 3) + 1 & 1, negated (0 - v) by j & 2 / (j & 3) + 1 & 2 (the constants
 * read from .rdata as the original does); writes out_a, out_b (16 bytes each); eax = out_a, ecx = the cookie, xmm0 =
 * xmm1 = out_b's value */
B_FN(0045e580, "sincos_ps", 0) {
	u32 sp = c->esp, o1 = M32(sp + 4), o2 = M32(sp + 8);
	X128 x = c->x[0], k1 = ex_ld(0x02663a00u), kc = ex_ld(0x02617fb0u), km = ex_ld(0x02631710u), t, j, jb, jc, y, z, z2, z3, p, q, sn, m1, m2, m3, m4, r1, r2;
	t = ex_mulps(x, ex_ld(0x011a5da0u));
	j = pc_cvtps2dq(t);
	jb = pc_pand(j, ex_ld(0x02610ae0u));
	jc = pc_paddd(kc, jb);
	y = pc_cvtdq2ps(j);
	z = ex_subps(x, ex_mulps(y, ex_ld(0x02666ef0u)));
	z = ex_subps(z, ex_mulps(y, ex_ld(0x010c9560u)));
	z2 = ex_mulps(z, z);
	z3 = ex_mulps(z2, z);
	p = ex_addps(ex_mulps(ex_ld(0x025c4310u), z2), ex_ld(0x02661970u));
	q = ex_addps(ex_mulps(ex_ld(0x025c4890u), z2), ex_ld(0x02666bd0u));
	p = ex_addps(ex_mulps(p, z2), ex_ld(0x026134e0u));
	q = ex_addps(ex_mulps(q, z2), ex_ld(0x010c5480u));
	p = ex_addps(ex_mulps(p, z2), ex_ld(0x010c6de0u));
	sn = ex_addps(ex_mulps(q, z3), z);
	m1 = ex_cmpps(pc_cvtdq2ps(pc_pand(jb, kc)), k1, 0);
	m2 = ex_cmpps(pc_cvtdq2ps(pc_pand(jc, kc)), k1, 0);
	r1 = ex_orps(ex_andnps(m1, p), ex_andps(m1, sn));
	r2 = ex_orps(ex_andnps(m2, p), ex_andps(m2, sn));
	m3 = ex_cmpps(pc_cvtdq2ps(pc_pand(jb, km)), k1, 0);
	m4 = ex_cmpps(pc_cvtdq2ps(pc_pand(jc, km)), k1, 0);
	r1 = ex_orps(ex_andnps(m3, ex_subps(k1, r1)), ex_andps(m3, r1));
	r2 = ex_orps(ex_andnps(m4, ex_subps(k1, r2)), ex_andps(m4, r2));
	BW128(o1, r1);
	BW128(o2, r2);
	c->x[0] = r2; c->x[1] = r2;
	c->eax = o1; c->ecx = M32(0x026e5a84u);
	c->esp = sp; xrc_stdcall_ret(c, 0);
}

/* 00462880: cdecl Matrix44 *(out, const Vec4 *e), ret: the rotation from the angles e.xyz (w masked off: andps with
 * cmpeqps(02663a00 zero, 0109f7b0 {0,0,0,1})) -- 0045e580(e & mask, &S, &C) into the aligned frame (S at A+0x200, C
 * at A+0x210, A = (ebp & ~15) - 0x280); nS = 0 - S, u1 = unpckhps(C, S), u2 = unpckhps(nS, C) & mask, w1 =
 * shufps(C, nS, 0x15), w2 = shufps(S, C, 0x15), Sx / Cx = S.x / C.x splat; rows r0 = u1 * w1, r1 = u2 * Cx + (u1 *
 * w2) * Sx, r2 = (u1 * w2) * Cx - u2 * Sx, r3 = 0109f7b0 (locals at A, A+0x10, A+0x20, A+0x30), then
 * out->00462cd0(r0, r1, r2, r3) (thiscall, ret 0x10: copies them to out); eax = out, ecx = the cookie */
B_FN(00462880, "Matrix44 from angles (sincos)", BR_EAX) {
	u32 sp = c->esp, out = M32(sp + 4), e = M32(sp + 8), bp = sp - 4, A = (bp & ~15u) - 0x280u, ck = M32(0x026e5a84u), k;
	X128 k1 = ex_ld(0x02663a00u), mk, v, vm, x2, S, C, nS, u1, u2, w1, w2, sx, cx, a1, r0, r1, r2;
	mk = ex_cmpps(ex_ld(0x0109f7b0u), k1, 0);
	v.u[0] = M32(e); v.u[1] = M32(e + 4); v.u[2] = M32(e + 8); v.u[3] = M32(e + 12);
	x2.u[0] = v.u[1]; x2.u[1] = v.u[3]; x2.u[2] = 0; x2.u[3] = 0; /* (unpcklps of the movss loads e.y, e.w) */
	vm = ex_andps(v, mk);
	B_ENTER(bp);
	M32(A - 4) = A + 0x210u; M32(A - 8) = A + 0x200u;
	c->x[0] = vm; c->x[1] = vm; c->x[2] = x2; c->x[3] = v;
	c->eax = e; c->ecx = A + 0x210u; c->edx = A + 0x200u;
	BCALL(A - 0xcu, 0x0045e580u, 0x0046294cu);
	S = ex_ld(A + 0x200u); C = ex_ld(A + 0x210u);
	nS = ex_subps(k1, S);
	u1 = ex_unpckhps(C, S);
	u2 = ex_andps(ex_unpckhps(nS, C), mk);
	w1 = ex_shufps(C, nS, 0x15);
	w2 = ex_shufps(S, C, 0x15);
	sx = ex_shufps(S, S, 0x00);
	cx = ex_shufps(C, C, 0x00);
	a1 = ex_mulps(u1, w2);
	r2 = ex_subps(ex_mulps(a1, cx), ex_mulps(u2, sx));
	r1 = ex_addps(ex_mulps(u2, cx), ex_mulps(a1, sx));
	r0 = ex_mulps(u1, w1);
	for (k = 0; k < 16; k += 4) M32(A + 0x30u + k) = M32(0x0109f7b0u + k);
	XST(A + 0x20u, r2); XST(A + 0x10u, r1); XST(A, r0);
	M32(A - 4) = A + 0x30u; M32(A - 8) = A + 0x20u; M32(A - 0xcu) = A + 0x10u; M32(A - 0x10u) = A;
	c->x[0] = ex_ss(r0.u[3]); c->x[1] = r0;
	c->eax = A; c->ecx = out; c->edx = A + 0x10u;
	BCALL(A - 0x14u, 0x00462cd0u, 0x00462cafu);
	c->eax = out; c->ecx = ck;
	B_RET(sp, 0);
}

/* 00492460: cdecl Matrix44 *(out, const Vec4 *t), ret: the translation matrix -- rows 010a6fa0, 010a6fb0, 010a6fc0
 * ({1,0,0,0}, {0,1,0,0}, {0,0,1,0}) and {t.x, t.y, t.z, [010aa950] (1.0)} (bit copies; t read before out is
 * written); eax = out, ecx = out + 0x30, edx = t.z, xmm0 = [010a6fac] (movss) */
B_FN(00492460, "Matrix44 translation", BR_EAX) {
	u32 sp = c->esp, out = M32(sp + 4), t = M32(sp + 8), t0 = M32(t), t1 = M32(t + 4), t2 = M32(t + 8), k;
	for (k = 0; k < 0x30u; k += 4) BW32(out + k, M32(0x010a6fa0u + k));
	BW32(out + 0x30u, t0); BW32(out + 0x34u, t1); BW32(out + 0x38u, t2); BW32(out + 0x3cu, M32(0x010aa950u));
	c->x[0] = ex_ldss(0x010a6facu);
	c->eax = out; c->ecx = out + 0x30u; c->edx = t2;
	c->esp = sp; xrc_stdcall_ret(c, 0);
}

/* 00dc4010: thiscall (), ret (optimized code): o = this->+8, with o->+0x44 bit 1: the n = (u16)o->+6 distance
 * constraints of o (32 bytes each at o + o->+0x34, ring order from index base = (u16)o->+8: k = base + i, wrapped
 * by n), each not disabled (+0x18 bit 1): points a, b = o + o->+0x30 + 0x90 * (+0, +4); d = b - a, len =
 * sqrtss(dot3(d)), m = (+0xc * o->+0x24) if not below 1e-5 else 1e-5; a movable when k >= base and not a->+0x88 bit
 * 1, b when not b->+0x88 bit 1. len > m: n = (1 / len) * d; without o->+0x44 bit 4 the velocities (+0x20) get
 * (len - m) * n (both movable: half each, a += / b -=); dv = m * n, else dv = d; then the positions: both: mid =
 * (a + b) * 0.5, b = 0.5 * dv + mid, a = mid - 0.5 * dv; only a: a = b - dv; only b: b = a + dv. Writes the
 * points' +0 / +0x20 (16 bytes); eax / ecx / edx and xmm0..7 as the original leaves them */
B_FN(00dc4010, "distance constraints (ring)", 0) {
	u32 sp = c->esp, th = c->ecx, o = M32(th + 8), fl = M32(o + 0x44), n, base, i, edx = 0;
	X128 x0, x1, x2, x3, x4, x5, x6, x7;
	if (!((fl >> 1) & 1)) { c->eax = o; c->edx = fl >> 1; c->esp = sp; xrc_stdcall_ret(c, 0); return; }
	n = M16(o + 6); base = M32(o + 8) & 0xffffu;
	if ((s32)n <= 0) { c->eax = 0; c->edx = fl >> 1; c->esp = sp; xrc_stdcall_ret(c, 0); return; }
	x0 = c->x[0]; x1 = c->x[1]; x2 = c->x[2]; x3 = c->x[3]; x7 = c->x[7];
	x5 = ex_ldss(0x010a0e7cu); x6 = pc_zero(); x4 = ex_ldss(0x0109b130u);
	for (i = 0; (s32)i < (s32)n; i++) {
		u32 k = (s32)i < (s32)(n - base) ? base + i : base - n + i, ob, e, pa, pb, fa, dl;
		X128 dist;
		ob = M32(th + 8);
		e = M32(ob + 0x34) + (k << 5) + ob;
		if ((M32(e + 0x18) >> 1) & 1) { edx = th; continue; }
		{
			u32 ia = M32(e), ib = M32(e + 4), o2 = M32(th + 8);
			pa = ia * 0x90u + M32(o2 + 0x30) + o2;
			pb = ib * 0x90u + M32(o2 + 0x30) + o2;
			x0 = ex_ld(pa); x2 = ex_ld(pb);
			fa = (s32)k >= (s32)base;
			x2 = ex_subps(x2, x0);
			if ((M32(pa + 0x88) >> 1) & 1) fa = 0;
			dl = (M32(pb + 0x88) >> 1) & 1 ? 0 : 1;
			edx = (o2 & 0xffffff00u) | dl;
		}
		{
			u32 o3 = M32(th + 8), e2 = M32(o3 + 0x34) + (k << 5) + o3;
			x0 = ex_mulps(x2, x2);
			x1 = ex_shufps(x0, x0, 0xaa);
			x1 = ex_addps(x1, x0);
			x0 = ex_shufps(x0, x0, 0x55);
			x1 = ex_addps(x1, x0);
			x0 = ex_ldss(e2 + 0xc);
			x0 = ex_mulss(x0, ex_ldss(o3 + 0x24));
		}
		x1 = ex_sqrtss(x1, x1);
		dist = x1;
		x1 = ex_subss(x0, x5);
		x3 = (ex_comiss(x1, x6) & 4) ? x5 : x0; /* (jb) */
		x1 = ex_ss(dist.u[0]);
		if (!(ex_comiss(x1, x3) & 5)) { /* (not jbe) */
			u32 o4 = M32(th + 8), f4 = M32(o4 + 0x44);
			x0 = ex_ldss(0x010aa950u);
			x0 = ex_divss(x0, x1);
			x0 = ex_shufps(x0, x0, 0x00);
			x0 = ex_mulps(x0, x2);
			if (!((f4 >> 4) & 1)) {
				x1 = ex_subss(x1, x3);
				x1 = ex_shufps(x1, x1, 0x00);
				x1 = ex_mulps(x1, x0);
				if (fa) {
					if (dl) {
						x7 = ex_ld(pa + 0x20);
						x2 = ex_mulps(pc_bc(x4.u[0]), x1);
						x1 = ex_addps(x2, x7);
						BW128(pa + 0x20, x1);
						x1 = ex_subps(ex_ld(pb + 0x20), x2);
						BW128(pb + 0x20, x1);
					} else {
						x2 = ex_ld(pa + 0x20);
						x1 = ex_addps(x1, x2);
						BW128(pa + 0x20, x1);
					}
				} else if (dl) {
					x2 = ex_subps(ex_ld(pb + 0x20), x1);
					BW128(pb + 0x20, x2);
				}
			}
			x2 = ex_mulps(pc_bc(x3.u[0]), x0);
		}
		if (fa) {
			x0 = ex_ld(pb);
			if (dl) {
				x3 = ex_ld(pa);
				x1 = pc_bc(x4.u[0]);
				x0 = ex_addps(x0, x3);
				x0 = ex_mulps(x0, x1);
				x1 = ex_mulps(x1, x2);
				x2 = ex_addps(x1, x0);
				BW128(pb, x2);
				x0 = ex_subps(x0, x1);
				BW128(pa, x0);
			} else {
				x0 = ex_subps(x0, x2);
				BW128(pa, x0);
			}
		} else if (dl) {
			x0 = ex_addps(ex_ld(pa), x2);
			BW128(pb, x0);
		}
	}
	c->x[0] = x0; c->x[1] = x1; c->x[2] = x2; c->x[3] = x3; c->x[4] = x4; c->x[5] = x5; c->x[6] = x6; c->x[7] = x7;
	c->eax = n; c->ecx = (base + n) << 5; c->edx = edx;
	c->esp = sp; xrc_stdcall_ret(c, 0);
}

/* 00dc8090: thiscall (), ret (optimized code, virtual): with this->+0x64: m = this->+0x64->vtbl[5](&M, this->+0x68)
 * (thiscall, ret 8: a 4x4 matrix into the aligned frame at A+0x1c, A = (ebp & ~15) - 0x6c); p = this->+0x40 by M
 * (rows 0..2 by x, y, z, + row 3), w = 1 -> this->+0x20 (16 bytes); v = this->+0x50 by rows 0..2, w = 0, normalized
 * where its dot3 > 0 (rsqrtps(len2) * v, cmpltps mask) with w 0 -> the local at A+0xc; this->00dc7900(&v) (thiscall,
 * ret 4). Without: eax = cookie ^ A. ecx = the cookie */
B_FN(00dc8090, "transformed point and direction", 0) {
	u32 sp = c->esp, th = c->ecx, bp = sp - 4, A = (bp & ~15u) - 0x6cu, ck = M32(0x026e5a84u), esi0 = c->esi, ob, m;
	X128 x0, x1, x2, x3, x4, x5;
	if (!M32(th + 0x64)) { c->eax = ck ^ A; c->ecx = ck; c->esp = sp; xrc_stdcall_ret(c, 0); return; }
	B_ENTER(bp);
	c->esi = th;
	ob = M32(th + 0x64);
	m = M32(M32(ob) + 0x14);
	M32(A - 8) = M32(th + 0x68); M32(A - 0xcu) = A + 0x1cu;
	c->ecx = ob; c->edx = A + 0x1cu; c->eax = m;
	BCALL(A - 0x10u, m, 0x00dc80c4u);
	x1 = ex_ld(th + 0x40); x3 = ex_ld(A + 0x1cu); x4 = ex_ld(A + 0x2cu); x5 = ex_ld(A + 0x3cu);
	x0 = ex_shufps(x1, x1, 0x00);
	x2 = ex_shufps(x1, x1, 0x55);
	x0 = ex_mulps(x0, x3);
	x2 = ex_mulps(x2, x4);
	x0 = ex_addps(x0, x2);
	x2 = ex_shufps(x1, x1, 0xaa);
	x1 = ex_ld(A + 0x4cu);
	x2 = ex_mulps(x2, x5);
	x0 = ex_addps(x0, x2);
	x2 = ex_ld(th + 0x50);
	x0 = ex_addps(x0, x1);
	x1 = ex_shufps(x2, x2, 0x00);
	x1 = ex_mulps(x1, x3);
	x3 = ex_shufps(x2, x2, 0x55);
	x3 = ex_mulps(x3, x4);
	x1 = ex_addps(x1, x3);
	x3 = ex_shufps(x2, x2, 0xaa);
	x3 = ex_mulps(x3, x5);
	x1 = ex_addps(x1, x3);
	x4 = pc_zero();
	x2 = x1;
	x1 = ex_unpckhps(x1, x4);
	x2 = ex_shufps(x2, x1, 0xc4);
	x1 = ex_mulps(x2, x2);
	x3 = ex_shufps(x1, x1, 0xd2);
	x3 = ex_addps(x3, x1);
	x1 = ex_shufps(x1, x1, 0xc9);
	x3 = ex_addps(x3, x1);
	x4 = ex_cmpps(x4, x3, 1);
	x1 = ex_rsqrtps(x3);
	x1 = ex_mulps(x1, x2);
	x1 = ex_andps(x1, x4);
	x4 = ex_andnps(x4, x2);
	x1 = ex_orps(x1, x4);
	x3 = x1;
	x1 = ex_unpckhps(x1, x2);
	x3 = ex_shufps(x3, x1, 0xc4);
	x1 = ex_ldss(0x010aa950u);
	x2 = x0;
	x1 = ex_shufps(x1, x1, 0x00);
	x0 = ex_unpckhps(x0, x1);
	x2 = ex_shufps(x2, x0, 0xc4);
	M32(A - 8) = A + 0xcu;
	XST(A + 0xcu, x3);
	BW128(th + 0x20, x2);
	c->x[0] = x0; c->x[1] = x1; c->x[2] = x2; c->x[3] = x3; c->x[4] = x4; c->x[5] = x5;
	c->ecx = th;
	BCALL(A - 0xcu, 0x00dc7900u, 0x00dc819au);
	c->esi = esi0; c->ecx = ck;
	B_RET(sp, 0);
}

/* 00d7c9d0: thiscall bool (const Aabb *b), ret 4 (optimized code): this = a ray {p at +0, d at +0x10}. p inside b
 * (min(p - b.min, b.max - p) lanes 0..2 without the sign bit, movmskps): 1. dot4(d) < 1e-6: 0. Else the slabs with
 * inv = (1 - rcp * d) * rcp + rcp (rcpps refined): tmin = inv * (b.min - p), tmax = inv * (b.max - p); per axis
 * 0..2: d^2 < 1e-5: p outside [min, max] on it -> 0; else enter = max(enter (0), lo), exit = min(exit (FLT_MAX),
 * hi) as the comiss branches do, enter > exit -> 0; at the end exit > enter. eax = 0 / 1, ecx = the cookie; xmm0..7
 * as the original leaves them; writes nothing */
B_FN(00d7c9d0, "ray / Aabb test", BR_EAX) {
	u32 sp = c->esp, th = c->ecx, bx = M32(sp + 4), l8, lc, r, l;
	X128 x0, x1, x2, x3, x4, x5 = c->x[5], x6 = c->x[6], x7 = c->x[7], s10, s20, s30, s40, s50;
	x1 = pc_zero();
	x0 = ex_ld(bx + 0x10); x2 = ex_ld(bx); x3 = ex_ld(th + 0x10);
	l8 = x1.u[0];
	x1 = ex_ldss(0x0109fb00u);
	lc = x1.u[0];
	x1 = ex_ld(th);
	x4 = ex_subps(x0, x1);
	x1 = ex_subps(x1, x2);
	x1 = ex_minps(x1, x4);
	if (!(pc_movmskps(x1) & 7)) { r = 1; goto done; }
	x6 = ex_ldss(0x010a523cu);
	x1 = ex_ss(x6.u[0]);
	s20 = x1;
	x5 = ex_mulps(x3, x3);
	x1 = ex_shufps(x5, x5, 0x55);
	x1 = ex_addps(x1, x5);
	x6 = ex_shufps(x5, x5, 0xaa);
	x1 = ex_addps(x1, x6);
	x6 = ex_shufps(x5, x5, 0xff);
	x1 = ex_addps(x1, x6);
	s10 = x1;
	x1 = ex_ss(s20.u[0]);
	if (!(ex_comiss(x1, ex_ss(s10.u[0])) & 5)) { r = 0; goto done; } /* (ja) */
	s30 = x0;
	x0 = ex_ldss(0x010aa950u);
	x1 = ex_rcpps(x3);
	x0 = ex_shufps(x0, x0, 0x00);
	x7 = x0;
	x6 = ex_mulps(x1, x3);
	x7 = ex_subps(x7, x6);
	x6 = ex_ld(th);
	s20 = x2;
	x7 = ex_mulps(x7, x1);
	x2 = ex_subps(x2, x6);
	x7 = ex_addps(x7, x1);
	x7 = ex_mulps(x7, x2);
	x2 = ex_mulps(x1, x3);
	x3 = ex_ss(lc);
	x0 = ex_subps(x0, x2);
	x0 = ex_mulps(x0, x1);
	x0 = ex_addps(x0, x1);
	x0 = ex_mulps(x0, x4);
	x4 = ex_ss(l8);
	s10 = x5;
	x5 = ex_ldss(0x010a0e7cu);
	s40 = x7; s50 = x0;
	for (l = 0; l < 3; l++) {
		if (!(ex_comiss(x5, ex_ss(s10.u[l])) & 5)) { /* (not jbe: the axis is degenerate) */
			x0 = ex_ldss(th + 4 * l);
			x1 = ex_ss(s20.u[l]);
			if (!(ex_comiss(x1, x0) & 5)) { r = 0; goto done; }
			if (!(ex_comiss(x0, ex_ss(s30.u[l])) & 5)) { r = 0; goto done; }
		} else {
			x0 = ex_ss(s40.u[l]); x1 = ex_ss(s50.u[l]);
			x2 = (ex_comiss(x1, x0) & 5) ? x1 : x0;
			if (ex_comiss(x0, x1) & 4) x0 = x1;
			if (ex_comiss(x4, x2) & 4) x4 = x2;
			if (ex_comiss(x0, x3) & 5) x3 = x0;
			if (!(ex_comiss(x4, x3) & 5)) { r = 0; goto done; }
		}
	}
	r = (ex_comiss(x3, x4) & 5) ? 0 : 1;
done:
	c->x[0] = x0; c->x[1] = x1; c->x[2] = x2; c->x[3] = x3; c->x[4] = x4; c->x[5] = x5; c->x[6] = x6; c->x[7] = x7;
	c->eax = r; c->ecx = M32(0x026e5a84u);
	c->esp = sp; xrc_stdcall_ret(c, 1);
}
/* ---- battle batch 2, part D: 008d6650, 00a81a20, 00a81ca0, 00ea31b0, 00f13f50 */

/* part D: x87 arithmetic as the translation models it (xrc.h XR87: each result in double, rounded to float when the
 * guest's FCW precision control is 24-bit; fld / fstp of a float: the conversions), and the SSE2 conversions, through
 * inline asm (fast math cannot fold or reorder them) */
static inline double pd_r(double v, int p24) { if (p24) __asm__("cvtsd2ss %0, %0\n\tcvtss2sd %0, %0" : "+x"(v)); return v; }
static inline double pd_f(u32 bits) { X128 a = ex_ss(bits); double r; __asm__("cvtss2sd %1, %0" : "=x"(r) : "x"(a.v)); return r; }
static inline u32 pd_tof(double v) { X128 r; __asm__("cvtsd2ss %1, %0" : "=x"(r.v) : "x"(v)); return r.u[0]; }
static inline double pd_add(double a, double b, int p24) { __asm__("addsd %1, %0" : "+x"(a) : "x"(b)); return pd_r(a, p24); }
static inline double pd_sub(double a, double b, int p24) { __asm__("subsd %1, %0" : "+x"(a) : "x"(b)); return pd_r(a, p24); }
static inline double pd_mul(double a, double b, int p24) { __asm__("mulsd %1, %0" : "+x"(a) : "x"(b)); return pd_r(a, p24); }
/* cvtss2sd dst, src (dst's low double from src's lane 0, its upper 64 bits kept); cvtsd2ss dst, src (lane 0 from src's
 * low double, lanes 1..3 of dst kept) */
static inline X128 pd_cvtss2sd(X128 d, X128 s) { __asm__("cvtss2sd %1, %0" : "+x"(d.v) : "x"(s.v)); return d; }
static inline X128 pd_cvtsd2ss(X128 d, X128 s) { __asm__("cvtsd2ss %1, %0" : "+x"(d.v) : "x"(s.v)); return d; }
/* a divss by a constant read from .rdata. The translation (xrc .rdata const folding) has the constant as a literal and
 * clang's fast math (-freciprocal-math, part of -ffast-math) would make that divide a mulss by the constant's reciprocal:
 * 1 ulp off the x86 divss for ~2% of inputs (16.666666 = 0x41855555: 1 / it = 0x3d75c290); the build has
 * -fno-reciprocal-math since 2026-11-25 (tools/xrc_build.sh, the launcher), so the translation divides too. Ours is the x86 divss;
 * -DPD_TRANSLATION_DIVC: the multiply instead (to check against a translation built with reciprocal math). */
#ifdef PD_TRANSLATION_DIVC
#define pd_divc(a_, cbits_) ex_mulss((a_), ex_divss(ex_ss(0x3f800000u), ex_ss(cbits_)))
#else
#define pd_divc(a_, cbits_) ex_divss((a_), ex_ss(cbits_))
#endif

/* 008d6650: thiscall (float dt), ret 4: advance the 8 timers (16 bytes each at this->+0xdc; no table: eax 0). Each
 * with a source p = e->+0 (else skipped): e->+8 bit 3: e->+0xc |= 1; else bit 4: +0xc &= ~2, e->+4 = p->+0xc, +0xc |= 1;
 * else bit 0 (looping): e->+4 += dt (addss), past p->+0xc (comiss >): e->+4 -= p->+0xc, and with e->+8 bit 1 +0xc bit 1
 * toggled; else with +0xc bit 0 clear: e->+4 += dt, past p->+0xc: e->+4 = p->+0xc, +0xc |= 1. Every value re-read
 * from memory where the original does. eax = 1 if any timer with a source has not both +0xc bits 0 and 2, else 0;
 * ecx 8, edx (the last timer's +0xc & 1, 0 without a source) and xmm0 as the original leaves them */
B_FN(008d6650, "timers: advance 8, any still running", BR_EAX) {
	u32 sp = c->esp, th = c->ecx, dt = M32(sp + 4), i, res = 0, edx = c->edx;
	X128 x0 = c->x[0];
	if (!M32(th + 0xdc)) { c->eax = 0; c->esp = sp; xrc_stdcall_ret(c, 1); return; }
	for (i = 0; i < 8; i++) {
		u32 e = i * 16u + M32(th + 0xdc), f, k;
		if (!M32(e)) { edx = 0; continue; }
		f = M32(e + 8);
		if (f & 8) BW32(e + 0xc, M32(e + 0xc) | 1u);
		else if (f & 0x10) {
			BW32(e + 0xc, M32(e + 0xc) & ~2u);
			x0 = ex_ldss(M32(e) + 0xc);
			BWSS(e + 4, x0);
			BW32(e + 0xc, M32(e + 0xc) | 1u);
		} else if ((f & 1) || !(M32(e + 0xc) & 1)) {
			X128 v, lim;
			x0 = ex_addss(ex_ldss(e + 4), ex_ss(dt));
			BWSS(e + 4, x0);
			v = ex_ldss(e + 4); lim = ex_ldss(M32(e) + 0xc);
			x0 = v;
			if (!(ex_comiss(v, lim) & 5)) { /* (jbe: CF or ZF -- not above, or unordered) */
				if (f & 1) {
					X128 a = ex_ldss(e + 4);
					x0 = ex_subss(a, ex_ldss(M32(e) + 0xc));
					BWSS(e + 4, x0);
					if (M32(e + 8) & 2) BW32(e + 0xc, M32(e + 0xc) ^ 2u);
				} else {
					x0 = ex_ldss(M32(e) + 0xc);
					BWSS(e + 4, x0);
					BW32(e + 0xc, M32(e + 0xc) | 1u);
				}
			}
		}
		k = M32(e + 0xc);
		edx = k & 1;
		if (!(k & 1) || !(k & 4)) res = 1;
	}
	c->eax = res; c->ecx = 8; c->edx = edx; c->x[0] = x0;
	c->esp = sp; xrc_stdcall_ret(c, 1);
}

/* 00a81a20: cdecl (keys, n, float t, float dflt), ret: a curve's value at t, on the x87 stack (st0). n < 2 (unsigned):
 * dflt. Else tt = t / 16.666666 (divss, the constant 01197f60); keys are 32 bytes (+0 time, an int; +4 value; +8 the
 * segment's kind; +0x10 .. +0x1c tangent data): keys[0].time > tt (comiss): keys[0].value; the first k (from keys[1])
 * with k.time > tt, pr = k - 1, by pr's kind -- 0 linear: (tt - pr.time) * ((k.value - pr.value) / (k.time - pr.time))
 * + pr.value (the slope in SSE, the rest x87); 2: the two tangents m0 = L(pi / 180 * pr->+0x14) * pr->+0x1c,
 * m1 = L(pi / 180 * k->+0x10) * k->+0x18 (L: 0x1010a81, the CRT's SSE2 libm, a double in and out in xmm0; pi and 180
 * the floats 0119fef8 / 0109c4d0, divss at each use), s = (tt - pr.time) /
 * (k.time - pr.time): (pr.value - k.value) * ((2 (s - 1) - 1) s^2) + ((tt - pr.time) ((s - 1) m0 + s m1) (s - 1) +
 * pr.value) (x87, operand order as the original); 3: pr->+0x10 d + 0.5 pr->+0x14 d d + pr.value (d = tt - pr.time);
 * else pr.value; none (tt past the last): keys[n - 1].value. Writes nothing outside its frame; eax, ecx, edx, xmm0,
 * xmm1 as the original leaves them */
B_FN(00a81a20, "curve: value at a time", BR_ST0) {
	u32 sp = c->esp, keys = M32(sp + 4), n = M32(sp + 8), bp = sp - 4, end, k, ck, pr, ty;
	int p24 = ((c->fcw >> 8) & 3u) == 0;
	X128 tt, x0, x1;
	double f0, f1, f2, f3;
	if (n < 2) { c->st0 = pd_f(M32(sp + 16)); c->esp = sp; xrc_cdecl_ret(c); return; }
	tt = pd_divc(ex_ss(M32(sp + 12)), M32(0x01197f60u));
	end = (n << 5) + keys;
	x0 = ex_cvtsi2ss(tt, (s32)M32(keys));
	if (!(ex_comiss(x0, tt) & 5)) {
		c->st0 = pd_f(M32(keys + 4));
		c->eax = keys; c->ecx = end; c->edx = keys; c->x[0] = x0;
		c->esp = sp; xrc_cdecl_ret(c); return;
	}
	ck = keys + 0x20;
	for (k = keys + 0x20;; k += 0x20) {
		if (k == end) {
			c->st0 = pd_f(M32(keys + ((n - 1u) << 5) + 4));
			c->eax = keys; c->ecx = ck; c->edx = (n - 1u) << 5; c->x[0] = x0;
			c->esp = sp; xrc_cdecl_ret(c); return;
		}
		ck = k;
		x0 = ex_cvtsi2ss(x0, (s32)M32(k));
		if (!(ex_comiss(x0, tt) & 5)) break;
	}
	pr = k - 0x20;
	ty = M32(pr + 8);
	if (ty == 0) {
		X128 a = ex_cvtsi2ss(x0, (s32)M32(pr)), dv, sl;
		u32 pv = M32(pr + 4);
		dv = ex_subss(ex_ldss(k + 4), ex_ss(pv));
		x1 = ex_subss(ex_cvtsi2ss(c->x[1], (s32)M32(k)), ex_ss(a.u[0]));
		sl = ex_divss(dv, x1);
		f0 = pd_f(tt.u[0]);
		f0 = pd_sub(f0, pd_f(a.u[0]), p24);
		f0 = pd_mul(f0, pd_f(sl.u[0]), p24);
		f0 = pd_add(f0, pd_f(pv), p24);
		c->x[0] = sl; c->x[1] = x1; c->eax = k; c->ecx = k; c->edx = pr;
	} else if (ty == 2) {
		X128 q, m0, m1, pt, num, s, sm1;
		u32 pv;
		B_ENTER(bp);
		q = ex_mulss(ex_divss(ex_ldss(0x0119fef8u), ex_ldss(0x0109c4d0u)), ex_ldss(pr + 0x14));
		c->x[0] = pd_cvtss2sd(q, q);
		c->eax = pr; c->ecx = k; c->edx = pr;
		BCALL(bp - 0x58, 0x01010a81u, 0x00a81b38u);
		x0 = pd_cvtsd2ss(c->x[0], c->x[0]);
		m0 = ex_mulss(ex_ss(x0.u[0]), ex_ldss(pr + 0x1c));
		q = ex_mulss(ex_divss(ex_ldss(0x0119fef8u), ex_ldss(0x0109c4d0u)), ex_ldss(k + 0x10));
		c->x[0] = pd_cvtss2sd(q, q);
		c->ecx = pr; c->edx = k;
		BCALL(bp - 0x58, 0x01010a81u, 0x00a81b7au);
		x0 = pd_cvtsd2ss(c->x[0], c->x[0]);
		m1 = ex_mulss(ex_ss(x0.u[0]), ex_ldss(k + 0x18));
		pt = ex_cvtsi2ss(m1, (s32)M32(pr));
		pv = M32(pr + 4);
		num = ex_subss(ex_ss(tt.u[0]), ex_ss(pt.u[0]));
		x1 = ex_subss(ex_cvtsi2ss(c->x[1], (s32)M32(k)), ex_ss(pt.u[0]));
		s = ex_divss(num, x1);
		sm1 = ex_subss(ex_ss(s.u[0]), ex_ldss(0x010aa950u));
		f0 = pd_sub(pd_f(pv), pd_f(M32(k + 4)), p24);
		f1 = pd_add(pd_f(sm1.u[0]), pd_f(sm1.u[0]), p24);
		f1 = pd_sub(f1, pd_f(M32(0x010aa950u)), p24);
		f2 = pd_mul(pd_f(s.u[0]), pd_f(s.u[0]), p24);
		f1 = pd_mul(f1, f2, p24);
		f0 = pd_mul(f0, f1, p24);
		f1 = pd_sub(pd_f(tt.u[0]), pd_f(pt.u[0]), p24);
		f2 = pd_mul(pd_f(sm1.u[0]), pd_f(m0.u[0]), p24);
		f3 = pd_mul(pd_f(s.u[0]), pd_f(m1.u[0]), p24);
		f2 = pd_add(f2, f3, p24);
		f2 = pd_mul(f2, pd_f(sm1.u[0]), p24);
		f1 = pd_mul(f1, f2, p24);
		f1 = pd_add(f1, pd_f(pv), p24);
		f0 = pd_add(f0, f1, p24);
		c->x[0] = sm1; c->x[1] = x1; c->eax = k; c->ecx = k; c->edx = pr;
		c->st0 = f0;
		B_RET(sp, 0);
		return;
	} else if (ty == 3) {
		u32 qd = M32(pr + 0x14), l = M32(pr + 0x10);
		X128 pt = ex_cvtsi2ss(ex_ss(l), (s32)M32(pr));
		x1 = ex_subss(tt, pt);
		f0 = pd_mul(pd_f(l), pd_f(x1.u[0]), p24);
		f1 = pd_mul(pd_f(M32(0x0109b130u)), pd_f(qd), p24);
		f1 = pd_mul(f1, pd_f(x1.u[0]), p24);
		f1 = pd_mul(f1, pd_f(x1.u[0]), p24);
		f0 = pd_add(f0, f1, p24);
		f0 = pd_add(f0, pd_f(M32(pr + 4)), p24);
		c->x[0] = pt; c->x[1] = x1; c->eax = pr; c->ecx = pr; c->edx = pr;
	} else {
		f0 = pd_f(M32(pr + 4));
		c->x[0] = x0; c->eax = pr; c->ecx = pr; c->edx = pr;
	}
	c->st0 = f0;
	c->esp = sp; xrc_cdecl_ret(c);
}

/* 00a81ca0: cdecl (curve, float t), ret: 00a81a20(curve->+0x10, curve->+8, t, the default 0119fefc[curve->+0]) (st0,
 * as it leaves it). Looping (curve->+4 bit 0) with dur = curve->+(0xc + 4 n)->time - curve->+0x10->time > 0 first:
 * D = dur * 16.666666, F = first.time * 16.666666 (mulss), r = (float)0x1012f2a(st1 t - F, st0 D) (the CRT's x87 fmod),
 * r < 0: r += D; t = F + r -- written to its argument slot (sp + 8) as the original does */
B_FN(00a81ca0, "curve: value at a time, looped", BR_ST0) {
	u32 sp = c->esp, cv = M32(sp + 4), bp = sp - 4, at = bp - 0x30, ty;
	int p24 = ((c->fcw >> 8) & 3u) == 0;
	B_ENTER(bp);
	if (M32(cv + 4) & 1) {
		u32 first = M32(cv + 0x10), last = M32(cv + M32(cv + 8) * 4u + 0xc);
		s32 dur = (s32)(M32(last) - M32(first));
		if (dur > 0) {
			X128 d, f, r;
			double t0;
			d = ex_mulss(ex_cvtsi2ss(c->x[0], dur), ex_ldss(0x01197f60u));
			f = ex_mulss(ex_cvtsi2ss(d, (s32)M32(first)), ex_ldss(0x01197f60u));
			t0 = pd_sub(pd_f(M32(sp + 8)), pd_f(f.u[0]), p24);
			c->fa[0] = pd_f(d.u[0]); c->fa[1] = t0;
			c->x[0] = f; c->eax = first; c->ecx = (u32)dur; c->edx = first;
			BCALL(bp - 0x20, 0x01012f2au, 0x00a81d15u);
			r = ex_ss(pd_tof(c->st0));
			if (!(ex_comiss(ex_ldss(0x025a3c20u), r) & 5)) r = ex_addss(r, ex_ss(d.u[0]));
			r = ex_addss(ex_ss(f.u[0]), r);
			M32(sp + 8) = r.u[0];
			c->x[0] = r;
		}
	}
	ty = M32(cv);
	M32(at + 16) = M32(ty * 4u + 0x0119fefcu); /* (fld / fstp float: the translation's plain copy) */
	M32(at + 12) = M32(sp + 8);
	M32(at + 8) = M32(cv + 8);
	M32(at + 4) = M32(cv + 0x10);
	c->eax = M32(cv + 8); c->ecx = cv; c->edx = M32(cv + 0x10);
	BCALL(at, 0x00a81a20u, 0x00a81d74u);
	B_RET(sp, 0);
}


/* 00f13f50: cdecl (m, p, q), ret (the frame aligned: S = (sp - 4 & ~15) - 0x140): P = *p at S+0x120, Q = *q at
 * S+0x110; m->004584c0(S+0x100, P), (m + 0x40)->004584c0(S+0xf0, P), (m + 0x80)->00711f40(S+0xe0, Q), (m + 0xc0)->
 * 00711f40(S+0xd0, Q); z = minps([S+0x100] + [S+0xe0], [S+0xf0] + [S+0xd0]); eax = 1 if no lane of z has its sign bit
 * (movmskps 0), else 0; xmm0 = xmm1 = z, ecx the cookie global */
B_FN(00f13f50, "two transformed points: no lane negative", BR_EAX) {
	u32 sp = c->esp, m = M32(sp + 4), p = M32(sp + 8), q = M32(sp + 12), bp = sp - 4, s = (bp & ~15u) - 0x140u, mask;
	X128 z;
	B_ENTER(bp);
	XST(s + 0x120, ex_ld(p));
	XST(s + 0x110, ex_ld(q));
	M32(s - 4) = s + 0x120; M32(s - 8) = s + 0x100;
	c->eax = s + 0x100; c->ecx = m; c->edx = s + 0x120;
	BCALL(s - 0xc, 0x004584c0u, 0x00f13f9eu);
	M32(s - 4) = s + 0x120; M32(s - 8) = s + 0xf0;
	c->ecx = m + 0x40; c->edx = s + 0xf0;
	BCALL(s - 0xc, 0x004584c0u, 0x00f13fb9u);
	M32(s - 4) = s + 0x110; M32(s - 8) = s + 0xe0;
	c->eax = s + 0x110; c->ecx = m + 0x80;
	BCALL(s - 0xc, 0x00711f40u, 0x00f13fd7u);
	M32(s - 4) = s + 0x110; M32(s - 8) = s + 0xd0;
	c->eax = s + 0xd0; c->ecx = m + 0xc0; c->edx = s + 0x110;
	BCALL(s - 0xc, 0x00711f40u, 0x00f13ff5u);
	z = ex_minps(ex_addps(ex_ld(s + 0x100), ex_ld(s + 0xe0)), ex_addps(ex_ld(s + 0xf0), ex_ld(s + 0xd0)));
	mask = (z.u[0] >> 31) | (z.u[1] >> 31) << 1 | (z.u[2] >> 31) << 2 | (z.u[3] >> 31) << 3;
	c->eax = mask ? 0u : 1u; c->ecx = M32(0x026e5a84u);
	c->x[0] = z; c->x[1] = z;
	B_RET(sp, 0);
}
/* ----------------------------------------------------------------------- battle batch 2, part E (00da7c90, 00dca0b0)
 * Both are optimized (/O2) SSE code, not /Od: the xmm registers are followed one for one (x[0..7] = the guest's
 * xmm0..7, from c->x at the entry, back to c->x around every call and at the return, as the translation does), the
 * stack locals are C values except those a callee reads (written at the original's addresses). rsqrtps is the
 * translation's exact 1 / sqrt (xv_rsqrtps), as in ovr_simd.c. */
static inline X128 pe_zero(void) { X128 r; r.u[0] = r.u[1] = r.u[2] = r.u[3] = 0; return r; }
/* movss xmm_a, xmm_b: lane 0 from b, lanes 1..3 of a */
static inline X128 pe_movss(X128 a, X128 b) { a.u[0] = b.u[0]; return a; }
/* cvtps2pd (lanes 0, 1 to two doubles) / cvtsd2ss a, b (lane 0 = (float)b.d[0], lanes 1..3 of a) */
static inline X128 pe_cvtps2pd(X128 a) { X128 r; __asm__("cvtps2pd %1, %0" : "=x"(r.v) : "x"(a.v)); return r; }
static inline X128 pe_cvtsd2ss(X128 a, X128 b) { __asm__("cvtsd2ss %1, %0" : "+x"(a.v) : "x"(b.v)); return a; }
/* the translation's x87 (doubles; XR87: rounded to float when the control word's precision is 24-bit) */
static inline double pe_mulsd(double a, double b) { __asm__("mulsd %1, %0" : "+x"(a) : "x"(b)); return a; }
static inline double pe_addsd(double a, double b) { __asm__("addsd %1, %0" : "+x"(a) : "x"(b)); return a; }
static inline double pe_pc(double v, u32 fcw) { return ((fcw >> 8) & 3u) == 0 ? (double)(float)v : v; }
static inline double pe_fd(u32 bits) { float f; memcpy(&f, &bits, 4); return (double)f; }
static inline u32 pe_fbits(double d) { float f = (float)d; u32 u; memcpy(&u, &f, 4); return u; }
/* fld dword [m] * fmul dword [mb] + fadd dword [ma] -> fstp dword: st0 the callee's result */
static inline u32 pe_x87_madd(double st0, u32 mb, u32 ma, u32 fcw) {
	double f = pe_pc(pe_mulsd(st0, pe_fd(M32(mb))), fcw);
	return pe_fbits(pe_pc(pe_addsd(f, pe_fd(M32(ma))), fcw));
}
/* the guest xmm0..xmm(n-1) to / from c->x; a call with them */
#define PE_GET7() (x[0] = c->x[0], x[1] = c->x[1], x[2] = c->x[2], x[3] = c->x[3], x[4] = c->x[4], x[5] = c->x[5], x[6] = c->x[6])
#define PE_PUT7() (c->x[0] = x[0], c->x[1] = x[1], c->x[2] = x[2], c->x[3] = x[3], c->x[4] = x[4], c->x[5] = x[5], c->x[6] = x[6])
#define PE_GET8() (PE_GET7(), x[7] = c->x[7])
#define PE_PUT8() (PE_PUT7(), c->x[7] = x[7])
#define PE_CALL7(at_, t_, ra_) do { PE_PUT7(); BCALL((at_), (t_), (ra_)); PE_GET7(); } while (0)
#define PE_CALL8(at_, t_, ra_) do { PE_PUT8(); BCALL((at_), (t_), (ra_)); PE_GET8(); } while (0)
/* (the check: before ours's first rand, the CRT seed (_getptd 01019e93 ()->+0x14) is saved for b_check to put back before the
 * translation runs, so both draw the same numbers; _getptd changes nothing else -- eax, ecx, edx kept) */
static inline void pe_seed(XCtx *c, u32 at) {
	u32 ea, ec, ed;
	if (bj_owner != c || b_rs) return;
	ea = c->eax; ec = c->ecx; ed = c->edx;
	BCALL(at, 0x01019e93u, 0x00da7e30u);
	b_rs_at = c->eax + 0x14u; b_rs_val = M32(b_rs_at); b_rs = 1;
	c->eax = ea; c->ecx = ec; c->edx = ed;
}

/* 00da7c90: thiscall (v, p, a3, a4, slot), ret 0x14, no result (callers 007b8dcc / 00da8250 read no register after
 * it). len2 = dot3(v, v) (lanes (z + x) + y); 0 >= len2: nothing. Else n = normalize3((0, 1, C) x v) (C the float
 * 3e072b02 at 02613b04; rsqrtps and one Newton step; lane 3 v.w), k = (1 - (1 - a4)^2) * a3, a34 = a3 * a4 (stored
 * into the a3 slot by the original: dead after the ret, not made). The slot's two sets of 4 waves (q = this +
 * slot * 0x70 + 0xa8: phase q[j], speed q+0x10[j], amplitude q+0x20[j]; phase q+0x30[j], speed q+0x40[j], amplitude
 * q+0x50[j]): phase += speed * a4 (written), >= 2pi: -= 2pi (written) and with this->+0x9e & 0xf clear new speed /
 * amplitude rand (00dcd2a0, x87) * p[2] + p[1], rand * p[4] + p[3] (second set p[6] + p[5], p[8] + p[7]). Flag clear:
 * q+0x60..0x68 = v.xyz, q+0x6c = p[0] (fld / fstp). Then for i < count (dwords this->+0x3c .. +0x40): j = i & 3,
 * s = (float)f(q[j]) * q+0x20[j] * k + q+0x6c * a34, t = (float)f(q+0x30[j]) * q+0x50[j] * k (f: 0100eed4, SSE2
 * libm, double in xmm0); this->+0x3c[i]->vtbl[0x80 / 4](&(s * v + t * n), 1) (thiscall, ret 8; the vector a local at
 * the original's address). */
B_FN(00da7c90, "wave phases and per-object offsets", 0) {
	u32 sp = c->esp, th = c->ecx, v = M32(sp + 4), p = M32(sp + 8), slot = M32(sp + 0x14), bp = sp - 4;
	u32 S = (bp & 0xfffffff0u) - 0x54u - 12u; /* esp after the frame and the 3 pushes */
	u32 ebx0 = c->ebx, esi0 = c->esi, edi0 = c->edi, fcw = c->fcw, n, q, j, i, k, a34;
	X128 x[8], nv, sv;
	PE_GET7();
	B_ENTER(bp);
	c->eax = th;
	c->edx = M32(th + 0x3c);
	n = 0;
	if (c->edx) c->ecx = n = (u32)((s32)(M32(th + 0x40) - c->edx) >> 2);
	x[4] = ex_ss(0x3f800000u);
	x[5] = ex_ldss(sp + 0x10);
	x[0] = ex_subss(x[4], x[5]);
	x[2] = ex_mulss(x[0], x[0]);
	x[0] = ex_ldss(sp + 0xc);
	x[1] = ex_subss(x[4], x[2]);
	x[1] = ex_mulss(x[1], x[0]);
	x[3] = pe_zero();
	x[0] = ex_mulss(x[0], x[5]);
	k = x[1].u[0];
	x[1] = ex_ld(v);
	a34 = x[0].u[0];
	x[0] = pe_zero();
	x[3] = pe_movss(x[3], x[0]);
	x[2] = ex_mulps(x[1], x[1]);
	x[3] = ex_shufps(x[2], x[2], 0xaa);
	x[3] = ex_addps(x[3], x[2]);
	x[2] = ex_shufps(x[2], x[2], 0x55);
	x[3] = ex_addps(x[3], x[2]);
	x[2] = pe_zero(); /* (movss from the zeroed local) */
	if (!(ex_comiss(x[2], x[3]) & 4)) goto out; /* jae */
	x[3] = ex_ss(0x3e072b02u);
	c->ecx = slot * 0x70u;
	x[6] = x[4];
	x[0] = ex_unpcklps(x[0], x[3]);
	x[3] = ex_shufps(x[1], x[1], 0xd2);
	x[2] = x[4];
	c->ebx = p;
	q = c->ecx + th + 0xa8u;
	c->edi = q;
	x[2] = ex_unpcklps(x[2], x[6]);
	x[0] = ex_unpcklps(x[0], x[2]);
	x[2] = ex_shufps(x[0], x[0], 0xc9);
	x[0] = ex_shufps(x[0], x[0], 0xd2);
	x[2] = ex_mulps(x[2], x[3]);
	x[3] = ex_shufps(x[1], x[1], 0xc9);
	x[0] = ex_mulps(x[0], x[3]);
	x[0] = ex_subps(x[0], x[2]);
	x[2] = x[0];
	x[0] = ex_unpckhps(x[0], x[1]);
	x[2] = ex_shufps(x[2], x[0], 0xc4);
	x[0] = ex_mulps(x[2], x[2]);
	x[1] = ex_shufps(x[0], x[0], 0xd2);
	x[1] = ex_addps(x[1], x[0]);
	x[0] = ex_shufps(x[0], x[0], 0xc9);
	x[1] = ex_addps(x[1], x[0]);
	x[3] = xv_rsqrtps(x[3], x[1]);
	x[6] = ex_mulps(x[3], x[1]);
	x[1] = ex_ss(0x3f000000u);
	x[1] = ex_shufps(x[1], x[1], 0);
	x[1] = ex_mulps(x[1], x[3]);
	x[0] = pe_movss(x[0], x[4]);
	x[0] = ex_shufps(x[0], x[0], 0);
	x[6] = ex_mulps(x[6], x[3]);
	x[0] = ex_subps(x[0], x[6]);
	x[0] = ex_mulps(x[0], x[1]);
	x[0] = ex_addps(x[0], x[3]);
	x[0] = ex_mulps(x[0], x[2]);
	x[1] = x[0];
	x[0] = ex_unpckhps(x[0], x[2]);
	x[1] = ex_shufps(x[1], x[0], 0xc4);
	nv = x[1];
	x[1] = ex_ss(0x40c90fdbu); /* 2pi */
	for (j = 0; j < 4; j++) {
		u32 e = q + 0x30u + j * 4u;
		unsigned fl;
		c->esi = e;
		x[0] = ex_ldss(e - 0x20);
		x[0] = ex_mulss(x[0], x[5]);
		x[0] = ex_addss(x[0], ex_ldss(e - 0x30));
		fl = ex_comiss(x[0], x[1]);
		BWSS(e - 0x30, x[0]);
		if (!(fl & 4)) {
			x[0] = ex_subss(x[0], x[1]);
			BWSS(e - 0x30, x[0]);
			if (!(M8(th + 0x9e) & 0xf)) {
				pe_seed(c, S - 4); PE_CALL7(S - 4, 0x00dcd2a0u, 0x00da7e30u);
				BW32(e - 0x20, pe_x87_madd(c->st0, p + 8, p + 4, fcw));
				pe_seed(c, S - 4); PE_CALL7(S - 4, 0x00dcd2a0u, 0x00da7e3eu);
				x[5] = ex_ldss(sp + 0x10);
				x[1] = ex_ss(0x40c90fdbu);
				c->eax = th;
				BW32(e - 0x10, pe_x87_madd(c->st0, p + 0x10, p + 0xc, fcw));
			}
		}
		x[0] = ex_ldss(e + 0x10);
		x[0] = ex_mulss(x[0], x[5]);
		x[0] = ex_addss(x[0], ex_ldss(e));
		fl = ex_comiss(x[0], x[1]);
		BWSS(e, x[0]);
		if (!(fl & 4)) {
			x[0] = ex_subss(x[0], x[1]);
			BWSS(e, x[0]);
			if (!(M8(th + 0x9e) & 0xf)) {
				pe_seed(c, S - 4); PE_CALL7(S - 4, 0x00dcd2a0u, 0x00da7e84u);
				BW32(e + 0x10, pe_x87_madd(c->st0, p + 0x18, p + 0x14, fcw));
				pe_seed(c, S - 4); PE_CALL7(S - 4, 0x00dcd2a0u, 0x00da7e92u);
				x[5] = ex_ldss(sp + 0x10);
				x[1] = ex_ss(0x40c90fdbu);
				c->eax = th;
				BW32(e + 0x20, pe_x87_madd(c->st0, p + 0x20, p + 0x1c, fcw));
			}
		}
	}
	c->esi = q + 0x40u;
	if (!(M8(th + 0x9e) & 0xf)) {
		c->eax = v;
		BW32(q + 0x60, x87_copy(M32(v)));
		BW32(q + 0x64, x87_copy(M32(v + 4)));
		BW32(q + 0x68, x87_copy(M32(v + 8)));
		BW32(q + 0x6c, x87_copy(M32(p)));
	}
	for (i = 0; i < n; i++) {
		u32 jj = i & 3u, o, vt, m;
		c->ebx = i; c->esi = jj;
		x[0] = ex_ldss(q + jj * 4u);
		x[0] = pe_cvtps2pd(x[0]);
		PE_CALL7(S - 4, 0x0100eed4u, 0x00da7f02u);
		x[1] = ex_ldss(q + 0x6c);
		x[1] = ex_mulss(x[1], ex_ss(a34));
		c->edx = v;
		x[0] = pe_cvtsd2ss(x[0], x[0]);
		x[0] = ex_mulss(x[0], ex_ldss(q + jj * 4u + 0x20));
		x[0] = ex_mulss(x[0], ex_ss(k));
		x[0] = ex_addss(x[0], x[1]);
		x[1] = ex_ld(v);
		x[0] = ex_shufps(x[0], x[0], 0);
		x[0] = ex_mulps(x[0], x[1]);
		sv = x[0];
		x[0] = ex_ldss(q + jj * 4u + 0x30);
		x[0] = pe_cvtps2pd(x[0]);
		PE_CALL7(S - 4, 0x0100eed4u, 0x00da7f41u);
		c->eax = th;
		c->ecx = M32(th + 0x3c);
		x[1] = sv;
		x[0] = pe_cvtsd2ss(x[0], x[0]);
		x[0] = ex_mulss(x[0], ex_ldss(q + jj * 4u + 0x50));
		x[0] = ex_mulss(x[0], ex_ss(k));
		M32(S - 4) = 1;
		c->eax = S + 0x40u;
		M32(S - 8) = S + 0x40u;
		x[0] = ex_shufps(x[0], x[0], 0);
		x[0] = ex_mulps(x[0], nv);
		x[1] = ex_addps(x[1], x[0]);
		XST(S + 0x40u, x[1]);
		o = M32(c->ecx + i * 4u); vt = M32(o); m = M32(vt + 0x80);
		c->ecx = o; c->edx = m;
		PE_CALL7(S - 12, m, 0x00da7f82u);
	}
out:
	c->ecx = M32(0x026e5a84u); /* (the cookie check's ecx: cookie ^ esp ^ esp) */
	c->ebx = ebx0; c->esi = esi0; c->edi = edi0;
	PE_PUT7();
	B_RET(sp, 5);
}

/* ---- battle batch 3 */
/* ---- battle batch 3, part A: 00f13c90, 00f0f640, 00f0ee60, 00f115b0 (/Od code: the octree / BVH culling of objects
 * against a view volume and the volume of an oriented box seen from a point)
 *
 * QA_CALLS 0 (default): the small leaf helpers these call are computed here with the same operations in the same
 * order -- their only effects are their out vectors (in the caller's frame, read back by the caller alone) and
 * registers nobody reads: Vec4::Dot4 00710860, Vec4::Scale 00727710, Aabb::Center 0075e920, Vec4::Cross 00458990,
 * Matrix33::Transform 00711f40, Matrix44 copy 007a8b50 / from rows 00c28db0 / Transpose 00728100 / Neg 00f14f90 /
 * Abs 00db4dd0, the getter 00ef5250 (this->+0x84[i]) and the id search 00f04630. QA_CALLS 1: every call made as the
 * original makes it (their inputs written at the original's frame addresses first). The other callees (00f13f50,
 * 00457e50, 00f14110, 00f14950, the recursions, the error report) are called either way. */
#ifndef QA_CALLS
#define QA_CALLS 0
#endif

static inline X128 qa_bc(u32 b) { X128 r; r.vi = _mm_set1_epi32((int)b); return r; }
static inline X128 qa_zero(void) { X128 r; r.vi = _mm_setzero_si128(); return r; }
static inline u32 qa_movmsk(X128 a) { return a.u[0] >> 31 | (a.u[1] >> 31) << 1 | (a.u[2] >> 31) << 2 | (a.u[3] >> 31) << 3; } /* movmskps */
/* 00710860 Vec4::Dot4 (this a, v b): p = a * b; ((p + p.yyyy) + p.zzzz) + p.wwww */
static inline X128 qa_dot4(X128 a, X128 b) {
	X128 p = ex_mulps(a, b), r;
	r = ex_addps(p, ex_shufps(p, p, 0x55));
	r = ex_addps(r, ex_shufps(p, p, 0xaa));
	return ex_addps(r, ex_shufps(p, p, 0xff));
}
/* 00458990 Vec4::Cross (this a, v b): d = a.yzx * b.zxy - b.yzx * a.zxy; {d.x, d.y, d.z, a.w} (shufps d, unpckhps(d, a)) */
static inline X128 qa_cross(X128 a, X128 b) {
	X128 l = ex_mulps(ex_shufps(a, a, 0xc9), ex_shufps(b, b, 0xd2)), r = ex_mulps(ex_shufps(b, b, 0xc9), ex_shufps(a, a, 0xd2));
	X128 d = ex_subps(l, r);
	return ex_shufps(d, ex_unpckhps(d, a), 0xc4);
}
/* 00711f40 Matrix33::Transform (rows m0..m2, v): (m0 * v.xxxx + m1 * v.yyyy) + m2 * v.zzzz */
static inline X128 qa_m33(X128 m0, X128 m1, X128 m2, X128 v) {
	X128 r = ex_addps(ex_mulps(m0, ex_shufps(v, v, 0x00)), ex_mulps(m1, ex_shufps(v, v, 0x55)));
	return ex_addps(r, ex_mulps(m2, ex_shufps(v, v, 0xaa)));
}
/* 00728100 Matrix44 transpose (its shufps 0x44 / 0xee, then 0x88 / 0xdd) */
static inline void qa_tr(X128 r0, X128 r1, X128 r2, X128 r3, X128 *o) {
	X128 a = ex_shufps(r0, r1, 0x44), b = ex_shufps(r0, r1, 0xee), e = ex_shufps(r2, r3, 0x44), f = ex_shufps(r2, r3, 0xee);
	o[0] = ex_shufps(a, e, 0x88); o[1] = ex_shufps(a, e, 0xdd); o[2] = ex_shufps(b, f, 0x88); o[3] = ex_shufps(b, f, 0xdd);
}
#if QA_CALLS
/* the calls as the original makes them: this at ta, v at va (written there: the values the original has there), out */
static inline X128 qa_c_cross(XCtx *c, u32 at, u32 ra, u32 ta, X128 a, u32 out, u32 va, X128 b) {
	XST(ta, a); XST(va, b);
	M32(at + 4) = out; M32(at + 8) = va; c->ecx = ta;
	BCALL(at, 0x00458990u, ra);
	return ex_ld(out);
}
static inline X128 qa_c_dot4(XCtx *c, u32 at, u32 ra, u32 ta, X128 a, u32 out, u32 va, X128 b) {
	XST(ta, a); XST(va, b);
	M32(at + 4) = out; M32(at + 8) = va; c->ecx = ta;
	BCALL(at, 0x00710860u, ra);
	return ex_ld(out);
}
static inline X128 qa_c_m33(XCtx *c, u32 at, u32 ra, u32 ta, const X128 *m, u32 out, u32 va, X128 v) {
	XST(ta, m[0]); XST(ta + 16, m[1]); XST(ta + 32, m[2]); XST(va, v);
	M32(at + 4) = out; M32(at + 8) = va; c->ecx = ta;
	BCALL(at, 0x00711f40u, ra);
	return ex_ld(out);
}
static inline void qa_c_tr(XCtx *c, u32 at, u32 ra, u32 ta, X128 r0, X128 r1, X128 r2, X128 r3, u32 out, X128 *o) {
	XST(ta, r0); XST(ta + 16, r1); XST(ta + 32, r2); XST(ta + 48, r3);
	M32(at + 4) = out; c->ecx = ta;
	BCALL(at, 0x00728100u, ra);
	o[0] = ex_ld(out); o[1] = ex_ld(out + 16); o[2] = ex_ld(out + 32); o[3] = ex_ld(out + 48);
}
#define QA_CROSS(at_, ra_, ta_, a_, out_, va_, b_) qa_c_cross(c, (at_), (ra_), (ta_), (a_), (out_), (va_), (b_))
#define QA_DOT4(at_, ra_, ta_, a_, out_, va_, b_) qa_c_dot4(c, (at_), (ra_), (ta_), (a_), (out_), (va_), (b_))
#define QA_M33(at_, ra_, ta_, m_, out_, va_, v_) qa_c_m33(c, (at_), (ra_), (ta_), (m_), (out_), (va_), (v_))
#define QA_TR(at_, ra_, ta_, r0_, r1_, r2_, r3_, out_, o_) qa_c_tr(c, (at_), (ra_), (ta_), (r0_), (r1_), (r2_), (r3_), (out_), (o_))
#else
#define QA_CROSS(at_, ra_, ta_, a_, out_, va_, b_) qa_cross((a_), (b_))
#define QA_DOT4(at_, ra_, ta_, a_, out_, va_, b_) qa_dot4((a_), (b_))
#define QA_M33(at_, ra_, ta_, m_, out_, va_, v_) qa_m33((m_)[0], (m_)[1], (m_)[2], (v_))
#define QA_TR(at_, ra_, ta_, r0_, r1_, r2_, r3_, out_, o_) qa_tr((r0_), (r1_), (r2_), (r3_), (o_))
#endif

/* 00f13c90: cdecl (u32 *objs_out, float *dist_out, u32 max, u32 *count, Node **nodes, Node *nd, Obj **objs, m, const
 * Vec4 *pos), ret (frame aligned: E = (sp - 4 & ~15) - 0x90, esp after the push esi): the BVH walk -- 00f13f50(m, nd,
 * nd + 0x10) (the node's box against m) false: nothing. Else for its 8 children i: the u16 nd->+0x48[i] != 0xffff:
 * recursion with nd = *nodes + child * 0x60; else the leaf's objects j in [nd->+0x20[i], nd->+0x24[i]) (unsigned):
 * o = objs[j] with o->+0x64 bit 0 and 00f13f50(m, o + 0x40, o + 0x50): d = Dot4(o->+0x40 - *pos) (lane 0, 00710860
 * on the local at E+0x20 into E+0x10), the first k < min(*count, max) with dist_out[k] > d (comiss ja); k < max:
 * *count += 1 if below max, the entries [k, *count - 1) shifted up by one (dist_out by its x87 fld / fstp: a plain
 * copy, as the translation folds it), dist_out[k] = d, objs_out[k] = o. Writes those; no result (callers read no
 * register); ecx = the cookie, esi as it was */
B_FN(00f13c90, "BVH: visible objects nearest first", 0) {
	u32 sp = c->esp, oo = M32(sp + 4), od = M32(sp + 8), mx = M32(sp + 0xc), pn = M32(sp + 0x10), nb = M32(sp + 0x14);
	u32 nd = M32(sp + 0x18), objs = M32(sp + 0x1c), fr = M32(sp + 0x20), pos = M32(sp + 0x24), bp = sp - 4;
	u32 E = (bp & ~15u) - 0x90u, i;
	B_ENTER(bp);
	M32(E - 4) = nd + 0x10u; M32(E - 8) = nd; M32(E - 12) = fr;
	c->eax = nd + 0x10u; c->ecx = nd; c->edx = fr;
	BCALL(E - 16, 0x00f13f50u, 0x00f13cbfu);
	if (c->eax & 0xffu) for (i = 0; i < 8; i++) {
		u32 ch = M16(nd + 0x48u + i * 2u);
		if (ch != 0xffffu) {
			M32(E - 4) = pos; M32(E - 8) = fr; M32(E - 12) = objs; M32(E - 16) = ch * 0x60u + M32(nb); M32(E - 20) = nb;
			M32(E - 24) = pn; M32(E - 28) = mx; M32(E - 32) = od; M32(E - 36) = oo;
			c->eax = mx; c->ecx = od; c->edx = oo;
			BCALL(E - 40, 0x00f13c90u, 0x00f13f2eu);
		} else {
			u32 j = M32(nd + 0x20u + i * 4u), end = M32(nd + 0x24u + i * 4u);
			for (; j < end; j++) {
				u32 ob = M32(objs + j * 4u), k, n, m, dist;
				X128 df;
				if (!(M32(ob + 0x64) & 1u)) continue;
				M32(E - 4) = ob + 0x50u; M32(E - 8) = ob + 0x40u; M32(E - 12) = fr;
				c->eax = ob + 0x50u; c->ecx = ob + 0x40u; c->edx = fr;
				BCALL(E - 16, 0x00f13f50u, 0x00f13db3u);
				if (!(c->eax & 0xffu)) continue;
				df = ex_subps(ex_ld(ob + 0x40u), ex_ld(pos));
#if QA_CALLS
				XST(E + 0x20u, df);
				M32(E - 4) = E + 0x20u; M32(E - 8) = E + 0x10u;
				c->eax = E + 0x20u; c->ecx = E + 0x20u; c->edx = pos;
				BCALL(E - 12, 0x00710860u, 0x00f13e12u);
				dist = M32(E + 0x10u);
#else
				dist = qa_dot4(df, df).u[0];
#endif
				n = M32(pn);
				for (k = 0; k < n && k < mx; k++)
					if (!(ex_comiss(ex_ldss(od + k * 4u), ex_ss(dist)) & 5)) break; /* (ja) */
				if (k >= mx) continue;
				if (n < mx) BW32(pn, n + 1u);
				for (m = M32(pn) - 1u; m > k; m--) {
					BW32(od + m * 4u, M32(od + m * 4u - 4u));
					BW32(oo + m * 4u, M32(oo + m * 4u - 4u));
				}
				BW32(od + k * 4u, dist);
				BW32(oo + k * 4u, ob);
			}
		}
	}
	c->ecx = M32(0x026e5a84u);
	B_RET(sp, 0);
}

/* 00f0f640: thiscall (Obj **objs, u32 n), ret 8 (frame S = (sp - 4 & ~15) - 0x130): the part visibility of each object
 * o = objs[i] (i < n unsigned) with o->+0xa0 bit 7: its o->+0x2a8 groups (0xc0 bytes each from o->+0x2ac; one group
 * with a part count <= 1 (signed): skipped), each group g's parts e = g->+0x84[l], l < g->+0x80 (00ef5250), up to the
 * first e without e->+0x19 bit 7 or with e->+0xe0 set. Before the object's first such part: 00457e50(this o + 0x70,
 * S+0xd0, o + 0x30) (its box), ctr = Center (0075e920) to S+0xc0, S+0x80 = box max - ctr, 00f14950(this->+0x28 +
 * 0x40, S+0xc0, S+0x80) (result unused). Per part: ctr = Center(e + 0x40) to S+0x60, S+0x10 = e->+0x50 - ctr; v = 1,
 * and 00f13f50(this->+0x28 + 0x40, S+0x60, S+0x10) true: v = this->+0x28->+0x184 byte ? 00f14110(e + 0x40,
 * this->+0x2c, this->+0x28->+0x160, this->+0x28 + 0x140) (al) : 0; e->+0x1a bit 0 = v (byte store). No result
 * (the caller reads no register); ecx = the cookie */
B_FN(00f0f640, "octree: objects' parts visible flags", 0) {
	u32 sp = c->esp, th = c->ecx, ar = M32(sp + 4), cnt = M32(sp + 8), bp = sp - 4, S = (bp & ~15u) - 0x130u, i;
	X128 hb = qa_bc(M32(0x0109b130u)); /* (0075e920's 0.5) */
	(void)hb;
	B_ENTER(bp);
	for (i = 0; i < cnt; i++) {
		u32 ob = M32(ar + i * 4u), n, k, first = 1;
		if ((M32(ob + 0xa0) & 0x80u) != 0x80u) continue;
		n = M32(ob + 0x2a8);
		if (n == 1 && (s32)M32(M32(ob + 0x2ac) + 0x80) <= 1) continue;
		for (k = 0; (s32)k < (s32)n; k++) {
			u32 pk = k * 0xc0u + M32(ob + 0x2ac), m = M32(pk + 0x80), l;
			for (l = 0; l < m; l++) {
				u32 e, w, vis = 1;
				X128 ctr, df;
#if QA_CALLS
				M32(S - 4) = l; c->ecx = pk;
				BCALL(S - 8, 0x00ef5250u, 0x00f0f7f8u);
				e = c->eax;
#else
				e = M32(M32(pk + 0x84) + l * 4u);
#endif
				if (!(M8(e + 0x19) & 0x80u)) break;
				if (M32(e + 0xe0)) break;
				if (first) {
					first = 0;
					M32(S - 4) = ob + 0x30u; M32(S - 8) = S + 0xd0u;
					c->eax = ob + 0x30u; c->ecx = ob + 0x70u; c->edx = 1;
					BCALL(S - 12, 0x00457e50u, 0x00f0f871u);
#if QA_CALLS
					M32(S - 4) = S + 0xc0u; c->ecx = S + 0xd0u; c->edx = S + 0xc0u;
					BCALL(S - 8, 0x0075e920u, 0x00f0f885u);
					ctr = ex_ld(S + 0xc0u);
#else
					ctr = ex_mulps(ex_addps(ex_ld(S + 0xd0u), ex_ld(S + 0xe0u)), hb);
					XST(S + 0xc0u, ctr);
#endif
					df = ex_subps(ex_ld(S + 0xe0u), ctr);
					XST(S + 0x80u, df);
					w = M32(th + 0x28) + 0x40u;
					M32(S - 4) = S + 0x80u; M32(S - 8) = S + 0xc0u; M32(S - 12) = w;
					c->eax = w; c->ecx = S + 0xc0u; c->edx = th;
					BCALL(S - 16, 0x00f14950u, 0x00f0f8f3u);
				}
				w = e + 0x40u;
#if QA_CALLS
				M32(S - 4) = S + 0x60u; c->eax = S + 0x60u; c->ecx = w; c->edx = w;
				BCALL(S - 8, 0x0075e920u, 0x00f0f915u);
				ctr = ex_ld(S + 0x60u);
#else
				ctr = ex_mulps(ex_addps(ex_ld(w), ex_ld(w + 0x10u)), hb);
				XST(S + 0x60u, ctr);
#endif
				df = ex_subps(ex_ld(e + 0x50u), ctr);
				XST(S + 0x10u, df);
				w = M32(th + 0x28) + 0x40u;
				M32(S - 4) = S + 0x10u; M32(S - 8) = S + 0x60u; M32(S - 12) = w;
				c->eax = S + 0x60u; c->ecx = th; c->edx = w;
				BCALL(S - 16, 0x00f13f50u, 0x00f0f96au);
				if (c->eax & 0xffu) {
					u32 t = 0;
					if (M8(M32(th + 0x28) + 0x184)) {
						M32(S - 4) = M32(th + 0x28) + 0x140u;
						M32(S - 8) = M32(M32(th + 0x28) + 0x160);
						M32(S - 12) = M32(th + 0x2c);
						M32(S - 16) = e + 0x40u;
						c->eax = th; c->ecx = M32(th + 0x2c); c->edx = e + 0x40u;
						BCALL(S - 20, 0x00f14110u, 0x00f0f9c8u);
						t = c->eax & 0xffu;
					}
					if (!t) vis = 0;
				}
				BW8(e + 0x1a, (M8(e + 0x1a) & 0xfeu) | vis);
			}
		}
	}
	c->ecx = M32(0x026e5a84u);
	B_RET(sp, 2);
}

/* 00f0ee60: thiscall (Node *nd, u32 mask), ret 8 (frame S = (sp - 4 & ~15) - 0x590): the octree walk collecting
 * visible objects. (nd->+0x70 & mask) != mask: nothing. Else half = (nd->+0x10 - nd->+0) * 0.5 (0109b130) at S+0x500,
 * ctr = Scale(nd->+0 + nd->+0x10, 0.5) (00727710, its float argument the x87 fld / fstp copy) at S+0x4b0;
 * 00f13f50(this->+0x28 + 0x40, S+0x4b0, S+0x500) false: nothing. Else for i < 8: nd->+0x28[i] set: the recursion
 * this->00f0ee60(it, mask); else id = nd->+0x48[i] (-1: none): j = 00f04630 (this nd->+0x20: the first table entry
 * (12 bytes from +0x108, +0x118 of them, unsigned) with +4 == id, else -1); for each following entry (j signed below
 * the count) still with +4 == id: ob = its +0, 00457e50(this ob + 0x70, S+0x470, ob + 0x30), ctr = Center (0075e920)
 * at S+0x460; with (ob->+0xa0 & mask) == mask: S+0x410 = box max - ctr, 00f13f50(this->+0x28 + 0x40, S+0x460,
 * S+0x410) true: v = this->+0x28->+0x184 byte ? 00f14110(S+0x470, this->+0x2c, this->+0x28->+0x160, this->+0x28 +
 * 0x140) (al) : 0; this->+0x28->+0x160 == 0 or !v: ob appended to the list this->+0x28->+0x164 (count +0x168, room
 * +0x16c; full: the error report 0100ea7c(S+8, 0x400, 0x3ff, 0263373c) then [026e4f20](S+8, 2) -- called once per
 * run, so twice in a checked call); v == 1: this->+0x44 += 1. No result (callers read no register); ecx = the
 * cookie (and eax = nd at the early exit) */
#if 0 /* (battle batch 3: 00f0ee60 not selected) */
B_FN(00f0ee60, "octree: collect visible objects", 0) {
	u32 sp = c->esp, th = c->ecx, nd = M32(sp + 4), mk = M32(sp + 8), bp = sp - 4, S = (bp & ~15u) - 0x590u, i, w;
	X128 mn, mx, half, sum, ctr, hb;
	if ((M32(nd + 0x70) & mk) != mk) { c->eax = nd; c->ecx = M32(0x026e5a84u); c->esp = sp; xrc_stdcall_ret(c, 2); return; }
	B_ENTER(bp);
	hb = qa_bc(M32(0x0109b130u));
	mn = ex_ld(nd); mx = ex_ld(nd + 0x10u);
	half = ex_mulps(ex_subps(mx, mn), hb);
	sum = ex_addps(mn, mx);
#if QA_CALLS
	XST(S + 0x4c0u, sum);
	M32(S - 4) = M32(0x0109b130u); M32(S - 8) = S + 0x4b0u;
	c->eax = S + 0x4b0u; c->ecx = S + 0x4c0u; c->edx = nd;
	BCALL(S - 12, 0x00727710u, 0x00f0ef90u);
	ctr = ex_ld(S + 0x4b0u);
#else
	ctr = ex_mulps(sum, qa_bc(M32(0x0109b130u)));
	XST(S + 0x4b0u, ctr);
#endif
	XST(S + 0x500u, half);
	w = M32(th + 0x28) + 0x40u;
	M32(S - 4) = S + 0x500u; M32(S - 8) = S + 0x4b0u; M32(S - 12) = w;
	c->eax = th; c->ecx = w; c->edx = S + 0x4b0u;
	BCALL(S - 16, 0x00f13f50u, 0x00f0efb3u);
	if (c->eax & 0xffu) for (i = 0; i < 8; i++) {
		u32 ch = M32(nd + 0x28u + i * 4u), id, tb, j;
		if (ch) {
			M32(S - 4) = mk; M32(S - 8) = ch;
			c->eax = ch; c->ecx = th; c->edx = nd;
			BCALL(S - 12, 0x00f0ee60u, 0x00f0f01du);
			continue;
		}
		id = M32(nd + 0x48u + i * 4u);
		if (id == 0xffffffffu) continue;
		tb = M32(nd + 0x20);
#if QA_CALLS
		M32(S - 4) = id;
		c->eax = id; c->ecx = tb; c->edx = id;
		BCALL(S - 8, 0x00f04630u, 0x00f0f05fu);
		j = c->eax;
#else
		for (j = 0;;) {
			if (M32(M32(tb + 0x108) + j * 0xcu + 4u) == id) break;
			if (++j >= M32(tb + 0x118)) { j = 0xffffffffu; break; }
		}
#endif
		for (;; j++) {
			u32 t = M32(nd + 0x20), ent, ob, vis;
			X128 df;
			ent = (s32)j < (s32)M32(t + 0x118) ? j * 0xcu + M32(t + 0x108) : 0;
			if (!ent || M32(ent + 4) != id) break;
			ob = M32(ent);
			M32(S - 4) = ob + 0x30u; M32(S - 8) = S + 0x470u;
			c->eax = ob + 0x30u; c->ecx = ob + 0x70u; c->edx = ob;
			BCALL(S - 12, 0x00457e50u, 0x00f0f108u);
#if QA_CALLS
			M32(S - 4) = S + 0x460u;
			c->ecx = S + 0x470u; c->edx = S + 0x460u;
			BCALL(S - 8, 0x0075e920u, 0x00f0f11cu);
			ctr = ex_ld(S + 0x460u);
#else
			ctr = ex_mulps(ex_addps(ex_ld(S + 0x470u), ex_ld(S + 0x480u)), hb);
#endif
			if ((M32(ob + 0xa0) & mk) != mk) continue;
			df = ex_subps(ex_ld(S + 0x480u), ctr);
			XST(S + 0x460u, ctr); XST(S + 0x410u, df);
			w = M32(th + 0x28) + 0x40u;
			M32(S - 4) = S + 0x410u; M32(S - 8) = S + 0x460u; M32(S - 12) = w;
			c->eax = w; c->ecx = S + 0x460u; c->edx = th;
			BCALL(S - 16, 0x00f13f50u, 0x00f0f1b1u);
			if (!(c->eax & 0xffu)) continue;
			vis = 0;
			if (M8(M32(th + 0x28) + 0x184)) {
				M32(S - 4) = M32(th + 0x28) + 0x140u;
				M32(S - 8) = M32(M32(th + 0x28) + 0x160);
				M32(S - 12) = M32(th + 0x2c);
				M32(S - 16) = S + 0x470u;
				c->eax = S + 0x470u; c->ecx = th; c->edx = M32(th + 0x2c);
				BCALL(S - 20, 0x00f14110u, 0x00f0f215u);
				vis = c->eax & 0xffu;
			}
			if (!M32(M32(th + 0x28) + 0x160) || !vis) {
				u32 wa = M32(th + 0x28), wb = M32(th + 0x28), cn = M32(wb + 0x168);
				if (cn < M32(wa + 0x16c)) {
					wa = M32(th + 0x28); wb = M32(th + 0x28); cn = M32(wb + 0x168);
					BW32(M32(wa + 0x164) + cn * 4u, ob);
					wa = M32(th + 0x28); cn = M32(wa + 0x168) + 1u;
					BW32(M32(th + 0x28) + 0x168u, cn);
				} else {
					M8(S + 0x407u) = 0;
					M32(S - 4) = 0x0263373cu; M32(S - 8) = 0x3ffu; M32(S - 12) = 0x400u; M32(S - 16) = S + 8u;
					c->eax = th; c->ecx = S + 8u; c->edx = cn;
					BCALL(S - 20, 0x0100ea7cu, 0x00f0f2d4u);
					M32(S - 4) = 2; M32(S - 8) = S + 8u;
					c->edx = S + 8u;
					BCALL(S - 12, M32(0x026e4f20u), 0x00f0f2e4u);
				}
			}
			if (vis == 1) BW32(th + 0x44, M32(th + 0x44) + 1u);
		}
	}
	c->ecx = M32(0x026e5a84u);
	B_RET(sp, 2);
}
#endif

/* 00f115b0: cdecl (Out *o, const Box *b, const Vec4 *p), ret (frame S = (sp - 4 & ~15) - 0x1850 (_chkstk 0100e590:
 * only probes the stack, not called)): the culling volume of the oriented box b (rows R0..R2 the half axes, R3 the
 * centre) seen from p. o->+0x90 = b->+0x60 (the kind). C = R3 - *p. Kind 0 (a quad: axes R0, R1, normal R2): the 4
 * edge planes n = Cross of its corners ((C -+ R0) -+ R1, Cross 00458990), T = Transpose{n4, n3, n2, n1} (00728100),
 * Dot4(R2, C) > [025a3c20] (comiss ja): R2 and T negated (0 - x, 00f14f90); o->+0 .. +0x20 = T rows 0..2, +0x30 ..
 * +0x50 = Abs(T) rows 0..2 (maxps(0 - r, r), 00db4dd0), +0x60 = (+-)R2, +0x70 = C. Kind 1 (a box): sel = the sign
 * bits 0..2 of 0 - Matrix33{Transpose{R0, R1, R2, 0}} * C (00711f40, movmskps), the 8 corners K = ((C +- R0) +- R1)
 * +- R2, its local tables (T7[8][7], T8[8], F8[8] in 0 / 2) pick the silhouette: Q0 = K[T8], A / B / Cc = Cross(K of
 * T7[0] / T7[F + 2] / T7[4 - F], Q0) (negated with F), X / Y / Z = R0 / R1 / R2 negated by sel's bits; six Dot4 sign
 * tests (A.Z, B.Z, Cc.X, A.X, B.Y, Cc.Y against [025a3c20], comiss jb) choose 4 planes (two pairs: three of them
 * Cross(Q0, ..), Cross(.., Q0) and two edges) or 6 (the hexagon's edges, one corner replaced by T8 under some tests)
 * and which of X / Y / Z are replaced by another; o->+0 .. +0x20 = Transpose{planes 0..2, 0} rows 0..2, +0x30 ..
 * +0x50 = Transpose{planes 3..5, 0} rows 0..2, +0x60 .. +0x80 = Transpose{X, Y, Z, 0} rows 0..2 (00c28db0 +
 * 00728100), +0x94 / 98 / 9c = Q0.xyz. Other kinds: only +0x90. All helper calls computed here (QA_CALLS 0). No
 * result (the caller reads no register); ecx = the cookie */
#if 0 /* (battle batch 3: 00f115b0 not selected) */
B_FN(00f115b0, "oriented box: culling planes from a point", 0) {
	static const u8 t7[8][7] = {{2, 3, 7, 4, 5, 1, 0}, {3, 0, 4, 5, 6, 2, 0}, {6, 5, 1, 0, 3, 7, 0}, {7, 6, 2, 1, 0, 4, 0},
	                            {1, 2, 6, 7, 4, 0, 0}, {0, 1, 5, 6, 7, 3, 0}, {5, 4, 0, 3, 2, 6, 0}, {4, 7, 3, 2, 1, 5, 0}};
	static const u8 t8[8] = {6, 7, 2, 3, 5, 4, 1, 0}, f8[8] = {0, 2, 2, 0, 2, 0, 0, 2};
	u32 sp = c->esp, o = M32(sp + 4), src = M32(sp + 8), pp = M32(sp + 12), bp = sp - 4, S = (bp & ~15u) - 0x1850u;
	u32 kind = M32(src + 0x60), kz = M32(0x025a3c20u);
	X128 R0, R1, R2, R3, C, z = qa_zero();
	(void)S;
#if QA_CALLS
	B_ENTER(bp);
#endif
	BW32(o + 0x90, kind);
#if QA_CALLS
	M32(S - 4) = src; c->ecx = S + 0x1800u;
	BCALL(S - 8, 0x007a8b50u, 0x00f115fbu);
	R0 = ex_ld(S + 0x1800u); R1 = ex_ld(S + 0x1810u); R2 = ex_ld(S + 0x1820u); R3 = ex_ld(S + 0x1830u);
#else
	R0 = ex_ld(src); R1 = ex_ld(src + 0x10u); R2 = ex_ld(src + 0x20u); R3 = ex_ld(src + 0x30u);
#endif
	C = ex_subps(R3, ex_ld(pp));
	if (kind == 0) {
		X128 E2 = R2, v4, v6, v8, v10, n1, n2, n3, n4, d, T[4], Ab[4];
		int k;
		v4 = ex_subps(ex_subps(C, R0), R1);
		v6 = ex_subps(ex_addps(C, R0), R1);
		v8 = ex_addps(ex_addps(C, R0), R1);
		v10 = ex_addps(ex_subps(C, R0), R1);
		n1 = QA_CROSS(S - 12, 0x00f11923u, S + 0x1570u, v10, S + 0x1560u, S + 0x16f0u, v4);
		n2 = QA_CROSS(S - 12, 0x00f11946u, S + 0x15f0u, v8, S + 0x1540u, S + 0x1570u, v10);
		n3 = QA_CROSS(S - 12, 0x00f11969u, S + 0x1670u, v6, S + 0x1520u, S + 0x15f0u, v8);
		n4 = QA_CROSS(S - 12, 0x00f1198cu, S + 0x16f0u, v4, S + 0x1500u, S + 0x1670u, v6);
		QA_TR(S - 8, 0x00f11a1fu, S + 0x14b0u, n4, n3, n2, n1, S + 0x1460u, T);
		d = QA_DOT4(S - 12, 0x00f11a3bu, S + 0x1770u, E2, S + 0x1450u, S + 0x17c0u, C);
		if (!(ex_comiss(d, ex_ss(kz)) & 5)) { /* (ja) */
			E2 = ex_subps(z, E2);
#if QA_CALLS
			M32(S - 4) = S + 0x1380u; c->ecx = S + 0x1460u;
			BCALL(S - 8, 0x00f14f90u, 0x00f11b20u);
			for (k = 0; k < 4; k++) { T[k] = ex_ld(S + 0x1380u + 16u * (u32)k); XST(S + 0x1460u + 16u * (u32)k, T[k]); }
#else
			for (k = 0; k < 4; k++) T[k] = ex_subps(z, T[k]);
#endif
		}
#if QA_CALLS
		M32(S - 4) = S + 0x1330u; c->ecx = S + 0x1460u;
		BCALL(S - 8, 0x00db4dd0u, 0x00f11bb6u);
		for (k = 0; k < 4; k++) Ab[k] = ex_ld(S + 0x1330u + 16u * (u32)k);
#else
		for (k = 0; k < 4; k++) Ab[k] = ex_maxps(ex_subps(z, T[k]), T[k]);
#endif
		BW128(o, T[0]); BW128(o + 0x10, T[1]); BW128(o + 0x20, T[2]);
		BW128(o + 0x30, Ab[0]); BW128(o + 0x40, Ab[1]); BW128(o + 0x50, Ab[2]);
		BW128(o + 0x60, E2); BW128(o + 0x70, C);
	} else if (kind == 1) {
		X128 Mt[4], K[8], Q0, Q1, Q2, Q3, A, B, Cc, X, Y, Z, d, a0, a1, a2, b0, b1, b2, ta[4], tb[4], tc[4];
		u32 sel, F, f1, f2, f3, f4, f5, f6;
		QA_TR(S - 8, 0x00f11d41u, S + 0x12b0u, R0, R1, R2, z, S + 0x1260u, Mt);
		sel = qa_movmsk(ex_subps(z, QA_M33(S - 12, 0x00f11d5du, S + 0x1260u, Mt, S + 0x1250u, S + 0x17c0u, C))) & 7u;
		K[0] = ex_subps(ex_subps(ex_subps(C, R0), R1), R2);
		K[1] = ex_subps(ex_subps(ex_addps(C, R0), R1), R2);
		K[2] = ex_addps(ex_subps(ex_addps(C, R0), R1), R2);
		K[3] = ex_addps(ex_subps(ex_subps(C, R0), R1), R2);
		K[4] = ex_subps(ex_addps(ex_subps(C, R0), R1), R2);
		K[5] = ex_subps(ex_addps(ex_addps(C, R0), R1), R2);
		K[6] = ex_addps(ex_addps(ex_addps(C, R0), R1), R2);
		K[7] = ex_addps(ex_addps(ex_subps(C, R0), R1), R2);
		F = f8[sel];
		Q0 = K[t8[sel]]; Q1 = K[t7[sel][0]]; Q2 = K[t7[sel][F + 2]]; Q3 = K[t7[sel][4 - F]];
		A = QA_CROSS(S - 12, 0x00f12a21u, S + 0x990u, Q1, S + 0x940u, S + 0x9b0u, Q0);
		B = QA_CROSS(S - 12, 0x00f12a3du, S + 0x970u, Q2, S + 0x930u, S + 0x9b0u, Q0);
		Cc = QA_CROSS(S - 12, 0x00f12a59u, S + 0x950u, Q3, S + 0x920u, S + 0x9b0u, Q0);
		X = (sel & 1u) ? ex_subps(z, R0) : R0;
		Y = (sel & 2u) ? ex_subps(z, R1) : R1;
		Z = (sel & 4u) ? ex_subps(z, R2) : R2;
		if (F) { A = ex_subps(z, A); B = ex_subps(z, B); Cc = ex_subps(z, Cc); }
		/* (jb: CF clear = not below, ordered) */
		d = QA_DOT4(S - 12, 0x00f12d3cu, S + 0x940u, A, S + 0x730u, S + 0x800u, Z);
		f1 = !(ex_comiss(d, ex_ss(kz)) & 4);
		d = QA_DOT4(S - 12, 0x00f12dd8u, S + 0x930u, B, S + 0x6d0u, S + 0x800u, Z);
		f2 = !(ex_comiss(ex_ss(kz), d) & 4);
		d = QA_DOT4(S - 12, 0x00f12e74u, S + 0x920u, Cc, S + 0x670u, S + 0x8c0u, X);
		f3 = !(ex_comiss(d, ex_ss(kz)) & 4);
		d = QA_DOT4(S - 12, 0x00f12f10u, S + 0x940u, A, S + 0x610u, S + 0x8c0u, X);
		f4 = !(ex_comiss(ex_ss(kz), d) & 4);
		d = QA_DOT4(S - 12, 0x00f12facu, S + 0x930u, B, S + 0x5b0u, S + 0x860u, Y);
		f5 = !(ex_comiss(d, ex_ss(kz)) & 4);
		d = QA_DOT4(S - 12, 0x00f13048u, S + 0x920u, Cc, S + 0x550u, S + 0x860u, Y);
		f6 = !(ex_comiss(ex_ss(kz), d) & 4);
		if (f1 & f2) {
			u32 i0 = t7[sel][2 * F], i1 = t7[sel][2 * F + 1], i2 = t7[sel][2 - F];
			X = Z; Y = Z;
			a0 = QA_CROSS(S - 12, 0x00f131a7u, S + 0x9b0u, Q0, S + 0x4e0u, S + 0x1160u + 16u * i2, K[i2]);
			a2 = QA_CROSS(S - 12, 0x00f131f6u, S + 0x1160u + 16u * i0, K[i0], S + 0x4a0u, S + 0x9b0u, Q0);
			b1 = QA_CROSS(S - 12, 0x00f1324fu, S + 0x1160u + 16u * i1, K[i1], S + 0x460u, S + 0x1160u + 16u * i0, K[i0]);
			b2 = QA_CROSS(S - 12, 0x00f13298u, S + 0x1160u + 16u * i2, K[i2], S + 0x430u, S + 0x1160u + 16u * i1, K[i1]);
			a1 = a0; b0 = a2;
		} else if (f3 & f4) {
			u32 j0 = t7[sel][4 - 2 * F], j1 = t7[sel][5 - 2 * F], j2 = t7[sel][F];
			Y = X; Z = X;
			a0 = QA_CROSS(S - 12, 0x00f1339du, S + 0x9b0u, Q0, S + 0x3f0u, S + 0x1160u + 16u * j2, K[j2]);
			a2 = QA_CROSS(S - 12, 0x00f133ecu, S + 0x1160u + 16u * j0, K[j0], S + 0x3d0u, S + 0x9b0u, Q0);
			b1 = QA_CROSS(S - 12, 0x00f13445u, S + 0x1160u + 16u * j1, K[j1], S + 0x3b0u, S + 0x1160u + 16u * j0, K[j0]);
			b2 = QA_CROSS(S - 12, 0x00f1348eu, S + 0x1160u + 16u * j2, K[j2], S + 0x390u, S + 0x1160u + 16u * j1, K[j1]);
			a1 = a0; b0 = a2;
		} else if (f5 & f6) {
			u32 k0 = t7[sel][2], k1 = t7[sel][3], k2 = t7[sel][4];
			X = Y; Z = Y;
			a0 = QA_CROSS(S - 12, 0x00f1354eu, S + 0x9b0u, Q0, S + 0x370u, S + 0x1160u + 16u * k2, K[k2]);
			a2 = QA_CROSS(S - 12, 0x00f1359du, S + 0x1160u + 16u * k0, K[k0], S + 0x350u, S + 0x9b0u, Q0);
			b1 = QA_CROSS(S - 12, 0x00f135f6u, S + 0x1160u + 16u * k1, K[k1], S + 0x330u, S + 0x1160u + 16u * k0, K[k0]);
			b2 = QA_CROSS(S - 12, 0x00f1363fu, S + 0x1160u + 16u * k2, K[k2], S + 0x310u, S + 0x1160u + 16u * k1, K[k1]);
			a1 = a0; b0 = a2;
		} else {
			u32 m0 = t7[sel][0], m1 = t7[sel][1], m2 = t7[sel][2], m3 = t7[sel][3], m4 = t7[sel][4], m5 = t7[sel][5];
			if (f1 | f2 | f3 | f4 | f5 | f6) {
				if (f1 | f6) { X = Y; if (F) m1 = t8[sel]; else m5 = t8[sel]; }
				else if (f2 | f3) { Y = Z; m3 = t8[sel]; }
				else { Z = X; if (F) m5 = t8[sel]; else m1 = t8[sel]; }
			}
			a0 = QA_CROSS(S - 12, 0x00f13833u, S + 0x1160u + 16u * m1, K[m1], S + 0x2e0u, S + 0x1160u + 16u * m0, K[m0]);
			a1 = QA_CROSS(S - 12, 0x00f1387cu, S + 0x1160u + 16u * m2, K[m2], S + 0x2c0u, S + 0x1160u + 16u * m1, K[m1]);
			a2 = QA_CROSS(S - 12, 0x00f138c5u, S + 0x1160u + 16u * m3, K[m3], S + 0x2a0u, S + 0x1160u + 16u * m2, K[m2]);
			b0 = QA_CROSS(S - 12, 0x00f1390eu, S + 0x1160u + 16u * m4, K[m4], S + 0x280u, S + 0x1160u + 16u * m3, K[m3]);
			b1 = QA_CROSS(S - 12, 0x00f13957u, S + 0x1160u + 16u * m5, K[m5], S + 0x260u, S + 0x1160u + 16u * m4, K[m4]);
			b2 = QA_CROSS(S - 12, 0x00f139a0u, S + 0x1160u + 16u * m0, K[m0], S + 0x240u, S + 0x1160u + 16u * m5, K[m5]);
		}
		QA_TR(S - 8, 0x00f13a58u, S + 0x1d0u, a0, a1, a2, z, S + 0x180u, ta);
		QA_TR(S - 8, 0x00f13af7u, S + 0x120u, b0, b1, b2, z, S + 0xd0u, tb);
#if QA_CALLS
		XST(S + 0x8c0u, X); XST(S + 0x860u, Y); XST(S + 0x800u, Z); XST(S + 0xb0u, z);
		M32(S - 4) = S + 0x70u; M32(S - 8) = S + 0xb0u; M32(S - 12) = S + 0x800u; M32(S - 16) = S + 0x860u; M32(S - 20) = S + 0x8c0u;
		c->ecx = S + 0x30u;
		BCALL(S - 24, 0x00c28db0u, 0x00f13b40u);
		c->ecx = c->eax;
		BCALL(S - 8, 0x00728100u, 0x00f13b47u);
		tc[0] = ex_ld(S + 0x70u); tc[1] = ex_ld(S + 0x80u); tc[2] = ex_ld(S + 0x90u);
#else
		qa_tr(X, Y, Z, z, tc);
#endif
		BW128(o, ta[0]); BW128(o + 0x10, ta[1]); BW128(o + 0x20, ta[2]);
		BW128(o + 0x30, tb[0]); BW128(o + 0x40, tb[1]); BW128(o + 0x50, tb[2]);
		BW128(o + 0x60, tc[0]); BW128(o + 0x70, tc[1]); BW128(o + 0x80, tc[2]);
		BW32(o + 0x94, Q0.u[0]); BW32(o + 0x98, Q0.u[1]); BW32(o + 0x9c, Q0.u[2]);
	}
	c->ecx = M32(0x026e5a84u);
#if QA_CALLS
	B_RET(sp, 0);
#else
	c->esp = sp; xrc_stdcall_ret(c, 0);
#endif
}
#endif
/* ---- battle batch 3, part B: 00f85b20, 00f9a340, 00f98b70, 00f314c0
 *
 * QB_LEAF_CALLS 1: the small leaf callees (the /Od Vec4 helpers 00515bb0 set, 00455880 get, 00455840 operator=,
 * 00456c90 * float, 00456d50 +=, 00456e50 *= float, 00572d80 /= float, 00ead960 / 00796ae0 constant vectors, 00ee2270
 * the light colour getter, the two devirtualised getters 00ef19f0 / 00ef1a10) are called as the original does (BCALL,
 * their arguments and the locals they read written into the frame at the original's addresses). 0 (default):
 * computed here, as the translation itself does for most of them (it inlines 00515bb0, 00ee2270, 00456c90, 00ead960
 * and devirtualises 00ef19f0 / 00ef1a10): their only effects are their out vector (in our frame) and eax, so nothing
 * a caller or the checker sees differs (their own dead frames below esp are not written). The same switch covers
 * 00f98b70's key helpers (00f98a90 / 00f98b00 group pointers, 00fa0cc0 / 00f98840 last-key value; the translation
 * inlines 00f98b00) and 00f314c0's render-state wrappers (00eac240 / 00eac230 getters, 00eacc90, 00eabdb0, 00eabdf0,
 * 00eabeb0, 00eabf60, 00eacbc0, 00ead120, the empty 00f76190: all inlined by the translation; with 0 the device
 * SetRenderState calls and 00eabbc0 / 00eac6a0 inside them are BCALLed directly, with their globals' stores). */
#ifndef QB_LEAF_CALLS
#define QB_LEAF_CALLS 0
#endif

static inline X128 qb_set1(u32 b) { X128 r; r.vi = _mm_set1_epi32((int)b); return r; }
static inline X128 qb_set4(u32 a, u32 b, u32 c_, u32 d) { X128 r; r.vi = _mm_set_epi32((int)d, (int)c_, (int)b, (int)a); return r; }
/* ucomiss a, b; lahf; test ah, 0x44; jp: taken when not equal or unordered (the /Od "!=") */
static inline int qb_ne(u32 a, u32 b) { return !b_feq(a, b); }
/* four dwords to a guest address (journaled), in order */
#define QB_W4(a_, v0_, v1_, v2_, v3_) do { u32 qa_ = (a_); BW32(qa_, (v0_)); BW32(qa_ + 4, (v1_)); BW32(qa_ + 8, (v2_)); BW32(qa_ + 12, (v3_)); } while (0)
/* a 16-byte copy as mov pairs (read a dword, write it, next) */
#define QB_CP4(d_, s_) do { u32 qd_ = (d_), qs_ = (s_); BW32(qd_, M32(qs_)); BW32(qd_ + 4, M32(qs_ + 4)); BW32(qd_ + 8, M32(qs_ + 8)); BW32(qd_ + 12, M32(qs_ + 12)); } while (0)

/* 00456c90: thiscall Vec4 *(out, float f), ret 8: out = *this * {f, f, f, f} (mulps this, f); eax = out */
static inline X128 qb_vscale(XCtx *c, u32 v, u32 f, u32 out, u32 at, u32 ra) {
#if QB_LEAF_CALLS
	M32(at + 4) = out; M32(at + 8) = f; c->ecx = v;
	BCALL(at, 0x00456c90u, ra);
	return ex_ld(c->eax);
#else
	(void)c; (void)out; (void)at; (void)ra;
	return ex_mulps(ex_ld(v), qb_set1(f));
#endif
}
/* a virtual getter o->vtbl[slot / 4]() (thiscall, eax = the method, ecx = o, edx = the vtable at the call); 00ef19f0
 * (o->+0xd4->+0x10) and 00ef1a10 (o->+0xd4 + 0x14) done here, as the translation devirtualises them */
static inline u32 qb_vget(XCtx *c, u32 o, u32 slot, u32 at, u32 ra) {
	u32 vt = M32(o), m = M32(vt + slot);
#if !QB_LEAF_CALLS
	if (m == 0x00ef19f0u) return M32(M32(o + 0xd4) + 0x10);
	if (m == 0x00ef1a10u) return M32(o + 0xd4) + 0x14;
#endif
	c->eax = m; c->ecx = o; c->edx = vt;
	BCALL(at, m, ra);
	return c->eax;
}
/* 00ee2270: thiscall Vec4 *(out), ret 4: out = {o->+0x1b8, +0x1bc, +0x1c0, clamp(o->+0x1c4 + o->+0x3a0, 0, 1)} (addss;
 * comiss 0 > t: 0, t > 1: 1, else t; constants 025a3c20 / 010aa950); eax = out */
static inline void qb_ee2270(XCtx *c, u32 o, u32 out, u32 at, u32 ra, u32 *r) {
#if QB_LEAF_CALLS
	M32(at + 4) = out; c->ecx = o;
	BCALL(at, 0x00ee2270u, ra);
	r[0] = M32(out); r[1] = M32(out + 4); r[2] = M32(out + 8); r[3] = M32(out + 12);
#else
	X128 t = ex_addss(ex_ldss(o + 0x1c4), ex_ldss(o + 0x3a0));
	u32 z = M32(0x025a3c20u), one = M32(0x010aa950u);
	(void)c; (void)out; (void)at; (void)ra;
	r[0] = M32(o + 0x1b8); r[1] = M32(o + 0x1bc); r[2] = M32(o + 0x1c0);
	if (!(ex_comiss(ex_ss(z), t) & 5)) r[3] = z;
	else if (!(ex_comiss(t, ex_ss(one)) & 5)) r[3] = one;
	else r[3] = t.u[0];
#endif
}

/* 00f85b20: thiscall (A, B), ret 8 (virtual: VertexLightShaderDriver::ApplyFixedLightParam, vtable 0265cf90 slot 2;
 * /Od, aligned frame S = (entry esp - 4 & ~15) - 0x2b0, ebp frame). dirty = [026e52a1]; A->+0x8fc: A->+0x904 = 0.
 * this->+0x13c = (A->+0x866 == 1 ? 1 : 0) if dirty or it differs (ucomiss !=: unordered differs), then dirty. V1 =
 * {1 / d, (-A->+0x17c) / d, A->+0x178, 1} (d = A->+0x180 - A->+0x17c, |d| below FLT_MIN [010c4360]: FLT_MIN; 00515bb0)
 * -> this->+0x100 (x, y, z compared); with A->+0x38c or +0x38d the colour 00ee2270(A) -> this->+0x114 (4 compared);
 * A->+0x878 * A->+0x888 (mulps, through 00455880 / 00455840: the x87 copy) -> this->+0x128 (4 compared). Dirty: the
 * pixel shader constants 00c27cf0(i, data) (thiscall, ecx = [028670bc] + 8, ret 8) for this->+0xfc (+0x100), +0x110
 * (+0x114, only with A->+0x38c / 38d), +0x124 (+0x128), +0x138 ({+0x13c x 4}, a local at S+0x154). Then this->+8 = 0,
 * acc = this->+0x1c; the lights (L->+0xfc must be 'DNTL' 0x4c544e44, else the engine assertion, lines 0x2ef / 0x310):
 * without A->+0x640 bit 4 the groups of A->+0x164 (its vtbl[4]() count, vtbl[5]() array; group G: G->+4 .. G->+8);
 * then the 36-byte slots of Q = A->+0x1b4 (Q->+0x14a of them, bit masks Q->+0x148 / +0x149) -- each while this->+8 <
 * 4: type (L->+0xdc) 4 (ambient): groups acc = L->+0xbc * L->+0xcc (00456c90; any = 1), slots acc += that
 * (00456d50), cnt++; else (slots with bit 4: only with its 0x149 bit) dirty if this->+0xc[this->+8] != L, and
 * (groups always, slots without its 0x148 bit) this->+0xc[this->+8++] = L. Not dirty: acc.xyz vs this->+0x1c (after
 * the groups), Q's xyz vs it (after the slots). A->+0x8fc: A->+0x904 = this->+8, not dirty: this->00f86fe0(A, 0) and
 * A->+0xb68 = this->+0x1c. Not dirty: done. Else this->+0x1c = Q's 16 bytes (no Q: 00ead960's {0, 0, 0, 1}); any:
 * acc /= cnt (00572d80), this->+0x1c += acc, *= 0.5 (00456d50, 00456e50); bit 4 and A->+0x640 bit 5: this->+0x1c =
 * 00796ae0's {1, 1, 1, 1}; this->+0x144 and +0x14c both 0: this->00f87990(A, B), done; else this->00f86fe0(A, 1),
 * A->+0x8fc: A->+0xb68 = this->+0x1c; this->+0x140: dev->SetPixelShaderConstantF(reg, this + 0x1c, 1), this->+0x144:
 * the same with {(float)this->+8 x 4} (a local at S+8) -- dev = [0283eac0] (its vtbl + 0x1b4), reg = ([028670bc]->+8)
 * [i * 2 - 1]. No result (a void virtual); the /GS cookie store and check skipped. Callees: setters (shader
 * constants), getters; 00f86fe0 / 00f87990 set constants from this and A (repeatable). */
#if 0 /* (battle batch 3: 00f85b20 not selected) */
B_FN(00f85b20, "fixed light params: apply", 0) {
	u32 sp = c->esp, th = c->ecx, A = M32(sp + 4), Bp = M32(sp + 8), bp = sp - 4, S = (bp & ~15u) - 0x2b0u;
	u32 dirty, fv, fl, Q, cnt = 0, any = 0, k, v1[4], v2[4], vc[4];
	X128 acc;
	B_ENTER(bp);
	dirty = M8(0x026e52a1u);
	if (M8(A + 0x8fc)) BW32(A + 0x904, 0);
	fv = M8(A + 0x866) == 1 ? M32(0x010aa950u) : M32(0x025a3c20u);
	if (dirty || qb_ne(M32(th + 0x13c), fv)) { dirty = 1; BW32(th + 0x13c, fv); }
	{
		X128 d = ex_subss(ex_ldss(A + 0x180), ex_ldss(A + 0x17c)), inv, nb;
		u32 eps = M32(0x010c4360u);
		if (!(ex_comiss(ex_ss(eps), ex_ss(d.u[0] & M32(0x0109b220u))) & 5)) d = ex_ss(eps); /* (not jbe) */
		inv = ex_divss(ex_ldss(0x010aa950u), d);
		nb = ex_mulss(ex_ss(M32(A + 0x17c) ^ M32(0x0109c600u)), inv);
		v1[0] = inv.u[0]; v1[1] = nb.u[0]; v1[2] = M32(A + 0x178); v1[3] = 0x3f800000u; /* (fld1) */
#if QB_LEAF_CALLS
		M32(S - 16) = v1[0]; M32(S - 12) = v1[1]; M32(S - 8) = v1[2]; M32(S - 4) = v1[3];
		c->ecx = S + 0x26cu;
		BCALL(S - 20, 0x00515bb0u, 0x00f85d28u);
		v1[0] = M32(S + 0x26cu); v1[1] = M32(S + 0x270u); v1[2] = M32(S + 0x274u); v1[3] = M32(S + 0x278u);
#endif
	}
	if (dirty || qb_ne(M32(th + 0x100), v1[0]) || qb_ne(M32(th + 0x104), v1[1]) || qb_ne(M32(th + 0x108), v1[2])) {
		QB_W4(th + 0x100, v1[0], v1[1], v1[2], v1[3]);
		dirty = 1;
	}
	if (M8(A + 0x38c) || M8(A + 0x38d)) {
		qb_ee2270(c, A, S + 0x240u, S - 8, 0x00f85e82u, v2);
		if (dirty || qb_ne(M32(th + 0x114), v2[0]) || qb_ne(M32(th + 0x118), v2[1]) || qb_ne(M32(th + 0x11c), v2[2]) || qb_ne(M32(th + 0x120), v2[3])) {
			QB_W4(th + 0x114, v2[0], v2[1], v2[2], v2[3]);
			dirty = 1;
		}
	}
	{
		X128 cv = ex_ld(A + 0x878), dv, pr;
#if QB_LEAF_CALLS
		XST(S + 0x210u, cv);
		M32(S - 4) = S + 0x200u; c->ecx = A + 0x888u;
		BCALL(S - 8, 0x00455880u, 0x00f8601cu);
		dv = ex_ld(c->eax);
		M32(S - 4) = S + 0x1f0u; c->ecx = S + 0x210u;
		BCALL(S - 8, 0x00455880u, 0x00f8603bu);
		cv = ex_ld(c->eax);
		pr = ex_mulps(cv, dv);
		XST(S + 0x1d0u, pr);
		M32(S - 4) = S + 0x1d0u; c->ecx = S + 0x1c0u;
		BCALL(S - 8, 0x00455840u, 0x00f86065u);
		vc[0] = M32(S + 0x1c0u); vc[1] = M32(S + 0x1c4u); vc[2] = M32(S + 0x1c8u); vc[3] = M32(S + 0x1ccu);
#else
		dv = ex_ld(A + 0x888);
		pr = ex_mulps(cv, dv);
		vc[0] = x87_copy(pr.u[0]); vc[1] = x87_copy(pr.u[1]); vc[2] = x87_copy(pr.u[2]); vc[3] = x87_copy(pr.u[3]);
#endif
	}
	if (dirty || qb_ne(M32(th + 0x128), vc[0]) || qb_ne(M32(th + 0x12c), vc[1]) || qb_ne(M32(th + 0x130), vc[2]) || qb_ne(M32(th + 0x134), vc[3])) {
		QB_W4(th + 0x128, vc[0], vc[1], vc[2], vc[3]);
		dirty = 1;
	}
	if (dirty) {
		u32 p, g;
		if ((p = M32(th + 0xfc)) != 0) {
			g = M32(0x028670bcu);
			M32(S - 4) = th + 0x100u; M32(S - 8) = p;
			c->eax = th + 0x100u; c->ecx = g + 8; c->edx = g;
			BCALL(S - 12, 0x00c27cf0u, 0x00f8629bu);
		}
		if (M32(th + 0x110) && (M8(A + 0x38c) || M8(A + 0x38d))) {
			p = M32(th + 0x110); g = M32(0x028670bcu);
			M32(S - 4) = th + 0x114u; M32(S - 8) = p;
			c->eax = th + 0x114u; c->ecx = g + 8; c->edx = g;
			BCALL(S - 12, 0x00c27cf0u, 0x00f86336u);
		}
		if ((p = M32(th + 0x124)) != 0) {
			g = M32(0x028670bcu);
			M32(S - 4) = th + 0x128u; M32(S - 8) = p;
			c->eax = th + 0x128u; c->ecx = g + 8; c->edx = g;
			BCALL(S - 12, 0x00c27cf0u, 0x00f86395u);
		}
		if (M32(th + 0x138)) {
			XST(S + 0x154u, qb_set4(M32(th + 0x13c), M32(th + 0x13c), M32(th + 0x13c), M32(th + 0x13c)));
			p = M32(th + 0x138); g = M32(0x028670bcu);
			M32(S - 4) = S + 0x154u; M32(S - 8) = p;
			c->eax = g; c->ecx = g + 8; c->edx = p;
			BCALL(S - 12, 0x00c27cf0u, 0x00f8649au);
		}
	}
	BW32(th + 8, 0);
	acc = ex_ld(th + 0x1c);
#if QB_LEAF_CALLS
	XST(S + 0x13cu, acc);
#endif
	fl = (M8(A + 0x640) >> 4) & 1u;
	if (!fl) {
		u32 na, arr, i;
		na = qb_vget(c, M32(A + 0x164), 0x10, S - 4, 0x00f86550u);
		arr = qb_vget(c, M32(A + 0x164), 0x14, S - 4, 0x00f8658au);
		for (i = 0; (s32)i < (s32)na; i++) {
			u32 G = M32(arr + i * 4u), m = 0, j;
			if (M32(G + 4)) m = (u32)((s32)(M32(G + 8) - M32(G + 4)) >> 2);
			for (j = 0; (s32)j < (s32)m; j++) {
				u32 L;
				if ((s32)M32(th + 8) >= 4) break;
				L = M32(M32(G + 4) + j * 4u);
				if (M32(L + 0xfc) != 0x4c544e44u) b_assert(c, S - 24, 0x00f88b70u, 0x0265cddcu, 0x01099d70u, 0x0265cdf8u, 0x2efu, 0x0265d208u, 0x00f86702u);
				if (M32(L + 0xdc) == 4) {
					acc = qb_vscale(c, L + 0xbc, M32(L + 0xcc), S + 0xdcu, S - 12, 0x00f86764u);
#if QB_LEAF_CALLS
					XST(S + 0x13cu, acc);
#endif
					any = 1; cnt++;
					continue;
				}
				if (M32(th + 0xc + M32(th + 8) * 4u) != L) dirty = 1;
				BW32(th + 0xc + M32(th + 8) * 4u, L);
				BW32(th + 8, M32(th + 8) + 1);
			}
		}
	}
	if (!dirty && (qb_ne(M32(th + 0x1c), acc.u[0]) || qb_ne(M32(th + 0x20), acc.u[1]) || qb_ne(M32(th + 0x24), acc.u[2]))) dirty = 1;
	if ((Q = M32(A + 0x1b4)) != 0) {
		for (k = 0; (s32)k < (s32)M8(Q + 0x14a); k++) {
			u32 L, bit = 1u << (k & 7);
			if ((s32)M32(th + 8) >= 4) break;
			L = M32(Q + k * 0x24u + 0x30);
			if (M32(L + 0xfc) != 0x4c544e44u) b_assert(c, S - 24, 0x00f88b70u, 0x0265cddcu, 0x01099d70u, 0x0265cdf8u, 0x310u, 0x0265d208u, 0x00f86a41u);
			if (M32(L + 0xdc) == 4) {
				X128 sv = qb_vscale(c, L + 0xbc, M32(L + 0xcc), S + 0x88u, S - 12, 0x00f86a9fu);
#if QB_LEAF_CALLS
				M32(S - 4) = c->eax; c->ecx = S + 0x13cu;
				BCALL(S - 8, 0x00456d50u, 0x00f86aacu);
				acc = ex_ld(S + 0x13cu);
				(void)sv;
#else
				acc = ex_addps(acc, sv);
#endif
				cnt++;
				continue;
			}
			if (fl && !(M8(Q + 0x149 + (k >> 3)) & bit)) continue;
			if (M32(th + 0xc + M32(th + 8) * 4u) != L) dirty = 1;
			if (!(M8(Q + 0x148 + (k >> 3)) & bit)) {
				BW32(th + 0xc + M32(th + 8) * 4u, L);
				BW32(th + 8, M32(th + 8) + 1);
			}
		}
		if (!dirty && (qb_ne(M32(th + 0x1c), M32(Q)) || qb_ne(M32(th + 0x20), M32(Q + 4)) || qb_ne(M32(th + 0x24), M32(Q + 8)))) dirty = 1;
	}
	if (M8(A + 0x8fc)) {
		BW32(A + 0x904, M32(th + 8));
		if (!dirty) {
			M32(S - 4) = 0; M32(S - 8) = A;
			c->eax = 0; c->ecx = th; c->edx = M32(th + 8);
			BCALL(S - 12, 0x00f86fe0u, 0x00f86cd8u);
			QB_CP4(A + 0xb68, th + 0x1c);
		}
	}
	if (!dirty) goto done;
	if (Q) QB_CP4(th + 0x1c, Q);
	else {
#if QB_LEAF_CALLS
		M32(S - 4) = S + 0x50u;
		BCALL(S - 8, 0x00ead960u, 0x00f86d3bu);
		QB_CP4(th + 0x1c, c->eax);
#else
		u32 k255 = M32(0x0109f7a4u);
		X128 z = ex_divps(qb_set4(0, 0, 0, k255), qb_set1(k255)); /* (00ead960: {0, 0, 0, 255} / 255) */
		QB_W4(th + 0x1c, z.u[0], z.u[1], z.u[2], z.u[3]);
#endif
	}
	if (any) {
		X128 fc = ex_cvtsi2ss(ex_ss(0), (s32)cnt);
#if QB_LEAF_CALLS
		M32(S - 4) = fc.u[0]; c->ecx = S + 0x13cu;
		BCALL(S - 8, 0x00572d80u, 0x00f86d9du);
		M32(S - 4) = S + 0x13cu; c->ecx = th + 0x1c;
		BCALL(S - 8, 0x00456d50u, 0x00f86db4u);
		M32(S - 4) = M32(0x0109b130u); c->ecx = th + 0x1c;
		BCALL(S - 8, 0x00456e50u, 0x00f86dcdu);
#else
		acc = ex_divps(acc, qb_set1(fc.u[0]));
		BW128(th + 0x1c, ex_addps(ex_ld(th + 0x1c), acc));
		BW128(th + 0x1c, ex_mulps(ex_ld(th + 0x1c), qb_set1(M32(0x0109b130u))));
#endif
	}
	if (fl && ((M8(A + 0x640) >> 5) & 1u)) {
#if QB_LEAF_CALLS
		M32(S - 4) = S + 0x38u;
		BCALL(S - 8, 0x00796ae0u, 0x00f86df9u);
		QB_CP4(th + 0x1c, c->eax);
#else
		X128 kv = qb_set1(M32(0x0109f7a4u)), o = ex_divps(kv, kv); /* (00796ae0: {255 x 4} / 255, 004558f0) */
		QB_W4(th + 0x1c, o.u[0], o.u[1], o.u[2], o.u[3]);
#endif
	}
	if (!M32(th + 0x144) && !M32(th + 0x14c)) {
		M32(S - 4) = Bp; M32(S - 8) = A;
		c->eax = Bp; c->ecx = th;
		BCALL(S - 12, 0x00f87990u, 0x00f86e60u);
		goto done;
	}
	M32(S - 4) = 1; M32(S - 8) = A;
	c->ecx = th; c->edx = A;
	BCALL(S - 12, 0x00f86fe0u, 0x00f86e77u);
	if (M8(A + 0x8fc)) QB_CP4(A + 0xb68, th + 0x1c);
	if (M32(th + 0x140)) {
		u32 p = M32(th + 0x140), g = M32(0x028670bcu), reg = M32(M32(g + 8) + p * 8u - 4u), dv = M32(0x0283eac0u), vt = M32(dv), m = M32(vt + 0x1b4);
		M32(S - 4) = 1; M32(S - 8) = th + 0x1cu; M32(S - 12) = reg; M32(S - 16) = dv;
		c->eax = m; c->ecx = vt; c->edx = dv;
		BCALL(S - 20, m, 0x00f86f24u);
	}
	if (M32(th + 0x144)) {
		X128 f = ex_cvtsi2ss(ex_ss(0), (s32)M32(th + 8));
		u32 p = M32(th + 0x144), g = M32(0x028670bcu), reg, dv, vt, m;
		XST(S + 8, qb_set1(f.u[0]));
		reg = M32(M32(g + 8) + p * 8u - 4u); dv = M32(0x0283eac0u); vt = M32(dv); m = M32(vt + 0x1b4);
		M32(S - 4) = 1; M32(S - 8) = S + 8; M32(S - 12) = reg; M32(S - 16) = dv;
		c->eax = vt; c->ecx = dv; c->edx = m;
		BCALL(S - 20, m, 0x00f86fc9u);
	}
done:
	B_RET(sp, 2);
}
#endif

/* x87 st0 (a callee's float result, double in c->st0) stored as a float: fstp dword (cvtsd2ss, as the translation) */
static inline u32 qb_d2f(double v) { X128 r = ex_ss(0); __asm__("cvtsd2ss %1, %0" : "+x"(r.v) : "x"(v)); return r.u[0]; }
/* (x, y, z, 1): movaps x3, r; unpckhps r, ones; shufps x3, r, 0xc4 */
static inline X128 qb_w1(X128 r, X128 ones) { return ex_shufps(r, ex_unpckhps(r, ones), 0xc4); }
/* 00f9a340 mode B: a bone's accumulated translation T / scale S written back (bl blend by w, ad additive, else plain);
 * the old S read after T is written, as the original */
static inline void qb_9a3_flush(XCtx *c, u32 pT, u32 pS, X128 T, X128 Sv, int bl, int ad, u32 wb, u32 oneb) {
	X128 w4 = qb_set1(wb), r;
	if (bl) {
		X128 om = qb_set1(ex_subss(ex_ss(oneb), ex_ss(wb)).u[0]);
		r = ex_addps(ex_mulps(T, w4), ex_mulps(om, ex_ld(pT)));
		BW128(pT, r);
		r = ex_addps(ex_mulps(Sv, w4), ex_mulps(ex_ld(pS), om));
		BW128(pS, r);
	} else if (ad) {
		X128 o1 = qb_set1(oneb);
		r = qb_w1(ex_addps(ex_mulps(T, w4), ex_ld(pT)), o1);
		BW128(pT, r);
		r = qb_w1(ex_mulps(ex_addps(ex_mulps(Sv, w4), o1), ex_ld(pS)), o1);
		BW128(pS, r);
	} else {
		BW128(pT, T);
		BW128(pS, Sv);
	}
}

/* 00f9a340: cdecl (base, hdr, bones, float t, float w, u8 add, const u16 *idx), ret (optimized code; aligned frame P =
 * (entry esp - 4 & ~15) - 0x100, the esp after its pushes; ebp frame): samples hdr->+6 + 1 animation channels (the
 * first at base + hdr->+0, the next 00f9a030(ch)), channel i for bone idx[i] (0xffff: none; bone records 96 bytes at
 * bones). bl = w < 1 (comiss) and not add. Rotation mode (hdr->+7 bit 0): bone target q = rec->+0x4c (0: skipped);
 * ch: word +4 = n groups (bit 7: the decode flag), +8 the u8 key counts per group, then the 6-byte keys (4-aligned);
 * g = (int)t / 32 (cvttss2si, toward zero) clamped to n - 1; t <= key 0's time (bits 43..47, _allshr done here):
 * R = 00f99040(key 0); in the last group with t >= its last key's time + 32 g: R = 00f99040(last key); else
 * 00f98690(t, g, counts, keys, &k0, &k1, &t0, &t1), 00f99680(&Q0, &Q1, k0, k1, flag), R = Q0->Slerp(Q1, (t - t0) /
 * (t1 - t0)) (005168e0). Then bl: *q = q->Slerp(R, w); add: *q = q->Mul(Identity->Slerp(R, w)) (00743c70); else
 * *q = R. Vector mode: per bone accumulators T / S (lanes 0..2 from the channels: types 4..9 00f98b70(ch + 8, ch->+4,
 * t) (x87, fstp float), 0xb..0x10 the float ch->+8, 0x11..0x13 00f98880(ch, t); T x, y, z, S x, y, z in that order;
 * else the engine assertion, line 0x48a), started on a bone change (rec->+0x48 = &T, +0x50 = &S): add: 0; bl: the
 * values at rec->+0x30's vtbl[9]() / vtbl[11]() (getters); else *&T / *&S; written back (qb_9a3_flush) when the
 * bone changes and at the end. The arguments t / w copied (fld / fstp) where pushed; the frame's locals that callees
 * read written at the original's addresses; the /GS cookie skipped. No result (the one caller, 00f9d125, reads no
 * register). Callees: pure (decoders, slerp / mul, curve evaluators, the channel stride, getters). */
B_FN(00f9a340, "animation channels: sample at t", 0) {
	u32 sp = c->esp, a1 = M32(sp + 4), a2 = M32(sp + 8), a3 = M32(sp + 0xc), wb = M32(sp + 0x14), bp = sp - 4;
	u32 P = (bp & ~15u) - 0xf4u - 12u, n = M8(a2 + 6) + 1u, rec = M32(a2) + a1, idxp = M32(sp + 0x1c), it, oneb = M32(0x010aa950u);
	int add = M8(sp + 0x18) != 0, bl = !(ex_comiss(ex_ss(oneb), ex_ss(wb)) & 5) && !add;
	X128 tt = ex_ldss(sp + 0x10);
	B_ENTER(bp);
	if ((s32)n <= 0) goto done;
	if (M8(a2 + 7) & 1) {
		for (it = n; it; it--) {
			u32 idx = M16(idxp), Tg;
			if (idx != 0xffffu && (Tg = M32(a3 + idx * 96u + 0x4c)) != 0) {
				u32 hw = M16(rec + 4), fl = (hw >> 7) & 1u, ng = hw & 0xff7fu, cn = rec + 8, ks = cn + ((ng + 3u) & 0xfffcu);
				u32 i = 0, s0 = 0, s1 = 0, tail = 0, last, src = 0, ra = 0;
				s32 g = ex_cvttss2si(tt);
				X128 R;
				g = (g + (g < 0 ? 31 : 0)) >> 5;
				if (g >= (s32)ng) g = (s32)ng - 1;
				if ((s32)ng >= 2) do { s0 += M8(cn + i); s1 += M8(cn + i + 1); i += 2; } while ((s32)i < (s32)(ng - 1));
				if ((s32)i < (s32)ng) tail = M8(cn + i);
				last = tail + (s0 + s1) - 1u;
				if (!(ex_comiss(ex_cvtsi2ss(tt, (s32)((M32(ks + 4) >> 11) & 0x1fu)), tt) & 4)) { src = ks; ra = 0x00f9a4f6u; } /* (not jb) */
				else if ((u32)g == ng - 1u) {
					u32 lk = ks + last * 6u;
					if (!(ex_comiss(tt, ex_cvtsi2ss(tt, (s32)(((M32(lk + 4) >> 11) & 0x1fu) + ((u32)g << 5)))) & 4)) { src = lk; ra = 0x00f9a54au; }
				}
				if (ra) {
					M32(P - 4) = fl; M32(P - 8) = src; M32(P - 12) = P + 0xa0u;
					BCALL(P - 16, 0x00f99040u, ra);
					R = ex_ld(P + 0xa0u);
				} else {
					X128 t0, u;
					M32(P - 4) = P + 0x4cu; M32(P - 8) = P + 0x44u; M32(P - 12) = P + 0x48u; M32(P - 16) = P + 0x40u;
					M32(P - 20) = ks; M32(P - 24) = cn; M32(P - 28) = (u32)g; M32(P - 32) = tt.u[0];
					BCALL(P - 36, 0x00f98690u, 0x00f9a579u);
					M32(P - 36) = fl; M32(P - 40) = M32(P + 0x48u); M32(P - 44) = M32(P + 0x40u); M32(P - 48) = P + 0x50u; M32(P - 52) = P + 0xe0u;
					BCALL(P - 56, 0x00f99680u, 0x00f9a59au);
					t0 = ex_ldss(P + 0x44u);
					u = ex_divss(ex_subss(tt, t0), ex_subss(ex_ldss(P + 0x4cu), t0));
					M32(P - 4) = u.u[0]; M32(P - 8) = P + 0x50u; M32(P - 12) = P + 0x70u;
					c->ecx = P + 0xe0u;
					BCALL(P - 16, 0x005168e0u, 0x00f9a5d8u);
					R = ex_ld(c->eax);
					if (bl || add) XST(P + 0xa0u, R);
				}
				if (bl) {
					M32(P - 4) = wb; M32(P - 8) = P + 0xa0u; M32(P - 12) = P + 0x90u;
					c->ecx = Tg;
					BCALL(P - 16, 0x005168e0u, 0x00f9a61cu);
					BW128(Tg, ex_ld(c->eax));
				} else if (add) {
					XST(P + 0xd0u, qb_set4(0, 0, 0, oneb));
					M32(P - 4) = wb; M32(P - 8) = P + 0xa0u; M32(P - 12) = P + 0x80u;
					c->ecx = P + 0xd0u;
					BCALL(P - 16, 0x005168e0u, 0x00f9a68fu);
					M32(P - 4) = c->eax; M32(P - 8) = P + 0x60u;
					c->ecx = Tg;
					BCALL(P - 12, 0x00743c70u, 0x00f9a6a0u);
					BW128(Tg, ex_ld(c->eax));
				} else BW128(Tg, R);
			}
			idxp += 2;
			M32(P - 4) = rec;
			BCALL(P - 8, 0x00f9a030u, 0x00f9a6e2u);
			rec = c->eax;
		}
	} else {
		u32 cur = 0xffffffffu, pT = 0, pS = 0, t0 = 0, t1 = 0, t2 = 0, t3 = 0, q0 = 0, q1 = 0, q2 = 0, q3 = 0;
		for (it = n; it; it--) {
			u32 idx = M16(idxp);
			if (cur != idx) {
				if ((s32)cur >= 0 && cur != 0xffffu) qb_9a3_flush(c, pT, pS, qb_set4(t0, t1, t2, t3), qb_set4(q0, q1, q2, q3), bl, add, wb, oneb);
				cur = idx;
				if (idx != 0xffffu) {
					u32 rb = idx * 96u + a3;
					X128 v;
					pT = M32(rb + 0x48); pS = M32(rb + 0x50);
					if (pT && add) { t0 = t1 = t2 = t3 = 0; q0 = q1 = q2 = q3 = 0; }
					else if (pT && bl) {
						v = ex_ld(BVGET(M32(rb + 0x30), 0x24, P - 4, 0x00f9a89cu));
						t0 = v.u[0]; t1 = v.u[1]; t2 = v.u[2]; t3 = v.u[3];
						v = ex_ld(BVGET(M32(rb + 0x30), 0x2c, P - 4, 0x00f9a8d8u));
						q0 = v.u[0]; q1 = v.u[1]; q2 = v.u[2]; q3 = v.u[3];
					} else {
						v = ex_ld(pT); t0 = v.u[0]; t1 = v.u[1]; t2 = v.u[2]; t3 = v.u[3];
						v = ex_ld(pS); q0 = v.u[0]; q1 = v.u[1]; q2 = v.u[2]; q3 = v.u[3];
					}
				}
			}
			if (pT) {
				u32 ty = M8(rec + 3), f;
				if (ty >= 4 && ty <= 9) {
					M32(P - 4) = tt.u[0]; M32(P - 8) = M16(rec + 4); M32(P - 12) = rec + 8;
					BCALL(P - 16, 0x00f98b70u, 0x00f9a91bu);
					f = qb_d2f(c->st0);
					switch (M8(rec + 3)) {
					case 4: t0 = f; break;
					case 5: t1 = f; break;
					case 6: t2 = f; break;
					case 7: q0 = f; break;
					case 8: q1 = f; break;
					case 9: q2 = f; break;
					}
				} else if (ty >= 0xb && ty <= 0x10) {
					f = M32(rec + 8);
					switch (ty) {
					case 0xb: t0 = f; break;
					case 0xc: t1 = f; break;
					case 0xd: t2 = f; break;
					case 0xe: q0 = f; break;
					case 0xf: q1 = f; break;
					default: q2 = f; break;
					}
				} else if (ty >= 0x11 && ty <= 0x13) {
					/* (never run in the battle + field coverage; checked line by line against f9aa42..f9aaa0:
					 * fld t / push / fstp [esp] -> P-4, push edi (rec) -> P-8, call 00f98880 (ret f9aa59 at P-12;
					 * it reads only its two stack arguments, no register), fstp float, bl (callee-saved) 0x11 /
					 * 0x12 / 0x13 -> T lanes 0 / 1 / 2 ([esp+0xb0 / b4 / b8]). Else: f9aaa2..f9aadb, the
					 * assertion handler at [026e80c0] (lazily 00cce7b0), args P-20..P-4, ret f9aadb at P-24) */
					M32(P - 4) = tt.u[0]; M32(P - 8) = rec;
					BCALL(P - 12, 0x00f98880u, 0x00f9aa59u);
					f = qb_d2f(c->st0);
					if (ty == 0x11) t0 = f;
					else if (ty == 0x12) t1 = f;
					else t2 = f;
				} else b_assert(c, P - 24, 0x00cce7b0u, 0x010d65b0u, 0x01099d70u, 0x0265e670u, 0x48au, 0x0265e718u, 0x00f9aadbu);
			}
			idxp += 2;
			M32(P - 4) = rec;
			BCALL(P - 8, 0x00f9a030u, 0x00f9aae9u);
			rec = c->eax;
		}
		if ((s32)cur >= 0 && cur != 0xffffu) qb_9a3_flush(c, pT, pS, qb_set4(t0, t1, t2, t3), qb_set4(q0, q1, q2, q3), bl, add, wb, oneb);
	}
done:
	B_RET(sp, 0);
}

/* the x87 as the translation models it (xrc.h: double arithmetic, XR87 = rounded to float when the FCW's precision
 * control is 24-bit; fld float / fild = exact conversions), through inline asm (fast math cannot fold or reorder) */
static inline double qb_r87(double v, int p24) { if (p24) __asm__("cvtsd2ss %0, %0\n\tcvtss2sd %0, %0" : "+x"(v)); return v; }
static inline double qb_fd(u32 bits) { X128 a = ex_ss(bits); double r; __asm__("cvtss2sd %1, %0" : "=x"(r) : "x"(a.v)); return r; }
static inline double qb_fi(s32 i) { double r; __asm__("cvtsi2sd %1, %0" : "=x"(r) : "r"(i)); return r; }
#define QB_DOP(name_, ins_) static inline double name_(double a, double b, int p24) { __asm__(ins_ " %1, %0" : "+x"(a) : "x"(b)); return qb_r87(a, p24); }
QB_DOP(qb_fadd, "addsd") QB_DOP(qb_fsub, "subsd") QB_DOP(qb_fmul, "mulsd") QB_DOP(qb_fdiv, "divsd")
/* a curve key's value: 16-bit (15 bits * [025e5d34] 1/32767, bit 15 the sign) or 8-bit (7 bits * [025df674] 1/127,
 * bit 7 the sign): cvtsi2ss, mulss, xorps [0109c600] (.rdata, folded by the translation: 38000100, 3c010204, 80000000) */
static inline u32 qb_kv(int b16, u32 v) {
	u32 r;
	if (b16) {
		r = ex_mulss(ex_cvtsi2ss(ex_ss(0), (s32)(v & 0x7fffu)), ex_ss(0x38000100u)).u[0];
		if (v & 0x8000u) r ^= 0x80000000u;
	} else {
		r = ex_mulss(ex_cvtsi2ss(ex_ss(0), (s32)(v & 0x7fu)), ex_ss(0x3c010204u)).u[0];
		if (v & 0x80u) r ^= 0x80000000u;
	}
	return r;
}
/* 00fa0cc0 (16-bit) / 00f98840 (8-bit): cdecl (v, float a1, float a0), ret: st0 = (key value v) * a1 + a0 (x87) */
static inline double qb_kx87(XCtx *c, int b16, u32 v, u32 a1, u32 a0, int p24, u32 at, u32 ra) {
#if QB_LEAF_CALLS
	(void)p24;
	M32(at + 4) = v; M32(at + 8) = a1; M32(at + 12) = a0;
	BCALL(at, b16 ? 0x00fa0cc0u : 0x00f98840u, ra);
	return c->st0;
#else
	(void)c; (void)at; (void)ra;
	return qb_fadd(qb_fmul(qb_fd(qb_kv(b16, v)), qb_fd(a1), p24), qb_fd(a0), p24);
#endif
}
/* 00f98a90 (16-bit: 3 bytes a key) / 00f98b00 (8-bit: 2): cdecl (u32 *pv, u32 *pt, s32 g, const u8 *cnts), ret:
 * *pt += (cnts[0] + .. + cnts[g - 1]) * 3 / 2 (pairwise, as the original), *pv = *pt + cnts[max(g, 0)] -- group g's
 * key times / values. *pt in, both out (as values; with QB_LEAF_CALLS through the frame slots pva / pta) */
static inline void qb_grp(XCtx *c, int b16, u32 pva, u32 pta, u32 *pv, u32 *pt, s32 g, u32 cnts, u32 at, u32 ra) {
#if QB_LEAF_CALLS
	M32(pta) = *pt;
	M32(at + 4) = pva; M32(at + 8) = pta; M32(at + 12) = (u32)g; M32(at + 16) = cnts;
	BCALL(at, b16 ? 0x00f98a90u : 0x00f98b00u, ra);
	*pt = M32(pta); *pv = M32(pva);
#else
	u32 i = 0, se = 0, so = 0, tl = 0;
	(void)c; (void)pva; (void)pta; (void)at; (void)ra;
	if (g >= 2) do { se += M8(cnts + i); so += M8(cnts + i + 1); i += 2; } while ((s32)i < g - 1);
	if ((s32)i < g) { tl = M8(cnts + i); i++; }
	*pt += (se + so + tl) * (b16 ? 3u : 2u);
	*pv = *pt + M8(cnts + i);
#endif
}

/* 00f98b70: cdecl (data, u16 w, float t), ret: a packed curve's value at t, on the x87 stack (st0). data: float a0,
 * float a1, n = w & 0x7fff u8 key counts (one per group of 255 time units), then per group its key times (u8) and
 * values (u16 with w bit 15, else u8: qb_kv). ti = (int)t (cvttss2si), g = ti / 255 clamped to n - 1, r = ti - 255 g.
 * ti below the first key's time: v0 * a1 + a0 (x87: fld, fmul, fadd); group g the last and r at or past its last
 * key: that key's v * a1 + a0 (00fa0cc0 / 00f98840); else the first key k of group g with time > r (none: group g + 1,
 * key 0), y1 = v_k * a1 + a0, T1 = time_k + 255 g' (SSE, u16), y0 / T0 the same for the key before (k = 0: group
 * g' - 1's last), u = (t - T0) / (T1 - T0) (fild / fsubr / fidiv; T as s16), st0 = y0 + (y1 - y0) * u. Groups'
 * keys by 00f98a90 / 00f98b00. Its argument slots are written as the original does (BW32: data <- a1, then T0 / y0;
 * w <- the key / value pointers, v, y0 / T0, dT; t <- dT (16-bit)). eax, ecx, edx are not results (the one caller,
 * 00f9a916, reads none) */
#if 0 /* (battle batch 3: 00f98b70 not selected) */
B_FN(00f98b70, "packed curve: value at t", BR_ST0) {
	u32 sp = c->esp, D = M32(sp + 4), cw = M16(sp + 8), tb = M32(sp + 0xc), L = sp - 0x14u, Q = L - 16u;
	u32 n = cw & 0x7fffu, a0 = M32(D), a1 = M32(D + 4), cn = D + 8, TB = n + D + 8, pv = 0, pt, cg, k, Tc, v1, y1, y0, T1, T0;
	int b16 = (cw & 0x8000u) != 0, p24 = ((c->fcw >> 8) & 3u) == 0;
	s32 ti = ex_cvttss2si(ex_ss(tb)), g = ti / 255, r, gs, dT;
	double f0;
	BW32(sp + 4, a1);
	BW32(sp + 8, TB);
	if (g >= (s32)n) g = (s32)n - 1;
	r = (s32)((u32)ti - (u32)g * 255u);
	if (ti < (s32)M8(TB)) {
		u32 v = b16 ? M16(TB + M8(cn)) : M8(TB + M8(cn)), x = qb_kv(b16, v);
		BW32(sp + 8, x);
		c->st0 = qb_fadd(qb_fmul(qb_fd(x), qb_fd(a1), p24), qb_fd(a0), p24);
		c->esp = sp; xrc_cdecl_ret(c);
		return;
	}
	pt = TB;
	qb_grp(c, b16, L + 4, sp + 8, &pv, &pt, g, cn, Q - 20, b16 ? 0x00f98c4cu : 0x00f98e74u);
	BW32(sp + 8, pt);
	cg = M8(cn + (u32)g);
	if (g == (s32)n - 1 && r >= (s32)M8(pt + cg - 1)) {
		u32 v = b16 ? M16(pv + cg * 2u - 2) : M8(pv + cg - 1);
		c->st0 = qb_kx87(c, b16, v, a1, a0, p24, Q - 16, b16 ? 0x00f98c88u : 0x00f98eb0u);
		c->esp = sp; xrc_cdecl_ret(c);
		return;
	}
	for (k = 0; k < cg && r >= (s32)M8(pt + k); k++) {}
	gs = g;
	if (k == cg) {
		BW32(sp + 8, TB);
		gs = g + 1; pt = TB;
		qb_grp(c, b16, L + 4, sp + 8, &pv, &pt, gs, cn, Q - 20, b16 ? 0x00f98cc9u : 0x00f98eeeu);
		BW32(sp + 8, pt);
		k = 0;
	}
	Tc = pt;
	v1 = b16 ? M16(pv + k * 2u) : M8(pv + k);
	y1 = ex_addss(ex_mulss(ex_ss(qb_kv(b16, v1)), ex_ss(a1)), ex_ss(a0)).u[0];
	T1 = (M8(Tc + k) + (u32)gs * 255u) & 0xffffu;
	if (!k) {
		u32 tp = TB, pvp = 0, cp, v0;
		s32 gp = gs - 1;
		BW32(sp + 8, 0);
		qb_grp(c, b16, sp + 8, L + 8, &pvp, &tp, gp, cn, Q - 20, b16 ? 0x00f98d54u : 0x00f98f74u);
		BW32(sp + 8, pvp);
		cp = M8(cn + (u32)gp);
		v0 = b16 ? M16(pvp + cp * 2u - 2) : M8(pvp + cp - 1);
		y0 = ex_addss(ex_mulss(ex_ss(qb_kv(b16, v0)), ex_ss(a1)), ex_ss(a0)).u[0];
		T0 = (M8(tp + cp - 1) + (u32)gp * 255u) & 0xffffu;
	} else {
		u32 v0 = b16 ? M16(pv + k * 2u - 2) : M8(pv + k - 1);
		y0 = ex_addss(ex_mulss(ex_ss(qb_kv(b16, v0)), ex_ss(a1)), ex_ss(a0)).u[0];
		T0 = (M8(Tc + k - 1) + (u32)gs * 255u) & 0xffffu;
	}
	dT = (s32)(s16)T1 - (s32)(s16)T0;
	f0 = qb_fi((s32)(s16)T0);
	if (b16) {
		BW32(sp + 4, (u32)(s32)(s16)T0);
		BW32(sp + 8, y0);
		f0 = qb_fsub(qb_fd(tb), f0, p24);
		BW32(sp + 0xc, (u32)dT);
	} else {
		BW32(sp + 8, (u32)(s32)(s16)T0);
		BW32(sp + 8, (u32)dT);
		BW32(sp + 4, y0);
		f0 = qb_fsub(qb_fd(tb), f0, p24);
	}
	f0 = qb_fdiv(f0, qb_fi(dT), p24);
	c->st0 = qb_fadd(qb_fd(y0), qb_fmul(qb_fsub(qb_fd(y1), qb_fd(y0), p24), f0, p24), p24);
	c->esp = sp; xrc_cdecl_ret(c);
}
#endif

/* 00f314c0's render-state wrappers. The translation inlines all but 00eacb40 (alpha func + ref, x87 + FCW) and the
 * helpers they tail into, so (QB_LEAF_CALLS 0) do they: their global stores (BW8 / BW32) and the device's
 * SetRenderState (IDirect3DDevice9 [0283eac0], vtbl + 0xe4, stdcall (this, state, value)) with the stack the wrapper
 * gives it -- `e` = the wrapper's esp (its return address there, its arguments at e + 4..). Each SetRenderState is
 * idempotent, so they repeat safely in the checked mode */
static inline void qb_srs(XCtx *c, u32 dev, u32 e, u32 rs, u32 v, u32 ra) {
	u32 fn = M32(M32(dev) + 0xe4u);
	M32(e - 12) = dev; M32(e - 8) = rs; M32(e - 4) = v;
	BCALL(e - 16, fn, ra);
}
/* 00eabdb0 (g 0283e538, D3DRS_ALPHATESTENABLE 0xf) / 00eabdf0 (0283e540, ALPHABLENDENABLE 0x1b) / 00eabeb0 (0283e568,
 * ZWRITEENABLE 0xe): cdecl (u8 v): [g] = v, SetRenderState(rs, v); rai = its return address inside the wrapper */
static inline void qb_rs8(XCtx *c, u32 fn, u32 g, u32 rs, u32 v, u32 e, u32 ra, u32 rai) {
#if QB_LEAF_CALLS
	(void)g; (void)rs; (void)rai;
	M32(e + 4) = v; BCALL(e, fn, ra);
#else
	u32 dev = M32(0x0283eac0u);
	(void)fn; (void)ra;
	BW8(g, v);
	qb_srs(c, dev, e, rs, v & 0xffu, rai);
#endif
}
/* 00eacc90: cdecl (v): [0283e5b8] = v, SetRenderState(D3DRS_CULLMODE 0x16, [026248d0 + 4 v]) */
static inline void qb_cull(XCtx *c, u32 v, u32 e, u32 ra) {
#if QB_LEAF_CALLS
	M32(e + 4) = v; BCALL(e, 0x00eacc90u, ra);
#else
	u32 dev = M32(0x0283eac0u);
	(void)ra;
	BW32(0x0283e5b8u, v);
	qb_srs(c, dev, e, 0x16, M32(v * 4u + 0x026248d0u), 0x00eaccb4u);
#endif
}
/* 00eabf60: cdecl (v): [0283e560] (byte) = v, then a tail jump to 00eabbc0 (v) (two SetRenderStates) */
static inline void qb_eabf60(XCtx *c, u32 v, u32 e, u32 ra) {
	M32(e + 4) = v;
#if QB_LEAF_CALLS
	BCALL(e, 0x00eabf60u, ra);
#else
	BW8(0x0283e560u, v);
	BCALL(e, 0x00eabbc0u, ra);
#endif
}
/* 00eacbc0: cdecl (a, b): [0283e544] = a, [0283e550] = b, SetRenderState(BLENDOP 0xab, [02624824 + 4 a]),
 * SetRenderState(BLENDOPALPHA 0xd1, [02624824 + 4 b]) (esi pushed: the calls 4 lower) */
static inline void qb_eacbc0(XCtx *c, u32 a, u32 b, u32 e, u32 ra) {
#if QB_LEAF_CALLS
	M32(e + 4) = a; M32(e + 8) = b; BCALL(e, 0x00eacbc0u, ra);
#else
	u32 x, dev;
	(void)ra;
	BW32(0x0283e544u, a);
	x = M32(a * 4u + 0x02624824u);
	BW32(0x0283e550u, b);
	dev = M32(0x0283eac0u);
	qb_srs(c, dev, e - 4, 0xab, x, 0x00eacbf4u);
	x = M32(b * 4u + 0x02624824u);
	dev = M32(0x0283eac0u);
	qb_srs(c, dev, e - 4, 0xd1, x, 0x00eacc11u);
#endif
}
/* 00ead120: cdecl (a, b): [0283e548] = a, [0283e54c] = b, [0283e554] = a, [0283e558] = b, 00eac6a0 (a, b, a, b) (the
 * four blend factors) */
static inline void qb_ead120(XCtx *c, u32 a, u32 b, u32 e, u32 ra) {
#if QB_LEAF_CALLS
	M32(e + 4) = a; M32(e + 8) = b; BCALL(e, 0x00ead120u, ra);
#else
	(void)ra;
	M32(e - 16) = a; M32(e - 12) = b; M32(e - 8) = a; M32(e - 4) = b;
	BW32(0x0283e548u, a); BW32(0x0283e54cu, b); BW32(0x0283e554u, a); BW32(0x0283e558u, b);
	BCALL(e - 20, 0x00eac6a0u, 0x00ead147u);
#endif
}
/* 00eacb40: cdecl (mode, float ref) (alpha func + ref: globals 0283e530 / 0283e534, 00eac630); called as is */
static inline void qb_eacb40(XCtx *c, u32 mode, u32 f, u32 e, u32 ra) {
	M32(e + 4) = mode; M32(e + 8) = f;
	BCALL(e, 0x00eacb40u, ra);
}
/* "1.0 > x" as comiss [010aa950] 1.0, x + ja (unordered: false) */
static inline int qb_lt1(u32 x) { return !(ex_comiss(ex_ss(0x3f800000u), ex_ss(x)) & 5u); }

/* 00f314c0: thiscall (A, B), ret 8; an ebp frame (calls at sp - 0xcc with one argument, sp - 0xd0 with two). Applies
 * this material pass's render states for object A: if [0283e64c] != [[0286719c] + 0x1c] or [0283e648] !=
 * [[028671a0] + 0x1c] (00eac240 / 00eac230): this->+8 = [028671a0], this->+c = [0286719c]. Unless this is the
 * last-applied pass [02867198] and neither A->+884 nor A->+894 is below 1.0: with R = this->+38 (u16 flags at R,
 * float at R + 4): cull mode (00eacc90 (R bit 3)); unless A->+862: A->+38c = !bit 6, A->+640 bits 3 / 4 / 5 = R bits
 * 10 / 11, this->+30 bit 0; unless A->+641 bit 0: alpha test (00eabdb0) on with alpha ref R+4 * A->+884 (x87 fmul,
 * fstp float; 00eacb40 (4, ref)) if bit 1, else off; A->+641 bit 5 cleared; bit 2: alpha test on, ref R+4 (00eacb40),
 * 00eabf60 (1), alpha blend off (00eabdf0 (0)); else 00eabf60 (0), blend op / factors by bits 8 / 9 (00eacbc0 +
 * 00ead120: (0,0)+(6,1) / (4,4)+(6,1) / (0,0)+(6,7)), alpha blend = bit 0 || A->+884 < 1 || A->+894 < 1; then if
 * A->+884 < 1 || A->+894 < 1 and A->+8f0 == 1: 00eabf60 (1), 00eabdf0 (0), 00eabdb0 (1); z write (00eabeb0) = bit 4;
 * 00f76190 (empty), [026e52a1] = 1. Always: [028671a0]->00f08810 (A), unless A->+862 [0286719c]->00f08f70 (A) (both
 * call their children's vtbl[2] (A, list)), [02867198] = this. Returns nothing (the one caller, 00f34a27, reads no
 * register); edx = this, and with A->+862 set eax = that byte, ecx = A, as the original leaves them */
#if 0 /* (battle batch 3: 00f314c0 not selected) */
B_FN(00f314c0, "material pass: apply render states", 0) {
	u32 sp = c->esp, th = c->ecx, A = M32(sp + 4), B = M32(sp + 8), e1 = sp - 0xccu, e2 = sp - 0xd0u, R, z;
	B_ENTER(sp - 4);
	{
		u32 e0 = M32(M32(0x0286719cu) + 0x1cu), g;
		int set;
#if QB_LEAF_CALLS
		BCALL(sp - 0xc8u, 0x00eac240u, 0x00f314dfu); g = c->eax;
#else
		g = M32(0x0283e64cu);
#endif
		set = g != e0;
		if (!set) {
			u32 e1v = M32(M32(0x028671a0u) + 0x1cu);
#if QB_LEAF_CALLS
			BCALL(sp - 0xc8u, 0x00eac230u, 0x00f314f5u); g = c->eax;
#else
			g = M32(0x0283e648u);
#endif
			set = g != e1v;
		}
		if (set) { BW32(th + 8, M32(0x028671a0u)); BW32(th + 0xc, M32(0x0286719cu)); }
	}
	if (M32(0x02867198u) == th && !qb_lt1(M32(A + 0x884)) && !qb_lt1(M32(A + 0x894))) goto tail;
	R = M32(th + 0x38);
	qb_cull(c, (M16(R) >> 3) & 1u, e1, 0x00f3159du);
	if (!M8(A + 0x862)) {
		u32 t30;
		BW8(A + 0x38c, ((M16(R) >> 6) & 1u) == 0);
		BW8(A + 0x640, (M8(A + 0x640) & 0xf7u) | ((M16(R) >> 10) & 1u) << 3);
		t30 = M8(th + 0x30);
		BW8(A + 0x640, (M8(A + 0x640) & 0xefu) | ((M16(R) >> 11) & 1u) << 4);
		BW8(A + 0x640, (M8(A + 0x640) & 0xdfu) | (t30 & 1u) << 5);
		if (!(M8(A + 0x641) & 1u)) {
			if ((M16(R) >> 1) & 1u) {
				u32 f = M32(R + 4), p;
				qb_rs8(c, 0x00eabdb0u, 0x0283e538u, 0xf, 1, e1, 0x00f316a9u, 0x00eabdd0u);
				p = qb_d2f(qb_fmul(qb_fd(f), qb_fd(M32(A + 0x884)), ((c->fcw >> 8) & 3u) == 0));
				qb_eacb40(c, 4, p, e2, 0x00f316d8u);
			} else
				qb_rs8(c, 0x00eabdb0u, 0x0283e538u, 0xf, 0, e1, 0x00f316e4u, 0x00eabdd0u);
			BW8(A + 0x641, M8(A + 0x641) & 0xdfu);
			if ((M16(R) >> 2) & 1u) {
				qb_rs8(c, 0x00eabdb0u, 0x0283e538u, 0xf, 1, e1, 0x00f31721u, 0x00eabdd0u);
				qb_eacb40(c, 4, M32(R + 4), e2, 0x00f31740u); /* fld / fstp float: a plain copy, as the translation */
				qb_eabf60(c, 1, e1, 0x00f3174au);
				qb_rs8(c, 0x00eabdf0u, 0x0283e540u, 0x1b, 0, e1, 0x00f31754u, 0x00eabe10u);
			} else {
				qb_eabf60(c, 0, e1, 0x00f31763u);
				if ((M16(R) >> 8) & 1u) {
					qb_eacbc0(c, 0, 0, e2, 0x00f3179au);
					qb_ead120(c, 6, 1, e2, 0x00f317bcu);
				} else if ((M16(R) >> 9) & 1u) {
					qb_eacbc0(c, 4, 4, e2, 0x00f31804u);
					qb_ead120(c, 6, 1, e2, 0x00f31832u);
				} else {
					qb_eacbc0(c, 0, 0, e2, 0x00f31862u);
					qb_ead120(c, 6, 7, e2, 0x00f31890u);
				}
				z = (M16(R) & 1u) || qb_lt1(M32(A + 0x884)) || qb_lt1(M32(A + 0x894));
				qb_rs8(c, 0x00eabdf0u, 0x0283e540u, 0x1b, z, e1, 0x00f31909u, 0x00eabe10u);
			}
			if ((qb_lt1(M32(A + 0x884)) || qb_lt1(M32(A + 0x894))) && M32(A + 0x8f0) == 1) {
				qb_eabf60(c, 1, e1, 0x00f31973u);
				qb_rs8(c, 0x00eabdf0u, 0x0283e540u, 0x1b, 0, e1, 0x00f3197du, 0x00eabe10u);
				qb_rs8(c, 0x00eabdb0u, 0x0283e538u, 0xf, 1, e1, 0x00f31987u, 0x00eabdd0u);
			}
			qb_rs8(c, 0x00eabeb0u, 0x0283e568u, 0xe, ((M16(R) >> 4) & 1u) != 0, e1, 0x00f319a9u, 0x00eabed0u);
		}
	}
#if QB_LEAF_CALLS
	M32(e2 + 4) = A; M32(e2 + 8) = B; c->ecx = M32(th + 0x38);
	BCALL(e2, 0x00f76190u, 0x00f319c2u);
#else
	(void)B;
#endif
	BW8(0x026e52a1u, 1);
tail:
	M32(e1 + 4) = A; c->ecx = M32(0x028671a0u);
	BCALL(e1, 0x00f08810u, 0x00f319d8u);
	z = M8(A + 0x862);
	if (!z) {
		M32(e1 + 4) = A; c->ecx = M32(0x0286719cu);
		BCALL(e1, 0x00f08f70u, 0x00f31a01u);
	} else { c->eax = z; c->ecx = A; }
	c->edx = th;
	BW32(0x02867198u, th);
	B_RET(sp, 2);
}
#endif
/* ---- battle batch 3, part C: 00f31a20, 00dc25d0, 00f4a3c0 */

/* 00f31a20 helper: the clamped float r = L->+4 (0 >= r, comiss not below [025a3c20] 0.0: r = [010aa204]), then
 * 00eabdb0(1) (cdecl, at bp - 0x64) and 00eacb40(4, r) (cdecl, at bp - 0x68; r through fld / fstp: the translation's
 * plain copy). ra1 / ra2 the return addresses; e1 / c1 / d1 eax / ecx / edx at the first call; the second's ecx (k 1),
 * eax (k 2) or edx (k 3) = bp - 0x64, the others as the first call leaves them. xmm0 as the original has it */
static inline void qc_f31_rs(XCtx *c, u32 bp, u32 L, u32 e1, u32 c1, u32 d1, int k, u32 ra1, u32 ra2) {
	u32 r = M32(L + 4);
	X128 x0 = ex_ldss(0x025a3c20u);
	if (!(ex_comiss(x0, ex_ss(r)) & 4)) { r = M32(0x010aa204u); x0 = ex_ss(r); } /* (jb skips) */
	M32(bp - 0x60) = 1;
	c->eax = e1; c->ecx = c1; c->edx = d1; c->x[0] = x0;
	BCALL(bp - 0x64, 0x00eabdb0u, ra1);
	M32(bp - 0x60) = r; M32(bp - 0x64) = 4;
	if (k == 1) c->ecx = bp - 0x64; else if (k == 2) c->eax = bp - 0x64; else c->edx = bp - 0x64;
	BCALL(bp - 0x68, 0x00eacb40u, ra2);
}
/* lahf after ucomiss: AH = ZF << 6 | PF << 2 | 2 | CF (SF, AF 0) */
static inline u32 qc_lahf(u32 eax, unsigned fl) { return (eax & 0xffff00ffu) | ((fl & 1u) << 14 | (fl & 2u) << 9 | 0x200u | (fl & 4u) << 6); }

/* 00f31a20: thiscall bool (p), ret 4 (/Od; one caller, 00f34d17, reads al): the render states of p from this's
 * material. this->+8 / +0xc != the globals 028671a0 / 0286719c: set to them. L = this->+0x38; p->+0x38c (byte) = 0,
 * p->+0x640 bit 3 cleared; 00eacc90(L->+0 word bit 3). p->+0x8e8 = [025a3c20] (0.0). A = this->+0x40 with
 * (s8)A->+0xe >= 0: p->+0x840 (byte) = 1, p->+0x844 = (s8)A->+0xe, p->+0x848 = (s8)A->+0xf, qc_f31_rs; then A->+8:
 * A->+8->00f4a1f0(0, A->+3 bit 5 ? A + 0x10 : 0) (thiscall, ret 8), else 00eabc60(0, 0). Else with L->+0 bit 2 and
 * A = this->+0x44: the same with p->+0x844 = 3, p->+0x848 = (s8)A->+0xf. Else p->+0x840 = 0, +0x844 = +0x848 = 0,
 * 00eabc60(0, 0), 00eabdb0(0). Then p->+0x884 == 0 or p->+0x894 == 0 (ucomiss, ordered): qc_f31_rs. Last
 * 0286719c->00f08f70(p) (thiscall, ret 4). eax = its eax with al = 1; ecx / edx / xmm0 as the callees leave them.
 * The callees set D3D state (repeatable). Every memory value re-read where the original reads it (after calls). */
#if 0 /* (battle batch 3: 00f31a20 not selected) */
B_FN(00f31a20, "material render states", BR_EAX) {
	u32 sp = c->esp, th = c->ecx, p = M32(sp + 4), bp = sp - 4, L, A, B, bit, arg;
	unsigned fl;
	B_ENTER(bp);
	if (M32(0x028671a0u) != M32(th + 8) || M32(0x0286719cu) != M32(th + 0xc)) {
		BW32(th + 8, M32(0x028671a0u));
		BW32(th + 0xc, M32(0x0286719cu));
	}
	L = M32(th + 0x38);
	BW8(p + 0x38c, 0);
	BW8(p + 0x640, M8(p + 0x640) & 0xf7u);
	bit = (M16(L) >> 3) & 1u;
	M32(bp - 0x60) = bit;
	c->eax = bp - 0x60; c->ecx = bit; c->edx = bit;
	BCALL(bp - 0x64, 0x00eacc90u, 0x00f31ab9u);
	BW32(p + 0x8e8, M32(0x025a3c20u));
	c->x[0] = ex_ldss(0x025a3c20u);
	A = M32(th + 0x40);
	if (A && (s8)M8(A + 0xe) >= 0) {
		u32 e = (u32)(s32)(s8)M8(A + 0xf), d = (u32)(s32)(s8)M8(A + 0xe);
		BW8(p + 0x840, 1);
		BW32(p + 0x844, d);
		BW32(p + 0x848, e);
		qc_f31_rs(c, bp, L, L, p, e, 1, 0x00f31b59u, 0x00f31b75u);
		A = M32(th + 0x40);
		c->eax = A; c->edx = th;
		if (M32(A + 8)) {
			B = M32(th + 0x40);
			arg = (M8(B + 3) & 0x20u) ? B + 0x10 : 0;
			M32(bp - 0x60) = arg; M32(bp - 0x64) = 0;
			B = M32(th + 0x40);
			c->eax = arg; c->edx = B; c->ecx = M32(B + 8);
			BCALL(bp - 0x68, 0x00f4a1f0u, 0x00f31bbfu);
		} else {
			M32(bp - 0x60) = 0; M32(bp - 0x64) = 0;
			BCALL(bp - 0x68, 0x00eabc60u, 0x00f31bcau);
		}
	} else if ((M16(L) >> 2) & 1u && M32(th + 0x44)) {
		u32 e;
		A = M32(th + 0x44);
		e = (u32)(s32)(s8)M8(A + 0xf);
		BW8(p + 0x840, 1);
		BW32(p + 0x844, 3);
		BW32(p + 0x848, e);
		qc_f31_rs(c, bp, L, p, e, L, 2, 0x00f31c57u, 0x00f31c73u);
		A = M32(th + 0x44);
		c->ecx = th; c->edx = A;
		if (M32(A + 8)) {
			B = M32(th + 0x44);
			arg = (M8(B + 3) & 0x20u) ? B + 0x10 : 0;
			M32(bp - 0x60) = arg; M32(bp - 0x64) = 0;
			B = M32(th + 0x44);
			c->eax = th; c->edx = arg; c->ecx = M32(B + 8);
			BCALL(bp - 0x68, 0x00f4a1f0u, 0x00f31cbdu);
		} else {
			M32(bp - 0x60) = 0; M32(bp - 0x64) = 0;
			BCALL(bp - 0x68, 0x00eabc60u, 0x00f31cc8u);
		}
	} else {
		BW8(p + 0x840, 0);
		BW32(p + 0x844, 0);
		BW32(p + 0x848, 0);
		M32(bp - 0x60) = 0; M32(bp - 0x64) = 0;
		c->eax = p; c->ecx = p; c->edx = p;
		BCALL(bp - 0x68, 0x00eabc60u, 0x00f31cfau);
		M32(bp - 0x60) = 0;
		BCALL(bp - 0x64, 0x00eabdb0u, 0x00f31d04u);
	}
	/* p->+0x884 == 0.0 (jnp after lahf / test ah, 0x44: ZF alone), else p->+0x894 == 0.0 */
	c->edx = p;
	c->x[0] = ex_ldss(p + 0x884);
	fl = ex_ucomiss(c->x[0], ex_ldss(0x025a3c20u));
	c->eax = qc_lahf(c->eax, fl);
	if (fl != 1u) {
		c->eax = p;
		c->x[0] = ex_ldss(p + 0x894);
		fl = ex_ucomiss(c->x[0], ex_ldss(0x025a3c20u));
		c->eax = qc_lahf(c->eax, fl);
	}
	if (fl == 1u) qc_f31_rs(c, bp, L, c->eax, L, p, 3, 0x00f31d7au, 0x00f31d96u);
	M32(bp - 0x60) = p;
	c->eax = p; c->ecx = M32(0x0286719cu);
	BCALL(bp - 0x64, 0x00f08f70u, 0x00f31da8u);
	c->eax = (c->eax & ~0xffu) | 1u;
	B_RET(sp, 1);
}
#endif


/* 00f4a3c0 helpers. qc_psc: the device's SetPixelShaderConstantF (vtable +0x1b4 of [0283eac0], stdcall: device,
 * register, data, 1) as the wrapper w makes it, inlined (w entered with esp E: its return address at E; the register
 * = [[[028670bc] + 8] + h * 8 - 4]); esi saved and restored through the stack as the wrapper does.
 *  w 1: 00c27cf0 (thiscall; push esi; the call at E - 24: eax the device, ecx the register, edx the method, esi h)
 *  w 2: 00c28790 (cdecl; push esi; the call at E - 24: eax the device, ecx the register, edx the method, esi the
 *       table; the caller sets al = 1 after)
 * (the third wrapper, 00c28280, is called only on the O->+0x30 path: the translation's, see qc_f4a_ok) */
static inline void qc_psc(XCtx *c, u32 E, int w, u32 h, u32 ptr) {
	u32 tb = M32(M32(0x028670bcu) + 8), dev = M32(0x0283eac0u), m = M32(M32(dev) + 0x1b4), reg = M32(tb + h * 8 - 4), at = E - 24;
	M32(E - 4) = c->esi;
	c->eax = dev; c->ecx = reg; c->edx = m; c->esi = w == 1 ? h : tb;
	M32(at + 4) = dev; M32(at + 8) = reg; M32(at + 12) = ptr; M32(at + 16) = 1;
	if (w == 1) BCALL(at, m, 0x00c27d13u);
	else BCALL(at, m, 0x00c287beu);
	c->esi = M32(E - 4);
}
/* fld / fstp (00455840 Vec4 copy, x87_copy) of 4 lanes: a signalling NaN gets its quiet bit */
static inline X128 qc_q4(X128 a) {
	X128 r;
	__m128i t = _mm_and_si128(a.vi, _mm_set1_epi32(0x7fffffff));
	__m128i m = _mm_and_si128(_mm_cmpgt_epi32(t, _mm_set1_epi32(0x7f800000)), _mm_cmpgt_epi32(_mm_set1_epi32(0x7fc00000), t));
	r.vi = _mm_or_si128(a.vi, _mm_and_si128(m, _mm_set1_epi32(0x00400000)));
	return r;
}
static inline X128 qc_splat(u32 b) { X128 r; r.vi = _mm_set1_epi32((int)b); return r; }
static inline X128 qc_v4(u32 x, u32 y, u32 z, u32 w) { X128 r; r.vi = _mm_set_epi32((int)w, (int)z, (int)y, (int)x); return r; }
/* 00515bb0 Vec4::Set(x, y, z, w) of rdata constants (fld / fstp pushes: plain copies) into a static */
static inline void qc_set4(XCtx *c, u32 a, u32 x, u32 y, u32 z, u32 w) { BW32(a, x); BW32(a + 4, y); BW32(a + 8, z); BW32(a + 12, w); }
/* 00f4a3c0: the paths ours makes (the only ones the battle + field coverage runs reached); anything else goes to the
 * translation at the entry, before any change. Ours: a1 != 0 (no assert); a2 0 or 1, and a2 == 0 when O->+0xc is
 * set; O->+0x30 clear (no 00c28280 / 00c27cf0 block); M->+4 >= 2 (both k run); for k 0 and 1 the n = 4 pattern
 * (O[k * 32 + 0x44 / 0x48 / 0x4c] set, O[k * 32 + 0x50] clear); every handle used set (O->+0x10 .. +0x2c (the
 * matrix gate and rows), O->+0x40 .. +0x4c and +0x60 .. +0x6c (the 4 offsets), O->+4, O->+8); M->+0x148 not > 0.0
 * (comiss). Each of these is data the original reads; the only code it runs before its own test is the
 * device's SetPixelShaderConstantF (the D3D proxy 00ec29b0 -> the HLE device, which stores the constants in host
 * state and writes no guest memory beyond its stack arguments), so the value tested here is the value it tests. */
static inline int qc_f4a_ok(u32 a1, u32 a2, u32 O, u32 M) {
	X128 h0, h1, h2, h3;
	__m128i z = _mm_setzero_si128(), m;
	if (!a1 || a2 > 1 || M32(O + 0x30) || (s32)M32(M + 4) < 2 || !M32(O + 4) || !M32(O + 8) || (a2 && M32(O + 0xc))) return 0;
	if (M32(O + 0x50) || M32(O + 0x70)) return 0;
	XLD(h0, O + 0x10); XLD(h1, O + 0x20); XLD(h2, O + 0x40); XLD(h3, O + 0x60);
	m = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi32(h0.vi, z), _mm_cmpeq_epi32(h1.vi, z)), _mm_or_si128(_mm_cmpeq_epi32(h2.vi, z), _mm_cmpeq_epi32(h3.vi, z)));
	if (_mm_movemask_epi8(m)) return 0;
	return (ex_comiss(ex_ldss(M + 0x148), ex_ldss(0x025a3c20u)) & 5u) != 0;
}

/* 00f4a3c0: cdecl void (u8 a1, int a2, O, const Vec4 *col, M, const u16 *cnt, float f1, float f2), ret (/Od; one
 * caller, 00f00432 (in the virtual 00f00130), reads nothing after). Sets pixel shader constants (through the wrappers
 * 00c27cf0 / 00c28790, inlined: qc_psc) from M, on the paths qc_f4a_ok admits (else the translation):
 *  k = 0, 1 (M->+4 >= 2): n = 4 offsets from the static table 02867370 (initialised once: flag bit 4 of 02867400),
 *  col4 = (.5, .5, 0, 0), inv = 1.0 / (float)cnt[k]; for j < 4: c = q(q(q(tab[j] * inv) * M->+0x1ec) + col4) (q: the
 *  x87_copy of 00455840) to O[k * 32 + 0x40 + j * 4]. Then R = static 028672b0 (flag bit 0x10: rows (.5,0,0,0)
 *  (0,K,0,0) (0,0,1,0) (0,0,0,1), 0 = [025a3c20]) Mul (00456970: m44_transform) M[k * 0x44 + 8] (16 dwords); rows
 *  0..2 to O[k * 4 + 0x10 / 0x18 / 0x20] (00c27cf0), row 3 to O[k * 4 + 0x28] (the device call made directly).
 *  O->+0xc: col to it (00c28790; a2 == 0 there). t = 0; O->+4: (.xy, 1, col.w (* M->+0x14c if a2 == 1)), a2 == 1:
 *  .xy = M->+0x1f0 / +0x1f4, t = clamp((f1 - M->+0x1f0) / M->+0x1f4, 0, 1) (comiss: NaN kept); a2 == 0: .xy =
 *  M->+0x120 / +0x1e8; !M->+0x204 (byte): .xy = ([026511e4], 1.0); to O->+4. O->+8: (M->+0x11c, M->+0x120, f2, t)
 *  to it. Writes: the statics' first-use inits; the data vectors in the original's frame (the device reads them).
 *  eax as the original leaves it (the last 00c28790's al = 1 over the method's eax), ecx the /GS cookie.
 *  Callees: D3D constant setters (repeatable) */
B_FN(00f4a3c0, "shader constants (material)", 0) {
	u32 sp = c->esp, bp = sp - 4, A = (bp & ~15u) - 0x330, E = A - 12;
	u32 a1 = M8(sp + 4), a2 = M32(sp + 8), O = M32(sp + 0xc), col = M32(sp + 0x10), M = M32(sp + 0x14), cnt = M32(sp + 0x18);
	u32 k, h, t, fg;
	if (!qc_f4a_ok(a1, a2, O, M)) { orig(c); return; }
	B_ENTER(bp);
	for (k = 0; k < 2; k++) { /* (M->+4 >= 2: both k) */
		u32 ob = O + k * 32, j, K5 = M32(0x0109b130u), Ka = M32(0x010a1458u), Kb = M32(0x010b2774u), Kc = M32(0x011552acu);
		X128 col4 = qc_v4(K5, K5, 0, 0), inv4;
		/* n = 4 (O[k * 32 + 0x44 .. 0x4c] set, +0x50 clear) */
		if (!((fg = M32(0x02867400u)) & 4)) {
			BW32(0x02867400u, fg | 4);
			qc_set4(c, 0x02867370u, Ka, Kb, 0, 0); qc_set4(c, 0x02867380u, Kb, K5, 0, 0);
			qc_set4(c, 0x02867390u, K5, Kc, 0, 0); qc_set4(c, 0x028673a0u, Kc, Ka, 0, 0);
		}
		inv4 = qc_splat(ex_divss(ex_ldss(0x010aa950u), ex_cvtsi2ss(ex_ss(0), (s32)M16(cnt + 2 * k))).u[0]);
		for (j = 0; j < 4; j++) {
			X128 r = qc_q4(ex_mulps(ex_ld(0x02867370u + 16 * j), inv4));
			r = qc_q4(ex_mulps(r, qc_splat(M32(M + 0x1ec))));
			r = qc_q4(ex_addps(r, col4));
			XST(A + 0x270, r);
			qc_psc(c, E, 1, M32(ob + 0x40 + 4 * j), A + 0x270); /* (the handle set) */
		}
		if (!((fg = M32(0x02867400u)) & 0x10)) {
			u32 z = M32(0x025a3c20u), one = M32(0x010aa950u);
			BW32(0x02867400u, fg | 0x10);
			qc_set4(c, 0x028672b0u, K5, z, z, z); qc_set4(c, 0x028672c0u, z, Ka, z, z);
			qc_set4(c, 0x028672d0u, z, z, one, z); qc_set4(c, 0x028672e0u, z, z, z, one);
		}
		{ /* (O[k * 4 + 0x10 / 0x18 / 0x20 / 0x28] set) */
			X128 r4[4];
			u32 nb = M + k * 0x44 + 8, i, dev, vt, m;
			XLD(r4[0], 0x028672b0u); XLD(r4[1], 0x028672c0u); XLD(r4[2], 0x028672d0u); XLD(r4[3], 0x028672e0u);
			for (i = 0; i < 4; i++) { X128 o = m44_transform(r4, ex_ld(nb + 16 * i)); XST(A + 0x128 + 16 * i, o); }
			qc_psc(c, E, 1, M32(O + k * 4 + 0x10), A + 0x128);
			qc_psc(c, E, 1, M32(O + k * 4 + 0x18), A + 0x138);
			qc_psc(c, E, 1, M32(O + k * 4 + 0x20), A + 0x148);
			h = M32(O + k * 4 + 0x28);
			dev = M32(0x0283eac0u); vt = M32(dev); m = M32(vt + 0x1b4);
			M32(A - 16) = dev; M32(A - 12) = M32(M32(M32(0x028670bcu) + 8) + h * 8 - 4); M32(A - 8) = A + 0x158; M32(A - 4) = 1;
			c->eax = vt; c->ecx = dev; c->edx = m;
			BCALL(A - 20, m, 0x00f4b353u);
		}
	}
	c->eax = O;
	if ((h = M32(O + 0xc)) != 0) { /* (a2 == 0 here: no w * M->+0x14c) */
		M32(A + 0x5c) = M32(col); M32(A + 0x60) = M32(col + 4); M32(A + 0x64) = M32(col + 8); M32(A + 0x68) = M32(col + 0xc);
		qc_psc(c, E, 2, h, A + 0x5c);
		c->eax = (c->eax & ~0xffu) | 1u;
	}
	t = M32(0x025a3c20u);
	{ /* (O->+4 set; a1 != 0: z = 1.0; a2 0 or 1) */
		u32 w = M32(col + 0xc), x, y, z = M32(0x010aa950u);
		if (a2 == 1) {
			X128 tt;
			w = ex_mulss(ex_ss(w), ex_ldss(M + 0x14c)).u[0];
			x = M32(M + 0x1f0); y = M32(M + 0x1f4);
			tt = ex_divss(ex_subss(ex_ldss(sp + 0x1c), ex_ldss(M + 0x1f0)), ex_ldss(M + 0x1f4));
			if (!(ex_comiss(ex_ldss(0x025a3c20u), tt) & 5u)) tt = ex_ldss(0x025a3c20u);
			if (!(ex_comiss(tt, ex_ldss(0x010aa950u)) & 5u)) tt = ex_ldss(0x010aa950u);
			t = tt.u[0];
		} else { x = M32(M + 0x120); y = M32(M + 0x1e8); }
		if (!M8(M + 0x204)) { x = M32(0x026511e4u); y = M32(0x010aa950u); }
		M32(A + 0x38) = x; M32(A + 0x3c) = y; M32(A + 0x40) = z; M32(A + 0x44) = w;
		qc_psc(c, E, 2, M32(O + 4), A + 0x38);
		c->eax = (c->eax & ~0xffu) | 1u;
	}
	{ /* (O->+8 set; M->+0x148 not > 0, M->+4 != 1: .x = M->+0x11c) */
		M32(A + 0x10) = M32(M + 0x11c); M32(A + 0x14) = M32(M + 0x120); M32(A + 0x18) = M32(sp + 0x20); M32(A + 0x1c) = t;
		qc_psc(c, E, 2, M32(O + 8), A + 0x10);
		c->eax = (c->eax & ~0xffu) | 1u;
	}
	c->ecx = M32(0x026e5a84u);
	B_RET(sp, 0);
}
/* ---- battle batch 3, part D: 00f87990, 00dc6330, 00c304e0 */

/* part-local helpers: a lane broadcast / zero / four lanes (SSE2 integer: exact bit moves, no stack round trip), the
 * SSE2 float <-> double conversions (inline asm: cvtss2sd dst, src keeps dst's upper 64 bits; cvtsd2ss dst, src keeps
 * dst's lanes 1..3) */
static inline X128 qd_bc(u32 b) { X128 r; r.vi = _mm_set1_epi32((int)b); return r; }
static inline X128 qd_zero(void) { X128 r; r.vi = _mm_setzero_si128(); return r; }
static inline X128 qd_v4(u32 x, u32 y, u32 z, u32 w) { X128 r; r.vi = _mm_set_epi32((int)w, (int)z, (int)y, (int)x); return r; }
static inline X128 qd_cvtss2sd(X128 d, X128 s) { __asm__("cvtss2sd %1, %0" : "+x"(d.v) : "x"(s.v)); return d; }
static inline X128 qd_cvtsd2ss(X128 d, X128 s) { __asm__("cvtsd2ss %1, %0" : "+x"(d.v) : "x"(s.v)); return d; }

/* the device's SetPixelShaderConstantF (vtbl + 0x1b4, stdcall (dev, reg, data, 1)) called at `at` (the return
 * address slot) with return address ra, the device [0283eac0] read as the original does. One inline cache for all
 * the sites (one target). Registers at the call as the site has them (a COM method reads none): pattern 0: eax = the
 * method, ecx = the vtable, edx = dev; 1: eax = the vtable, ecx = dev, edx = the method; 2: eax = dev, ecx = the
 * method, edx = the vtable */
static inline void qd_setps(XCtx *c, u32 at, u32 reg, u32 data, u32 ra, int pat) {
	u32 d = M32(0x0283eac0u), vt = M32(d), m = M32(vt + 0x1b4u);
	M32(at + 4) = d; M32(at + 8) = reg; M32(at + 12) = data; M32(at + 16) = 1;
	if (pat == 1) { c->eax = vt; c->ecx = d; c->edx = m; } else if (pat == 2) { c->eax = d; c->ecx = m; c->edx = vt; } else { c->eax = m; c->ecx = vt; c->edx = d; }
	BCALL(at, m, ra);
}
/* the register of constant slot r: [[028670bc] + 8][r * 8 - 4] (the table re-read as the original does) */
static inline u32 qd_psreg(u32 r) { return M32(M32(M32(0x028670bcu) + 8u) + r * 8u - 4u); }
/* 00c27cf0 (thiscall (i, data) on tbl + 8, ret 8: dev->SetPixelShaderConstantF([tbl->+8][i * 8 - 4], data, 1)),
 * done here as its fast override xo_00c27cf0 does: called at `at`, its push esi at at - 4, the device call's
 * arguments at at - 20 .. - 8, return address 00c27d13; eax = dev, ecx = the register, edx = the method, esi = i at
 * the call; esi restored after */
static inline void qd_c27cf0(XCtx *c, u32 at, u32 i, u32 data) {
	u32 tb = M32(M32(0x028670bcu) + 8u), d = M32(0x0283eac0u), m = M32(M32(d) + 0x1b4u), reg = M32(tb + i * 8u - 4u), es = c->esi;
	M32(at - 8) = 1; M32(at - 12) = data; M32(at - 16) = reg; M32(at - 20) = d;
	c->eax = d; c->ecx = reg; c->edx = m; c->esi = i;
	BCALL(at - 24, m, 0x00c27d13u);
	c->esi = es;
}

/* 00f87990: thiscall (r, p), ret 8 (/Od; frame S = ((entry esp - 4) & ~15) - 0x350): the pixel shader constants of
 * this's lights. n0 / n1 / n2 = the bit counts (00effb30) of nibbles 0 / 1 / 2 of p->+0x20; one4 = (1, 1, 1, 1);
 * counters c0 = c1 = c2 = 0. For each i < this->+8 (signed), o = this->+0xc[i] (o->+0xfc != 'LTND': the engine's
 * assertion): A = o->ee32e0() at S+0x2a0, B = o->ee31e0() at S+0x290 (getters, called), C = o->+0xbc * splat(o->+0xcc),
 * D = 0; k = o->+0xdc (p->+0x24 bit 4, k == 1, c0 >= n0, c1 < n1: k = 3, B = -A * [010ac040]). By k -- 2 (spot, two
 * cos) / 3: the slot groups this->[c2 * 16 + 0xbc..] / [c1 * 12 + 0x8c..]. Else: c0 < n0 and this->[c0 * 8 + 0x6c]:
 * A.w = C.w = [025a3c20], slots this->[c0 * 8 + 0x6c / 70] = C / A (00c27cf0, done here); c0++. i == 0: one4 lane 0 =
 * [025a3c20]. r->+0x8fc: a debug copy to r->+0xa38 + 0x4c i. After: slots j = c0..3 of [j * 8 + 0x6c / 70], c1..3 of
 * [j * 12 + 0x8c..], c2..3 of [j * 16 + 0xbc..] (a group only when its first slot is set) get 0; r->+0x8fc:
 * r->+0xb68 = this->+0x1c; slot this->+0x140 = this->+0x1c; slot this->+0x16c = one4 (00c28790, done here). A slot:
 * 0 skips; else dev->SetPixelShaderConstantF([[028670bc]->+8][slot * 8 - 4], data, 1), the data a local at the
 * original's address. No result (the caller 00f85b20 returns its eax: void); ecx = the cookie.
 * Done here: only the paths the battle / field checks ran -- every light 'LTND' with k not 2 / 3, this->+8 <= n0,
 * slots [i * 8 + 0x6c / 70] (i < this->+8) both set, the later [j * 8 + 0x6c] groups complete (0x70 set when 0x6c
 * is), no [j * 12 + 0x8c] / [j * 16 + 0xbc] slot, r->+0x8fc clear; anything else: the translation, decided at the
 * entry before any change (the entry pre-scan below). Then c0 == i < n0 at light i (every earlier light took the
 * else path and counted), so the k = 3 forcing (needs c0 >= n0) cannot happen and c1 = c2 = 0, one4 = ([025a3c20] or
 * 1, 1, 1, 1). The pre-scan reads only what the original reads (this->+8, this->+0xc[i], o->+0xfc / +0xdc, the slot
 * table this->+0x6c..0xc8, r->+0x8fc, p->+0x20) and nothing the original calls before its own tests can change it:
 * ee32e0 / ee31e0 (and their callees ee31c0, 457750, ee55e0, 7279f0, 457540) write only their outputs in the
 * frame, SetPixelShaderConstantF is the native D3D proxy (writes no guest memory). Callees repeatable. */
B_FN(00f87990, "light shader constants", 0) {
	u32 sp = c->esp, th = c->ecx, a1 = M32(sp + 4), a2 = M32(sp + 8), bp = sp - 4, S = (bp & ~15u) - 0x350u;
	u32 n0 = (u32)__builtin_popcount(M32(a2 + 0x20) & 0xfu), L = M32(th + 8), c0, i, j, r, v0 = 0x3f800000u;
	X128 z = qd_zero();
	/* the entry pre-scan (no change made yet) */
	if (M8(a1 + 0x8fc) || (s32)L > (s32)n0) { orig(c); return; }
	for (j = 0; j < 4; j++) {
		u32 s6c = M32(th + j * 8u + 0x6c), s70 = M32(th + j * 8u + 0x70);
		if ((s32)j < (s32)L ? (!s6c || !s70) : (s6c && !s70)) { orig(c); return; }
		if (M32(th + j * 12u + 0x8c) || M32(th + j * 16u + 0xbc)) { orig(c); return; }
	}
	for (i = 0; (s32)i < (s32)L; i++) {
		u32 o = M32(th + 0xcu + i * 4u), k = M32(o + 0xdc);
		if (M32(o + 0xfc) != 0x4c544e44u || k == 2 || k == 3) { orig(c); return; }
	}
	B_ENTER(bp);
	for (i = 0; (s32)i < (s32)L; i++) {
		u32 o = M32(th + 0xcu + i * 4u);
		M32(S - 4) = S + 0x2a0u;
		c->eax = S + 0x2a0u; c->ecx = o;
		BCALL(S - 8, 0x00ee32e0u, 0x00f87b77u);
		M32(S - 4) = S + 0x290u;
		c->ecx = o;
		BCALL(S - 8, 0x00ee31e0u, 0x00f87b8bu);
		XST(S + 0x230u, ex_mulps(ex_ld(o + 0xbc), qd_bc(M32(o + 0xcc))));
		/* k not 2 / 3, c0 == i < n0, this->[i * 8 + 0x6c] and [i * 8 + 0x70] set (the pre-scan) */
		M32(S + 0x2acu) = M32(0x025a3c20u);
		M32(S + 0x23cu) = M32(0x025a3c20u);
		qd_c27cf0(c, S - 0xcu, M32(th + i * 8u + 0x6c), S + 0x230u);
		qd_c27cf0(c, S - 0xcu, M32(th + i * 8u + 0x70), S + 0x2a0u);
		if (!i) v0 = M32(0x025a3c20u);
	}
	c0 = (s32)L > 0 ? L : 0;
	for (j = c0; (s32)j < 4; j++) {
		if (!(r = M32(th + j * 8u + 0x6c))) continue;
		XST(S + 0x104u, z); qd_setps(c, S - 0x14u, qd_psreg(r), S + 0x104u, 0x00f885c8u, 0);
		r = M32(th + j * 8u + 0x70); /* (set: the pre-scan) */
		XST(S + 0xe8u, z); qd_setps(c, S - 0x14u, qd_psreg(r), S + 0xe8u, 0x00f88646u, 1);
	}
	/* (c1 = c2 = 0 and no [j * 12 + 0x8c] / [j * 16 + 0xbc] slot: those groups set nothing; r->+0x8fc clear) */
	if ((r = M32(th + 0x140))) qd_setps(c, S - 0x14u, qd_psreg(r), th + 0x1cu, 0x00f88b22u, 2);
	if ((r = M32(th + 0x16c))) {
		/* 00c28790 (cdecl (r, &one4) at S - 0xc), done here: push esi, the device call's arguments, return address
		 * 00c287be; eax = dev, ecx = the register, edx = the method, esi = the table; al = 1 after */
		u32 at = S - 0xcu, tb = M32(M32(0x028670bcu) + 8u), d = M32(0x0283eac0u), m = M32(M32(d) + 0x1b4u), reg = M32(tb + r * 8u - 4u), es = c->esi;
		M32(S + 0x320u) = v0; M32(S + 0x324u) = 0x3f800000u; M32(S + 0x328u) = 0x3f800000u; M32(S + 0x32cu) = 0x3f800000u;
		M32(at - 8) = 1; M32(at - 12) = S + 0x320u; M32(at - 16) = reg; M32(at - 20) = d;
		c->eax = d; c->ecx = reg; c->edx = m; c->esi = tb;
		BCALL(at - 24, m, 0x00c287beu);
		c->esi = es;
		c->eax = (c->eax & ~0xffu) | 1u;
	}
	c->ecx = M32(0x026e5a84u); /* (the cookie check's ecx) */
	B_RET(sp, 2);
}

/* 00dc6330 helpers. A point / direction by three rows, the two operand orders the code uses (splats first in each
 * mulps): yx: (v.y * b + v.x * a) + v.z * c; xy: (v.x * a + v.y * b) + v.z * c */
static inline X128 qd_yx(X128 v, X128 a, X128 b, X128 cc) {
	return ex_addps(ex_addps(ex_mulps(ex_shufps(v, v, 0x55), b), ex_mulps(ex_shufps(v, v, 0x00), a)), ex_mulps(ex_shufps(v, v, 0xaa), cc));
}
static inline X128 qd_xy(X128 v, X128 a, X128 b, X128 cc) {
	return ex_addps(ex_addps(ex_mulps(ex_shufps(v, v, 0x00), a), ex_mulps(ex_shufps(v, v, 0x55), b)), ex_mulps(ex_shufps(v, v, 0xaa), cc));
}
/* dot4 as the shuffles add it: (v.55 + v) + v.aa + v.ff (lane 0 the sum) / (v.4e + v.39) + (v.93 + v) (every lane) */
static inline X128 qd_dot4a(X128 v) { return ex_addps(ex_addps(ex_addps(ex_shufps(v, v, 0x55), v), ex_shufps(v, v, 0xaa)), ex_shufps(v, v, 0xff)); }
static inline X128 qd_dot4b(X128 v) { return ex_addps(ex_addps(ex_shufps(v, v, 0x4e), ex_shufps(v, v, 0x39)), ex_addps(ex_shufps(v, v, 0x93), v)); }
/* lane 0 alone (movss from memory: lanes 1..3 zero), cvtps2pd */
static inline X128 qd_lo(X128 v) { X128 r; r.vi = _mm_and_si128(v.vi, _mm_set_epi32(0, 0, 0, -1)); return r; }
static inline X128 qd_cvtps2pd(X128 a) { X128 r; __asm__("cvtps2pd %1, %0" : "=x"(r.v) : "x"(a.v)); return r; }
/* (v.xyz, w): shufps(v, unpckhps(v, w4), 0xc4) -- lane 3 from w4's lane 3 */
static inline X128 qd_w(X128 v, X128 w4) { return ex_shufps(v, ex_unpckhps(v, w4), 0xc4); }
/* 00d95d40 (cdecl (out, a, b), optimized, a leaf: done here): the rotation taking b to a -- n = normalize(cross as the
 * shuffles make it, w = b.w) (rsqrtps, cmpltps mask), cc = clamp(dot4(a * b), -1, 1) by the comiss branches, s =
 * -sqrt(max(1 - cc^2, 0)) (xorps [0109c600]); rows (1 - cc) n.x n + .., .. as the original's shuffles; row 3 =
 * [0109f810]. Rows to o[0..3] */
static inline void qd_d95d40(X128 a, X128 b, X128 *o) {
	X128 z = qd_zero(), one = ex_ss(M32(0x010aa950u)), x0, x1, x2, x3, x4, x5, x6, n, m;
	x1 = ex_mulps(ex_shufps(a, a, 0xc9), ex_shufps(b, b, 0xd2));
	x0 = ex_subps(ex_mulps(ex_shufps(a, a, 0xd2), ex_shufps(b, b, 0xc9)), x1);
	x4 = ex_shufps(x0, ex_unpckhps(x0, b), 0xc4);
	x1 = qd_dot4b(ex_mulps(x4, x4));
	x2 = ex_mulps(a, b);
	m = ex_cmpps(z, x1, 1);
	x1 = ex_mulps(ex_rsqrtps(x1), x4);
	n = ex_orps(ex_andps(x1, m), ex_andnps(m, x4));
	x0 = qd_lo(qd_dot4a(x2)); /* (stored, movss back) */
	if (ex_comiss(ex_subss(one, x0), z) & 4) x0 = one;              /* (jae) */
	x4 = ex_ss(M32(0x0109f7c0u));
	if (!(ex_comiss(ex_subss(x0, x4), z) & 4)) x4 = x0;           /* (jb) */
	x0 = ex_subss(one, ex_mulss(x4, x4));
	if (ex_comiss(x0, z) & 4) x0 = z;                               /* (jae) */
	x0 = ex_sqrtss(x0, x0);
	x0 = ex_xorps(x0, ex_ld(0x0109c600u));
	x2 = ex_subss(one, x4);
	x4 = ex_shufps(x4, x4, 0x00);
	x0 = ex_shufps(x0, x0, 0x00);
	x0 = ex_mulps(x0, n);
	x3 = ex_subps(z, x0);
	x0 = ex_shufps(x0, x0, 0xd2);
	x5 = x0;
	x3 = ex_shufps(x3, x3, 0xc9);
	x0 = ex_shufps(x4, x5, 0x44);
	x4 = ex_shufps(x4, x5, 0xee);
	x5 = ex_shufps(x3, z, 0x44);
	x6 = ex_shufps(x3, z, 0xee);
	x4 = ex_shufps(x4, x6, 0x88);
	x3 = ex_shufps(x0, x5, 0x88);
	x2 = ex_shufps(x2, x2, 0x00);
	x2 = ex_mulps(x2, n);
	x0 = ex_shufps(x0, x5, 0xdd);
	o[0] = ex_addps(ex_mulps(ex_shufps(x2, x2, 0x00), n), x3);
	x0 = ex_shufps(x0, x0, 0xd2);
	o[1] = ex_addps(ex_mulps(ex_shufps(x2, x2, 0x55), n), x0);
	o[2] = ex_addps(ex_mulps(ex_shufps(x2, x2, 0xaa), n), ex_shufps(x4, x4, 0xc9));
	o[3] = ex_ld(0x0109f810u);
}

/* 00dc6330: thiscall (s, bool post), ret 8 (optimized code; frame W = ((entry esp - 4) & ~15) - 0x2c0, the esp after
 * the pushes of ebx / esi / edi): the look-at / aim constraints of this (this->+4 of them; o8 = s->+8: E = o8 +
 * o8->+0x34 + 0x20 i (+0 / +4 joint indices j0 / j1), R = o8 + o8->+0x38 + 0xa0 i, P = o8 + o8->+0x30 + 0x90 j0).
 * Nothing when this->+0x14 is 0, this->+0x10 == 0.0 (ucomiss / lahf / jnp) or this->+4 == 0 (< 0: only the end
 * part). Per i with P->+0x80 != -1 and R->+0x74 == 0: V0 = s->vtbl[0x28](j0), V1 = s->vtbl[0x28](j1) (thiscall (out,
 * j), ret 8: into the frame) plus the chain R->+0x70 (k: E_k->+4, next R_k->+0x70) averaged (divps by the count);
 * w = 1 both; M = (this->+8)->vtbl[0x14](P->+0x84) (a matrix); Rinv = its inverse (cross products, rcpps refined
 * once: an exact divide as the translation's rcpps), w lanes 0 / 1; sum of squares of its 16 lanes > 1e10 (or NaN):
 * next i. Q = Rinv * V0 (point), or with this->+0xc and bits 2 / 1 of P->+0x88 (00dc3d20, done here): N =
 * vtbl[0x14](P->+0x80), Q = Rinv * N.r3, V0 = (N.r3.xyz, 1) (bit 0: + N * P->+0x60). E->+0x18 bit 0: V1 =
 * 00dccde0 (00dc34d0's thunk done here: thiscall on o8 + o8->+0x40 + 0x60 (s16)E->+0x1c, (out, &M, &V0, &V1)),
 * stored to P_j1 (00dc3c40, done here: 16 bytes). G = 00d95d40(R->+0x40, normalize(Rinv * (V1 - V0))) (done here);
 * nb = (s16)R->+0x76 > 0: the angle about R->+0x40 (w 0) from the nb points R->+0x78[k] (s->vtbl[0x28]): atan2 of
 * their Rinv offsets from V'1 on G * R->+0x50 / +0x60 (0100f08c) less R->+0x80[k], wrapped by 2pi two ways (mean of
 * the one with the smaller spread), its rotation matrix (cos 0100f5a2, sin 0100eed4) applied to G (00d957e0, done
 * here). G = G * R (rows of R, 00d957e0); this->+0x10 < 1: X = (this->+8)->vtbl[0x20](P->+0x80), without this->+0xc
 * Q = (Q - X.r3) * t + X.r3, G = 00dc2070(&X, &G, t) (cdecl, called). (this->+8)->vtbl[0x24](P->+0x80, &G, &Q).
 * post set: for each of the (u16)o8->+4 records P_k with P_k->+0x80 >= 0: (this->+8)->vtbl[0x34](it). The frame
 * locals the callees read are written at the original's addresses. Writes P_j1's first 16 bytes (the rest: the
 * callees). No result (the caller 00dc37e0 returns its eax: void); ecx = the cookie; ebx / esi / edi as at the call
 * at each call, restored. Callees: getters (vtbl 0x28 / 0x14 / 0x20, dccde0, dc2070, libm), setters (vtbl 0x24 /
 * 0x34): repeatable as far as their code here shows (virtual: not seen). */
#if 0 /* (battle batch 3: 00dc6330 not selected) */
B_FN(00dc6330, "aim constraints", 0) {
	u32 sp = c->esp, th = c->ecx, a1 = M32(sp + 4), bp = sp - 4, W = (bp & ~15u) - 0x2c0u;
	u32 eb0 = c->ebx, es0 = c->esi, ed0 = c->edi, i;
	X128 z = qd_zero(), one4 = qd_bc(M32(0x010aa950u));
	B_ENTER(bp);
	c->esi = a1;
	if (!M8(th + 0x14)) goto out;
	if ((ex_ucomiss(z, ex_ldss(th + 0x10)) & 3u) == 1u) goto out; /* (equal: lahf / test ah, 0x44 / jnp) */
	if (!M32(th + 4)) goto out;
	if ((s32)M32(th + 4) > 0) for (i = 0;; ) {
		u32 o8 = M32(a1 + 8), E1 = M32(o8 + 0x34) + i * 0x20u + o8, j0 = M32(E1), j1 = M32(E1 + 4);
		u32 R = M32(o8 + 0x38) + i * 0xa0u + o8, P0 = M32(o8 + 0x30) + j0 * 0x90u + o8, m80 = M32(P0 + 0x80), ed, nb, m, o;
		X128 V0, V1, r0, r1, r2, r3, R0, R1, R2, R3, Q, G[4], x0, x1, x2, x3, x4, x5, x6, x7;
		if (m80 == 0xffffffffu || M8(R + 0x74)) goto next;
		c->ebx = E1; c->edi = j0 * 0x90u;
		m = M32(M32(a1) + 0x28);
		M32(W - 4) = j0; M32(W - 8) = W + 0x70u;
		c->eax = W + 0x70u; c->ecx = a1; c->edx = m;
		BCALL(W - 0xcu, m, 0x00dc63f3u);
		m = M32(M32(a1) + 0x28);
		M32(W - 4) = j1; M32(W - 8) = W + 0x90u;
		c->eax = m; c->ecx = a1; c->edx = W + 0x90u;
		BCALL(W - 0xcu, m, 0x00dc6409u);
		V1 = ex_ld(W + 0x90u);
		{
			u32 ob = M32(a1 + 8), k = M32(M32(ob + 0x38) + ob + i * 0xa0u + 0x70), cnt = 1;
			while ((s32)k >= 0) {
				u32 arg = M32((k << 5) + M32(ob + 0x34) + ob + 4);
				m = M32(M32(a1) + 0x28);
				M32(W - 4) = arg; M32(W - 8) = W + 0x250u;
				c->eax = W + 0x250u; c->ecx = a1; c->edx = m; c->edi = k;
				BCALL(W - 0xcu, m, 0x00dc6443u);
				x0 = ex_ld(c->eax);
				ob = M32(a1 + 8);
				cnt++;
				V1 = ex_addps(V1, x0);
				k = M32(k * 0xa0u + M32(ob + 0x38) + ob + 0x70);
			}
			if ((s32)cnt > 1) { x0 = ex_cvtsi2ss(z, (s32)cnt); V1 = ex_divps(V1, ex_shufps(x0, x0, 0x00)); }
		}
		V0 = qd_w(ex_ld(W + 0x70u), one4);
		V1 = qd_w(V1, one4);
		o = M32(th + 8); m = M32(M32(o) + 0x14);
		M32(W - 4) = M32(P0 + 0x84); M32(W - 8) = W + 0x210u;
		c->eax = m; c->ecx = o; c->edx = W + 0x210u; c->edi = P0;
		BCALL(W - 0xcu, m, 0x00dc64e7u);
		/* the inverse of M (rows r0..r3) */
		r0 = ex_ld(W + 0x210u); r1 = ex_ld(W + 0x220u); r2 = ex_ld(W + 0x230u); r3 = ex_ld(W + 0x240u);
		x6 = ex_shufps(r1, r1, 0xd2);
		x7 = ex_shufps(r1, r1, 0xc9);
		x5 = ex_shufps(r0, r0, 0xc9);
		x2 = ex_shufps(r0, r0, 0xd2); /* ([W+0x50]) */
		x0 = ex_subps(ex_mulps(x6, x5), ex_mulps(x7, x2));            /* c01 */
		x4 = ex_shufps(r2, r2, 0xd2);
		x1 = ex_shufps(r2, r2, 0xc9);
		x3 = ex_subps(ex_mulps(x4, x7), ex_mulps(x1, x6));            /* c12 */
		x6 = ex_subps(ex_mulps(x1, x2), ex_mulps(x4, x5));            /* c20 */
		x4 = ex_subps(z, r3);
		x2 = ex_addps(ex_mulps(ex_shufps(x0, x0, 0x09), ex_shufps(r2, r2, 0x09)), ex_mulps(x0, r2));
		x2 = ex_addps(x2, ex_mulps(ex_shufps(x0, x0, 0x52), ex_shufps(r2, r2, 0x52)));
		x2 = ex_shufps(x2, x2, 0x00);
		x5 = ex_rcpps(x2);
		x1 = ex_addps(ex_mulps(ex_subps(one4, ex_mulps(x5, x2)), x5), x5);   /* 1 / det */
		x2 = ex_unpcklps(x3, x0);
		x3 = ex_unpckhps(x3, x0);
		x5 = ex_unpcklps(x2, x6);
		x6 = ex_shufps(x6, x6, 0x1e);
		x2 = ex_unpckhps(x2, x6);
		x3 = ex_unpcklps(x3, x6);
		x0 = ex_mulps(ex_shufps(x4, x4, 0x55), x2);
		x2 = ex_mulps(x2, x1);
		x0 = ex_addps(x0, ex_mulps(ex_shufps(x4, x4, 0x00), x5));
		x5 = ex_mulps(x5, x1);
		x6 = ex_mulps(ex_shufps(x4, x4, 0xaa), x3);
		x3 = ex_mulps(x3, x1);
		x0 = ex_addps(x0, x6);
		x1 = ex_mulps(x1, x0);
		R0 = qd_w(x5, z); R1 = qd_w(x2, z); R2 = qd_w(x3, z); R3 = qd_w(x1, one4);
		x6 = ex_addss(ex_addss(ex_addss(qd_dot4a(ex_mulps(R0, R0)), qd_dot4a(ex_mulps(R1, R1))), qd_dot4a(ex_mulps(R2, R2))), qd_dot4a(ex_mulps(R3, R3)));
		if (ex_comiss(ex_ss(M32(0x026121c8u)), x6) & 4) goto next; /* (jb: 1e10 < it, or NaN) */
		ed = P0;
		if (M8(th + 0xc)) {
			u32 pb = P0 + 0x88;
			ed = pb; c->edi = pb;
			if ((M32(pb) & 4u) || (M32(pb) & 2u)) { /* (00dc3d20(2), (1)) */
				X128 N0, N1, N2, n3;
				o = M32(th + 8); m = M32(M32(o) + 0x14);
				M32(W - 4) = m80; M32(W - 8) = W + 0xa0u;
				c->eax = m; c->ecx = o; c->edx = W + 0xa0u;
				BCALL(W - 0xcu, m, 0x00dc675bu);
				n3 = ex_ld(W + 0xd0u);
				Q = ex_addps(qd_yx(n3, R0, R1, R2), R3);
				V0 = qd_w(n3, one4);
				if (M32(pb) & 1u) { /* (00dc3d20(0)) */
					N0 = ex_ld(W + 0xa0u); N1 = ex_ld(W + 0xb0u); N2 = ex_ld(W + 0xc0u);
					V0 = ex_addps(qd_xy(ex_ld(P0 + 0x60), N0, N1, N2), V0);
				}
				goto have_q;
			}
		}
		Q = ex_addps(qd_yx(V0, R0, R1, R2), R3);
	have_q:
		if (M8(E1 + 0x18) & 1u) {
			/* 00dc34d0 (thiscall on s: (out W+0x260, i, &M, &V0, &V1), ret 0x14), its thunk done here: push esi; 00dccde0
			 * (thiscall on o8 + o8->+0x40 + 0x60 (s16)E->+0x1c: (out, &M, &V0, &V1), ret 0x10) with eax = o8, edx = 32 i,
			 * esi = out; its result: out */
			u32 ob = M32(a1 + 8);
			XST(W + 0x70u, V0); XST(W + 0x90u, V1);
			M32(W - 0x20u) = W + 0x90u; M32(W - 0x24u) = W + 0x70u; M32(W - 0x28u) = W + 0x210u; M32(W - 0x2cu) = W + 0x260u;
			c->eax = ob; c->ecx = (u32)(s32)(s16)M16(M32(ob + 0x34) + i * 0x20u + ob + 0x1c) * 0x60u + M32(ob + 0x40) + ob;
			c->edx = i << 5; c->esi = W + 0x260u; c->edi = ed;
			BCALL(W - 0x30u, 0x00dccde0u, 0x00dc3509u);
			c->esi = a1;
			V1 = ex_ld(W + 0x260u);
			V0 = ex_ld(W + 0x70u);
			ob = M32(a1 + 8);
			BW128(M32(ob + 0x30) + j1 * 0x90u + ob, V1); /* (00dc3c40) */
			c->eax = ob;
		}
		{
			X128 Rv = ex_ld(R + 0x40), d, L, mk;
			d = ex_subps(V1, V0);
			L = qd_yx(d, R0, R1, R2);
			x2 = qd_dot4b(ex_mulps(L, L));
			mk = ex_cmpps(z, x2, 1);
			x2 = ex_mulps(ex_rsqrtps(x2), L);
			qd_d95d40(Rv, ex_orps(ex_andps(x2, mk), ex_andnps(mk, L)), G);
			nb = (u32)(s32)(s16)M16(R + 0x76);
			if ((s32)nb > 0) {
				X128 Ut, Vt, B0, s0 = z, s1 = z, q0 = z, q1 = z, pi = z, k1, k2, a, b, inv, ta, axis, uh, cs, omc, sn, sa, sy, sz;
				X128 row0, row1, row2, t, e00, e01, e02, e10, e11, e12, e20, e21, e22;
				u32 k;
				Ut = qd_xy(ex_ld(R + 0x50), G[0], G[1], G[2]);
				Vt = qd_xy(ex_ld(R + 0x60), G[0], G[1], G[2]);
				m = M32(M32(a1) + 0x28);
				M32(W - 4) = j1; M32(W - 8) = W + 0x180u;
				c->eax = m; c->ecx = a1; c->edx = W + 0x180u; c->ebx = R; c->edi = nb;
				BCALL(W - 0xcu, m, 0x00dc6a40u);
				B0 = ex_ld(W + 0x180u);
				for (k = 0; k < nb; k++) {
					u32 ix = (u32)(s32)(s16)M16(R + 0x78 + 2 * k);
					X128 dl, u, v;
					m = M32(M32(a1) + 0x28);
					M32(W - 4) = ix; M32(W - 8) = W + 0x1b0u;
					c->eax = ix; c->ecx = a1; c->edx = m; c->ebx = R + 0x78 + 2 * k; c->edi = R + 0x80 + 4 * k;
					BCALL(W - 0xcu, m, 0x00dc6a95u);
					dl = qd_yx(ex_subps(ex_ld(W + 0x1b0u), B0), R0, R1, R2);
					u = ex_mulps(dl, Ut);
					v = ex_mulps(dl, Vt);
					c->x[0] = qd_cvtps2pd(qd_lo(qd_dot4a(v)));
					c->x[1] = qd_cvtps2pd(qd_lo(qd_dot4a(u)));
					BCALL(W - 4, 0x0100f08cu, 0x00dc6b51u);
					pi = ex_ldss(0x02613ff0u);
					a = ex_subss(qd_cvtsd2ss(c->x[0], c->x[0]), ex_ldss(R + 0x80 + 4 * k));
					x1 = (ex_comiss(a, z) & 4) ? ex_subss(a, pi) : ex_addss(a, pi);
					k1 = ex_ldss(0x026134dcu); k2 = ex_ldss(0x010a1fa0u);
					x1 = ex_mulss(x1, k1);
					x1 = ex_mulss(ex_cvtsi2ss(x1, ex_cvttss2si(x1)), k2);
					a = ex_subss(a, x1);
					b = ex_subss(a, pi);
					x2 = (ex_comiss(b, z) & 4) ? ex_subss(b, pi) : ex_addss(b, pi);
					x2 = ex_mulss(x2, k1);
					x2 = ex_mulss(ex_cvtsi2ss(x2, ex_cvttss2si(x2)), k2);
					b = ex_subss(b, x2);
					q0 = ex_addss(ex_mulss(a, a), q0);
					s0 = ex_addss(a, s0);
					q1 = ex_addss(ex_mulss(b, b), q1);
					s1 = ex_addss(b, s1);
				}
				c->ebx = R; c->edi = nb;
				inv = ex_divss(ex_ss(M32(0x010aa950u)), ex_cvtsi2ss(z, (s32)nb));
				x2 = ex_addss(ex_mulss(inv, s1), pi);
				x1 = (ex_comiss(x2, z) & 4) ? ex_subss(x2, pi) : ex_addss(x2, pi);
				if (!(ex_comiss(ex_subss(q1, q0), z) & 4)) ta = ex_mulss(inv, s0);
				else {
					x1 = ex_mulss(x1, ex_ldss(0x026134dcu));
					ta = ex_subss(x2, ex_mulss(ex_cvtsi2ss(s0, ex_cvttss2si(x1)), ex_ldss(0x010a1fa0u)));
				}
				/* the rotation by ta about axis = (R->+0x40.xyz, 0) */
				uh = ex_unpckhps(Rv, z);
				axis = ex_shufps(Rv, uh, 0xc4);
				c->x[0] = qd_cvtss2sd(uh, ta);
				BCALL(W - 4, 0x0100f5a2u, 0x00dc6cb4u);
				cs = qd_cvtsd2ss(c->x[2], c->x[0]); /* (cvtsd2ss xmm2, xmm0: lanes 1..3 xmm2's after the call, which reach the sin argument's high half) */
				omc = ex_subss(ex_ss(M32(0x010aa950u)), cs);
				t = ex_mulss(omc, axis); row0 = ex_mulps(ex_shufps(t, t, 0x00), axis);
				t = ex_mulss(omc, ex_shufps(axis, axis, 0xaa)); row2 = ex_mulps(ex_shufps(t, t, 0x00), axis);
				t = ex_mulss(omc, ex_shufps(axis, axis, 0x55)); row1 = ex_mulps(ex_shufps(t, t, 0x00), axis);
				e00 = ex_addss(cs, row0);
				e22 = ex_addss(cs, ex_shufps(row2, row2, 0xaa));
				e11 = ex_addss(cs, ex_shufps(row1, row1, 0x55));
				c->x[0] = qd_cvtss2sd(e11, ta);
				BCALL(W - 4, 0x0100eed4u, 0x00dc6d70u);
				sn = qd_cvtsd2ss(c->x[0], c->x[0]);
				sa = ex_mulps(ex_shufps(sn, sn, 0x00), axis);
				sy = ex_shufps(sa, sa, 0x55); sz = ex_shufps(sa, sa, 0xaa);
				e01 = ex_addss(ex_shufps(row0, row0, 0x55), sz);
				e12 = ex_addss(ex_shufps(row1, row1, 0xaa), sa);
				e20 = ex_addss(row2, sy);
				e02 = ex_subss(ex_shufps(row0, row0, 0xaa), sy);
				e10 = ex_subss(row1, sz);
				e21 = ex_subss(ex_shufps(row2, row2, 0x55), sa);
				/* 00d957e0(&G, &rot) (done here): row i = (rot_i.y G1 + rot_i.x G0) + rot_i.z G2, row 3 kept */
				x0 = G[0]; x1 = G[1]; x2 = G[2];
				G[0] = ex_addps(ex_addps(ex_mulps(ex_shufps(e01, e01, 0), x1), ex_mulps(ex_shufps(e00, e00, 0), x0)), ex_mulps(ex_shufps(e02, e02, 0), x2));
				G[1] = ex_addps(ex_addps(ex_mulps(ex_shufps(e11, e11, 0), x1), ex_mulps(ex_shufps(e10, e10, 0), x0)), ex_mulps(ex_shufps(e12, e12, 0), x2));
				G[2] = ex_addps(ex_addps(ex_mulps(ex_shufps(e21, e21, 0), x1), ex_mulps(ex_shufps(e20, e20, 0), x0)), ex_mulps(ex_shufps(e22, e22, 0), x2));
			}
		}
		/* 00d957e0(&G, R) (done here) */
		x0 = G[0]; x1 = G[1]; x2 = G[2];
		G[0] = qd_yx(ex_ld(R), x0, x1, x2);
		G[1] = qd_yx(ex_ld(R + 0x10), x0, x1, x2);
		G[2] = qd_yx(ex_ld(R + 0x20), x0, x1, x2);
		c->ebx = R; c->edi = th;
		if (!(ex_comiss(ex_ss(M32(0x010aa950u)), ex_ldss(th + 0x10)) & 5)) { /* (1 > t: not jbe) */
			u32 r;
			o = M32(th + 8); m = M32(M32(o) + 0x20);
			M32(W - 4) = m80; M32(W - 8) = W + 0xa0u;
			c->eax = W + 0xa0u; c->ecx = o; c->edx = m;
			BCALL(W - 0xcu, m, 0x00dc6eebu);
			if (!M8(th + 0xc)) {
				x3 = ex_ld(W + 0xd0u);
				Q = ex_addps(ex_mulps(ex_subps(Q, x3), qd_bc(M32(th + 0x10))), x3);
			}
			XST(W + 0x130u, G[0]); XST(W + 0x140u, G[1]); XST(W + 0x150u, G[2]); XST(W + 0x160u, G[3]);
			M32(W - 4) = M32(th + 0x10); M32(W - 8) = W + 0x130u; M32(W - 0xcu) = W + 0xa0u; M32(W - 0x10u) = W + 0x270u;
			c->eax = W + 0x130u; c->ecx = W + 0xa0u; c->edx = W + 0x270u;
			BCALL(W - 0x14u, 0x00dc2070u, 0x00dc6f3fu);
			r = c->eax;
			G[0] = ex_ld(r); G[1] = ex_ld(r + 0x10); G[2] = ex_ld(r + 0x20); G[3] = ex_ld(r + 0x30);
		}
		XST(W + 0x130u, G[0]); XST(W + 0x140u, G[1]); XST(W + 0x150u, G[2]); XST(W + 0x160u, G[3]);
		XST(W + 0x120u, Q);
		o = M32(th + 8); m = M32(M32(o) + 0x24);
		M32(W - 4) = W + 0x120u; M32(W - 8) = W + 0x130u; M32(W - 0xcu) = m80;
		c->eax = m; c->ecx = o; c->edx = m80;
		BCALL(W - 0x10u, m, 0x00dc6f90u);
	next:
		i++;
		if ((s32)i >= (s32)M32(th + 4)) break;
	}
	if (M8(sp + 8)) {
		u32 cnt = M32(M32(a1 + 8) + 4) & 0xffffu, e = 0;
		c->ebx = th;
		if (cnt) do {
			u32 ob = M32(a1 + 8), v = M32(M32(ob + 0x30) + e + ob + 0x80);
			if ((s32)v >= 0) {
				u32 o = M32(th + 8), vt = M32(o), m = M32(vt + 0x34);
				M32(W - 4) = v;
				c->eax = m; c->ecx = o; c->edx = vt; c->edi = e;
				BCALL(W - 8, m, 0x00dc6feeu);
			}
			e += 0x90u;
		} while (--cnt);
	}
out:
	c->ecx = M32(0x026e5a84u); /* (the cookie check's ecx) */
	c->ebx = eb0; c->esi = es0; c->edi = ed0;
	B_RET(sp, 2);
}
#endif

/* 00c304e0 helpers */
/* ordered-equal to 0 (ucomiss / lahf / test ah, 0x44: jnp taken) */
static inline int qd_eq0(X128 a) { return (ex_ucomiss(a, qd_zero()) & 3u) == 1u; }
/* 004559b0 Matrix44::Transform (fast override's order, as the original): ((r0 * v.x + r1 * v.y) + r2 * v.z) + r3 * v.w */
static inline X128 qd_xf(const X128 *r, X128 v) {
	X128 a = ex_mulps(r[0], ex_shufps(v, v, 0x00));
	a = ex_addps(a, ex_mulps(r[1], ex_shufps(v, v, 0x55)));
	a = ex_addps(a, ex_mulps(r[2], ex_shufps(v, v, 0xaa)));
	return ex_addps(a, ex_mulps(r[3], ex_shufps(v, v, 0xff)));
}
/* 004fbeb0 (thiscall (out), ret 4, /Od): l = (v^2 + v^2.zxyw) + v^2.yzxw; r = 004e5110(l) (r0 = rsqrtps (exact), ((1 -
 * r0 (r0 l)) (r0 0.5)) + r0); lanes with 0 < l: v r, else v; w = v.w */
static inline X128 qd_norm(X128 v) {
	X128 sq = ex_mulps(v, v), l, r0, r, m, p;
	l = ex_addps(ex_addps(sq, ex_shufps(sq, sq, 0xd2)), ex_shufps(sq, sq, 0xc9));
	r0 = ex_rsqrtps(l);
	r = ex_addps(ex_mulps(ex_subps(qd_bc(0x3f800000u), ex_mulps(r0, ex_mulps(r0, l))), ex_mulps(r0, qd_bc(M32(0x0109b130u)))), r0);
	m = ex_cmpps(qd_zero(), l, 1);
	p = ex_orps(ex_andnps(m, v), ex_andps(m, ex_mulps(v, r)));
	return ex_shufps(p, ex_unpckhps(p, v), 0xc4);
}
/* 00c4ffb0 (cdecl (q, b), a leaf: the x87 result as float bits): q all 0 (ucomiss): 1.0; else L = |b.xyz| (((z^2 + x^2)
 * + y^2), sqrtss), A = q0 L + q1, B = q2 L + q3; q0 = q1 = 0: A = B, else q2 = q3 = 0: B = A; x = B > A ? A : B; 1 <= x
 * (or NaN): 1.0, x > 0: x, else 0 */
static inline u32 qd_c4ffb0(u32 q, X128 b) {
	X128 z = qd_zero(), q0 = ex_ldss(q), q1 = ex_ldss(q + 4), q2 = ex_ldss(q + 8), q3 = ex_ldss(q + 12), sq, l, A, B;
	if (qd_eq0(q0) && qd_eq0(q1) && qd_eq0(q2) && qd_eq0(q3)) return 0x3f800000u;
	sq = ex_mulps(b, b);
	l = ex_addps(ex_addps(ex_shufps(sq, sq, 0xaa), sq), ex_shufps(sq, sq, 0x55));
	l = qd_lo(ex_sqrtss(l, l));
	A = ex_addss(ex_mulss(q0, l), q1);
	B = ex_addss(ex_mulss(q2, l), q3);
	if (qd_eq0(q0) && qd_eq0(q1)) A = B;
	else if (qd_eq0(q2) && qd_eq0(q3)) B = A;
	if (!(ex_comiss(B, A) & 5u)) B = A;
	if (ex_comiss(ex_ss(0x3f800000u), B) & 5u) return 0x3f800000u;
	return (ex_comiss(B, z) & 5u) ? 0u : B.u[0];
}
/* min(max(x, 0), 1) as the comiss branches pick it: 1 <= x or NaN: 1.0, x > 0: x, else 0 */
static inline X128 qd_clamp01(X128 x) {
	if (ex_comiss(ex_ss(0x3f800000u), x) & 5u) return ex_ss(0x3f800000u);
	return (ex_comiss(x, qd_zero()) & 5u) ? qd_zero() : x;
}
static inline double qd_fd(u32 bits) { float f; memcpy(&f, &bits, 4); return (double)f; }
/* the translation's x87 fmul dword / fstp dword: (float)XR87(st0 * (double)m). XR87 under 24-bit precision control
 * is (double)(float)v, and (float)(double)(float)v == (float)v (a float is exact as a double), so the stored float
 * is (float)(st0 * m) whatever the control word: no precision test needed */
static inline u32 qd_x87mul(double st0, u32 m) {
	double d = qd_fd(m);
	float g;
	u32 u;
	__asm__("mulsd %1, %0" : "+x"(st0) : "x"(d));
	g = (float)st0; memcpy(&u, &g, 4);
	return u;
}

/* 00c304e0: cdecl (p), ret (optimized; frame W = ((entry esp - 4) & ~15) - 0xc0, the esp after the pushes of ebx / esi /
 * edi; the real extent runs to c309bb: the block c309b5 after the ret is part of it): a look-at weight. s = [p->+0xc0] *
 * p->+0x2c; f = p->+0xb0 > 0 and p->+0xb4 > 0 and not all of p->+0xa0 / a4 / a8 == 0. s != 0 and (p->+0x53 or
 * p->+0x82 or f): mat = 00c6d9e0(p)->vtbl[0x14]() (00c6d9e0 done here), T = mat * (p->+0xc4)->+0x30 (Transform, done
 * here); p->+0x53: s = (float)(00c4ffb0(p + 0x40, &T) (done here) * s) (x87); f and |T.z| > p->+0xb0: pos = p + 0x20 +
 * p->+0xa0 * clamp01(..) (a frame copy); p->+0x82: a = 010101dd (SSE2 libm, double in xmm0)(dot(norm(T),
 * -norm((mat * p->+0xc4) * (0,0,1,0)))) (Matrix44::Mul, Transform, 004fbeb0 x2 done here) * 025df1c0, a > p->+0x84:
 * s = 0 (a >= lo + range) or s * clamp01(1 - (a - lo) / range) (range == 0: 0). Then: p->+0x52 == 3: 00c4e350(p,
 * p + 0x60, p + 0x70); p->+0x10 = p->+0x30 * pos (16 bytes); o = 00cb9ff0(00c6dd40(p)) (00c6dd40 done here);
 * p->+0x1c = s; o->vtbl[0xc](2) (al): s = (float)(o->vtbl[0x24]() * s) (x87); p->+0x1c = s * p->+0x1c (re-read). No
 * result (callers: tail wrappers / a switch whose other cases return anything): eax / ecx / edx are the last callee's,
 * as the original leaves them; xmm0 mirrored. Callees: getters (vtbl 0x14, cb9ff0, c4e350, libm) and virtual
 * vtbl[0xc] / vtbl[0x24] (not seen): repeatable as far as shown.
 * Done here: only the paths the battle / field checks ran -- p->+0xb0 <= 0 (or NaN: so f = 0, pos = p + 0x20),
 * s != 0, p->+0x53 set (so the main block runs), 00c6d9e0's [..]->+0x18->+3 clear; anything else: the translation,
 * decided at the entry before any change. These are read by the original before its first call (p->+0xb0,
 * [p->+0xc0], p->+0x2c, p->+0x53: c304f4..c3057b) or by its first callee, 00c6d9e0 (its chain: nothing runs before
 * it), so nothing can change them before the original's tests. */
B_FN(00c304e0, "look-at weight", 0) {
	u32 sp = c->esp, p = M32(sp + 4), bp = sp - 4, W = (bp & ~15u) - 0xc0u;
	u32 eb0 = c->ebx, es0 = c->esi, ed0 = c->edi, pv = p + 0x20u, ob, o2, vt, m, n, mat;
	X128 z = qd_zero(), sv, R, r[4], T;
	/* the entry pre-scan (no change made yet) */
	sv = ex_mulss(ex_ldss(M32(p + 0xc0)), ex_ldss(p + 0x2c));
	if (!(ex_comiss(ex_ldss(p + 0xb0), z) & 5u) || qd_eq0(sv) || !M8(p + 0x53)) { orig(c); return; }
	{ /* 00c6d9e0 (cdecl (p), done here) */
		u32 o = M32(p), a = M32(o + 8), d = M32(a + 8), q = M32(a), e = M32(d + 0x18);
		if (M8(e + 3)) { orig(c); return; }
		q = M32(q + 0x10); e = M32(q + 0xc);
		ob = (s32)M32(M32(e + 0xbc) + 0x64) >= 0x10 ? M32(e + 0xb4) : M32(e + 0xb0);
	}
	B_ENTER(bp);
	c->esi = p; c->edi = pv;
	mat = BVGET(ob, 0x14, W - 4, 0x00c305a3u);
	n = M32(p + 0xc4);
	c->ebx = n; c->edi = mat;
	r[0] = ex_ld(mat); r[1] = ex_ld(mat + 0x10); r[2] = ex_ld(mat + 0x20); r[3] = ex_ld(mat + 0x30);
	T = qd_xf(r, ex_ld(n + 0x30));
	/* (c305be: re-read after the getter, as the original; set at the entry) */
	if (M8(p + 0x53)) sv = ex_ss(qd_x87mul(qd_fd(qd_c4ffb0(p + 0x40, T)), sv.u[0])); /* (fld dword: st0 = (double)it) */
	/* (f = 0: c305e0 je c30693) */
	if (M8(p + 0x82)) {
		X128 nr[4], mm[4], N1, N2, pr, a, lo;
		int k;
		nr[0] = ex_ld(n); nr[1] = ex_ld(n + 0x10); nr[2] = ex_ld(n + 0x20); nr[3] = ex_ld(n + 0x30);
		for (k = 3; k >= 0; k--) mm[k] = qd_xf(r, nr[k]); /* (Matrix44::Mul) */
		N2 = qd_norm(qd_xf(mm, qd_v4(0, 0, 0x3f800000u, 0)));
		N1 = qd_norm(T);
		pr = ex_mulps(N1, ex_subps(z, N2));
		a = ex_addps(ex_addps(ex_shufps(pr, pr, 0xaa), pr), ex_shufps(pr, pr, 0x55));
		c->x[0] = qd_cvtps2pd(qd_lo(a));
		BCALL(W - 4, 0x010101ddu, 0x00c307f7u);
		a = ex_mulss(qd_cvtsd2ss(c->x[0], c->x[0]), ex_ldss(0x025df1c0u));
		lo = ex_ldss(p + 0x84);
		if (!(ex_comiss(a, lo) & 5u)) {
			X128 rg = ex_ldss(p + 0x88), v;
			if (!(ex_comiss(a, ex_addss(rg, lo)) & 4u)) sv = z;
			else {
				v = qd_eq0(rg) ? z : ex_subss(ex_ss(0x3f800000u), ex_divss(ex_subss(a, lo), rg));
				sv = ex_mulss(qd_clamp01(v), sv);
			}
		}
	}
	c->edi = pv; /* (c3089e: [W+0x20], never moved: f = 0) */
	if (M8(p + 0x52) == 3) {
		M32(W - 4) = p + 0x70u; M32(W - 8) = p + 0x60u; M32(W - 0xcu) = p;
		c->ecx = p + 0x70u; c->edx = p + 0x60u;
		BCALL(W - 0x10u, 0x00c4e350u, 0x00c308b6u);
	}
	R = ex_mulps(ex_ld(p + 0x30), ex_ld(pv));
	BW128(p + 0x10, R);
	{ /* 00c6dd40 (cdecl (p), done here), then 00cb9ff0 (thiscall) */
		u32 o = M32(p), e = M32(o + 8);
		ob = M32(e);
		c->eax = ob; c->ecx = ob; c->edx = e;
		BCALL(W - 4, 0x00cb9ff0u, 0x00c30971u);
	}
	o2 = c->eax;
	c->edi = o2;
	BW32(p + 0x1c, sv.u[0]);
	vt = M32(o2); m = M32(vt + 0xc);
	M32(W - 4) = 2;
	c->eax = vt; c->ecx = o2; c->edx = m;
	BCALL(W - 8, m, 0x00c30989u);
	if (c->eax & 0xffu) {
		vt = M32(o2); m = M32(vt + 0x24);
		c->eax = vt; c->ecx = o2; c->edx = m;
		BCALL(W - 4, m, 0x00c30996u);
		sv = ex_ss(qd_x87mul(c->st0, sv.u[0]));
	}
	sv = ex_mulss(ex_ss(sv.u[0]), ex_ldss(p + 0x1c)); /* (movss from the frame: lanes 1..3 0) */
	BW32(p + 0x1c, sv.u[0]);
	c->x[0] = sv;
	c->ebx = eb0; c->esi = es0; c->edi = ed0;
	B_RET(sp, 0);
}
/* ---- battle batch 3, part E: 00cd35f0, 00dca630, 00f30350, 00dc4e90 */

/* part E helpers. Animation tracks of 00cd35f0: trk = th->+4->+0->+4 (16 bytes per channel: +0 -> key data, +4 -> the
 * key times (dwords), +0xc the key count, u16); a key time k (bounded: bound < count, else -1) */
static inline u32 qe_trk(u32 th) { return M32(M32(M32(th + 4u)) + 4u); }
static inline u32 qe_kt(u32 tr, u32 bound, u32 idx) { return bound < (u32)M16(tr + 0xcu) ? M32(M32(tr + 4u) + idx * 4u) : 0xffffffffu; }
/* x87 as the translation models it (xrc.h XR87: double, rounded to float under the 24-bit precision control): fld of
 * a float, fstp to a float, fmul */
static inline double qe_f2d(u32 bits) { X128 a = ex_ss(bits); double r; __asm__("cvtss2sd %1, %0" : "=x"(r) : "x"(a.v)); return r; }
static inline u32 qe_d2f(double v) { X128 r; r.vi = _mm_setzero_si128(); __asm__("cvtsd2ss %1, %0" : "+x"(r.v) : "x"(v)); return r.u[0]; }
static inline double qe_r87(double v, int p24) { if (p24) __asm__("cvtsd2ss %0, %0\n\tcvtss2sd %0, %0" : "+x"(v)); return v; }
static inline double qe_mul87(double a, double b, int p24) { __asm__("mulsd %1, %0" : "+x"(a) : "x"(b)); return qe_r87(a, p24); }
static inline int qe_p24(XCtx *c) { return ((c->fcw >> 8) & 3u) == 0; }
/* lanes: zero (xorps), movss reg, reg (lane 0 of b, 1..3 of a), lane 0 only (xorps + movss reg, reg), one dword in
 * all four (movss + shufps 0) */
static inline X128 qe_zero(void) { X128 r; r.vi = _mm_setzero_si128(); return r; }
static inline X128 qe_movss(X128 a, X128 b) { X128 r; r.v = _mm_move_ss(a.v, b.v); return r; }
static inline X128 qe_lo(X128 a) { X128 r; r.v = _mm_move_ss(_mm_setzero_ps(), a.v); return r; }
static inline X128 qe_bc(u32 bits) { X128 r; r.vi = _mm_set1_epi32((int)bits); return r; }

/* 00cd35f0's callback calls: xmm0 and eax / ecx / edx as at the original's call; the callee's back */
#define QE_CB(at_, t_, ra_, ecx_, edx_) do { u32 qt_ = (t_); c->x[0] = x0; c->eax = qt_; c->ecx = (ecx_); c->edx = (edx_); BCALL((at_), qt_, (ra_)); x0 = c->x[0]; } while (0)

/* 00cd35f0: thiscall (P, t0, t1), ret 0xc (optimized code, no ebp frame; F = esp - 0x50, the body's esp): the
 * channels of an animation player th (th->+0x10: channel pointers, n = P->+0->+8 (u16) of them; tracks qe_trk(th),
 * th->+4->+0x1c: s16 key strides per channel) stepped to time t1 (t0 the step's start) through P's descriptor
 * desc = P->+8 callbacks: f24 = desc->+0x24 (cdecl (out, key_b, key_a, float dt, ctx)), f20 = desc->+0x20 (cdecl
 * (out, key, float frac, ctx)). ctx is the original's frame block (0x14 bytes) whose address the callbacks get: F+0x1c
 * (desc->+1 bit 3 clear: "A") or F+0x3c ("B"), written and re-read at the same addresses where the original does.
 * P->+0xc != 0 ("2"): channels with ch->+0x28 == 0 skipped, out = ch + 0x30 (else ch + 0x20). Per channel ch: cur =
 * ch->+0x18 (s16) < 0: (no keys: skipped) ch->+0x18 = 0, f24(out, key0, key0, 0.0f, ctx), first key j = 0; else j =
 * ch->+0x1c + cur, ch->+0x18 = j, ch->+0x1c = 0. Then for each key j + 1 whose time nx <= t1: A f24(out, key(j + 1),
 * key j, (nx - time j) * 0.0002f, ctx); B f20(out, key j, (float)x87((nx - time j) * 0.0002f), ctx) and f24(out,
 * key(j + 1), key j, the same, ctx), ctx->+0xc / +8 / +0x10 the SSE times (nx - t1 + t0) * 0.0002f, t0 * 0.0002f.
 * Finally ch->+0x1c = keys passed, ch->+0x14 = frac = (t1 - time j) * 0.0002f, f20(out, key j, frac, ctx). Times
 * (s32) and their lookups as the original (bounded by the key count, else -1). Writes ch->+0x14 / +0x18 / +0x1c,
 * the frame's ctx and call arguments; the callbacks write the rest. Non-repeatable callees: the callbacks include
 * event firers (00cd6c50, 00cd71a0: they advance out->+0xc and call the event handlers), run twice in the checked
 * mode. eax / ecx / edx and xmm0 left as the original (the callee's after a call; A1 eax = the channel's track
 * offset); no result. */
#if 0 /* (battle batch 3: 00cd35f0 not selected) */
B_FN(00cd35f0, "animation channels: key callbacks up to t1", 0) {
	u32 E = c->esp, F = E - 0x50u, th = c->ecx, P = M32(E + 4), t0 = M32(E + 8), t1 = M32(E + 12);
	u32 n = M16(M32(P) + 8), b24, desc, fl, two, ch4 = 0, ch16 = 0, sidx = 0;
	X128 x0 = c->x[0], K;
	if (n == 0) { c->eax = 0; c->ecx = P; c->esp = E; xrc_stdcall_ret(c, 3); return; }
	b24 = M8(P + 0x24); two = M32(P + 0xc) != 0; desc = M32(P + 8); fl = M8(desc + 1);
	K = ex_ldss(0x025df544u);
	if (!(fl & 8) && !two) { /* A1: ctx = F+0x1c: b24, ch4 (the channel's offset, re-read), nx, out, key(j + 1) */
		const u32 ctx = F + 0x1cu;
		M32(F + 0x1c) = b24; M32(F + 0x20) = 0;
		for (;;) {
			u32 ch = M32(M32(th + 0x10) + M32(F + 0x20)), nk = M16(qe_trk(th) + ch16 + 0xc), out = ch + 0x20, j, jn, start;
			s32 cur = (s16)M16(ch + 0x18);
			M32(F + 0x28) = out;
			if (cur < 0 && !nk) { c->ecx = (u32)cur; c->edx = out; }
			else {
				if (cur < 0) {
					u32 base;
					BW16(ch + 0x18, 0);
					base = M32(M32(qe_trk(th) + ch16));
					M32(F - 4) = ctx; M32(F - 8) = 0; M32(F - 0xc) = base; M32(F - 0x10) = base; M32(F - 0x14) = out;
					start = 0;
					QE_CB(F - 0x18, M32(desc + 0x24), 0x00cd36a5u, ctx, desc);
				} else {
					start = M16(ch + 0x1c) + (u32)cur;
					BW16(ch + 0x18, start); BW16(ch + 0x1c, 0);
				}
				j = start; jn = start + 1;
				while ((s32)j < (s32)(nk - 1)) {
					u32 tr = qe_trk(th) + ch16, nx = qe_kt(tr, jn, j + 1), X, tr2, base, kc, kn, ct;
					s32 st;
					M32(F + 0x24) = nx;
					if ((s32)nx > (s32)t1) break;
					X = M32(th + 4); st = (s16)M16(M32(X + 0x1c) + sidx); tr2 = M32(M32(X) + 4) + ch16; base = M32(M32(tr2));
					kc = (jn - 1) * (u32)st + base; kn = jn * (u32)st + base;
					M32(F + 0x2c) = kn;
					ct = qe_kt(tr2, jn - 1, j);
					x0 = ex_mulss(ex_cvtsi2ss(x0, (s32)(nx - ct)), K);
					M32(F - 4) = ctx; M32(F - 8) = x0.u[0]; M32(F - 0xc) = kc; M32(F - 0x10) = kn; M32(F - 0x14) = out;
					M32(F + 0x28) = out;
					QE_CB(F - 0x18, M32(desc + 0x24), 0x00cd377cu, kn, desc);
					j++; jn++;
				}
				BW16(ch + 0x1c, j - start);
				{
					u32 tr = qe_trk(th) + ch16, ct = qe_kt(tr, j, j), X, base, key, o2;
					s32 st;
					x0 = ex_cvtsi2ss(x0, (s32)(t1 - ct));
					X = M32(th + 4); base = M32(M32(M32(M32(X) + 4) + ch16)); st = (s16)M16(M32(X + 0x1c) + sidx);
					x0 = ex_mulss(x0, K);
					key = (u32)st * j + base;
					o2 = M32(F + 0x28);
					BW32(ch + 0x14, x0.u[0]);
					M32(F - 4) = ctx; M32(F - 8) = x0.u[0]; M32(F - 0xc) = key; M32(F - 0x10) = o2;
					QE_CB(F - 0x14, M32(desc + 0x20), 0x00cd380eu, o2, desc);
				}
			}
			c->eax = ch16;
			if (--n == 0) break;
			M32(F + 0x20) += 4; ch16 += 0x10; sidx += 2;
		}
	} else if (!(fl & 8)) { /* A2: ctx = F+0x1c: b24, the first key j (re-read), ch (re-read), nx, out (re-read) */
		const u32 ctx = F + 0x1cu;
		M32(F + 0x1c) = b24;
		for (;;) {
			u32 ch = M32(M32(th + 0x10) + ch4);
			M32(F + 0x24) = ch;
			if (M32(ch + 0x28) == 0) { c->eax = ch; c->ecx = M32(th + 0x10); c->edx = ch4; }
			else {
				u32 nk = M16(qe_trk(th) + ch16 + 0xc), out = ch + 0x30, j, jn;
				s32 cur = (s16)M16(ch + 0x18);
				M32(F + 0x2c) = out;
				if (cur < 0 && !nk) { c->eax = ch; c->ecx = (u32)cur; c->edx = out; }
				else {
					if (cur < 0) {
						u32 base;
						BW16(ch + 0x18, 0);
						base = M32(M32(qe_trk(th) + ch16));
						M32(F - 4) = ctx; M32(F - 8) = 0; M32(F - 0xc) = base; M32(F - 0x10) = base; M32(F - 0x14) = out;
						j = 0; M32(F + 0x20) = 0;
						QE_CB(F - 0x18, M32(desc + 0x24), 0x00cd3b60u, ctx, desc);
					} else {
						j = M16(ch + 0x1c) + (u32)cur;
						M32(F + 0x20) = j;
						BW16(ch + 0x18, j); BW16(ch + 0x1c, 0);
					}
					if ((s32)j < (s32)(nk - 1)) {
						jn = M32(F + 0x20) + 1;
						do {
							u32 tr = qe_trk(th) + ch16, nx = qe_kt(tr, jn, j + 1), X, tr2, base, kc, kn, ct, o2;
							s32 st;
							M32(F + 0x28) = nx;
							if ((s32)nx > (s32)t1) break;
							X = M32(th + 4); st = (s16)M16(M32(X + 0x1c) + sidx); tr2 = M32(M32(X) + 4) + ch16; base = M32(M32(tr2));
							kc = (jn - 1) * (u32)st + base; kn = jn * (u32)st + base;
							ct = qe_kt(tr2, jn - 1, j);
							x0 = ex_mulss(ex_cvtsi2ss(x0, (s32)(M32(F + 0x28) - ct)), K);
							o2 = M32(F + 0x24) + 0x30;
							M32(F - 4) = ctx; M32(F - 8) = x0.u[0]; M32(F - 0xc) = kc; M32(F - 0x10) = kn; M32(F - 0x14) = o2;
							M32(F + 0x2c) = o2;
							QE_CB(F - 0x18, M32(desc + 0x24), 0x00cd3c36u, kn, desc);
							j++; jn++;
						} while ((s32)j < (s32)(nk - 1));
					}
					{
						u32 chm = M32(F + 0x24), tr, ct, X, base, key, o2;
						s32 st;
						BW16(chm + 0x1c, j - M32(F + 0x20));
						tr = qe_trk(th) + ch16; ct = qe_kt(tr, j, j);
						x0 = ex_cvtsi2ss(x0, (s32)(t1 - ct));
						X = M32(th + 4); st = (s16)M16(sidx + M32(X + 0x1c)); base = M32(M32(M32(M32(X) + 4) + ch16));
						x0 = ex_mulss(x0, K);
						key = (u32)st * j + base;
						BW32(chm + 0x14, x0.u[0]);
						o2 = M32(F + 0x2c);
						M32(F - 4) = ctx; M32(F - 8) = x0.u[0]; M32(F - 0xc) = key; M32(F - 0x10) = o2;
						QE_CB(F - 0x14, M32(desc + 0x20), 0x00cd3cccu, o2, desc);
					}
				}
			}
			if (--n == 0) break;
			ch4 += 4; ch16 += 0x10; sidx += 2;
		}
	} else { /* B1 / B2: ctx = F+0x3c: b24, desc->+1 bit 4, then three SSE times (+8, +0xc, +0x10: re-read) */
		const u32 ctx = F + 0x3cu, oo = two ? 0x30u : 0x20u;
		const int p24 = qe_p24(c); /* (the FCW as the translation reads it: at the entry) */
		x0 = qe_zero();
		M32(F + 0x3c) = b24; M32(F + 0x40) = (fl >> 4) & 1; c->ecx = (fl >> 4) & 1;
		for (;;) {
			u32 ch = M32(M32(th + 0x10) + ch4);
			if (two && M32(ch + 0x28) == 0) { c->eax = ch; c->edx = M32(th + 0x10); }
			else {
				u32 nk = M16(qe_trk(th) + ch16 + 0xc), out = ch + oo, j, start;
				s32 cur = (s16)M16(ch + 0x18);
				if (cur < 0 && !nk) { c->eax = ch; c->ecx = (u32)cur; c->edx = out; }
				else {
					if (cur < 0) {
						u32 base;
						BW16(ch + 0x18, 0);
						base = M32(M32(qe_trk(th) + ch16));
						M32(F - 4) = ctx; M32(F - 8) = 0; M32(F - 0xc) = base; M32(F - 0x10) = base; M32(F - 0x14) = out;
						j = 0;
						M32(F + 0x44) = x0.u[0]; M32(F + 0x48) = x0.u[0]; M32(F + 0x4c) = x0.u[0];
						QE_CB(F - 0x18, M32(desc + 0x24), two ? 0x00cd3d8eu : 0x00cd38c4u, ctx, desc);
					} else {
						j = M16(ch + 0x1c) + (u32)cur;
						BW16(ch + 0x18, j); BW16(ch + 0x1c, 0);
						M32(F + 0x48) = x0.u[0];
					}
					start = j;
					x0 = ex_mulss(ex_cvtsi2ss(x0, (s32)t0), K);
					M32(F + 0x4c) = x0.u[0];
					while ((s32)j < (s32)(nk - 1)) {
						u32 tr = qe_trk(th) + ch16, nx = qe_kt(tr, j + 1, j + 1), X, tr2, kc, kn, ct, dtb;
						s32 st;
						if ((s32)nx > (s32)t1) break;
						X = M32(th + 4);
						x0 = ex_ldss(F + 0x48); M32(F + 0x44) = x0.u[0];
						tr2 = M32(M32(X) + 4) + ch16;
						st = (s16)M16(M32(X + 0x1c) + sidx);
						kc = (u32)st * j + M32(M32(tr2));
						ct = qe_kt(tr2, j, j);
						x0 = ex_mulss(ex_cvtsi2ss(x0, (s32)(nx - t1 + t0)), K);
						dtb = qe_d2f(qe_mul87((double)(s32)(nx - ct), qe_f2d(K.u[0]), p24)); /* fild, fmul, fst */
						M32(F - 4) = ctx; M32(F - 8) = dtb; M32(F - 0xc) = kc; M32(F - 0x10) = out;
						M32(F + 0x48) = x0.u[0];
						QE_CB(F - 0x14, M32(desc + 0x20), two ? 0x00cd3e91u : 0x00cd39c7u, out, desc);
						x0 = ex_ldss(F + 0x48); M32(F + 0x44) = x0.u[0];
						X = M32(th + 4);
						st = (s16)M16(M32(X + 0x1c) + sidx);
						kn = (u32)st * (j + 1) + M32(M32(M32(M32(X) + 4) + ch16));
						M32(F - 4) = ctx; M32(F - 8) = dtb; M32(F - 0xc) = kc; M32(F - 0x10) = kn; M32(F - 0x14) = out;
						QE_CB(F - 0x18, M32(desc + 0x24), two ? 0x00cd3ed8u : 0x00cd3a0eu, out, desc);
						j++;
					}
					x0 = ex_ldss(F + 0x48); M32(F + 0x44) = x0.u[0];
					BW16(ch + 0x1c, j - start);
					{
						u32 tr = qe_trk(th) + ch16, ct = qe_kt(tr, j, j), X, base, key, frac;
						s32 st;
						x0 = ex_cvtsi2ss(x0, (s32)(t1 - ct));
						X = M32(th + 4); st = (s16)M16(M32(X + 0x1c) + sidx); base = M32(M32(M32(M32(X) + 4) + ch16));
						x0 = ex_mulss(x0, K);
						key = (u32)st * j + base;
						frac = x0.u[0];
						BW32(ch + 0x14, frac);
						x0 = ex_ldss(F + 0x4c);
						M32(F - 4) = ctx; M32(F - 8) = frac; M32(F - 0xc) = key; M32(F - 0x10) = out;
						M32(F + 0x48) = x0.u[0];
						QE_CB(F - 0x14, M32(desc + 0x20), two ? 0x00cd3f82u : 0x00cd3ab8u, out, desc);
						x0 = qe_zero();
					}
				}
			}
			if (--n == 0) break;
			ch4 += 4; ch16 += 0x10; sidx += 2;
		}
	}
	c->x[0] = x0;
	c->esp = E; xrc_stdcall_ret(c, 3);
}
#endif

/* 00dca630's register mirror around a call: xmm0..7 out to the context, the callee's back */
#define QE_PUT8() do { c->x[0] = x0; c->x[1] = x1; c->x[2] = x2; c->x[3] = x3; c->x[4] = x4; c->x[5] = x5; c->x[6] = x6; c->x[7] = x7; } while (0)
#define QE_GET8() do { x0 = c->x[0]; x1 = c->x[1]; x2 = c->x[2]; x3 = c->x[3]; x4 = c->x[4]; x5 = c->x[5]; x6 = c->x[6]; x7 = c->x[7]; } while (0)
/* the 4-lane sum x + shuf 55 + shuf aa + shuf ff as the original adds it */
static inline X128 qe_hsum(X128 a) { return ex_addps(ex_addps(ex_addps(ex_shufps(a, a, 0x55), a), ex_shufps(a, a, 0xaa)), ex_shufps(a, a, 0xff)); }
/* the 4-lane sum by rotations: (shuf 4e + shuf 39) + (shuf 93 + x) */
static inline X128 qe_rsum(X128 a) { return ex_addps(ex_addps(ex_shufps(a, a, 0x4e), ex_shufps(a, a, 0x39)), ex_addps(ex_shufps(a, a, 0x93), a)); }

/* 00dca630: thiscall (obj, idx), ret 8 (virtual; push ebp frame, aligned: S = ((ebp) & ~15) - 0x200 the body's esp):
 * cloth / rope collision of obj's constraints (cl = obj->+8: constraint k at cl + cl->+0x34 + k * 32 (+0 / +4 particle
 * indexes, +8 radius add, +0x14 bit 0 disabled), particles at cl + cl->+0x30 + i * 0x90 (+0 position, +0x10 previous,
 * +0x20 velocity-like, +0x40 a float)) against th's capsule (th->+0x20 / +0x30 the segment, +0x40 / +0x50 the previous
 * one, +0x80 radius, +0x10 / +0x14 / +8 the mix). idx >= 0: constraint idx, else all (obj->00dc35e0(); obj->00dc35f0(i)
 * maps; disabled ones skipped). flagB = obj->vtbl[0x3c]() (al; 0 when th->+8): 0: closest points of the segments
 * (00db7310 into the frame), push out along d when closer than radius r: d * (r - |d|) / |d| (|d| == 0: d = (0, 0.1,
 * 0, 0), |d| = 0.1), with k < cl->+8 (u16, "flagL") divided by the clamped cosine of d and the constraint (rsqrtps
 * normalized, <0.01: skipped, <0.4: 0.4); else the swept test 00dbd120 (the frame's segments, r * 0.99, &t) and at
 * t the interpolated segments: their closest points, 00dca040's two segment parameters (x87), the same push with
 * friction along the normal (velocity of th's segment at the contact). Applied: flagL: B += s1 * push, B+0x20 +=
 * s2 * push; else both A and B; their +0x40 = 1 - th->+0x10 (s1, s2 = 1, th->+0x14, or th->+0x14, 0 with th->+8).
 * Writes the particles (journaled) and the frame blocks the callees read (at the original's addresses). eax = the
 * last index + 1 (or the first when none), ecx the cookie, edx / xmm0..7 as the original; no result (void). */
#if 0 /* (battle batch 3: 00dca630 not selected) */
B_FN(00dca630, "constraints: capsule collision push", 0) {
	u32 E = c->esp, bp = E - 4u, S = (bp & ~15u) - 0x200u, th = c->ecx, obj = M32(E + 4), p2 = M32(E + 8), cl = M32(obj + 8);
	u32 first, last, i, flagB, s1, s2, omt;
	X128 x0, x1 = c->x[1], x2 = c->x[2], x3 = c->x[3], x4 = c->x[4], x5 = c->x[5], x6 = c->x[6], x7 = c->x[7];
	B_ENTER(bp);
	x0 = ex_subss(ex_ldss(0x010aa950u), ex_ldss(th + 0x10)); omt = x0.u[0];
	{
		u32 vt = M32(obj), m = M32(vt + 0x3cu);
		s32 d;
		int k = b_getter_kind(m, &d);
		c->eax = vt; c->ecx = obj; c->edx = m;
		if (k == 1) c->eax = M32(obj + (u32)d);
		else if (k == 2) c->eax = obj + (u32)d;
		else { QE_PUT8(); BCALL(S - 4, m, 0x00dca679u); QE_GET8(); }
	}
	flagB = c->eax & 0xffu;
	x6 = qe_zero();
	if (M32(th + 8) == 0) { s1 = M32(0x010aa950u); x0 = ex_ldss(th + 0x14); s2 = x0.u[0]; }
	else { x0 = ex_ldss(th + 0x14); s1 = x0.u[0]; s2 = x6.u[0]; flagB = 0; }
	if ((s32)p2 >= 0) first = last = p2;
	else {
		first = 0;
		c->ecx = obj;
		QE_PUT8(); BCALL(S - 4, 0x00dc35e0u, 0x00dca6d7u); QE_GET8();
		x6 = qe_zero();
		last = c->eax - 1;
	}
	c->ecx = first; c->eax = first;
	if ((s32)first <= (s32)last) for (i = first;; i++) {
		u32 k = i, con, pA, pB, flagL, rad, s;
		X128 d, push;
		if (first != last) {
			M32(S - 4) = i; c->eax = i; c->ecx = obj;
			QE_PUT8(); BCALL(S - 8, 0x00dc35f0u, 0x00dca71au); QE_GET8();
			x6 = qe_zero();
			k = c->eax;
		}
		con = (k << 5) + cl + M32(cl + 0x34);
		c->ecx = con; c->edx = M32(cl + 8) & 0xffffu;
		flagL = (s32)k < (s32)c->edx;
		if ((s32)p2 < 0 && (M8(con + 0x14) & 1)) goto next;
		pA = M32(con) * 0x90u + M32(cl + 0x30) + cl; pB = M32(con + 4) * 0x90u + M32(cl + 0x30) + cl;
		x0 = ex_addss(ex_ldss(th + 0x80), ex_ldss(con + 8)); rad = x0.u[0];
		if (!flagB) {
			XST(S + 0x80, ex_ld(pA)); XST(S + 0x90, ex_ld(pB)); XST(S + 0xe0, ex_ld(th + 0x20));
			x0 = ex_ld(th + 0x30); XST(S + 0xf0, x0);
			M32(S - 4) = S + 0x80; M32(S - 8) = S + 0xe0; M32(S - 0xc) = S + 0x100;
			c->eax = S + 0x80; c->ecx = S + 0xe0; c->edx = S + 0x100;
			QE_PUT8(); BCALL(S - 0x10, 0x00db7310u, 0x00dca7d0u); QE_GET8();
			x0 = ex_ld(S + 0x100); x3 = ex_ld(S + 0x110); x6 = qe_zero();
			x1 = ex_subps(x3, x0);
			x0 = ex_mulps(x1, x1);
			x4 = ex_shufps(x0, x0, 0xaa);
			x2 = qe_hsum(x0); x2 = ex_sqrtss(x2, x2);
			x0 = qe_lo(x2);
			x2 = ex_ss(rad);
			if (ex_comiss(x2, x0) & 5) goto next; /* (jbe) */
			if ((ex_ucomiss(x6, x0) & 3) == 1) { /* |d| == 0 */
				x0 = ex_ldss(0x0109c5ecu); x1 = x6; x4 = x0; x5 = x6; x7 = x6;
				x1 = ex_unpcklps(x1, x5); x4 = ex_unpcklps(x4, x7); x1 = ex_unpcklps(x1, x4);
			}
			d = x1;
			x2 = ex_divss(ex_subss(x2, x0), x0); x0 = x2; s = x0.u[0];
			if (flagL) {
				X128 sa0, m60, sc0;
				x7 = ex_ld(pA); x4 = ex_ld(pB);
				x4 = ex_subps(x4, x7); x3 = ex_subps(x3, x7);
				x0 = ex_mulps(x4, x4);
				x2 = ex_addps(ex_shufps(x0, x0, 0x4e), ex_shufps(x0, x0, 0x39));
				x1 = ex_addps(ex_shufps(x0, x0, 0x93), x0);
				x2 = ex_addps(x2, x1);
				x0 = x3; sa0 = x0;
				x0 = ex_mulps(x0, x0);
				x3 = ex_shufps(x0, x0, 0x4e);
				x7 = ex_shufps(x0, x0, 0x39);
				x1 = qe_zero(); x5 = x1;
				x3 = ex_addps(x3, x7);
				x7 = ex_addps(ex_shufps(x0, x0, 0x93), x0);
				x6 = ex_rsqrtps(x2);
				x3 = ex_addps(x3, x7);
				x7 = ex_ldss(0x0109b130u);
				x1 = ex_cmpps(x1, x3, 1); m60 = x1;
				x1 = qe_bc(M32(0x010aa950u));
				x5 = ex_cmpps(x5, x2, 1);
				x0 = ex_rsqrtps(x3); sc0 = x0;
				x0 = ex_mulps(ex_mulps(x6, x2), x6);
				x2 = x0;
				x0 = ex_subps(x1, x2);
				x7 = ex_shufps(x7, x7, 0);
				x2 = ex_mulps(x6, x7);
				x0 = ex_mulps(x0, x2);
				x2 = sc0;
				x0 = ex_addps(x0, x6);
				x0 = ex_mulps(x0, x4);
				x0 = ex_andps(x0, x5);
				x5 = ex_andnps(x5, x4);
				x4 = ex_mulps(ex_mulps(x2, x3), x2);
				x1 = ex_subps(x1, x4);
				x3 = ex_mulps(x2, x7);
				x1 = ex_mulps(x1, x3);
				x3 = sa0;
				x1 = ex_addps(x1, x2);
				x2 = m60;
				x1 = ex_mulps(x1, x3);
				x1 = ex_andps(x1, x2);
				x2 = ex_andnps(x2, x3);
				x1 = ex_orps(x1, x2);
				x0 = ex_orps(x0, x5);
				x0 = ex_mulps(x0, x1);
				x1 = ex_addps(ex_shufps(x0, x0, 0x55), x0);
				x2 = ex_shufps(x0, x0, 0xaa);
				x1 = ex_addps(x1, x2);
				x0 = ex_shufps(x0, x0, 0xff);
				x1 = ex_addps(x1, x0);
				x0 = qe_lo(x1);
				x1 = ex_ldss(0x010a0558u);
				if (!(ex_comiss(x1, x0) & 5)) { x6 = qe_zero(); goto next; } /* 0.01 > cos (ja) */
				x1 = ex_ldss(0x0109f7d4u);
				if (!(ex_comiss(x1, x0) & 5)) x0 = x1; /* 0.4 > cos */
				x1 = ex_ss(s); x6 = qe_zero();
				x1 = ex_divss(x1, x0); x0 = x1; x1 = d;
			}
			x0 = ex_mulps(ex_shufps(x0, x0, 0), x1);
			x2 = x0;
		} else {
			X128 dd, z;
			u32 u1, u2;
			x1 = ex_ld(pA);
			x0 = ex_mulss(x0, ex_ldss(0x010ae3fcu));
			XST(S + 0x120, x1);
			x1 = ex_ld(pB); XST(S + 0x130, x1);
			x1 = ex_ld(pA + 0x10); XST(S + 0x100, x1);
			x1 = ex_ld(pB + 0x10); XST(S + 0x110, x1);
			x1 = ex_ld(th + 0x20); XST(S + 0x60, x1);
			x1 = ex_ld(th + 0x30); XST(S + 0x70, x1);
			x1 = ex_ld(th + 0x40); XST(S + 0xe0, x1);
			x1 = ex_ld(th + 0x50); XST(S + 0xf0, x1);
			M32(S + 0x24) = x6.u[0];
			M32(S - 4) = S + 0x24; M32(S - 8) = x0.u[0]; M32(S - 0xc) = S + 0x120; M32(S - 0x10) = S + 0x60; M32(S - 0x14) = S + 0x100;
			M32(S - 0x18) = S + 0xe0;
			c->eax = S + 0x100; c->ecx = S + 0xe0; c->edx = S + 0x60;
			QE_PUT8(); BCALL(S - 0x1c, 0x00dbd120u, 0x00dcaacdu); QE_GET8();
			if (!(c->eax & 0xffu)) { x6 = qe_zero(); goto next; }
			/* at t: the segments A (pA, pA+0x10) -> S+0xa0 / +0xb0, th's -> S+0xc0 / +0xd0 */
			x0 = ex_ldss(S + 0x24);
			x2 = ex_ldss(0x010aa950u);
			x5 = ex_ld(S + 0x120);
			x6 = ex_ld(S + 0x100);
			x2 = ex_subss(x2, x0);
			x1 = qe_lo(x0);
			x0 = qe_lo(x2);
			x2 = x0;
			x0 = ex_shufps(x1, x1, 0);
			x4 = ex_mulps(x0, x5);
			x3 = ex_shufps(x2, x2, 0);
			x5 = ex_mulps(x3, x6);
			x4 = ex_addps(x4, x5);
			XST(S + 0xa0, x4);
			x4 = ex_ld(S + 0x130);
			x0 = ex_mulps(x0, x4);
			x4 = ex_ld(S + 0x110);
			x3 = ex_mulps(x3, x4);
			x4 = ex_ld(S + 0xe0);
			x0 = ex_addps(x0, x3);
			x3 = ex_ld(S + 0x60);
			XST(S + 0xb0, x0);
			x1 = ex_shufps(x1, x1, 0);
			x0 = ex_mulps(x1, x3);
			x2 = ex_shufps(x2, x2, 0);
			x3 = ex_mulps(x2, x4);
			x0 = ex_addps(x0, x3);
			XST(S + 0xc0, x0);
			x0 = ex_ld(S + 0x70);
			x1 = ex_mulps(x1, x0);
			x0 = ex_ld(S + 0xf0);
			x2 = ex_mulps(x2, x0);
			x1 = ex_addps(x1, x2);
			XST(S + 0xd0, x1);
			M32(S - 4) = S + 0xa0; M32(S - 8) = S + 0xc0; M32(S - 0xc) = S + 0x80;
			c->eax = S + 0xc0; c->ecx = S + 0x80; c->edx = S + 0xa0;
			QE_PUT8(); BCALL(S - 0x10, 0x00db7310u, 0x00dcabb8u); QE_GET8();
			x1 = ex_ld(S + 0x80); x0 = ex_ld(S + 0x90);
			x0 = ex_subps(x0, x1); dd = x0;
			M32(S - 4) = S + 0x80; c->ecx = S + 0xc0; c->edx = S + 0x80;
			QE_PUT8(); BCALL(S - 8, 0x00dca040u, 0x00dcabe7u); QE_GET8();
			u1 = qe_d2f(c->st0);
			M32(S - 4) = S + 0x90; c->eax = S + 0x90; c->ecx = S + 0xa0;
			QE_PUT8(); BCALL(S - 8, 0x00dca040u, 0x00dcabffu); QE_GET8();
			u2 = qe_d2f(c->st0);
			x4 = dd; x6 = qe_zero();
			x0 = ex_mulps(x4, x4);
			x2 = ex_shufps(x0, x0, 0xaa);
			x1 = qe_hsum(x0); x1 = ex_sqrtss(x1, x1);
			x0 = qe_lo(x1);
			if ((ex_ucomiss(x6, x0) & 3) == 1) { /* |d| == 0 */
				x0 = ex_ldss(0x0109c5ecu); x4 = x6; x1 = x0; x2 = x6; x3 = x6;
				x4 = ex_unpcklps(x4, x2); x1 = ex_unpcklps(x1, x3); x4 = ex_unpcklps(x4, x1);
			}
			x1 = ex_divss(ex_subss(ex_ss(rad), x0), x0); s = x1.u[0];
			if (flagL) {
				x0 = ex_ss(u2); x2 = ex_ldss(0x010a0558u);
				if (!(ex_comiss(x2, x0) & 4)) goto next; /* 0.01 >= u (jae) */
				x2 = ex_ldss(0x0109f7d4u);
				if (!(ex_comiss(x2, x0) & 5)) x0 = x2;
				x1 = ex_divss(x1, x0); s = x1.u[0];
			}
			/* n = d normalized (rsqrtps + a Newton step), minus its part along the segment at t */
			x5 = ex_ldss(0x0109b130u);
			x0 = ex_mulps(x4, x4);
			x2 = qe_rsum(x0);
			x1 = ex_shufps(x0, x0, 0x93);
			x1 = ex_addps(x1, x0);
			x1 = ex_ldss(0x010aa950u);
			x0 = ex_rsqrtps(x2);
			x7 = ex_mulps(ex_mulps(x0, x2), x0);
			x2 = x0;
			x1 = ex_shufps(x1, x1, 0);
			x5 = ex_shufps(x5, x5, 0);
			x2 = ex_mulps(x2, x5);
			x3 = ex_subps(x1, x7);
			x3 = ex_mulps(x3, x2);
			x2 = ex_ld(S + 0xb0);
			x3 = ex_addps(x3, x0);
			x0 = ex_ld(S + 0xa0);
			x2 = ex_subps(x2, x0);
			x0 = ex_mulps(x2, x2);
			x7 = ex_addps(ex_addps(ex_shufps(x0, x0, 0xaa), x0), ex_shufps(x0, x0, 0x55));
			x0 = ex_shufps(x0, x0, 0x55);
			x0 = qe_lo(x7);
			{
				unsigned f = ex_comiss(x0, x6);
				x3 = ex_mulps(x3, x4);
				if (!(f & 5)) {
					X128 l2;
					x0 = ex_mulps(x2, x2);
					x6 = ex_addps(ex_addps(ex_shufps(x0, x0, 0xaa), x0), ex_shufps(x0, x0, 0x55));
					l2 = x6;
					x0 = ex_mulps(x3, x2);
					x7 = ex_shufps(x0, x0, 0xaa);
					x6 = ex_addps(ex_addps(ex_addps(ex_shufps(x0, x0, 0x55), x0), x7), ex_shufps(x0, x0, 0xff));
					x0 = ex_shufps(x0, x0, 0xff);
					x7 = qe_lo(l2);
					x0 = qe_lo(x7);
					x6 = ex_divss(x6, x0);
					x0 = ex_shufps(x6, x6, 0);
					x6 = qe_zero();
					x0 = ex_mulps(x0, x2);
					x3 = ex_subps(x3, x0);
				}
			}
			x0 = ex_mulps(qe_bc(s), x4);
			x4 = qe_lo(x6); z = x4;
			x2 = x0; push = x2;
			x0 = ex_mulps(x3, x3);
			x7 = ex_shufps(x0, x0, 0xff);
			x4 = qe_lo(qe_hsum(x0));
			if (ex_comiss(x4, qe_lo(z)) & 5) goto apply; /* |n| == 0 */
			/* friction: n normalized; the contact's velocity along n */
			x2 = qe_rsum(x0);
			x4 = ex_addps(ex_shufps(x0, x0, 0x93), x0);
			x0 = ex_rsqrtps(x2);
			x4 = ex_mulps(ex_mulps(x0, x2), x0);
			x2 = ex_mulps(x0, x5);
			x5 = ex_ldss(0x010aa950u);
			x1 = ex_subps(x1, x4);
			x4 = ex_ld(S + 0x70);
			x1 = ex_mulps(x1, x2);
			x2 = ex_ss(u1);
			x1 = ex_addps(x1, x0);
			x0 = ex_ld(S + 0xc0);
			x1 = ex_mulps(x1, x3);
			x3 = ex_ld(S + 0x60);
			x3 = ex_subps(x3, x0);
			x0 = ex_ld(S + 0xd0);
			x4 = ex_subps(x4, x0);
			x0 = ex_subss(x5, x2);
			x2 = ex_shufps(x2, x2, 0);
			x2 = ex_mulps(x2, x4);
			x4 = ex_ldss(S + 0x24);
			x0 = ex_shufps(x0, x0, 0);
			x0 = ex_mulps(x0, x3);
			x0 = ex_addps(x0, x2);
			x0 = ex_mulps(x0, x1);
			x3 = ex_shufps(x0, x0, 0xaa);
			x2 = qe_hsum(x0);
			x0 = qe_lo(x2);
			if (ex_comiss(x0, x6) & 5) x2 = x6;
			else x2 = ex_mulss(ex_subss(x5, x4), x0);
			if (flagL) {
				x3 = ex_ld(pB); x5 = ex_ld(pB + 0x10);
				x0 = ex_mulps(ex_subps(x3, x5), x1);
				x7 = ex_shufps(x0, x0, 0xaa);
				x5 = qe_lo(qe_hsum(x0));
				x0 = ex_addss(ex_mulss(x5, x4), x2);
				x2 = push;
				x0 = ex_subss(x0, x5);
				x0 = ex_addps(ex_mulps(ex_shufps(x0, x0, 0), x1), x3);
				BW128(pB, x0);
			} else {
				X128 vA, vB;
				x3 = ex_ld(pA); x5 = ex_ld(pA + 0x10);
				x0 = ex_mulps(ex_subps(x3, x5), x1);
				vA = qe_hsum(x0);
				x0 = ex_ld(pB); x5 = ex_ld(pB + 0x10);
				x0 = ex_mulps(ex_subps(x0, x5), x1);
				x7 = ex_shufps(x0, x0, 0xaa);
				vB = qe_hsum(x0);
				x5 = qe_lo(vA);
				x0 = ex_subss(ex_addss(ex_mulss(x5, x4), x2), x5);
				x0 = ex_addps(ex_mulps(ex_shufps(x0, x0, 0), x1), x3);
				x3 = qe_lo(vB);
				BW128(pA, x0);
				x0 = ex_addss(ex_mulss(x3, x4), x2);
				x2 = push;
				x0 = ex_subss(x0, x3);
				x0 = ex_mulps(ex_shufps(x0, x0, 0), x1);
				x1 = ex_ld(pB);
				x0 = ex_addps(x0, x1);
				BW128(pB, x0);
			}
		}
	apply:
		if (flagL) { /* only B */
			x1 = ex_ld(pB);
			x0 = ex_addps(ex_mulps(qe_bc(s1), x2), x1);
			BW128(pB, x0);
			x1 = ex_ld(pB + 0x20);
			x0 = ex_addps(ex_mulps(qe_bc(s2), x2), x1);
			BW128(pB + 0x20, x0);
			x0 = ex_ss(omt); BW32(pB + 0x40, omt);
		} else {
			x3 = ex_ld(pA);
			x0 = ex_mulps(qe_bc(s1), x2);
			x1 = ex_addps(x0, x3);
			BW128(pA, x1);
			x1 = ex_mulps(qe_bc(s2), x2);
			x2 = ex_addps(ex_ld(pA + 0x20), x1);
			BW128(pA + 0x20, x2);
			x2 = ex_ss(omt); BW32(pA + 0x40, omt);
			x3 = ex_ld(pB);
			x0 = ex_addps(x0, x3);
			BW128(pB, x0);
			x0 = ex_ld(pB + 0x20);
			x1 = ex_addps(x1, x0);
			BW128(pB + 0x20, x1);
			BW32(pB + 0x40, omt);
		}
	next:
		c->eax = i + 1;
		if ((s32)(i + 1) > (s32)last) break;
	}
	c->x[0] = x0; c->x[1] = x1; c->x[2] = x2; c->x[3] = x3; c->x[4] = x4; c->x[5] = x5; c->x[6] = x6; c->x[7] = x7;
	c->ecx = M32(0x026e5a84u);
	B_RET(E, 2);
}
#endif

/* 00f30350: thiscall (p, unused), ret 8 (virtual; push ebp frame, aligned: S = (ebp & ~15) - 0x1b0 the body's esp; no
 * register saves): a material's shader parameters. a = p->+0x4f4 when >= 0 (comiss, not below), else p->+0x4f0;
 * a *= p->+0x4fc; 00eacdd0(p->+0x4f8, a * 100000) (cdecl: depth-bias render states); with th->+0x2c a vector (a * 100)
 * x4 (00515bb0) to 00c28280(th->+0x2c, &v) (thiscall on *0x028670bc: SetPixelShaderConstantF); p->+0x20 and p->+0x60
 * as matrices (00462f80) whose rows (00455840 copies) go to th->+0xc..+0x18 (00c28280) and th->+0x1c..+0x28 (00c27cf0
 * on *0x028670bc + 8, each only when nonzero); a vector zeroed (005be970), with p->+0x840 its lane p->+0x844 = 1.0
 * and f = (float)p->+0x848 (else 0); with th->+4: 00c28790(th->+4, &v), then with th->+8 the device's
 * SetPixelShaderConstantF (vtbl +0x1b4, stdcall) of (f, f, f, f). The fld / fstp float copies of the arguments are
 * bit copies (as the translation's). Writes only the frame's call arguments and the blocks the callees read and write
 * (at the original's addresses). Callees: D3D state / constant setters (the same values when run twice in the checked
 * mode). Fallback: p->+0x840 set with a lane index p->+0x844 >= 4 (a store outside the vector). eax / edx / xmm0
 * as the original, ecx the cookie; no result (void). */
B_FN(00f30350, "material: shader constants and depth bias", 0) {
	u32 E = c->esp, bp = E - 4u, S = (bp & ~15u) - 0x1b0u, th = c->ecx, p = M32(E + 4), h, k;
	X128 a, v100, v1e5, x0;
	if (M8(p + 0x840) && M32(p + 0x844) >= 4u) { orig(c); return; }
	B_ENTER(bp);
	a = ex_ldss(p + 0x4f0);
	c->eax = p;
	if (!(ex_comiss(ex_ldss(p + 0x4f4), ex_ldss(0x025a3c20u)) & 4)) { c->edx = p; a = ex_ldss(p + 0x4f4); } /* (not jb) */
	a = ex_mulss(a, ex_ldss(p + 0x4fc));
	v100 = ex_mulss(a, ex_ldss(0x0114f038u));
	v1e5 = ex_mulss(a, ex_ldss(0x026511e4u));
	x0 = ex_ldss(p + 0x4f8);
	M32(S - 4) = v1e5.u[0]; M32(S - 8) = x0.u[0];
	c->ecx = p; c->x[0] = x0;
	BCALL(S - 0xc, 0x00eacdd0u, 0x00f30463u);
	c->edx = th;
	if (M32(th + 0x2c) != 0) {
		M32(S - 4) = v100.u[0]; M32(S - 8) = v100.u[0]; M32(S - 0xc) = v100.u[0]; M32(S - 0x10) = v100.u[0];
		c->ecx = S + 0x17c;
		BCALL(S - 0x14, 0x00515bb0u, 0x00f304abu);
		h = M32(th + 0x2c);
		M32(S - 4) = S + 0x17c; M32(S - 8) = h;
		c->eax = h; c->ecx = M32(0x028670bcu); c->edx = S + 0x17c;
		BCALL(S - 0xc, 0x00c28280u, 0x00f304d7u);
	}
	/* p->+0x20: rows 0..3 to th->+0xc, +0x10, +0x14, +0x18 */
	M32(S - 4) = p + 0x20; c->ecx = S + 0x130;
	BCALL(S - 8, 0x00462f80u, 0x00f304eau);
	M32(S - 4) = S + 0x130; c->ecx = S + 0x120; c->edx = S + 0x130;
	BCALL(S - 8, 0x00455840u, 0x00f304feu);
	h = M32(th + 0xc);
	M32(S - 4) = S + 0x120; M32(S - 8) = h; c->eax = h; c->ecx = M32(0x028670bcu); c->edx = S + 0x120;
	BCALL(S - 0xc, 0x00c28280u, 0x00f3052au);
	M32(S - 4) = S + 0x140; c->ecx = S + 0x10c;
	BCALL(S - 8, 0x00455840u, 0x00f3053eu);
	h = M32(th + 0x10);
	M32(S - 4) = S + 0x10c; M32(S - 8) = h; c->eax = h; c->ecx = M32(0x028670bcu); c->edx = h;
	BCALL(S - 0xc, 0x00c28280u, 0x00f3056au);
	M32(S - 4) = S + 0x150; c->eax = S + 0x150; c->ecx = S + 0xf8;
	BCALL(S - 8, 0x00455840u, 0x00f3057eu);
	h = M32(th + 0x14);
	M32(S - 4) = S + 0xf8; M32(S - 8) = h; c->eax = S + 0xf8; c->ecx = M32(0x028670bcu); c->edx = h;
	BCALL(S - 0xc, 0x00c28280u, 0x00f305aau);
	M32(S - 4) = S + 0x160; c->ecx = S + 0xe4; c->edx = S + 0x160;
	BCALL(S - 8, 0x00455840u, 0x00f305beu);
	h = M32(th + 0x18);
	M32(S - 4) = S + 0xe4; M32(S - 8) = h; c->eax = h; c->ecx = M32(0x028670bcu); c->edx = S + 0xe4;
	BCALL(S - 0xc, 0x00c28280u, 0x00f305eau);
	/* p->+0x60: rows 0..3 to th->+0x1c, +0x20, +0x24, +0x28 (each when nonzero) */
	M32(S - 4) = p + 0x60; c->ecx = S + 0xa0;
	BCALL(S - 8, 0x00462f80u, 0x00f305fdu);
	{
		static const u32 rw[4] = {0xa0u, 0xb0u, 0xc0u, 0xd0u}, cp[4] = {0x90u, 0x78u, 0x60u, 0x48u};
		static const u32 r1[4] = {0x00f30611u, 0x00f30669u, 0x00f306afu, 0x00f306f5u}, r2[4] = {0x00f30658u, 0x00f3069eu, 0x00f306e4u, 0x00f3072au};
		for (k = 0; k < 4; k++) {
			u32 mgr;
			M32(S - 4) = S + rw[k]; c->ecx = S + cp[k]; c->edx = S + rw[k];
			BCALL(S - 8, 0x00455840u, r1[k]);
			h = M32(th + 0x1c + k * 4); mgr = M32(0x028670bcu);
			c->eax = th; c->ecx = h; c->edx = mgr;
			if (h != 0) {
				M32(S - 4) = S + cp[k]; M32(S - 8) = h;
				c->eax = S + cp[k]; c->ecx = mgr + 8;
				BCALL(S - 0xc, 0x00c27cf0u, r2[k]);
			}
		}
	}
	M32(S - 4) = S + 0x30; c->edx = S + 0x30;
	BCALL(S - 8, 0x005be970u, 0x00f30734u);
	x0 = ex_ldss(0x025a3c20u);
	{
		u32 fl = M8(p + 0x840);
		c->eax = p; c->ecx = (c->ecx & ~0xffu) | fl; c->edx = fl;
		if (fl) {
			u32 idx = M32(p + 0x844), at = S + 0x30 + idx * 4;
			x0 = ex_ldss(0x010aa950u);
			if (idx < 4u) M32(at) = x0.u[0]; else BW32(at, x0.u[0]);
			c->edx = idx;
			c->ecx = M32(p + 0x848);
			x0 = ex_cvtsi2ss(x0, (s32)c->ecx);
		}
	}
	c->edx = th;
	if (M32(th + 4) != 0) {
		u32 f = x0.u[0], h8, mgr;
		M32(S - 4) = S + 0x30; M32(S - 8) = M32(th + 4);
		c->eax = S + 0x30; c->ecx = th; c->edx = M32(th + 4); c->x[0] = x0;
		BCALL(S - 0xc, 0x00c28790u, 0x00f307b9u);
		x0 = ex_ss(f);
		M32(S + 0x10) = f; M32(S + 0x14) = f; M32(S + 0x18) = f; M32(S + 0x1c) = f;
		h8 = M32(th + 8); mgr = M32(0x028670bcu);
		c->eax = th; c->ecx = h8; c->edx = mgr;
		if (h8 != 0) {
			u32 dev = M32(0x0283eac0u), vt = M32(dev);
			M32(S - 4) = 1; M32(S - 8) = S + 0x10; M32(S - 0xc) = M32(M32(mgr + 8) + h8 * 8 - 4); M32(S - 0x10) = dev;
			c->eax = vt; c->ecx = dev; c->edx = M32(vt + 0x1b4); c->x[0] = x0;
			BCALL(S - 0x14, c->edx, 0x00f30839u);
			x0 = c->x[0];
		}
	}
	c->x[0] = x0;
	c->ecx = M32(0x026e5a84u);
	B_RET(E, 2);
}

/* ---- battle batch 3, part F: 00ee65c0, 00f832d0, 00ba7340, 00fa0d60 */

/* the x87 as the translation computes it (xrc.h: double arithmetic, XR87 = rounded to float after each operation when
 * the FCW's precision control (c->fcw at the function's entry, as x_fcw) is 24-bit; fld / fild / fstp of a float /
 * int: the conversions) -- SSE2 double instructions through inline asm, so fast math cannot fold or reorder them */
static inline double qf_f2d(u32 bits) { X128 a = ex_ss(bits); double r; __asm__("cvtss2sd %1, %0" : "=x"(r) : "x"(a.v)); return r; }
static inline u32 qf_d2f(double v) { X128 r = ex_ss(0); __asm__("cvtsd2ss %1, %0" : "+x"(r.v) : "x"(v)); return r.u[0]; }
static inline double qf_r(double v, int p24) { if (p24) __asm__("cvtsd2ss %0, %0\n\tcvtss2sd %0, %0" : "+x"(v)); return v; }
#define QF_DOP(name_, ins_) static inline double name_(double a, double b, int p24) { __asm__(ins_ " %1, %0" : "+x"(a) : "x"(b)); return qf_r(a, p24); }
QF_DOP(qf_addd, "addsd") QF_DOP(qf_subd, "subsd") QF_DOP(qf_muld, "mulsd") QF_DOP(qf_divd, "divsd")
/* cvtss2sd x, x (the low double from lane 0, the upper 64 bits kept) / cvtsd2ss x, x (lane 0 from the low double, lanes
 * 1..3 kept): the SSE2 libm's argument and result */
static inline X128 qf_cvtss2sd(X128 a) { __asm__("cvtss2sd %0, %0" : "+x"(a.v)); return a; }
static inline X128 qf_cvtsd2ss(X128 a) { __asm__("cvtsd2ss %0, %0" : "+x"(a.v)); return a; }
/* four dwords as a vector / lane 3 replaced (SSE2 integer ops: no stack round trip) */
static inline X128 qf_v4(u32 x, u32 y, u32 z, u32 w) { X128 r; r.vi = _mm_set_epi32((int)w, (int)z, (int)y, (int)x); return r; }
static inline X128 qf_setw(X128 a, u32 w) {
	X128 r;
	r.vi = _mm_or_si128(_mm_and_si128(a.vi, _mm_set_epi32(0, -1, -1, -1)), _mm_set_epi32((int)w, 0, 0, 0));
	return r;
}

/* 00ee65c0: thiscall (), ret (/Od; T = this, E = T->+0, re-read at each use as the original). A = the frame's esp
 * (entry esp - 0x1b8; push ebp / mov ebp, esp / sub esp, 0x1b4). T->+8 = 0; 00515bb0 (Vec4 ctor, 4 x fld1) into the
 * local ebp-0x60 -> E->+0x878 (16 bytes); E->+0x168 = 0; 00f37060(1).
 * Pass 1, when (T->+0x89 && ![026e51f8]) || T->+0x8a: the first list i (i < T->+0x14's count (+0x18 begin, +0x1c end:
 * 0 without a begin), bit i of T->+0x2c) only, L = T->+0x18[i]: buckets j = L->+0xc - 1 down to 0 (L->+4[j]), each a
 * node chain N (->+0x14). Per node: material m = N->+0x10 changed (from 0): ee6470(m), ee7730(m) (thiscall T, ret 4),
 * 00455840 (Vec4 copy through the x87) of m + 0xb0 into ebp-0x80; E->+0x878 = {its x, y, z, [010aa950] (1.0)},
 * E->+0x1b4 = m->+0x10c. Then N->vtbl[T->+0x8a && N->+0x19 bit 3 ? 1 : 2](E) (thiscall, ret 4); E->+0x168 =
 * (s16)N->+4; the callback T->+0xa8 (if set, no args) returning al != 0: E->+0x168 = 0.
 * After it: E->+0x878 = 4 x [010aa950], E->+0x168 = 0; T->+0x8a set: done. Else pass 2 over every list k with bit k:
 * the buckets of L->+4 as pass 1 (00455840 into ebp-0xa0; the w lane 1.0 only when m->+0x13a & 3 == 0; slot 1); once
 * per call (flag b2): F = E->+0x8f4 set: F(1, 0, 0) (cdecl), E->+0x168 = 0, f37060(1). Then the buckets of L->+0
 * (j = L->+8 - 1 .. 0; prev = (1 << L->+0x18) - 1 per bucket), per node: lv = (s8)N->+0x18 (negative: 5); prev >= 3,
 * lv < 3 and once (flag b1): F(0, 1, 1), E->+0x168 = 0, f37060(1); m changed: ee6470, ee7730, E->+0x878 = m->+0xb0
 * (movss bit copies), E->+0x1b4 = m->+0x10c; N->vtbl[1](E); E->+0x168 = (s16)N->+4, = 0 when 1.0 > m->+0xbc,
 * +0xc0 or +0xc4 (comiss, ja); the callback; prev = lv. After each list, b1 not yet set: as the b1 call above.
 * At the end E->+0x8d8 = 0; eax = T, ecx = edx = E (the callers 00ee5f08 / 00ee718a read none). The registers at each
 * call and xmm0 (movss loads) as the original has them.
 * Non-repeatable in the checked mode (each callee runs twice): the node methods (N->vtbl[1] / [2]: draw submission),
 * the callback, F (a render-pass hook); ee6470 (shader set: its own change test), ee7730, f37060 (globals) repeat. */
#define QF_E65_VCALL(N_, slot_, ra_) do {                                                                       \
		u32 vt_, m_;                                                                                         \
		M32(A - 4) = M32(T);                                                                                 \
		vt_ = M32(N_); m_ = M32(vt_ + (slot_));                                                              \
		c->eax = m_; c->ecx = (N_); c->edx = vt_;                                                            \
		BCALL(A - 8, m_, (ra_));                                                                             \
	} while (0)
#define QF_E65_CB(ra_) do {                                                                                     \
		u32 cb_ = M32(T + 0xa8u);                                                                            \
		if (cb_) {                                                                                           \
			c->eax = cb_; c->ecx = T; c->edx = T;                                                            \
			BCALL(A - 4, cb_, (ra_));                                                                        \
			if (c->eax & 0xffu) { u32 e_ = M32(T); BW32(e_ + 0x168u, 0); c->eax = e_; c->ecx = e_; c->edx = T; } \
		}                                                                                                    \
	} while (0)
/* ee6470(m), ee7730(m), thiscall T */
#define QF_E65_MAT(m_, N_, ra1_, ra2_) do {                                                                     \
		M32(A - 4) = (m_); c->eax = (m_); c->ecx = T; c->edx = (N_);                                         \
		BCALL(A - 8, 0x00ee6470u, (ra1_));                                                                   \
		M32(A - 4) = (m_); c->ecx = T; c->edx = (m_);                                                        \
		BCALL(A - 8, 0x00ee7730u, (ra2_));                                                                   \
	} while (0)
/* F(a0, a1, a2) (cdecl, at A - 16), E->+0x168 = 0, f37060(1); the registers at the calls: the b2 site (fst_ 1) eax = E,
 * ecx = F, edx = T, then eax = ecx = E, edx = T; the b1 sites eax = T, ecx = E, edx = F, then eax = T, ecx = edx = E */
#define QF_E65_HOOK(fst_, a0_, a1_, a2_, ra1_, ra2_) do {                                                       \
		u32 f_ = M32(M32(T) + 0x8f4u);                                                                       \
		if (f_) {                                                                                            \
			u32 e_;                                                                                          \
			M32(A - 4) = (a2_); M32(A - 8) = (a1_); M32(A - 12) = (a0_);                                     \
			if (fst_) { c->eax = M32(T); c->ecx = f_; c->edx = T; }                                          \
			else { c->eax = T; c->ecx = M32(T); c->edx = f_; }                                               \
			BCALL(A - 16, f_, (ra1_));                                                                       \
			done_ = 1;                                                                                       \
			e_ = M32(T); BW32(e_ + 0x168u, 0);                                                               \
			M32(A - 4) = 1;                                                                                  \
			if (fst_) { c->eax = e_; c->ecx = e_; c->edx = T; }                                              \
			else { c->eax = T; c->ecx = e_; c->edx = e_; }                                                   \
			BCALL(A - 8, 0x00f37060u, (ra2_));                                                               \
		}                                                                                                    \
	} while (0)
#if 0 /* (battle batch 3: 00ee65c0 not selected) */
B_FN(00ee65c0, "render lists: set materials, draw the nodes", 0) {
	u32 sp = c->esp, T = c->ecx, bp = sp - 4, A = bp - 0x1b4u, K1 = M32(0x010aa950u), E, L, m, j, N, k, bg, cnt, v0, v1, v2, v3;
	int b1 = 0, b2 = 0;
	B_ENTER(bp);
	BW32(T + 8, 0);
	M32(A - 4) = 0x3f800000u; M32(A - 8) = 0x3f800000u; M32(A - 12) = 0x3f800000u; M32(A - 16) = 0x3f800000u; /* (fld1, fstp) */
	c->eax = T; c->ecx = bp - 0x60u;
	BCALL(A - 20, 0x00515bb0u, 0x00ee65fcu);
	E = M32(T);
	BW32(E + 0x878u, M32(bp - 0x60u)); BW32(E + 0x87cu, M32(bp - 0x5cu)); BW32(E + 0x880u, M32(bp - 0x58u)); BW32(E + 0x884u, M32(bp - 0x54u));
	E = M32(T);
	BW32(E + 0x168u, 0);
	M32(A - 4) = 1; c->eax = E; c->ecx = T; c->edx = E;
	BCALL(A - 8, 0x00f37060u, 0x00ee665du);
	/* pass 1 */
	if ((M8(T + 0x89u) && !M8(0x026e51f8u)) || M8(T + 0x8au)) {
		for (k = 0;; k++) {
			bg = M32(T + 0x18u);
			cnt = bg ? (u32)((s32)(M32(T + 0x1cu) - bg) >> 2) : 0;
			if (k >= cnt) break;
			if (!((1u << (k & 31u)) & M32(T + 0x2cu))) continue;
			L = M32(M32(T + 0x18u) + k * 4u);
			m = 0;
			for (j = M32(L + 0xcu) - 1u; j != 0xffffffffu; j--)
				for (N = M32(M32(L + 4u) + j * 4u); N; N = M32(N + 0x14u)) {
					if (m != M32(N + 0x10u)) {
						m = M32(N + 0x10u);
						QF_E65_MAT(m, N, 0x00ee6785u, 0x00ee6794u);
						M32(A - 4) = m + 0xb0u; c->eax = m + 0xb0u; c->ecx = bp - 0x80u; c->edx = m;
						BCALL(A - 8, 0x00455840u, 0x00ee67a5u);
						v0 = M32(bp - 0x80u); v1 = M32(bp - 0x7cu); v2 = M32(bp - 0x78u);
						c->x[0] = ex_ss(K1);
						E = M32(T);
						BW32(E + 0x878u, v0); BW32(E + 0x87cu, v1); BW32(E + 0x880u, v2); BW32(E + 0x884u, K1);
						v3 = M32(m + 0x10cu);
						BW32(M32(T) + 0x1b4u, v3);
					}
					if (M8(T + 0x8au) && (M8(N + 0x19u) & 8u)) QF_E65_VCALL(N, 4u, 0x00ee6870u);
					else QF_E65_VCALL(N, 8u, 0x00ee6888u);
					v0 = (u32)(s32)(s16)M16(N + 4u);
					BW32(M32(T) + 0x168u, v0);
					QF_E65_CB(0x00ee68d2u);
				}
			break; /* (only the first list) */
		}
	}
	c->x[0] = ex_ss(K1);
	E = M32(T);
	BW32(E + 0x878u, K1); BW32(E + 0x87cu, K1); BW32(E + 0x880u, K1); BW32(E + 0x884u, K1);
	BW32(M32(T) + 0x168u, 0);
	/* pass 2 */
	if (!M8(T + 0x8au)) {
		for (k = 0;; k++) {
			bg = M32(T + 0x18u);
			cnt = bg ? (u32)((s32)(M32(T + 0x1cu) - bg) >> 2) : 0;
			if (k >= cnt) break;
			if (!((1u << (k & 31u)) & M32(T + 0x2cu))) continue;
			L = M32(M32(T + 0x18u) + k * 4u);
			m = 0;
			for (j = M32(L + 0xcu) - 1u; j != 0xffffffffu; j--)
				for (N = M32(M32(L + 4u) + j * 4u); N; N = M32(N + 0x14u)) {
					if (m != M32(N + 0x10u)) {
						m = M32(N + 0x10u);
						QF_E65_MAT(m, N, 0x00ee6ac8u, 0x00ee6ad7u);
						M32(A - 4) = m + 0xb0u; c->eax = m + 0xb0u; c->ecx = bp - 0xa0u; c->edx = m;
						BCALL(A - 8, 0x00455840u, 0x00ee6aebu);
						v0 = M32(bp - 0xa0u); v1 = M32(bp - 0x9cu); v2 = M32(bp - 0x98u); v3 = M32(bp - 0x94u);
						if (!(M8(m + 0x13au) & 3u)) { v3 = K1; c->x[0] = ex_ss(K1); }
						E = M32(T);
						BW32(E + 0x878u, v0); BW32(E + 0x87cu, v1); BW32(E + 0x880u, v2); BW32(E + 0x884u, v3);
						v3 = M32(m + 0x10cu);
						BW32(M32(T) + 0x1b4u, v3);
					}
					QF_E65_VCALL(N, 4u, 0x00ee6bb0u);
					v0 = (u32)(s32)(s16)M16(N + 4u);
					BW32(M32(T) + 0x168u, v0);
					QF_E65_CB(0x00ee6bfau);
				}
			if (!b2) {
				int done_ = 0;
				QF_E65_HOOK(1, 1u, 0u, 0u, 0x00ee6c77u, 0x00ee6ca3u);
				if (done_) b2 = 1;
			}
			m = 0;
			for (j = M32(L + 8u) - 1u; j != 0xffffffffu; j--) {
				u32 prev;
				N = M32(M32(L) + j * 4u);
				prev = (1u << (M32(L + 0x18u) & 31u)) - 1u;
				for (; N; N = M32(N + 0x14u)) {
					s32 lv = (s32)(s8)M8(N + 0x18u);
					X128 one = ex_ss(K1);
					if (lv < 0) lv = 5;
					if ((s32)prev >= 3 && lv < 3 && !b1) {
						int done_ = 0;
						QF_E65_HOOK(0, 0u, 1u, 1u, 0x00ee6d61u, 0x00ee6d8du);
						if (done_) b1 = 1;
					}
					if (m != M32(N + 0x10u)) {
						u32 pm;
						m = M32(N + 0x10u);
						QF_E65_MAT(m, N, 0x00ee6db7u, 0x00ee6dc6u);
						pm = m + 0xb0u;
						v0 = M32(pm); v1 = M32(pm + 4u); v2 = M32(pm + 8u); v3 = M32(pm + 12u);
						c->x[0] = ex_ss(v3);
						E = M32(T);
						BW32(E + 0x878u, v0); BW32(E + 0x87cu, v1); BW32(E + 0x880u, v2); BW32(E + 0x884u, v3);
						v3 = M32(m + 0x10cu);
						BW32(M32(T) + 0x1b4u, v3);
					}
					QF_E65_VCALL(N, 4u, 0x00ee6eadu);
					v0 = (u32)(s32)(s16)M16(N + 4u);
					BW32(M32(T) + 0x168u, v0);
					if (!(ex_comiss(one, ex_ldss(m + 0xbcu)) & 5u) || !(ex_comiss(one, ex_ldss(m + 0xc0u)) & 5u) ||
					    !(ex_comiss(one, ex_ldss(m + 0xc4u)) & 5u)) /* (ja: 1.0 above, ordered) */
						BW32(M32(T) + 0x168u, 0);
					c->x[0] = one;
					QF_E65_CB(0x00ee6f81u);
					prev = (u32)lv;
				}
			}
			if (!b1) {
				int done_ = 0;
				QF_E65_HOOK(0, 0u, 1u, 1u, 0x00ee7004u, 0x00ee7030u);
				if (done_) b1 = 1;
			}
		}
	}
	E = M32(T);
	BW32(E + 0x8d8u, 0);
	c->eax = T; c->ecx = E; c->edx = E;
	B_RET(sp, 0);
}
#endif
#undef QF_E65_VCALL
#undef QF_E65_CB
#undef QF_E65_MAT
#undef QF_E65_HOOK

/* 00f832d0: thiscall (P, Q), ret 8 (/Od; the frame aligned: S = ((entry esp - 4) & ~15) - 0x210, the esp after it). For
 * the 4 slots i (T = this): i < (s32)T->+8: R = T->+0xc[i] (R->+0xfc != 'DNTL' 0x4c544e44: the engine assertion,
 * default handler 00f88b70); memset(S+0x188, 0, 0x4c) (0100df80); V0 = *R->ee31e0(S+0x178); C0 = x87 copy (00455840)
 * of *R->+0xbc (00455880 into S+0x140) * splat(R->+0xcc) (mulps, the product at S+0x120); by kind R->+0xdc -- 2: V1 =
 * *R->ee32e0(S+0xe0) with w 1.0 ([010aa950]); a = R->+0xe8, cc = R->+0xec * [0119c674] (2.0) + a, s1 = (float)cos(a
 * * [0109b130] (0.5)), s2 = (float)cos(cc * 0.5) (0100f5a2, the SSE2 libm: a double in / out in xmm0); V2 = 00515bb0
 * (Vec4 ctor, the 4 args pushed by fld / fstp) (s2, s1, R->+0xe0, R->+0xe4) at S+0xa0; B[i].1 = 1. 3: V2 = {-1, -1,
 * R->+0xe0, R->+0xe4} ([0109f7c0]), V1 = {0, 0, 1, 1} ([025a3c20], [010aa950]); B[i].1 = 1. Else: V1 = *R->ee32e0(
 * S+0x58) with w 0, V2 = {-1, -1, 0, [0109f7f0]}, C0.w = 0, V0 = *005be970(S+0x34) (cdecl); B[i].0 = 1. Then with Q and
 * T->+0x2c + 16 i: for the dwords r = T->+0x2c / 30 / 34 / 38 + 16 i (each re-read after the previous call) not 0:
 * dev->SetVertexShaderConstantF(Q->+8[2 r - 1], &C0 / &V1 / &V0 / &V2 (the locals S+0x100 / cc / 164 / 90), 1)
 * (dev = [0283eac0], vtbl + 0x178). With P->+0x8fc (re-read): (P + 0x908 + 0x4c i)->ef1c70(S+0x188 = {C0, V1, V0, V2,
 * kind, kind (both 0 when P->+0x8fc was clear at the first read), (u8)(T->+0x2c + 16 i != 0)}) (thiscall, ret 4).
 * i >= T->+8, with Q and T->+0x2c + 16 i: v = 005be970(S+0xc); dev->SetVertexShaderConstantF(Q->+8[2 r - 1], v, 1) with r
 * = T->+0x2c + 16 i re-read, not 0. At the end eaba40(6, T->+8 != 0), eaba20(8, B (the 8 dwords at S+0x1e8), 8) (cdecl;
 * vertex shader bool constants). The /GS cookie store and check skipped. eax / edx as eaba20 leaves them, ecx = the
 * cookie (callers 00f83198 / 00f831e9 read none); xmm0 / xmm1 as the original has them at its calls. Writes nothing
 * outside its frame itself (only the assertion's handler globals); its stack locals that callees read are written at
 * their addresses. Non-repeatable callees in the checked mode: none known (the device calls set state; ef1c70 copies) */
#define QF_F83_VSC(r_, ptr_, ra_) do {                                                                          \
		u32 dv_ = M32(0x0283eac0u), vt_ = M32(dv_), m_;                                                      \
		M32(S - 4) = 1; M32(S - 8) = (ptr_); M32(S - 12) = M32(M32(Q + 8u) + (r_) * 8u - 4u); M32(S - 16) = dv_; \
		m_ = M32(vt_ + 0x178u);                                                                              \
		c->eax = dv_; c->ecx = m_; c->edx = vt_;                                                             \
		BCALL(S - 20, m_, (ra_));                                                                            \
	} while (0)
#if 0 /* (battle batch 3: 00f832d0 not selected) */
B_FN(00f832d0, "material: 4 vertex shader constant slots", 0) {
	u32 sp = c->esp, T = c->ecx, P = M32(sp + 4), Q = M32(sp + 8), bp = sp - 4, S = (bp & ~15u) - 0x210u, i, fm = 0;
	u32 K1 = M32(0x010aa950u), K0 = M32(0x025a3c20u), Km1 = M32(0x0109f7c0u), Kf0 = M32(0x0109f7f0u);
	X128 K2 = ex_ldss(0x0119c674u), Kh = ex_ldss(0x0109b130u);
	B_ENTER(bp);
	for (i = 0; i < 4u; i++) {
		u32 ti = T + i * 16u;
		if ((s32)i < (s32)M32(T + 8u)) {
			u32 R = M32(T + i * 4u + 0xcu), kind, a8;
			X128 C0, V0, V1, V2, s4;
			if (M32(R + 0xfcu) != 0x4c544e44u)
				b_assert(c, S - 0x18u, 0x00f88b70u, 0x0263193cu, 0x01099d70u, 0x0265cdf8u, 0x10du, 0x0265cea8u, 0x00f83427u);
			M32(S - 4) = 0x4cu; M32(S - 8) = 0; M32(S - 12) = S + 0x188u; c->ecx = S + 0x188u;
			BCALL(S - 16, 0x0100df80u, 0x00f8343bu); /* memset */
			M32(S - 4) = S + 0x178u; c->ecx = R; c->edx = S + 0x178u;
			BCALL(S - 8, 0x00ee31e0u, 0x00f83452u);
			V0 = ex_ld(c->eax);
			s4 = ex_ldss(R + 0xccu);
			s4 = ex_shufps(s4, s4, 0x00);
			M32(S - 4) = S + 0x140u; c->eax = R; c->ecx = R + 0xbcu; c->x[0] = s4;
			BCALL(S - 8, 0x00455880u, 0x00f834deu);
			C0 = ex_mulps(ex_ld(c->eax), s4);
			XST(S + 0x120u, C0);
			c->x[0] = C0; c->x[1] = s4;
			M32(S - 4) = S + 0x120u; c->ecx = S + 0x110u; c->edx = S + 0x120u;
			BCALL(S - 8, 0x00455840u, 0x00f83508u);
			C0 = ex_ld(S + 0x110u);
			a8 = M8(P + 0x8fcu);
			kind = M32(R + 0xdcu);
			if (a8) a8 = kind;
			if (kind == 2) {
				X128 a, cc;
				u32 d, e, s1, s2;
				M32(S - 4) = S + 0xe0u; c->eax = R; c->ecx = R; c->edx = S + 0xe0u;
				BCALL(S - 8, 0x00ee32e0u, 0x00f835dfu);
				V1 = ex_ld(c->eax);
				a = ex_ldss(R + 0xe8u);
				cc = ex_addss(ex_mulss(ex_ldss(R + 0xecu), K2), a);
				d = M32(R + 0xe4u); e = M32(R + 0xe0u);
				c->x[0] = qf_cvtss2sd(ex_mulss(a, Kh));
				c->eax = R; c->ecx = R; c->edx = R;
				BCALL(S - 4, 0x0100f5a2u, 0x00f836b1u);
				s1 = qf_cvtsd2ss(c->x[0]).u[0];
				c->x[0] = qf_cvtss2sd(ex_mulss(cc, Kh));
				BCALL(S - 4, 0x0100f5a2u, 0x00f836d8u);
				c->x[0] = qf_cvtsd2ss(c->x[0]);
				s2 = c->x[0].u[0];
				M32(S - 4) = x87_copy(d); M32(S - 8) = x87_copy(e); M32(S - 12) = x87_copy(s1); M32(S - 16) = x87_copy(s2);
				c->ecx = S + 0xa0u;
				BCALL(S - 20, 0x00515bb0u, 0x00f8371du);
				V2 = ex_ld(S + 0xa0u);
				V1 = qf_setw(V1, K1);
				c->x[0] = ex_ss(K1);
				fm |= 2u << (2 * i);
			} else if (kind == 3) {
				V2 = qf_v4(Km1, Km1, M32(R + 0xe0u), M32(R + 0xe4u));
				V1 = qf_v4(K0, K0, K1, K1);
				c->x[0] = ex_ss(K1);
				fm |= 2u << (2 * i);
			} else {
				M32(S - 4) = S + 0x58u; c->eax = R; c->ecx = R; c->edx = S + 0x58u;
				BCALL(S - 8, 0x00ee32e0u, 0x00f838abu);
				V1 = qf_setw(ex_ld(c->eax), K0);
				V2 = qf_v4(Km1, Km1, K0, Kf0);
				C0 = qf_setw(C0, K0);
				M32(S - 4) = S + 0x34u; c->eax = Kf0; c->ecx = S + 0x34u; c->edx = K0; c->x[0] = ex_ss(K0);
				BCALL(S - 8, 0x005be970u, 0x00f8396au);
				V0 = ex_ld(c->eax);
				fm |= 1u << (2 * i);
			}
			if (Q && M32(ti + 0x2cu)) {
				u32 r = M32(ti + 0x2cu);
				if (r) { XST(S + 0x100u, C0); QF_F83_VSC(r, S + 0x100u, 0x00f83a23u); }
				r = M32(ti + 0x30u);
				if (r) { XST(S + 0xccu, V1); QF_F83_VSC(r, S + 0xccu, 0x00f83a72u); }
				r = M32(ti + 0x34u);
				if (r) { XST(S + 0x164u, V0); QF_F83_VSC(r, S + 0x164u, 0x00f83ac1u); }
				r = M32(ti + 0x38u);
				if (r) { XST(S + 0x90u, V2); QF_F83_VSC(r, S + 0x90u, 0x00f83b10u); }
			}
			if (M8(P + 0x8fcu)) {
				XST(S + 0x188u, C0); XST(S + 0x198u, V1); XST(S + 0x1a8u, V0); XST(S + 0x1b8u, V2);
				M32(S + 0x1c8u) = a8; M32(S + 0x1ccu) = a8;
				M8(S + 0x1d0u) = M32(ti + 0x2cu) != 0;
				M32(S - 4) = S + 0x188u; c->eax = S + 0x188u; c->ecx = P + i * 0x4cu + 0x908u; c->edx = P;
				BCALL(S - 8, 0x00ef1c70u, 0x00f83c4du);
			}
		} else if (Q && M32(ti + 0x2cu)) {
			u32 pv, r;
			M32(S - 4) = S + 0xcu; c->eax = i << 4; c->ecx = T; c->edx = S + 0xcu;
			BCALL(S - 8, 0x005be970u, 0x00f83c77u);
			pv = c->eax;
			r = M32(ti + 0x2cu);
			if (r) {
				u32 dv = M32(0x0283eac0u), vt = M32(dv), m;
				M32(S - 4) = 1; M32(S - 8) = pv; M32(S - 12) = M32(M32(Q + 8u) + r * 8u - 4u); M32(S - 16) = dv;
				m = M32(vt + 0x178u);
				c->eax = vt; c->ecx = dv; c->edx = m;
				BCALL(S - 20, m, 0x00f83ccbu);
			}
		}
	}
	for (i = 0; i < 8u; i++) M32(S + 0x1e8u + i * 4u) = (fm >> i) & 1u;
	{
		u32 f = M32(T + 8u) != 0;
		M32(S - 4) = f; M32(S - 8) = 6; c->eax = T; c->ecx = (c->ecx & ~0xffu) | f; c->edx = f;
		BCALL(S - 12, 0x00eaba40u, 0x00f83ce9u);
		M32(S - 4) = 8; M32(S - 8) = S + 0x1e8u; M32(S - 12) = 8; c->eax = S + 0x1e8u;
		BCALL(S - 16, 0x00eaba20u, 0x00f83cfdu);
	}
	c->ecx = M32(0x026e5a84u);
	B_RET(sp, 2);
}
#endif
#undef QF_F83_VSC

/* 00ba7340: thiscall (float t, u16 mask, float *o1, float *o2, a5), ret 0x14 (/Od): k = 0; for the mask bits 1, 2, 4
 * (o1[0..2]) and 0x10, 0x20, 0x40 (o2[0..2]) set: e = this->+4[k++], o[j] = (float)e->vtbl[3](t) (thiscall, ret 4; t
 * pushed by fld / fstp: the x87 copy; the result in st0, fstp float); bit 0x100: e = this->+4[k], e->vtbl[4](a5, t)
 * (no result). The mask's argument slot gets the cleared bits (dead after the ret 0x14: not made). eax / ecx / edx as
 * the last callee leaves them, not set after it (the callers 00ba7aef / 00ba7c7c / 00ba7cb4 read none); xmm0 the last
 * result (movss). The callees run twice in the checked mode (curve evaluators, the hot one 00fa1260 -> 00fa0d60: pure) */
#define QF_BA7_ONE(bit_, out_, ra_) do {                                                                        \
		if (msk & (bit_)) {                                                                                  \
			u32 e_ = M32(M32(T + 4u) + k * 4u), vt_ = M32(e_), m_ = M32(vt_ + 0xcu), r_;                     \
			k++;                                                                                             \
			M32(sp - 0x44u) = tq;                                                                            \
			c->eax = m_; c->ecx = e_; c->edx = vt_;                                                          \
			BCALL(sp - 0x48u, m_, (ra_));                                                                    \
			r_ = qf_d2f(c->st0);                                                                             \
			c->x[0] = ex_ss(r_);                                                                             \
			BW32((out_), r_);                                                                                \
		}                                                                                                    \
	} while (0)
B_FN(00ba7340, "curves: evaluate the masked channels at t", 0) {
	u32 sp = c->esp, T = c->ecx, bp = sp - 4, msk = M16(sp + 8), tq = x87_copy(M32(sp + 4)), k = 0;
	B_ENTER(bp);
	QF_BA7_ONE(1u, M32(sp + 0xcu), 0x00ba7390u);
	QF_BA7_ONE(2u, M32(sp + 0xcu) + 4u, 0x00ba73dfu);
	QF_BA7_ONE(4u, M32(sp + 0xcu) + 8u, 0x00ba742fu);
	QF_BA7_ONE(0x10u, M32(sp + 0x10u), 0x00ba747fu);
	QF_BA7_ONE(0x20u, M32(sp + 0x10u) + 4u, 0x00ba74ceu);
	QF_BA7_ONE(0x40u, M32(sp + 0x10u) + 8u, 0x00ba751eu);
	if (msk & 0x100u) {
		u32 e = M32(M32(T + 4u) + k * 4u), vt = M32(e), m = M32(vt + 0x10u);
		M32(sp - 0x44u) = tq; M32(sp - 0x48u) = M32(sp + 0x14u);
		c->eax = m; c->ecx = e; c->edx = vt;
		BCALL(sp - 0x4cu, m, 0x00ba7577u);
	}
	B_RET(sp, 5);
}
#undef QF_BA7_ONE

/* 00fa0d60's key values: 16-bit (w & 0x7fff) * k, 8-bit (b & 0x7f) * k (cvtsi2ss, mulss), the sign bit from bit 15 / 7
 * (xorps with 0109c600's 0x80000000: lane 0 is what is used); then v * B + A (mulss, addss) */
static inline u32 qf_dec(u32 w, int w16, u32 k, u32 sg) {
	u32 v = ex_mulss(ex_cvtsi2ss(ex_ss(0), (s32)(w & (w16 ? 0x7fffu : 0x7fu))), ex_ss(k)).u[0];
	return (w & (w16 ? 0x8000u : 0x80u)) ? v ^ sg : v;
}
static inline u32 qf_madd(u32 v, u32 b, u32 a) { return ex_addss(ex_mulss(ex_ss(v), ex_ss(b)), ex_ss(a)).u[0]; }

/* 00fa0d60: cdecl float (p, float t), ret (optimized code; the result on the x87 stack, c->st0): a packed curve. p: u16
 * hdr (n = hdr & 0x7fff blocks of 255 time units, bit 15: 16-bit values (else 8-bit)), float A at +2, B at +6, u8 key
 * counts ct[n] at +0xa, then the keys D = ct + n (per block: the times (u8), then the values). ti = cvttss2si(t), q =
 * ti / 255 (the 0x80808081 multiply) clamped to n - 1, r = ti - 255 q.
 * Done here only for n = 1 and ti >= D[0] (the time of key 0); else orig at the entry (ENTRY fallback, before any
 * change): ti < D[0] is the original's own first test (fa0de1 / fa0ffb: its short path), n != 1 covers the paths that
 * need a block other than 0 (the next block when r is past block q's last key with q < n - 1 (fa0ea3 / fa10bf), the
 * previous block's last key when k = 0 (fa0f2a / fa1145)). The test reads hdr, t and D[0], what the original reads
 * before its first call (its stores before them: the pushes into its frame and D into the p slot, not the curve).
 * With n = 1 and ti >= D[0] >= 0: q = 0 (ti / 255 >= 0, clamped to 0), r = ti; f98a90 / f98b00 (16 / 8-bit; cdecl (&o,
 * &d, 0, ct)) with block 0 leave d = D (the sum of no counts) and set o = D + ct[0]; q = n - 1 always, so r >= the
 * last time (D[ct[0] - 1]; ct[0] = 0 reads D[-1] = ct[0] = 0 <= r): fa0cc0 / f98840 (cdecl (value, B, A): the x87
 * value * B + A) of the last value (st0 as it leaves it). Else ct[0] >= 1 and D[ct[0] - 1] > r >= D[0]: the first key
 * k with time > r has 1 <= k <= ct[0] - 1 (no block change, a key before in the block). v1 / v0 = value k / k - 1 * B
 * + A (SSE), t1 / t0 their times; u = (t - t0) / (t1 - t0) (fild, fsubr, fidiv), result v0 + (v1 - v0) * u (x87; XR87
 * by the entry FCW, as the translation's x_fcw). Its argument slots (the caller's memory: BW32, so the checked mode's
 * undo restores p and t before the translation's run) as the original leaves them: p <- D at the entry (then the
 * callee's d through it), at the end p <- v0 (y0), t <- t1 - t0. The key scales [025e5d34] / [025df674] and the sign
 * mask [0109c600] as the translation folds them (read-only data). The frame slots a callee reads written at their
 * addresses, its other stores not made. eax / ecx / edx and xmm0..4 not mirrored: left as the entry or the last callee
 * leaves them (the callers 00fa126c, 00fa1430 .. 00fa1475 read none: they fstp the result). ebx / esi / edi / ebp
 * untouched (no frame pointer: the original uses ebp as a register; its pushes of them are not made, no callee reads
 * them) */
B_FN(00fa0d60, "packed curve: value at t", BR_ST0) {
	u32 sp = c->esp, p = M32(sp + 4), hdr = M16(p), n = hdr & 0x7fffu, ct = p + 0xau, D = ct + n, tb = M32(sp + 8);
	u32 E, A, Bv, K, sg = 0x80000000u, r, dp, o1, cnt, kk, v0, v1, t0, t1;
	s32 ti = ex_cvttss2si(ex_ss(tb));
	int w16, p24;
	double u, d;
	if (n != 1u || ti < (s32)M8(D)) { orig(c); return; }
	E = sp - 0x28u; A = M32(p + 2); Bv = M32(p + 6);
	w16 = (int)(hdr >> 15); p24 = ((c->fcw >> 8) & 3u) == 0;
	K = w16 ? 0x38000100u : 0x3c010204u; /* ([025e5d34] / [025df674]) */
	r = (u32)ti; /* (q = 0) */
	BW32(E + 0x2cu, D); /* (the p slot: the d the callee reads) */
	M32(E + 0x1cu) = 0;
	M32(E - 4) = ct; M32(E - 8) = 0; M32(E - 12) = E + 0x2cu; M32(E - 16) = E + 0x1cu;
	if (w16) BCALL(E - 20, 0x00f98a90u, 0x00fa0e41u); else BCALL(E - 20, 0x00f98b00u, 0x00fa1057u);
	dp = M32(E + 0x2cu);
	cnt = M8(ct);
	if (!((s32)r < (s32)M8(cnt + dp - 1u))) {
		o1 = M32(E + 0x1cu);
		M32(E - 4) = x87_copy(A); M32(E - 8) = x87_copy(Bv);
		if (w16) { M32(E - 12) = M16(o1 + cnt * 2u - 2u); BCALL(E - 16, 0x00fa0cc0u, 0x00fa0e7fu); }
		else { M32(E - 12) = M8(cnt + o1 - 1u); BCALL(E - 16, 0x00f98840u, 0x00fa1095u); }
		c->esp = sp; xrc_cdecl_ret(c);
		return;
	}
	for (kk = 0; kk < cnt; kk++) if ((s32)r < (s32)M8(kk + dp)) break; /* (1 <= kk <= cnt - 1: see above) */
	o1 = M32(E + 0x1cu);
	v1 = qf_madd(qf_dec(w16 ? M16(o1 + kk * 2u) : M8(kk + o1), w16, K, sg), Bv, A);
	v0 = qf_madd(qf_dec(w16 ? M16(o1 + kk * 2u - 2u) : M8(kk + o1 - 1u), w16, K, sg), Bv, A);
	t1 = M8(kk + dp); t0 = M8(kk - 1u + dp); /* (u16 + 0 * 255, as s16: the byte) */
	BW32(sp + 8, t1 - t0);
	BW32(E + 0x2cu, v0);
	u = qf_subd(qf_f2d(tb), (double)(s32)t0, p24);
	u = qf_divd(u, (double)((s32)t1 - (s32)t0), p24);
	d = qf_subd(qf_f2d(v1), qf_f2d(v0), p24);
	d = qf_muld(d, u, p24);
	c->st0 = qf_addd(qf_f2d(v0), d, p24);
	c->esp = sp; xrc_cdecl_ret(c);
}
#else /* (not x86-64: the battle batches run the translations) */
void XRC_CC xo_00f99040(XCtx *c) { ORIG(0x00f99040u); orig(c); }
void XRC_CC xo_00f98690(XCtx *c) { ORIG(0x00f98690u); orig(c); }
void XRC_CC xo_00fa0700(XCtx *c) { ORIG(0x00fa0700u); orig(c); }
void XRC_CC xo_00cd8240(XCtx *c) { ORIG(0x00cd8240u); orig(c); }
void XRC_CC xo_00a39b20(XCtx *c) { ORIG(0x00a39b20u); orig(c); }
void XRC_CC xo_00f99680(XCtx *c) { ORIG(0x00f99680u); orig(c); }
void XRC_CC xo_00d7e170(XCtx *c) { ORIG(0x00d7e170u); orig(c); }
void XRC_CC xo_00db7310(XCtx *c) { ORIG(0x00db7310u); orig(c); }
void XRC_CC xo_00462d70(XCtx *c) { ORIG(0x00462d70u); orig(c); }
void XRC_CC xo_0045e580(XCtx *c) { ORIG(0x0045e580u); orig(c); }
void XRC_CC xo_00462880(XCtx *c) { ORIG(0x00462880u); orig(c); }
void XRC_CC xo_00492460(XCtx *c) { ORIG(0x00492460u); orig(c); }
void XRC_CC xo_00dc4010(XCtx *c) { ORIG(0x00dc4010u); orig(c); }
void XRC_CC xo_00dc8090(XCtx *c) { ORIG(0x00dc8090u); orig(c); }
void XRC_CC xo_00d7c9d0(XCtx *c) { ORIG(0x00d7c9d0u); orig(c); }
void XRC_CC xo_008d6650(XCtx *c) { ORIG(0x008d6650u); orig(c); }
void XRC_CC xo_00a81a20(XCtx *c) { ORIG(0x00a81a20u); orig(c); }
void XRC_CC xo_00a81ca0(XCtx *c) { ORIG(0x00a81ca0u); orig(c); }
void XRC_CC xo_00f13f50(XCtx *c) { ORIG(0x00f13f50u); orig(c); }
void XRC_CC xo_00da7c90(XCtx *c) { ORIG(0x00da7c90u); orig(c); }
void XRC_CC xo_00f13c90(XCtx *c) { ORIG(0x00f13c90u); orig(c); }
void XRC_CC xo_00f0f640(XCtx *c) { ORIG(0x00f0f640u); orig(c); }
void XRC_CC xo_00f9a340(XCtx *c) { ORIG(0x00f9a340u); orig(c); }
void XRC_CC xo_00f4a3c0(XCtx *c) { ORIG(0x00f4a3c0u); orig(c); }
void XRC_CC xo_00f87990(XCtx *c) { ORIG(0x00f87990u); orig(c); }
void XRC_CC xo_00c304e0(XCtx *c) { ORIG(0x00c304e0u); orig(c); }
void XRC_CC xo_00f30350(XCtx *c) { ORIG(0x00f30350u); orig(c); }
void XRC_CC xo_00ba7340(XCtx *c) { ORIG(0x00ba7340u); orig(c); }
void XRC_CC xo_00fa0d60(XCtx *c) { ORIG(0x00fa0d60u); orig(c); }
#endif

const XrcManual xrc_manual_simd[] = {
	{0x00f99040u, xo_00f99040, "quaternion: unpack from 64 bits (fast, battle 2)"},
	{0x00f98690u, xo_00f98690, "keyframe groups: the keys around a time (fast, battle 2)"},
	{0x00fa0700u, xo_00fa0700, "quaternion track: slerped key at t (fast, battle 2)"},
	{0x00cd8240u, xo_00cd8240, "curve: polynomial rows at x (fast, battle 2)"},
	{0x00a39b20u, xo_00a39b20, "color: 4 floats to a byte-swapped dword (fast, battle 2)"},
	{0x00f99680u, xo_00f99680, "packed rotation keys: decode two (fast, battle 2)"},
	{0x00d7e170u, xo_00d7e170, "triangle rows, winding by sign (fast, battle 2)"},
	{0x00db7310u, xo_00db7310, "segment-segment closest points (fast, battle 2)"},
	{0x00462d70u, xo_00462d70, "Matrix44 identity (fast, battle 2)"},
	{0x0045e580u, xo_0045e580, "sincos_ps (fast, battle 2)"},
	{0x00462880u, xo_00462880, "Matrix44 from angles (sincos) (fast, battle 2)"},
	{0x00492460u, xo_00492460, "Matrix44 translation (fast, battle 2)"},
	{0x00dc4010u, xo_00dc4010, "distance constraints (ring) (fast, battle 2)"},
	{0x00dc8090u, xo_00dc8090, "transformed point and direction (fast, battle 2)"},
	{0x00d7c9d0u, xo_00d7c9d0, "ray / Aabb test (fast, battle 2)"},
	{0x008d6650u, xo_008d6650, "timers: advance 8, any still running (fast, battle 2)"},
	{0x00a81a20u, xo_00a81a20, "curve: value at a time (fast, battle 2)"},
	{0x00a81ca0u, xo_00a81ca0, "curve: value at a time, looped (fast, battle 2)"},
	{0x00f13f50u, xo_00f13f50, "two transformed points: no lane negative (fast, battle 2)"},
	{0x00da7c90u, xo_00da7c90, "wave phases and per-object offsets (fast, battle 2)"},
	{0x00f13c90u, xo_00f13c90, "BVH: visible objects nearest first (fast, battle 3)"},
	{0x00f0f640u, xo_00f0f640, "octree: objects' parts visible flags (fast, battle 3)"},
	{0x00f9a340u, xo_00f9a340, "animation channels: sample at t (fast, battle 3)"},
	{0x00f4a3c0u, xo_00f4a3c0, "shader constants (material) (fast, battle 3)"},
	{0x00f87990u, xo_00f87990, "light shader constants (fast, battle 3)"},
	{0x00c304e0u, xo_00c304e0, "look-at weight (fast, battle 3)"},
	{0x00f30350u, xo_00f30350, "material: shader constants and depth bias (fast, battle 3)"},
	{0x00ba7340u, xo_00ba7340, "curves: evaluate the masked channels at t (fast, battle 3)"},
	{0x00fa0d60u, xo_00fa0d60, "packed curve: value at t (fast, battle 3)"},
	{0x0101a95du, xo_0101a95d, "__VEC_memzero, the CRT memset zero fill (fast)"},
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
	{0x00e64d50u, xo_00e64d50, "Bone::LocalMatrix (fast)"},
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
