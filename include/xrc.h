/* xrc.h - what the C that xrc generates compiles against (the generated pages include only this). A copy of the header
 * the FINAL FANTASY XIII port's runtime uses: the runtime (memory, threads, the dispatcher, the platform layer) is not part of xrc.
 *
 * A translated function `void xf_<addr>(XCtx *c)` keeps the x86 registers in locals (r_eax..r_edi, r_esp), the
 * flags as separate values computed only where read, the x87 stack as doubles at statically known depths, and
 * reaches guest memory through M8/M16/M32/... (guest address -> host pointer: XM + address).
 *
 * Two builds of the same C:
 *   XRC_INGAME  (VC++ 2008, 32-bit, linked into the retail exe by the port's own linker tool): guest addresses are host
 *               addresses (XM = 0). `sub_<addr>` is a naked entry thunk that saves the real registers into an XCtx
 *               and calls xf_<addr>; calls out of a translated function run the original (native) code with the
 *               real registers and esp loaded (XCALL). The guest stack of a translated function is a copy of its
 *               incoming frame placed below its own C frame (XRC_PROLOGUE), so native callees never overwrite
 *               the C frames. This build exists to verify the translation in the running game (live-check).
 *   portable    (any C compiler, 32/64-bit, any CPU): XM is the base of a 4 GB guest memory block; calls dispatch
 *               to other translated functions or to the high-level platform layer by guest address (xrc_dispatch).
 */
#ifndef XRC_H
#define XRC_H

typedef unsigned char u8;
typedef signed char s8;
typedef unsigned short u16;
typedef short s16;
typedef unsigned int u32;
typedef int s32;
#ifdef _MSC_VER
typedef unsigned __int64 u64;
typedef __int64 s64;
#define XRC_INLINE static __inline
#else
typedef unsigned long long u64;
typedef long long s64;
#define XRC_INLINE static inline
#endif

/* XRC_VEC: an x86 host with SSE2 (the portable GCC builds): the packed SSE ops are the host's own instructions (xv_*
 * below), bit-identical to the lane-wise C they replace; other hosts (or -DXRC_NO_VEC, to compare) get that lane-wise C */
#if defined(__GNUC__) && defined(__SSE2__) && !defined(XRC_INGAME) && !defined(XRC_CHECK) && !defined(XRC_NO_VEC)
#include <emmintrin.h>
#define XRC_VEC 1
#endif
/* an SSE register; generated code reads/writes lanes (X[n].f[0], X[n].u[2], ...) */
typedef union X128 {
#ifdef XRC_VEC
	__m128 v;
	__m128i vi;
#endif
	float f[4];
	u32 u[4];
	s32 s[4];
	double d[2];
	u64 q[2];
	u16 w[8];
	s16 sw[8];
	u8 b[16];
} X128;

typedef struct XCtx {
	u32 eax, ecx, edx, ebx, esp, ebp, esi, edi; /* 0..28 */
	u32 eip;                                    /* 32: where the function returns (popped by its ret) */
	u32 pad;
	double st0;                                 /* 40: x87 st(0) of a float result (out) */
	double fa[2];                               /* 48: x87 st(0), st(1) passed to a callee (portable XCALLX) */
	X128 x[8];                                  /* 64: xmm0..xmm7 */
	/* 192: the guest thread's control state (portable): the x87 control word, MXCSR, EFLAGS' non-arithmetic bits.
	 * Kept here, not in thread-local storage, as the translated code reads them often (x87 function entry, rep
	 * movs / stos) and some toolchains emulate TLS with a call per access (mingw). One XCtx per guest thread. */
	u32 fcw, mxcsr, eflags_sys;
	u32 pre_n;                                  /* 204: calls since the last xrc_preempt_n (XRC_PREEMPT) */
} XCtx;

/* XRC_PNONE (an A/B experiment; clang, x86-64 only): translated functions use clang's preserve_none calling
 * convention -- the callee saves no host registers (no push / pop at every entry and exit), the caller keeps only
 * what it needs across the call. XRC_CC marks every function of that type: the translations (xrc emits it) and
 * the runtime's overrides and manual entries. Pages, dispatch table and runtime must all be built with it or without. */
#if defined(XRC_PNONE) && defined(__clang__) && defined(__x86_64__)
#define XRC_CC __attribute__((preserve_none))
#else
#undef XRC_PNONE
#define XRC_CC
#endif
typedef void (XRC_CC *XFn)(XCtx *);

/* ---- guest memory ---- */
/* x86-64 (not macOS: its programs reserve the low 4 GB): guest memory at a fixed host address by default, so its base
 * is a constant -- below 2 GB it folds into every access's displacement, a register freed (about 4% of the main
 * thread). -DXRC_NO_FIXED_BASE: the base found at run time instead (xrc_mem). 0x7fff0000: the only such base on Windows (just above its shared data page at 0x7ffe0000, in every process); its
 * 4 GB end at 0x17fff0000, clear of a Linux program's C heap (brk, after its image) -- the Windows exe is linked above
 * (--image-base 0x200000000). ARM64 has no such displacement: a register either way, so the base is found there. */
#if defined(__x86_64__) && !defined(__APPLE__) && !defined(XRC_NO_FIXED_BASE) && !defined(XRC_FIXED_BASE) && !defined(XRC_INGAME) && !defined(XRC_CHECK)
#define XRC_FIXED_BASE 0x7fff0000
#endif
#if defined(XRC_INGAME) || defined(XRC_CHECK)
#define XM ((u8 *)0)
#elif defined(XRC_FIXED_BASE)
/* XRC_FIXED_BASE (another one: -DXRC_FIXED_BASE=0x...): guest memory at that host address (reserve_guest: there or
 * not at all) */
extern u8 *xrc_mem;
#define XM ((u8 *)(unsigned long long)(XRC_FIXED_BASE))
#else
extern u8 *xrc_mem;
#ifdef XRC_PAGE
/* in the translated functions: a copy of the base made by XRC_PROLOGUE. With the global, every store to guest memory
 * (any type: -fno-strict-aliasing) could have changed it, so the compiler reloaded it after each one; the local is
 * never stored to, so it stays in a register. (xrc_mem is set once, before any guest code runs.) */
#define XM xm_
#else
#define XM xrc_mem
#endif
#endif
#define M8(a) (*(u8 *)(XM + (u32)(a)))
#define M16(a) (*(u16 *)(XM + (u32)(a)))
#define M32(a) (*(u32 *)(XM + (u32)(a)))
#define M64(a) (*(u64 *)(XM + (u32)(a)))
#define MF32(a) (*(float *)(XM + (u32)(a)))
#define MF64(a) (*(double *)(XM + (u32)(a)))
/* MX*: an access whose address is a u64 the translator built as (u32)(registers) + a small signed displacement (it
 * does that for [reg + disp]: the displacement then folds into the host instruction's, not a separate 32-bit add that
 * wraps). Only an address that would have wrapped past 0 or 4 GB differs, and none of those is a valid one. With
 * XM 0 (in-game / check builds) the same as M*. */
#if defined(XRC_INGAME) || defined(XRC_CHECK)
#define XEA_T u32
#define MX8(a) M8(a)
#define MX16(a) M16(a)
#define MX32(a) M32(a)
#define MX64(a) M64(a)
#define MXF32(a) MF32(a)
#define MXF64(a) MF64(a)
#else
#define XEA_T unsigned long long
#define MX8(a) (*(u8 *)(XM + (a)))
#define MX16(a) (*(u16 *)(XM + (a)))
#define MX32(a) (*(u32 *)(XM + (a)))
#define MX64(a) (*(u64 *)(XM + (a)))
#define MXF32(a) (*(float *)(XM + (a)))
#define MXF64(a) (*(double *)(XM + (a)))
#endif

/* per guest thread (portable: each guest thread is a host thread) */
#ifdef _MSC_VER
#define XRC_THREAD __declspec(thread)
#else
#define XRC_THREAD _Thread_local
#endif

/* fs: (the TEB) */
#ifdef XRC_INGAME
unsigned long __readfsdword(unsigned long);
void __writefsdword(unsigned long, unsigned long);
#pragma intrinsic(__readfsdword, __writefsdword)
#define FS32(off) ((u32)__readfsdword((unsigned long)(off)))
#define FS32_SET(off, v) __writefsdword((unsigned long)(off), (unsigned long)(v))
#elif defined(XRC_CHECK)
#define FS32(off) (0u)
#define FS32_SET(off, v) ((void)0)
#else
/* the runtime's per-thread state, one block per (host = guest) thread. On Windows reached by hand through a TLS slot
 * of the TEB (gs:[0x1480 + 8 * slot]): MinGW GCC has only emulated TLS (a call to __emutls_get_address per access,
 * ~3% of the main thread). A block is made at a thread's first use (xrc_tls_new) */
typedef struct XrcTls {
	u32 teb;              /* the guest TEB (fs:) */
	int imp_depth;        /* xrc_rt.c: XRC_IMPORT_TIME nesting */
	XCtx *import_ctx;
	void *ts;             /* xrc_rt.c: this thread's guest call stack (TStack *) */
	unsigned char thr[512] __attribute__((aligned(16))); /* xrc_thread.c's (ThrTls) */
} XrcTls;
#if defined(_WIN32)
extern unsigned xrc_tls_slot;
XrcTls *xrc_tls_new(void);
static inline XrcTls *xrc_tls(void) {
	XrcTls *p;
	__asm__("movq %%gs:(%1), %0" : "=r"(p) : "r"((unsigned long long)(0x1480u + 8u * xrc_tls_slot)));
	return p ? p : xrc_tls_new();
}
#else
extern XRC_THREAD XrcTls xrc_tls_v;
#define xrc_tls() (&xrc_tls_v)
#endif
#define xrc_teb (xrc_tls()->teb)
#define FS32(off) M32(xrc_teb + (u32)(off))
#define FS32_SET(off, v) (M32(xrc_teb + (u32)(off)) = (u32)(v))
#endif

/* fs: / gs:[ea] is XRC_SEG_FS / XRC_SEG_GS + ea: the segment's base, which the runtime keeps per guest thread. In a
 * Windows program fs: is the thread's TEB (the SEH chain at fs:[0], the TLS array at fs:[0x2c]); in a Linux one gs:
 * is TLS, set up by set_thread_area. A runtime may define either itself (the FINAL FANTASY XIII one: XRC_SEG_FS is
 * xrc_teb). */
#ifndef XRC_SEG_FS
extern XRC_THREAD u32 xrc_seg_fs;
#define XRC_SEG_FS xrc_seg_fs
#endif
#ifndef XRC_SEG_GS
extern XRC_THREAD u32 xrc_seg_gs;
#define XRC_SEG_GS xrc_seg_gs
#endif
/* int N (Linux's system call int 0x80, ...): the registers are in c (c->eip: the next instruction) and are
 * reloaded after it */
void xrc_int(XCtx *c, u32 n);
/* a segment register loaded (mov gs, ax ...; seg: 0 es, 1 cs, 2 ss, 3 ds, 4 fs, 5 gs): the runtime looks the selector up
 * (the descriptor set_thread_area made) and sets XRC_SEG_FS / XRC_SEG_GS */
void xrc_set_seg(XCtx *c, u32 seg, u32 sel);

/* ---- helpers ---- */
#define PAR8(v) ((u8)(~((0x6996u >> (((v) ^ ((v) >> 4)) & 0xf)) & 1)) & 1) /* x86 PF: even number of set bits */
XRC_INLINE u32 xbsf(u32 v) { u32 i = 0; while (!(v & 1u)) { v >>= 1; i++; } return i; } /* v != 0 */
XRC_INLINE u32 xbsr(u32 v) { u32 i = 31; while (!(v >> 31)) { v <<= 1; i--; } return i; }
XRC_INLINE u32 rol32(u32 v, u32 n) { n &= 31; return n ? (v << n) | (v >> (32 - n)) : v; }
XRC_INLINE u32 ror32(u32 v, u32 n) { n &= 31; return n ? (v >> n) | (v << (32 - n)) : v; }
XRC_INLINE u32 bswap32(u32 v) { return (v >> 24) | ((v >> 8) & 0xff00) | ((v << 8) & 0xff0000) | (v << 24); }
XRC_INLINE float f32_of(u32 u) { union { u32 u; float f; } x; x.u = u; return x.f; }
XRC_INLINE u32 u32_of(float f) { union { u32 u; float f; } x; x.f = f; return x.u; }
XRC_INLINE double f64_of(u64 u) { union { u64 u; double d; } x; x.u = u; return x.d; }
XRC_INLINE u64 u64_of(double d) { union { u64 u; double d; } x; x.d = d; return x.u; }
/* NaN tests on the bits: they hold under -ffast-math too (the launcher's Fast Math), which assumes no NaN and may
 * drop an `x != x` or an unordered outcome of a compare -- the x86 semantics these helpers give must not change */
/* -DXRC_NAN_PROBE (pages and runtime): counts of the NaNs these tests find [0] and of the infinities x87 / SSE
 * compares see [1] (the fps line prints them) -- whether the game meets non-finite values, which -ffast-math assumes
 * it never does */
#ifdef XRC_NAN_PROBE
extern unsigned long long xrc_nonfinite_n[2];
#define XRC_PROBE(k_, c_) ((c_) ? (void)__atomic_fetch_add(&xrc_nonfinite_n[k_], 1, __ATOMIC_RELAXED) : (void)0)
#else
#define XRC_PROBE(k_, c_) ((void)0)
#endif
XRC_INLINE int xrc_isnan(double d) { int n = (u64_of(d) & 0x7fffffffffffffffull) > 0x7ff0000000000000ull; XRC_PROBE(0, n); return n; }
XRC_INLINE int xrc_isnanf(float f) { union { u32 u; float f; } x; x.f = f; return (x.u & 0x7fffffffu) > 0x7f800000u; }
/* x87 FSW condition bits of a compare (C0 0x100, C2 0x400, C3 0x4000) */
XRC_INLINE u32 fsw_cmp(double a, double b) { XRC_PROBE(1, (u64_of(a) << 1) == 0xffe0000000000000ull || (u64_of(b) << 1) == 0xffe0000000000000ull); return xrc_isnan(a) || xrc_isnan(b) ? 0x4500 : a > b ? 0 : a < b ? 0x100 : 0x4000; }
/* rounding without libm or compiler helper calls (VC9 /arch:SSE2 inlines s32<->double; s64 conversions would call
 * CRT helpers the emulator cannot resolve). 2^52 trick: adding and subtracting 2^52 rounds to nearest-even. */
#define XRC_2P52 4503599627370496.0
XRC_INLINE double xrc_rne(double d) {
#if defined(__GNUC__) && (defined(__SSE4_1__) || defined(__FAST_MATH__)) && !defined(XRC_INGAME) && !defined(XRC_CHECK)
	/* (the host's rint: roundsd with SSE4.1 (-march=native); the host's rounding mode stays nearest-even. Also under
	 * -ffast-math, which may fold the 2^52 trick below to nothing) */
	return __builtin_rint(d);
#else
	if (!(d < XRC_2P52 && d > -XRC_2P52)) return d; /* huge, inf, nan: already integral */
	return d >= 0 ? (d + XRC_2P52) - XRC_2P52 : -((-d + XRC_2P52) - XRC_2P52);
#endif
}
/* x87/SSE rounding modes: 0 nearest-even, 1 down, 2 up, 3 toward zero */
XRC_INLINE double xrc_rint(double d, u32 mode) {
	double r = xrc_rne(d);
	if (xrc_isnan(r) || r == d) return r;
	switch (mode & 3) {
	case 1: return r > d ? r - 1 : r;
	case 2: return r < d ? r + 1 : r;
	case 3: return d >= 0 ? (r > d ? r - 1 : r) : (r < d ? r + 1 : r);
	}
	return r;
}
/* the SSE control / status register (ldmxcsr / stmxcsr). The 32-bit MSVC builds use the real one: the translated
 * code's own SSE arithmetic then runs under the guest's rounding / flush-to-zero settings, as the original did.
 * Portable keeps the guest's value in xrc_mxcsr (the runtime's; reset 0x1f80) and honours only its rounding control
 * (in conversions; FTZ / DAZ and the exception flags are not modelled). */
#if defined(XRC_INGAME) || defined(XRC_CHECK)
#include <xmmintrin.h>
#define XRC_STMXCSR() ((u32)_mm_getcsr())
#define XRC_LDMXCSR(v_) _mm_setcsr((unsigned int)(v_))
#else
#define XRC_STMXCSR() (c->mxcsr)
#define XRC_LDMXCSR(v_) (c->mxcsr = (u32)(v_))
#endif
#define XRC_SSE_RC() ((XRC_STMXCSR() >> 13) & 3) /* 0 nearest, 1 down, 2 up, 3 toward zero (as xrc_rint) */

/* to integer the way x86 does it: out of range / nan -> the "integer indefinite" (0x80000000...) */
XRC_INLINE u32 xrc_i32(double r) { return (xrc_isnan(r) || r >= 2147483648.0 || r < -2147483648.0) ? 0x80000000u : (u32)(s32)r; }
XRC_INLINE u32 cvt_trunc(double d) { return xrc_i32(xrc_rint(d, 3)); }
/* cvt(ss|sd|ps|pd)2(si|dq): rounding per MXCSR.RC */
XRC_INLINE u32 cvt_round_(double d, int rc) { return xrc_i32(xrc_rint(d, rc)); }
#define cvt_round(d_) cvt_round_((d_), (int)XRC_SSE_RC()) /* (the rounding mode of the XCtx c in scope) */
XRC_INLINE u16 xrc_i16(double r) { return (xrc_isnan(r) || r >= 32768.0 || r < -32768.0) ? 0x8000u : (u16)(s16)(s32)r; }
XRC_INLINE u64 xrc_i64(double r) {
	double hi, lo; u32 l;
	if (xrc_isnan(r) || r >= 9223372036854775808.0 || r < -9223372036854775808.0) return (u64)1 << 63;
	hi = xrc_rint(r / 4294967296.0, 1); lo = r - hi * 4294967296.0; /* both exact; 0 <= lo < 2^32 */
	l = lo >= 2147483648.0 ? (u32)(s32)(lo - 4294967296.0) : (u32)(s32)lo;
	return ((u64)(u32)(s32)hi << 32) | l;
}
XRC_INLINE double xrc_u32d(u32 u) { return (double)(s32)(u ^ 0x80000000u) + 2147483648.0; }
XRC_INLINE double xrc_i64d(u64 v) { return (double)(s32)(u32)(v >> 32) * 4294967296.0 + xrc_u32d((u32)v); }
XRC_INLINE double xrc_fabs(double d) { return f64_of(u64_of(d) & 0x7fffffffffffffffull); }
/* x87 precision control (FCW bits 8-9): 0 = 24-bit (what D3D9 sets), 2 = 53-bit, 3 = 64-bit (we compute 53) */
XRC_INLINE double xrc_pc(double v, u32 fcw) { return ((fcw >> 8) & 3) == 0 ? (double)(float)v : v; }
#define XR87(v) xrc_pc((v), x_fcw)
/* lane shifts without the CRT's __allshl/__aullshr */
XRC_INLINE u64 xsh64l(u64 v, u32 n) { u32 lo = (u32)v, hi = (u32)(v >> 32); if (n > 63) return 0; if (n >= 32) { hi = lo << (n - 32); lo = 0; } else if (n) { hi = (hi << n) | (lo >> (32 - n)); lo <<= n; } return ((u64)hi << 32) | lo; }
XRC_INLINE u64 xsh64r(u64 v, u32 n) { u32 lo = (u32)v, hi = (u32)(v >> 32); if (n > 63) return 0; if (n >= 32) { lo = hi >> (n - 32); hi = 0; } else if (n) { lo = (lo >> n) | (hi << (32 - n)); hi >>= n; } return ((u64)hi << 32) | lo; }
/* 16 bytes from/to guest memory (any alignment) */
#ifdef XRC_VEC
#define XLD(x_, a_) ((x_).vi = _mm_loadu_si128((const __m128i *)(XM + (u32)(a_))))
#define XST(a_, x_) _mm_storeu_si128((__m128i *)(XM + (u32)(a_)), (x_).vi)
#define XLDX(x_, a_) ((x_).vi = _mm_loadu_si128((const __m128i *)(XM + (a_))))
#define XSTX(a_, x_) _mm_storeu_si128((__m128i *)(XM + (a_)), (x_).vi)
#else
#define XLD(x_, a_) ((x_).u[0] = M32(a_), (x_).u[1] = M32((a_) + 4), (x_).u[2] = M32((a_) + 8), (x_).u[3] = M32((a_) + 12))
#define XST(a_, x_) (M32(a_) = (x_).u[0], M32((a_) + 4) = (x_).u[1], M32((a_) + 8) = (x_).u[2], M32((a_) + 12) = (x_).u[3])
#define XLDX(x_, a_) XLD(x_, (u32)(a_))
#define XSTX(a_, x_) XST((u32)(a_), x_)
#endif


/* sqrt / reciprocal estimates / transcendental x87 ops. In the 32-bit MSVC builds these are the real instructions
 * (intrinsics / inline x87) so results are bit-identical to the original; portable uses libm (rcp/rsqrt are exact
 * there, the hardware estimates differ in the low bits - functionally equivalent). */
#if defined(XRC_INGAME) || defined(XRC_CHECK)
#include <emmintrin.h>
XRC_INLINE float xsqrtf(float x) { return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))); }
XRC_INLINE double xsqrt(double x) { return _mm_cvtsd_f64(_mm_sqrt_sd(_mm_set_sd(0), _mm_set_sd(x))); }
XRC_INLINE float xrcpf(float x) { return _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x))); }
XRC_INLINE float xrsqrtf(float x) { return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))); }
#define XRC_X87_1(name_, insn_) static double name_(double x) { double r; __asm { fld qword ptr x } __asm { insn_ } __asm { fstp qword ptr r } return r; }
#define XRC_X87_2(name_, insn_) static double name_(double y, double x) { double r; __asm { fld qword ptr y } __asm { fld qword ptr x } __asm { insn_ } __asm { fstp qword ptr r } return r; }
XRC_X87_1(xf_fsin, fsin)
XRC_X87_1(xf_fcos, fcos)
XRC_X87_1(xf_f2xm1, f2xm1)
XRC_X87_2(xf_fpatan, fpatan)  /* atan2(y, x): st1 = y, st0 = x */
XRC_X87_2(xf_fyl2x, fyl2x)    /* y * log2(x) */
static double xf_fptan(double x) { double r; __asm { fld qword ptr x } __asm { fptan } __asm { fstp st(0) } __asm { fstp qword ptr r } return r; }
static double xf_fscale(double x, double s) { double r; __asm { fld qword ptr s } __asm { fld qword ptr x } __asm { fscale } __asm { fstp st(1) } __asm { fstp qword ptr r } return r; }
static double xf_fprem(double x, double y) { double r; __asm { fld qword ptr y } __asm { fld qword ptr x } l_: __asm { fprem } __asm { fnstsw ax } __asm { test ah, 4 } __asm { jnz l_ } __asm { fstp st(1) } __asm { fstp qword ptr r } return r; }
static double xf_fprem1(double x, double y) { double r; __asm { fld qword ptr y } __asm { fld qword ptr x } l_: __asm { fprem1 } __asm { fnstsw ax } __asm { test ah, 4 } __asm { jnz l_ } __asm { fstp st(1) } __asm { fstp qword ptr r } return r; }
#else
#include <math.h>
XRC_INLINE float xsqrtf(float x) { return sqrtf(x); }
XRC_INLINE double xsqrt(double x) { return sqrt(x); }
XRC_INLINE float xrcpf(float x) { return 1.0f / x; }
XRC_INLINE float xrsqrtf(float x) { return 1.0f / sqrtf(x); }
XRC_INLINE double xf_fsin(double x) { return sin(x); }
XRC_INLINE double xf_fcos(double x) { return cos(x); }
XRC_INLINE double xf_f2xm1(double x) { return exp2(x) - 1.0; }
XRC_INLINE double xf_fpatan(double y, double x) { return atan2(y, x); }
XRC_INLINE double xf_fyl2x(double y, double x) { return y * log2(x); }
XRC_INLINE double xf_fptan(double x) { return tan(x); }
XRC_INLINE double xf_fscale(double x, double s) { return ldexp(x, (int)xrc_rint(s, 3)); }
XRC_INLINE double xf_fprem(double x, double y) { return fmod(x, y); }
XRC_INLINE double xf_fprem1(double x, double y) { return remainder(x, y); }
#endif

/* packed SSE ops on whole registers: xv_<op>(dest, src) -> the new dest (the translator's sse_vec). Each is exactly
 * the lane-wise C: min/max give the second operand when equal or unordered (as minps / maxps), the compares are the
 * cmpps predicates, rcp / rsqrt are exact divides (as xrcpf / xrsqrtf here), cvttps2dq gives 0x80000000 out of range */
#ifdef XRC_VEC
#define XV_FN static inline __attribute__((always_inline)) X128
#define XV_F(name_, e_) XV_FN name_(X128 a, X128 b) { X128 r; (void)a; (void)b; r.v = (e_); return r; }
#define XV_I(name_, e_) XV_FN name_(X128 a, X128 b) { X128 r; (void)a; (void)b; r.vi = (e_); return r; }
/* lanes where a or b is a NaN: all ones */
static inline __attribute__((always_inline)) __m128i xv_nan_(__m128i a, __m128i b) {
	const __m128i abs_ = _mm_set1_epi32(0x7fffffff), inf_ = _mm_set1_epi32(0x7f800000);
	return _mm_or_si128(_mm_cmpgt_epi32(_mm_and_si128(a, abs_), inf_), _mm_cmpgt_epi32(_mm_and_si128(b, abs_), inf_));
}
XV_F(xv_addps, _mm_add_ps(a.v, b.v))
XV_F(xv_subps, _mm_sub_ps(a.v, b.v))
XV_F(xv_mulps, _mm_mul_ps(a.v, b.v))
XV_F(xv_divps, _mm_div_ps(a.v, b.v))
XV_F(xv_minps, _mm_min_ps(a.v, b.v))
XV_F(xv_maxps, _mm_max_ps(a.v, b.v))
XV_F(xv_sqrtps, _mm_sqrt_ps(b.v))
XV_F(xv_rcpps, _mm_div_ps(_mm_set1_ps(1.0f), b.v))
XV_F(xv_rsqrtps, _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(b.v)))
XV_F(xv_andps, _mm_and_ps(a.v, b.v))
XV_F(xv_andnps, _mm_andnot_ps(a.v, b.v))
XV_F(xv_orps, _mm_or_ps(a.v, b.v))
XV_F(xv_xorps, _mm_xor_ps(a.v, b.v))
XV_F(xv_cmpps0, _mm_cmpeq_ps(a.v, b.v))
XV_F(xv_cmpps1, _mm_cmplt_ps(a.v, b.v))
XV_F(xv_cmpps2, _mm_cmple_ps(a.v, b.v))
XV_I(xv_cmpps3, xv_nan_(a.vi, b.vi)) /* (unordered / ordered on the bits: see xrc_isnan) */
XV_F(xv_cmpps4, _mm_cmpneq_ps(a.v, b.v))
XV_F(xv_cmpps5, _mm_cmpnlt_ps(a.v, b.v))
XV_F(xv_cmpps6, _mm_cmpnle_ps(a.v, b.v))
XV_I(xv_cmpps7, _mm_xor_si128(xv_nan_(a.vi, b.vi), _mm_set1_epi32(-1)))
XV_F(xv_unpcklps, _mm_unpacklo_ps(a.v, b.v))
XV_F(xv_unpckhps, _mm_unpackhi_ps(a.v, b.v))
XV_F(xv_cvtdq2ps, _mm_cvtepi32_ps(b.vi))
XV_I(xv_cvttps2dq, _mm_cvttps_epi32(b.v))
#define XV_SHUFPS(a_, b_, k_) __extension__({ X128 r_; r_.v = _mm_shuffle_ps((a_).v, (b_).v, (k_)); r_; })
#else
#ifdef __GNUC__
#define XV_FN static inline __attribute__((always_inline)) X128
#else
#define XV_FN XRC_INLINE X128
#endif
#define XV_L(name_, ty_, e_) XV_FN name_(X128 a, X128 b) { X128 r; int i; (void)a; (void)b; for (i = 0; i < 4; i++) r.ty_[i] = (e_); return r; }
XV_L(xv_addps, f, a.f[i] + b.f[i])
XV_L(xv_subps, f, a.f[i] - b.f[i])
XV_L(xv_mulps, f, a.f[i] * b.f[i])
XV_L(xv_divps, f, a.f[i] / b.f[i])
XV_L(xv_minps, f, a.f[i] < b.f[i] ? a.f[i] : b.f[i])
XV_L(xv_maxps, f, a.f[i] > b.f[i] ? a.f[i] : b.f[i])
XV_L(xv_sqrtps, f, xsqrtf(b.f[i]))
XV_L(xv_rcpps, f, xrcpf(b.f[i]))
XV_L(xv_rsqrtps, f, xrsqrtf(b.f[i]))
XV_L(xv_andps, u, a.u[i] & b.u[i])
XV_L(xv_andnps, u, ~a.u[i] & b.u[i])
XV_L(xv_orps, u, a.u[i] | b.u[i])
XV_L(xv_xorps, u, a.u[i] ^ b.u[i])
XV_L(xv_cmpps0, u, a.f[i] == b.f[i] ? 0xffffffffu : 0u)
XV_L(xv_cmpps1, u, a.f[i] < b.f[i] ? 0xffffffffu : 0u)
XV_L(xv_cmpps2, u, a.f[i] <= b.f[i] ? 0xffffffffu : 0u)
XV_L(xv_cmpps3, u, (xrc_isnanf(a.f[i]) || xrc_isnanf(b.f[i])) ? 0xffffffffu : 0u)
XV_L(xv_cmpps4, u, (xrc_isnanf(a.f[i]) || xrc_isnanf(b.f[i]) || !(a.f[i] == b.f[i])) ? 0xffffffffu : 0u)
XV_L(xv_cmpps5, u, (xrc_isnanf(a.f[i]) || xrc_isnanf(b.f[i]) || !(a.f[i] < b.f[i])) ? 0xffffffffu : 0u)
XV_L(xv_cmpps6, u, (xrc_isnanf(a.f[i]) || xrc_isnanf(b.f[i]) || !(a.f[i] <= b.f[i])) ? 0xffffffffu : 0u)
XV_L(xv_cmpps7, u, (!xrc_isnanf(a.f[i]) && !xrc_isnanf(b.f[i])) ? 0xffffffffu : 0u)
XV_FN xv_unpcklps(X128 a, X128 b) { X128 r; r.u[0] = a.u[0]; r.u[1] = b.u[0]; r.u[2] = a.u[1]; r.u[3] = b.u[1]; return r; }
XV_FN xv_unpckhps(X128 a, X128 b) { X128 r; r.u[0] = a.u[2]; r.u[1] = b.u[2]; r.u[2] = a.u[3]; r.u[3] = b.u[3]; return r; }
XV_L(xv_cvtdq2ps, f, (float)b.s[i])
XV_L(xv_cvttps2dq, u, cvt_trunc((double)b.f[i]))
XV_FN xv_shufps_(X128 a, X128 b, u32 k) { X128 r; r.u[0] = a.u[k & 3]; r.u[1] = a.u[(k >> 2) & 3]; r.u[2] = b.u[(k >> 4) & 3]; r.u[3] = b.u[(k >> 6) & 3]; return r; }
#define XV_SHUFPS(a_, b_, k_) xv_shufps_((a_), (b_), (k_))
#endif

/* 32x32->64 multiplies (MSVC: the intrinsics, so no __allmul call) */
#ifdef _MSC_VER
__int64 __emul(int, int);
unsigned __int64 __emulu(unsigned int, unsigned int);
#pragma intrinsic(__emul, __emulu)
#define XMULU(a, b) ((u64)__emulu((unsigned int)(a), (unsigned int)(b)))
#define XMULS(a, b) ((s64)__emul((int)(a), (int)(b)))
#else
#define XMULU(a, b) ((u64)(u32)(a) * (u64)(u32)(b))
#define XMULS(a, b) ((s64)(s32)(a) * (s64)(s32)(b))
#endif

/* div / idiv edx:eax by a 32-bit value without the CRT's 64-bit division helpers (__aulldvrm/__alldvrm): plain
 * 32-bit division when edx only extends eax (xor edx,edx / cdq, the usual case), else a shift-subtract long division.
 * (A quotient overflow or a zero divisor raises #DE on x86; here the result is unspecified.) */
XRC_INLINE void xdiv64_32(u32 hi, u32 lo, u32 v, u32 *q, u32 *r) {
	u32 qq = 0; int i;
	for (i = 0; i < 32; i++) {
		u32 top = hi >> 31;
		hi = (hi << 1) | (lo >> 31); lo <<= 1; qq <<= 1;
		if (top || hi >= v) { hi -= v; qq |= 1; }
	}
	*q = qq; *r = hi;
}
XRC_INLINE void xdiv32(u32 *a, u32 *d, u32 v) {
	if (v == 0) return;
	if (*d == 0) { u32 q = *a / v, r = *a % v; *a = q; *d = r; return; }
	xdiv64_32(*d, *a, v, a, d);
}
XRC_INLINE void xidiv32(u32 *a, u32 *d, u32 v_) {
	s32 v = (s32)v_;
	u32 hi = *d, lo = *a, mv, q, r; int neg;
	if (v == 0) return;
	if (hi == (u32)((s32)lo >> 31)) {
		s32 x = (s32)lo;
		if (x == (s32)0x80000000 && v == -1) return;
		*a = (u32)(x / v); *d = (u32)(x % v); return;
	}
	neg = (hi >> 31) != 0;
	if (neg) { lo = ~lo + 1; hi = ~hi + (lo == 0); }
	mv = v < 0 ? (u32)0 - (u32)v : (u32)v;
	xdiv64_32(hi, lo, mv, &q, &r);
	*a = (neg != (v < 0)) ? (u32)0 - q : q;
	*d = neg ? (u32)0 - r : r;
}

/* div/idiv edx:eax as generated code uses them. The 32-bit MSVC builds run the real instruction, so a zero divisor
 * or a quotient overflow raises #DE (EXCEPTION_INT_DIVIDE_BY_ZERO / _OVERFLOW) exactly where the original would;
 * portable uses the helpers above (TODO: raise the guest exception through the platform layer). */
#if defined(_MSC_VER) && (defined(XRC_INGAME) || defined(XRC_CHECK))
#define XRC_DIV(v_) { u32 dv_ = (v_); __asm { mov eax, r_eax } __asm { mov edx, r_edx } __asm { div dv_ } __asm { mov r_eax, eax } __asm { mov r_edx, edx } }
#define XRC_IDIV(v_) { u32 dv_ = (v_); __asm { mov eax, r_eax } __asm { mov edx, r_edx } __asm { idiv dv_ } __asm { mov r_eax, eax } __asm { mov r_edx, edx } }
#else
#define XRC_DIV(v_) xdiv32(&r_eax, &r_edx, (v_))
#define XRC_IDIV(v_) xidiv32(&r_eax, &r_edx, (v_))
#endif

/* ---- sub-register write (the 32-bit home register is read/written whole; 8/16-bit ops touch a slice) ---- */
#define SET8L(reg, v) ((reg) = ((reg) & 0xffffff00u) | ((u32)(v) & 0xffu))
#define SET8H(reg, v) ((reg) = ((reg) & 0xffff00ffu) | (((u32)(v) & 0xffu) << 8))
#define SET16(reg, v) ((reg) = ((reg) & 0xffff0000u) | ((u32)(v) & 0xffffu))
/* GET8H(reg) is (reg>>8)&0xff; low/16-bit reads are plain casts the generator emits inline */

/* ---- lazy condition flags ----
 * Each flag-setting integer op stores its kind, operand size, and up to three values (a, b=operand, r=result) in
 * cc_op/cc_sz/cc_a/cc_b/cc_r; the carry flag lives separately in `cf` so CF-preserving ops (inc/dec, logic that we
 * still want ZF/SF from) leave it alone. xcc(op,sz,a,b,r) returns ZF|SF|OF|PF packed; CF is read from `cf`. A branch
 * or setcc calls xcond(). cmp/test immediately followed by a jcc are emitted as a direct comparison instead.
 */
#define CC_ADD 1
#define CC_SUB 2
#define CC_LOGIC 3 /* and/or/xor/test: CF=OF=0 */
#define CC_INC 4   /* add 1, CF preserved */
#define CC_DEC 5   /* sub 1, CF preserved */
#define CC_SHL 6
#define CC_SHR 7   /* logical right */
#define CC_SAR 8   /* arithmetic right */
#define CC_NEG 9   /* r = -a ; CF = a!=0 (set in cf separately) */
#define CC_IMUL 10 /* CF=OF set when the high half is significant (set in cf) */
#define CC_RAW 11  /* cc_r already holds ZF|SF|OF|PF (fcomi, comiss, sahf) */
#define CC_MUL 12  /* mul/imul: OF = CF = cc_a (the high half is significant); ZF/SF/PF from the low result (as
                    * qemu and current Intel parts compute them; architecturally undefined) */
/* packed result bits of xcc() */
#define XF_ZF 1u
#define XF_SF 2u
#define XF_OF 4u
#define XF_PF 8u
XRC_INLINE u32 xcc_mask(u8 sz) { return sz >= 4 ? 0xffffffffu : (1u << (sz * 8)) - 1u; }
XRC_INLINE u32 xcc_sign(u8 sz) { return 1u << (sz * 8 - 1); }
/* ZF|SF|OF|PF for the stored op (CF handled via `cf`) */
XRC_INLINE u32 xcc(u8 op, u8 sz, u32 a, u32 b, u32 r) {
	u32 m = xcc_mask(sz), s = xcc_sign(sz), rr = r & m, f = 0;
	if (op == CC_RAW) return r;
	if (rr == 0) f |= XF_ZF;
	if (rr & s) f |= XF_SF;
	if (PAR8((u8)rr)) f |= XF_PF;
	switch (op) {
	case CC_ADD: case CC_INC: if ((~(a ^ b) & (a ^ r)) & s) f |= XF_OF; break;
	case CC_SUB: case CC_DEC: if (((a ^ b) & (a ^ r)) & s) f |= XF_OF; break;
	case CC_NEG: if ((a & m) == s) f |= XF_OF; break;
	case CC_MUL: if (a) f |= XF_OF; break;
	/* CC_LOGIC, CC_SHL/SHR/SAR, CC_IMUL: OF either 0 or already folded into cf for our purposes */
	default: break;
	}
	return f;
}
/* pushfd / popfd. The arithmetic flags come from / go to the lazy flags; the rest of EFLAGS (IF, DF, AC, ID, ...)
 * is the guest's in xrc_eflags_sys, so a popfd that flips ID (the CPUID-support test) reads back flipped, and DF
 * (cld / std / popfd) gives the string instructions their direction. */
#define XRC_EFL_ARITH 0x8d5u /* CF PF AF ZF SF OF */
#define XRC_EFL_DF 0x400u    /* the string instructions' direction (cld / std) */
#if defined(XRC_INGAME) || defined(XRC_CHECK)
static u32 xrc_eflags_sys = 0x202u; /* per translation unit: the ID-flag test reads it back, and std / rep movs /
                                    * cld come in the same function (memmove's backward copy) */
#define XRC_EFLSYS xrc_eflags_sys
#else
#define XRC_EFLSYS (c->eflags_sys)
#ifndef XRC_RUNTIME
#define xrc_eflags_sys (c->eflags_sys) /* (the name the translations use: cld / std / rep) */
#endif
#endif
#define xrc_eflags_pack(f_, cf_) xrc_eflags_pack_(XRC_EFLSYS, (f_), (cf_))
#define xrc_eflags_unpack(v_) xrc_eflags_unpack_(&XRC_EFLSYS, (v_))
XRC_INLINE u32 xrc_eflags_pack_(u32 sys, u32 f, u32 cf) {
	return (sys & ~XRC_EFL_ARITH) | 2u | (cf ? 1u : 0u) | ((f & XF_PF) ? 4u : 0u) | ((f & XF_ZF) ? 0x40u : 0u) |
	       ((f & XF_SF) ? 0x80u : 0u) | ((f & XF_OF) ? 0x800u : 0u);
}
/* the ZF|SF|OF|PF of an EFLAGS value (for CC_RAW; CF goes to cf), keeping its system bits */
XRC_INLINE u32 xrc_eflags_unpack_(u32 *sys, u32 v) {
	*sys = (v & ~XRC_EFL_ARITH) | 2u;
	return ((v & 0x40u) ? XF_ZF : 0u) | ((v & 0x80u) ? XF_SF : 0u) | ((v & 0x800u) ? XF_OF : 0u) | ((v & 4u) ? XF_PF : 0u);
}

/* an invalid instruction (ud2, FE /7 ...): #UD. MSVC builds execute a real ud2 (the guest's SEH sees it); portable
 * stops with a report. */
#if defined(XRC_INGAME) || defined(XRC_CHECK)
#define XRC_UD(a_) __asm { ud2 }
#else
void xrc_ud(XCtx *c, u32 addr);
#define XRC_UD(a_) do { xrc_ud(c, (a_)); return; } while (0)
#endif

/* cpuid: one fixed virtual CPU in every build, so the game's feature checks pick the same code paths everywhere --
 * and only paths the translator handles: x87, TSC, CX8, CMOV, FXSR, SSE, SSE2. No MMX, no SSE3 or later (the
 * translator does not translate them; real CPUs all have MMX, but the game's checks gate on SSE / SSE2). One core,
 * one logical processor. Leaf 0 says "GenuineIntel" with max leaf 1; extended leaves are absent. */
XRC_INLINE void xrc_cpuid(u32 *a, u32 *b, u32 *c, u32 *d) {
	u32 leaf = *a;
	if (leaf == 0) {
		*a = 1; *b = 0x756e6547u; *d = 0x49656e69u; *c = 0x6c65746eu; /* "Genu" "ineI" "ntel" */
	} else if (leaf == 1) {
		*a = 0x000006fbu;  /* family 6, model 15, stepping 11 (a Core 2) */
		*b = 0x00010800u;  /* CLFLUSH line 8 qwords, 1 logical processor */
		*c = 0;            /* no SSE3 .. */
		*d = (1u << 0) | (1u << 4) | (1u << 5) | (1u << 8) | (1u << 15) | (1u << 19) | (1u << 24) | (1u << 25) | (1u << 26);
		                   /* FPU TSC MSR CX8 CMOV CLFSH FXSR SSE SSE2 */
	} else if (leaf == 0x80000000u) {
		*a = 0x80000000u; *b = *c = *d = 0;
	} else {
		*a = *b = *c = *d = 0;
	}
}

/* the saturating SIMD integer ops (packs, padds / psubs, pmaddubsw): v clamped to [lo, hi] */
XRC_INLINE s32 xsat(s32 v, s32 lo, s32 hi) { return v < lo ? lo : v > hi ? hi : v; }
/* rdtsc: the host's time-stamp counter on an x86 host, elsewhere a counter that only goes up (code reads it for
 * timings and as a seed) */
XRC_INLINE u64 xrc_rdtsc(void) {
#if defined(__x86_64__) || defined(__i386__)
	return __builtin_ia32_rdtsc();
#else
	static u64 t;
	return t += 1000;
#endif
}

/* carry produced by add/sub/neg (shifts set cf inline) */
XRC_INLINE u32 xcf_add(u8 sz, u32 a, u32 r) { return (r & xcc_mask(sz)) < (a & xcc_mask(sz)); }
XRC_INLINE u32 xcf_sub(u8 sz, u32 a, u32 b) { return (a & xcc_mask(sz)) < (b & xcc_mask(sz)); }
/* carry out of adc / borrow out of sbb, with carry-in c */
XRC_INLINE u32 xcf_adc(u8 sz, u32 a, u32 b, u32 c) { u32 m = xcc_mask(sz); a &= m; b &= m; return c ? ((a + b + 1) & m) <= a : ((a + b) & m) < a; }
XRC_INLINE u32 xcf_sbb(u8 sz, u32 a, u32 b, u32 c) { u32 m = xcc_mask(sz); a &= m; b &= m; return c ? a <= b : a < b; }
/* x86 condition codes 0..15 (tttn): O NO B AE E NE BE A S NS P NP L GE LE G */
XRC_INLINE int xcond(int cc, u32 cf, u32 f) {
	int zf = (f & XF_ZF) != 0, sf = (f & XF_SF) != 0, of = (f & XF_OF) != 0, pf = (f & XF_PF) != 0, c = cf != 0;
	switch (cc) {
	case 0: return of;              case 1: return !of;
	case 2: return c;               case 3: return !c;
	case 4: return zf;              case 5: return !zf;
	case 6: return c || zf;         case 7: return !(c || zf);
	case 8: return sf;              case 9: return !sf;
	case 10: return pf;             case 11: return !pf;
	case 12: return sf != of;       case 13: return sf == of;
	case 14: return zf || (sf != of); case 15: return !(zf || (sf != of));
	}
	return 0;
}

/* a translation made with XRC_FL_VERIFY=1 (src/xrc_flags.odin): each condition evaluated the flag plan's way
 * (n) and the old way (o); a difference is reported (xrc_rt.c), the old one is used */
void xrc_flbad(u32 addr);
XRC_INLINE int xrc_flchk(u32 addr, int n, int o) {
	if ((n != 0) != (o != 0)) xrc_flbad(addr);
	return o;
}

/* fcomi/fucomi/comiss/ucomiss/comisd/ucomisd: ZF PF CF = 000 a>b, 001 a<b, 100 a==b, 111 unordered; OF=SF=0 */
#define XFCOMI(a_, b_) { double xa_ = (a_), xb_ = (b_); cc_op = CC_RAW; \
	XRC_PROBE(1, (u64_of(xa_) << 1) == 0xffe0000000000000ull || (u64_of(xb_) << 1) == 0xffe0000000000000ull); \
	if (xrc_isnan(xa_) || xrc_isnan(xb_)) { cc_r = XF_ZF | XF_PF; cf = 1; } \
	else { cc_r = xa_ > xb_ ? 0u : xa_ < xb_ ? 0u : XF_ZF; cf = xa_ < xb_; } }
/* the same compare fused into its readers (src/xrc_flags.odin: cc_fa / cc_fb): unordered, by bits (the pages
 * are built with finite math), and XFCOMI's probe */
XRC_INLINE int xfc_un(double a, double b) { return xrc_isnan(a) || xrc_isnan(b); }
#define XFC_PROBE(a_, b_) XRC_PROBE(1, (u64_of(a_) << 1) == 0xffe0000000000000ull || (u64_of(b_) << 1) == 0xffe0000000000000ull)

/* ---- trimmed register syncs (xrc_translator x_sync_plan) ----
 * Which guest registers a call or a return stores to XCtx (x_sm_: the ones the function changed since XCtx last had
 * them -- the others are there already) and which a call reloads after (x_fm_: the ones read before the next write or
 * call -- the others' locals are stale until then, never stored). Bits: 1 eax, 2 ecx, 4 edx, 8 ebx, 0x10 ebp,
 * 0x20 esi, 0x40 edi; esp always both. The translator puts a block's own enum around each call / return it trims;
 * everything else gets this file-scope one (all). -DXRC_NO_SYNC_TRIM: all everywhere (to compare). */
#if defined(XRC_PAGE) && !defined(XRC_ABI_CHECK) && !defined(XRC_NO_SYNC_TRIM) && !defined(XRC_INGAME) && !defined(XRC_CHECK)
enum { x_sm_ = 0x7f, x_fm_ = 0x7f };
#define XRC_SM_ x_sm_
#define XRC_FM_ x_fm_
#else
#define XRC_SM_ 0x7f
#define XRC_FM_ 0x7f
#endif
#define XRC_SPILL_REGS { if (XRC_SM_ & 1) c->eax = r_eax; if (XRC_SM_ & 2) c->ecx = r_ecx; if (XRC_SM_ & 4) c->edx = r_edx; \
	if (XRC_SM_ & 8) c->ebx = r_ebx; if (XRC_SM_ & 0x10) c->ebp = r_ebp; if (XRC_SM_ & 0x20) c->esi = r_esi; if (XRC_SM_ & 0x40) c->edi = r_edi; }

/* ---- return (ret / ret N): pop the guest return address, publish registers, map esp back ---- */
#define XRC_RETURN(nbytes)                                                                                          \
	do {                                                                                                            \
		u32 _ip_ = MX32(r_esp);                                                                                      \
		r_esp += 4u + (u32)(nbytes);                                                                                \
		XRC_SPILL_REGS                                                                                              \
		c->esp = XRC_ESP_OUT; c->eip = _ip_;                                                                        \
		return;                                                                                                     \
	} while (0)

/* ---- an inlined leaf's ret (xrc x_inline_site): the return address its call pushed popped (checked: a callee
 * that changed its return address is fatal, as it is for a call -- xrc_bad_return), the arguments too (ret N), then on
 * past the body (lbl_, the site's) */
void xrc_bad_return(XCtx *c, u32 target);
/* (XRC_STACK_VERIFY=1 translations: a direct call to a function the translator found stack-balanced left esp elsewhere) */
void xrc_stack_bad(XCtx *c, u32 target, u32 esp, u32 expected);
/* the call macros inside an inlined SSE body (xrc_translator x_inline_plan): the same ones, under names the
 * translator's renaming of an SSE caller's calls (to XCALLM*: its own xmm registers around them) does not touch --
 * the body moves its own (XI) around the call itself */
#define XCDK_ XCALLDK
#define XCD_ XCALLD
#define XC_ XCALL
#define XRC_INL_RET(nbytes, ra_, tgt_, lbl_)                                                                         \
	{                                                                                                               \
		u32 _ip_ = MX32(r_esp);                                                                                      \
		r_esp += 4u + (u32)(nbytes);                                                                                \
		if (_ip_ != (ra_)) {                                                                                        \
			c->eax = r_eax; c->ecx = r_ecx; c->edx = r_edx; c->ebx = r_ebx;                                         \
			c->ebp = r_ebp; c->esi = r_esi; c->edi = r_edi; c->esp = r_esp; c->eip = _ip_;                           \
			xrc_bad_return(c, (tgt_));                                                                              \
		}                                                                                                           \
		goto lbl_;                                                                                                  \
	}

/* ---- entry / calls ---- */
#if defined(XRC_INGAME) || defined(XRC_CHECK)

/* Stacks. The guest code runs on the real thread stack, exactly where the original would (so stack addresses, the
 * security cookie's esp mix, arguments of any size, varargs and SEH frames are all the original's), and nothing of the
 * translation is ever written to it beyond what the original's own call/ret write -- uninitialized guest locals see
 * the same bytes as in the original. The C code of the translations runs on a separate host stack: the entry thunk
 * swaps esp with a per-thread slot (`xchg esp, [slot]`, no register needed) that holds the current top of the host
 * stack; a call out (XCALLX) sets the slot below its own frame so a translation entered from the native callee
 * stacks its host frame below.
 *   XRC_CHECK : the slot is a fixed word in the emulator's stack region (single thread); host stack below
 *               0x6000fff0, the guest stack starts at 0x6001f000. Nothing outside the stack region is written.
 *   XRC_INGAME: the slot is fs:[0x14] (NT_TIB.ArbitraryUserPointer, left to the application; the loader restores it
 *               around image mapping); a thread's host stack is allocated at its first entry. */
#ifdef XRC_CHECK
#define XRC_SLOT_ASM dword ptr ds:[0x6000fffc]
#define XRC_SLOT_GET() (*(u32 *)0x6000fffcu)
#define XRC_SLOT_SET(v) (*(u32 *)0x6000fffcu = (u32)(v))
/* first entry of a run (the emulator zero-fills its stack): start the host stack */
#define XRC_SLOT_INIT_ASM __asm { mov dword ptr ds:[0x6000fffc], 0x6000fff0 }
#else
#define XRC_SLOT_ASM dword ptr fs:[0x14]
#define XRC_SLOT_GET() ((u32)__readfsdword(0x14))
#define XRC_SLOT_SET(v) __writefsdword(0x14, (unsigned long)(v))
#define XRC_HOST_SIZE 0x100000u
__declspec(dllimport) void *__stdcall VirtualAlloc(void *, unsigned long, unsigned long, unsigned long);
/* a thread's first entry: its host stack (runs once per thread, on the guest stack: the only guest-stack writes) */
static void __cdecl xrc_host_alloc(void) {
	u8 *p = (u8 *)VirtualAlloc(0, XRC_HOST_SIZE, 0x3000 /* MEM_COMMIT|MEM_RESERVE */, 4 /* PAGE_READWRITE */);
	XRC_SLOT_SET(p + XRC_HOST_SIZE - 16);
}
#define XRC_SLOT_INIT_ASM __asm { pushad } __asm { call xrc_host_alloc } __asm { popad }
#endif

/* blk (on the host stack; &blk[42] is the host top S the slot held), in words:
 *   [0..1] st0 out   [2..33] xmm0..xmm7   [34] ebp [35] edi [36] esi [37] ebx [38] edx [39] ecx [40] eax
 *   [41] the guest esp at entry (pointing at the xf_ function the thunk pushed; the return address is above it) */
static void __cdecl xrc_enter_c(u32 *blk) {
	XCtx c;
	int k;
	u32 g = blk[41];
	c.ebp = blk[34]; c.edi = blk[35]; c.esi = blk[36]; c.ebx = blk[37];
	c.edx = blk[38]; c.ecx = blk[39]; c.eax = blk[40];
	c.esp = g + 4;
	c.eip = 0; c.st0 = 0;
	for (k = 0; k < 32; k++) ((u32 *)c.x)[k] = blk[2 + k];
	((XFn) * (u32 *)g)(&c);
	for (k = 0; k < 32; k++) blk[2 + k] = ((u32 *)c.x)[k];
	*(double *)&blk[0] = c.st0;
	blk[34] = c.ebp; blk[35] = c.edi; blk[36] = c.esi; blk[37] = c.ebx;
	blk[38] = c.edx; blk[39] = c.ecx; blk[40] = c.eax;
	/* the thunk's final `xchg esp, [slot]; ret` returns to c.eip at exactly the final guest esp. (For a plain ret
	 * the word below the final esp is the return address slot itself; after ret N it is the last argument's.) */
	*(u32 *)(c.esp - 4) = c.eip;
	XRC_SLOT_SET(c.esp - 4);
}
/* a float-returning translated function sets st(0) for the xrc_enterf thunk */
#define XRC_FRET(v) (c->st0 = (double)(v))
#define XRC_ENTER_BODY(fld_)                                                                                      \
	__asm { xrc_retry_: }                                                                                         \
	__asm { xchg esp, XRC_SLOT_ASM }                                                                              \
	__asm { test esp, esp }                                                                                       \
	__asm { jnz xrc_have_host_ }                                                                                  \
	__asm { xchg esp, XRC_SLOT_ASM }                                                                              \
	XRC_SLOT_INIT_ASM                                                                                             \
	__asm { jmp xrc_retry_ }                                                                                      \
	__asm { xrc_have_host_: }                                                                                     \
	__asm { push XRC_SLOT_ASM }                                                                                   \
	__asm { push eax }  __asm { push ecx }  __asm { push edx }  __asm { push ebx }                                \
	__asm { push esi }  __asm { push edi }  __asm { push ebp }                                                    \
	__asm { sub esp, 136 }                                                                                        \
	__asm { movups [esp + 8], xmm0 }   __asm { movups [esp + 24], xmm1 }                                         \
	__asm { movups [esp + 40], xmm2 }  __asm { movups [esp + 56], xmm3 }                                         \
	__asm { movups [esp + 72], xmm4 }  __asm { movups [esp + 88], xmm5 }                                         \
	__asm { movups [esp + 104], xmm6 } __asm { movups [esp + 120], xmm7 }                                        \
	__asm { push esp }                                                                                            \
	__asm { call xrc_enter_c }                                                                                    \
	__asm { add esp, 4 }                                                                                          \
	__asm { movups xmm0, [esp + 8] }   __asm { movups xmm1, [esp + 24] }                                         \
	__asm { movups xmm2, [esp + 40] }  __asm { movups xmm3, [esp + 56] }                                         \
	__asm { movups xmm4, [esp + 72] }  __asm { movups xmm5, [esp + 88] }                                         \
	__asm { movups xmm6, [esp + 104] } __asm { movups xmm7, [esp + 120] }                                        \
	fld_                                                                                                          \
	__asm { add esp, 136 }                                                                                        \
	__asm { pop ebp }  __asm { pop edi }  __asm { pop esi }  __asm { pop ebx }                                    \
	__asm { pop edx }  __asm { pop ecx }  __asm { pop eax }                                                       \
	__asm { add esp, 4 }                                                                                          \
	__asm { xchg esp, XRC_SLOT_ASM }                                                                              \
	__asm { ret }
/* the shared body of the entry thunks: [esp] = xf_ function, [esp+4] = return address */
static __declspec(naked) void xrc_enter(void) { XRC_ENTER_BODY(;) }
/* the same for a function that returns a float in st(0) */
static __declspec(naked) void xrc_enterf(void) { XRC_ENTER_BODY(__asm { fld qword ptr [esp] }) }
#define XRC_THUNK(addr) __declspec(naked) void sub_##addr(void) { __asm { push offset xf_##addr } __asm { jmp xrc_enter } }
#define XRC_THUNKF(addr) __declspec(naked) void sub_##addr(void) { __asm { push offset xf_##addr } __asm { jmp xrc_enterf } }

#define XRC_PROLOGUE { r_esp = c->esp; }
#define XRC_ESP_OUT (r_esp)

/* xmm0-7 around a native call: a function that uses SSE keeps the guest xmm registers in c->x (X) and calls out
 * with XCALLXM / XCALLM, which load them into the real registers for the call and read them back after it */
#define XRC_XLOAD1                                                                                                \
	__asm { mov eax, X }                                                                                          \
	__asm { movups xmm0, [eax] }      __asm { movups xmm1, [eax + 16] }                                         \
	__asm { movups xmm2, [eax + 32] } __asm { movups xmm3, [eax + 48] }                                         \
	__asm { movups xmm4, [eax + 64] } __asm { movups xmm5, [eax + 80] }                                         \
	__asm { movups xmm6, [eax + 96] } __asm { movups xmm7, [eax + 112] }
#define XRC_XSAVE1                                                                                                \
	__asm { mov eax, X }                                                                                          \
	__asm { movups [eax], xmm0 }      __asm { movups [eax + 16], xmm1 }                                         \
	__asm { movups [eax + 32], xmm2 } __asm { movups [eax + 48], xmm3 }                                         \
	__asm { movups [eax + 64], xmm4 } __asm { movups [eax + 80], xmm5 }                                         \
	__asm { movups [eax + 96], xmm6 } __asm { movups [eax + 112], xmm7 }
#define XRC_XLOAD0
#define XRC_XSAVE0

/* a call to native code: all eight registers are the guest's for the call, ebp included -- CRT helpers such as
 * __EH_prolog3 / __SEH_prolog4 build the caller's frame and return a new ebp, EH funclets run on the parent's ebp,
 * and ff check locates stack objects from it. The C frame's state is parked in the host-stack area just below this
 * frame (x_hs: [0] host ebp, [4] host esp, [8] host ebx -- VC9 addresses parameters via ebx in realigned frames),
 * which is also what the per-thread slot points at during the call, so after the callee returns `xchg esp, slot`
 * reaches it with no free register. The target goes in the guest stack word the call's push then overwrites.
 * kin_ x87 values (x_fa[0] = st0, x_fa[1] = st1) are pushed on the real FPU first; kout_ = 1 pops a float result
 * into x_fret. kin_/kout_ are constants, so the ifs fold away. */
#define XRC_CALL_(tgt_, kin_, kout_, XLOAD_, XSAVE_)                                                              \
	{                                                                                                             \
		u32 t_ = (tgt_);                                                                                          \
		__asm { mov x_hesp, esp }                                                                                 \
		x_hs = x_hesp - 64;                                                                                       \
		x_hold = XRC_SLOT_GET();                                                                                  \
		XRC_SLOT_SET(x_hs); /* a translation entered from the callee stacks its host frame below x_hs */          \
		M32(r_esp - 4) = t_;                                                                                      \
		XLOAD_                                                                                                    \
		if ((kin_) >= 2) { __asm { fld qword ptr x_fa[8] } }                                                     \
		if ((kin_) >= 1) { __asm { fld qword ptr x_fa[0] } }                                                     \
		__asm { mov eax, x_hs }                                                                                   \
		__asm { mov [eax], ebp }                                                                                  \
		__asm { mov [eax + 4], esp }                                                                              \
		__asm { mov [eax + 8], ebx }                                                                              \
		__asm { mov ecx, r_ecx }                                                                                  \
		__asm { mov edx, r_edx }                                                                                  \
		__asm { mov ebx, r_ebx }                                                                                  \
		__asm { mov esi, r_esi }                                                                                  \
		__asm { mov edi, r_edi }                                                                                  \
		__asm { mov eax, r_eax }                                                                                  \
		__asm { mov esp, r_esp }                                                                                  \
		__asm { mov ebp, r_ebp }                                                                                  \
		__asm { call dword ptr [esp - 4] }                                                                        \
		__asm { xchg esp, XRC_SLOT_ASM }                                                                          \
		__asm { push ebp }                                                                                        \
		__asm { mov ebp, [esp + 4] }                                                                              \
		__asm { pop r_ebp }                                                                                       \
		__asm { mov r_eax, eax }                                                                                  \
		__asm { mov r_ecx, ecx }                                                                                  \
		__asm { mov r_edx, edx }                                                                                  \
		__asm { mov r_ebx, ebx }                                                                                  \
		__asm { mov r_esi, esi }                                                                                  \
		__asm { mov r_edi, edi }                                                                                  \
		__asm { mov eax, XRC_SLOT_ASM }                                                                           \
		__asm { mov r_esp, eax }                                                                                  \
		__asm { mov ebx, [esp + 8] }                                                                              \
		__asm { mov esp, [esp + 4] }                                                                              \
		XRC_SLOT_SET(x_hold);                                                                                     \
		XSAVE_                                                                                                    \
		if ((kout_) >= 1) { __asm { fstp qword ptr x_fret } }                                                    \
	}
#define XCALLX(tgt_, kin_, kout_) XRC_CALL_(tgt_, kin_, kout_, XRC_XLOAD0, XRC_XSAVE0)
#define XCALLXM(tgt_, kin_, kout_) XRC_CALL_(tgt_, kin_, kout_, XRC_XLOAD1, XRC_XSAVE1)
#define XCALL(tgt) XCALLX(tgt, 0, 0)
#define XCALLM(tgt) XCALLXM(tgt, 0, 0)
/* a call whose target is known (fn: its translation): native here, as every call */
#define XCALLD(tgt, fn) XCALL(tgt)
#define XCALLMD(tgt, fn) XCALLM(tgt)
#define XRC_SPIN_YIELD()
#define XRC_INL_PREEMPT()
#define XCALLXD(tgt, kin_, kout_, fn) XCALLX(tgt, kin_, kout_)
#define XCALLXMD(tgt, kin_, kout_, fn) XCALLXM(tgt, kin_, kout_)
#define XCALLDK(tgt, fn, m_) XCALLD(tgt, fn)
#define XCALLMDK(tgt, fn, m_) XCALLMD(tgt, fn)
#define XCALLXDK(tgt, kin_, kout_, fn, m_) XCALLXD(tgt, kin_, kout_, fn)
#define XCALLXMDK(tgt, kin_, kout_, fn, m_) XCALLXMD(tgt, kin_, kout_, fn)
#define XRC_LOCALS u32 x_hesp, x_hs, x_hold, x_ra_; double x_fret, x_fa[2];
/* With double/SSE locals VC9 realigns the stack and addresses the parameters (c) through ebx (warning C4731):
 * XCALLX saves and restores the host ebx around the guest's, so that is safe. */
#pragma warning(disable : 4731)
/* the x87 control word is the real one (precision / rounding of the running game) */
#define XRC_FCW_INIT { u16 w_; __asm { fnstcw w_ } x_fcw = w_; }
#define XRC_FLDCW(v_) { u16 w_ = (u16)(v_); x_fcw = w_; __asm { fldcw w_ } }

/* (the xmm registers stay in c->x, which XCALLXM loads into the real ones) */
#define XRC_FN_ENTER(a_)
#define XRC_XDECL X128 *X = c->x;
#define XRC_XENTRY
#define XRC_RETURNX(n) XRC_RETURN(n)
#else /* portable */

void xrc_dispatch(XCtx *c, u32 target); /* runs the function at guest address target with c (pushes nothing) */
#define XRC_THUNK(addr)
/* every translated function's entry: nothing, or with -DXRC_COUNT_CALLS a count of its calls (an analysis build; the
 * runtime lists the most called ones: XRC_COUNT_FILE) */
#ifdef XRC_COUNT_CALLS
typedef struct XrcCount { u32 addr, reg; unsigned long long n, base; struct XrcCount *next; } XrcCount;
void xrc_count_reg(XrcCount *k);
#define XRC_FN_ENTER(a_) { static XrcCount k_ = {(a_), 0, 0, 0, 0}; if (!k_.reg) xrc_count_reg(&k_); k_.n++; }
#else
#define XRC_FN_ENTER(a_)
#endif
#define XRC_THUNKF(addr)
#define XRC_PROLOGUE u8 *const xm_ = xrc_mem; r_esp = c->esp;
#define XRC_ESP_OUT (r_esp)
#define XRC_SPILL XRC_SPILL_REGS c->esp = r_esp;
#define XRC_FILL { if (XRC_FM_ & 1) r_eax = c->eax; if (XRC_FM_ & 2) r_ecx = c->ecx; if (XRC_FM_ & 4) r_edx = c->edx; \
	if (XRC_FM_ & 8) r_ebx = c->ebx; if (XRC_FM_ & 0x10) r_ebp = c->ebp; if (XRC_FM_ & 0x20) r_esi = c->esi; if (XRC_FM_ & 0x40) r_edi = c->edi; \
	r_esp = c->esp; }
/* a guest call: push the original's return address (x_ra_), which the callee's ret pops (checked against c->eip) */
extern volatile int xrc_gil_waiters;
void xrc_preempt(void);
void xrc_preempt_n(XCtx *c);
/* a call is a point where a thread waiting for the GIL may get it: counted in the guest thread's XCtx (a register
 * away), xrc_preempt (thread-local counter, the clock) only every XRC_PREEMPT_N calls */
#define XRC_PREEMPT_N 64u
#define XRC_PREEMPT() { if (xrc_gil_waiters && ++c->pre_n >= XRC_PREEMPT_N) xrc_preempt_n(c); }
#ifdef XRC_NO_IC
#define XCALL(tgt) { u32 t_ = (tgt); r_esp -= 4; MX32(r_esp) = x_ra_; XRC_SPILL xrc_dispatch(c, t_); XRC_FILL }
#else
/* an inline cache per call site (an indirect call: a vtable, a function pointer, a callback): the last target and
 * its function; a hit calls it as XCALLD does (no search by address, no call-stack record), a miss goes through
 * xrc_dispatch_ic: the site's previous target (addr2: a polymorphic site's other class) is called the same way from
 * there, anything else goes through xrc_dispatch, then the cache is refilled (the old entry moving to addr2). imp: an
 * import's slot + 1 (xrc_imp_enter sets the current-import globals its handler reads). Guest code runs under the GIL,
 * so the updates are not torn. */
typedef struct { u32 addr, imp; XFn fn; u32 addr2, imp2; XFn fn2; } XIc;
void xrc_dispatch_ic(XCtx *c, u32 target, XIc *ic);
void xrc_imp_enter(u32 slot);
extern volatile int xrc_gil_waiters;
void xrc_preempt(void);
void xrc_bad_return(XCtx *c, u32 target);
#define XCALL(tgt) { static XIc ic_ = {0xffffffffu, 0, 0}; u32 t_ = (tgt); r_esp -= 4; MX32(r_esp) = x_ra_; XRC_SPILL \
	if (t_ == ic_.addr) { XRC_PREEMPT() if (ic_.imp) xrc_imp_enter(ic_.imp - 1u); XRC_DS_PUSH(t_) ic_.fn(c); XRC_DS_POP() \
		if (c->eip != x_ra_) xrc_bad_return(c, t_); } \
	else xrc_dispatch_ic(c, t_, &ic_); XRC_FILL }
#endif
#define XCALLX(tgt, kin_, kout_) { if ((kin_) >= 1) c->fa[0] = x_fa[0]; if ((kin_) >= 2) c->fa[1] = x_fa[1]; XCALL(tgt) if ((kout_) >= 1) x_fret = c->st0; }
/* the guest xmm registers a function uses are its locals X[k] (registers to the C compiler: nothing aliases them),
 * from c->x at its entry, back to c->x around calls and at its returns (XRC_XIN_ / XRC_XOUT_, defined per function) */
#define XRC_XDECL X128 X[8];
#define XRC_XENTRY XRC_XIN_
#define XRC_RETURNX(n) do { XRC_XOUT_ XRC_RETURN(n); } while (0)
#define XCALLM(tgt) { XRC_XOUT_ XCALL(tgt) XRC_XIN_ }
#define XCALLXM(tgt, kin_, kout_) { XRC_XOUT_ XCALLX(tgt, kin_, kout_) XRC_XIN_ }
/* a call whose target is known and translated (fn): called directly, without xrc_dispatch's search by address and
 * its call-stack record (crash reports and the profiler's guest column then show the nearest dispatched caller).
 * Still a point where a waiting guest thread gets the GIL, and the callee's return is checked. */
extern volatile int xrc_gil_waiters;
void xrc_preempt(void);
void xrc_bad_return(XCtx *c, u32 target);
/* XRC_DEEP_STACKS (the launcher's Debug & Profile Tools): direct calls are recorded in the thread's guest call stack
 * too (xrc_tstacks, xrc_rt.c), so stack dumps (XRC_WATCH, crashes, "blocked for 20 s") and the profiler's callers
 * show every frame, not only the dispatched ones -- two stores a call */
extern volatile u32 xrc_dwatch_on; /* XRC_DATA_WATCH (xrc_rt.c): a guest dword whose changes are reported */
void xrc_dwatch_check(XCtx *c);
#ifdef XRC_DEEP_STACKS
#define XRC_DEEP_MAX 4096
typedef struct XrcTStack { u32 calls[XRC_DEEP_MAX]; volatile int depth; u32 tid; unsigned long long n; XCtx *ctx; } XrcTStack;
extern XrcTStack xrc_tstacks[];
#define XRC_DS_PUSH(t_) { XrcTStack *ds_ = c->pad ? &xrc_tstacks[c->pad - 1u] : 0; if (xrc_dwatch_on) xrc_dwatch_check(c); \
	if (ds_) { if (ds_->depth < XRC_DEEP_MAX) ds_->calls[ds_->depth] = (t_); ds_->depth++; }
#define XRC_DS_POP() if (ds_) ds_->depth--; }
#else
#define XRC_DS_PUSH(t_) {
#define XRC_DS_POP() }
#endif
/* XRC_ABI_CHECK (an experiment's build: XRC_PAGE_DEFS): after each direct call, whether the callee kept the
 * registers the x86 ABI has it preserve (ebx esi edi ebp) -- each one that did not is reported once (xrc_rt.c) */
#ifdef XRC_ABI_CHECK
void xrc_abi_violation(u32 target, u32 ra, unsigned mask);
#define XRC_ABI_NOTE(tgt) { unsigned m_ = (c->ebx != r_ebx) | (c->esi != r_esi) << 1 | (c->edi != r_edi) << 2 | (c->ebp != r_ebp) << 3; \
	if (m_) xrc_abi_violation((tgt), x_ra_, m_); }
#else
#define XRC_ABI_NOTE(tgt)
#endif
#define XCALLD(tgt, fn) { r_esp -= 4; MX32(r_esp) = x_ra_; XRC_SPILL XRC_PREEMPT() XRC_DS_PUSH(tgt) fn(c); XRC_DS_POP() \
	if (c->eip != x_ra_) xrc_bad_return(c, (tgt)); XRC_ABI_NOTE(tgt) XRC_FILL }
/* the backward jump of a call-free loop with an atomic in it (xchg / lock / pause: a spin lock): the GIL to a waiting
 * thread, which may be the one holding the lock (without this the spinner keeps the GIL forever: a deadlock) */
void xrc_spin_yield(void);
#define XRC_SPIN_YIELD() { if (xrc_gil_waiters) xrc_spin_yield(); }
/* an inlined call (xrc_translator x_inline_site) is still a point where a waiting thread may get the GIL, as the call
 * was (a loop polling a flag through a getter that is now inlined would hold it for good otherwise); the registers
 * go to XCtx first, as at a call, only when it does */
#define XRC_INL_PREEMPT() { if (xrc_gil_waiters && ++c->pre_n >= XRC_PREEMPT_N) { XRC_SPILL xrc_preempt_n(c); } }
#define XCALLXD(tgt, kin_, kout_, fn) { if ((kin_) >= 1) c->fa[0] = x_fa[0]; if ((kin_) >= 2) c->fa[1] = x_fa[1]; XCALLD(tgt, fn) if ((kout_) >= 1) x_fret = c->st0; }
#define XCALLMD(tgt, fn) { XRC_XOUT_ XCALLD(tgt, fn) XRC_XIN_ }
#define XCALLXMD(tgt, kin_, kout_, fn) { XRC_XOUT_ XCALLXD(tgt, kin_, kout_, fn) XRC_XIN_ }
/* XCALLDK: a direct call whose callee keeps the callee-saved registers outside m_ (xrc x_abi_plan; bits 0 ebx,
 * 1 esi, 2 edi, 3 ebp: the ones it may change) -- only eax ecx edx esp and those are read back from XCtx, the caller's
 * locals hold the rest across the call (registers to the C compiler). m_ is a constant: the ifs fold away.
 * -DXRC_NO_ABI_KEEP: every register read back (as XCALLD); XRC_ABI_CHECK: reported when a callee broke it. */
#if defined(XRC_NO_ABI_KEEP) || defined(XRC_ABI_CHECK)
#define XRC_FILL_K(m_) XRC_FILL
#else
#define XRC_FILL_K(m_) { if (XRC_FM_ & 1) r_eax = c->eax; if (XRC_FM_ & 2) r_ecx = c->ecx; if (XRC_FM_ & 4) r_edx = c->edx; r_esp = c->esp; \
	if (((m_) & 1) && (XRC_FM_ & 8)) r_ebx = c->ebx; if (((m_) & 2) && (XRC_FM_ & 0x20)) r_esi = c->esi; \
	if (((m_) & 4) && (XRC_FM_ & 0x40)) r_edi = c->edi; if (((m_) & 8) && (XRC_FM_ & 0x10)) r_ebp = c->ebp; }
#endif
#ifdef XRC_ABI_CHECK
#define XRC_ABI_NOTE_K(tgt, m_) { unsigned m2_ = ((c->ebx != r_ebx) | (c->esi != r_esi) << 1 | (c->edi != r_edi) << 2 | (c->ebp != r_ebp) << 3) & ~(unsigned)(m_); \
	if (m2_) xrc_abi_violation((tgt), x_ra_, m2_); }
#else
#define XRC_ABI_NOTE_K(tgt, m_)
#endif
#define XCALLDK(tgt, fn, m_) { r_esp -= 4; MX32(r_esp) = x_ra_; XRC_SPILL XRC_PREEMPT() XRC_DS_PUSH(tgt) fn(c); XRC_DS_POP() \
	if (c->eip != x_ra_) xrc_bad_return(c, (tgt)); XRC_ABI_NOTE_K(tgt, m_) XRC_FILL_K(m_) }
#define XCALLXDK(tgt, kin_, kout_, fn, m_) { if ((kin_) >= 1) c->fa[0] = x_fa[0]; if ((kin_) >= 2) c->fa[1] = x_fa[1]; XCALLDK(tgt, fn, m_) if ((kout_) >= 1) x_fret = c->st0; }
#define XCALLMDK(tgt, fn, m_) { XRC_XOUT_ XCALLDK(tgt, fn, m_) XRC_XIN_ }
#define XCALLXMDK(tgt, kin_, kout_, fn, m_) { XRC_XOUT_ XCALLXDK(tgt, kin_, kout_, fn, m_) XRC_XIN_ }
#define XRC_FRET(v) (c->st0 = (double)(v))
/* x_ra_: the return address the next call pushes (set by the translation just before it: a constant, the original's) */
#define XRC_LOCALS u32 x_ra_; double x_fret, x_fa[2];
/* the guest x87 control word: c->fcw (per thread) */
#define XRC_FCW_INIT { x_fcw = c->fcw; }
#define XRC_FLDCW(v_) { x_fcw = c->fcw = (u16)(v_); }

#endif
#endif
