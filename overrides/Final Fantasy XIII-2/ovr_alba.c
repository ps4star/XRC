/* ovr_alba.c -- FINAL FANTASY XIII-2's hand-written routines (AlbaRecomp). The runtime links four tables by name
 * (xrc_rt.c: crt_math, crt_sort, game, simd).
 *
 * crt_math: the C runtime's x87 math entry points (VC9 libcmt, a later build than FF13's: same routines, other
 * addresses; found by their SSE2 dispatch and their descriptors' names). Their x87 internals keep values at depths
 * that differ by path, which xrc's static x87 model cannot express, so they are replaced at their entry points
 * (overrides.tsv: manual; the internals behind them: unreachable). Results in double with the host libm.
 *   __CI*      x87 arguments in c->fa[0] (st0) and c->fa[1] (st1), the result in c->st0; cdecl return.
 *              Two-argument forms take (st1, st0) as (x, y): pow(x, y), fmod(st1, st0).
 *   plain      double(s) on the stack (cdecl), the result in st0. Their x87 entries (the je target taken without
 *              SSE2) are the same function from the same stack.
 *   _ftol2     st0 -> edx:eax (truncated); _ftol2_sse -> eax. */
#include <math.h>
#include <stdio.h>
#include "xrc_rt.h"

#define FA0 (c->fa[0])
#define FA1 (c->fa[1])
#define CI1(name_, expr_) static void XRC_CC name_(XCtx *c) { double x = FA0; c->st0 = (expr_); xrc_cdecl_ret(c); }
#define CI2(name_, expr_) static void XRC_CC name_(XCtx *c) { double x = FA1, y = FA0; c->st0 = (expr_); xrc_cdecl_ret(c); }
static double stack_arg(XCtx *c, int k) { return f64_of(M64(c->esp + 4 + 8 * k)); }
#define ST1(name_, expr_) static void XRC_CC name_(XCtx *c) { double x = stack_arg(c, 0); c->st0 = (expr_); xrc_cdecl_ret(c); }
#define ST2(name_, expr_) static void XRC_CC name_(XCtx *c) { double x = stack_arg(c, 0), y = stack_arg(c, 1); c->st0 = (expr_); xrc_cdecl_ret(c); }

CI1(m_CIsin, sin(x))
CI1(m_CIcos, cos(x))
CI1(m_CItan, tan(x))
CI1(m_CIlog, log(x))
CI1(m_CIexp, exp(x))
CI1(m_CIacos, acos(x))
CI1(m_CIasin, asin(x))
CI2(m_CIpow, pow(x, y))
CI2(m_CIfmod, fmod(x, y))
ST1(m_sin, sin(x))
ST1(m_cos, cos(x))
ST1(m_tan, tan(x))
ST1(m_log, log(x))
ST1(m_exp, exp(x))
ST1(m_acos, acos(x))
ST1(m_asin, asin(x))
ST2(m_pow, pow(x, y))
ST2(m_fmod, fmod(x, y))

static void XRC_CC m_ftol2(XCtx *c) {
	u64 v = xrc_i64(xrc_rint(FA0, 3));
	c->eax = (u32)v;
	c->edx = (u32)(v >> 32);
	xrc_cdecl_ret(c);
}
static void XRC_CC m_ftol2_sse(XCtx *c) {
	c->eax = cvt_trunc(FA0);
	xrc_cdecl_ret(c);
}
/* __set_statfp_raise: raises the x87 exception flags it is given; the guest's FP exception flags are not modelled */
static void XRC_CC m_raise_fpexc(XCtx *c) { xrc_cdecl_ret(c); }

const XrcManual xrc_manual_crt_math[] = {
	{0x00ceb6b0u, m_acos, "acos"},
	{0x00ceb6f0u, m_CIacos, "__CIacos"},
	{0x00ceb73fu, m_acos, "acos (x87 entry)"},
	{0x00ceca30u, m_fmod, "fmod"},
	{0x00ceca3au, m_CIfmod, "__CIfmod"},
	{0x00cea080u, m_ftol2_sse, "_ftol2_sse"},
	{0x00cea0b6u, m_ftol2, "_ftol2"},
	{0x00cee420u, m_asin, "asin"},
	{0x00cee460u, m_CIasin, "__CIasin"},
	{0x00cee4afu, m_asin, "asin (x87 entry)"},
	{0x00cef8c0u, m_pow, "pow"},
	{0x00cef900u, m_CIpow, "__CIpow"},
	{0x00cef954u, m_pow, "pow (x87 entry)"},
	{0x00cf29d0u, m_exp, "exp"},
	{0x00cf2a0cu, m_CIexp, "__CIexp"},
	{0x00cf2a47u, m_exp, "exp (x87 entry)"},
	{0x00cf2a60u, m_log, "log"},
	{0x00cf2aa0u, m_CIlog, "__CIlog"},
	{0x00cf2aefu, m_log, "log (x87 entry)"},
	{0x00cf2bb0u, m_cos, "cos"},
	{0x00cf2bf0u, m_CIcos, "__CIcos"},
	{0x00cf2c3fu, m_cos, "cos (x87 entry)"},
	{0x00cf2ce0u, m_sin, "sin"},
	{0x00cf2d20u, m_CIsin, "__CIsin"},
	{0x00cf2d6fu, m_sin, "sin (x87 entry)"},
	{0x00cfc820u, m_tan, "tan"},
	{0x00cfc860u, m_CItan, "__CItan"},
	{0x00cfc8afu, m_tan, "tan (x87 entry)"},
	{0x00d038c0u, m_raise_fpexc, "__set_statfp_raise"},
};
const unsigned xrc_manual_crt_math_n = sizeof xrc_manual_crt_math / sizeof xrc_manual_crt_math[0];
/* qsort (00ced4f0, VC9 libcmt): the translation, wrapped -- XRC_QSORT_STATS=1 tallies the comparators (calls, elements)
 * and logs the top ones every 10 s, to find the sorts worth a native path (FF13: ovr_crt_sort.c) */
#include <time.h>
static struct { u32 cmp, width; unsigned long long calls, elems; } qs_tally[64];
static int qs_n;
/* (from FF13's ovr_crt_sort.c) */
#include <stdlib.h>
#include <string.h>
/* ascending u32s: an LSD radix sort (4 passes of 8 bits; a pass whose byte is the same everywhere is skipped) -- no
 * comparator calls, linear in n */
static void sort_u32(u32 *v, u32 n) {
	static u32 *tmp;
	static u32 cap;
	static u32 cnt[4][256];
	u32 *src = v, *dst, i, b;
	if (n < 2) return;
	/* (the table is often in order already, from the frame before: one pass that stops at the first inversion) */
	for (i = 1; i < n && v[i - 1] <= v[i]; i++) {}
	if (i == n) return;
	if (n > cap) { free(tmp); tmp = (u32 *)malloc((size_t)n * 4); cap = n; }
	dst = tmp;
	/* the four byte histograms in one read of the array */
	memset(cnt, 0, sizeof cnt);
	for (i = 0; i < n; i++) {
		u32 x = v[i];
		cnt[0][x & 0xff]++; cnt[1][x >> 8 & 0xff]++; cnt[2][x >> 16 & 0xff]++; cnt[3][x >> 24]++;
	}
	for (b = 0; b < 4; b++) {
		u32 sum = 0, *t, *c = cnt[b], sh = b * 8;
		if (c[(v[0] >> sh) & 0xff] == n) continue; /* (all the same byte: already in order for this pass) */
		for (i = 0; i < 256; i++) { u32 k = c[i]; c[i] = sum; sum += k; }
		for (i = 0; i < n; i++) { u32 x = src[i]; dst[c[(x >> sh) & 0xff]++] = x; }
		t = src; src = dst; dst = t;
	}
	if (src != v) memcpy(v, src, (size_t)n * 4);
}
static void XRC_CC m_qsort(XCtx *c) {
	static XFn orig;
	static int stats = -1;
	if (!orig) orig = xrc_translation_of(0x00ced4f0u);
	/* 00c32010: an unsigned compare of two u32s -- the ~32k-entry table the game sorts every frame (FF13: 00ec4440, the
	 * same sort): a total order whose equal elements are identical, so the radix sort gives the CRT's very array */
	if (M32(c->esp + 16) == 0x00c32010u && M32(c->esp + 12) == 4 && M32(c->esp + 8) < 0x10000000u) {
		sort_u32((u32 *)(xrc_mem + M32(c->esp + 4)), M32(c->esp + 8));
		xrc_cdecl_ret(c);
		return;
	}
	if (stats < 0) stats = xrc_tool_env("XRC_QSORT_STATS") != 0;
	if (stats) {
		u32 num = M32(c->esp + 8), width = M32(c->esp + 12), cmp = M32(c->esp + 16);
		static time_t last;
		int i;
		for (i = 0; i < qs_n && (qs_tally[i].cmp != cmp || qs_tally[i].width != width); i++) {}
		if (i == qs_n && qs_n < 64) { qs_tally[i].cmp = cmp; qs_tally[i].width = width; qs_n++; }
		if (i < 64) { qs_tally[i].calls++; qs_tally[i].elems += num; }
		if (time(0) - last >= 10) {
			last = time(0);
			for (i = 0; i < qs_n; i++)
				if (qs_tally[i].elems > 1000) fprintf(stderr, "xrc: qsort cmp %08x width %u: %llu sorts, %llu elements\n", qs_tally[i].cmp, qs_tally[i].width, qs_tally[i].calls, qs_tally[i].elems);
		}
	}
	orig(c);
}
const XrcManual xrc_manual_crt_sort[] = {
	{0x00ced4f0u, m_qsort, "qsort"},
};
const unsigned xrc_manual_crt_sort_n = sizeof xrc_manual_crt_sort / sizeof xrc_manual_crt_sort[0];
/* game: d4f000 (the MSSMIXER section: Miles' CPU check, pushfd / cpuid / push ds): the features it reports, bit 0
 * MMX, 1 3DNow!, 2 SSE, 3 SSE2 */
static void XRC_CC m_mss_cpu(XCtx *c) { c->eax = 0xd; xrc_cdecl_ret(c); }
/* game: the frame pacer (as FF13's, ovr_game.c there). A 1 ms timeSetEvent callback (cd7270: ReleaseSemaphore) wakes
 * the pacer thread (cd72f0), which calls the tick c027f0 (thiscall; the object at [05167208]): it counts fixed ticks of
 * 1/[this] s ([this] = 59.94) from [this + 0x48] (QPC) and, when [this + 0x3c] ticks are down and the main thread
 * waits (c027b0: [this + 0x40] == -1, then the event [this + 0x30]), releases it -- inlined here (c028d3): a time
 * stamp (us) into the ring [this + 8 + 8 * i] (i = [this + 0x28], 4 entries, spin lock [this + 0x2c]), [this + 0x34] =
 * 1 ("released"), SetEvent, then [this + 0x3c] = (int)([this] / [this + 4] + 1e-4) ticks for the next ([this + 4]: the
 * target rate). So frames snap to whole 1/59.94 s ticks: one of 20 ms takes 33.
 * XRC_FRAME_LIMIT=60 (or 30; build define -DXRC_FRAME_LIMIT, or the variable): the release once 1/limit s (or the
 * target's period, if longer) has passed since the last one, keeping the cadence while frames are on time, else at
 * once. 0: the game's own tick. */
static int frame_limit = -1;
static void pacer_release(XCtx *c, u32 s) {
	static u32 set_event;
	u32 i, arg;
	float a, b;
	unsigned long long us;
	if (!set_event) set_event = M32(0x00d500dcu); /* (the IAT: SetEvent) */
	while (__atomic_exchange_n((u32 *)(xrc_mem + s + 0x2c), 1u, __ATOMIC_ACQ_REL)) {}
	us = xrc_qpc() / 10; /* (QueryPerformanceFrequency is 10 MHz here: 100 ns units) */
	i = M32(s + 0x28) & 3;
	M32(s + 8 + i * 8) = (u32)us; M32(s + 12 + i * 8) = (u32)(us >> 32);
	M32(s + 0x28) = (i + 1) & 3;
	__atomic_store_n((u32 *)(xrc_mem + s + 0x2c), 0u, __ATOMIC_RELEASE);
	__atomic_store_n((u32 *)(xrc_mem + s + 0x34), 1u, __ATOMIC_RELEASE);
	arg = M32(s + 0x30);
	xrc_call_guest(c, set_event, 1, &arg);
	memcpy(&a, xrc_mem + s, 4); memcpy(&b, xrc_mem + s + 4, 4);
	M32(s + 0x3c) = b != 0.0f ? (u32)(int)(a / b + 9.9999997e-5f) : 1u;
	M32(s + 0x40) = 0;
}
/* the release when due (the game's guest code runs under the GIL: the tick thread and the main thread one at a time) */
static void pacer_try(XCtx *c, u32 s) {
	static unsigned long long last_rel;
	static unsigned long long unlock_at = 1; /* (1: not read yet; 0: never) */
	if (unlock_at == 1) {
		const char *e = xrc_tool_env("XRC_UNLOCKED");
		unlock_at = e ? xrc_now_us() + (unsigned long long)(atof(e) * 1e6) + 2 : 0;
	}
	if (M32(s + 0x40) == 0xffffffffu) {
		unsigned long long period = 10000000ull / (unsigned)frame_limit, now = xrc_qpc(), due;
		float fps;
		memcpy(&fps, xrc_mem + s + 4, 4);
		if (fps >= 1.0f && fps <= 1000.0f && (unsigned long long)(1e7 / fps) > period) period = (unsigned long long)(1e7 / fps);
		/* XRC_UNLOCKED=secs (tools builds; XRC_BENCH sets it): from secs on, no frame limit at all -- a benchmark of what a
		 * frame costs (as FF13's ovr_game.c) */
		if (unlock_at && xrc_now_us() >= unlock_at) period = 0;
		due = last_rel + period;
		if (!last_rel || now + 10000ull >= due) { /* (within the 1 ms tick) */
			last_rel = !last_rel || now > due + 10000ull ? now : due;
			pacer_release(c, s);
		}
	}
}
static void XRC_CC m_pacer_tick(XCtx *c) {
	static XFn orig;
	u32 s = c->ecx;
	if (frame_limit < 0) {
		const char *e = getenv("XRC_FRAME_LIMIT");
#ifdef XRC_FRAME_LIMIT
		frame_limit = XRC_FRAME_LIMIT;
#else
		frame_limit = 0;
#endif
		if (e) frame_limit = atoi(e);
		if (frame_limit < 0 || frame_limit > 1000) frame_limit = 0;
		fprintf(stderr, frame_limit ? "xrc: frame limiter: up to %d fps, not snapped to the game's 1/60 s ticks\n" : "xrc: frame limiter: the game's own (%d)\n", frame_limit);
	}
	if (!frame_limit) {
		if (!orig) orig = xrc_translation_of(0x00c027f0u);
		orig(c);
		return;
	}
	pacer_try(c, s);
	xrc_cdecl_ret(c);
}
/* c02660 (thiscall, the pacer; arg: u32 *out): the frame's step for the game's logic, in 1/300000 s -- the last two ring
 * stamps' difference (us; <= 0 or >= 1 s: the target's period) rounded to whole ticks of 300000 / [this] (5005: 1/59.94
 * s). With the game's pacer every frame is whole ticks anyway. With XRC_FRAME_LIMIT a frame of 20 ms stepped 1 tick
 * (16.7 ms): the game ran at ~80% speed. Here the rounding's remainder is carried to the next frame -- still whole
 * ticks each (as the engine expects), real time on average. (Once per frame: the same ring entry, the same answer.) */
static void XRC_CC m_frame_step(XCtx *c) {
	static XFn orig;
	static long long carry;
	static u32 seen_i = ~0u, seen_out;
	u32 s = c->ecx, out = M32(c->esp + 4), i, i1, i2;
	long long dt, units, tick, total, n;
	float rate, tr;
	if (frame_limit <= 0) {
		if (!orig) orig = xrc_translation_of(0x00c02660u);
		orig(c);
		return;
	}
	while (__atomic_exchange_n((u32 *)(xrc_mem + s + 0x2c), 1u, __ATOMIC_ACQ_REL)) {}
	i = M32(s + 0x28);
	i1 = (i - 1) & 3; i2 = (i - 2) & 3;
	dt = (long long)(int)(M32(s + 8 + i1 * 8) - M32(s + 8 + i2 * 8));
	__atomic_store_n((u32 *)(xrc_mem + s + 0x2c), 0u, __ATOMIC_RELEASE);
	if (i == seen_i) { M32(out) = seen_out; c->eax = out; xrc_stdcall_ret(c, 1); return; }
	memcpy(&tr, xrc_mem + s + 4, 4); memcpy(&rate, xrc_mem + s, 4);
	if (dt <= 0 || dt >= 1000000) dt = tr != 0.0f ? (long long)(1e6f / tr) : 16683;
	tick = rate != 0.0f ? (long long)(300000.0f / rate) : 5005;
	if (tick <= 0) tick = 5005;
	units = dt * 300000 / 1000000;
	total = units + carry;
	n = (total + tick / 2) / tick;
	if (n < 1) n = 1;
	carry = total - n * tick;
	if (carry > tick) carry = tick; /* (a hitch is not paid back over the frames after) */
	if (carry < -tick) carry = -tick;
	seen_i = i; seen_out = (u32)(n * tick);
	{ /* (XRC_STEP_LOG=1: the logic's time against real time, every 600 frames) */
		static int lg = -1; static long long su, sd, s0; static int sn;
		if (lg < 0) lg = getenv("XRC_STEP_LOG") != 0;
		if (lg) { su += n * tick; sd += dt; s0 += (units + tick / 2) / tick * tick; if (++sn == 600) { fprintf(stderr, "xrc: frame step: logic %.2f s in %.2f s real (%.1f%%; the game's rounding alone %.1f%%)\n", (double)su / 300000.0, (double)sd / 1e6, 100.0 * (double)su / 300000.0 / ((double)sd / 1e6), 100.0 * (double)s0 / 300000.0 / ((double)sd / 1e6)); su = sd = s0 = 0; sn = 0; } }
	}
	M32(out) = seen_out;
	c->eax = out; /* (returns the pointer: the callers read the step through it) */
	xrc_stdcall_ret(c, 1);
}
/* c027b0 (thiscall, the pacer): the main thread's frame wait -- c1f2d0 (this), c1af30 (this = 0516b400), c1cd10, then
 * [this + 0x40] = -1 ("waiting"), the idle callback [this + 0x38] (loading work while not released), the wait on the
 * event [this + 0x30], [this + 0x34] = 0. With XRC_FRAME_LIMIT a frame already due is released here, after the idle
 * callback, instead of at the pacer's next 1 ms tick. */
static void XRC_CC m_frame_wait(XCtx *c) {
	static XFn orig;
	static u32 wait_fn;
	u32 s = c->ecx, cb, args[2];
	if (frame_limit <= 0) {
		if (!orig) orig = xrc_translation_of(0x00c027b0u);
		orig(c);
		return;
	}
	if (!wait_fn) wait_fn = M32(0x00d500e8u); /* (the IAT: WaitForSingleObject) */
	c->ecx = s; xrc_call_guest(c, 0x00c1f2d0u, 0, 0);
	c->ecx = 0x0516b400u; xrc_call_guest(c, 0x00c1af30u, 0, 0);
	xrc_call_guest(c, 0x00c1cd10u, 0, 0);
	M32(s + 0x40) = 0xffffffffu;
	cb = M32(s + 0x38);
	if (cb) { c->ecx = s; xrc_call_guest(c, cb, 0, 0); }
	pacer_try(c, s);
	args[0] = M32(s + 0x30); args[1] = 0xffffffffu;
	xrc_call_guest(c, wait_fn, 2, args);
	__atomic_store_n((u32 *)(xrc_mem + s + 0x34), 0u, __ATOMIC_RELEASE);
	xrc_cdecl_ret(c);
}
const XrcManual xrc_manual_game[] = {
	{0x00d4f000u, m_mss_cpu, "MSS CPU features"},
	{0x00c027f0u, m_pacer_tick, "frame pacer tick (XRC_FRAME_LIMIT)"},
	{0x00c02660u, m_frame_step, "frame step (XRC_FRAME_LIMIT: remainder carried)"},
	{0x00c027b0u, m_frame_wait, "frame wait (XRC_FRAME_LIMIT: released at once if due)"},
};
const unsigned xrc_manual_game_n = sizeof xrc_manual_game / sizeof xrc_manual_game[0];

/* the frame statistics the runtime reports (FF13's ovr_game.c has them with its frame pacer; XIII-2's pacer is not
 * replaced yet: the renderer's own numbers only) */
unsigned long long xrc_frame_wait_us;
unsigned xrc_swap_gen;
unsigned xrc_swap_shown;
unsigned long long xrc_acquire_wait_us;
double xrc_gpu_busy_ms_total;
double xrc_rt_work_ms;
double xrc_gpu_ms_total = -1;
