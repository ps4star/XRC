/* ovr_game.c - hand-written replacements for functions of the game itself (compile-time options).
 *
 * XRC_FRAME_LIMIT=30 or 60 (build with XRC_DEFS=-DXRC_FRAME_LIMIT=60): the frame limiter's rate, without its snapping.
 * The game paces its frames like a vsync. A 1 ms timeSetEvent callback runs its pacer (00e8d220, on its own thread),
 * which counts fixed 1/60 s ticks ([this] = 60.0) and lets the waiting main thread go only on a tick boundary, once
 * round([this] / [this + 4]) ticks (the game's target rate) have passed since the last frame: at a 60 target a frame
 * that needs 17 ms takes 33 (fps snaps to 60, 30, 20 ...). The release (00e8d3c0) puts a QueryPerformanceCounter time
 * stamp into the frame-time ring and ups the semaphore the main thread waits on in 00e8d460 (flag [this + 0x40] ==
 * -1 while it waits).
 * With XRC_FRAME_LIMIT the pacer releases the waiting main thread once 1/XRC_FRAME_LIMIT s (or the game's own target
 * period, if longer) has passed since the last release, keeping that cadence while frames are on time:
 *   30  a steady 30 fps (a frame up to 33 ms takes 33; a longer one is not stretched to 50)
 *   60  up to 60 fps (a 20 ms frame takes 20 ms, not 33)
 * Everything else (the waits, the threads, the ring) is the game's. (The engine steps its logic by the last frame
 * time rounded to whole 1/60 s ticks, 00e8d490: a frame time between multiples of 1/60 s steps the nearest one.)
 *
 * The idle work (all builds): the frame wait first runs an idle callback ([this + 0x38]: 0040e880), which builds
 * resources -- effect nodes (00428c20) and the ExResourceManager's queue (00ccb570: in "checker mode", one item after
 * another while 00404770 says go on) -- until the frame is due: 00e8d5a0 ("the pacer has released the frame",
 * [pacer + 0x34]). The game presents without waiting (D3DPRESENT_INTERVAL_IMMEDIATE) and runs fast natively, so on a
 * PC the rest of each frame goes to that. Here a frame is often due already when its wait starts (slower code; a FIFO
 * present that waits for the vblank), and then each frame got one unit of idle work: loading a zone, whose effects
 * have thousands of instances to build, seemed to hang on its loading screen ("Effect NodeBuild wait", 00428da0).
 * xo_00e8d5a0 keeps "not due" for the first XRC_MIN_IDLE_MS (default 4; tools builds read the variable) of each
 * frame's idle work -- frames with nothing to build lose nothing: the callback stops when it has no work. And the
 * frame wait (m_frame_wait) runs the callback before releasing a frame that is due already: released first, it did
 * no work at all (fullscreen on the AMD driver, Present ~12 ms a frame: a loading screen hung until windowed). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xrc_rt.h"

/* 00e8d5a0 (cdecl, no arguments): is the frame due (bool) -- the original, unless this frame's idle work has not had
 * its minimum yet. The idle period: from the frame wait's start (xrc_idle_begin, XRC_FRAME_LIMIT builds), or else
 * from a call more than 2 ms after the last one. */
static unsigned long long idle_t0, idle_last;
unsigned long long xrc_frame_wait_us; /* the main thread's time in the frame wait (XRC_FRAME_LIMIT builds; the fps report) */
unsigned long long xrc_frame_block_us; /* of it, blocked on the release (the idle callback's loading work not counted:
                                       * Android's performance hint, xrc_android.c) */
unsigned xrc_swap_gen; /* swap chains made (vk_d3d9.c): the frame limiter empties a new one's queue */
unsigned xrc_swap_shown; /* the window shown again after being minimized (vk_d3d9.c): the queue watched a while */
unsigned long long xrc_acquire_wait_us; /* the main thread's time waiting for a swap chain image (vk_d3d9.c: the queue full) */
double xrc_gpu_busy_ms_total; /* the GPU busy (its command buffers' spans, overlaps once: vk_d3d9.c), ms so far */
double xrc_rt_work_ms;        /* the render thread replaying commands (vk_cs.h), ms so far */
double xrc_gpu_ms_total = -1; /* the GPU's time on the frames so far, in ms (the fps report); -1: not measured (vk_d3d9.c) */
static int idle_new;
void xrc_idle_begin(void) { idle_new = 1; }
void XRC_CC xo_00e8d5a0(XCtx *c) {
	static XFn orig;
	static unsigned long long min_us = ~0ull;
	unsigned long long now = xrc_now_us();
	if (!orig) orig = xrc_translation_of(0x00e8d5a0u);
	if (min_us == ~0ull) {
		const char *e = xrc_tool_env("XRC_MIN_IDLE_MS");
		min_us = e ? (unsigned long long)(atof(e) * 1000.0) : 4000ull;
	}
#ifndef XRC_NO_TOOLS
	static int idiag = -1;
	static unsigned long long id_us, id_calls, id_n, id_due0, id_at;
	if (idiag < 0) idiag = getenv("XRC_HITCH_MS") != 0;
#endif
	if (idle_new || now - idle_last > 2000ull) {
#ifndef XRC_NO_TOOLS
		if (idiag) { /* (XRC_HITCH_MS: the idle work each frame really had, every 2 s) */
			if (idle_t0) { id_us += idle_last - idle_t0; id_n++; }
			if (!id_at) id_at = now;
			if (now - id_at > 2000000ull && id_n) {
				fprintf(stderr, "xrc: idle work: %.2f ms a frame in %.1f polls (%llu frames with idle work; %.0f%% of them due at the first poll)\n",
				        (double)id_us / 1000.0 / (double)id_n, (double)id_calls / (double)id_n, id_n, 100.0 * (double)id_due0 / (double)id_n);
				id_us = id_calls = id_n = id_due0 = 0; id_at = now;
			}
		}
#endif
		idle_t0 = now; idle_new = 0;
	}
	idle_last = now;
	orig(c);
#ifndef XRC_NO_TOOLS
	if (idiag) { id_calls++; if (now == idle_t0 && (c->eax & 0xffu)) id_due0++; }
#endif
	if ((c->eax & 0xffu) && now - idle_t0 < min_us) c->eax = 0;
}

#ifdef XRC_FRAME_LIMIT
int xrc_vblank_start(void);
unsigned long long xrc_vblank_count(double *hz);
#if XRC_FRAME_LIMIT != 30 && XRC_FRAME_LIMIT != 60
#error "XRC_FRAME_LIMIT: 30 or 60"
#endif
/* 00e8d220 (thiscall, no arguments): the pacer's tick */
/* the release when due: from the pacer's 1 ms tick, and from the main thread as it starts to wait (00e8d560), so a
 * frame that is already late goes on at once instead of at the next tick (~1 ms a frame otherwise) */
static void try_release(XCtx *c, u32 self) {
	static int said;
	static unsigned long long last_rel;
	static unsigned long long last_due; /* (locked: when the last frame was due on the exact 1/rate schedule) */
	float fps;
	unsigned long long period = 10000000ull / XRC_FRAME_LIMIT, now;
	static unsigned long long unlock_at = 1; /* (1: not read yet; 0: never) */
	if (unlock_at == 1) {
		const char *e = xrc_tool_env("XRC_UNLOCKED");
		unlock_at = e ? xrc_now_us() + (unsigned long long)(atof(e) * 1e6) + 2 : 0;
	}
	memcpy(&fps, xrc_mem + self + 4, 4);
	if (fps >= 1.0f && fps <= 1000.0f && (unsigned long long)(1e7 / fps) > period) period = (unsigned long long)(1e7 / fps);
	/* XRC_UNLOCKED=secs (tools builds): from secs on, no frame limit at all (and an immediate present, vk_d3d9.c) -- a
	 * benchmark of what a frame costs (the game's logic steps by whole 1/60 s ticks: it may then run fast or slow) */
	if (unlock_at && xrc_now_us() >= unlock_at) period = 0;
	if (!said) { said = 1; fprintf(stderr, "xrc: XRC_FRAME_LIMIT: frames at up to %d fps, not snapped\n", XRC_FRAME_LIMIT); }
	/* locked to the display's refresh (vblank.c), when it runs at a whole multiple of the rate (60 Hz at 60, 120 at
	 * 60 ...: every k-th vertical blank): each frame released at a vertical blank -- one clock, not two drifting
	 * apart. Else (144 Hz, not measured yet, XRC_VSYNC_LOCK=0) the limiter's own clock. */
	if (period && xrc_vblank_start()) {
		static unsigned long long last_vb;
		static int was_locked = -1, was_lock_n;
		double hz, r;
		unsigned long long vb = xrc_vblank_count(&hz);
		int k = 0;
		r = hz * (double)period / 1e7; /* (refreshes per frame) */
		if (hz > 0) {
			k = (int)(r + 0.5);
			if (k < 1 || r < k * 0.97 || r > k * 1.03) k = 0;
		}
		if ((k != 0) != was_locked) {
			was_locked = k != 0;
			if (k) was_lock_n++;
			if (k) fprintf(stderr, "xrc: frame limiter: locked to the display's refresh (%.2f Hz, every %d)\n", hz, k);
			else if (hz > 0) fprintf(stderr, "xrc: frame limiter: its own clock (the display's %.2f Hz is not a multiple of the rate)\n", hz);
		}
		if (k) {
			/* The swap chain's queue: when the limiter locks, it may be full -- filled on the limiter's own clock before
			 * (a little faster than the display's), and at one frame a refresh it stays full: every frame on screen a
			 * refresh late (input lag at a steady 60). One refresh more, once, empties it: on the first locked release,
			 * after a new swap chain (a resize, another window: xrc_swap_gen), and on the frame after a stall (over 100
			 * ms from release to release; one every 2 s at most, so loading does not slow down). (A stall of the main
			 * thread empties the queue itself -- nothing presented while the display goes on -- but one of the GPU fills
			 * it: the frames presented meanwhile all become ready at once. Which one it was is not known here; the
			 * frame right after a hitch is where a 33 ms one shows least.) A refresh more: the release at least k + 1
			 * refreshes after the last one -- a frame already that late is not held. Never otherwise: waits for a swap chain image do
			 * not tell a full queue from frames a little longer than a refresh (CPU + GPU: each frame shown a refresh
			 * later, every image busy) -- a refresh more for those was a 33 ms frame every ~10 (the field's hitches). */
			static unsigned long long drained_at, lock_gen = ~0ull;
			static int after_stall, after_shown, aw_n;
			static unsigned seen_shown;
			static unsigned long long shown_at, aw_last, owed_at;
			static int owed, free_n;
			extern unsigned xrc_swap_shown;
			static unsigned seen_swap = ~0u;
			static int diag = -1;
			extern unsigned xrc_swap_gen;
			unsigned long long now_q = xrc_qpc(), need = (unsigned long long)k;
			int waiting = M32(self + 0x40) == 0xffffffffu, why = 0;
			if (diag < 0) diag = getenv("XRC_HITCH_MS") != 0;
			static unsigned long long lock_seen = ~0ull;
			if (lock_seen != (unsigned long long)was_lock_n) { lock_seen = (unsigned long long)was_lock_n; last_vb = vb; } /* (counted from the lock) */
			if (lock_gen != (unsigned long long)was_lock_n) why = 1;
			else if (seen_swap != xrc_swap_gen) why = 2;
			else if (after_stall) why = 3;
			else if (after_shown) why = 4;
			if (why) need++;
			/* A queue that stays full: each frame waits for an image, the frame wait ~0, loading starves. It fills when
			 * the display shows nothing for a moment -- shown again after being minimized (Windows' restore), or the
			 * window behind others (Windows then takes its frames when it likes) -- and, without present wait, when a
			 * frame's GPU work misses its refresh (that one repeated on the screen, the queue a frame deeper after).
			 * Full: 3 releases in a row whose frame waited over 8 ms for an image (over 2 ms for 3 s after being shown
			 * again). (A frame of CPU + GPU a little over a refresh waits only a few ms: not counted -- the misfire of
			 * an earlier, 2 ms rule.)
			 *   - In the 3 s after being shown again (or a new swap chain): a refresh more at once (repeated while it stays full, every 0.5 s
			 *     at most).
			 *   - Else the refresh more is owed, not spent: a frame repeated on the screen for it would be a hitch of
			 *     its own. The next frame that is late anyway (released over a refresh after the one before) empties
			 *     the queue with nothing repeated -- the frame waiting in it is shown meanwhile -- and the debt is
			 *     gone; so is it when the frames stop waiting for images (3 releases waiting under 2 ms). Owed for
			 *     3 s: spent then (mostly a loading screen, where a repeated frame does not show and a full queue
			 *     starves the loading).
			 *   - "Shown again" is also a new swap chain (the window resized, fullscreen on or off). */
			static unsigned seen_gen = ~0u;
			/* (shown again, or a new swap chain -- the window resized, fullscreen on or off: Windows shows nothing of it
			 * for a moment either way; the 3 s after: a full queue drained at once) */
			if (seen_shown != xrc_swap_shown || seen_gen != xrc_swap_gen) {
				seen_shown = xrc_swap_shown; seen_gen = xrc_swap_gen;
				shown_at = now_q; aw_n = 0; aw_last = xrc_acquire_wait_us;
			}
			/* (and never sooner than a refresh period minus 2 ms after the last release, k refreshes a frame: a vertical
			 * blank counted twice must not release two frames in one refresh -- the queue then fills for good) */
			/* (and never before its time on the exact schedule, 1/rate apart -- variable refresh (FreeSync, G-Sync: the
			 * window focused, Windows' independent flip) refreshes the display when a frame arrives, so the vertical blanks
			 * follow the frames: locked to them alone, the frames ran ahead, the rate measured from those blanks rose with
			 * them until it fell out of the lock's 3% (2026-11-23, PresentMon: 57-66 fps in a 4 s cycle, smooth unfocused).
			 * With a fixed refresh the blanks come at that time anyway; on time, the schedule is kept (a frame let go up to
			 * 1 ms early by the tick does not move it), late by over 2 ms it starts again from then) */
			/* (the schedule follows the measured refresh -- a fixed display a little over the rate keeps its own pace --
			 * but never over 0.2% faster than the rate: under variable refresh that measure follows the frames) */
			double hz_cap = (double)k * 1e7 / (double)period * 1.002;
			unsigned long long sched = (unsigned long long)(1e7 * (double)need / (hz < hz_cap ? hz : hz_cap));
			if (waiting && vb >= last_vb + need && (!last_rel || now_q - last_rel + 20000ull >= (unsigned long long)(1e7 * (double)need / hz)) &&
			    (!last_due || now_q + 10000ull >= last_due + sched)) {
				last_due = !last_due || now_q > last_due + sched + 20000ull ? now_q : last_due + sched;
				after_stall = !why && last_rel && now_q - last_rel > 1000000ull && now_q - drained_at > 20000000ull;
				{
					unsigned long long aw = xrc_acquire_wait_us - aw_last;
					int late = vb > last_vb + need; /* (this frame over a refresh after the last: late anyway) */
					aw_last = xrc_acquire_wait_us;
					int fresh = shown_at && now_q - shown_at < 30000000ull;
					after_shown = 0;
					aw_n = aw > (fresh ? 2000ull : 8000ull) ? aw_n + 1 : 0;
					if (aw_n >= 3 && !why) {
						aw_n = 0;
						if (fresh) { if (now_q - drained_at > 5000000ull || !drained_at) { after_shown = 1; owed = 0; } }
						else if (!owed) { owed = 1; owed_at = now_q; free_n = 0; if (diag) fprintf(stderr, "xrc: frame limiter: at %llu us, the queue full: a refresh owed\n", xrc_now_us()); }
					} else if (owed && !why) {
						free_n = aw < 2000ull ? free_n + 1 : 0;
						if (late || free_n >= 3) {
							owed = 0;
							if (diag) fprintf(stderr, "xrc: frame limiter: at %llu us, the owed refresh not needed (%s)\n", xrc_now_us(), late ? "a late frame emptied the queue" : "no longer waiting for images");
						} else if (now_q - owed_at > 30000000ull) { owed = 0; after_shown = 1; }
					}
				}
				if (why) {
					after_stall = 0;
					if (diag) fprintf(stderr, "xrc: frame limiter: at %llu us, a refresh more (%s)\n", xrc_now_us(), why == 1 ? "locked to the refresh" : why == 2 ? "a new swap chain" : why == 3 ? "after a stall" : "the queue full");
					lock_gen = (unsigned long long)was_lock_n;
					seen_swap = xrc_swap_gen;
					drained_at = now_q;
				}
				last_vb = vb;
				last_rel = xrc_qpc();
				xrc_call_guest(c, 0x00e8d3c0u, 0, 0);
				M32(self + 0x40) = 0;
			}
			return;
		}
	}
	if (M32(self + 0x40) == 0xffffffffu) {
		unsigned long long due = last_rel + period;
		now = xrc_qpc();
		if (!last_rel || now + 10000ull >= due) {
			last_rel = !last_rel || now > due + 10000ull ? now : due;
			xrc_call_guest(c, 0x00e8d3c0u, 0, 0);
			M32(self + 0x40) = 0;
		}
	}
}
static void XRC_CC m_pacer_tick(XCtx *c) {
	try_release(c, c->ecx);
	xrc_cdecl_ret(c);
}
/* the main thread's frame wait (this in ecx): released first if due, then the game's wait (on the semaphore) */
static void XRC_CC m_frame_wait(XCtx *c) {
	static XFn orig;
	if (!orig) orig = xrc_translation_of(0x00e8d560u);
	if (!orig) xrc_fatal(c, "XRC_FRAME_LIMIT: no translation of 00e8d560");
	unsigned long long t0 = xrc_now_us();
	u32 self = c->ecx, cb = M32(self + 0x38);
	/* 00e8d560: the idle callback ([this+0x38]) once, then the wait on the release semaphore. The callback does its
	 * work (loading) only while the frame is not released yet -- released first (it was due already), it did nothing.
	 * A frame is due already at its wait whenever the frame took its time elsewhere: in fullscreen on the AMD driver
	 * the swap chain's queue stays full and Present takes ~12 ms, so the callback ran in only a few frames a second
	 * and a loading screen hung until windowed again. So: the callback first (the pacer's tick releases the frame
	 * meanwhile when due; xo_00e8d5a0 keeps it working its minimum), then the release, then the wait (without the
	 * callback a second time). */
	xrc_idle_begin();
	if (cb) {
		c->ecx = self;
		xrc_call_guest(c, cb, 0, 0);
		c->ecx = self;
	}
	try_release(c, self);
	if (cb) M32(self + 0x38) = 0;
	c->ecx = self;
	{
		unsigned long long tb = xrc_now_us();
		int tsp = xrc_ts_push(XS_LIMIT);
		orig(c); /* (the wait alone now: the frame report's limiter time) */
		xrc_ts_pop(tsp);
		xrc_frame_block_us += xrc_now_us() - tb;
	}
	if (cb && !M32(self + 0x38)) M32(self + 0x38) = cb;
	xrc_frame_wait_us += xrc_now_us() - t0;
}
const XrcManual xrc_manual_game[] = {
	{0x00e8d220u, m_pacer_tick, "frame pacer tick (XRC_FRAME_LIMIT)"},
	{0x00e8d560u, m_frame_wait, "frame wait (XRC_FRAME_LIMIT: released at once if due)"},
	{0x00e8d5a0u, xo_00e8d5a0, "frame due (fast: a minimum of idle work)"},
};
const unsigned xrc_manual_game_n = 3;
#else
const XrcManual xrc_manual_game[] = {{0x00e8d5a0u, xo_00e8d5a0, "frame due (fast: a minimum of idle work)"}};
const unsigned xrc_manual_game_n = 1;
#endif
