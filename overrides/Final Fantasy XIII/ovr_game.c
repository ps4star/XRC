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
#ifdef XRC_FRAME_LIMIT
/* Vsync pacing (2026-11-24, XRC_FRAME_LIMIT=60): when the display alone paces the frames -- the swap chain presents
 * with FIFO (each frame waits for its vertical blank) and keeps one frame queued at most (present wait, or a swap chain
 * of 3 images at most: the acquire then waits until the frame before is on the screen, the moment present wait waits
 * for; vk_d3d9.c xrc_vk_vsync), into a window shown, on a display whose mode runs at 59-61 Hz (xrc_display_hz), with
 * the game's own target at 60 -- the limiter holds nothing: a frame waiting is released at once and the present's
 * vsync sets the pace. The limiter's own release timing (locked to the counted vertical blanks, on an exact schedule)
 * is not used then: under variable refresh (G-Sync, FreeSync: Windows' independent flip) the blanks follow the frames,
 * and a schedule made from their measured rate could drift under 60 -- a smooth ~57 fps for a second or two in battle
 * (PresentMon, 2026-11-24). Otherwise the limiter as before: no FIFO, a longer queue (more than 3 images without
 * present wait), 120 / 144 Hz modes, a 30 limit, the game's own target under 60. And frames coming faster than 66 a
 * second are not paced by vsync (the driver's "vertical sync: off" overrides FIFO): the limiter then, until a new swap
 * chain or the window shown again. XRC_VSYNC_PACE=0: the limiter always.
 * The release is made as the frame wait starts (its frame-time stamp, which steps the logic, then right after the
 * present that waited for the vertical blank: steady 1/60 s apart). The idle work (loading) had the time until the
 * release; now it gets the frame's slack instead: the period less the main thread's work a frame (from one frame wait
 * to the next, less its time blocked presenting) and 2 ms for the GPU (xo_00e8d5a0). */
static int vs_on;                  /* vsync pacing now (vsync_pace decides it at each release check) */
static unsigned long long vs_work; /* the main thread's work a frame, us (m_frame_wait: up by half the difference, down by
                                    * 1/8; a frame counts as at most a period + 2 ms, so one hitch does not starve the next ones) */
static int vs_in_wait;             /* the idle callback running in the frame wait: the budget applies (elsewhere the original) */
static int vs_frame;               /* this frame released by vsync pacing (presented since the last one): the budget applies */
#endif
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
#ifdef XRC_FRAME_LIMIT
	if (vs_on && vs_in_wait && vs_frame) { /* (vsync pacing: due once the frame's slack is used -- the release, made at the wait's start, says nothing.
	              * A frame that misses its refresh anyway -- the main thread's work alone a period or more: a slow machine --
	              * has XRC_MIN_IDLE_MS, so loading never starves; one that fits keeps its refresh: half a ms at least) */
		unsigned long long p = 1000000ull / XRC_FRAME_LIMIT, b = p > vs_work + 2000ull ? p - vs_work - 2000ull : 0;
		if (vs_work >= p) { if (b < min_us) b = min_us; }
		else if (b < 500ull) b = 500ull;
		c->eax = (c->eax & ~0xffu) | (now - idle_t0 >= b ? 1u : 0u);
		return;
	}
#endif
	if ((c->eax & 0xffu) && now - idle_t0 < min_us) c->eax = 0;
}

#ifdef XRC_FRAME_LIMIT
int xrc_vblank_start(void);
unsigned long long xrc_vblank_count(double *hz);
#if XRC_FRAME_LIMIT != 30 && XRC_FRAME_LIMIT != 60
#error "XRC_FRAME_LIMIT: 30 or 60"
#endif
/* vsync pacing (above): decided at each release check; 1: paced by vsync -- a frame waiting released at once */
static int vsync_pace(XCtx *c, u32 self, unsigned long long period, unsigned long long *last_rel) {
	static int env = -1, why_was = -1;
	static unsigned long long win_at, win_n;
	static int off; /* (vsync found not pacing: until a new swap chain or the window shown again) */
	static unsigned seen_gen = ~0u, seen_shown = ~0u, seen_pf;
	extern unsigned xrc_swap_gen, xrc_swap_shown;
	extern int xrc_vk_vsync, xrc_display_hz;
	unsigned long long now = xrc_qpc();
	int why = 0;
	if (env < 0) { const char *e = getenv("XRC_VSYNC_PACE"); env = !(e && e[0] == '0'); }
	if (seen_gen != xrc_swap_gen || seen_shown != xrc_swap_shown) { seen_gen = xrc_swap_gen; seen_shown = xrc_swap_shown; off = 0; win_at = 0; }
	if (!env) why = 1;
	else if (XRC_FRAME_LIMIT != 60) why = 2;
	else if (!xrc_vk_vsync) why = 3;
	else if (xrc_display_hz < 59 || xrc_display_hz > 61) why = 4;
	else if (period > 10000000ull / 58ull) why = 5;
	else if (off) why = 6;
	vs_on = !why;
	if (why != why_was) {
		why_was = why;
		if (!why) { extern int xrc_vk_present_wait; xrc_log("xrc: frame limiter: not holding frames -- the display's vsync paces them (FIFO, %s, %d Hz)\n", xrc_vk_present_wait ? "present wait" : "3 swap chain images", xrc_display_hz); }
		else xrc_log("xrc: frame limiter: on (%s)\n", why == 1 ? "XRC_VSYNC_PACE=0" : why == 2 ? "a 30 fps limit" :
		             why == 3 ? "no FIFO present with one frame queued at most, or the window not shown" : why == 4 ? "the display not at 60 Hz" :
		             why == 5 ? "the game's own rate under 60" : "vsync not pacing the frames");
	}
	if (!vs_on) return 0;
	if (M32(self + 0x40) == 0xffffffffu) {
		extern unsigned xrc_vk_vsync_frames;
		int presented = xrc_vk_vsync_frames != seen_pf;
		/* (no frame presented since the last release -- loading that draws nothing: no vsync to pace it, the period; its
		 * idle work then has the time until the release, as with the limiter) */
		if (!presented && *last_rel && now + 10000ull < *last_rel + period) { vs_frame = 0; return 1; }
		vs_frame = presented;
		if (presented) {
			seen_pf = xrc_vk_vsync_frames;
			/* (the rate, over half-second windows: over 66 a second, vsync is not pacing them) */
			if (!win_at) { win_at = now; win_n = 0; }
			else win_n++;
		}
		if (win_at && now - win_at >= 5000000ull) {
			double fps = (double)win_n * 1e7 / (double)(now - win_at);
			win_at = 0;
			if (fps > 66.0) {
				off = 1;
				xrc_log("xrc: frame limiter: %.0f fps with vsync -- it is not pacing the frames (a driver setting?): the limiter\n", fps);
			}
		}
		*last_rel = now;
		xrc_call_guest(c, 0x00e8d3c0u, 0, 0);
		M32(self + 0x40) = 0;
	}
	return 1;
}
static int tr_main; /* (try_release called by the main thread, from its frame wait: the diagnostic) */
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
	if (period && vsync_pace(c, self, period, &last_rel)) return;
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
#ifndef XRC_NO_TOOLS
			{ /* (XRC_HITCH_MS: the own clock's releases every 2 s -- the period, how late, how often the schedule restarted) */
				static int od = -1;
				static unsigned long long o_at, o_n, o_late, o_re, o_main;
				if (od < 0) od = getenv("XRC_HITCH_MS") != 0;
				if (od && last_rel) {
					o_n++; o_late += now > due ? now - due : 0; o_re += now > due + 10000ull; o_main += tr_main;
					if (!o_at) o_at = now;
					if (now - o_at > 20000000ull) {
						fprintf(stderr, "xrc: frame limiter: own clock: period %.3f ms, %llu releases, %.3f ms late on average, %llu restarts, %llu by the main thread\n",
						        (double)period / 1e4, o_n, (double)o_late / 1e4 / (double)o_n, o_re, o_main);
						o_at = now; o_n = o_late = o_re = o_main = 0;
					}
				}
			}
#endif
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
	static unsigned long long fw_end, fw_blk0; /* (the last frame wait's end; the time blocked presenting then) */
	extern unsigned long long xrc_present_block_us;
	if (fw_end && t0 > fw_end) { /* (vsync pacing: the main thread's work since the last frame wait, less its time blocked presenting) */
		unsigned long long w = t0 - fw_end, b = xrc_present_block_us - fw_blk0;
		w = w > b ? w - b : 0;
		if (w > 1000000ull / XRC_FRAME_LIMIT + 2000ull) w = 1000000ull / XRC_FRAME_LIMIT + 2000ull;
		vs_work = w >= vs_work ? vs_work + (w - vs_work + 1) / 2 : vs_work - (vs_work - w) / 8;
	}
	u32 self = c->ecx, cb = M32(self + 0x38);
	/* 00e8d560: the idle callback ([this+0x38]) once, then the wait on the release semaphore. The callback does its
	 * work (loading) only while the frame is not released yet -- released first (it was due already), it did nothing.
	 * A frame is due already at its wait whenever the frame took its time elsewhere: in fullscreen on the AMD driver
	 * the swap chain's queue stays full and Present takes ~12 ms, so the callback ran in only a few frames a second
	 * and a loading screen hung until windowed again. So: the callback first (the pacer's tick releases the frame
	 * meanwhile when due; xo_00e8d5a0 keeps it working its minimum), then the release, then the wait (without the
	 * callback a second time). */
	xrc_idle_begin();
	if (vs_on) { try_release(c, self); c->ecx = self; } /* (vsync pacing: released as the wait starts; the idle work has its budget) */
	vs_in_wait = 1;
	if (cb) {
		c->ecx = self;
		xrc_call_guest(c, cb, 0, 0);
		c->ecx = self;
	}
	vs_in_wait = 0;
	tr_main = 1;
	try_release(c, self);
	tr_main = 0;
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
	fw_end = xrc_now_us(); fw_blk0 = xrc_present_block_us;
}
#endif

/* Boosters (the launcher's Features & Mods > Boosters: XRC_BOOST=1 in the .cfg; every build). In the game F1..F4 turn
 * each one on and off -- the window's key events take those keys from the game then (vk_d3d9.c boost_key ->
 * xrc_boost_press, taken here once a frame) --, all off at the start. The start, and each change, shows the states at
 * the top left for a few seconds (xrc_osd_show). Each booster is transient: off, the game is as it was.
 *   F1  game speed 1x -> 2x -> 4x -> 8x -> 1x: the game logic's step x N. The main loop takes the last frame's time
 *       (00e8d490 at 00e8a3b3: 1/300000 s units, rounded to whole 1/60 s ticks), clamps it (the fps limits, a fixed step)
 *       and runs the update phases with it -- 00e8b910(UPDATE_BEGIN, UPDATE_END, &step) at 00e8a652, then the INTERVAL
 *       ones with the same step (the phase names: the table at 02667d98). x N there: each frame steps N frames' worth --
 *       battles, the field, scenes in the engine. (The sound manager's phase, 00e8a73d with 00e8d490's time from
 *       00e8a6f2, keeps the real time; the movies and the frame rate are their own.) The play time keeps the real time.
 *   F2  max stats: the party's battle max HP, Strength and Magic at their highest (99,999 / 9,999 / 9,999): the effective
 *       values of a battle character's parameters (004b8270, below), on the party's side. The battle's copy only: the
 *       Crystarium's stats and the Status menu are unchanged. Turned on or off in a battle, the party's characters
 *       there (bst_pc: their parameter sets, as 004b8270 recomputes them) are recomputed at once: the game's own values
 *       come back when off (buffs included -- only the result is changed), and the HP keep their share of the max HP.
 *   F3  max gil: 999,999,999 (the game's own cap) in [[0282b230] + 0x2578], held there each frame. The real gil is kept
 *       meanwhile (bst_gil_real): what the game adds through 00b6c620 (gil + delta, clamped) goes to it, what is spent
 *       is free, a value set (00b6c5c0, the scripts') is it; another value appearing there (a loaded save: 00b6e600) is
 *       taken as the real gil. Off: the real gil back.
 *       (A save made while on holds 999,999,999.)
 *   F4  one-hit kills: an enemy losing HP loses all of it: 0046e6b0 (a battle character's HP + delta, clamped to its
 *       floor +0x29c and its max) with delta < 0 on the enemy side (+0x270 bits 12..15: 1, 5 or 7, as 0046dcf0) takes
 *       -0x3fffffff -- the floor still holds (an enemy a scene needs alive).
 *   F8  max CP: every character's CP locked at 999,999 (the game's cap). The characters' records: [0282b230] + 0x8be0 +
 *       n * 0xf30, n < 32 (00b64be0; +0xc6c bit 0: the character is there), CP at +0xc80 (+0xc84 the CP earned in all,
 *       up to 999,999,999). Each one's real CP is kept meanwhile (bst_cp_real): CP earned (00b6ffd0, a character's CP +
 *       delta; 00b693e0 one, 00b69450 all) goes to it as the game would add it, CP spent (00b694e0, the Crystarium's:
 *       CP - cost) is free; any other value appearing there (a set: 00b69260, 00b73480; a loaded save; a new record,
 *       00b71660) is taken as the real CP. Off: each one's real CP back. (Off, the game keeps its CP itself. A save
 *       made while on holds 999,999.)
 * XRC_BOOST_SPEED=2|4|8 / _STATS=1 / _GIL=1 / _ONE_HIT=1 / _CP=1: the start states (tests; the launcher writes only
 * XRC_BOOST). Tools builds: XRC_BOOST_LOG=1 (each HP change, the parameter recomputes, the step each 600th, the CP
 * changes), XRC_BOOST_FROM=N (the game speed only from logic update N on), XRC_BOOST_KEYS=N:M,.. (at logic update N the
 * keys of mask M pressed: 1 F1, 2 F2, 4 F3, 8 F4, 16 F8 -- tests without a keyboard; 32: 100 CP given to all, 64: 500 gil added), XRC_BOOST_PROBE=1 (the party's
 * characters and what points to them) */
static int bst_ready, bst_keys, bst_speed = 1, bst_stats, bst_gil, bst_one_hit, bst_cp, bst_log, bst_probe, bst_rescale;
static unsigned bst_from, bst_nkey, bst_key_at[16], bst_key_mask[16];
static u32 bst_gil_real, bst_gil_have;
#define BST_CHR 32
#define BST_CP_MAX 999999u
static u32 bst_cp_real[BST_CHR], bst_cp_base; /* (F8: each character's real CP; the records' base they are of) */
static unsigned char bst_cp_have[BST_CHR];
#define BST_PC 24
static struct { u32 ps, ch, vt; } bst_pc[BST_PC]; /* (the party's battle characters seen: parameter set, character, vtable) */
static int bst_npc;
static void bst_read(void) {
	const char *e;
	if (bst_ready) return;
	bst_keys = xrc_boost_on();
	e = getenv("XRC_BOOST_SPEED");
	if (e && (atoi(e) == 2 || atoi(e) == 4 || atoi(e) == 8)) bst_speed = atoi(e);
	bst_stats = (e = getenv("XRC_BOOST_STATS")) != 0 && e[0] == '1';
	bst_gil = (e = getenv("XRC_BOOST_GIL")) != 0 && e[0] == '1';
	bst_one_hit = (e = getenv("XRC_BOOST_ONE_HIT")) != 0 && e[0] == '1';
	bst_cp = (e = getenv("XRC_BOOST_CP")) != 0 && e[0] == '1';
	bst_log = xrc_tool_env("XRC_BOOST_LOG") != 0;
	bst_probe = xrc_tool_env("XRC_BOOST_PROBE") != 0;
	if ((e = xrc_tool_env("XRC_BOOST_FROM")) != 0) bst_from = (unsigned)atoi(e);
	if ((e = xrc_tool_env("XRC_BOOST_KEYS")) != 0)
		while (*e && bst_nkey < 16) {
			char *end;
			bst_key_at[bst_nkey] = (unsigned)strtoul(e, &end, 10);
			if (*end != ':') break;
			bst_key_mask[bst_nkey++] = (unsigned)strtoul(end + 1, &end, 10);
			e = *end == ',' ? end + 1 : end;
		}
	if (bst_keys || bst_speed > 1 || bst_stats || bst_gil || bst_one_hit || bst_cp)
		xrc_log("xrc: boosters%s: game speed %d00%%%s%s%s%s\n", bst_keys ? " (F1-F4, F8)" : "", bst_speed, bst_stats ? ", max stats" : "",
		        bst_gil ? ", max gil" : "", bst_one_hit ? ", one-hit kills" : "", bst_cp ? ", max CP" : "");
	bst_ready = 1;
}
/* a battle character's side (+0x270 bits 12..15, signed) and party slot (bits 4..7, signed; -1: none), as the game takes them */
static int bst_side(u32 ch) { return (s32)(M32(ch + 0x270) << 16) >> 28; }
static int bst_slot(u32 ch) { return (s32)(M32(ch + 0x270) << 24) >> 28; }
static int bst_party(u32 ch) { int s = bst_side(ch); return s == 0 || s == 3; }
/* the states at the top left for ms (and in the log) */
static void bst_osd(int ms) {
	char l[6][32];
	const char *p[6];
	int k;
	snprintf(l[0], sizeof l[0], "BOOSTERS");
	snprintf(l[1], sizeof l[1], "%cF1 SPEED %dX", bst_speed > 1 ? '+' : '-', bst_speed);
	snprintf(l[2], sizeof l[2], "%cF2 MAX STATS %s", bst_stats ? '+' : '-', bst_stats ? "ON" : "OFF");
	snprintf(l[3], sizeof l[3], "%cF3 MAX GIL %s", bst_gil ? '+' : '-', bst_gil ? "ON" : "OFF");
	snprintf(l[4], sizeof l[4], "%cF4 ONE-HIT KILLS %s", bst_one_hit ? '+' : '-', bst_one_hit ? "ON" : "OFF");
	snprintf(l[5], sizeof l[5], "%cF8 MAX CP %s", bst_cp ? '+' : '-', bst_cp ? "ON" : "OFF");
	for (k = 0; k < 6; k++) p[k] = l[k];
	xrc_osd_show(p, 6, ms);
	xrc_log("xrc: boosters: speed %dx, max stats %s, max gil %s, one-hit kills %s, max CP %s\n", bst_speed, bst_stats ? "on" : "off",
	        bst_gil ? "on" : "off", bst_one_hit ? "on" : "off", bst_cp ? "on" : "off");
}
/* (tools, XRC_BOOST_PROBE: a new party character -- its vtable's first entries, and the globals in .data / .bss pointing
 * to it or to a block holding it near its start) */
static void bst_probe_pc(u32 ps, u32 ch) {
	u32 vt = M32(ch), a, v, o;
	xrc_log("xrc: boost probe: chara %08x (params %08x) vt %08x: %08x %08x %08x %08x %08x %08x, side %d slot %d\n", ch, ps, vt,
	        M32(vt), M32(vt + 4), M32(vt + 8), M32(vt + 12), M32(vt + 16), M32(vt + 20), bst_side(ch), bst_slot(ch));
	for (a = 0x0269e000u; a < 0x02872000u; a += 4) {
		v = M32(a);
		if (v == ch || v == ps) { xrc_log("xrc: boost probe:   [%08x] = %s\n", a, v == ch ? "chara" : "params"); continue; }
		if (v < 0x10000u || v - ch + 0x01000000u >= 0x02000000u) continue; /* (only a block within 16 MB of it) */
		for (o = 0; o < 0x200; o += 4) if (M32(v + o) == ch) { xrc_log("xrc: boost probe:   [%08x] = %08x, +%x = chara\n", a, v, o); break; }
	}
}
/* a party character's parameter set, as 004b8270 recomputes it: kept (once) for F2 in a battle */
static void bst_pc_add(u32 ps, u32 ch) {
	int k;
	for (k = 0; k < bst_npc; k++) if (bst_pc[k].ps == ps) { bst_pc[k].ch = ch; bst_pc[k].vt = M32(ch); return; }
	if (bst_npc == BST_PC) { memmove(&bst_pc[0], &bst_pc[1], sizeof bst_pc[0] * (BST_PC - 1)); bst_npc--; }
	bst_pc[bst_npc].ps = ps; bst_pc[bst_npc].ch = ch; bst_pc[bst_npc].vt = M32(ch);
	bst_npc++;
	if (bst_probe) bst_probe_pc(ps, ch);
}
/* one still a party character: its set still its own, the same class, on the party's side */
static int bst_pc_live(int k) { return M32(bst_pc[k].ps + 0x688) == bst_pc[k].ch && M32(bst_pc[k].ch) == bst_pc[k].vt && bst_party(bst_pc[k].ch); }
/* F2: the party's characters' max HP, Strength and Magic recomputed (by the game: 004b8270, and so bst_param), HP scaled */
static void bst_stats_apply(XCtx *c) {
	static const u32 idx[3] = {1, 5, 6};
	u32 eax = c->eax, ecx = c->ecx, edx = c->edx;
	int k, j, n = 0;
	bst_rescale = 1;
	for (k = 0; k < bst_npc; k++) {
		if (!bst_pc_live(k)) continue;
		for (j = 0; j < 3; j++) { c->ecx = bst_pc[k].ps; xrc_call_guest(c, 0x004b8270u, 1, &idx[j]); }
		n++;
	}
	bst_rescale = 0;
	c->eax = eax; c->ecx = ecx; c->edx = edx;
	if (bst_log || bst_probe) xrc_log("xrc: boost: max stats %s: %d of %d party characters recomputed\n", bst_stats ? "on" : "off", n, bst_npc);
}
/* F3 on / off: the real gil kept, or put back */
static void bst_gil_set(int on) {
	u32 b = M32(0x0282b230u);
	if (on) {
		bst_gil_have = 0;
		if (b) { bst_gil_real = M32(b + 0x2578); bst_gil_have = 1; M32(b + 0x2578) = 999999999u; }
	} else if (b && bst_gil_have)
		M32(b + 0x2578) = bst_gil_real;
	bst_gil = on;
	if (bst_log) xrc_log("xrc: boost: max gil %s: gil %u (real %u)\n", on ? "on" : "off", b ? M32(b + 0x2578) : 0u, bst_gil_real);
}
/* F8: character record n (0 if none, or not there) and the number of a record (-1: not one) */
static u32 bst_chr(u32 b, int n) { u32 r = b + 0x8be0u + (u32)n * 0xf30u; return b && (M32(r + 0xc6c) & 1u) ? r : 0; }
static int bst_chr_n(u32 b, u32 r) {
	u32 o = r - b - 0x8be0u;
	return b && o % 0xf30u == 0 && o / 0xf30u < BST_CHR ? (int)(o / 0xf30u) : -1;
}
/* max CP on: each character's CP the game set itself taken as its real CP (one not seen yet; or a value other than the
 * lock: a set, a loaded save, a new record), then locked at 999,999. Once a frame and after each change */
static void bst_cp_hold(void) {
	u32 b = M32(0x0282b230u), r, v;
	int n;
	if (!b) return;
	if (b != bst_cp_base) { memset(bst_cp_have, 0, sizeof bst_cp_have); bst_cp_base = b; }
	for (n = 0; n < BST_CHR; n++) {
		if (!(r = bst_chr(b, n))) { bst_cp_have[n] = 0; continue; }
		v = M32(r + 0xc80);
		if (!bst_cp_have[n] || v != BST_CP_MAX) {
			if (bst_log && (!bst_cp_have[n] || v != bst_cp_real[n])) xrc_log("xrc: boost: cp %d: real %u (the game's)\n", n, v);
			bst_cp_real[n] = v; bst_cp_have[n] = 1;
		}
		if (v != BST_CP_MAX) M32(r + 0xc80) = BST_CP_MAX;
	}
}
/* the lock back on every record (after the game changed CP: its real CP kept as it is) */
static void bst_cp_relock(void) {
	u32 b = M32(0x0282b230u), r;
	int n;
	if (b != bst_cp_base) { bst_cp_hold(); return; }
	for (n = 0; n < BST_CHR; n++) if ((r = bst_chr(b, n)) != 0 && bst_cp_have[n]) M32(r + 0xc80) = BST_CP_MAX;
}
/* F8 on / off: the real CP kept, or put back */
static void bst_cp_set(int on) {
	u32 b = M32(0x0282b230u), r;
	int n;
	if (on) {
		memset(bst_cp_have, 0, sizeof bst_cp_have);
		bst_cp_base = b;
		bst_cp = 1;
		bst_cp_hold();
	} else {
		if (b && b == bst_cp_base)
			for (n = 0; n < BST_CHR; n++)
				if ((r = bst_chr(b, n)) != 0 && bst_cp_have[n]) {
					if (bst_log) xrc_log("xrc: boost: cp %d: %u back\n", n, bst_cp_real[n]);
					M32(r + 0xc80) = bst_cp_real[n];
				}
		bst_cp = 0;
	}
}
/* 00b6ffd0 (thiscall (delta), ret 4: a character record): CP + delta (to 999,999), the CP earned + delta (to
 * 999,999,999). With max CP on: the real CP + delta as the game would, the lock kept */
static void XRC_CC bst_cp_add(XCtx *c) {
	static XFn orig;
	u32 r = c->ecx, b = M32(0x0282b230u);
	s32 d = (s32)M32(c->esp + 4);
	int n;
	if (!orig) orig = xrc_translation_of(0x00b6ffd0u);
	bst_read();
	if (!bst_cp || (n = bst_chr_n(b, r)) < 0) { orig(c); return; }
	if (b != bst_cp_base || !bst_cp_have[n]) bst_cp_hold();
	{
		s64 v = (s64)bst_cp_real[n] + d;
		bst_cp_real[n] = v < 0 ? 0u : v > BST_CP_MAX ? BST_CP_MAX : (u32)v;
		bst_cp_have[n] = 1;
	}
	if (bst_log) xrc_log("xrc: boost: cp %d: %+d -> real %u\n", n, d, bst_cp_real[n]);
	orig(c);
	M32(r + 0xc80) = BST_CP_MAX;
}
/* 00b694e0 (cdecl (character id, cost)): CP - cost (the Crystarium). With max CP on: free -- the real CP kept */
static void XRC_CC bst_cp_spend(XCtx *c) {
	static XFn orig;
	if (!orig) orig = xrc_translation_of(0x00b694e0u);
	bst_read();
	if (!bst_cp) { orig(c); return; }
	if (bst_log) xrc_log("xrc: boost: cp spent %d by character %d: free\n", (s32)M32(c->esp + 8), (s32)M32(c->esp + 4));
	orig(c);
	bst_cp_relock();
}
/* 00b6e600 (thiscall, a virtual: the party data read from a save): after it, the gil and CP there are the real ones --
 * taken at the next frame even when they are the boosters' caps */
static void XRC_CC bst_load(XCtx *c) {
	static XFn orig;
	u32 self = c->ecx;
	if (!orig) orig = xrc_translation_of(0x00b6e600u);
	orig(c);
	if (self != M32(0x0282b230u)) return; /* (only the game's own party data: not a copy read elsewhere) */
	bst_gil_have = 0;
	memset(bst_cp_have, 0, sizeof bst_cp_have);
	if (bst_log && (bst_gil || bst_cp)) xrc_log("xrc: boost: party data loaded: gil and CP taken as the real ones\n");
}
/* 00e8b910 (cdecl (first, last, &step)): the game's update phases first .. last with the step -- at the main loop's
 * logic update (the call at 00e8a652) the keys taken, the step x the game speed (the INTERVAL phases after it, 00e8a6af,
 * get the same step), the gil held (once a frame). The play time ([[0282b230] + 0x2580], the same units: it adds the
 * step) keeps the real time: a call that added the step x N to it gets the real step added instead. */
static void XRC_CC bst_update(XCtx *c) {
	static XFn orig;
	static s32 real;
	u32 ra = M32(c->esp), stp = M32(c->esp + 12), b, pt = 0;
	if (!orig) orig = xrc_translation_of(0x00e8b910u);
	bst_read();
	if (ra == 0x00e8a657u) {
		static unsigned nlog, nupd, nframe;
		int n;
		if (bst_keys) {
			unsigned p = __atomic_exchange_n(&xrc_boost_press, 0u, __ATOMIC_SEQ_CST), k;
			for (k = 0; k < bst_nkey; k++) if (bst_key_at[k] == nframe) p |= bst_key_mask[k];
			if (nframe++ == 0) bst_osd(6000);
			if (p & 1u) bst_speed = bst_speed >= 8 ? 1 : bst_speed * 2;
			if (p & 2u) { bst_stats = !bst_stats; bst_stats_apply(c); }
			if (p & 4u) bst_gil_set(!bst_gil);
			if (p & 8u) bst_one_hit = !bst_one_hit;
			if (p & 16u) bst_cp_set(!bst_cp);
			if (p & 31u) bst_osd(3000);
			if (p & 32u) { /* (tests, XRC_BOOST_KEYS mask 32: the game gives every character 100 CP -- 00b69450, as a battle's) */
				u32 a = 100, eax = c->eax, ecx = c->ecx, edx = c->edx;
				xrc_call_guest(c, 0x00b69450u, 1, &a);
				c->eax = eax; c->ecx = ecx; c->edx = edx;
				xrc_log("xrc: boost: test: 100 CP given to all\n");
			}
			if (p & 64u) { /* (tests, mask 64: the game adds 500 gil -- 00b6c620) */
				u32 a = 500, eax = c->eax, ecx = c->ecx, edx = c->edx, b0 = M32(0x0282b230u);
				xrc_call_guest(c, 0x00b6c620u, 1, &a);
				c->eax = eax; c->ecx = ecx; c->edx = edx;
				xrc_log("xrc: boost: test: 500 gil added (gil %u, real %u)\n", b0 ? M32(b0 + 0x2578) : 0u, bst_gil_real);
			}
		}
		if (bst_cp) bst_cp_hold();
		if (bst_probe && nframe % 600 == 0)
			for (n = 0; n < bst_npc; n++)
				xrc_log("xrc: boost probe: frame %u: chara %08x %s (vt %08x, set's chara %08x, side %d)\n", nframe, bst_pc[n].ch,
				        bst_pc_live(n) ? "live" : "GONE", M32(bst_pc[n].ch), M32(bst_pc[n].ps + 0x688), bst_side(bst_pc[n].ch));
		n = nupd < bst_from ? (nupd++, 1) : bst_speed;
		real = (s32)M32(stp);
		if (bst_log && (nlog++ % 600) == 0) xrc_log("xrc: boost: step %d x %d\n", real, n);
		if (n > 1) M32(stp) = (u32)(real * n);
		if (bst_gil && (b = M32(0x0282b230u)) != 0) {
			u32 g = M32(b + 0x2578);
			if (!bst_gil_have) { bst_gil_real = g; bst_gil_have = 1; }
			else if (g != 999999999u) bst_gil_real = g; /* (a value of the game's own: a loaded save) */
			if (g != 999999999u) M32(b + 0x2578) = 999999999u;
		}
	}
	if (bst_speed <= 1 || (ra != 0x00e8a657u && ra != 0x00e8a6b4u) || (s32)M32(stp) != real * bst_speed || !(b = M32(0x0282b230u))) {
		orig(c);
		return;
	}
	pt = M32(b + 0x2580);
	orig(c);
	if (M32(0x0282b230u) == b && M32(b + 0x2580) - pt == (u32)(real * bst_speed)) M32(b + 0x2580) = pt + (u32)real;
}
/* 00b6c620 (cdecl (delta)): gil + delta, clamped to 0 .. 999,999,999, returned. With max gil on the change goes to the
 * real gil; the game's stays at the cap */
static void XRC_CC bst_gil_add(XCtx *c) {
	static XFn orig;
	u32 b;
	s64 v;
	if (!orig) orig = xrc_translation_of(0x00b6c620u);
	bst_read();
	if (!bst_gil) { orig(c); return; }
	v = (s64)bst_gil_real + ((s32)M32(c->esp + 4) > 0 ? (s32)M32(c->esp + 4) : 0); /* (gil spent meanwhile is free) */
	bst_gil_real = v < 0 ? 0u : v > 999999999 ? 999999999u : (u32)v;
	bst_gil_have = 1;
	orig(c);
	if ((b = M32(0x0282b230u)) != 0) M32(b + 0x2578) = 999999999u;
	c->eax = 999999999u;
}
/* 00b6c5c0 (cdecl (value)): gil = value, clamped (the scripts' set). With max gil on the value is the real gil's */
static void XRC_CC bst_gil_put(XCtx *c) {
	static XFn orig;
	u32 b;
	s32 v = (s32)M32(c->esp + 4);
	if (!orig) orig = xrc_translation_of(0x00b6c5c0u);
	bst_read();
	if (!bst_gil) { orig(c); return; }
	bst_gil_real = v < 0 ? 0u : v > 999999999 ? 999999999u : (u32)v;
	bst_gil_have = 1;
	orig(c);
	if ((b = M32(0x0282b230u)) != 0) M32(b + 0x2578) = 999999999u;
}
/* 0046e6b0 (thiscall (delta), ret 4): a battle character's HP + delta (returns the change made) */
static void XRC_CC bst_add_hp(XCtx *c) {
	static XFn orig;
	u32 ch = c->ecx, d = M32(c->esp + 4);
	int side = bst_side(ch);
	if (!orig) orig = xrc_translation_of(0x0046e6b0u);
	bst_read();
	if (bst_one_hit && (s32)d < 0 && ((1u << (side & 31)) & 0xa2u)) M32(c->esp + 4) = 0xc0000001u;
	if (bst_log)
		xrc_log("xrc: boost: hp %08x side %d slot %d: %d %+d (max %d, floor %d)\n", ch, side, bst_slot(ch), (s32)M32(ch + 0x290),
		        (s32)M32(c->esp + 4), (s32)M32(ch + 0x6c8), (s32)M32(ch + 0x29c));
	orig(c);
}
/* 004b8270 (thiscall (idx), ret 4): a battle character's parameter set (chara + 0x2c4: pages of 64 dwords, page 4 the
 * effective values; its character at +0x688) recomputes parameter idx's effective value (004b84b0, to +0x400 + idx * 4)
 * -- each change of a base value or a modifier (004b52f0) does. After it, on the party's side (the character's +0x270
 * bits 12..15: 0 or 3, as 0046d8d0), with max stats: max HP (1) 99,999, Strength (5) and Magic (6) 9,999 (kept x 100
 * there: 999,900; their limits, 004b84b0's table 010a2950 by the type 010a2850[idx], are 1 .. 2^31 - 1). When the max HP
 * changes by that -- or by F2's recompute, either way --, the HP keep their share of it (rounded up: alive stays alive).
 * Each party character's set is kept for F2 (bst_pc). */
static void XRC_CC bst_param(XCtx *c) {
	static XFn orig;
	u32 ps = c->ecx, idx = M32(c->esp + 4), ch, was, now, maxv, game;
	int ours = 0;
	if (!orig) orig = xrc_translation_of(0x004b8270u);
	bst_read();
	if ((!bst_keys && !bst_stats && !bst_log) || (idx != 1 && idx != 5 && idx != 6) || !(ch = M32(ps + 0x688))) { orig(c); return; }
	was = M32(ps + 0x400 + idx * 4);
	orig(c);
	now = M32(ps + 0x400 + idx * 4);
	if (!bst_party(ch)) {
		if (bst_log) xrc_log("xrc: boost: param %u of %08x (side %d slot %d) = %d (was %d)\n", idx, ch, bst_side(ch), bst_slot(ch), (s32)now, (s32)was);
		return;
	}
	if (bst_keys) bst_pc_add(ps, ch);
	game = now;
	maxv = idx == 1 ? 99999u : 999900u; /* (Strength and Magic are kept x 100: 9,999 as the game shows them) */
	if (bst_stats && now < maxv) { M32(ps + 0x400 + idx * 4) = now = maxv; ours = 1; }
	if (idx == 1 && (ours || bst_rescale) && (s32)was > 0 && now != was) {
		u32 hp = M32(ch + 0x290);
		if ((s32)hp > 0) {
			unsigned long long v = ((unsigned long long)hp * now + was - 1) / was;
			M32(ch + 0x290) = (u32)(v < 1 ? 1 : v > now ? now : v);
		}
	}
	if (bst_log)
		xrc_log("xrc: boost: param %u of %08x (side %d slot %d) = %d (was %d, the game's %d; HP %d)\n", idx, ch, bst_side(ch), bst_slot(ch),
		        (s32)now, (s32)was, (s32)game, (s32)M32(ch + 0x290));
}
#define BOOSTERS                                                                                                       \
	{0x00e8b910u, bst_update, "update phases (Boosters: keys, game speed, max gil)"},                                   \
	{0x00b6c620u, bst_gil_add, "gil + delta (Boosters: max gil keeps the real gil)"},                                  \
	{0x00b6c5c0u, bst_gil_put, "gil = value (Boosters: max gil keeps the real gil)"},                                    \
	{0x00b6ffd0u, bst_cp_add, "a character's CP + delta (Boosters: max CP keeps the real CP)"},                           \
	{0x00b694e0u, bst_cp_spend, "CP spent in the Crystarium (Boosters: free with max CP)"},                               \
	{0x00b6e600u, bst_load, "party data from a save (Boosters: the real gil and CP)"},                                    \
	{0x0046e6b0u, bst_add_hp, "battle HP change (Boosters: one-hit kills)"},                                          \
	{0x004b8270u, bst_param, "battle parameter recompute (Boosters: max stats)"},

#ifdef XRC_FRAME_LIMIT
const XrcManual xrc_manual_game[] = {
	BOOSTERS
	{0x00e8d220u, m_pacer_tick, "frame pacer tick (XRC_FRAME_LIMIT)"},
	{0x00e8d560u, m_frame_wait, "frame wait (XRC_FRAME_LIMIT: released at once if due)"},
	{0x00e8d5a0u, xo_00e8d5a0, "frame due (fast: a minimum of idle work)"},
};
#else
const XrcManual xrc_manual_game[] = {BOOSTERS {0x00e8d5a0u, xo_00e8d5a0, "frame due (fast: a minimum of idle work)"}};
#endif
const unsigned xrc_manual_game_n = sizeof xrc_manual_game / sizeof xrc_manual_game[0];
