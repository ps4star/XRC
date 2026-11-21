/* Guest functions run without the GIL: the translation itself, unchanged, but with the GIL released around it, so the
 * thread that calls it runs at the same time as the others (as on Windows, where every thread has a core) instead of
 * only while they wait.
 *
 * Only for a function that is a leaf in every sense: it calls nothing (no other guest function, no import, nothing of
 * the runtime: its translation has no call at all), and the memory it reads and writes is its caller's alone while it
 * runs (the guest stack of its thread, buffers no other thread touches until the caller says so) -- so nothing it does
 * can race with what the other threads do under the GIL. Reads of read-only tables are fine.
 *
 *   00c77820  zlib's inflate_fast (the game's statically linked zlib 1.1: the bit buffer, inflate_mask at 026df450,
 *             the length / distance tables): the decompression of the streamed resources, on the game's loader thread.
 *             A profile of the opening battle (2026-11-19): 80% of that thread's time, and 74% of it inside the frames
 *             over 20 ms -- the main thread waiting for a resource while the loader could run only in its waits (one
 *             frame: 12 ms of the main thread's work, then 50 ms of waiting for the loader). Its translation calls
 *             nothing (checked in build/xrc_gen/src/page_00c7.c).
 *
 * XRC_LOCKED_INFLATE=1: as before (the GIL kept), to compare. */
#include <stdlib.h>
#include "xrc.h"
#include "xrc_rt.h"

void XRC_CC xo_00c77820(XCtx *c) {
	static XFn orig;
	static int locked = -1;
	if (!orig) orig = xrc_translation_of(0x00c77820u);
	if (locked < 0) locked = getenv("XRC_LOCKED_INFLATE") != 0;
	if (locked || !xrc_gil_held()) { orig(c); return; }
	xrc_gil_release();
	orig(c);
	xrc_gil_acquire();
}
