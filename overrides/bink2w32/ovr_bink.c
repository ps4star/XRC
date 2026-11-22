/* ovr_bink.c - hand-written replacements for bink2w32.dll functions the translator cannot do.
 *
 *   ftol64 (100033c0, and the jmp thunk to it at 100033f0): st0 -> edx:eax, truncated toward zero (the original sets
 *   the x87 rounding to chop, fistp qword, restores it). The x87 argument is c->fa[0]; cdecl return. */
#include "xrc_rt.h"

static void XRC_CC m_ftol64(XCtx *c) {
	u64 v = xrc_i64(xrc_rint(c->fa[0], 3));
	c->eax = (u32)v;
	c->edx = (u32)(v >> 32);
	xrc_cdecl_ret(c);
}

const XrcManual xrc_manual_bink[] = {
	{0x100033c0u, m_ftol64, "ftol64"},
	{0x100033f0u, m_ftol64, "ftol64 (thunk)"},
};
const unsigned xrc_manual_bink_n = sizeof xrc_manual_bink / sizeof xrc_manual_bink[0];
