/* ovr_crt_math.c - hand-written replacements for the CRT's floating-point entry points (VC9 libcmt).
 *
 * The x87 halves of the CRT math library (__CIsin & co. and their dispatchers __cintrindisp* / __ctrandisp* /
 * __trandisp*, __fload, _ftol2 ...) keep values on the x87 stack at depths that differ by path and pass them between
 * helpers, which the translator's static stack model cannot express. They are replaced here at their entry points,
 * which makes the internals behind them unreachable (overrides.tsv lists them as unreachable, with why, and xrc
 * reads it for its coverage report). Results are computed in
 * double with the host libm: functionally equivalent, not bit-identical to the x87 (80-bit) originals.
 *
 * Calling conventions, as the translated callers use them (src/include/xrc.h XCALLX):
 *   __CI*   x87 arguments in c->fa[0] (st0) and c->fa[1] (st1), the result in c->st0; cdecl return.
 *           Two-argument forms take (st1, st0) as (x, y): pow(x, y), atan2(y = st1, x = st0), fmod(st1, st0).
 *   _ftol2  st0 -> edx:eax (truncated); _ftol2_sse -> eax (cvttsd2si: the SSE2 path the original takes).
 *   __libm_sse2_sin / cos: xmm0 -> xmm0. */
#include <math.h>
#include "xrc_rt.h"

#define FA0 (c->fa[0])
#define FA1 (c->fa[1])
#define CI1(name_, expr_) static void XRC_CC name_(XCtx *c) { double x = FA0; c->st0 = (expr_); xrc_cdecl_ret(c); }
#define CI2(name_, expr_) static void XRC_CC name_(XCtx *c) { double x = FA1, y = FA0; c->st0 = (expr_); xrc_cdecl_ret(c); }

CI1(m_CIsin, sin(x))
CI1(m_CIcos, cos(x))
CI1(m_CItan, tan(x))
CI1(m_CIsqrt, sqrt(x))
CI1(m_CIlog, log(x))
CI1(m_CIexp, exp(x))
CI1(m_CIatan, atan(x))
CI1(m_CIacos, acos(x))
CI2(m_CIpow, pow(x, y))
CI2(m_CIatan2, atan2(x, y))
CI2(m_CIfmod, fmod(x, y))

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
/* sin / cos with the double on the stack (cdecl), the result in st0 */
static double stack_arg(XCtx *c) { return f64_of(M64(c->esp + 4)); }
static void XRC_CC m_cos(XCtx *c) { c->st0 = cos(stack_arg(c)); xrc_cdecl_ret(c); }
static void XRC_CC m_sin(XCtx *c) { c->st0 = sin(stack_arg(c)); xrc_cdecl_ret(c); }
static void XRC_CC m_tan(XCtx *c) { c->st0 = tan(stack_arg(c)); xrc_cdecl_ret(c); }
static void XRC_CC m_sse2_sin(XCtx *c) { c->x[0].d[0] = sin(c->x[0].d[0]); xrc_cdecl_ret(c); }
static void XRC_CC m_sse2_cos(XCtx *c) { c->x[0].d[0] = cos(c->x[0].d[0]); xrc_cdecl_ret(c); }
/* __handle_exc's helper that raises the x87 exception flags it is given (fld / fistp of crafted values): the guest's
 * floating-point exception flags are not modelled */
static void XRC_CC m_raise_fpexc(XCtx *c) { xrc_cdecl_ret(c); }

const XrcManual xrc_manual_crt_math[] = {
	{0x0100eed4u, m_sse2_sin, "__libm_sse2_sin"},
	{0x0100f5a2u, m_sse2_cos, "__libm_sse2_cos"},
	{0x0100f75au, m_CIatan2, "__CIatan2"},
	{0x0100f770u, m_cos, "cos"},
	{0x0100f7b0u, m_CIcos, "__CIcos"},
	{0x0100f7ffu, m_cos, "cos (x87 entry)"},
	{0x0100f8a0u, m_sin, "sin"},
	{0x0100f8e0u, m_CIsin, "__CIsin"},
	{0x0100f92fu, m_sin, "sin (x87 entry)"},
	{0x0100fd40u, m_ftol2_sse, "_ftol2_sse"},
	{0x0100fd76u, m_ftol2, "_ftol2"},
	{0x010108a0u, m_CIsqrt, "__CIsqrt"},
	{0x01012f2au, m_CIfmod, "__CIfmod"},
	{0x01013b90u, m_CIacos, "__CIacos"},
	{0x01013df0u, m_CIpow, "__CIpow"},
	{0x010153b0u, m_CIlog, "__CIlog"},
	{0x010154fcu, m_CIexp, "__CIexp"},
	{0x01015590u, m_CIatan, "__CIatan"},
	{0x0101f110u, m_tan, "tan"},
	{0x0101f150u, m_CItan, "__CItan"},
	{0x01026facu, m_raise_fpexc, "__set_statfp_raise"},
};
const unsigned xrc_manual_crt_math_n = sizeof xrc_manual_crt_math / sizeof xrc_manual_crt_math[0];
