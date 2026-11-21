/* xrc_rt.h - the types of the dispatch table (xrc_table.c includes this header from the runtime). A real runtime
 * declares much more here; these three are what the table needs. */
#ifndef XRC_RT_H
#define XRC_RT_H
#include "xrc.h"
typedef struct XrcFn { u32 addr; XFn fn; } XrcFn;               /* a translation: guest address -> xf_<addr> */
typedef struct XrcOverride { u32 addr; const char *why; } XrcOverride; /* a function not translated, and why */
typedef struct XrcRange { u32 addr, len; } XrcRange;            /* a data range the table's hash covers */
extern const XrcFn xrc_fns[];
extern const unsigned xrc_nfns;
extern const XrcOverride xrc_overrides[];
extern const unsigned xrc_noverrides;
extern const unsigned xrc_entry;
#endif
