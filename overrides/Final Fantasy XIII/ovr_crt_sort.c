/* ovr_crt_sort.c - the CRT's qsort (VC9 libcmt, 01011d90) for the comparators known to be plain key orders.
 *
 * The game sorts a 32768-entry pointer table every frame (00ec4560) with 00ec4440, an unsigned compare of the two
 * u32 elements: some 130k comparator calls a frame, each through the dispatcher. Its order is total and equal
 * elements are identical, so any correct sort gives the very array the CRT's would: that one is radix-sorted natively.
 * 00f049f0 orders u16 indices by a 64-bit key from a table (render items, every frame): equal keys can belong to
 * different elements, so its sort is the CRT's own algorithm (crt_qsort below, written from the game's code) with the comparator
 * native -- the same array, no dispatcher call per comparison.
 * Any other comparator runs the CRT's qsort as translated (it stays in the dispatch table: overrides.tsv lists it as
 * "wrapped", so nothing calls it directly). XRC_QSORT_CHECK=1 also runs the translation on a copy and compares. */
#include <stdlib.h>
#include <string.h>
#include "xrc_rt.h"

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
/* a native comparator: <0, 0, >0 as the CRT's */
typedef int (*NCmp)(const u8 *a, const u8 *b);
/* The game's CRT qsort as its machine code runs it, step for step: the same comparisons and swaps in the same order,
 * so equal elements end up in the same order too. An explicit-stack quicksort: a range of more than 8 elements is
 * split around the median of its first, middle and last (q_split), the larger part is put aside and the smaller one
 * done next; a range of 8 or fewer is finished by sort_small. */
static void swap_w(u8 *a, u8 *b, u32 w) {
	if (a != b) while (w--) { u8 t = *a; *a++ = *b; *b++ = t; }
}
/* [first, last] in order: the largest element swapped to the end, the range shortened by one, again */
static void sort_small(u8 *first, u8 *last, u32 w, NCmp cmp) {
	for (; last > first; last -= w) {
		u8 *big = first, *p;
		for (p = first + w; p <= last; p += w)
			if (cmp(p, big) > 0) big = p;
		swap_w(big, last, w);
	}
}
/* [first, last] (more than 8 elements) split around a pivot: what is left to sort is [first, below] and
 * [above, last] (the elements equal to the pivot between them are in place) */
typedef struct { u8 *below, *above; } QSplit;
static QSplit q_split(u8 *first, u8 *last, u32 w, NCmp cmp) {
	size_t count = (size_t)(last - first) / w + 1;
	u8 *piv = first + count / 2 * w, *up = first, *down = last;
	QSplit s;
	/* the pivot: the median of the three, which also leaves first <= pivot <= last */
	if (cmp(first, piv) > 0) swap_w(first, piv, w);
	if (cmp(first, last) > 0) swap_w(first, last, w);
	if (cmp(piv, last) > 0) swap_w(piv, last, w);
	for (;;) {
		/* up: to the next element above the pivot (first the ones before it, then past it) */
		if (up < piv) {
			do up += w; while (up < piv && cmp(up, piv) <= 0);
		}
		if (up >= piv) {
			do up += w; while (up <= last && cmp(up, piv) <= 0);
		}
		/* down: to the next element not above it */
		do down -= w; while (down > piv && cmp(down, piv) > 0);
		if (down < up) break;
		swap_w(up, down, w);
		if (piv == down) piv = up; /* (the pivot itself moved) */
	}
	/* down: back over the elements equal to the pivot */
	down += w;
	if (down > piv) {
		do down -= w; while (down > piv && cmp(down, piv) == 0);
	}
	if (down <= piv) {
		do down -= w; while (down > first && cmp(down, piv) == 0);
	}
	s.below = down;
	s.above = up;
	return s;
}
static void crt_qsort(u8 *base, u32 num, u32 w, NCmp cmp) {
	u8 *later_first[64], *later_last[64]; /* (the ranges put aside) */
	u8 *first, *last;
	int nlater = 0;
	if (num < 2 || !w) return;
	first = base;
	last = base + (size_t)w * (num - 1);
	for (;;) {
		if ((size_t)(last - first) / w + 1 <= 8) sort_small(first, last, w, cmp);
		else {
			QSplit s = q_split(first, last, w, cmp);
			/* the larger part put aside (if it has two elements or more), the smaller one next */
			if (s.below - first >= last - s.above) {
				if (first < s.below) { later_first[nlater] = first; later_last[nlater] = s.below; nlater++; }
				if (s.above < last) { first = s.above; continue; }
			} else {
				if (s.above < last) { later_first[nlater] = s.above; later_last[nlater] = last; nlater++; }
				if (first < s.below) { last = s.below; continue; }
			}
		}
		if (nlater == 0) return;
		nlater--;
		first = later_first[nlater];
		last = later_last[nlater];
	}
}
/* 00f049f0: elements' u16 index i; key ((u64)T[i].+4 << 32) | T[i].+8, T = *(*(0x028670a0) + 0x108) (12-byte
 * records), compared unsigned (the original's __allshl of the sign-extended +4 by 32 keeps its low word only) */
static const u8 *cmp_f049f0_t;
static u64 key_f049f0(const u8 *e) {
	const u8 *r = cmp_f049f0_t + 12u * (u32)(e[0] | e[1] << 8);
	u32 hi, lo;
	memcpy(&hi, r + 4, 4); memcpy(&lo, r + 8, 4);
	return (u64)hi << 32 | lo;
}
static int cmp_f049f0(const u8 *a, const u8 *b) {
	u64 x = key_f049f0(a), y = key_f049f0(b);
	return x > y ? 1 : x < y ? -1 : 0;
}
static void XRC_CC m_qsort(XCtx *c) {
	static XFn orig;
	static int check = -1;
	u32 base = xrc_arg(c, 0), num = xrc_arg(c, 1), width = xrc_arg(c, 2), cmp = xrc_arg(c, 3);
	if (!orig) orig = xrc_translation_of(0x01011d90u);
	if (check < 0) check = xrc_tool_env("XRC_QSORT_CHECK") != 0;
	if (cmp == 0x00ec4440u && width == 4 && num < 0x10000000u) {
		u32 *v = (u32 *)(xrc_mem + base);
		u32 *want = 0;
		if (check && num) { /* the translation's result, for comparison */
			u32 *save = (u32 *)malloc((size_t)num * 4);
			memcpy(save, v, (size_t)num * 4);
			orig(c);
			want = (u32 *)malloc((size_t)num * 4);
			memcpy(want, v, (size_t)num * 4);
			memcpy(v, save, (size_t)num * 4);
			free(save);
		}
		sort_u32(v, num);
		if (want) {
			if (memcmp(want, v, (size_t)num * 4)) xrc_fatal(c, "ovr qsort: the native sort differs from the CRT's (%u elements)", num);
			free(want);
			return; /* (orig already returned for this call) */
		}
		xrc_cdecl_ret(c);
		return;
	}
	static int native_f049f0 = -1; /* (XRC_QSORT_NATIVE=0: the translated CRT's sort, to compare) */
	if (native_f049f0 < 0) native_f049f0 = !(xrc_tool_env("XRC_QSORT_NATIVE") && atoi(xrc_tool_env("XRC_QSORT_NATIVE")) == 0);
	if (cmp == 0x00f049f0u && native_f049f0 && width >= 2 && num < 0x10000000u && (unsigned long long)num * width < 0x40000000ull) {
		u8 *v = xrc_mem + base, *want = 0;
		size_t bytes = (size_t)num * width;
		cmp_f049f0_t = xrc_mem + M32(M32(0x028670a0u) + 0x108u);
		if (check && num) {
			u8 *save = (u8 *)malloc(bytes);
			memcpy(save, v, bytes);
			orig(c);
			want = (u8 *)malloc(bytes);
			memcpy(want, v, bytes);
			memcpy(v, save, bytes);
			free(save);
		}
		crt_qsort(v, num, width, cmp_f049f0);
		if (want) {
			static unsigned nchk;
			if (memcmp(want, v, bytes)) xrc_fatal(c, "ovr qsort: the native f049f0 sort differs from the CRT's (%u x %u bytes)", num, width);
			free(want);
			if ((++nchk, nchk & (nchk - 1)) == 0) fprintf(stderr, "xrc: ovr qsort: %u f049f0 sorts checked (this one %u x %u bytes)\n", nchk, num, width);
			return;
		}
		xrc_cdecl_ret(c);
		return;
	}
	orig(c);
}

const XrcManual xrc_manual_crt_sort[] = {
	{0x01011d90u, m_qsort, "qsort"},
};
const unsigned xrc_manual_crt_sort_n = sizeof xrc_manual_crt_sort / sizeof xrc_manual_crt_sort[0];
