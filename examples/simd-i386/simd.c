/* simd.c - checks xrc's integer SIMD translations (SSE2 xmm and MMX mm forms, psrldq / pslldq, SSSE3 pmaddubsw /
 * pshufb, lock xchg, pause) against the CPU: every op on many inputs (random, and the lane edge values that saturate),
 * each result printed in hex. run.sh compares the native run with the translation's. No C library (int 0x80). */
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

static int sys(int n, int a, int b, int c) {
	int r;
	__asm__ volatile("int $0x80" : "=a"(r) : "a"(n), "b"(a), "c"(b), "d"(c) : "memory");
	return r;
}
static char obuf[4096];
static int on;
static void flush(void) { sys(4, 1, (int)obuf, on); on = 0; }
static void outc(char ch) { if (on == sizeof obuf) flush(); obuf[on++] = ch; }
static void out(const char *s) { while (*s) outc(*s++); }
static void hex(u32 v) { for (int i = 28; i >= 0; i -= 4) outc("0123456789abcdef"[(v >> i) & 15]); }
static void res(const char *name, const u8 *r, int n) {
	out(name); out(" ");
	for (int i = n - 4; i >= 0; i -= 4) hex(*(const u32 *)(r + i));
	out("\n");
}

static u32 seed = 12345;
static u32 rnd(void) { seed = seed * 1103515245u + 12345u; return seed >> 8 ^ seed << 13; }
/* lane edge values: 0, 1, -1, the signed and unsigned byte / word / dword limits */
static const u32 edges[] = {0, 1, 0xffffffffu, 0x7f7f7f7fu, 0x80808080u, 0x7fff7fffu, 0x80008000u, 0x7fffffffu,
	0x80000000u, 0x00ff00ffu, 0xff00ff00u, 0x01000100u, 0x0000ffffu, 0xffff0000u, 0x00800080u, 0x00008000u};
static void fill(u8 *p, int it) {
	for (int i = 0; i < 16; i += 4) {
		u32 v = rnd();
		if (it & 1) v = edges[(v >> 4) & 15];
		else if (it & 2) v &= 0x0f0f0f0fu; /* small values: shift counts, pshufb indexes */
		*(u32 *)(p + i) = v;
	}
}

static u8 A[16] __attribute__((aligned(16))), B[16] __attribute__((aligned(16))), R[16] __attribute__((aligned(16)));

#define XOP(op) do { \
	__asm__ volatile("movdqa %1, %%xmm0\n\tmovdqa %2, %%xmm1\n\t" #op " %%xmm1, %%xmm0\n\tmovdqa %%xmm0, %0" \
		: "=m"(R) : "m"(A), "m"(B) : "xmm0", "xmm1"); res(#op " x", R, 16); \
	__asm__ volatile("movdqa %1, %%xmm2\n\t" #op " %2, %%xmm2\n\tmovdqa %%xmm2, %0" \
		: "=m"(R) : "m"(A), "m"(B) : "xmm2"); res(#op " xm", R, 16); \
} while (0)
#define MOP(op) do { \
	__asm__ volatile("movq %1, %%mm0\n\tmovq %2, %%mm1\n\t" #op " %%mm1, %%mm0\n\tmovq %%mm0, %0\n\temms" \
		: "=m"(*(u64 *)R) : "m"(*(u64 *)A), "m"(*(u64 *)B) : "mm0", "mm1"); res(#op " m", R, 8); \
	__asm__ volatile("movq %1, %%mm3\n\t" #op " %2, %%mm3\n\tmovq %%mm3, %0\n\temms" \
		: "=m"(*(u64 *)R) : "m"(*(u64 *)A), "m"(*(u64 *)(B + 8)) : "mm3"); res(#op " mm", R, 8); \
} while (0)
#define BOTH(op) do { XOP(op); MOP(op); } while (0)
#define XIMM(op, n) do { \
	__asm__ volatile("movdqa %1, %%xmm4\n\t" #op " $" #n ", %%xmm4\n\tmovdqa %%xmm4, %0" : "=m"(R) : "m"(A) : "xmm4"); \
	res(#op " " #n " x", R, 16); } while (0)
#define MIMM(op, n) do { \
	__asm__ volatile("movq %1, %%mm5\n\t" #op " $" #n ", %%mm5\n\tmovq %%mm5, %0\n\temms" \
		: "=m"(*(u64 *)R) : "m"(*(u64 *)A) : "mm5"); res(#op " " #n " m", R, 8); } while (0)

/* the shifts by a register count: counts 0..63 from B's low byte (small), and past the lane widths */
static void shifts_by_reg(void) {
	static const u32 counts[] = {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 200};
	for (unsigned k = 0; k < sizeof counts / sizeof counts[0]; k++) {
		*(u32 *)B = counts[k]; *(u32 *)(B + 4) = 0;
		*(u32 *)(B + 8) = counts[k]; *(u32 *)(B + 12) = 0;
		MOP(psrlw); MOP(psrld); MOP(psrlq); MOP(psraw); MOP(psrad); MOP(psllw); MOP(pslld); MOP(psllq);
	}
}

void _start(void) {
	for (int it = 0; it < 48; it++) {
		fill(A, it); fill(B, it >> 1);
		BOTH(punpcklbw); BOTH(punpcklwd); BOTH(punpckldq); BOTH(punpckhbw); BOTH(punpckhwd); BOTH(punpckhdq);
		BOTH(packsswb); BOTH(packuswb); BOTH(packssdw);
		BOTH(pcmpgtb); BOTH(pcmpgtw); BOTH(pcmpgtd); BOTH(pcmpeqb); BOTH(pcmpeqw); BOTH(pcmpeqd);
		BOTH(paddb); BOTH(paddw); BOTH(paddd); BOTH(paddq); BOTH(psubb); BOTH(psubw); BOTH(psubd); BOTH(psubq);
		BOTH(paddsb); BOTH(paddsw); BOTH(paddusb); BOTH(paddusw); BOTH(psubsb); BOTH(psubsw); BOTH(psubusb); BOTH(psubusw);
		BOTH(pmullw); BOTH(pmulhw); BOTH(pmulhuw); BOTH(pmaddwd); BOTH(pmuludq); BOTH(psadbw);
		BOTH(pavgb); BOTH(pavgw); BOTH(pminub); BOTH(pmaxub); BOTH(pminsw); BOTH(pmaxsw);
		BOTH(pand); BOTH(pandn); BOTH(por); BOTH(pxor);
		XOP(pmaddubsw); XOP(pshufb);
		XIMM(psrldq, 0); XIMM(psrldq, 1); XIMM(psrldq, 2); XIMM(psrldq, 8); XIMM(psrldq, 15); XIMM(psrldq, 16); XIMM(psrldq, 200);
		XIMM(pslldq, 0); XIMM(pslldq, 1); XIMM(pslldq, 3); XIMM(pslldq, 8); XIMM(pslldq, 15); XIMM(pslldq, 16); XIMM(pslldq, 255);
		MIMM(psrlw, 0); MIMM(psrlw, 3); MIMM(psrlw, 15); MIMM(psrlw, 16); MIMM(psraw, 1); MIMM(psraw, 15); MIMM(psraw, 40);
		MIMM(psllw, 4); MIMM(psllw, 16); MIMM(psrld, 5); MIMM(psrld, 32); MIMM(psrad, 7); MIMM(psrad, 31); MIMM(psrad, 99);
		MIMM(pslld, 4); MIMM(pslld, 31); MIMM(psrlq, 1); MIMM(psrlq, 33); MIMM(psrlq, 64); MIMM(psllq, 12); MIMM(psllq, 63);
		MIMM(psllq, 90);
		if ((it & 7) == 0) shifts_by_reg();
	}
	/* lock xchg with memory, and pause */
	{
		u32 m = 0x12345678u, r = 0xcafef00du;
		__asm__ volatile("lock xchgl %0, %1\n\tpause" : "+r"(r), "+m"(m));
		out("lock xchg "); hex(r); out(" "); hex(m); out("\n");
	}
	flush();
	sys(1, 0, 0, 0);
}
