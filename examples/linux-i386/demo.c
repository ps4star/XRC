/* demo.c - a small Linux i386 program for trying xrc: no C library, system calls through int 0x80.
 * It exercises what compiled C code is made of: a switch (a jump table), recursion, function pointers, x87 floating
 * point, struct copies (rep movs), 64-bit arithmetic, thread-local storage through gs: (set up with
 * set_thread_area) and, built with lib.c as a shared library, calls through the PLT into it. Its output is the same
 * run natively and run as xrc's translation (run.sh compares the two). */

static int sys(int n, int a, int b, int c) {
	int r;
	__asm__ volatile("int $0x80" : "=a"(r) : "a"(n), "b"(a), "c"(b), "d"(c) : "memory");
	return r;
}
static void out(const char *s) { int n = 0; while (s[n]) n++; sys(4, 1, (int)s, n); }
/* u / 10 with shifts and adds (no libgcc: a 64-bit divide would be a call to __udivdi3) */
static unsigned long long div10(unsigned long long n, unsigned *rem) {
	unsigned long long q = (n >> 1) + (n >> 2);
	q += q >> 4; q += q >> 8; q += q >> 16; q += q >> 32; q >>= 3;
	unsigned r = (unsigned)(n - ((q << 3) + (q << 1)));
	if (r > 9) { q++; r -= 10; }
	*rem = r;
	return q;
}
static void out_int(long long v) {
	char b[24]; int i = 23; int neg = v < 0; unsigned long long u = neg ? -(unsigned long long)v : (unsigned long long)v;
	b[i] = 0;
	do { unsigned r; u = div10(u, &r); b[--i] = (char)('0' + r); } while (u);
	if (neg) b[--i] = '-';
	out(b + i);
}
static void line(const char *k, long long v) { out(k); out(" = "); out_int(v); out("\n"); }

__attribute__((noinline)) static int classify(int x, int y) {
	switch (x) {
	case 0: return y + 11; case 1: return y * 23; case 2: return y ^ 37; case 3: return y - 41;
	case 4: return (y << 3) + 53; case 5: return y * y + 67; case 6: return 71 - y; case 7: return y | 89;
	default: return -1;
	}
}
__attribute__((noinline)) static int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
static int add(int a, int b) { return a + b; }
static int mul(int a, int b) { return a * b; }
static int sub(int a, int b) { return a - b; }
static int (*const ops[3])(int, int) = {add, mul, sub};

struct big { int v[16]; };
__attribute__((noinline)) static struct big make(int k) { struct big b; for (int i = 0; i < 16; i++) b.v[i] = k * i; return b; }

__attribute__((noinline)) static double area(double r) { return 3.14159265358979 * r * r; }
__attribute__((noinline)) static double root(double x) { double g = x; for (int i = 0; i < 20; i++) g = 0.5 * (g + x / g); return g; }

/* thread-local storage the way Linux i386 does it: a descriptor from set_thread_area, gs: its selector */
struct user_desc { unsigned entry, base, limit; unsigned flags; };
static int tls_block[4] = {0, 0, 0, 0};
static int tls_setup(void) {
	struct user_desc d = {(unsigned)-1, (unsigned)tls_block, 0xfffff, 0x51}; /* seg_32bit, limit_in_pages, useable */
	if (sys(243, (int)&d, 0, 0) != 0) return -1;
	unsigned short sel = (unsigned short)(d.entry * 8 + 3);
	__asm__ volatile("movw %0, %%gs" ::"r"(sel));
	return 0;
}
__attribute__((noinline)) static int tls_get(void) { int v; __asm__ volatile("movl %%gs:4, %0" : "=r"(v)); return v; }
__attribute__((noinline)) static void tls_set(int v) { __asm__ volatile("movl %0, %%gs:4" ::"r"(v)); }

#ifdef WITH_LIB
int lib_scale(int x, int k);
const char *lib_name(void);
#endif

void demo_main(void) {
	int s = 0;
	for (int i = -1; i < 9; i++) s = s * 3 + classify(i, i + s % 7);
	line("switch", s);
	line("fib(24)", fib(24));
	int acc = 7;
	for (int i = 0; i < 9; i++) acc = ops[i % 3](acc, i + 2);
	line("function pointers", acc);
	struct big b = make(5), c = b;
	line("struct copy", c.v[15] + c.v[3]);
	line("x87 area*1000", (long long)(area(2.5) * 1000.0));
	line("x87 sqrt(2)*1e9", (long long)(root(2.0) * 1e9));
	unsigned long long h = 1469598103934665603ull;
	for (int i = 0; i < 100; i++) h = (h ^ (unsigned)i) * 1099511628211ull;
	line("64-bit hash", (long long)(h >> 1));
	if (tls_setup() == 0) { tls_set(1234); line("gs: TLS", tls_get() + tls_block[1]); }
#ifdef WITH_LIB
	out("library: "); out(lib_name()); out("\n");
	line("lib_scale", lib_scale(21, 2));
#endif
	sys(1, 0, 0, 0); /* exit(0) */
}

/* the entry point: the stack is aligned for the C code, then demo_main */
__asm__(".globl _start\n_start:\n\tandl $-16, %esp\n\tcall demo_main\n\thlt\n");
