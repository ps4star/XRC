/* lib.c - a shared library for demo.c (the dynamic build). A runtime does not translate it: the example runtime has
 * its own native lib_scale and lib_name and binds the PLT's slots to them, the way a real runtime implements a
 * program's libraries (libc, ...) natively. */
int lib_scale(int x, int k) { return x * k; }
const char *lib_name(void) { return "native lib"; }
