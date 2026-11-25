# xrc

**xrc** is a static recompiler: it translates the machine code of a 32-bit x86 program into C, ahead of time, one C
function per x86 function. Compiled together with a runtime, the generated C runs the original program's logic
natively on any CPU and OS a C compiler targets: x86-64, ARM64, Linux, Windows, macOS, Android.

It reads the common 32-bit x86 executable formats:

| Format | Files |
|---|---|
| **PE** | Windows `.exe` and `.dll` |
| **ELF** | Linux and BSD executables, PIEs and shared objects (`.so`), static or dynamically linked |
| **Mach-O** | macOS i386 executables, dylibs and bundles, and the i386 slice of a fat (universal) file |
| **Raw** | a flat image loaded at a given address (boot code, firmware, a dumped image) |

It was written for, and is used by, a port of **FINAL FANTASY XIII** (2014 Steam release) to 64-bit PCs, Linux, macOS
and Android. For that game it translates all 55,770 functions of the game's executable, and the result runs the whole
game. The translator itself is not tied to the game: what is specific to one program lives in its folder under
`overrides/`, and `overrides/Final Fantasy XIII/` is the example. `examples/linux-i386/` is a complete small one: a
Linux program, translated and run on x86-64 or ARM64 with a 200-line runtime.

xrc is written in [Odin](https://odin-lang.org/). It does not include any game: you supply the executable you own.

---

## Contents

- [What it does](#what-it-does)
- [What it does not do](#what-it-does-not-do)
- [Building](#building)
- [Usage](#usage)
- [Quick start: a Linux program](#quick-start-a-linux-program)
- [Executable formats](#executable-formats)
- [Command-line options](#command-line-options)
- [Environment variables](#environment-variables)
- [The game folder](#the-game-folder)
- [What xrc writes](#what-xrc-writes)
- [Compiling the output: the runtime](#compiling-the-output-the-runtime)
- [Tutorial: overrides for your own executable](#tutorial-overrides-for-your-own-executable)
- [How the translation works](#how-the-translation-works)
- [Optimizations](#optimizations)
- [Profiles for inlining and devirtualization](#profiles-for-inlining-and-devirtualization)
- [The LLVM IR backend (experimental)](#the-llvm-ir-backend-experimental)
- [Repository layout](#repository-layout)
- [License](#license)
- [Copyright and game files](#copyright-and-game-files)

---

## What it does

Given a 32-bit x86 executable and a map of its functions (its own symbols, or a function map you supply), xrc:

1. **Loads** the executable's sections or segments, its symbols and import stubs, and the program's tables: the
   function map and optional profiles. A PIE or shared object is placed at a load address, with its relocations
   applied.
2. **Finds the functions the map is missing**: addresses that direct calls, `push offset f`, jump tables or data
   reach become functions too.
3. **Translates each function to C**, `void xf_<addr>(XCtx *c)`:
   - Every integer, control-flow, string, x87, MMX and SSE/SSE2 instruction the binary uses is handled.
   - Switch jump tables become C `switch` statements, so no byte of the original code is read at run time. Both
     absolute tables (`jmp [idx*4 + table]`) and the GOT-relative ones of position-independent code are handled.
   - Software interrupts (`int 0x80`, Linux's system calls) and `fs:` / `gs:` thread-local storage go to the
     runtime.
4. **Optimizes** what it writes, as far as it can prove safe:
   - Flags are computed only where something reads them, and compare-and-branch pairs are fused.
   - Calls between translated functions are direct.
   - Registers are saved and reloaded around calls only when needed.
   - Small functions are inlined.
   - Hot virtual calls are devirtualized.
5. **Writes** one C file per 64 KB page of the executable, plus a dispatch table that maps every guest address to its
   translation.
6. **Copies the game's hand-written override routines** next to the output. These are C replacements for the few
   functions that should not, or cannot, be translated mechanically.

The generated C calls nothing but what `include/xrc.h` declares, so it compiles with any C99 compiler (Clang, GCC)
for any target.

## What it does not do

- **It is not a runtime.** The generated code needs one: guest memory, threads, the dispatcher, and the operating
  system's API (Win32 and Direct3D, or libc and Linux system calls, ...) implemented on the host. This repository has
  the interface the output expects (`include/xrc.h`, [below](#compiling-the-output-the-runtime)) and a minimal
  example runtime for small Linux programs (`examples/linux-i386/`), not a full runtime. The FINAL FANTASY XIII port
  has its own runtime, which is not part of this repository.
- **It does not translate shared libraries along with a program.** A program's imports (libc, `kernel32.dll`, ...)
  are the runtime's to provide, usually natively. A library can be translated on its own like any other executable.
- **It does not unpack or decrypt.** DRM-wrapped executables (e.g. SteamStub) and packed ones (UPX) must be unpacked
  first. xrc reads the sections as they are in the file.
- **It is not a decompiler.** The output is machine-level C (registers as variables, memory as `M32(addr)`), meant to
  be compiled, not read.
- **It handles 32-bit x86 only** (i386 through SSE2, the subset MSVC- and GCC-era programs use). No 64-bit code, no
  AVX.
- **It assumes code is not generated or modified at run time** (no JIT, no self-modifying code). What it translates
  is what is on disk. ELF files with text relocations (`DT_TEXTREL`: non-PIC code linked as a PIE) are not supported.
## Building

xrc needs the **Odin compiler** with a small fix to its x86 decoder, `core:rexcode`. Three i386 decoding cases are
wrong upstream:
- the accumulator-immediate forms (`05 id` decodes as `ADD AX, imm16`);
- `moffs` operands (`A0`–`A3`), whose offset is read as 8 bytes instead of 4;
- SIB with no base and `mod=0`, which needs a disp32 (`jmp [eax*4 + table]`).

The fix is `patches/odin-rexcode-i386.patch`. It was made against Odin commit `e28f5d1ff` and applies to later ones
while that code is unchanged.

```sh
git clone https://github.com/odin-lang/Odin && cd Odin
git apply /path/to/xrc/patches/odin-rexcode-i386.patch
# then build Odin as its README says (build.bat on Windows, build_odin.sh on Linux / macOS)
```

Then build xrc with that Odin:

```sh
# Windows
set ODIN=C:\path\to\Odin\odin.exe
build.bat

# Linux / macOS
ODIN=/path/to/Odin/odin ./build.sh
```

The output is `xrc.exe` (Windows) or `xrc` beside the scripts, and a build takes a few seconds. The translator
recurses deeply on large functions, so it is linked with a 16 MB stack. The Windows build is the one in daily use.
The Linux/macOS script is provided but has seen less testing: if xrc crashes on a big binary there, run it with
`ulimit -s 16384` or higher.

## Usage

```sh
xrc --exe ffxiiiimg.exe.unpacked.exe --override "Final Fantasy XIII" --all \
    --out gen/src --table gen/xrc_table.c --report coverage.tsv
```

What this command does:
- It reads the executable and the game folder `overrides/Final Fantasy XIII/`: its tables in `data/`, its `xrc.cfg`
  and its `overrides.tsv`.
- It translates every function into `gen/src/page_XXXX.c`, and writes the dispatch table and a coverage report.
- It copies the game's override routines to `gen/src/overrides/`.

Only pages whose content changed are rewritten, so a build system recompiles only what changed.

A run prints a summary. For FINAL FANTASY XIII:

```
xrc: PE i386 executable, 5 sections, entry 01013b3c; 52856 functions (function_map.csv)
xrc: 2914 entry points found from direct calls and address references (not in the function map)
xrc: 49082 of 55770 functions keep ebx esi edi ebp (direct calls to them reload only eax ecx edx esp)
xrc: 26028 functions of up to 24 instructions inlined at their direct calls (11384 leaves, 14644 that call)
xrc: 1810 indirect call sites devirtualized, 85 of them with two targets (profile ...devirt_sites.txt: 6369 sites)
xrc: 202 page files in gen/src, 202 changed; 72519 calls inlined, 1790 indirect calls devirtualized
xrc: register syncs trimmed at 178699 calls / returns: 579919 of 1250893 stores, 322213 of 542487 loads left out
xrc: 6 override files from overrides/Final Fantasy XIII in gen/src/overrides, 6 changed
xrc: 55770 functions  55717 ok (99.9%)  15 hand-written  38 unreachable  0 need an override (covered 100.00%)
```

An executable that has symbols (most Linux and macOS programs, unless stripped) needs no game folder at all. Its
symbols are the function map:

```sh
xrc --exe ./myprogram --all --out gen --table gen/xrc_table.c --report coverage.tsv
```

**For a new program:**
1. If it has no symbols (most Windows programs, stripped Linux ones), make a game folder (see
   [The game folder](#the-game-folder)) with at least `data/function_map.csv`, a Ghidra export of its functions.
2. Run xrc without `--out` first. It writes only the report, which lists every function it could not translate and
   why.
3. Each of those needs a hand-written override, or a fix to xrc. The
   [tutorial](#tutorial-overrides-for-your-own-executable) shows how.

## Quick start: a Linux program

`examples/linux-i386/` takes a small C program from source to a translation running on a 64-bit machine:

```sh
cd examples/linux-i386
sh run.sh
```

`run.sh` does the following:
1. It builds `demo.c` as a 32-bit Linux program four ways: static, PIE, dynamically linked with a shared library
   (`lib.c`), and a PIE with that library.
2. It translates each build with xrc.
3. It compiles each translation for the machine you are on, with the example runtime.
4. It runs each one, and compares the output with the program run natively.

The demo covers a switch (a jump table), recursion, function pointers, x87 floating point, struct copies, 64-bit
arithmetic, `gs:` thread-local storage, Linux system calls, and calls into a shared library through the PLT. It
needs clang with the i386 target and GNU ld or lld. On an x86-64 Linux that can run 32-bit programs, every run is
compared with the native one. On ARM64 the translation runs the same; there is just nothing native to compare with.

The example runtime, `examples/linux-i386/runtime/runtime.c`, is about 200 lines. It shows what any runtime does:
- it loads the segments into guest memory and applies a PIE's relocations;
- it dispatches by guest address;
- it binds imports to native C functions;
- it implements system calls and `gs:`.

## Executable formats

xrc detects the format from the file. Every format comes down to the same model: sections (or segments) with an
address, a size and code/data/write flags, an entry point, and functions with an address and a size.

| | PE | ELF | Mach-O | Raw (`--raw ADDR`) |
|---|---|---|---|---|
| Sections | section table | `PT_LOAD` segments | `LC_SEGMENT` segments | the whole file at ADDR |
| Code range | executable sections | `SHF_EXECINSTR` sections, else executable segments | instruction sections (not the self-modifying `__jump_table`) | the whole file |
| Functions, with no map | none: a function map is needed | `.symtab` / `.dynsym` (`STT_FUNC`, `STT_GNU_IFUNC`) | `LC_SYMTAB`, each to the next symbol | none: a function map is needed |
| Import stubs | (calls through the IAT) | every `.plt` / `.plt.sec` / `.plt.got` entry | every `__symbol_stub` | — |
| Entry point | `AddressOfEntryPoint` | `e_entry` | `LC_MAIN` or `LC_UNIXTHREAD` | ADDR |
| Load address | the image base | its addresses; a PIE / `.so` at `--base` | its addresses | ADDR |

Notes on each format:

- **PE.**
  - `fs:` accesses address the current thread's TEB: `fs:[ea]` becomes `XRC_SEG_FS + ea`, so the runtime points
    `XRC_SEG_FS` at each guest thread's TEB. That covers the SEH exception-handler chain at `fs:[0]`, which MSVC's
    frames push and pop, and implicit TLS through the array at `fs:[0x2c]`. The runtime builds the TEB fields the
    program reads.
  - Segment registers read as the WOW64 selectors.
- **ELF.**
  - **PIE and shared-object placement.** A PIE or shared object, linked at address 0, is placed at `--base` (default
    `56555000`, where Linux puts a 32-bit PIE). At address 0, small integers would pass for code addresses.
  - **Relocations.** Its `R_386_RELATIVE` and `R_386_JMP_SLOT` relocations, and its `R_386_32` / `R_386_GLOB_DAT`
    relocations to its own symbols, are applied, so pointer tables in its data hold run-time addresses.
  - **Load base in the table.** The dispatch table then has `xrc_load_base`, which is where the runtime must load it.
  - **Thread-local storage.** `gs:` and `fs:` accesses become `XRC_SEG_GS + address` (and `XRC_SEG_FS`). Loading a
    segment register calls `xrc_set_seg`, so the runtime can look up the descriptor `set_thread_area` made.
- **Mach-O.**
  - Fat files are read for their i386 slice.
  - The loader is tested on i386 Mach-O object files only. No i386 Mach-O linker was available to build a test
    executable.
  - `__jump_table` stubs are filled in by dyld at load time, so they are not code to translate. Calls to them go
    through the dispatcher, where the runtime binds them.
- **Raw.** The whole file is code and data at ADDR, writable, with its entry point at ADDR. A function map is needed.

**Position-independent code** (ELF PIEs and shared objects, Mach-O):
- **GOT-relative switch tables.** Switches through GOT-relative tables are recognized in two forms: GCC's
  `mov r, [got + idx*4 + table]; add r, got; jmp r`, and the `add got, [got + idx*4 + table]; jmp got` that clang and
  newer GCC emit. Each becomes a C `switch` on the computed target.
- **The `call $+5; pop reg` idiom**, which reads the code's own address, is translated as the push it is.

## Command-line options

| Option | What it does |
|---|---|
| `--exe FILE` | **Required.** The 32-bit x86 executable to translate: a PE, ELF or Mach-O file (detected from its contents), or a flat image with `--raw`. It must be unpacked, since xrc reads it as it is in the file. |
| `--raw ADDR` | `--exe` is a flat image: the whole file is code and data, loaded at `ADDR` (hex), with its entry point at `ADDR`. Needs a function map. |
| `--base ADDR` | Where a PIE or shared object (an ELF linked at 0) is placed: its addresses plus `ADDR` (hex). The default is `56555000`, where Linux puts a 32-bit PIE. The runtime must load it at the same address (the table's `xrc_load_base`). |
| `--override NAME` | The game folder. xrc looks for `overrides/NAME/` beside the xrc executable, then in the current directory, then takes `NAME` as a path. From that folder it uses the tables in `data/`, `xrc.cfg`, `overrides.tsv`, and the hand-written `.c`/`.h` files, which it copies to `<out>/overrides/`. `--override=NAME` works too. Without it, nothing is overridden. |
| `--data DIR` | Take the tables from `DIR` instead of the game folder's `data/`. `--data=DIR` works too. Without a function map there, xrc uses the executable's own symbols; with neither, it stops. |
| `--all` | Translate every function in the function map, plus the entry points xrc finds itself. |
| `--list FILE` | Translate only the functions listed in `FILE`: a hex address first on each line, `#` for comments. Optimizations that need the whole program (direct calls, inlining, devirtualization) are off. |
| `--out DIR` | Write the page files (`page_XXXX.c`) to `DIR`. Without it nothing is written but the report: a coverage pass. |
| `--report FILE` | Write a TSV of every function with the columns `address`, `status`, `nins` (instruction count) and `reason`. `status` is `ok`, `manual` (hand-written), `unreachable`, or `override`, which means it needs a hand-written override and `reason` names the first instruction or construct xrc cannot do. |
| `--table FILE` | Write the dispatch table, `xrc_table.c` (see [What xrc writes](#what-xrc-writes)). |
| `--matched FILE` | Functions (hex addresses) that other code replaces in the dispatcher, for example functions recompiled from matching source. Translations call them only through the dispatcher, never directly. An empty file is fine. |
| `--llvm` | Also write `page_XXXX.ll`: the functions the experimental LLVM IR backend can do (see [below](#the-llvm-ir-backend-experimental)). |
| `-h`, `--help` | Print the options. |

## Environment variables

Every variable is optional. The defaults are the settings the FINAL FANTASY XIII port ships with.

### Optimizations

| Variable | Default | Effect |
|---|---|---|
| `XRC_FLAGS_OLD=1` | off | Turns flag liveness and compare-branch fusion off: every flag is computed lazily, as in xrc's first output. For comparison. |
| `XRC_SYNC_TRIM=0` | on | Turns register-sync trimming off, so every call and return saves and reloads every register. |
| `XRC_FLAT_EA=0` | on | Turns flat addressing off. When on, `[reg + small displacement]` addresses are 64-bit sums (`MX*` accesses) and `esp`/`ebp` live in 64-bit C variables, so the host compiler folds each displacement into one instruction instead of a separate 32-bit add (FF13: code 5% smaller, main thread 3-4% faster). Only addresses that would wrap past 0 or 4 GB differ, and none of those is valid. |
| `XRC_RDATA_FOLD=0` | on | Turns read-only constant folding off. When on, a read of a fixed address in a section that is neither writable nor executable (a float constant, a mask, a string's bytes) becomes the value itself (`M32(_ea)` -> `0x3f800000u`), so the host compiler can fold what is computed from it. The import tables and `xrc.cfg`'s `patched` entries are never folded; an instruction that also writes the address is left as it was. FF13: 39 467 of 72 119 such reads, binary 0.3 MB smaller; main thread battle -1.8%, crystarium -0.8% (alternating A/B, 4 runs each); 128-bit loads (`XLD` -> `XLDC`, 4080 sites) a further -3.7% in battle. |
| `XRC_STACK=0` | on | Turns the stack-balance analysis off. When on, a data flow over each function finds the `N` of a `ret N` that every way out of it returns with (FF13: 51076 of 55770 functions); a direct call to such a function then sets `esp` to its value before the call `+ N` instead of reading it back. `XRC_STACK_VERIFY=1` checks every such call at run time (`xrc_stack_bad`). |
| `XRC_LAYOUT=FILE` / `XRC_LAYOUT=0` | `data/layout_hot.txt` | Code layout for the instruction cache. The hot functions of a profile (lines `address weight`, any positive weights) move out of their address pages into hot pages of their own (`page_10000.c`, `page_10001.c`, ...), clustered by their direct calls (C3) and ordered by density. Compilers keep a file's functions in order and the partial link keeps the objects' order, so the hot code is contiguous on every target. FF13 (2946 functions, 1.8 MB of guest code): field -3.5%, battle -6.5%, crystarium -8% main-thread time. `0` turns it off. |
| `XRC_ABI_KEEP=0` | on | Turns the callee-saved register plan off: every direct call reloads ebx, esi, edi and ebp afterwards. |
| `XRC_INLINE=0` | on | Turns inlining off. |
| `XRC_INLINE_MAX=N` | 24 | Inlines functions of up to N instructions. |
| `XRC_INLINE_HOTMAX=N` | 64 | Inlines the hot list's functions of up to N instructions. |
| `XRC_INLINE_HOTLIST=FILE` | `<data>/inline_hot.txt` | The hot list: hex addresses, one per line. `0` means no hot list. |
| `XRC_INLINE_HOT=FILE` | none | Inlines only the functions listed in FILE, for testing. |
| `XRC_DEVIRT=FILE` | `<data>/devirt_sites.txt` | The call-site profile for devirtualization. `0` turns devirtualization off. With no file, nothing is devirtualized. |
| `XRC_DEVIRT_MIN=N` | 1000 | Devirtualizes a call site only if the profile saw at least N calls there. |

### Checking and debugging

| Variable | Effect |
|---|---|
| `XRC_TIME=1` | Prints how long each phase took. |
| `XRC_THREADS=N` | Translates on N threads (default: one per logical CPU). The output is the same for any N. |
| `XRC_FL_VERIFY=1` | Writes a checking translation: every branch condition is computed both by the flag plan and the plain way, and the runtime reports `xrc: flag plan wrong at <addr>` if they ever differ. Inlining and devirtualization are off in this mode. |
| `XRC_FL_DEBUG=1` | Prints flag-plan statistics, and why each flag reader that was not fused stayed generic. |
| `XRC_INLINE_WHY=ADDR` | Prints why the function at ADDR is or is not inlined. |
| `XRC_ABI_WHY=ADDR[,ADDR...]` | Prints which callee-saved registers each of those functions keeps, and which instructions, calls or tail jumps decide that. |
| `XRC_SYNC_RANGE=LO-HI` | Trims register syncs only in functions starting between LO and HI (hex), to bisect a wrong trim. |
| `XRC_SLOT_DEBUG=1` | Prints the stack slots that stayed in memory, and why. |
| `XRC_X87_DEBUG=1` | Prints the x87 stack analysis's merge conflicts. |
| `XRC_RX_SLOW=1` | Decodes with rexcode's general decoder instead of xrc's one-instruction fast path, to check the decoder. |

### LLVM IR backend (`--llvm`)

| Variable | Effect |
|---|---|
| `XRC_IR_WHY=1` | Prints why each function stays C. |
| `XRC_IR_OFF=feat,...` | Sends functions using those features back to C, to bisect a wrong translation. The features are `btm`, `cmpxchg`, `cpuid`, `div`, `flags`, `mmx`, `mul816`, `mxcsr`, `psr`, `rcr`, `rep`, `rot816`, `shld`, `x87t`, `xadd`, `xchgm`. |
| `XRC_IR_INLINE=0` | Turns inlining off in the IR. |

## The game folder

Everything specific to one program lives in one folder, `overrides/<NAME>/`:

```
overrides/Final Fantasy XIII/
├── xrc.cfg              facts about the binary that xrc cannot work out itself
├── overrides.tsv        the functions that are hand-written, wrapped, fast or unreachable
├── ovr_*.c / *.h        the hand-written routines (copied to <out>/overrides/)
└── data/
    ├── function_map.csv every function: address, name, size (optional if the executable has symbols)
    ├── devirt_sites.txt (optional) the call-site profile for devirtualization
    ├── inline_hot.txt   (optional) the hot functions for inlining
    └── size_overrides.txt (optional) corrections to the map's sizes
```

### `data/function_map.csv`

```
address,name,size,thunk,size_source
00401000,`vector_constructor_iterator',35,0,ghidra
```

- A header line, then one function per line: the address in hex, the name, and the size in bytes. Further columns are
  ignored.
- A Ghidra export of the program's functions works.
- The names matter only for the C runtime's x87 helpers, which xrc recognizes by name: `__CI*`, `*ftol*`,
  `__cintrindisp*`, `__ctrandisp*` and `__security_check_cookie`. Any other name is just a label.
- Without the file, the executable's own symbols are the map (ELF `.symtab` / `.dynsym`, Mach-O `LC_SYMTAB`). With
  it, the file is the map, and the executable's import stubs (PLT entries, Mach-O symbol stubs) are added.

xrc needs no import table. Calls to imports are indirect calls through their slots (the IAT, the GOT), which hold
whatever addresses the runtime puts there. Binding a program's imports is the runtime's job, from the executable's
own import information.

### `data/size_overrides.txt`

`ADDR SIZE` lines, `#` for comments, that correct sizes the function map got wrong.

### `xrc.cfg`

One entry per line, `#` for comments, addresses in hex:

| Entry | Meaning |
|---|---|
| `frame_prolog ADDR` | A compiler frame-setup helper such as MSVC's `__SEH_prolog4` or `__EH_prolog3`. It pushes ebx, esi, edi and the caller's ebp on behalf of the function that calls it, which then writes them freely. xrc treats the call as that function's own prologue. |
| `frame_epilog ADDR` | Its counterpart, such as `__SEH_epilog4` or `__EH_epilog3`. It restores all four. |
| `x87_helper ADDR TAKES GIVES` | A function whose x87 interface the stack analysis cannot derive, because its x87 paths are dynamic (the C runtime's `__CIsin` and similar). It takes TAKES values from the x87 stack and leaves GIVES there. |
| `patched ADDR [LEN]` | Read-only data (LEN bytes, default 4) that the runtime changes at run time, such as a constant it patches for a feature. Its reads are not folded into the code as constants (`XRC_RDATA_FOLD`). |

The file is optional. Without it, these calls are plain calls: correct, as long as the binary has no such helpers.

### `overrides.tsv`

```
# address	kind	name / why
0100eed4	manual	__libm_sse2_sin
```

One function per line, tab-separated:

| Kind | Meaning |
|---|---|
| `manual` | Not translated. A hand-written function replaces it. |
| `wrapped` | Translated, but reached only through the dispatcher, where a hand-written function takes over and may call the translation for the cases it does not handle. Nothing calls the translation directly. |
| `fast` | Translated, but translations call the hand-written `xo_<addr>` directly instead. The translation stays as the reference. |
| `unreachable` | Not translated. The function is dead, or reached only from replaced functions. If it is ever dispatched, the runtime stops with the reason given. |

### The hand-written routines

The `.c` and `.h` files in the folder are copied unchanged to `<out>/overrides/` for the game's build to compile with
the runtime. A file is rewritten only when its contents changed, and one that is no longer in the folder is removed.
How the routines reach the dispatcher is the runtime's business. In the FINAL FANTASY XIII port, each file exports a
table:

```c
const XrcManual xrc_manual_crt_math[] = {
	{0x0100eed4u, m_sse2_sin, "__libm_sse2_sin"},
	...
};
```

A `fast` routine is a function `void XRC_CC xo_<addr>(XCtx *c)` with the translation's calling protocol: registers in
`c`, and the return through `c->eip`.

## What xrc writes

| File | Contents |
|---|---|
| `<out>/page_XXXX.c` | The translations of the functions in the 64 KB page `XXXX0000`. Each has a `void XRC_CC xf_<addr>(XCtx *c)` and an `XRC_THUNK(<addr>)` entry, and the page includes only `xrc.h`. |
| `<out>/.xrc_pages` | Each page file's hash, size and time. A page whose content did not change is not rewritten, so its timestamp stays and a build skips it. |
| `<out>/overrides/` | The game folder's `.c`/`.h` files. |
| `<out>/page_XXXX.ll` | With `--llvm`: the functions done as LLVM IR, which are then left out of the `.c` page. |
| `--table` file | The dispatch table, described below. |
| `--report` file | One line per function: `address`, `status`, `nins`, `reason`. |

The dispatch table (`xrc_table.c`) contains:
- `xrc_fns[]` / `xrc_nfns`: every translated function as `{guest address, xf_<addr>}`, sorted.
- `xrc_overrides[]` / `xrc_noverrides`: every function that is not translated, with why.
- `xrc_entry`: the program's real entry point.
- `xrc_load_base`, only for a PIE or shared object: where xrc placed it, so where the runtime loads it.
- `xrc_hash_ranges[]` / `xrc_data_hash`: an FNV-1a hash of every non-code section as the file has them. The runtime
  can refuse to run with a different version of the program than the one translated. Code sections are left out,
  so a copy whose code is still encrypted (SteamStub) hashes the same.

A translated function looks like this:

```c
/* ---- 00401000 ---- */
void XRC_CC xf_00401000(XCtx *c);
XRC_THUNK(00401000)
void XRC_CC xf_00401000(XCtx *c) {
    u32 r_eax = c->eax, r_ecx = c->ecx, r_edx = c->edx, r_ebx = c->ebx;
    u32 r_esp = c->esp, r_ebp = c->ebp, r_esi = c->esi, r_edi = c->edi;
    ...
    /* 00401000 push push ebp */
    r_esp -= 4; M32(r_esp) = (r_ebp);
    /* 00401001 mov mov ebp, esp */
    r_ebp = (r_esp);
    ...
```

## Compiling the output: the runtime

The pages include only `include/xrc.h`. It defines:
- the guest CPU state, `XCtx`: the eight registers, `eip`, the x87 and SSE state;
- the memory macros (`M8`/`M16`/`M32`/`M64`, as offsets from `xrc_mem`, the base of a 4 GB block of guest memory);
- the flag, x87 and SSE helpers;
- the call macros (`XCALL`, `XCALLD*`, `XCALLX`, ...).

A runtime has to provide what the generated code calls. The minimal set, as the example runtime implements it:

| Symbol | What it is |
|---|---|
| `xrc_mem` | The base of guest memory, a 4 GB block; the executable's sections are loaded at their addresses in it. On x86-64 (not macOS), `xrc.h` puts guest memory at the fixed host address `0x7fff0000` unless `-DXRC_NO_FIXED_BASE` is given; then the runtime sets `xrc_mem`. |
| `xrc_dispatch_ic(c, target, ic)` | Runs the function at a guest address: a translation from `xrc_fns`, a hand-written routine, or an import. It may fill the call site's cache `ic` with a translation. |
| `xrc_imp_enter(slot)` | Called before a cached import runs, for runtimes that track the current import (may do nothing). |
| `xrc_preempt_n(c)`, `xrc_gil_waiters` | Cooperative scheduling of guest threads. The generated code checks `xrc_gil_waiters` at loop back-edges and calls. A single-threaded runtime keeps it 0. |
| `xrc_bad_return(c, target)`, `xrc_ud(c, addr)` | A function returned somewhere its caller did not expect, or an undefined or privileged instruction ran. |
| `xrc_int(c, n)` | `int n`, e.g. Linux's `int 0x80` system calls. The registers are in `c`, and `c->eip` is the next instruction. |
| `xrc_set_seg(c, seg, sel)`, `xrc_seg_fs`, `xrc_seg_gs` | A segment register was loaded (`seg` 4 is fs, 5 is gs), and the per-thread base `fs:` / `gs:` accesses add: for a Windows program, `xrc_seg_fs` is the thread's TEB (a runtime may define `XRC_SEG_FS` / `XRC_SEG_GS` itself instead). |

Some builds need more:

| Build | Symbols |
|---|---|
| `-DXRC_NO_IC` | `xrc_dispatch` in place of the inline cache |
| `XRC_FL_VERIFY=1` translations | `xrc_flbad` |
| `-DXRC_ABI_CHECK` | `xrc_abi_violation` |
| a spin-wait loop in the program | `xrc_spin_yield` |
| `-DXRC_DEEP_STACKS`, `-DXRC_COUNT_CALLS` | `xrc_tstacks`, `xrc_count_reg` |
| a Windows host | `xrc_tls_slot` / `xrc_tls_new` |

The dispatch table (`xrc_table.c`) includes `xrc_rt.h`, the runtime's own header, for three types: `XrcFn`,
`XrcOverride` and `XrcRange`. `examples/linux-i386/runtime/xrc_rt.h` is the minimal version.

Beyond those, a runtime implements the program's imports and loads the executable into guest memory:
- **Imports** are `kernel32`, `user32`, `d3d9` and the rest for a Windows program, or libc and system calls for a Linux
  one, implemented on the host OS.
- **A PIE** is loaded at `xrc_load_base`, with the same relocations xrc applied.

The FINAL FANTASY XIII port's runtime does all of that for that game. It is a separate project and is not included
here. `include/xrc.h` is a copy of the header that runtime uses, with the additions for other formats.

## Tutorial: overrides for your own executable

An **override** is a C function you write that runs in place of one guest function. You need one when xrc cannot
translate a function, and you may want one when a function is better done natively: a C library routine, a hot math
kernel, a busy-wait loop. This tutorial goes from a new executable to working overrides. `MyGame` stands for your
program throughout.

### Step 1: make the game folder

```
overrides/MyGame/
├── data/
│   └── function_map.csv (not needed if the executable has symbols)
├── overrides.tsv        (empty at first)
└── xrc.cfg              (optional, empty at first)
```

**`function_map.csv`** is only needed for an executable without symbols: most Windows programs, stripped Linux ones,
and raw images.
1. Load the unpacked executable into Ghidra (or another disassembler) and let it analyze.
2. Export every function's address, name and size as `address,name,size` lines under a header line (see
   [the format](#datafunction_mapcsv)). Names are only labels, except the C runtime x87 helpers listed there.
3. Make sure every size is right. A map that misses functions is fine, since xrc finds the ones that calls and
   pointers reach, but a wrong size is not. Correct one with `data/size_overrides.txt` instead of editing the export.

For an ELF with symbols, `readelf -sW` gives the same thing:

```sh
{ echo address,name,size; readelf -sW prog | awk '$4 == "FUNC" && $3 > 0 {printf "%s,%s,%d\n", $2, $8, $3}'; } > function_map.csv
```

### Step 2: find what needs an override

Run a coverage pass. Without `--out`, xrc writes nothing but the report:

```sh
xrc --exe MyGame.exe --override MyGame --all --report cov.tsv
awk -F'\t' '$2 == "override"' cov.tsv
```

Each line printed is a function xrc cannot translate, with the first instruction or construct it stopped at in the
`reason` column. The summary's last line counts them: `... N need an override`. The usual causes are:
- an x87 stack whose depth differs between paths through the function (C runtime math internals);
- an instruction xrc does not implement yet (the report names it);
- hand-written assembly that jumps into the middle of other functions.

For each one, decide:

| Situation | Kind |
|---|---|
| You will write the whole function in C. | `manual` |
| The function is dead, or only called from functions you replace. | `unreachable` |
| The translation works, but you want C for some cases and the translation for the rest. | `wrapped` |
| The translation works, and you want translated callers to call your faster C version directly. | `fast` |

A C runtime routine that the rest of the program calls through one entry point (`sin`, `_ftol2`, `qsort`) is best
replaced at that entry point: its internal helpers then become `unreachable`.

### Step 3: list it in `overrides.tsv`

```
# address	kind	name / why
00451a20	manual	_ftol2 (ovr_crt.c)
00451a7c	unreachable	_ftol2's x87 helper, only called by _ftol2
```

Tab-separated, one function per line, the address in hex as in the function map. Run the coverage pass again: the
function now counts as `manual` or `unreachable` instead of `override`, and for `manual` and `unreachable` xrc writes
no translation.

### Step 4: write the routine

Put it in any `.c` file in the game folder (e.g. `overrides/MyGame/ovr_crt.c`). xrc copies it to `<out>/overrides/`
and your build compiles it with the runtime and the pages.

An override has the same signature and protocol as a translation:

```c
#include "xrc.h"

void XRC_CC my_ftol2(XCtx *c);
```

**On entry**, `c` holds the guest registers as the call left them:
- `c->esp` points at the return address, so the first stack argument is `M32(c->esp + 4)`, the second
  `M32(c->esp + 8)`, and so on.
- Register arguments are in `c->ecx` (`this` for `thiscall`, the first argument for `fastcall`), `c->edx`, and so on.
- x87 arguments are in `c->fa[0]` (st0) and `c->fa[1]` (st1); SSE arguments in `c->x[0]` ... `c->x[7]`.

**Before it returns**, the routine must:
1. **Put the result where the original does**: `c->eax` (and `c->edx` for 64-bit integers), `c->st0` for an x87
   float result, `c->x[0]` for SSE.
2. **Do the return**: pop the return address into `c->eip`. A `cdecl` function pops only that. A `stdcall` /
   `thiscall` function also pops its N stack arguments, like `ret 4*N`:

   ```c
   c->eip = M32(c->esp);
   c->esp += 4;          /* cdecl */
   c->esp += 4 + 4 * N;  /* stdcall / thiscall with N stack arguments */
   ```

   The FINAL FANTASY XIII runtime has helpers for exactly this, `xrc_cdecl_ret(c)` and `xrc_stdcall_ret(c, N)`.
3. **Keep ebx, esi, edi and ebp** as they were (the callee-saved registers of the x86 calling conventions, Windows'
   and the System V i386 ABI's alike). The others
   (eax, ecx, edx, the flags) may be changed.

Guest memory is reached with the same macros as the translations: `M8` / `M16` / `M32` / `M64` take a guest address.
A guest pointer is a `u32`; the host pointer is `xrc_mem + addr` (`XM + addr` in `xrc.h`).

A complete example, `_ftol2` (st0 truncated to a 64-bit integer in edx:eax, `cdecl`):

```c
#include "xrc.h"

void XRC_CC my_ftol2(XCtx *c) {
	long long v = (long long)c->fa[0];
	c->eax = (u32)v;
	c->edx = (u32)((unsigned long long)v >> 32);
	c->eip = M32(c->esp);
	c->esp += 4;
}
```

And a `thiscall` method with one stack argument, `int Buffer::Get(int i)` returning `this->data[i]`:

```c
void XRC_CC my_buffer_get(XCtx *c) {
	u32 self = c->ecx, i = M32(c->esp + 4);
	c->eax = M32(M32(self + 0x10) + 4 * i);   /* this->data at offset 0x10 */
	c->eip = M32(c->esp);
	c->esp += 8;                              /* return address + 1 argument */
}
```

Read the original's disassembly to get the convention, the arguments and the result right. Its `ret` tells you how
many bytes it pops.

### Step 5: tell the runtime

xrc writes every overridden function into the dispatch table's `xrc_overrides[]` (address and why), so the runtime
knows these addresses have no translation. Binding an address to your routine is the runtime's job. The FINAL FANTASY
XIII runtime takes one table per file, which it registers at start-up:

```c
const XrcManual xrc_manual_crt[] = {
	{0x00451a20u, my_ftol2, "_ftol2"},
};
const unsigned xrc_manual_crt_n = sizeof xrc_manual_crt / sizeof xrc_manual_crt[0];
```

A call to a `manual` function always goes through the dispatcher (translations never call it directly), and the
dispatcher finds your routine there. If an `unreachable` function is ever dispatched after all, the runtime stops and
names it, with the reason from `overrides.tsv`.

### `wrapped`: C for some calls, the translation for the rest

The function is still translated, but every call to it goes through the dispatcher, which runs your routine. For the
cases you do not handle, call the translation with the context unchanged: it does its own return.

```c
void XRC_CC my_qsort(XCtx *c) {
	static XFn orig;
	u32 base = M32(c->esp + 4), num = M32(c->esp + 8), width = M32(c->esp + 12), cmp = M32(c->esp + 16);
	if (!orig) orig = xrc_translation_of(0x00452000u);   /* runtime: the translation's xf_ function */
	if (cmp == 0x00401230u && width == 4) {              /* a comparator you know: sort natively */
		sort_u32_natively(xrc_mem + base, num);
		c->eip = M32(c->esp);
		c->esp += 4;
		return;
	}
	orig(c);                                             /* anything else: the original's way */
}
```

`xrc_translation_of` is the FINAL FANTASY XIII runtime's lookup in `xrc_fns`. Any runtime can offer the same, or the
routine can name the translation's `xf_<addr>` directly. This is also the way to check a replacement: run both on a
copy of the data and compare, as `ovr_crt_sort.c` does under `XRC_QSORT_CHECK=1`.

### `fast`: a native version translated callers call directly

The routine must be named `xo_<addr>` (eight lowercase hex digits) and exported, because the translations call it by
that name:

```c
void XRC_CC xo_004559b0(XCtx *c) { ... }
```

The translation stays, as the reference to compare against, and the dispatcher should map the address to `xo_` too.
**Keep every register the original keeps.** xrc decides which registers a caller reloads after the call by analysing
the original's code, so your version may only change the registers the original changes. When unsure, change only
eax, ecx and edx.

### Step 6: check it

- Run the coverage pass again: the summary should say `0 need an override`.
- Rebuild and run. For a replacement of a function that also has a translation (`wrapped`, `fast`), a check mode that
  runs both and compares the results finds mistakes early. `ovr_simd.c` in the FINAL FANTASY XIII folder runs every
  call both ways when its check mode is on.
- A wrong return (esp popped by the wrong amount) shows up as the runtime's `xrc_bad_return` error at the caller, or as
  a crash soon after: check the original's `ret N`.

The FINAL FANTASY XIII folder has working examples of every kind:

| File | What it shows |
|---|---|
| `ovr_crt_math.c` | `manual`: the C runtime's x87 math entry points, done with the host's libm |
| `ovr_crt_sort.c` | `wrapped`: qsort done natively for known comparators, the translation for the rest, and a check mode |
| `ovr_simd.c` | `fast`: vector math kernels as SSE intrinsics, with a check mode against the translations |
| `ovr_game.c` | `wrapped`: the game's frame pacing replaced |

## How the translation works

**Decoding.** `src/rxdecode.odin` decodes with Odin's `core:rexcode` (i386 mode), with a fast path for one
instruction at a time. It also walks jump tables. An indirect `jmp [reg*4 + table]` with a bounds check before it
becomes a C `switch` over the table's targets, which are read from the file at translation time. The byte-indexed
two-level tables MSVC emits are handled too.

**Registers and memory.** The eight registers are C locals (`r_eax` ... `r_edi`, `r_esp`) for the whole function.
They are loaded from `XCtx` on entry and stored back only where needed: calls, returns, and anything opaque. Guest
memory is `M32(addr)` and similar: an unaligned load or store at `xrc_mem + addr`.

**Flags.** `src/xrc_flags.odin` plans the flags per function:
- A flag nothing reads is never computed.
- A `cmp`/`test` followed by its conditional jump becomes one C comparison (`if ((s32)a < (s32)b)`).
- The few readers that are left (`adc`, `pushf`, flags live across calls) use a lazy flags record.

**x87.** `src/xrc_fp.odin` tracks the x87 register stack's depth statically through each function. Each stack slot
becomes a C `double` local, so `fld`/`fadd`/`fstp` are plain C arithmetic.
- Functions that take or return x87 values are found by flow analysis, or given in `xrc.cfg`.
- Precision is double, not 80-bit.

**SSE/SSE2/MMX.** These map to SSE intrinsics (`emmintrin.h`). On non-x86 hosts `xrc.h` provides them through the
compiler's vector extensions, so the same C runs on ARM64.

**Calls.**
- A direct call to a known function becomes a direct C call (`XCALLD*`) with the protocol's return-address check.
- Indirect calls go through the dispatcher with a per-site inline cache.
- Calls through the import table go to the runtime's import implementation.
- A return checks that it returns where the caller expects, so code that manipulates return addresses still works
  (the runtime falls back to the dispatcher).

**Interrupts and threads.** The generated code polls a flag at loop back-edges and calls, so the runtime can switch
guest threads cooperatively (a global lock, the "GIL") without OS signals.

**The translator itself** runs on every core: entry discovery, the callee-saved register scan, the x87 analysis, the
inlining plan and the translation (a 64 KB page per job) are split across threads, and the results are put together
in address order, so the output is byte-identical to a one-thread run (`XRC_THREADS=1`). The x87 analysis depends on
the order of recursion, so the functions that reach a call cycle keep the one-thread order. FINAL FANTASY XIII (55k
functions) translates in about 6 s on 12 threads, from 28 s on one.

## Optimizations

All of these are on by default unless noted, and each has a switch to turn it off for comparison
([Environment variables](#environment-variables)).

- **Flag liveness and compare-branch fusion** (`src/xrc_flags.odin`). Described above. This is the largest single
  win.
- **Callee-saved register plan** (`x_abi_*` in `src/xrc.odin`). xrc proves which of ebx, esi, edi and ebp each
  function keeps, from its own instructions and, transitively, its callees. For FINAL FANTASY XIII, 49,082 of 55,770
  functions keep all four. A direct call to such a function does not reload them afterwards. The runtime can check
  the claim (`xrc_abi_violation`).
- **Register-sync trimming** (`src/xrc_sync.odin`). At each call and return, only the registers the callee may read
  are stored, and only those it may change are reloaded. A liveness analysis over the function decides this. For
  FINAL FANTASY XIII, 46% of the stores and 59% of the loads around calls are left out.
- **Inlining.** A function of up to 24 instructions (64 for functions on the hot list) is pasted into each place that
  calls it directly, instead of a call that saves and reloads registers.
  - Its labels get a per-site suffix, its flag variables are renamed, and its SSE registers become the site's own.
  - Its returns check the popped return address, as a real call does.
  - Only one level: calls inside an inlined body stay calls.
  - x87 and MMX code, stack slots and functions that never return are not inlined.
- **Devirtualization.** An indirect call site (a virtual method or function pointer) gets a guarded fast path if a
  profile saw at least 90% of its calls go to one function, or 95% to two. `if (target == F)` runs F's inlined body
  or a direct call, and anything else takes the normal dispatch. The guard compares the real target, so a wrong
  profile only costs speed.

Measured on FINAL FANTASY XIII (headless, main-thread CPU per frame, Clang without PGO), inlining and
devirtualization together took the field from 7.40 to 6.88 ms and battle from 7.10 to about 6.55 ms. The binary grew
12%.

## Profiles for inlining and devirtualization

`data/devirt_sites.txt` and `data/inline_hot.txt` come from running the program with a runtime that counts calls.
- `devirt_sites.txt`: lines of `calls return-address target`, one per indirect call site and target.
- `inline_hot.txt`: one hex address per line, the functions called most often.

Profiles of several runs can be concatenated, and counts for the same site are summed.

The FINAL FANTASY XIII port's runtime writes them in a build made with `-DXRC_COUNT_CALLS`, run with:
- `XRC_SITE_FILE=FILE` for the sites;
- `XRC_COUNT_FILE=FILE` for the functions;
- `XRC_PROFILE_DELAY=s` to start counting after that many seconds.

The shipped profiles cover that game's Chapter 4 field and Chapter 1 opening battle. `inline_hot.txt` lists the
functions called over about 10,000 times a second in either scene.

Without profiles, xrc still inlines small functions. It just has no hot list and devirtualizes nothing.

## The LLVM IR backend (experimental)

`--llvm` (`src/xrc_ir.odin`) writes `page_XXXX.ll` beside each `page_XXXX.c`.
- Each function goes into the `.ll` file instead of the C file, with the same `xf_<addr>(XCtx*)` symbol and calling
  protocol. The dispatch table, the C pages and the runtime link to it unchanged.
- At the end, xrc prints how many functions became IR, and the most common reasons the rest stayed C.

How it works:
- **CFG.** Blocks start at the entry, at every branch target, and after every branch, return or call.
- **SSA.** The registers and the five flags (ZF SF CF OF PF) are SSA values, joined by phis at block starts.
- **Flags** are computed eagerly as `i1` values. LLVM deletes the unused ones and folds `cmp` + `jcc` into one
  compare.
- **`XCtx` is `noalias`.** No `XCtx` ever lives in guest memory, so LLVM can keep its fields in registers across
  guest stores.
- **Calls** follow the C path's protocol. Indirect calls use an inline cache through `xrc_ir_icall`, which the
  runtime provides.
- **The C path's inlining set** is inlined too, as `alwaysinline` copies, along with profiled devirtualization.

Coverage on FINAL FANTASY XIII is every function. Compile the `.ll` files with `clang -O2` and link them with the C
pages. The backend works but is not what the port ships, which uses the C output.

## Repository layout

```
xrc/
├── README.md
├── build.bat, build.sh       build xrc with Odin
├── patches/
│   └── odin-rexcode-i386.patch   the i386 decoder fixes xrc needs (see Building)
├── include/
│   └── xrc.h                 what the generated C compiles against
├── src/
│   ├── main.odin             command line
│   ├── config.odin           xrc.cfg
│   ├── data.odin             the PE / ELF / Mach-O / raw loaders, PIE relocation, the tables
│   ├── overrides.odin        --override: overrides.tsv and copying the routines
│   ├── rxdecode.odin         decoding (rexcode), jump tables
│   ├── xrc.odin              the translator: functions, calls, the ABI plan, inlining, devirtualization,
│   │                         stack slots, pages, the dispatch table
│   ├── xrc_flags.odin        flag liveness and compare-branch fusion
│   ├── xrc_rdfold.odin       read-only constant folding
│   ├── xrc_sync.odin         register-sync trimming
│   ├── xrc_fp.odin           x87 and SSE
│   └── xrc_ir.odin           the LLVM IR backend
├── examples/
│   └── linux-i386/           a Linux program translated and run (see Quick start)
│       ├── demo.c, lib.c     the program and its shared library
│       ├── run.sh            build, translate, compile, run, compare
│       └── runtime/          a minimal runtime (runtime.c, xrc_rt.h)
└── overrides/
    └── Final Fantasy XIII/   the example game folder (see The game folder)
```

## License

xrc is released into the **public domain** under [The Unlicense](https://unlicense.org/) (see `LICENSE`): no
copyright, no copyleft, no conditions. Use, change, sell or redistribute it, with or without credit. That covers
everything in this repository, the FINAL FANTASY XIII folder included.

The C that xrc generates from an executable is a translation of that executable, so its copyright is the
executable's, not xrc's. `patches/odin-rexcode-i386.patch` changes files of the Odin compiler, which has its own
license.

## Copyright and game files

This repository contains no game files and no code from any game:
- **The translator** (`src/`), **the header** (`include/`) and **the patch** are original work.
- **The FINAL FANTASY XIII folder** holds facts about the game's executable, not its code: function addresses, sizes
  and Ghidra's library-signature names in `function_map.csv`; call counts and addresses in the profiles; and a
  handful of addresses in `xrc.cfg`.
- **Its `ovr_*.c` routines** are hand-written C that does what the corresponding functions of the game do, written
  from their behavior. For example, its sort follows the game's own sort step for step, so that equal elements come
  out in the same order.

To use xrc on FINAL FANTASY XIII you need your own copy of the game: xrc reads its executable from your installation
and writes the translation on your machine.

FINAL FANTASY XIII is a registered trademark of Square Enix Holdings Co., Ltd. This project is not affiliated with or
endorsed by Square Enix.
