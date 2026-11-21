@echo off
rem Builds xrc.exe (the x86 -> C translator) from src\ with Odin on Windows. Needs an Odin compiler with the i386 fixes
rem of patches\odin-rexcode-i386.patch applied (see README.md); ODIN: its odin.exe (default: odin on PATH).
rem Run from this folder. The translator recurses deeply on large functions: it is linked with a 16 MB stack.
setlocal
if "%ODIN%"=="" set ODIN=odin
"%ODIN%" build src -out:xrc.exe -o:speed -extra-linker-flags:/STACK:16777216 %*
