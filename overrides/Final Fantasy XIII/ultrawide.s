# Ultrawide (wider than 16:9): the code patches of the 13UW mod's [Black Bars] section (patches/main.patch,
# version.dll's patch format), as xrc.cfg `patch` entries -- tools/uw_patches.sh in ff13decomp2 assembles this file
# and writes them into xrc.cfg between its "# ultrawide" markers.
#
# Every hook jumps to code here that runs the mod's version only while UW_FLAG is set (the runtime sets it, and the
# values below, when the back buffer is wider than 16:9: xrc_widescreen, src/runtime/hle_d3d9.c), and the game's own
# instructions otherwise. The sites patched in place read a value whose default is the constant they read before
# (16:9, 2.0), the same bits. The mod's graphics-settings change (00e851c0's 1280 x 720 sizes and 9/16 aspects -> the
# back buffer's) is the runtime's too, at CreateDevice. From its [FOV] section: the field camera's added degrees (its F5 / F6 / F7
# keys, the runtime's). Not taken: its battle and cutscene FOV multipliers, its resolution override and 60 Hz unlock
# (the launcher has those).
#
# The values (the end of .data's last page, past its virtual size: zeros in the image, as the mod used it):
.set UW_FLAG,   0x02871f00 # 1: ultrawide, 2: narrower than 16:9 (16:10; the minimap as at 16:9)
.set UW_ASP,    0x02871f04 # the back buffer's width / height
.set UW_ASP16,  0x02871f08 # UW_ASP while UW_FLAG, else 16/9 (010cb154's bits)
.set UW_UIDIV,  0x02871f0c # 2 x aspect / 1.80307 (the mod's 3fe6cab7) while UW_FLAG, else 2.0 (0119c674's)
.set UW_CANVAS, 0x02871f10 # 720 x aspect while UW_FLAG, else 1280.0
.set UW_MM,     0x02871f14 # 4/3 while UW_FLAG, else 1.0
.set UW_SCRH,   0x02871f18 # the back buffer's height (int)
.set UW_FOVADD, 0x02871f1c # degrees added to the field camera's fov (F5 / F6 / F7: vk_d3d9.c fov_key; 0 = none)
.set UW_BBW,    0x02871f20 # the back buffer's width, height (any aspect; 0 before CreateDevice)
.set UW_BBH,    0x02871f24
.set UW_VERT,   0x02871f28 # narrower than 16:9 (16:10): (16/9) / aspect, the factor on tan(fov / 2) that keeps 16:9's width (0 = off)
.set UW_CANS,   0x02871f2c # 2D canvas: its height / 720 (16:10: 800 / 720; else 1.0)
.set UW_CANY,   0x02871f30 # 2D canvas: its top above the UI's (16:10: 40.0; else 0)
.set UW_CANH,   0x02871f34 # 2D canvas: its height (16:10: 800.0; else 720.0)
.set UW_LBY,    0x02871f38 # 16:10: the movie quad's height, aspect x 9 / 16 (letterboxed)
.set UW_GMW,    0x02871f3c # the GraphicsManager's own width, height (the back buffer's, while UW_FLAG; c_905ba8 sets its +0x20 / +0x24 to 1280 x 720)
.set UW_GMH,    0x02871f40
# the game's constants
.set F_1280,    0x0114a9e4 # 1280.0
.set F_HALF,    0x0109b130 # 0.5
.set F_16_9,    0x010cb154 # 16/9
.set F_360,     0x010a76ec # 360.0
.set F_ONE,     0x010aa950 # 1.0

.intel_syntax noprefix

# a hook: a jmp at SITE to CAVE (the instruction it replaces is in the cave, as the game's path)
.macro HOOK site, cave
.section .h_\site,"ax"
	jmp \cave
.endm
# an instruction changed in place
.macro AT site
.section .h_\site,"ax"
.endm

# 009460b0 (the UI's construction): laid out at 1280 x 720, not at the GraphicsManager's size (00905f70 / 00905f90)
HOOK 009461aa, c_9461aa
HOOK 009461cb, c_9461cb
# 00629e00: the aspect it hands 00629dc0 the screen's, not [[this+0xbc]+0x70]
HOOK 00629e18, c_629e18
# the cameras' default aspect (010cb154, 16:9): 007bebc0, 00844bf0, the Vfx BaseCamera 00ce2c80 / 00ce35f0
AT 007bec5b
	movss xmm0, dword ptr [UW_ASP16]
AT 00844c75
	mulss xmm0, dword ptr [UW_ASP16]
AT 00ce2c8b
	movss xmm2, dword ptr [UW_ASP16]
AT 00ce3663
	movss xmm0, dword ptr [UW_ASP16]
# 00905b80 (the UI camera): the GraphicsManager's size back to 1280 x 720, its fov halved by UW_UIDIV
HOOK 00905ba8, c_905ba8
AT 00905bad
	divss xmm0, dword ptr [UW_UIDIV]
# 00a05290 (the minimap)
HOOK 00a0548d, c_a0548d
HOOK 00a054ef, c_a054ef
HOOK 00a0567c, c_a0567c
# 006b84e0: the 2D canvas's width 720 x aspect, its x moved by half the extra width
HOOK 006b84f7, c_6b84f7
HOOK 006b865f, c_6b865f
# (16:10: its y on the taller canvas, the UI's 720 in its middle: y x UW_CANS - UW_CANY)
HOOK 006b8694, c_6b8694
# 00ce2c80 (the Vfx BaseCamera's defaults): its centre x and width from the canvas
HOOK 00ce2c83, c_ce2c83
HOOK 00ce2d0c, c_ce2d0c
# (its height: UW_CANH)
# 00fee4e0 (a size's width / height): the screen's aspect
HOOK 00fee4f8, c_fee4f8
# 00a68b70: the scissor it sets (00eabfe0: x, y, w, h) from x 1, 8192 wide
HOOK 00a68ebb, c_a68ebb
# 0094c890 (the battle's enemy intel): the Notes line's scissor, 00906240(x, y, w, h), in the UI's 1280 x 720 pixels
# (0, 496, 00905f70's 1280, (lines + 8) x 5) whatever the back buffer -- the retail game's bug: the line clipped away
# above 720p. Not the mod's: scaled to the back buffer, y and h by its height / 720, x and w by its width / 1280
# (ultrawide: 0 and its width). With no back buffer size yet (UW_BBH 0): as it was
HOOK 0094c934, c_94c934
# 0062c000 (the field camera's fov, returned on the x87 stack): UW_FOVADD degrees more
HOOK 0062c03a, c_62c03a
# 006b65f0 (a camera's render update: its projection from 007bc210, the camera's fov -- vertical, radians: cot(fov / 2) is the
# matrix's y scale): with UW_VERT the fov it reads widened to 2 atan(tan(fov / 2) x UW_VERT) -- the picture keeps 16:9's width
# and shows more above and below (Vert+). Its four reads, so the change it compares with its last one (006b7138) is
# like for like; the camera's own value is unchanged (the game reads it elsewhere)
AT 006b6758
	call c_fovv
AT 006b7138
	call c_fovv
AT 006b733c
	call c_fovv
AT 006b738f
	call c_fovv
# 01008e00 (the movie's quad: its Y, U, V textures, 01008dd5): ultrawide, a centred 16:9 quad; 16:10, full width and UW_LBY
# high (letterboxed)
HOOK 01008e5f, c_1008e5f
HOOK 01008ee8, c_1008ee8
HOOK 01008f33, c_1008f33
HOOK 01008fb6, c_1008fb6
# 008fab80 (the GUI filter -- the battle transition: its render targets, from the pool by the GraphicsManager's
# size, / 2 or / 4): the pool's were made at the GraphicsManager's own size, so the sizes asked for x UW_GMW / its
# width (1280 after c_905ba8), the heights the same
HOOK 008fad4d, c_8fad4d

.section .cave,"ax"
c_8fad4d:
	cmp dword ptr [UW_FLAG], 0
	je 1f
	cmp dword ptr [UW_GMW], 0
	je 1f
	mov edx, dword ptr [0x0283e36c]
	mov ecx, dword ptr [edx + 0x20]
	test ecx, ecx
	jz 1f
	mov eax, dword ptr [ebp - 0x18]
	imul eax, dword ptr [UW_GMW]
	cdq
	idiv ecx
	mov dword ptr [ebp - 0x18], eax
	mov edx, dword ptr [0x0283e36c]
	mov ecx, dword ptr [edx + 0x24]
	test ecx, ecx
	jz 1f
	mov eax, dword ptr [ebp - 0x1c]
	imul eax, dword ptr [UW_GMH]
	cdq
	idiv ecx
	mov dword ptr [ebp - 0x1c], eax
1:	mov edx, dword ptr [0x026a5480]
	jmp 0x008fad53

c_fovv:
	call 0x007bc210
	cmp dword ptr [UW_VERT], 0
	je 1f
	fmul dword ptr [F_HALF]
	fptan
	fstp st(0)
	fmul dword ptr [UW_VERT]
	fld1
	fpatan
	fadd st(0), st(0)
1:	ret

c_62c03a:
	cmp dword ptr [UW_FOVADD], 0
	je 1f
	fadd dword ptr [UW_FOVADD]
1:	mov esp, ebp
	pop ebp
	ret

c_9461aa:
	cmp dword ptr [UW_FLAG], 0
	jne 1f
	call 0x00905f70
	jmp 0x009461af
1:	mov eax, 1280
	jmp 0x009461af

c_9461cb:
	cmp dword ptr [UW_FLAG], 0
	jne 1f
	call 0x00905f90
	jmp 0x009461d0
1:	mov eax, 720
	jmp 0x009461d0

c_629e18:
	cmp dword ptr [UW_FLAG], 0
	jne 1f
	movss xmm0, dword ptr [edx + 0x70]
	movss dword ptr [ebp - 4], xmm0
	push ecx
	fld dword ptr [ebp - 4]
	jmp 0x00629e26
1:	push ecx
	fld dword ptr [UW_ASP]
	jmp 0x00629e26

# (ecx: the GraphicsManager, as 00905f90 left it)
c_905ba8:
	movss xmm0, dword ptr [ebp + 0x10]
	cmp dword ptr [UW_FLAG], 0
	je 1f
	mov dword ptr [ecx + 0x20], 1280
	mov dword ptr [ecx + 0x24], 720
1:	jmp 0x00905bad

# (x 1.0 without UW_FLAG: exact)
c_a0548d:
	fld dword ptr [esp + 0x294]
	fmul dword ptr [UW_MM]
	jmp 0x00a05494

# the minimap's x: 640 + ([edx+0x1f20] - 640) x (16/9) / aspect
c_a054ef:
	cmp dword ptr [UW_FLAG], 1
	je 1f
	movss dword ptr [esp + 0x234], xmm0
	jmp 0x00a054f8
1:	fld dword ptr [F_1280]
	fmul dword ptr [F_HALF]
	fld st(0)
	fisubr dword ptr [edx + 0x1f20]
	fld dword ptr [UW_ASP]
	fdiv dword ptr [F_16_9]
	.byte 0xde, 0xf9 # fdivp st(1), st(0): st(1) / st(0) (as bytes: GNU as swaps fdivp's operands)
	.byte 0xde, 0xc1 # faddp st(1), st(0)
	fstp dword ptr [esp + 0x234]
	jmp 0x00a054f8

c_a0567c:
	cmp dword ptr [UW_FLAG], 1
	je 1f
	fmul dword ptr [esp + 0x20c]
	jmp 0x00a05683
1:	fstp st(0)
	fld dword ptr [F_360]
	jmp 0x00a05683

c_6b84f7:
	mov ecx, dword ptr [ebp - 0x40]
	mov edx, dword ptr [ecx + 0x20]
	mov dword ptr [ebp - 0x44], edx
	cmp dword ptr [UW_FLAG], 0
	je 1f
	fld dword ptr [UW_CANVAS]
	fistp dword ptr [ebp - 0x44]
1:	jmp 0x006b8500

c_6b865f:
	mov edx, 1
	cmp dword ptr [UW_FLAG], 0
	je 1f
	fld dword ptr [F_1280]
	fsub dword ptr [UW_CANVAS]
	fmul dword ptr [F_HALF]
	fadd dword ptr [ecx]
	fstp dword ptr [ecx]
1:	jmp 0x006b8664

c_6b8694:
	mov ecx, dword ptr [ebp + 8]
	cmp dword ptr [UW_FLAG], 2
	jne 1f
	mulss xmm0, dword ptr [UW_CANS]
	subss xmm0, dword ptr [UW_CANY]
1:	movss dword ptr [ecx + 4], xmm0
	jmp 0x006b869c

# (1280 x 0.5 = 640 and 1280 without UW_FLAG: exact)
c_ce2c83:
	movss xmm1, dword ptr [UW_CANVAS]
	mulss xmm1, dword ptr [F_HALF]
	jmp 0x00ce2c8b

c_ce2d0c:
	fld dword ptr [UW_CANVAS]
	fistp dword ptr [ecx + 0xbc]
	fld dword ptr [UW_CANH]
	fistp dword ptr [ecx + 0xc0]
	jmp 0x00ce2d20

c_fee4f8:
	movss dword ptr [ebp - 4], xmm0
	cmp dword ptr [UW_FLAG], 0
	jne 1f
	fld dword ptr [ebp - 4]
	jmp 0x00fee500
1:	fld dword ptr [UW_ASP]
	jmp 0x00fee500

c_a68ebb:
	cmp dword ptr [UW_FLAG], 0
	je 1f
	mov dword ptr [esp], 1
	mov dword ptr [esp + 8], 0x2000
1:	call 0x00eabfe0
	jmp 0x00a68ec0

c_94c934:
	cmp dword ptr [UW_BBH], 0
	je 9f
	cmp dword ptr [UW_FLAG], 2
	je 3f
	mov ecx, 720
	mov eax, dword ptr [esp + 4]
	imul eax, dword ptr [UW_BBH]
	cdq
	idiv ecx
	mov dword ptr [esp + 4], eax
	mov eax, dword ptr [esp + 12]
	imul eax, dword ptr [UW_BBH]
	cdq
	idiv ecx
	mov dword ptr [esp + 12], eax
	cmp dword ptr [UW_FLAG], 0
	jne 2f
4:	mov ecx, 1280
	mov eax, dword ptr [esp]
	imul eax, dword ptr [UW_BBW]
	cdq
	idiv ecx
	mov dword ptr [esp], eax
	mov eax, dword ptr [esp + 8]
	imul eax, dword ptr [UW_BBW]
	cdq
	idiv ecx
	mov dword ptr [esp + 8], eax
	jmp 9f
2:	mov dword ptr [esp], 0
	mov eax, dword ptr [UW_BBW]
	mov dword ptr [esp + 8], eax
	jmp 9f
# (16:10: y and h from the UI's 720 in the middle of the canvas, UW_CANH high: (y + UW_CANY) x height / UW_CANH; x as at 16:9)
3:	fild dword ptr [esp + 4]
	fadd dword ptr [UW_CANY]
	fimul dword ptr [UW_BBH]
	fdiv dword ptr [UW_CANH]
	fistp dword ptr [esp + 4]
	fild dword ptr [esp + 12]
	fimul dword ptr [UW_BBH]
	fdiv dword ptr [UW_CANH]
	fistp dword ptr [esp + 12]
	jmp 4b
9:	call 0x00906240
	jmp 0x0094c939

# the quad's left edge -(16/9)/aspect (-1 - [ebp-0x6c], [ebp-0x6c] = (H x 16/9 - H x aspect) / its width), its
# right edge (16/9)/aspect ([ebp-0x44], [ebp-0x30]: their stores below skipped)
c_1008e5f:
	cmp dword ptr [UW_FLAG], 1
	je 1f
	movss xmm0, dword ptr [F_ONE]
	divss xmm0, dword ptr [ebp - 0x5c]
	movss dword ptr [ebp - 0x6c], xmm0
	jmp 0x01008e71
1:	fild dword ptr [UW_SCRH]
	fmul dword ptr [F_16_9]
	fild dword ptr [UW_SCRH]
	fmul dword ptr [UW_ASP]
	.byte 0xde, 0xe9 # fsubp st(1), st(0): st(1) - st(0)
	fstp dword ptr [ebp - 0x6c]
	movss xmm0, dword ptr [ebp - 0x6c]
	divss xmm0, dword ptr [ebp - 0x5c]
	movss dword ptr [ebp - 0x6c], xmm0
	fild dword ptr [UW_SCRH]
	fmul dword ptr [F_16_9]
	fild dword ptr [UW_SCRH]
	fmul dword ptr [UW_ASP]
	.byte 0xde, 0xf9 # fdivp st(1), st(0): st(1) / st(0)
	fst dword ptr [ebp - 0x44]
	fstp dword ptr [ebp - 0x30]
	jmp 0x01008e71

c_1008ee8:
	cmp dword ptr [UW_FLAG], 1
	je 1f
	movss dword ptr [ebp - 0x44], xmm0
1:	jmp 0x01008eed

c_1008f33:
	cmp dword ptr [UW_FLAG], 1
	je 1f
	movss dword ptr [ebp - 0x30], xmm0
1:	jmp 0x01008f38

# (the four vertices' y: -0x54 -0x40 -0x2c -0x18)
c_1008fb6:
	cmp dword ptr [UW_FLAG], 2
	jne 1f
	movss xmm0, dword ptr [ebp - 0x54]
	mulss xmm0, dword ptr [UW_LBY]
	movss dword ptr [ebp - 0x54], xmm0
	movss xmm0, dword ptr [ebp - 0x40]
	mulss xmm0, dword ptr [UW_LBY]
	movss dword ptr [ebp - 0x40], xmm0
	movss xmm0, dword ptr [ebp - 0x2c]
	mulss xmm0, dword ptr [UW_LBY]
	movss dword ptr [ebp - 0x2c], xmm0
	movss xmm0, dword ptr [ebp - 0x18]
	mulss xmm0, dword ptr [UW_LBY]
	movss dword ptr [ebp - 0x18], xmm0
1:	push ecx
	mov dword ptr [ebp - 0x9c], esp
	jmp 0x01008fbd
