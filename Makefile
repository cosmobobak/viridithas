EXE = Viridithas
LXE = viridithas
ifeq ($(OS),Windows_NT)
	NAME := $(EXE).exe
	V1NAME := $(LXE)-x86_64-win-v1.exe
	V2NAME := $(LXE)-x86_64-win-v2.exe
	V3NAME := $(LXE)-x86_64-win-v3.exe
else
	NAME := $(EXE)
	V1NAME := $(LXE)-x86_64-linux-v1
	V2NAME := $(LXE)-x86_64-linux-v2
	V3NAME := $(LXE)-x86_64-linux-v3
endif

rule:
	cargo rustc --release -- -C target-cpu=native --emit link=$(NAME)

release:
	cargo rustc --release -- -C target-feature=+crt-static,+fxsr,+sse,+sse2 --emit link=$(V1NAME)
	cargo rustc --release -- -C target-feature=+crt-static,+fxsr,+sse,+sse2,+cmpxchg16b,+popcnt,+sse3,+sse4.1,+sse4.2,+ssse3 --emit link=$(V2NAME)
	cargo rustc --release -- -C target-feature=+crt-static,+fxsr,+sse,+sse2,+cmpxchg16b,+popcnt,+sse3,+sse4.1,+sse4.2,+ssse3,+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+lzcnt,+movbe --emit link=$(V3NAME)