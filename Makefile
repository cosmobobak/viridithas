EXE = Viridithas
ifeq ($(OS),Windows_NT)
	NAME := $(EXE).exe
	V1NAME := $(EXE)-x86_64-v1.exe
	V2NAME := $(EXE)-x86_64-v2.exe
	V3NAME := $(EXE)-x86_64-v3.exe
else
	NAME := $(EXE)
	V1NAME := $(EXE)-x86_64-v1
	V2NAME := $(EXE)-x86_64-v2
	V3NAME := $(EXE)-x86_64-v3
endif

rule:
	cargo rustc --release -- -C target-cpu=native --emit link=$(NAME)

release:
	cargo rustc --release -- -C target-feature=+crt-static,+fxsr,+sse,+sse2 --emit link=$(V1NAME)
	cargo rustc --release -- -C target-feature=+crt-static,+fxsr,+sse,+sse2,+cmpxchg16b,+popcnt,+sse3,+sse4.1,+sse4.2,+ssse3 --emit link=$(V2NAME)
	cargo rustc --release -- -C target-feature=+crt-static,+fxsr,+sse,+sse2,+cmpxchg16b,+popcnt,+sse3,+sse4.1,+sse4.2,+ssse3,+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+lzcnt,+movbe --emit link=$(V3NAME)