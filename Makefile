EXE = Viridithas
ifeq ($(OS),Windows_NT)
	NAME := $(EXE).exe
else            
	NAME := $(EXE)
endif

rule:
	cargo rustc --release -- -C target-cpu=native --emit link=$(NAME)

release:
	cargo rustc --release -- -C target-feature=+crt-static,+fxsr,+sse,+sse2 --emit link=viridithas-x86_64-v1
	cargo rustc --release -- -C target-feature=+crt-static,+fxsr,+sse,+sse2,+cmpxchg16b,+popcnt,+sse3,+sse4.1,+sse4.2,+ssse3 --emit link=viridithas-x86_64-v2
	cargo rustc --release -- -C target-feature=+crt-static,+fxsr,+sse,+sse2,+cmpxchg16b,+popcnt,+sse3,+sse4.1,+sse4.2,+ssse3,+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+lzcnt,+movbe --emit link=viridithas-x86_64-v3