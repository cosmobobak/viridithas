EXE = Viridithas
LXE = viridithas
ifeq ($(OS),Windows_NT)
	NAME := $(EXE).exe
	V1NAME := $(LXE)-x86_64-win-v1.exe
	V2NAME := $(LXE)-x86_64-win-v2.exe
	V3NAME := $(LXE)-x86_64-win-v3.exe
	V4NAME := $(LXE)-x86_64-win-v4.exe
else
	NAME := $(EXE)
	V1NAME := $(LXE)-x86_64-linux-v1
	V2NAME := $(LXE)-x86_64-linux-v2
	V3NAME := $(LXE)-x86_64-linux-v3
	V4NAME := $(LXE)-x86_64-linux-v4
endif

rule:
	cargo rustc --release -- -C target-cpu=native --emit link=$(NAME)

final-release:
	cargo rustc --release --features syzygy,bindgen,final-release -- -C target-feature=+crt-static -C target-cpu=x86-64 --emit link=$(V1NAME)
	cargo rustc --release --features syzygy,bindgen,final-release -- -C target-feature=+crt-static -C target-cpu=x86-64-v2 --emit link=$(V2NAME)
	cargo rustc --release --features syzygy,bindgen,final-release -- -C target-feature=+crt-static -C target-cpu=x86-64-v3 --emit link=$(V3NAME)
	cargo rustc --release --features syzygy,bindgen,final-release -- -C target-feature=+crt-static -C target-cpu=x86-64-v4 --emit link=$(V4NAME)

release:
	cargo rustc --release --features syzygy,bindgen, -- -C target-feature=+crt-static -C target-cpu=x86-64 --emit link=$(V1NAME)
	cargo rustc --release --features syzygy,bindgen, -- -C target-feature=+crt-static -C target-cpu=x86-64-v2 --emit link=$(V2NAME)
	cargo rustc --release --features syzygy,bindgen, -- -C target-feature=+crt-static -C target-cpu=x86-64-v3 --emit link=$(V3NAME)
	cargo rustc --release --features syzygy,bindgen, -- -C target-feature=+crt-static -C target-cpu=x86-64-v4 --emit link=$(V4NAME)

datagen:
	cargo rustc --release --features syzygy,bindgen,datagen -- -C target-cpu=native