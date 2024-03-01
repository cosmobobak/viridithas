EXE = Viridithas
LXE = viridithas
VERSION = ???
_THIS := $(realpath $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
TMPDIR := $(_THIS)/tmp

ifeq ($(OS),Windows_NT)
	NAME := $(EXE).exe
	V1NAME := $(LXE)-$(VERSION)-x86_64-win-v1.exe
	V2NAME := $(LXE)-$(VERSION)-x86_64-win-v2.exe
	V3NAME := $(LXE)-$(VERSION)-x86_64-win-v3.exe
	V4NAME := $(LXE)-$(VERSION)-x86_64-win-v4.exe
else
	NAME := $(EXE)
	V1NAME := $(LXE)-$(VERSION)-x86_64-linux-v1
	V2NAME := $(LXE)-$(VERSION)-x86_64-linux-v2
	V3NAME := $(LXE)-$(VERSION)-x86_64-linux-v3
	V4NAME := $(LXE)-$(VERSION)-x86_64-linux-v4
endif

openbench:
	cargo rustc --release -- -C target-cpu=native --emit link=$(NAME)

final-release:
	cargo rustc --release --features syzygy,bindgen,final-release -- -C target-feature=+crt-static -C target-cpu=x86-64 --emit link=$(V1NAME)
	cargo rustc --release --features syzygy,bindgen,final-release -- -C target-feature=+crt-static -C target-cpu=x86-64-v2 --emit link=$(V2NAME)
	cargo rustc --release --features syzygy,bindgen,final-release -- -C target-feature=+crt-static -C target-cpu=x86-64-v3 --emit link=$(V3NAME)
	cargo rustc --release --features syzygy,bindgen,final-release -- -C target-feature=+crt-static -C target-cpu=x86-64-v4 --emit link=$(V4NAME)

release:
	cargo rustc --release --features syzygy,bindgen -- -C target-feature=+crt-static -C target-cpu=x86-64 --emit link=$(V1NAME)
	cargo rustc --release --features syzygy,bindgen -- -C target-feature=+crt-static -C target-cpu=x86-64-v2 --emit link=$(V2NAME)
	cargo rustc --release --features syzygy,bindgen -- -C target-feature=+crt-static -C target-cpu=x86-64-v3 --emit link=$(V3NAME)
	cargo rustc --release --features syzygy,bindgen -- -C target-feature=+crt-static -C target-cpu=x86-64-v4 --emit link=$(V4NAME)

datagen:
	cargo rustc --release --features syzygy,bindgen,datagen -- -C target-cpu=native

tmp-dir:
	mkdir -p $(TMPDIR)

x86-64 x86-64-v2 x86-64-v3 x86-64-v4 native: tmp-dir
	cargo rustc -r --features syzygy,bindgen,final-release -- -C target-cpu=$@ -C profile-generate=$(TMPDIR) --emit link=$(LXE)-$(VER)-$@$(EXT)
	./$(LXE)-$(VER)-$@$(EXT) bench
	llvm-profdata merge -o $(TMPDIR)/merged.profdata $(TMPDIR)

	cargo rustc -r --features syzygy,bindgen,final-release -- -C target-feature=+crt-static -C target-cpu=$@ -C profile-use=$(TMPDIR)/merged.profdata --emit link=$(LXE)-$(VER)-$@$(EXT)

	rm -rf $(TMPDIR)/*
	rm -f *.pdb

bench:
	cargo rustc --release -- -C target-cpu=native --emit link=$(NAME)
	target/release/$(NAME) bench