EXE = Viridithas
LXE = viridithas
VERSION = ???
TMPDIR := tmp

ifeq ($(OS),Windows_NT)
    INF := win
    EXT := .exe
    RMFILE := del
    RMDIR := rmdir /s /q
    MKDIR := mkdir
    NAME := $(EXE).exe
else
    INF := linux
    EXT :=
    RMFILE := rm
    RMDIR := rm -rf
    MKDIR := mkdir -p
    NAME := $(EXE)
endif

V1NAME := $(LXE)-$(VERSION)-$(INF)-x86_64-v1$(EXT)
V2NAME := $(LXE)-$(VERSION)-$(INF)-x86_64-v2$(EXT)
V3NAME := $(LXE)-$(VERSION)-$(INF)-x86_64-v3$(EXT)
V4NAME := $(LXE)-$(VERSION)-$(INF)-x86_64-v4$(EXT)

openbench:
	cargo rustc --release -- -C target-cpu=native --emit link=$(NAME)

final-release:
	cargo rustc --release --features final-release -- -C target-feature=+crt-static -C target-cpu=x86-64 --emit link=$(V1NAME)
	cargo rustc --release --features final-release -- -C target-feature=+crt-static -C target-cpu=x86-64-v2 --emit link=$(V2NAME)
	cargo rustc --release --features final-release -- -C target-feature=+crt-static -C target-cpu=x86-64-v3 --emit link=$(V3NAME)
	cargo rustc --release --features final-release -- -C target-feature=+crt-static -C target-cpu=x86-64-v4 --emit link=$(V4NAME)

release:
	cargo rustc --release --features syzygy,bindgen -- -C target-feature=+crt-static -C target-cpu=x86-64 --emit link=$(V1NAME)
	cargo rustc --release --features syzygy,bindgen -- -C target-feature=+crt-static -C target-cpu=x86-64-v2 --emit link=$(V2NAME)
	cargo rustc --release --features syzygy,bindgen -- -C target-feature=+crt-static -C target-cpu=x86-64-v3 --emit link=$(V3NAME)
	cargo rustc --release --features syzygy,bindgen -- -C target-feature=+crt-static -C target-cpu=x86-64-v4 --emit link=$(V4NAME)

datagen:
	cargo rustc --release --features syzygy,bindgen,datagen -- -C target-cpu=native

tmp-dir:
	mkdir $(TMPDIR)

x86-64 x86-64-v2 x86-64-v3 x86-64-v4 native: tmp-dir
	cargo rustc -r --features final-release -- -C target-feature=+crt-static -C target-cpu=$@ -C profile-generate=$(TMPDIR) --emit link=$(LXE)-$(VERSION)-$(INF)-$@$(EXT)
	./$(LXE)-$(VERSION)-$(INF)-$@$(EXT) bench
	llvm-profdata merge -o $(TMPDIR)/merged.profdata $(TMPDIR)

	cargo rustc -r --features final-release -- -C target-feature=+crt-static -C target-cpu=$@ -C profile-use=$(TMPDIR)/merged.profdata --emit link=$(LXE)-$(VERSION)-$(INF)-$@$(EXT)

	$(RMDIR) $(TMPDIR)
	$(RMFILE) *.pdb

bench:
	cargo rustc --release -- -C target-cpu=native --emit link=$(NAME)
	target/release/$(NAME) bench
