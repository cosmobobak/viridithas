EXE = Viridithas
LXE = viridithas
VERSION = ???
TMPDIR := $(CURDIR)/tmp

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
    RMFILE := rm -f
    RMDIR := rm -rf
    MKDIR := mkdir -p
    NAME := $(EXE)
endif

V1NAME := $(LXE)-$(VERSION)-$(INF)-x86_64-v1$(EXT)
V2NAME := $(LXE)-$(VERSION)-$(INF)-x86_64-v2$(EXT)
V3NAME := $(LXE)-$(VERSION)-$(INF)-x86_64-v3$(EXT)
V4NAME := $(LXE)-$(VERSION)-$(INF)-x86_64-v4$(EXT)

openbench:
	cargo rustc -r -- -C target-cpu=native --emit link=$(NAME)

tmp-dir:
	$(MKDIR) $(TMPDIR)

x86-64 x86-64-v2 x86-64-v3 x86-64-v4 native: tmp-dir
	cargo rustc -r --features final-release -- -C target-feature=+crt-static -C target-cpu=$@ -C profile-generate=$(TMPDIR) --emit link=$(LXE)-$(VERSION)-$(INF)-$@$(EXT)
	./$(LXE)-$(VERSION)-$(INF)-$@$(EXT) bench
	llvm-profdata merge -o $(TMPDIR)/merged.profdata $(TMPDIR)/*.profraw

	cargo rustc -r --features final-release -- -C target-feature=+crt-static -C target-cpu=$@ -C profile-use=$(TMPDIR)/merged.profdata --emit link=$(LXE)-$(VERSION)-$(INF)-$@$(EXT)

	$(RMDIR) $(TMPDIR)
	$(RMFILE) *.pdb

aarch64-apple: tmp-dir
	cargo rustc -r --target=aarch64-apple-darwin --features final-release -- -C target-feature=+crt-static -C profile-generate=$(TMPDIR) --emit link=$(LXE)-$(VERSION)-macos-aarch64
	./$(LXE)-$(VERSION)-macos-aarch64 bench
	llvm-profdata merge -o $(TMPDIR)/merged.profdata $(TMPDIR)/*.profraw

	cargo rustc -r --target=aarch64-apple-darwin --features final-release -- -C target-feature=+crt-static -C profile-use=$(TMPDIR)/merged.profdata --emit link=$(LXE)-$(VERSION)-macos-aarch64

	$(RMDIR) $(TMPDIR)

aarch64-android: tmp-dir
	cargo rustc -r --target=aarch64-linux-android --features final-release -- -C target-feature=+crt-static -C profile-generate=$(TMPDIR) --emit link=$(LXE)-$(VERSION)-android-aarch64
	./$(LXE)-$(VERSION)-android-aarch64 bench
	llvm-profdata merge -o $(TMPDIR)/merged.profdata $(TMPDIR)/*.profraw

	cargo rustc -r --target=aarch64-linux-android --features final-release -- -C target-feature=+crt-static -C profile-use=$(TMPDIR)/merged.profdata --emit link=$(LXE)-$(VERSION)-android-aarch64

	$(RMDIR) $(TMPDIR)

x86-64-datagen x86-64-v2-datagen x86-64-v3-datagen x86-64-v4-datagen native-datagen: tmp-dir
	cargo rustc -r --features datagen -- -C target-feature=+crt-static -C target-cpu=$(subst -datagen,,$@) -C profile-generate=$(TMPDIR) --emit link=$(LXE)-$(VERSION)-$(INF)-$(subst -datagen,,$@)$(EXT)
	./$(LXE)-$(VERSION)-$(INF)-$(subst -datagen,,$@)$(EXT) bench
	llvm-profdata merge -o $(TMPDIR)/merged.profdata $(TMPDIR)/*.profraw

	cargo rustc -r --features datagen -- -C target-feature=+crt-static -C target-cpu=$(subst -datagen,,$@) -C profile-use=$(TMPDIR)/merged.profdata --emit link=$(LXE)-$(VERSION)-$(INF)-$(subst -datagen,,$@)$(EXT)

	$(RMDIR) $(TMPDIR)
	$(RMFILE) *.pdb

bench:
	cargo rustc --release -- -C target-cpu=native --emit link=$(NAME)
	target/release/$(NAME) bench
