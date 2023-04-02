ifeq ($(OS),Windows_NT)
    NAME := $(EXE).exe
    BIN := viridithas.exe
else
    NAME := $(EXE)
    BIN := viridithas
endif

rule:
    RUSTFLAGS="-C target-cpu=native" cargo build --release
    cp ../target/release/$(BIN) $(NAME)