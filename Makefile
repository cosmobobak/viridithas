EXE    	= viridithas
ifeq ($(OS),Windows_NT)
	NAME := $(EXE).exe
else
	NAME := $(EXE)
endif

rule:
	RUSTFLAGS="-C target-cpu=native" cargo build --release
	cp target/release/$(NAME) $(NAME)