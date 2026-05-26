# Build for AMX bf16 GEMM benchmark on Sapphire Rapids.
# Use gcc-toolset/13.
CC      := gcc
CFLAGS  ?= -O3 -std=gnu11 -Wall -Wextra
ARCHFLAGS ?= -march=sapphirerapids -mtune=sapphirerapids -mamx-tile -mamx-bf16 -mamx-int8 -mavx512bf16
EXTRA   ?=
MKLROOT ?= /home/app/inteloneapi/2025.1.3/mkl/latest
MKL_INC ?= -I$(MKLROOT)/include
MKL_LIB ?= -L$(MKLROOT)/lib/intel64 -Wl,-rpath,$(MKLROOT)/lib/intel64 \
           -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
MKL_DEF ?= -DUSE_MKL -DMKL_ILP64

SRCDIR := src
BUILD  := build

SRCS := $(SRCDIR)/main.c $(SRCDIR)/naive_gemm.c $(SRCDIR)/avx512_gemm.c $(SRCDIR)/amx_gemm.c $(SRCDIR)/mkl_gemm.c
OBJS := $(patsubst $(SRCDIR)/%.c,$(BUILD)/%.o,$(SRCS))

TARGET := bench

.PHONY: all clean

all: $(TARGET)

$(BUILD)/%.o: $(SRCDIR)/%.c | $(BUILD)
	$(CC) $(CFLAGS) $(ARCHFLAGS) $(EXTRA) $(MKL_DEF) $(MKL_INC) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(ARCHFLAGS) $(EXTRA) $(OBJS) $(MKL_LIB) -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD) $(TARGET)
