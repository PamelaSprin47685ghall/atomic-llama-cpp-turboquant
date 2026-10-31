# Makefile — command record for the 13022 executor (NOT run on 12022: that
# host forbids all compilation/testing; 13022 holds the serial lock and also
# forbids GPU).
#
#   BLOCKED: if 13022's serial-lock path/protocol is not confirmed by the
#   executor, NOTHING here runs (no build, no run, no ldd). The lock is
#   13022-local; all targets are CPU-only and GPU stays forbidden on both
#   hosts. The original floor1 tree is never patched in place — TREE must
#   point at an independent PATCHED COPY (copy the pristine tree, apply
#   nout-instrument-v1.patch to the copy only).
#
#   TREE  = 已应用 nout-instrument-v1.patch 的独立副本树（llama.h 必须含
#           instrumentation API 声明；纯净树的 llama.h 没有它们，编译会失败）
#
#   make build TREE=... BUILD=...            # build the driver
#   make run   TREE=... BUILD=... MODEL=...  # run with LLAMA_NOUT_INSTR set

.RECIPEPREFIX = >

ifeq ($(strip $(TREE)),)
$(error TREE must be set to the PATCHED copy of the floor1 tree (nout-instrument-v1.patch applied); the pristine tree's llama.h lacks the instrumentation API declarations)
endif
BUILD ?= /home/kunweiz/work/lead-nout-driver-20261030/build
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -I$(TREE)/include
LDLIBS   := -L$(BUILD) -lllama

build:
>mkdir -p $(BUILD)
>$(CXX) $(CXXFLAGS) driver/nout_floor1_driver.cpp $(LDLIBS) -o $(BUILD)/nout_floor1_driver
>ldd $(BUILD)/nout_floor1_driver

run: build
>LD_LIBRARY_PATH=$(BUILD) LLAMA_NOUT_INSTR=$$(pwd)/nout_instrument_events.jsonl $(BUILD)/nout_floor1_driver --model $${MODEL:?(set MODEL to a real GGUF)}

.PHONY: build run
