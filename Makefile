# Env/Tools
#
# Python env to handle deps.yaml
VENV ?= .venv
ifeq ($(OS),Windows_NT)
BOOTSTRAP_PYTHON ?= py
PYTHON ?= $(VENV)/Scripts/python.exe
else
BOOTSTRAP_PYTHON ?= python3
PYTHON ?= $(VENV)/bin/python
endif
#
DEPS_MANIFEST := deps.yaml
DEPS_ROOT := 3rdparty
FETCHER := scripts/fetch_deps.py
#
# CPP
CXX = cl
SHELL := cmd.exe
.SHELLFLAGS := /D /S /C
VSWHERE ?= ${ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe
VSINSTALL := $(shell "$(VSWHERE)" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
VSDEVCMD ?= $(VSINSTALL)/Common7/Tools/VsDevCmd.bat
#
CPPFLAGS += /I3rdparty /I. /I"$(CUDA_PATH)/include"
CXXFLAGS += /nologo /std:c++17 /O2 /W4 /EHsc /MD
LDLIBS += ws2_32.lib

# CUDA
NVCC ?= nvcc
CUDA_ARCH ?= native
NVCCFLAGS += -O2 -std=c++17 -arch=$(CUDA_ARCH) -Iinclude -Xcompiler "/MD /EHsc /nologo" $(NVCC_EXTRA)
CUDA_LIBDIR ?= $(CUDA_PATH)/lib/x64

# Applications
MAIN := epi.exe
TEST := epitest.exe

# Libs
BUILD_DIR := build
BRIDGE_DIR := $(DEPS_ROOT)/bed-reader-cpp/0.1.0
BRIDGE_LIB := $(BRIDGE_DIR)/libs/x86_64-pc-windows-msvc/bed_reader_bridge.lib
# System libraries required by the Rust static library.
RUST_SYSLIBS := advapi32.lib bcrypt.lib ntdll.lib userenv.lib kernel32.lib shell32.lib ole32.lib

# Objs
OBJS := $(BUILD_DIR)/main.obj $(BUILD_DIR)/test.obj $(BUILD_DIR)/epi_gpu.obj
MAIN_OBJS := $(BUILD_DIR)/main.obj $(BUILD_DIR)/epi_gpu.obj
TEST_OBJS := $(BUILD_DIR)/test.obj $(BUILD_DIR)/epi_gpu.obj

# Headers
EPI_HEADERS := include/epi_core.h include/epi_cell.h include/epi_stats.h include/epi_scanner.h
MAIN_HEADERS := $(EPI_HEADERS) include/bed_dataset.h $(BRIDGE_DIR)/include/bed_reader_bridge.h include/epi_cpu.h include/epi_gpu.cuh
TEST_HEADERS := $(EPI_HEADERS) include/epi_cpu.h include/epi_gpu.cuh
GPU_HEADERS := $(EPI_HEADERS) include/epi_gpu.cuh

# Run a command inside the MSVC developer environment (cl/link/nvcc host compiler).
vsrun = cmd /D /C "call "$(VSDEVCMD)" -arch=x64 -host_arch=x64 >NUL && $(1)"

.PHONY: all debug setup deps deps-force doctor clean clean-deps

all: deps $(MAIN) $(TEST)

debug:
	$(MAKE) MAIN=build/debug/epi.exe     BUILD_DIR=build/debug CXXFLAGS="/nologo /std:c++17 /Od /Z7 /W4 /EHsc /MD" NVCCFLAGS="-O0 -g -std=c++17 -arch=$(CUDA_ARCH) -Iinclude -Xcompiler \"/MD /EHsc /nologo /Od /Z7\" $(NVCC_EXTRA)" LDFLAGS="$(LDFLAGS) /DEBUG /INCREMENTAL:NO" build/debug/epi.exe
	$(MAKE) TEST=build/debug/epitest.exe BUILD_DIR=build/debug CXXFLAGS="/nologo /std:c++17 /Od /Z7 /W4 /EHsc /MD" NVCCFLAGS="-O0 -g -std=c++17 -arch=$(CUDA_ARCH) -Iinclude -Xcompiler \"/MD /EHsc /nologo /Od /Z7\" $(NVCC_EXTRA)" LDFLAGS="$(LDFLAGS) /DEBUG /INCREMENTAL:NO" build/debug/epitest.exe

setup:
	$(BOOTSTRAP_PYTHON) -c "import sys; assert sys.version_info >= (3, 12), 'Python 3.12+ is required'"
	$(BOOTSTRAP_PYTHON) -m venv $(VENV)
	"$(PYTHON)" -m pip install "PyYAML>=6,<7"

deps:
	"$(PYTHON)" $(FETCHER) --manifest $(DEPS_MANIFEST) --root $(DEPS_ROOT)

deps-force:
	"$(PYTHON)" $(FETCHER) --manifest $(DEPS_MANIFEST) --root $(DEPS_ROOT) --force

doctor:
	@"$(PYTHON)" --version
	@$(MAKE) --version
	@"$(PYTHON)" -c "import yaml; print('PyYAML: ' + yaml.__version__)"

$(BUILD_DIR):
	if not exist "$(subst /,\,$(BUILD_DIR))" mkdir "$(subst /,\,$(BUILD_DIR))"

$(BUILD_DIR)/main.obj: src/main.cpp $(MAIN_HEADERS) | $(BUILD_DIR)
	$(call vsrun,$(CXX) $(CPPFLAGS) $(CXXFLAGS) /c src/main.cpp /Fo:$@)

$(BUILD_DIR)/test.obj: src/test_epi.cpp $(TEST_HEADERS) | $(BUILD_DIR)
	$(call vsrun,$(CXX) $(CPPFLAGS) $(CXXFLAGS) /c src/test_epi.cpp /Fo:$@)

# cuPrintf.cu was legacy code; epi_gpu.cu now includes epi_gpu.cuh declarations.
$(BUILD_DIR)/epi_gpu.obj: src/epi_gpu.cu $(GPU_HEADERS) | $(BUILD_DIR)
	$(call vsrun,$(NVCC) $(NVCCFLAGS) -c src/epi_gpu.cu -o $@)

$(MAIN): $(MAIN_OBJS) $(BRIDGE_LIB)
	$(call vsrun,link /nologo $(MAIN_OBJS) $(BRIDGE_LIB) cudart.lib $(LDLIBS) $(RUST_SYSLIBS) /LIBPATH:"$(CUDA_LIBDIR)" $(LDFLAGS) /OUT:$@)

$(TEST): $(TEST_OBJS)
	$(call vsrun,link /nologo $(TEST_OBJS) cudart.lib $(LDLIBS) /LIBPATH:"$(CUDA_LIBDIR)" $(LDFLAGS) /OUT:$@)

clean:
	if exist $(BUILD_DIR) rmdir /S /Q $(BUILD_DIR)
	if exist $(MAIN) del /Q $(MAIN)
	if exist $(TEST) del /Q $(TEST)

clean-deps:
	rm -rf $(DEPS_ROOT)


