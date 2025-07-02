#===================================================================================================
# Global Configuration
#===================================================================================================
BUILD_TYPE ?= release
VERBOSE ?= yes

#===================================================================================================
# Directories
#===================================================================================================
ROOT_DIRECTORY := $(CURDIR)
BINARIES_DIRECTORY := $(ROOT_DIRECTORY)/bin
PLOTS_DIRECTORY := $(ROOT_DIRECTORY)/scripts/plot

#===================================================================================================
# Toolchain Configuration
#===================================================================================================
CARGO ?= $(HOME)/.cargo/bin/cargo

ifeq ($(BUILD_TYPE),release)
CARGO_FLAGS := --release
else ifeq ($(BUILD_TYPE),debug)
CARGO_FLAGS :=
else
$(error Invalid BUILD_TYPE '$(BUILD_TYPE)'. Must be 'debug' or 'release')
endif

#===================================================================================================
# Global Build Rules
#===================================================================================================
all: all-v_sim

MAKE_DIRECTORY_COMAND=mkdir -p $(BINARIES_DIRECTORY)

make-directories:
	@$(MAKE_DIRECTORY_COMAND)


PLOTS_VENV_DIRECTORY := $(PLOTS_DIRECTORY)/venv

all-plots:
	@if [ ! -d $(PLOTS_VENV_DIRECTORY) ]; then python3 -m venv $(PLOTS_VENV_DIRECTORY); fi
	@$(PLOTS_VENV_DIRECTORY)/bin/pip3 install -r ./scripts/plot/requirements.txt > /dev/null
	@$(PLOTS_VENV_DIRECTORY)/bin/python3 $(PLOTS_DIRECTORY)/results.py

check: check-v_sim

clean: clean-v_sim
	rm -rf target
	rm -rf $(BINARIES_DIRECTORY)
#===================================================================================================
# Build Rules for "v_sim" Project
#===================================================================================================
BUILD_COMMAND=$(CARGO) build $(CARGO_FLAGS) -p v_sim
BUILD_CHECK_COMMNAD=$(CARGO) check $(CARGO_FLAGS) -p v_sim --message-format=json
ifeq ($(BUILD_TYPE),debug)
TARGET_DIRECTORY=target/debug
else
TARGET_DIRECTORY=target/release
endif

all-v_sim: make-directories
ifeq ($(VERBOSE),)
	@$(BUILD_COMMAND) --quiet
	@cp $(TARGET_DIRECTORY)/v_sim $(BINARIES_DIRECTORY)
else
	$(BUILD_COMMAND)
	cp $(TARGET_DIRECTORY)/v_sim $(BINARIES_DIRECTORY)
endif

check-v_sim:
	$(CARGO) check $(CARGO_FLAGS) --mesage-format=json -p v_sim

clean-v_sim:
	$(CARGO) clean -p v_sim
	rm -f $(BINARIES_DIRECTORY)/v_sim

#===================================================================================================
# Rules to run clippy on each project with the right target
#===================================================================================================

clippy:
	$(CARGO) clippy -p v_sim

