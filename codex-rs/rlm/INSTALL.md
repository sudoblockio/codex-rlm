# RLM Runtime Installation Guide

## Prerequisites

### Rust

The repo includes `rust-toolchain.toml`, so rustup will automatically install the correct version.

```bash
# Install rustup if you don't have it
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Python 3.11+

RLM embeds a Python runtime via PyO3. You need Python 3.11 or later **with shared library support**.

#### macOS (Homebrew)

```bash
brew install python@3.11
# or python@3.12, python@3.13, python@3.14
```

Homebrew Python includes shared libraries by default.

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install python3.11 python3.11-dev
# The -dev package includes the shared library
```

#### Fedora/RHEL

```bash
sudo dnf install python3.11 python3.11-devel
```

#### Linuxbrew

```bash
brew install python@3.14
```

**Important:** You may need to set `LD_LIBRARY_PATH` (see Troubleshooting below).

#### pyenv

If using pyenv, you **must** build Python with shared library support:

```bash
PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.11.0
pyenv global 3.11.0
```

---

## Build Instructions

### 1. Clone and checkout

```bash
git clone https://github.com/openai/codex.git
cd codex/codex-rs
git checkout feat/rlm-runtime
```

### 2. Build

```bash
cargo build --release -p codex-tui --features rlm
```

The binary will be at `target/release/codex-tui`.

### 3. (Optional) Install to PATH

```bash
# Option A: Copy to /usr/local/bin
sudo cp target/release/codex-tui /usr/local/bin/codex-rlm

# Option B: Symlink
sudo ln -sf $(pwd)/target/release/codex-tui /usr/local/bin/codex-rlm

# Option C: Add to PATH in your shell config
echo 'export PATH="$HOME/codex/codex-rs/target/release:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

---

## Configuration

Add the RLM tools to your config at `~/.codex/config.toml`:

```toml
experimental_supported_tools = [
  "rlm_load",
  "rlm_load_append",
  "rlm_exec",
  "rlm_query",
  "rlm_memory",
  "rlm_helpers",
  "glob",
  "grep"
]

# Optional: Configure RLM gateway model
[rlm.gateway]
model = "gpt-4"

# Optional: Sub-agent tool policy
[rlm.sub_agent_policy]
allowed_tool_overrides = ["shell", "apply_patch"]
require_approval = false
```

---

## Verify Installation

```bash
# Run the TUI
./target/release/codex-tui

# Or if installed to PATH
codex-rlm
```

Once in a session, the model should have access to:
- `rlm_load` - Load context from file/directory
- `rlm_load_append` - Append additional context
- `rlm_exec` - Execute Python code over context
- `rlm_query` - Quick search + summarize
- `rlm_memory_*` - Session memory
- `rlm_helpers_*` - Python helper management

---

## Troubleshooting

### `libpython3.X.so: cannot open shared object file`

The Python shared library isn't in the library search path.

**Fix 1: Set LD_LIBRARY_PATH**

```bash
# Find your Python's lib directory
python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"

# Export it (add to ~/.bashrc or ~/.zshrc for persistence)
export LD_LIBRARY_PATH=/path/to/python/lib:$LD_LIBRARY_PATH
```

**Common paths:**
- Homebrew (macOS): `/opt/homebrew/opt/python@3.X/Frameworks/Python.framework/Versions/3.X/lib`
- Linuxbrew: `/home/linuxbrew/.linuxbrew/lib`
- Ubuntu system Python: `/usr/lib/x86_64-linux-gnu`
- pyenv: `~/.pyenv/versions/3.X.X/lib`

**Fix 2: Install Python dev package**

On Linux, make sure you have the `-dev` or `-devel` package:

```bash
# Ubuntu/Debian
sudo apt install python3.11-dev

# Fedora/RHEL
sudo dnf install python3.11-devel
```

**Fix 3: Rebuild Python with shared library (pyenv)**

```bash
PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.11.0
```

### `error: the package 'codex-tui' does not contain this feature: rlm`

Make sure you're on the correct branch:

```bash
git checkout feat/rlm-runtime
git pull origin feat/rlm-runtime
```

### RLM tools not appearing in session

1. Check your `~/.codex/config.toml` has `experimental_supported_tools` configured
2. Make sure you built with `--features rlm`
3. Restart the TUI after config changes

### Build fails with PyO3 errors

Ensure Python is discoverable:

```bash
# Check Python version
python3 --version  # Should be 3.11+

# PyO3 uses python3-config to find Python
which python3-config
python3-config --ldflags
```

If using a virtual environment, PyO3 may have trouble finding the base Python. Try building outside the venv or set:

```bash
export PYO3_PYTHON=$(which python3)
```

---

## Available Tools

| Tool | Description |
|------|-------------|
| `rlm_load(path)` | Load context from file/directory (resets session) |
| `rlm_load_append(path)` | Append additional context (preserves state) |
| `rlm_exec(code)` | Execute Python, return structured `result_json` |
| `rlm_query(prompt)` | Quick scan + summarize via sub-agent |
| `rlm_memory_get/put/list/clear` | Session memory for multi-pass workflows |
| `rlm_helpers_add/list/remove` | Reusable Python helper management |
| `glob` | List directory contents (alias for list_dir) |
| `grep` | Search file contents (alias for grep_files) |

---

## Quick Test

Once running, try:

```
Load the current directory and tell me about its structure
```

The model should use `rlm_load` to load your codebase and then explore it with Python.
