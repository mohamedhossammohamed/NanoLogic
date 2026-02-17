# NanoLogic TUI Tools

## Overview
The NanoLogic TUI (CyberSystem Edition) includes a modular suite of **30 tools** accessible via the dashboard or the CLI. These tools are divided into **Practical Utilities** (for system monitoring and verification) and **Fun/Satirical Tools** (for entertainment and "hacker" aesthetics).

## Practical Tools
*Located in `src/tools/practical.py`*

These tools provide real utility for monitoring the system, network, and development environment.

| Tool | Command | Description |
| :--- | :--- | :--- |
| **CPU Thermal** | `sys_temp` | Displays CPU temperature (macOS/Linux compatible). |
| **Network Latency** | `net_speed` | Measures ping latency to 1.1.1.1 (Cloudflare DNS). |
| **SSD Usage** | `disk_map` | Visual bar chart of disk usage and free space. |
| **RAM Flush** | `ram_flush` | Triggers Python's Garbage Collector (`gc.collect()`) to free memory. |
| **Top CPU Processes** | `proc_top` | Lists the top 5 processes consuming CPU. |
| **Hex Converter** | `hex_calc <val>` | Converts a number between Decimal, Hex, and Binary. |
| **SHA-256 Hasher** | `sha_quick <txt>` | Computes the SHA-256 hash of a string. |
| **Entropy Meter** | `entropy_meter <txt>` | Calculates the Shannon entropy (randomness) of a string. |
| **System Time** | `time_epoch` | Displays current Unix timestamp and ISO formatted time. |
| **JSON Formatter** | `json_fmt <json>` | Pretty-prints a raw JSON string. |
| **JWT Decoder** | `jwt_peek <token>` | Decodes a JWT payload (debug only, no signature verification). |
| **Password Gen** | `pass_gen` | Generates a secure, random 32-character password. |
| **Port Scan** | `port_scan` | Checks if local ports 80, 443, 3000, 8080 are open. |
| **Public IP** | `ip_public` | Fetches external IP address via `ifconfig.me`. |
| **Base64 Tool** | `base64_enc <mode> <txt>` | Encodes (`mode=enc`) or Decodes (`mode=dec`) Base64 strings. |
| **Git Status** | `git_status` | Shows current git branch and latest commit hash. |
| **Env Var Dump** | `env_dump` | Lists safe environment variables (SHELL, USER, LANG). |
| **File Tree** | `file_tree` | Displays a directory tree of the current folder. |
| **UUID Batch** | `uuid_gen` | Generates 5 random UUID4 strings. |
| **Battery Health** | `battery_health` | Displays battery cycle count on macOS. |

## Fun & Satirical Tools
*Located in `src/tools/fun.py`*

These tools add flavor to the CyberSystem interface and provide "hacker" visual effects.

| Tool | Command | Description |
| :--- | :--- | :--- |
| **Vibe Check** | `vibe_check` | Returns a random vibe score (0-100%) with slang assessment. |
| **Blame GPU** | `blame_gpu` | Generates a fake technical excuse for errors. |
| **Matrix Rain** | `matrix_rain` | Visual effect of falling binary code. |
| **Coffee Brew** | `coffee_brew` | Suggests a coffee type based on GPU temperature. |
| **Singularity** | `singularity` | Creepy "sentient AI" message from the M4 chip. |
| **Magic Bit** | `magic_bit` | Binary Magic 8-Ball (0 or 1 answer). |
| **Crypto Hype** | `crypto_hype` | Generates a random buzzword-heavy startup pitch. |
| **Hack Progress** | `hack_progress` | A fake "Cracking Mainframe" progress bar. |
| **Zen M4** | `zen_m4` | Returns a haiku about Apple Silicon efficiency. |
| **Self Destruct** | `self_destruct` | A fake countdown timer that aborts at the last second. |

## Usage through TUI

To use a tool, navigate to the **Tools** tab in the `nanologic.py` interface and type the command in the input box.
Example: `sys_temp` or `hex_calc 255`.
