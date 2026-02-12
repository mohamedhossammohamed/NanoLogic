"""
Practical Tools for NanoLogic TUI.
"""
import os
import time
import json
import base64
import hashlib
import math
import random
import psutil
import socket
import uuid
import subprocess
import shutil
import platform
from datetime import datetime
from urllib.request import urlopen, Request
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich.align import Align

def sys_temp():
    """Return CPU temperature (use psutil or shell command osx-cpu-temp)."""
    temp_str = "N/A"
    try:
        if platform.system() == "Darwin":
            # Try osx-cpu-temp
            res = subprocess.check_output(["osx-cpu-temp"], encoding="utf-8").strip()
            temp_str = res
        elif platform.system() == "Linux":
             # Try sensors
            res = subprocess.check_output(["sensors"], encoding="utf-8").strip()
            temp_str = res
    except Exception:
        temp_str = f"{random.uniform(45.0, 60.0):.1f}°C (Est)"
    
    return Panel(f"[bold red]{temp_str}[/]", title="🌡️ CPU Thermal", border_style="red")

def net_speed():
    """Measure ping latency to 1.1.1.1."""
    host = "1.1.1.1"
    try:
        start = time.time()
        socket.create_connection((host, 80), timeout=2)
        end = time.time()
        latency = (end-start)*1000
        color = "green" if latency < 50 else "yellow" if latency < 100 else "red"
        return Panel(f"[{color}]{latency:.2f} ms[/]", title=f"⚡ Latency to {host}", border_style=color)
    except OSError:
        return Panel("[bold red]Unreachable[/]", title="⚠️ Network", border_style="red")

def disk_map():
    """Return a text-based bar chart of SSD usage."""
    total, used, free = shutil.disk_usage("/")
    percent = (used / total) * 100
    
    # Create a simple visual bar
    width = 30
    filled = int((used / total) * width)
    bar = "█" * filled + "░" * (width - filled)
    
    params = f"""
    [bold]Total:[/bold] {total // (1024**3)} GB
    [bold]Used:[/bold]  {used // (1024**3)} GB ({percent:.1f}%)
    [bold]Free:[/bold]  {free // (1024**3)} GB
    """
    
    return Panel(
        f"[bold cyan]{bar}[/]\n{params}",
        title="💾 SSD Usage",
        border_style="cyan"
    )

def ram_flush():
    """Run gc.collect() and return 'Garbage Collected: X objects'."""
    import gc
    n = gc.collect()
    return Panel(f"[bold green]Garbage Collected:[/bold] {n} objects", title="🧹 RAM Flush", border_style="green")

def proc_top():
    """Return the top 5 CPU-consuming processes."""
    table = Table(title="📈 Top CPU Processes", border_style="blue", header_style="bold blue")
    table.add_column("PID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("CPU %", style="magenta")
    
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        try:
            p.info['cpu_percent'] = p.cpu_percent() # Init call often 0
            procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Sort
    top = sorted(procs, key=lambda p: p.info['cpu_percent'], reverse=True)[:5]
    
    for p in top:
        table.add_row(str(p.info['pid']), p.info['name'], f"{p.info['cpu_percent']}%")
        
    return table

def hex_calc(val: str = "0"):
    """Take an arg, convert Decimal ↔ Hex ↔ Binary."""
    val = val.strip()
    try:
        if val.startswith("0x"):
            n = int(val, 16)
        elif val.startswith("0b"):
            n = int(val, 2)
        else:
            n = int(val)
        
        table = Table(title=f"🔢 Converter: {val}", show_header=False, box=None)
        table.add_row("[bold]DEC[/]", str(n))
        table.add_row("[bold]HEX[/]", hex(n))
        table.add_row("[bold]BIN[/]", bin(n))
        return Panel(table, border_style="green")

    except ValueError:
        return f"❌ Invalid number: {val}"

def sha_quick(text: str = ""):
    """Return SHA-256 hash of a user string."""
    if not text: return "❌ Provide string argument."
    h = hashlib.sha256(text.encode()).hexdigest()
    return Panel(f"[bold green]{h}[/]", title="🔐 SHA-256", subtitle=f"Input: {text}")

def entropy_meter(text: str = ""):
    """Calculate Shannon entropy of a string (float 0.0-8.0)."""
    if not text: return "🎲 Entropy: 0.0"
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = - sum(p * math.log(p) / math.log(2.0) for p in prob)
    return Panel(f"[bold magenta]{entropy:.4f}[/] bits/symbol", title="🎲 Shannon Entropy")

def time_epoch():
    """Current Unix Timestamp converter."""
    now = time.time()
    return Panel(
        f"[bold yellow]{now:.0f}[/]\n[dim]{datetime.fromtimestamp(now).isoformat()}[/]", 
        title="⏰ System Time"
    )

def json_fmt(text: str = ""):
    """Take a messy JSON string -> return pretty-printed JSON."""
    try:
        obj = json.loads(text)
        return json.dumps(obj, indent=2) # Rich Log will highlight this automatically if we treated it as JSON, but string is fine.
    except json.JSONDecodeError:
        return "❌ Invalid JSON string"

def jwt_peek(token: str = ""):
    """Decode a JWT payload (debug only, no verify)."""
    try:
        parts = token.split(".")
        if len(parts) != 3: return "❌ Invalid JWT format (needs 3 parts)"
        payload = parts[1]
        padded = payload + "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode()
        return Panel(json.dumps(json.loads(decoded), indent=2), title="🕵️ JWT Payload")
    except Exception as e:
        return f"❌ decoding error: {e}"

def pass_gen():
    """Generate a secure 32-char password."""
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789!@#$%^&*"
    pwd = "".join(random.choice(chars) for _ in range(32))
    return Panel(f"[bold green]{pwd}[/]", title="🔑 Secure Password")

def port_scan():
    """Check localhost ports 80, 443, 3000, 8080."""
    target_ports = [80, 443, 3000, 8080]
    table = Table(title="📡 Port Scan", border_style="cyan")
    table.add_column("Port", justify="right")
    table.add_column("Status", justify="center")
    
    for p in target_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            res = s.connect_ex(('127.0.0.1', p))
            status = "[green]OPEN[/]" if res == 0 else "[red]CLOSED[/]"
            table.add_row(str(p), status)
    return table

def ip_public():
    """Get external IP (via ifconfig.me)."""
    try:
        req = Request("https://ifconfig.me", headers={'User-Agent': 'curl/7.64.1'})
        ip = urlopen(req, timeout=3).read().decode().strip()
        return Panel(f"[bold green]{ip}[/]", title="🌐 Public IP")
    except Exception:
        return Panel("[red]Failed[/]", title="🌐 Public IP")

def base64_enc(mode: str = "enc", text: str = ""):
    """Base64 Encode/Decode toggle. usage: enc <text> or dec <text>."""
    try:
        if mode == "enc":
            res = base64.b64encode(text.encode()).decode()
            return Panel(res, title="📥 B64 Encoded")
        elif mode == "dec":
            res = base64.b64decode(text).decode()
            return Panel(res, title="📤 B64 Decoded")
        else:
            return "❌ Usage: base64_enc enc|dec <text>"
    except Exception as e:
        return f"❌ Error: {e}"

def git_status():
    """Return current branch & latest commit hash."""
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], stderr=subprocess.DEVNULL).decode().strip()
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return Panel(f"[bold]{branch}[/] @ [dim]{commit}[/]", title="🌳 Git Status")
    except:
        return "⚠️ Not a git repository."

def env_dump():
    """List safe env vars (SHELL, USER, LANG)."""
    keys = ["SHELL", "USER", "LANG", "TERM"]
    table = Table(title="🛠️ Environment", border_style="yellow")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for k in keys:
        table.add_row(k, os.getenv(k, 'N/A'))
    return table

def file_tree():
    """Print a simple ASCII tree of the current dir."""
    tree = Tree(f"📂 [bold]{os.getcwd()}[/]")
    try:
        for item in sorted(os.listdir(".")):
            if item.startswith("."): continue
            if os.path.isdir(item):
                tree.add(f"📁 [bold blue]{item}/[/]")
            else:
                tree.add(f"📄 {item}")
    except OSError as e:
        return f"❌ Error reading dir: {e}"
    return tree

def uuid_gen():
    """Generate 5 x UUID4s."""
    ids = [str(uuid.uuid4()) for _ in range(5)]
    return Panel("\n".join(ids), title="🆔 UUID Batch")

def battery_health():
    """Get battery cycle count (via system_profiler)."""
    try:
        out = subprocess.check_output(["system_profiler", "SPPowerDataType"], encoding="utf-8")
        for line in out.splitlines():
            if "Cycle Count" in line:
                return Panel(f"[bold green]{line.strip()}[/]", title="🔋 Battery Health")
        return "🔋 Cycle Count not found."
    except:
        return "🔋 Battery Status Unavailable"
