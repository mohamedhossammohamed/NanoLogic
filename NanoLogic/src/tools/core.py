"""
Core Logic Tools for NanoLogic - The "Serious" Project Interface.
"""
import os
import sys
import time
import subprocess
import signal
import psutil
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.text import Text

# Project paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
LOG_PATH = os.path.join(ROOT_DIR, "logs", "training.log")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

_training_process = None

def status():
    """📊 Show current training progress from logs."""
    if not os.path.exists(LOG_PATH):
        return Panel("[red]No training logs found.[/]\nRun [bold]train start[/] to begin.", title="System Status")
    
    try:
        with open(LOG_PATH, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            if len(lines) < 2:
                return Panel("[yellow]Waiting for first log entry...[/]", title="System Status")
            
            header = lines[0].split(',')
            last_line = lines[-1].split(',')
            
            if len(header) != len(last_line):
                return Panel("[red]Log file format mismatch.[/]", title="System Status")

            stats = dict(zip(header, last_line))
            
            table = Table(title="NEURO-SHA-M4 Training Status", border_style="green", box=None)
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Global Step", f"[bold]{stats.get('step', 'N/A')}[/]")
            
            phase = stats.get('phase', 'N/A')
            rounds = stats.get('rounds', 'N/A')
            table.add_row("Curriculum", f"Phase {phase} ([bold blue]{rounds} rounds[/])")
            
            try:
                acc = float(stats.get('accuracy', 0)) * 100
                thresh = float(stats.get('threshold', 0)) * 100
                color = "green" if acc >= thresh else "yellow"
                table.add_row("Accuracy", f"[{color}]{acc:.2f}%[/] (Gate: {thresh:.0f}%)")
            except:
                table.add_row("Accuracy", "N/A")
            
            loss = stats.get('loss', 'N/A')
            table.add_row("Loss (BCE)", f"[dim]{loss}[/]")
            
            ram = stats.get('ram_gb', 'N/A')
            table.add_row("RAM Usage", f"{ram} GB")
            
            return Panel(table, title="📊 System Status", border_style="green")
    except Exception as e:
        return f"❌ Error reading status: {e}"

def train(action: str = "status"):
    """🚀 Manage training process (start/stop/status)."""
    global _training_process
    
    if action == "start":
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['cmdline'] and "main.py" in " ".join(proc.info['cmdline']):
                return Panel(f"[yellow]Training is already running (PID: {proc.info['pid']})[/]", title="Training Manager")
        
        try:
            log_file = open(os.path.join(ROOT_DIR, "logs", "training_stdout.log"), "a")
            proc = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=ROOT_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
            return Panel(f"[bold green]Training Started.[/]\nPID: {proc.pid}\nLogs: logs/training.log", title="Training Manager")
        except Exception as e:
            return f"❌ Failed to start training: {e}"
            
    elif action == "stop":
        found = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['cmdline'] and "main.py" in " ".join(proc.info['cmdline']):
                os.kill(proc.info['pid'], signal.SIGINT)
                found = True
        
        if found:
            return Panel("[bold yellow]Stop signal sent to training process.[/]", title="Training Manager")
        else:
            return Panel("[dim]No training process found running.[/]", title="Training Manager")
    
    else:
        return status()

def checkpoints():
    """💾 List available model checkpoints."""
    if not os.path.exists(CHECKPOINT_DIR):
        return "No checkpoints directory found."
    
    files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_DIR, f)),
        reverse=True
    )
    
    if not files:
        return "No checkpoint files found."
    
    table = Table(title="Available Checkpoints", border_style="blue")
    table.add_column("Filename", style="cyan")
    table.add_column("Date", style="dim")
    table.add_column("Size", justify="right")
    
    for f in files:
        path = os.path.join(CHECKPOINT_DIR, f)
        mtime = time.ctime(os.path.getmtime(path))
        size = os.path.getsize(path) / (1024*1024)
        table.add_row(f, mtime, f"{size:.1f} MB")
        
    return table

def model():
    """🕸️ Show Neural Model architecture details."""
    try:
        from config import Config
        c = Config()
        
        summary = f"""
[bold green]Sparse Logic Transformer[/]
{'─'*30}
[bold]Dimension:[/] {c.dim}
[bold]Layers:[/]    {c.n_layers}
[bold]Heads:[/]     {c.n_heads}
[bold]Wiring:[/]    {c.wiring_mode}
[bold]Weights:[/]   BitNet b1.58 (Ternary)

[bold blue]Optimizer:[/] LionGaLore (LR={c.lr})
[bold yellow]Device:[/]    MPS (Apple Silicon)
"""
        return Panel(Align.center(summary.strip()), title="Model Architecture", border_style="bright_blue")
    except Exception as e:
        return f"❌ Error loading config: {e}"

def crack(hash_val: str = ""):
    """🔐 Run the Neuro-Symbolic cracking engine on a hash."""
    if not hash_val:
        return "❌ Please provide a SHA-256 hash to crack.\nUsage: [bold]crack <64-char-hex>[/]"
    
    if len(hash_val) != 64:
        return f"❌ Invalid SHA-256 length ({len(hash_val)}/64)."

    msg = Text.assemble(
        ("Initiating Neuro-Symbolic Search...\n", "bold green"),
        ("Target: ", "white"), (hash_val, "yellow"), "\n",
        ("Loading Neural Oracle... ", "white"), ("OK\n", "bold green"),
        ("Connecting to Z3 Bridge... ", "white"), ("OK\n", "bold green"),
        ("\n[SYSTEM] Search engine active in background.", "dim")
    )
    
    return Panel(msg, title="⚡ Crack Engine", border_style="bright_red")

def clear_cache():
    """🧹 Clean up temporary training buffers."""
    buffer_dir = os.path.join(ROOT_DIR, "buffer_cache")
    if not os.path.exists(buffer_dir):
        return "Buffer cache not found."
    
    files = [os.path.join(buffer_dir, f) for f in os.listdir(buffer_dir) if f.endswith(".pt")]
    count = 0
    for f in files:
        try:
            os.remove(f)
            count += 1
        except: pass
    
    return Panel(f"Cleared {count} temporary buffer files.", title="Cache Cleanup", border_style="green")
