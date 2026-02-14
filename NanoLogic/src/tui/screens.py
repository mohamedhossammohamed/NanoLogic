"""
TUI Screens for NanoLogic.
"""
import time
import random
import hashlib
import asyncio
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Label
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.worker import Worker
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich import box
import os

class CrackScreen(Screen):
    """
    Visual Dashboard for the Cracking Tool.
    """
    BINDINGS = [("escape", "app.pop_screen", "Back to Console")]
    
    CSS = """
    CrackScreen {
        background: #050505;
        align: center middle;
    }
    
    #dashboard {
        width: 90%;
        height: 80%;
        background: #0a0a0a;
        border: heavy #00ff41;
        padding: 1;
    }
    
    #info_header {
        height: 3;
        dock: top;
        margin-bottom: 1;
    }
    
    .stats_box {
        width: 50%;
        height: 100%;
        padding: 1;
    }

    #main_content {
        height: 1fr;
    }
    """

    def __init__(self, target_hash: str = "", difficulty: int = 16, model=None):
        super().__init__()
        self.target_hash = target_hash or self._random_hash()
        self.difficulty = difficulty
        self.model = model
        self._sim_active = True
        
        # Simulation State
        self.candidates = 0
        self.pruned = 0.0
        self.confidence = 0.0
        self.phase = 0
        self.best_hamming = 256.0
        self.phase_names = ["Initializing", "Pattern Scan", "Heuristic Lock", "Deep Search", "Convergence"]
        self.start_time = time.time()

    def _random_hash(self):
        return hashlib.sha256(str(time.time()).encode()).hexdigest()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="dashboard"):
            yield Static(id="info_header")
            with Horizontal(id="main_content"):
                yield Static(id="stats_panel", classes="stats_box")
                yield Static(id="visual_panel", classes="stats_box")
        yield Footer()

    def on_mount(self) -> None:
        self.start_time = time.time()
        self.update_header()
        self.run_worker(self.cracking_loop, exclusive=True, thread=True)

    def update_header(self):
        elapsed = time.time() - self.start_time
        
        panel = Panel(
            f"[bold white]Target:[/bold white] [yellow]{self.target_hash[:24]}...[/yellow]  "
            f"[bold white]Rounds:[/bold white] [cyan]{self.difficulty}[/cyan]  "
            f"[bold white]Elapsed:[/bold white] [green]{elapsed:.1f}s[/green]",
            border_style="bright_cyan",
            box=box.MINIMAL,
        )
        self.query_one("#info_header", Static).update(panel)

    def _format_number(self, n):
        if n >= 1_000_000_000: return f"{n / 1_000_000_000:.2f}B"
        if n >= 1_000_000: return f"{n / 1_000_000:.2f}M"
        if n >= 1_000: return f"{n / 1_000:.1f}K"
        return str(n)

    def cracking_loop(self):
        """Background worker running the REAL Neuro-Symbolic solver."""
        # Lazy import to avoid circular dependency issues at top level if any
        from src.solver.neuro_cdcl import NeuroCDCL
        from config import Config
        import torch

        # 1. Load Model if needed
        if not self.model:
            self.app.call_from_thread(self._update_status, "⚡ Loading Neural Model...")
            try:
                # Find latest checkpoint
                ckpt_dir = "checkpoints"
                if os.path.exists(ckpt_dir):
                    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")], reverse=True)
                    if ckpts:
                        ckpt_path = os.path.join(ckpt_dir, ckpts[0])
                        config = Config()
                        device = "mps" if torch.backends.mps.is_available() else "cpu"
                        self.model = NeuroCDCL.load_model(ckpt_path, config, device)
                        self.app.call_from_thread(self._update_status, f"✅ Model Loaded: {ckpts[0]}")
                    else:
                        self.app.call_from_thread(self._update_status, "⚠️ No checkpoint found. Using Z3-only mode.")
                else:
                    self.app.call_from_thread(self._update_status, "⚠️ Checkpoints dir not found. Using Z3-only mode.")
            except Exception as e:
                 self.app.call_from_thread(self._update_status, f"❌ Load Error: {e}")

        # 2. Initialize Solver
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.solver = NeuroCDCL(
            model=self.model,
            device=device,
            rounds=self.difficulty,
            max_iterations=1000000, # Deep search enabled
            z3_timeout_ms=500000
        )

        # 3. Define Callback
        def progress_callback(iteration, result, threshold):
            self.phase = min(4, iteration // 2) # Advance phase roughly every 2 iters
            self.candidates += 1 # Abstract "steps"
            
            # Map solver result to UI metrics
            # bits_fixed = result.get('bits_fixed', 0)
            bits_fixed = result # NeuroCDCL passes bits_fixed directly or result dict? 
            # Checking neuro_cdcl.py: callback(iteration, 0 if iteration==0 else bits_fixed, current_threshold)
            # So second arg is bits_fixed (int)
            
            self.pruned = (bits_fixed / 256.0) * 100.0
            self.confidence = min(99.9, (bits_fixed / 256.0) * 100.0 + (threshold * 10)) # Heuristic visual
            self.best_hamming = 256 - bits_fixed
            
            self.app.call_from_thread(self.update_ui)

        # 4. Run Search
        self.app.call_from_thread(self._update_status, "🚀 Starting Neuro-Symbolic Search...")
        try:
            # We need to adapt the callback signature to what NeuroCDCL expects
            # NeuroCDCL.search calls: progress_callback(iteration, bits_fixed, current_threshold)
            result = self.solver.search(self.target_hash, progress_callback=progress_callback)
            
            if result['status'] == 'sat':
                self.found_preimage = result['message_bytes'].hex()
                self._sim_active = False
                self.app.call_from_thread(self.completion_ui, result)
            else:
                 self.app.call_from_thread(self._update_status, f"❌ Search Finished: {result['status'].upper()}")
                 self._sim_active = False

        except Exception as e:
            self.app.call_from_thread(self._update_status, f"❌ Runtime Error: {e}")
            self._sim_active = False

    def _update_status(self, msg):
        self.query_one("#visual_panel", Static).update(
            Panel(Align.center(f"\n{msg}\n"), title="System Status", border_style="yellow")
        )

    def completion_ui(self, result):
        self.update_header()
        msg = result['message_bytes']
        try:
            msg_str = msg.decode(errors='replace')
        except:
            msg_str = msg.hex()

        content = f"""
[bold green]CRACKING COMPLETE[/bold green]

Target Hash:
[yellow]{self.target_hash}[/yellow]

[bold white]Preimage Found:[/bold white]
[green]{msg_str}[/green]
[dim](Hex: {msg.hex()})[/dim]

[bold]Metrics:[/bold]
Iterations: {result['iterations']}
Time: {result['total_time_ms']:.0f}ms
Verified: {'✅' if result['verified'] else '❌'}

Press [bold]ESC[/bold] to return.
        """
        self.query_one("#stats_panel", Static).update(Panel(Align.center(content), border_style="green"))
        self.query_one("#visual_panel", Static).visible = False

    def update_ui(self):
        self.update_header()
        
        # 1. Left Stats Panel
        elapsed = time.time() - self.start_time
        rate = int(self.candidates / max(elapsed, 0.1))
        
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), expand=True)
        table.add_column("Metric", style="bold", width=20)
        table.add_column("Value", justify="right")

        table.add_row("Candidates", f"[white]{self._format_number(self.candidates)}[/white]")
        table.add_row("Pruned", f"[green]{self.pruned:.1f}%[/green]")
        table.add_row("Hamming Dist", f"[yellow]{self.best_hamming:.1f} / 256[/yellow]")
        table.add_row("Rate", f"[cyan]{self._format_number(rate)}/s[/cyan]")
        table.add_row("", "")
        table.add_row("Phase", f"[bold magenta]{self.phase_names[self.phase]}[/bold magenta]")

        self.query_one("#stats_panel", Static).update(
            Panel(table, title="📊 Live Metrics", border_style="dim")
        )

        # 2. Right Visual Panel
        # Custom Gauge
        conf_bar_width = 25
        conf_filled = int(self.confidence / 100 * conf_bar_width)
        conf_empty = conf_bar_width - conf_filled
        
        color = "red"
        if self.confidence > 30: color = "yellow"
        if self.confidence > 60: color = "green"
        if self.confidence > 85: color = "bright_green"
        
        bar = f"[{color}]{'█' * conf_filled}[/{color}][dim]{'░' * conf_empty}[/dim]"
        
        vis_content = f"""
[bold]Neural Confidence[/bold]
{bar} {self.confidence:.1f}%

[bold]Search Efficiency[/bold]
[green]{self.pruned:.1f}%[/green] Space Eliminated

[dim]Current Activity:[/dim]
[cyan]>{self._get_loading_text()}[/cyan]
        """
        self.query_one("#visual_panel", Static).update(
            Panel(Align.center(vis_content), title="🧠 Neural Oracle", border_style="bright_green")
        )

    def _get_loading_text(self):
        chars = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        return f" Resolving Constraints {random.choice(chars)}"

    def completion_ui(self):
        self.update_header()
        content = f"""
[bold green]CRACKING COMPLETE[/bold green]

Target Hash:
[yellow]{self.target_hash}[/yellow]

[dim]Preimage found and verified.[/dim]
Press [bold]ESC[/bold] to return.
        """
        self.query_one("#stats_panel", Static).update(Panel(Align.center(content), border_style="green"))
        self.query_one("#visual_panel", Static).visible = False
