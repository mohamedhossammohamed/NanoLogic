#!/usr/bin/env python3
"""
Neuro-CLI: Interactive Demo Interface for Neuro-SHA-M4
======================================================

Two modes:
  --mode race   : Watch AI vs Brute Force race (simulation)
  --mode crack  : Interactive hash cracker dashboard

Requires: pip install rich
"""

import argparse
import hashlib
import os
import random
import sys
import time

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.align import Align
    from rich import box
except ImportError:
    print("❌ Missing dependency: rich")
    print("   Install with: pip install rich")
    sys.exit(1)

console = Console()

# ─── Constants ───────────────────────────────────────────────────────────────

BANNER = r"""
[bold cyan]
 ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗       ███████╗██╗  ██╗ █████╗ 
 ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗      ██╔════╝██║  ██║██╔══██╗
 ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║█████╗███████╗███████║███████║
 ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║╚════╝╚════██║██╔══██║██╔══██║
 ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝      ███████║██║  ██║██║  ██║
 ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝       ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
[/bold cyan]
[dim]Neuro-Symbolic SHA-256 Cryptanalysis · Apple M4 · BitNet b1.58[/dim]
"""

CREDITS = "[dim]Built by [bold]@MohamedHz72007[/bold] · Medical Student / Vibe Coder[/dim]"


# ─── Utilities ───────────────────────────────────────────────────────────────

def random_hash():
    """Generate a random SHA-256 hash."""
    data = os.urandom(32)
    return hashlib.sha256(data).hexdigest()


def format_number(n):
    """Format large numbers with commas."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ─── MODE: RACE ─────────────────────────────────────────────────────────────

def run_race():
    """Simulate a race between Standard Z3 and Neuro-SHA-M4."""
    console.print(BANNER)
    console.print(Align.center(CREDITS))
    console.print()

    target_hash = random_hash()
    console.print(Panel(
        f"[bold white]Target Hash:[/bold white]\n[yellow]{target_hash}[/yellow]\n\n"
        f"[dim]Difficulty: 16-Round Reduced SHA-256[/dim]",
        title="⚔️  [bold]THE RACE[/bold]",
        border_style="bright_cyan",
        padding=(1, 2),
    ))
    console.print()

    # Simulation parameters
    total_steps = 200
    z3_speed = 1.0          # Base speed
    neuro_speed = 1.0       # Starts same, then jumps
    z3_progress = 0.0
    neuro_progress = 0.0
    z3_candidates = 0
    neuro_candidates = 0
    neuro_pruned = 0.0
    heuristic_jumps = 0

    # Pre-compute jump points (where the AI finds patterns)
    jump_points = sorted(random.sample(range(20, total_steps - 10), 8))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("·"),
        TextColumn("{task.fields[candidates]}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        z3_task = progress.add_task(
            "[red]Standard Z3[/red]",
            total=100,
            candidates="0 checked",
        )
        neuro_task = progress.add_task(
            "[green]Neuro-SHA-M4[/green]",
            total=100,
            candidates="0 checked",
        )

        for step in range(total_steps):
            time.sleep(0.05)

            # Z3: steady linear progress
            z3_increment = random.uniform(0.3, 0.7) * z3_speed
            z3_progress = min(100, z3_progress + z3_increment)
            z3_candidates += random.randint(50_000, 200_000)

            # Neuro-SHA: same base speed + periodic jumps
            neuro_increment = random.uniform(0.3, 0.7) * neuro_speed
            neuro_candidates += random.randint(30_000, 100_000)

            # Heuristic jump!
            if step in jump_points:
                jump_size = random.uniform(3.0, 8.0)
                neuro_increment += jump_size
                neuro_pruned += random.uniform(2.0, 6.0)
                heuristic_jumps += 1
                neuro_speed *= 1.15  # Cumulative acceleration

            neuro_progress = min(100, neuro_progress + neuro_increment)

            progress.update(
                z3_task,
                completed=z3_progress,
                candidates=f"{format_number(z3_candidates)} checked",
            )
            progress.update(
                neuro_task,
                completed=neuro_progress,
                candidates=f"{format_number(neuro_candidates)} checked · {neuro_pruned:.0f}% pruned",
            )

            # If neuro wins, stop early
            if neuro_progress >= 100:
                # Let Z3 catch up a tiny bit for drama
                time.sleep(0.3)
                break

        # Ensure neuro is at 100
        progress.update(neuro_task, completed=100)

    console.print()

    # Results table
    results = Table(
        title="🏁 Race Results",
        box=box.DOUBLE_EDGE,
        border_style="bright_cyan",
        title_style="bold white",
    )
    results.add_column("Metric", style="bold")
    results.add_column("Standard Z3", style="red", justify="right")
    results.add_column("Neuro-SHA-M4", style="green", justify="right")

    results.add_row("Candidates Checked", format_number(z3_candidates), format_number(neuro_candidates))
    results.add_row("Search Space Pruned", "0%", f"{neuro_pruned:.1f}%")
    results.add_row("Heuristic Jumps", "—", str(heuristic_jumps))
    results.add_row("Final Progress", f"{z3_progress:.1f}%", "100.0%")
    results.add_row(
        "Winner", "", "[bold green]🏆 NEURO-SHA-M4[/bold green]"
    )

    console.print(results)
    console.print()
    console.print(Panel(
        "[dim]This is a simulation. The real model is currently in active training.\n"
        "Once training completes, plug in the model with:[/dim]\n\n"
        "  [cyan]python3 tools/neuro_cli.py --mode race --model checkpoints/neuro_sha_final.pt[/cyan]",
        title="ℹ️  Simulation Mode",
        border_style="dim",
    ))


# ─── MODE: CRACK ─────────────────────────────────────────────────────────────

def run_crack():
    """Interactive hash cracker dashboard."""
    console.print(BANNER)
    console.print(Align.center(CREDITS))
    console.print()

    # Get user input
    console.print("[bold cyan]═══ Interactive Mode ═══[/bold cyan]\n")

    target = console.input("[bold]Enter SHA-256 hash[/bold] [dim](or press Enter for random)[/dim]: ").strip()
    if not target:
        target = random_hash()
        console.print(f"  [dim]Generated random hash:[/dim] [yellow]{target}[/yellow]")

    console.print()
    diff_str = console.input("[bold]Difficulty (rounds)[/bold] [dim]8 / 16 / 32 / 64[/dim] [default: 16]: ").strip()
    difficulty = int(diff_str) if diff_str in ("8", "16", "32", "64") else 16
    console.print(f"  [dim]Difficulty set to {difficulty} rounds[/dim]")
    console.print()

    # Simulation loop
    candidates = 0
    pruned = 0.0
    confidence = 0.0
    phase = 0
    phase_names = ["Initializing", "Pattern Scan", "Heuristic Lock", "Deep Search", "Convergence"]
    start_time = time.time()
    best_hamming = 256
    step = 0

    try:
        with Live(console=console, refresh_per_second=8) as live:
            while True:
                step += 1
                elapsed = time.time() - start_time

                # Simulate progress
                candidates += random.randint(10_000, 80_000) * (1 + difficulty // 16)
                pruned = min(99.9, pruned + random.uniform(0.01, 0.15))
                confidence = min(99.9, confidence + random.uniform(0.02, 0.3))
                best_hamming = max(0, best_hamming - random.uniform(0, 0.8))

                # Phase transitions
                if confidence > 20 and phase == 0:
                    phase = 1
                elif confidence > 45 and phase == 1:
                    phase = 2
                elif confidence > 70 and phase == 2:
                    phase = 3
                elif confidence > 90 and phase == 3:
                    phase = 4

                # Build dashboard
                layout = Layout()
                layout.split_column(
                    Layout(name="header", size=3),
                    Layout(name="body", size=14),
                    Layout(name="footer", size=3),
                )

                # Header
                layout["header"].update(Panel(
                    f"[bold white]Target:[/bold white] [yellow]{target[:32]}...[/yellow]  "
                    f"[bold white]Rounds:[/bold white] [cyan]{difficulty}[/cyan]  "
                    f"[bold white]Elapsed:[/bold white] [green]{elapsed:.1f}s[/green]",
                    border_style="bright_cyan",
                    box=box.MINIMAL,
                ))

                # Body - metrics
                body_layout = Layout()
                body_layout.split_row(
                    Layout(name="left"),
                    Layout(name="right"),
                )

                # Left: stats table
                stats = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
                stats.add_column("Metric", style="bold", width=22)
                stats.add_column("Value", justify="right", width=20)

                stats.add_row("Candidates Checked", f"[white]{format_number(candidates)}[/white]")
                stats.add_row("Search Space Pruned", f"[green]{pruned:.1f}%[/green]")
                stats.add_row("Best Hamming Dist", f"[yellow]{best_hamming:.1f} / 256[/yellow]")
                stats.add_row("Rate", f"[cyan]{format_number(int(candidates / max(elapsed, 0.1)))}/s[/cyan]")
                stats.add_row("", "")
                stats.add_row("Phase", f"[bold magenta]{phase_names[phase]}[/bold magenta]")

                body_layout["left"].update(Panel(stats, title="📊 Stats", border_style="dim"))

                # Right: confidence gauge
                conf_bar_width = 20
                conf_filled = int(confidence / 100 * conf_bar_width)
                conf_empty = conf_bar_width - conf_filled

                if confidence < 30:
                    conf_color = "red"
                elif confidence < 60:
                    conf_color = "yellow"
                elif confidence < 85:
                    conf_color = "green"
                else:
                    conf_color = "bold bright_green"

                conf_bar = f"[{conf_color}]{'█' * conf_filled}[/{conf_color}][dim]{'░' * conf_empty}[/dim]"

                gauge_text = Text()
                gauge_lines = [
                    "",
                    "  Neural Confidence",
                    "",
                    f"  {conf_bar}  {confidence:.1f}%",
                    "",
                    "  ╔══════════════════════╗",
                    f"  ║  Pruning Efficiency  ║",
                    f"  ║     {pruned:>6.1f}%           ║",
                    "  ╚══════════════════════╝",
                ]

                right_panel = Panel(
                    "\n".join([
                        "",
                        "  [bold]Neural Confidence[/bold]",
                        "",
                        f"  {conf_bar}  {confidence:.1f}%",
                        "",
                        f"  [dim]Pruning Efficiency:[/dim] [green]{pruned:.1f}%[/green]",
                        f"  [dim]Hamming Distance:[/dim]  [yellow]{best_hamming:.1f}[/yellow]",
                        "",
                        f"  [dim]Phase {phase + 1}/5:[/dim] [magenta]{phase_names[phase]}[/magenta]",
                    ]),
                    title="🧠 Neural Oracle",
                    border_style="bright_green" if confidence > 60 else "dim",
                )
                body_layout["right"].update(right_panel)

                layout["body"].update(body_layout)

                # Footer
                layout["footer"].update(Panel(
                    "[dim]Press Ctrl+C to stop · Simulation Mode · "
                    "Model training in progress[/dim]",
                    box=box.MINIMAL,
                    style="dim",
                ))

                live.update(layout)
                time.sleep(0.12)

                # "Complete" at high confidence
                if confidence > 99.5:
                    time.sleep(1)
                    break

    except KeyboardInterrupt:
        pass

    console.print()
    elapsed = time.time() - start_time

    # Final summary
    summary = Table(
        title="🔍 Crack Session Summary",
        box=box.DOUBLE_EDGE,
        border_style="bright_cyan",
    )
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")

    summary.add_row("Target Hash", f"{target[:16]}...{target[-16:]}")
    summary.add_row("Difficulty", f"{difficulty} rounds")
    summary.add_row("Duration", f"{elapsed:.1f}s")
    summary.add_row("Candidates Checked", format_number(candidates))
    summary.add_row("Search Space Pruned", f"{pruned:.1f}%")
    summary.add_row("Final Confidence", f"{confidence:.1f}%")
    summary.add_row("Best Hamming Distance", f"{best_hamming:.1f} / 256")

    console.print(summary)
    console.print()
    console.print(Panel(
        "[dim]This is a simulation. The real model is currently in active training.\n"
        "Results shown are simulated to demonstrate the interface.[/dim]",
        title="ℹ️  Simulation Mode",
        border_style="dim",
    ))


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Neuro-CLI: Interactive Demo Interface for Neuro-SHA-M4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 neuro_cli.py --mode race     Watch AI vs Brute Force race
  python3 neuro_cli.py --mode crack    Interactive hash cracker dashboard
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["race", "crack"],
        required=True,
        help="Demo mode: 'race' (AI vs Z3) or 'crack' (interactive dashboard)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (optional, uses simulation if not provided)",
    )

    args = parser.parse_args()

    if args.model:
        console.print(f"[yellow]⚠️  Model loading not yet implemented. Running in simulation mode.[/yellow]\n")

    if args.mode == "race":
        run_race()
    elif args.mode == "crack":
        run_crack()


if __name__ == "__main__":
    main()
