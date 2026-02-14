#!/usr/bin/env python3
"""
NanoLogic TUI v3.0 - CyberSystem Edition.
Powered by Textual & Rich.
"""
import sys
import os
import inspect
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog
from textual.containers import Container
from rich.text import Text
from rich.panel import Panel
from rich.align import Align

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Tools
from src.tools import practical, fun, execution, core

class NanoLogicApp(App):
    """
    Cyber-Aesthetic Command Center for Neuro-SHA-M4.
    """
    
    CSS = """
    Screen {
        background: #050505;
        color: #00ff41;
    }
    
    Header {
        background: #0d0d0d;
        color: #00ff41;
        dock: top;
        height: 3;
        content-align: center middle;
        text-style: bold;
        border-bottom: solid #00ff41;
    }

    #main_container {
        height: 1fr;
        border: heavy #00ff41;
        background: #0a0a0a;
        margin: 1 2;
        padding: 1;
    }

    RichLog {
        background: #0a0a0a;
        color: #e0f2f1;
        scrollbar-gutter: stable;
        overflow-y: scroll;
        border: none;
    }

    Input {
        dock: bottom;
        height: 3;
        margin: 0 2 1 2;
        border: solid #00ff41;
        background: #111;
        color: #00ff41;
    }
    Input:focus {
        border: double #00ff41;
    }

    .timestamp {
        color: #555555;
    }
    """

    TITLE = "NANO.LOGIC // NEURO-SHA-M4"
    SUB_TITLE = "CORE.ACTIVE"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main_container"):
            yield RichLog(id="log-view", markup=True, wrap=True)
        yield Input(placeholder="ENTER COMMAND (type 'help' for core tools)...", id="cmd-input")
        yield Footer()

    def on_mount(self) -> None:
        self.log_view = self.query_one(RichLog)
        
        # Welcome Message
        welcome = Panel(
            Align.center(
                "[bold green]NEURO-SHA-M4 SYSTEM ONLINE[/]\n"
                "[dim]Target: SHA-256 Symbolic Cryptanalysis[/]\n"
                "[dim]Hardware: Apple Silicon M4 Neural Engine[/]\n\n"
                "Type [bold cyan]help[/] for core project tools."
            ),
            border_style="green",
            title="SYSTEM BOOT"
        )
        self.log_view.write(welcome)
        
        # Load Core Tools by Default
        self.tools = {}
        self._register_module(core)
        self.unlocked = False
        
        self.log_view.write(f"[dim]Core Modules Loaded: {len(self.tools)} commands active.[/]")

    def _register_module(self, module):
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("_"):
                self.tools[name] = func

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        raw_cmd = message.value.strip()
        message.input.value = ""
        if not raw_cmd: return

        self.log_view.write(f"[bold cyan]>[/] [white]{raw_cmd}[/]")
        
        parts = raw_cmd.split()
        cmd = parts[0].lower()
        args = parts[1:]

        # 1. Internal Router
        if cmd in ["exit", "quit"]:
            self.exit()
        elif cmd == "help":
            self.show_help()
        elif cmd == "clear":
            self.log_view.clear()
            self.log_view.write("[dim]Console Cleared.[/]")
        elif cmd == "unlock":
            if not self.unlocked:
                self._register_module(practical)
                self._register_module(fun)
                self.unlocked = True
                self.log_view.write("[bold yellow]🔓 Access Unlocked: Auxiliary and Entertainment modules online.[/]")
            else:
                self.log_view.write("[dim]System already fully unlocked.[/]")
        
        # 2. Tool Router
        elif cmd == "crack":
            # Launch interactive screen instead of static output
            from src.tui.screens import CrackScreen
            target = args[0] if args else ""
            self.push_screen(CrackScreen(target_hash=target))
            
        elif cmd in self.tools:
            # Use smart execution engine
            result = execution.execute_tool(self.tools[cmd], args)
            self.log_view.write(result)
        
        else:
            self.log_view.write(f"[bold red]UNKNOWN COMMAND:[/] {cmd}")

    def show_help(self):
        from rich.table import Table
        
        table = Table(title="AVAILABLE COMMANDS", border_style="green", box=None)
        table.add_column("Command", style="bold cyan")
        table.add_column("Module", style="dim")
        table.add_column("Description", style="white")
        
        # Sort tools
        for name in sorted(self.tools.keys()):
            func = self.tools[name]
            module_name = func.__module__.split(".")[-1]
            doc = (inspect.getdoc(func) or "").split("\n")[0]
            table.add_row(name, module_name, doc)
            
        self.log_view.write(table)

if __name__ == "__main__":
    app = NanoLogicApp()
    app.run()
