"""
Execution Engine for NanoLogic Tools.
Handles smart argument parsing, type casting, and error management.
"""
import inspect
import shlex
from typing import Any, Callable, Dict, List, get_type_hints

def execute_tool(func: Callable, args: List[str]) -> Any:
    """
    Execute a tool function with smart argument parsing.
    
    Args:
        func: The function to execute.
        args: List of string arguments from the CLI.
        
    Returns:
        The result of the function call.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    hints = get_type_hints(func)
    
    # 1. Handle "catch-all" text argument
    # If function has exactly 1 argument named 'text', 'msg', 'query', or 'prompt',
    # and we have multiple args, join them into one string.
    if len(params) == 1 and params[0].name in ['text', 'msg', 'query', 'prompt', 'val'] and len(args) > 1:
        args = [" ".join(args)]
        
    # 2. Map arguments to parameters
    mapped_args = []
    
    for i, param in enumerate(params):
        if i < len(args):
            val = args[i]
            
            # Type Casting
            target_type = hints.get(param.name)
            if target_type:
                try:
                    if target_type == int:
                        # Handle hex/bin strings for int
                        if val.lower().startswith("0x"):
                            val = int(val, 16)
                        elif val.lower().startswith("0b"):
                            val = int(val, 2)
                        else:
                            val = int(val)
                    elif target_type == float:
                        val = float(val)
                    elif target_type == bool:
                        val = val.lower() in ('true', '1', 'yes', 'on')
                except ValueError:
                    # If cast fails, keep as string (or let function error out)
                    pass
            
            mapped_args.append(val)
        else:
            # No argument provided for this parameter
            if param.default == inspect.Parameter.empty:
                 # Missing required argument
                 return f"❌ Missing argument: {param.name}"
            # Use default
            mapped_args.append(param.default)
            
    try:
        if len(args) > len(params) and not any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
             # Too many args, but we might want to be lenient? 
             # For now, strict: warn user
             pass 

        return func(*mapped_args)
    except Exception as e:
        return f"❌ Execution Error: {e}"

def get_tool_help(func: Callable) -> str:
    """Generate a help string for a function."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or "No description."
    params = []
    for param in sig.parameters.values():
        default = "" if param.default == inspect.Parameter.empty else f"={param.default}"
        params.append(f"{param.name}{default}")
    
    usage = f"{func.__name__} {' '.join(params)}"
    return f"[bold cyan]{usage}[/]\n[dim]{doc}[/]"
