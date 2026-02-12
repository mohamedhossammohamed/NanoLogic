import sys
import os
import inspect
from rich.console import Console
from rich.panel import Panel

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools import execution, practical, fun

console = Console()

def test_execution():
    print("Testing Execution Engine...")
    
    # Test valid int conversion
    def types_int(x: int):
        return x * 2
    
    res = execution.execute_tool(types_int, ["10"])
    assert res == 20, f"Failed int conversion: {res} (Expected 20)"
    
    # Test hex conversion
    def types_hex(x: int):
        return x
        
    res = execution.execute_tool(types_hex, ["0x10"])
    assert res == 16, f"Failed hex conversion: {res}"
    
    # Test catch-all
    def types_text(text: str):
        return text
        
    res = execution.execute_tool(types_text, ["hello", "world"])
    assert res == "hello world", f"Failed catch-all: {res}"
    
    print("✅ Execution Engine: PASS")

def test_tools_rich_compatibility():
    print("Testing Tool Return Values...")
    
    # Practical
    p_tools = [practical.sys_temp, practical.disk_map, practical.net_speed]
    for tool in p_tools:
        res = tool()
        # Just check it doesn't crash and returns something
        assert res is not None
        # We expect Rich objects (Panel, Table) or strings
        console.print(res)
        
    # Fun
    f_tools = [fun.vibe_check, fun.matrix_rain]
    for tool in f_tools:
        res = tool()
        assert res is not None
        console.print(res)
        
    print("✅ Tool Compatibility: PASS")

if __name__ == "__main__":
    try:
        test_execution()
        test_tools_rich_compatibility()
        print("🎉 ALL TESTS PASSED")
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        sys.exit(1)
