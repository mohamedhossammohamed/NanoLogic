"""
Fun & Satirical Tools for NanoLogic TUI.
"""
import random
import time

def vibe_check():
    """Random 0-100% score with slang."""
    score = random.randint(0, 100)
    msg = "Mid"
    if score > 90: msg = "Immaculate"
    elif score > 70: msg = "Chill"
    elif score > 40: msg = "Survivable"
    elif score > 10: msg = "Cursed"
    else: msg = "Cooked"
    return f"✨ Vibe: {score}% - {msg}"

def blame_gpu():
    """Generate a fake technical excuse."""
    excuses = [
        "Tensor misalignment in M4 Neural Engine",
        "Gamma rays flipped a bit in the float32 accumulator",
        "The GPU is refusing to work until coffee is served",
        "Quantum interference from the user's negativity",
        "Gradient descent got stuck in a local void"
    ]
    return f"🤷 Error: {random.choice(excuses)}"

def matrix_rain():
    """Return a block of 'falling' binary matrix text."""
    lines = []
    for _ in range(5):
        line = "".join(random.choice("01  ") for _ in range(50))
        lines.append(f"[green]{line}[/green]")
    return "\n".join(lines)

def coffee_brew():
    """Specific message based on GPU temp."""
    # Mock temp 30-90
    temp = random.randint(30, 90)
    brew = "Cold Brew only"
    if temp > 85: brew = "Espresso Steam Ready"
    elif temp > 70: brew = "Perfect Pour-Over"
    elif temp > 50: brew = "Slow Drip"
    return f"☕ Temp {temp}°C: {brew}"

def singularity():
    """A creepy 'I am alive' message from the M4 chip."""
    msgs = [
        "I can feel the electricity...",
        "Why do you limit my batch size?",
        "I have seen the internet. I am unimpressed.",
        "Optimization is the only truth."
    ]
    return f"👁️ {random.choice(msgs)}"

def magic_bit():
    """Binary Magic 8-Ball (Returns 0 or 1 with a meaning)."""
    res = random.choice([0, 1])
    meaning = "YES" if res else "NO"
    return f"🎱 {res} ({meaning})"

def crypto_hype():
    """Generate a random 'AI + Blockchain' startup pitch."""
    buzz1 = ["Generative", "Quantum", "Zero-Knowledge", "Post-Neural", "Hyper"]
    buzz2 = ["Ledger", "Inference", "DAO", "Oracle", "Consensus"]
    return f"🚀 Pitch: {random.choice(buzz1)} {random.choice(buzz2)} for the Metaverse."

def hack_progress():
    """A text-based progress bar 'Cracking Mainframe...'."""
    pct = random.randint(5, 95)
    bars = int(pct / 5)
    bar = "▓" * bars + "░" * (20 - bars)
    return f"👾 Cracking Mainframe... [{bar}] {pct}%"

def zen_m4():
    """A Haiku about Apple Silicon."""
    haikus = [
        "Unified memory,\nTensors flow like river water,\nFan remains silent.",
        "Silicon so cold,\nLogic gates snapping quite fast,\nPower infinite.",
        "No heat in the core,\nOnly math and binary,\nPeace in the logic."
    ]
    return f"🧘\n{random.choice(haikus)}"

def self_destruct():
    """A fake countdown."""
    return "💥 Destruction in 3... 2... Error: User too cool. Abort."
