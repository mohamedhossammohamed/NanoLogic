import csv
import os

def analyze_logs(file_path):
    steps = []
    accuracy = []
    loss = []
    phases = []
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row['step']))
                accuracy.append(float(row['accuracy']))
                loss.append(float(row['loss']))
                phases.append(int(row['phase']))
            except (ValueError, KeyError):
                continue

    if not steps:
        print("No data found in log file.")
        return

    # Subsample for display if too long (keep ~30 points for readability)
    display_points = 30
    stride = max(1, len(steps) // display_points)
    s_steps = steps[::stride]
    s_acc = accuracy[::stride]
    s_loss = loss[::stride]
    s_phases = phases[::stride]

    print("\n" + "="*70)
    print("      NANOLOGIC TRAINING ANALYSIS (Visual Report)")
    print("="*70)
    
    # 1. Accuracy Trend
    print("\n--- BIT-LEVEL ACCURACY TREND ---")
    print("(50% = Random | 80%+ = Structural Mapping)")
    print("-" * 65)
    print(f"{'Step':>8} | {'Accuracy Visualization (0% to 100%)':<40} | {'Value'}")
    print("-" * 65)
    for i in range(len(s_steps)):
        bar_len = int(s_acc[i] * 40)
        bar = "#" * bar_len + "-" * (40 - bar_len)
        phase_marker = f" [P{s_phases[i]}]" if (i == 0 or s_phases[i] != s_phases[i-1]) else ""
        print(f"{s_steps[i]:>8} | {bar} | {s_acc[i]:>6.2%}{phase_marker}")
    print("-" * 65)

    # 2. Summary Statistics
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Total Training Steps: {steps[-1]}")
    print(f"Current Phase:        {phases[-1]} (Round Difficulty: {phases[-1]*8+8 if phases[-1] < 4 else 64})")
    print(f"Peak Accuracy:        {max(accuracy):.2%}")
    print(f"Final Accuracy:       {accuracy[-1]:.2%}")
    print(f"Minimum Loss:         {min(loss):.4f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    analyze_logs("logs/training.log")
