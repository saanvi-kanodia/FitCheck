import json
import math
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ===== BENCHMARKS (male defaults) =====
benchmarks = {
    "pushups": {"excellent": 30, "good": 20, "average": 10, "poor": 0},
    "plank_hold": {"excellent": 150, "good": 90, "average": 45, "poor": 0},
    "shuttle_run": {"excellent": 10, "good": 12, "average": 14, "poor": 20},  # lower = better
    "vertical_jump": {"excellent": 60, "good": 50, "average": 40, "poor": 0}
}

# ===== DOMAIN MAPPING =====
test_to_domain = {
    "pushups": "strength",
    "plank": "core",
    "shuttle_run": "agility",
    "vertical_jump": "power"
}

bands = ["poor", "average", "good", "excellent"]

# ===== PARSE TXT FILE =====
def parse_report(filepath):
    """Read exercise txt report and return dict"""
    data = {}
    with open(filepath, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                data[key.strip().lower()] = value.strip()
    print(f"Parsed {filepath}: {data}")  # Add this line
    # Normalize fields
    if "value" in data: data["value"] = float(data["value"])
    if "form_score" in data: data["form_score"] = float(data["form_score"])
    if "max_form_score" in data: data["max_form_score"] = float(data["max_form_score"])
    else: data["max_form_score"] = 4.0
    # Parse session duration if present
    if "session duration" in data:
        try:
            # Accepts 'Session Duration: 11' or 'Session Duration: 11 seconds'
            data["session_duration"] = float(data["session duration"].split()[0])
        except Exception:
            data["session_duration"] = None
    return data

# ===== CALCULATE BAND / PERCENTILE / PENALTY =====
def analyze(test_data):
    test = test_data["exercise"].lower()
    value = float(test_data["value"])
    # Normalize pushups to reps per minute if session_duration is present
    if test == "pushups" and "session_duration" in test_data and test_data["session_duration"]:
        duration = float(test_data["session_duration"])
        if duration > 0:
            value = value / (duration / 60)  # reps per minute
    # Convert vertical jump from meters to centimeters for benchmarking
    if test == "vertical_jump":
        value = value * 100
    form_score = float(test_data.get("form_score", 4))
    max_form_score = float(test_data.get("max_form_score", 4))
    domain = test_to_domain.get(test, "other")

    thresholds = [benchmarks[test][b] for b in bands]

    # --- Band ---
    band_result = "poor"
    for b in bands:
        if test == "shuttle_run":
            if value <= benchmarks[test][b]:
                band_result = b
        else:
            if value >= benchmarks[test][b]:
                band_result = b

    # --- Percentile ---
    if test == "shuttle_run":
        # Lower is better
        if value <= thresholds[0]:
            percentile = 100
        elif value >= thresholds[3]:
            percentile = 0
        else:
            percentile = int(max(0, 100 - (value - thresholds[0]) * 100 / (thresholds[3] - thresholds[0])))
    else:
        if value >= thresholds[3]:
            percentile = 100
        elif value <= thresholds[0]:
            percentile = 0
        else:
            percentile = int((value - thresholds[0]) * 100 / (thresholds[3] - thresholds[0]))

    # --- Penalty (form score out of 4) ---
    penalty = round(100 * (1 - form_score / max_form_score))
    adj_percentile = max(0, round(percentile * (1 - penalty / 100)))

    return {
        "test": test,
        "value": value,
        "band": band_result,
        "percentile": adj_percentile,
        "penalty": penalty,
        "domain": domain
    }

# ===== MAIN PIPELINE =====
def main():
    import glob
    import re
    from datetime import datetime

    reports_dir = "placeholder"

    # Patterns for each test type
    patterns = {
        "pushups": "pushup_report_*.txt",
        "plank": "plank_report_*.txt",
        "shuttle_run": "shuttlerun_report_*.txt",
        "vertical_jump": "verticaljump_report_*.txt"
    }

    def get_timestamp_from_filename(filename):
        match = re.search(r'(\d{8}_\d{6})', filename)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        return None

    results = []
    for test, pattern in patterns.items():
        files = glob.glob(os.path.join(reports_dir, pattern))
        if not files:
            print(f"⚠️ Missing file for {test}")
            continue
        # Sort files by timestamp
        files_with_ts = [(f, get_timestamp_from_filename(f)) for f in files]
        files_with_ts = [x for x in files_with_ts if x[1] is not None]
        if not files_with_ts:
            print(f"⚠️ No valid timestamped file for {test}")
            continue
        # Get most recent file
        most_recent = max(files_with_ts, key=lambda x: x[1])[0]
        test_data = parse_report(most_recent)
        results.append(analyze(test_data))

    # --- Domain averages ---
    domain_scores = {}
    for r in results:
        domain_scores.setdefault(r["domain"], []).append(r["percentile"])
    domain_percentiles = {d: int(sum(vals) / len(vals)) for d, vals in domain_scores.items()}

    # --- Print Exercise Metrics ---
    print("\n====== INDIVIDUAL EXERCISE METRICS ======")
    for r in results:
        print(f"Test: {r['test']} ({r['domain']})")
        print(f"  Value: {r['value']}")
        print(f"  Band: {r['band']}")
        print(f"  Percentile (penalized): {r['percentile']}")
        print(f"  Penalty: {r['penalty']}%\n")

    # --- Print Domain Metrics ---
    print("====== FITNESS DOMAIN METRICS ======")
    print(json.dumps(domain_percentiles, indent=2))

    # --- Save JSON ---
    output_folder = "./placeholder"
    os.makedirs(output_folder, exist_ok=True)

    json_path = os.path.join(output_folder, "fitness_domain_graph.json")
    graph_path = os.path.join(output_folder, "fitness_domain_radar.png")

    with open(json_path, "w") as out:
        json.dump(domain_percentiles, out, indent=2)
    print(f"\n✅ Saved JSON: {json_path}")

    # --- Radar Chart ---
    labels = list(domain_percentiles.keys())
    stats = list(domain_percentiles.values())

    if not labels or not stats:
        print("No data available for radar chart. Please check your report files.")
        return

    labels += [labels[0]]
    stats += [stats[0]]

    angles = np.linspace(0, 2 * np.pi, len(labels))
    
    # Color based on average percentile (red to green)
    avg_percentile = sum(stats[:-1]) / len(stats[:-1])
    color = cm.RdYlGn(avg_percentile / 100)
    print(f"Average percentile: {avg_percentile:.1f}, chart color: {color}")
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, 'o-', linewidth=2, color=color)
    ax.fill(angles, stats, alpha=0.25, color=color)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1])
    ax.set_title("Fitness Domain Radar Chart", size=14)

    plt.savefig(graph_path)
    print(f"✅ Saved Radar Chart: {graph_path}")
    plt.show(block=True)

    # --- Donut Charts for Each Domain ---
    num_domains = len(domain_percentiles)
    cols = 2  # 2 columns
    rows = (num_domains + cols - 1) // cols  # Calculate rows needed
    fig, axes = plt.subplots(rows, cols, figsize=(8, 6))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing
    
    for i, (domain, percentile) in enumerate(domain_percentiles.items()):
        ax = axes[i]
        color = cm.RdYlGn(percentile / 100)
        ax.pie([percentile, 100 - percentile], 
               colors=[color, 'lightgray'], 
               wedgeprops=dict(width=0.3),  # Makes it a donut
               startangle=90)
        ax.set_title(f"{domain.capitalize()}\n{percentile}%", size=12)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    donut_path = os.path.join(output_folder, "donut_charts.png")
    plt.savefig(donut_path)
    print(f"✅ Saved Combined Donut Charts: {donut_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()