import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# Regex patterns
# -----------------------------
header_re = re.compile(r"^---\s+([^-]+(?:-tol[0-9.]+)?)-(\d+)\s+---\s*$")
l1_re = re.compile(r"^vCPU\s+\d+\s*-\s*L1:\s*accesses=(\d+),\s*hits=(\d+),\s*misses=(\d+)\s*$")
l2_re = re.compile(r"^L2\s*\(.*?\):\s*accesses=(\d+),\s*hits=(\d+),\s*misses=(\d+)\s*$")
l3_re = re.compile(r"^L3\s*\(.*?\):\s*accesses=(\d+),\s*hits=(\d+),\s*misses=(\d+)\s*$")
task_wait_re = re.compile(r"^Task\s+\d+\s+-\s+.*Waiting time:\s*([0-9.eE+-]+)s")

# -----------------------------
# Strategy sorting
# -----------------------------
def strategy_sort_key(name: str):
    if name == "Balanced":
        return (0, 0.0)
    if name == "Jaccard":
        return (1, 0.0)
    if name.startswith("CDF-tol"):
        try:
            tol = float(name.split("tol", 1)[1])
        except Exception:
            tol = 999.0
        return (2, tol)
    return (3, 0.0)

# -----------------------------
# Parse cache results
# -----------------------------
def parse_and_aggregate_cache(filename: str):
    """
    Parse results.txt and aggregate raw hits/accesses per execution per strategy.
    Returns: dict[strategy] -> list of dicts per run with levels L1/L2/L3
    """
    runs = defaultdict(list)
    current_strategy = None
    run_hits = {"L1": 0, "L2": 0, "L3": 0}
    run_accesses = {"L1": 0, "L2": 0, "L3": 0}
    in_block = False

    def flush_block():
        nonlocal current_strategy, run_hits, run_accesses, in_block
        if in_block and current_strategy is not None:
            runs[current_strategy].append({
                lvl: (run_hits[lvl], run_accesses[lvl])
                for lvl in ("L1", "L2", "L3")
            })
        current_strategy = None
        run_hits = {"L1": 0, "L2": 0, "L3": 0}
        run_accesses = {"L1": 0, "L2": 0, "L3": 0}
        in_block = False

    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()
            m = header_re.match(line)
            if m:
                flush_block()
                base_name = m.group(1)
                current_strategy = base_name
                in_block = True
                continue

            if not in_block:
                continue

            for lvl, regex in zip(("L1","L2","L3"), (l1_re,l2_re,l3_re)):
                m_lvl = regex.match(line)
                if m_lvl:
                    hits = int(m_lvl.group(2))
                    misses = int(m_lvl.group(3))
                    acc = hits + misses
                    run_hits[lvl] += hits
                    run_accesses[lvl] += acc
                    break

    flush_block()
    return runs

# --------------------------------
# Compute mean + std per strategy
# --------------------------------

def compute_mean_std(runs):
    """
    Compute mean and stddev of hit rates per level per strategy.
    Sum hits & accesses per run, then mean/std across runs.
    Returns: dict[strategy] -> dict[level] -> {'mean':..., 'std':...}
    """
    summary = {}
    for strat, run_list in runs.items():
        summary[strat] = {}
        for lvl in ("L1","L2","L3"):
            # Compute hit rates per run
            hrates = []
            for run in run_list:
                hits, accesses = run[lvl]
                hr = hits / accesses if accesses > 0 else 0.0
                hrates.append(hr)
            summary[strat][lvl] = {
                "mean": np.mean(hrates),
                "std": np.std(hrates)
            }
    return summary

# -----------------------------
# Normalize to baseline
# -----------------------------

def normalize_to_baseline(summary, baseline_key="Balanced"):
    base = summary[baseline_key]
    norm = {}
    for strat, lvls in summary.items():
        if strat == baseline_key:
            continue
        norm[strat] = {}
        for lvl in ("L1","L2","L3"):
            b_mean = base[lvl]["mean"]
            norm[strat][lvl] = {
                "mean": (lvls[lvl]["mean"]/b_mean) if b_mean>0 else float("nan"),
                "std": (lvls[lvl]["std"]/b_mean) if b_mean>0 else float("nan")
            }
    return norm

# -----------------------------
# Plotting
# -----------------------------

def plot_normalized_cache(norm_rates, baseline_key="Balanced"):
    strategies = sorted(norm_rates.keys(), key=strategy_sort_key)
    levels = ["L1","L2","L3"]
    x = np.arange(len(strategies))
    width = 0.25

    plt.figure(figsize=(12, 6))
    for j, lvl in enumerate(levels):
        vals = [norm_rates[s][lvl]["mean"] for s in strategies]
        errs = [norm_rates[s][lvl]["std"] for s in strategies]
        plt.bar(x + (j-1)*width, vals, width, yerr=errs, capsize=5, label=lvl, zorder=2)

    plt.xticks(x, strategies, rotation=45, ha="right")
    plt.ylabel(f"Normalized Hit Rate (relative to {baseline_key})")
    plt.title("Normalized Hit Rates per Strategy by Cache Level\n(with stddev error bars)")
    plt.axhline(1.0, linestyle="--", color="red", label=f"Baseline = {baseline_key}")
    plt.legend(loc="lower right")
    plt.grid(linestyle="--",alpha=0.5,zorder=1)
    plt.tight_layout()
    plt.savefig("hit_rate.png")
    # plt.show()


# -----------------------------
# Parse task waiting times
# -----------------------------
def parse_task_waiting_times(filename: str):
    runs = defaultdict(list)
    current_strategy = None
    current_waits = []
    in_block = False

    def flush_block():
        nonlocal current_strategy, current_waits, in_block
        if in_block and current_strategy:
            runs[current_strategy].append(current_waits)
        current_strategy = None
        current_waits = []
        in_block = False

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            m = header_re.match(line)
            if m:
                flush_block()
                current_strategy = m.group(1)
                in_block = True
                continue
            if not in_block:
                continue
            m_task = task_wait_re.match(line)
            if m_task:
                current_waits.append(float(m_task.group(1)))
    flush_block()
    return runs

# -----------------------------
# Compute percentiles
# -----------------------------
def compute_percentile_summary(runs, percentile=99, baseline=None, normalize=False, cdf_only=False):
    summary = {}
    raw_means = {}
    raw_stds = {}
    for strat, run_list in runs.items():
        if cdf_only and not strat.startswith("CDF"):
            continue
        pct_values = []
        for run_waits in run_list:
            if run_waits:
                sorted_times = np.sort(run_waits)
                pct = np.percentile(sorted_times, percentile)
                pct_values.append(pct)
        if pct_values:
            raw_means[strat] = np.mean(pct_values)
            raw_stds[strat] = np.std(pct_values)

    if normalize:
        if baseline not in raw_means:
            raise ValueError(f"Baseline '{baseline}' not found.")
        base_mean = raw_means[baseline]
        for strat in raw_means:
            summary[strat] = {"mean": raw_means[strat]/base_mean, "std": raw_stds[strat]/base_mean}
    else:
        for strat in raw_means:
            summary[strat] = {"mean": raw_means[strat], "std": raw_stds[strat]}
    return summary

def plot_percentile(summary, percentile=99, title="", filename="percentile.png", excluded_baseline=None):
    strategies = sorted(summary.keys(), key=strategy_sort_key)

    if excluded_baseline:
        strategies = [s for s in strategies if s != excluded_baseline]
    
    means = [summary[s]["mean"] for s in strategies]
    stds = [summary[s]["std"] for s in strategies]
    x = np.arange(len(strategies))
    plt.figure(figsize=(12,6))
    plt.bar(x, means, yerr=stds, capsize=5, color="skyblue", zorder=2)
    plt.xticks(x, strategies, rotation=45, ha="right")
    ylabel = f"Task Waiting Time p{percentile}"
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axhline(1.0 if "Normalized" in title else 0, linestyle="--", color="red", label="Baseline" if "Normalized" in title else "")
    plt.grid(linestyle="--",alpha=0.5,zorder=1)
    plt.tight_layout()
    plt.savefig(filename)

# -----------------------------
# Parse execution's time line
# -----------------------------

def parse_time_line(line):
    """Parse the first line: total_time_s, contention_ns, scheduling_ns, ..."""
    parts = line.strip().split(",")
    total_s = float(parts[0].replace("s", ""))
    contention_ns = int(parts[1])
    scheduling_ns = int(parts[2])
    
    # Convert ns to seconds
    contention_s = contention_ns / 1e9
    scheduling_s = scheduling_ns / 1e9
    remaining_s = total_s - contention_s - scheduling_s
    return total_s, contention_s, scheduling_s, remaining_s

def aggregate_time_data(filename):
    """
    Aggregate time data per strategy (mean over multiple runs).
    Returns: dict[strategy] -> dict with mean times (scheduling, contention, remaining)
    """
    from collections import defaultdict
    import re

    header_re = re.compile(r"^---\s+([^-]+(?:-tol[0-9.]+)?)-\d+\s+---")
    times = defaultdict(list)
    current_strategy = None
    parsed_first_line = False  # track if we parsed the first line after header

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            m = header_re.match(line)
            if m:
                current_strategy = m.group(1)
                parsed_first_line = False  # reset for new block
                continue

            if current_strategy and not parsed_first_line:
                # This is the first line after the header
                parts = line.split(",")
                if len(parts) >= 3:
                    total_s = float(parts[0].replace("s", ""))
                    contention_s = int(parts[1]) / 1e9
                    scheduling_s = int(parts[2]) / 1e9
                    remaining_s = total_s - contention_s - scheduling_s
                    times[current_strategy].append((scheduling_s, contention_s, remaining_s))
                    parsed_first_line = True  # only parse once per block

    # Compute mean for each strategy
    means = {}
    for strat, runs in times.items():
        runs_arr = np.array(runs)
        mean_vals = np.mean(runs_arr, axis=0)
        means[strat] = {"scheduling": mean_vals[0],
                        "contention": mean_vals[1],
                        "remaining": mean_vals[2]}
    return means


def plot_stacked_times(time_means):
    strategies = sorted(time_means.keys(), key=strategy_sort_key)
    sched = [time_means[s]["scheduling"] for s in strategies]
    cont  = [time_means[s]["contention"] for s in strategies]
    rest  = [time_means[s]["remaining"] for s in strategies]

    y = np.arange(len(strategies))

    plt.figure(figsize=(12, 6))
    plt.barh(y, sched, color="orange", label="Scheduling time", zorder=2)
    plt.barh(y, cont, left=sched, color="red", label="Contention time", zorder=2)
    plt.barh(y, rest, left=np.array(sched)+np.array(cont), color="skyblue", label="Remaining time", zorder=2)

    plt.yticks(y, strategies)
    plt.xlabel("Time (seconds)")
    plt.title("Execution Time Breakdown per Strategy (mean of runs)")
    plt.grid(linestyle="--",alpha=0.5,zorder=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("stacked_time.png")
    # plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    filename = "results.txt"

    # ---- Cache hit rates ----
    totals = parse_and_aggregate_cache(filename)
    summary = compute_mean_std(totals)
    norm_rates = normalize_to_baseline(summary, baseline_key="Balanced")
    plot_normalized_cache(norm_rates, baseline_key="Balanced")

    # ---- Task waiting times ----
    runs = parse_task_waiting_times(filename)

    # CDF-only p99
    summary_cdf = compute_percentile_summary(runs, percentile=99, normalize=False, cdf_only=True)
    plot_percentile(summary_cdf, percentile=99, title="Task Waiting Time p99 (CDF strategies only)", filename="p99_cdf_only.png")

    # All strategies normalized to Balanced
    summary_all_norm = compute_percentile_summary(runs, percentile=99, baseline="Balanced", normalize=True, cdf_only=False)
    plot_percentile(summary_all_norm, percentile=99, title="Normalized Task Waiting Time p99 (All strategies)", filename="p99_all_norm.png", excluded_baseline="Balanced")

    # Execution's stacked times
    time_means = aggregate_time_data("results.txt")
    plot_stacked_times(time_means)
    
    # Optional: print summary
    print("---- Cache hit rates (absolute) ----")
    for strat in sorted(summary.keys(), key=strategy_sort_key):
        l1,l2,l3 = summary[strat]["L1"]["mean"], summary[strat]["L2"]["mean"], summary[strat]["L3"]["mean"]
        print(f"{strat:>12} | L1={l1:.4f}  L2={l2:.4f}  L3={l3:.4f}")
        
    print("---- Task waiting time percentiles (CDF only) ----")
    for s in sorted(summary_cdf.keys(), key=strategy_sort_key):
        print(f"{s:>12} | mean p99={summary_cdf[s]['mean']:.4f} ± {summary_cdf[s]['std']:.4f}")

    print("---- Task waiting time percentiles normalized to Balanced ----")
    for s in sorted(summary_all_norm.keys(), key=strategy_sort_key):
        print(f"{s:>12} | mean p99 normalized={summary_all_norm[s]['mean']:.4f} ± {summary_all_norm[s]['std']:.4f}")

