import re
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_simulation_file(filepath):
    algo_regex = re.compile(r'^(round-?\s*robin|jaccard|cdf|balanced|graph)', re.IGNORECASE)
    header_regex = re.compile(r'^(\d+\.\d+)s,(\d+),(\d+),([\d.]+)')
    task_regex = re.compile(
        r'^Task (\d+) - Response time: ([\d.]+)(ms|s), '
        r'Waiting time: ([\d.]+)s, '
        r'Execution Time: ([\d.]+)µs'
    )
    vcpu_regex = re.compile(
        r'^vCPU (\d+) - L1: accesses=(\d+), hits=(\d+), misses=(\d+)'
    )
    l2_regex = re.compile(
        r'^L2 \(grupo (\d+)\): accesses=(\d+), hits=(\d+), misses=(\d+)'
    )
    l3_regex = re.compile(
        r'^L3 \(global\): accesses=(\d+), hits=(\d+), misses=(\d+)'
    )

    results = {}
    current_algo = None
    current_run = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            algo_match = algo_regex.match(line)
            if algo_match:
                current_algo = algo_match.group(1).strip().title()
                results[current_algo] = []
                continue

            if re.match(r'^\d+\s*-\s*$', line):
                if current_algo is None:
                    continue
                current_run = {
                    "header": None,
                    "tasks": [],
                    "vcpus": [],
                    "l2": [],
                    "l3": None
                }
                results[current_algo].append(current_run)
                continue

            header_match = header_regex.match(line)
            if header_match and current_run:
                current_run["header"] = {
                    "total_time_s": float(header_match.group(1)),
                    "total_contention_ns": int(header_match.group(2)),
                    "total_sched_time_ns": int(header_match.group(3)),
                    "ignored_value": float(header_match.group(4))
                }
                continue

            task_match = task_regex.match(line)
            if task_match and current_run:
                task_id = int(task_match.group(1))
                resp_time = float(task_match.group(2))
                resp_unit = task_match.group(3)
                if resp_unit == 'ms':
                    resp_time /= 1000.
                waiting_time = float(task_match.group(4))
                exec_time_us = float(task_match.group(5))
                current_run["tasks"].append({
                    "id": task_id,
                    "response_time_s": resp_time,
                    "waiting_time_s": waiting_time,
                    "execution_time_us": exec_time_us
                })
                continue

            vcpu_match = vcpu_regex.match(line)
            if vcpu_match and current_run:
                current_run["vcpus"].append({
                    "id": int(vcpu_match.group(1)),
                    "accesses": int(vcpu_match.group(2)),
                    "hits": int(vcpu_match.group(3)),
                    "misses": int(vcpu_match.group(4))
                })
                continue

            l2_match = l2_regex.match(line)
            if l2_match and current_run:
                current_run["l2"].append({
                    "group": int(l2_match.group(1)),
                    "accesses": int(l2_match.group(2)),
                    "hits": int(l2_match.group(3)),
                    "misses": int(l2_match.group(4))
                })
                continue

            l3_match = l3_regex.match(line)
            if l3_match and current_run:
                current_run["l3"] = {
                    "accesses": int(l3_match.group(1)),
                    "hits": int(l3_match.group(2)),
                    "misses": int(l3_match.group(3))
                }
                continue

    return results

def plot_normalized_cache_hit_rates(dados):
    algos = list(dados.keys())

    def calc_means_stds(algo_runs):
        l1_vals = []
        l2_vals = []
        l3_vals = []
        for run in algo_runs:
            l1_hits = sum(vcpu['hits'] for vcpu in run['vcpus'])
            l1_access = sum(vcpu['accesses'] for vcpu in run['vcpus'])
            l2_hits = sum(l2['hits'] for l2 in run['l2'])
            l2_access = sum(l2['accesses'] for l2 in run['l2'])
            l3_hits = run['l3']['hits']
            l3_access = run['l3']['accesses']

            l1_vals.append(l1_hits / l1_access if l1_access > 0 else 0)
            l2_vals.append(l2_hits / l2_access if l2_access > 0 else 0)
            l3_vals.append(l3_hits / l3_access if l3_access > 0 else 0)
        return (
            np.mean(l1_vals), np.std(l1_vals),
            np.mean(l2_vals), np.std(l2_vals),
            np.mean(l3_vals), np.std(l3_vals)
        )

    stats = {}
    for algo in algos:
        stats[algo] = calc_means_stds(dados[algo])

    l1_base, _, l2_base, _, l3_base, _ = stats["Balanced"]
    algos_no_balanced = [a for a in algos if a != "Balanced"]

    l1_means_norm = []
    l1_stds_norm = []
    l2_means_norm = []
    l2_stds_norm = []
    l3_means_norm = []
    l3_stds_norm = []

    for algo in algos_no_balanced:
        l1_mean, l1_std, l2_mean, l2_std, l3_mean, l3_std = stats[algo]
        l1_means_norm.append(l1_mean / l1_base if l1_base != 0 else 0)
        l1_stds_norm.append(l1_std / l1_base if l1_base != 0 else 0)
        l2_means_norm.append(l2_mean / l2_base if l2_base != 0 else 0)
        l2_stds_norm.append(l2_std / l2_base if l2_base != 0 else 0)
        l3_means_norm.append(l3_mean / l3_base if l3_base != 0 else 0)
        l3_stds_norm.append(l3_std / l3_base if l3_base != 0 else 0)

    x = np.arange(len(algos_no_balanced))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width, l1_means_norm, width, yerr=l1_stds_norm, capsize=5, label='L1 Cache Hit Rate')
    rects2 = ax.bar(x, l2_means_norm, width, yerr=l2_stds_norm, capsize=5, label='L2 Cache Hit Rate')
    rects3 = ax.bar(x + width, l3_means_norm, width, yerr=l3_stds_norm, capsize=5, label='L3 Cache Hit Rate')

    ax.axhline(1, color='black', linestyle='--', linewidth=1, label='Balanced baseline (1.0)')

    ax.set_ylabel('Normalized Cache Hit Rate (relative to Balanced)')
    ax.set_title('Normalized Cache Hit Rates by Algorithm (with Std Dev)')
    ax.set_xticks(x)
    ax.set_xticklabels(algos_no_balanced)
    ax.legend(loc="lower right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = "normalized_cache_hit_rates_with_std.png"
    filepath = os.path.join(os.getcwd(), filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


def plot_random_waiting_times_cdf(dados):
    random_run = 1

    plt.figure(figsize=(10, 6))

    balanced_99_time = None
        
    for algo, runs in dados.items():
        run_data = runs[random_run]
        waiting_times = sorted(task["waiting_time_s"] for task in run_data["tasks"])
        percentiles = [(i + 1) / len(waiting_times) * 100 for i in range(len(waiting_times))]

        plt.plot(waiting_times, percentiles, linestyle="--", label=algo)


        if algo.lower() == "balanced":
            idx_99 = int(len(waiting_times) * 0.99) - 1
            idx_99 = max(0, min(idx_99, len(waiting_times) - 1))
            balanced_99_time = waiting_times[idx_99]

    if balanced_99_time is not None:
        plt.axvline(x=balanced_99_time, color='red', linestyle=':', linewidth=2,
                    label='Balanced 99th percentile')
        
    plt.xlabel("Waiting time (s)")
    plt.ylabel("Task (%)")
    plt.title(f"Cumulative Distribution Function of Tasks' Waiting Times by Strategy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"waiting_times_cdf_run_{random_run}.png"
    filepath = os.path.join(os.getcwd(), filename)

    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")

def plot_stacked_contention_sched_times(dados):
    algos = list(dados.keys())


    sched_means = []
    contention_means = []

    for algo in algos:
        sched_times = []
        contention_times = []
        for run in dados[algo]:
            if run["header"] is not None:
                sched_times.append(run["header"]["total_sched_time_ns"])
                contention_times.append(run["header"]["total_contention_ns"])

        avg_sched = np.mean(sched_times) if sched_times else 0
        avg_contention = np.mean(contention_times) if contention_times else 0
        sched_means.append(avg_sched / 1_000_000)
        contention_means.append(avg_contention / 1_000_000)

    diff_means = [max(c - s, 0) for c, s in zip(contention_means, sched_means)]

    y_pos = np.arange(len(algos))

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.barh(y_pos, sched_means, color='tab:blue', label='Avg Total Sched Time (ns)')
    ax.barh(y_pos, diff_means, left=sched_means, color='tab:orange', label='Avg Contention Time (ns) excluding Sched')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(algos)
    ax.set_xlabel('Time (miliseconds)')
    ax.set_title('Average Total Contention and Scheduling Times by Algorithm')
    ax.legend()
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = "stacked_contention_sched_times.png"
    filepath = os.path.join(os.getcwd(), filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")

    

if __name__ == "__main__":
    arquivo = "partial_results.txt"
    dados = parse_simulation_file(arquivo)
    plot_random_waiting_times_cdf(dados)
    plot_normalized_cache_hit_rates(dados)
    plot_stacked_contention_sched_times(dados)
