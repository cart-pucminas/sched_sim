import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join, realpath

PROJ_ROOT = join(dirname(dirname(dirname(realpath(__file__)))))
CODE_DIR = join(PROJ_ROOT, "bin", "v_sim")
PLOTS_DIR = join(PROJ_ROOT, "src", "v_sim", "plot")
BINARY_PATH = join("bin", "v_sim")  
FIGSIZE = (8, 4)
COLORS = {
    "DARK_RED"  : "#8B0000",
    "DARK_GREEN": "#006400",
    "DARK_BLUE": "#00008B",
}

NUM_ITERATIONS = 10
base_tasks = 25
num_threads1 = [1, 2, 4, 8, 12]
num_threads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
num_cpus = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def run_iteration(args):
    try:
        result = subprocess.run(
            [BINARY_PATH] + args,
            capture_output=True,
            text=True,
            check=True
        )
        line = result.stdout.strip()
        parts = line.split(",")

        exec_time_s = float(parts[0].strip().replace("s", ""))
        contention_ns = int(parts[1].strip())
        contention_ratio = float(parts[2].strip())
        return exec_time_s, contention_ns, contention_ratio

    except Exception as e:
        print(f"Erro na execução: {e}")
        return None
    
def results_threads_strong():
    times_per_thread = {}

    for threads in num_threads:
        print(f"\n>> Testando: -num_vcpus {threads}")
        times = []
        for i in range(NUM_ITERATIONS):
            print(f"  Iteração {i+1}/{NUM_ITERATIONS}...", end="\r")
            args = ["--num_vcpus", str(threads), "--seed", str(i)]
            result = run_iteration(args)
            if result:
                times.append(result[0])  # apenas tempo de execução

        if not times:
            print(f"[ERRO] Nenhum resultado válido para {threads} threads.")
            continue

        times_per_thread[threads] = times
        print(f"\n   >> Média de tempo {np.mean(times)}")

    return times_per_thread
    
def plot_normalized_speedup(times_per_thread):
    if 1 not in times_per_thread:
        print("Erro: tempo com 1 thread (sequencial) necessário para normalização.")
        return

    t_seq = np.mean(times_per_thread[1])
    # threads = sorted([t for t in times_per_thread.keys() if t != 1])
    threads = sorted([t for t in times_per_thread.keys() if t in num_threads1 and t != 1])

    speedups = []
    std_errors = []

    for t in threads:
        avg = np.mean(times_per_thread[t])
        speedup = t_seq / avg
        sem = np.std(times_per_thread[t]) / np.sqrt(NUM_ITERATIONS)

        speedups.append(speedup)
        std_errors.append(sem)

    color_palette = plt.get_cmap('tab10')
    bar_colors = [color_palette(i) for i in range(len(threads))]

    plt.figure(figsize=FIGSIZE)
    x_labels = [str(t) for t in threads]
    x_pos = np.arange(len(threads))

    bars = plt.bar(
        x_pos,
        speedups,
        yerr=std_errors,
        align="center",
        alpha=0.9,
        capsize=6,
        color=bar_colors,
        edgecolor="black",
        label="Speedup"
    )

    plt.axhline(y=1.0, color='blue', linestyle='--', label="Sequential Execution")
    plt.xticks(x_pos, x_labels)
    plt.ylabel("Speedup")
    plt.xlabel("Number of vCPUs")
    plt.title("Speedup (Comp. to sequential execution)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.legend()

    plt.savefig(f"{PLOTS_DIR}_speedup_forte.png")
    print(f"Gráfico salvo em: {PLOTS_DIR}")

def plot_efficiency_vs_speedup(times_per_thread):
    if 1 not in times_per_thread:
        print("Erro: tempo com 1 thread (sequencial) necessário para normalização.")
        return

    t_seq = np.mean(times_per_thread[1])
    threads = sorted([t for t in times_per_thread.keys() if t != 1])

    speedups = []
    efficiencies = []
    labels = []

    for (i, t) in enumerate(threads):
        times = times_per_thread[t]
        avg = np.mean(times)
        speedup = t_seq / avg
        efficiency = speedup / num_cpus[i]

        speedups.append(speedup)
        efficiencies.append(efficiency)
        labels.append(str(t))

    speedups_np = np.array(speedups)
    efficiencies_np = np.array(efficiencies)
        
    poly3 = np.poly1d(np.polyfit(speedups_np, efficiencies_np, deg=3))
    y_poly3 = poly3(speedups_np)

    poly1 = np.poly1d(np.polyfit(speedups_np, efficiencies_np, deg=1))
    y_poly1 = poly1(speedups_np)


    plt.figure(figsize=FIGSIZE)
    scatter = plt.scatter(speedups_np, efficiencies_np, c=range(len(threads)), cmap='viridis', s=80, edgecolors='black')

    for i, label in enumerate(labels):
        plt.annotate(f"{label} vCPU", (speedups_np[i] + 0.15, efficiencies_np[i]), fontsize=9)

    x_fit = np.linspace(min(speedups), max(speedups), 200)
    plt.plot(x_fit, poly3(x_fit), color=COLORS["DARK_RED"], linestyle='--', linewidth=2, label=f'Sim. Behaviour')
    # plt.plot(x_fit, poly1(x_fit), color=COLORS["DARK_GREEN"], linestyle='--', linewidth=2, label=f'Linear ')
    plt.axhline(y=1.0, color='blue', linestyle='--', label="Sequential Execution")
    plt.axvline(x=1.0, color='blue', linestyle='--')

    plt.xlabel("Speedup")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Speedup")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}_efficiency_speedup_forte.png")
    print(f"Gráfico salvo em: {PLOTS_DIR}")








    
#################################
## WEAK
################################





def results_threads_weak():
    times_per_thread = {}

    num_tasks_list = [base_tasks * t for t in num_threads]

    for threads, tasks in zip(num_threads, num_tasks_list):
        print(f"\n>> Testando: -num_vcpus {threads} -num_tasks {tasks}")
        times = []
        for i in range(NUM_ITERATIONS):
            print(f"  Iteração {i+1}/{NUM_ITERATIONS}...", end="\r")
            args = ["--num_vcpus", str(threads), "--num_tasks", str(tasks), "--seed", str(i)]
            result = run_iteration(args)
            if result:
                times.append(result[0])  # apenas tempo de execução

        if not times:
            print(f"[ERRO] Nenhum resultado válido para {threads} threads.")
            continue

        times_per_thread[threads] = times
        print(f"\n   >> Média de tempo {np.mean(times)}")

    return times_per_thread

def plot_weak_scaling_bars(times_per_thread):
    threads = sorted(times_per_thread.keys())

    num_tasks_per_thread = {t: t * base_tasks for t in times_per_thread.keys()}
    
    t_seq = np.mean(times_per_thread[1])

    # Dados para o gráfico
    avg_times = [np.mean(times_per_thread[t]) for t in threads]
    sems = [np.std(times_per_thread[t], ddof=1) / np.sqrt(len(times_per_thread[t])) for t in threads]
    normalized_times = [t / t_seq for t in avg_times]

    # Geração do gráfico
    fig, ax1 = plt.subplots(figsize=FIGSIZE)

    # Barras com tempo normalizado
    bars = ax1.bar(
        [str(t) for t in threads],
        normalized_times,
        yerr=sems,
        capsize=4,
        color="skyblue",
        edgecolor='black',
        label='Norm. Execution Time',
    )

    # # Rótulos numéricos nas barras
    # for bar, value in zip(bars, normalized_times):
    #     height = bar.get_height()
    #     ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{value:.2f}", ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel("Number of vCPUs")
    ax1.set_ylabel("Normalized Execution Time")
    ax1.set_title("Weak Scalability - Normalized Times")
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, max(normalized_times) * 1.3)
    plt.axhline(y=1.0, color='blue', linestyle='--', label="Sequential Execution")

    # Segundo eixo Y para número de tarefas
    ax2 = ax1.twinx()
    tasks = [num_tasks_per_thread[t] for t in threads]
    ax2.plot([str(t) for t in threads], tasks, color=COLORS["DARK_RED"], marker='o', linestyle='--', label='Number of tasks')
    ax2.set_ylabel("Number of Tasks", color=COLORS["DARK_RED"])
    ax2.tick_params(axis='y', labelcolor=COLORS["DARK_RED"])

    # Legendas
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}_fraca.png")
    print(f"Gráfico salvo em: {PLOTS_DIR}")
    
if __name__ == "__main__":
    times = results_threads_strong()
    plot_efficiency_vs_speedup(times)
    plot_normalized_speedup(times)
    # times = results_threads_weak()
    # plot_weak_scaling_bars(times)
