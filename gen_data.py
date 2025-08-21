import subprocess
from datetime import datetime

# schedulers = ["Balanced", "Jaccard", "CDF"]
schedulers = ["Balanced", "CDF"]
tols = [round(x * 0.1, 1) for x in range(1, 10)]  # 0.1 -> 0.9
iterations = 5
output_file = "results.txt"

with open(output_file, "w") as f:
    for scheduler in schedulers:
        if scheduler == "CDF":
            for tol in tols:
                for i in range(1, iterations + 1):
                    tag = f"{scheduler}-tol{tol}-{i}"
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Running {tag} at {timestamp}...")

                    cmd = f"./bin/v_sim --num_vcpus 12 --scheduler {scheduler} --num_tasks 80 --tol {tol}"
        
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    stdout, _ = process.communicate()
                    
                    f.write(f"\n--- {tag} ---\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(stdout)
                    f.write("\n")
                    print(stdout)
        else:
            for i in range(1, iterations + 1):
                tag = f"{scheduler}-{i}"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Running {tag} at {timestamp}...")

                cmd = f"./bin/v_sim --num_vcpus 12 --scheduler {scheduler} --num_tasks 80"

                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                stdout, _ = process.communicate()

                f.write(f"\n--- {tag} ---\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(stdout)
                f.write("\n")
                print(stdout)
