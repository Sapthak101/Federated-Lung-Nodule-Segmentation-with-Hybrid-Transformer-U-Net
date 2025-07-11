# run_federated_experiment.py

import subprocess
import sys
import time

def run_clients(model_name):
    print(f"🚀 Starting federated experiment with model: {model_name}")

    processes = []

    # Launch all 5 clients in parallel
    for i in range(1, 6):
        client_file = f"clients/client_{i}.py"
        print(f"🧠 Launching Client {i} with {model_name}...")
        
        # Start each client as a subprocess and pass model_name as argument
        p = subprocess.Popen(["python", client_file, model_name])
        processes.append(p)
        time.sleep(2)  # Slight delay to avoid connection race with Flower server

    # Wait for all client processes to complete
    for p in processes:
        p.wait()

    print("✅ Federated experiment completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❗Usage: python run_federated_experiment.py model_1")
        sys.exit(1)

    model_name = sys.argv[1]
    run_clients(model_name)
