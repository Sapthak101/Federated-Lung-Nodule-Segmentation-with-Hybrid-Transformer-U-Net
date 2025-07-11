# ------------------------------
# 🌐 Federated Learning Server Setup
# ------------------------------

import flwr as fl
import sys
import numpy as np

# Define the strategy (FedAvg is default)
strategy = fl.server.strategy.FedAvg()

# Start the Flower server
fl.server.start_server(
    server_address="127.0.0.1:18080", 
    config=fl.server.ServerConfig(num_rounds=4),
    strategy=strategy
)