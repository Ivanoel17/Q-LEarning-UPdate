import socket
import json
import numpy as np
import logging

# Constants
HOST = '127.0.0.1'
PORT = 5000
EPSILON = 0.1  # Fixed exploration rate

# Simplified state discretization
POWER_BINS = [5, 15, 25, 30]
BEACON_BINS = [1, 5, 10, 20]
CBR_BINS = [0.0, 0.3, 0.6, 1.0]

# Load the trained Q-table model (already trained and saved as q_table_model.npy)
q_table = np.load('q_table_model.npy')

def discretize(value, bins):
    """Discretize continuous values into indices based on predefined bins."""
    return np.digitize(value, bins) - 1

def select_action(state):
    """Select the action based on the current state using the trained Q-table (greedy selection)."""
    power_idx = discretize(state[0], POWER_BINS)
    beacon_idx = discretize(state[1], BEACON_BINS)
    cbr_idx = discretize(state[2], CBR_BINS)

    # Directly use the model (no epsilon-greedy or random choice)
    return np.argmax(q_table[power_idx, beacon_idx, cbr_idx])  # Choose the best action based on Q-table

def handle_client(conn):
    """Handle incoming client connection and process batch data."""
    while True:
        data = conn.recv(4096)
        if not data:
            break
        
        try:
            # Parse the batch data from client (vehicle information)
            batch_data = json.loads(data.decode())
            logging.info(f"[BATCH] Received {len(batch_data)} vehicles' data")
            response_data = {}

            for veh_id, vehicle_data in batch_data.items():
                # Retrieve vehicle-specific data
                current_cbr = vehicle_data['CBR']
                current_power = vehicle_data['transmissionPower']
                current_beacon = vehicle_data['beaconRate']
                
                # Select the action for the vehicle based on the current state (power, beacon rate, and CBR)
                action = select_action((current_power, current_beacon, current_cbr))
                
                # Apply action to the current state
                if action == 0:  # Increase beacon rate by 1
                    new_beacon = min(20, current_beacon + 1)
                    new_tx_power = current_power
                elif action == 1:  # Decrease beacon rate by 1
                    new_beacon = max(1, current_beacon - 1)
                    new_tx_power = current_power
                elif action == 2:  # Increase tx power by 1
                    new_beacon = current_beacon
                    new_tx_power = min(30, current_power + 1)
                elif action == 3:  # Decrease tx power by 1
                    new_beacon = current_beacon
                    new_tx_power = max(5, current_power - 1)
                
                # Prepare response for this vehicle
                response_data[veh_id] = {
                    "transmissionPower": new_tx_power,
                    "beaconRate": new_beacon,
                    "MCS": 1  # Static MCS for simplicity, can be adjusted as needed
                }

            # Send the processed response back to the client (with updated transmissionPower, beaconRate, and MCS)
            conn.send(json.dumps(response_data).encode())
            logging.info(f"Sending RL response to client: {json.dumps(response_data)}")
        
        except Exception as e:
            logging.error(f"Error processing batch data: {e}")
            break

def start_server():
    """Start the server and accept client connections."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Server listening on {HOST}:{PORT}")
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    
    while True:
        conn, addr = server.accept()
        logging.info(f"Connected to client: {addr}")
        handle_client(conn)
        conn.close()

if __name__ == "__main__":
    start_server()
