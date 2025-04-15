import numpy as np
import random
import logging
import csv

# Constants
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1  # Fixed exploration rate

# Simplified state discretization
POWER_BINS = [5, 15, 25, 30]
BEACON_BINS = [1, 5, 10, 20]
CBR_BINS = [0.0, 0.3, 0.6, 1.0]

# Initialize Q-table
q_table = np.zeros((len(POWER_BINS), len(BEACON_BINS), len(CBR_BINS), 4))  # 4 actions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Open CSV file for writing the training rewards (create if doesn't exist)
def write_to_csv(episode, cum_reward):
    with open('training_rewards.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, cum_reward])  # Write episode and cumulative reward to CSV

def discretize(value, bins):
    return np.digitize(value, bins) - 1

def calculate_reward(cbr):
    """
    Reward function based on how close CBR is to the target.
    - Positive reward when CBR is between 0.3 and 0.6.
    - Negative penalty when CBR is outside 0.3 to 0.6.
    """
    if 0.3 <= cbr <= 0.6:
        return 100  # Positive reward when CBR is in the desired range
    else:
        return -50  # Moderate penalty for CBR being out of range, avoid too large negative value

def select_action(state):
    """Choose an action based on the current state using epsilon-greedy strategy."""
    power_idx = discretize(state[0], POWER_BINS)
    beacon_idx = discretize(state[1], BEACON_BINS)
    cbr_idx = discretize(state[2], CBR_BINS)
    
    if random.random() < EPSILON:
        return random.choice([0, 1, 2, 3])  # 0: increase beacon, 1: decrease beacon, 2: increase tx power, 3: decrease tx power
    return np.argmax(q_table[power_idx, beacon_idx, cbr_idx])  # Choose the best action based on Q-table

def update_q_table(state, action, reward, new_state):
    """Update Q-table using the Q-learning formula."""
    old_idx = discretize(state[0], POWER_BINS), discretize(state[1], BEACON_BINS), discretize(state[2], CBR_BINS)
    new_idx = discretize(new_state[0], POWER_BINS), discretize(new_state[1], BEACON_BINS), discretize(new_state[2], CBR_BINS)
    
    old_q = q_table[old_idx + (action,)]
    max_new_q = np.max(q_table[new_idx])
    q_table[old_idx + (action,)] = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_new_q - old_q)

    # Log Q-table after update
    logger.info(f"Updated Q-table for state {state} with action {action}: {q_table[old_idx + (action,)]}")
    logger.info(f"New Q-table value: {q_table[old_idx + (action,)]}")
    logger.info(f"Q-table after update:\n{q_table}")  # Log the entire Q-table

def train():
    """Simulate the training process of the Q-learning agent."""
    cum_reward = 0  # Initialize cumulative reward for the episode
    max_cum_reward = 2000  # Set the maximum cumulative reward limit
    min_cum_reward = -2000  # Set the minimum cumulative reward limit
    
    for episode in range(10000):  # Number of training episodes
        # Random initial state
        state = (random.choice(POWER_BINS), random.choice(BEACON_BINS), random.choice(CBR_BINS))
        action = select_action(state)
        
        # Apply the action to the state (action 0 to 3)
        if action == 0:  # Increase beacon rate by 1
            new_beacon = min(20, state[1] + 1)
            new_tx_power = state[0]  # tx power remains the same
        elif action == 1:  # Decrease beacon rate by 1
            new_beacon = max(1, state[1] - 1)
            new_tx_power = state[0]  # tx power remains the same
        elif action == 2:  # Increase tx power by 1
            new_beacon = state[1]  # beacon rate remains the same
            new_tx_power = min(30, state[0] + 1)
        elif action == 3:  # Decrease tx power by 1
            new_beacon = state[1]  # beacon rate remains the same
            new_tx_power = max(5, state[0] - 1)
        
        # Calculate new CBR based only on the new beacon rate, not on the old CBR
        # Assuming CBR is proportional to beacon rate, for example:
        new_cbr = min(1.0, new_beacon * 0.03)  # Just an example formula for new CBR
        
        # Calculate reward based on new CBR
        reward = calculate_reward(new_cbr)
        
        # Update cumulative reward, and clamp it to be within the max and min limits
        cum_reward += reward
        cum_reward = max(min(cum_reward, max_cum_reward), min_cum_reward)  # Clamp to the range [-2000, 2000]
        
        # Update Q-table
        update_q_table(state, action, reward, (new_tx_power, new_beacon, new_cbr))
        
        # Write the reward and cumulative reward to the CSV file every episode
        write_to_csv(episode, cum_reward)
        
        if episode % 1000 == 0:
            print(f"Episode {episode}, Cumulative Reward: {cum_reward}, Q-table updated.")
    
    # Save the trained Q-table to a .npy file
    np.save('q_table_model.npy', q_table)
    print("Training complete. Q-table saved.")

if __name__ == "__main__":
    train()
