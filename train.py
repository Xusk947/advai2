import torch
import numpy as np
import time
from typing import Dict

from src.envs.pacman_env import PacmanEnv
from src.agents.dqn import DQNAgent, preprocess_obs
from src.config import (
    EPS_START, EPS_END, EPS_DECAY, 
    TARGET_UPDATE_FREQ, TRAIN_START, 
    DEVICE
)

def train() -> None:
    env = PacmanEnv(level_path="assets/level1.png", render_mode=None)
    h, w = env.level.height, env.level.width
    n_actions = 4
    
    agent_names = ["pacman", "blinky", "pinky", "inky", "clyde"]
    agents = {name: DQNAgent(name, h, w, n_actions) for name in agent_names}
    
    epsilon = EPS_START
    total_steps = 0
    episode = 0
    
    print(f"Training on {DEVICE}...")
    
    while True:
        obs, info = env.reset()
        state = preprocess_obs(obs)
        episode_reward = 0
        done = False
        
        while not done:
            # Select actions for all agents
            actions = {}
            for name, agent in agents.items():
                actions[name] = agent.select_action(state, epsilon)
            
            # Step environment
            next_obs, rewards, done, truncated, info = env.step(actions)
            next_state = preprocess_obs(next_obs)
            
            # Store transitions in memory
            for name, agent in agents.items():
                reward = rewards.get(name, 0.0)
                agent.memory.push(state, actions[name], next_state, reward, done)
            
            state = next_state
            episode_reward += rewards.get("pacman", 0.0)
            total_steps += 1
            
            # Train agents
            if total_steps > TRAIN_START:
                for agent in agents.values():
                    agent.optimize_model()
                    
                if total_steps % TARGET_UPDATE_FREQ == 0:
                    for agent in agents.values():
                        agent.update_target_network()
            
            epsilon = max(EPS_END, epsilon * EPS_DECAY)
            
        episode += 1
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Steps {total_steps}, Pacman Reward: {episode_reward:.1f}, Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    train()
