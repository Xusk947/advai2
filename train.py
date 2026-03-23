import torch
import numpy as np
import os
from typing import Dict, List
import cv2

from src.envs.pacman_env import PacmanEnv
from src.agents.dqn import DQNAgent, preprocess_obs
from src.config import (
    EPS_START, EPS_END, EPS_DECAY, 
    TARGET_UPDATE_FREQ, TRAIN_START, 
    DEVICE
)

def save_mar_weights(agents: Dict[str, DQNAgent], episode: int) -> None:
    os.makedirs("weights", exist_ok=True)
    for name, agent in agents.items():
        agent.save(f"weights/{name}_ep{episode}.pth")

def record_video(agents: Dict[str, DQNAgent], env: PacmanEnv, filename: str) -> None:
    obs, info = env.reset()
    state = preprocess_obs(obs)
    done = False
    
    # Capture frames using environment rendering (rgb_array)
    frames = []
    
    while not done and len(frames) < 1000:
        actions = {name: agent.select_action(state, epsilon=0.0) for name, agent in agents.items()}
        obs, rewards, done, truncated, info = env.step(actions)
        state = preprocess_obs(obs)
        
        # Render frame
        frame = env.render_array() # We'll need to add this method to PacmanEnv
        frames.append(frame)
        
    if frames:
        h, w, c = frames[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()

def train() -> None:
    # Use render_mode="rgb_array" for video capture
    env = PacmanEnv(level_path="assets/level1.png", render_mode="rgb_array")
    h, w = env.level.height, env.level.width
    n_actions = 4
    
    agent_names = ["pacman", "blinky", "pinky", "inky", "clyde"]
    agents = {name: DQNAgent(name, h, w, n_actions) for name in agent_names}
    
    epsilon = EPS_START
    total_steps = 0
    episode = 0
    
    print(f"Training on {DEVICE}...")
    
    while episode < 1000:
        obs, info = env.reset()
        state = preprocess_obs(obs)
        episode_reward = 0
        done = False
        
        while not done:
            actions = {name: agent.select_action(state, epsilon) for name, agent in agents.items()}
            next_obs, rewards, done, truncated, info = env.step(actions)
            next_state = preprocess_obs(next_obs)
            
            for name, agent in agents.items():
                agent.memory.push(state, actions[name], next_state, rewards.get(name, 0.0), done)
            
            state = next_state
            episode_reward += rewards.get("pacman", 0.0)
            total_steps += 1
            
            if total_steps > TRAIN_START:
                for agent in agents.values():
                    agent.optimize_model()
                    
                if total_steps % TARGET_UPDATE_FREQ == 0:
                    for agent in agents.values():
                        agent.update_target_network()
            
            epsilon = max(EPS_END, epsilon * EPS_DECAY)
            
        episode += 1
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Steps {total_steps}, Pacman Reward: {episode_reward:.1f}")
            save_mar_weights(agents, episode)
            record_video(agents, env, f"replay_ep{episode}.mp4")

    save_mar_weights(agents, "final")
    env.close()

if __name__ == "__main__":
    train()
