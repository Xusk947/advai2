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
    # Mandatory Telegram Check
    from src.utils.telegram import send_telegram_message
    if not send_telegram_message("🚀 Обучение на Kaggle началось!"):
        print("CRITICAL: BOT_TOKEN or CHAT_ID is missing or invalid. Telegram notification failed.")
        print("The script will now exit as requested.")
        return

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
            actions = {}
            for name, agent in agents.items():
                state_tensor = preprocess_obs(obs)
                actions[name] = agent.select_action(state_tensor)
            
            next_obs, rewards, env_done, _, info = env.step(actions)
            
            for name, agent in agents.items():
                episode_reward[name] += rewards[name]
                state_tensor = preprocess_obs(obs)
                next_state_tensor = preprocess_obs(next_obs)
                agent.memory.push(state_tensor, actions[name], torch.tensor([rewards[name]], device=DEVICE), next_state_tensor, torch.tensor([env_done], device=DEVICE))
                agent.optimize_model()
            
            obs = next_obs
            steps += 1
            if env_done or steps > 1000:
                done = True

        # End of episode
        for agent in agents.values():
            agent.update_target()
        
        # Collect Metrics
        ep_metrics = {
            "episode": episode,
            "world": {"score": info["score"], "length": steps},
            "agents": {name: {"reward": episode_reward[name]} for name in agents}
        }
        metrics_history.append(ep_metrics)

        # Every 10 episodes: Save & Notify
        if episode % 10 == 0:
            from src.utils.metrics import save_metrics, format_summary
            from src.utils.telegram import send_telegram_message
            save_metrics(metrics_history, episode)
            summary = format_summary(metrics_history, window=10)
            send_telegram_message(summary)

        # Recordings and Weights at milestones (100, 500, 1000)
        if episode in [100, 500, 1000]:
            print(f"Recording episode {episode}...")
            video_path = f"replay_ep{episode}.mp4"
            record_video(agents, env, video_path)
            save_mar_weights(agents, f"ep{episode}")
            
            # Send to Telegram
            from src.utils.telegram import send_telegram_document, send_telegram_video
            send_telegram_video(video_path, caption=f"Pac-Man Replay Episode {episode}")
            for name in agents:
                w_path = f"weights/{name}_ep{episode}.pth"
                send_telegram_document(w_path, caption=f"{name} weights Ep {episode}")
                if os.path.exists(w_path): os.remove(w_path)
            
            if os.path.exists(video_path): os.remove(video_path)

    save_mar_weights(agents, "final")
    env.close()

if __name__ == "__main__":
    train()
