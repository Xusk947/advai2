import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

from src.config import (
    DEVICE, LEARNING_RATE, GAMMA, BATCH_SIZE, 
    REPLAY_BUFFER_SIZE, TARGET_UPDATE_FREQ
)

class DQN(nn.Module):
    def __init__(self, h: int, w: int, outputs: int) -> None:
        super().__init__()
        # Input channel 8: Wall, Pill, PowerPill, Pacman, Ghost, Barrier, FrightenedGhost, DeadGhost
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        def conv2d_size_out(size: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> int:
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32
        
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.head(x.view(x.size(0), -1))

class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done) -> None:
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

class DQNAgent:
    def __init__(self, name: str, h: int, w: int, n_actions: int) -> None:
        self.name = name
        self.policy_net = DQN(h, w, n_actions).to(DEVICE)
        self.target_net = DQN(h, w, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.memory = ReplayMemory(REPLAY_BUFFER_SIZE)
        
        self.n_actions = n_actions
        self.steps_done = 0

    def select_action(self, state: torch.Tensor, epsilon: float) -> int:
        if random.random() > epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1).item()
        
        return random.randrange(self.n_actions)

    def optimize_model(self) -> Optional[float]:
        if len(self.memory) < BATCH_SIZE:
            return None

        transitions = self.memory.sample(BATCH_SIZE)
        
        batch_state = torch.cat([t[0] for t in transitions])
        batch_action = torch.tensor([t[1] for t in transitions], device=DEVICE).view(-1, 1)
        batch_next_state = torch.cat([t[2] for t in transitions])
        batch_reward = torch.tensor([t[3] for t in transitions], device=DEVICE).float()
        batch_done = torch.tensor([t[4] for t in transitions], device=DEVICE).float()

        state_action_values = self.policy_net(batch_state).gather(1, batch_action)
        
        with torch.no_grad():
            next_state_values = self.target_net(batch_next_state).max(1)[0]
        
        expected_state_action_values = (next_state_values * GAMMA * (1 - batch_done)) + batch_reward

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict())

def preprocess_obs(obs: np.ndarray) -> torch.Tensor:
    # 0: Empty, 1: Wall, 2: Pill, 3: Power Pill, 4: Pacman, 5: Ghost, 6: Frightened Ghost, 7: Dead Ghost
    h, w = obs.shape
    tensor = torch.zeros((1, 8, h, w), device=DEVICE)
    
    for val in range(1, 8):
        # Val 1-7 maps to channels 0-6. 
        tensor[0, val-1] = torch.from_numpy(obs == val).float().to(DEVICE)
    
    return tensor
