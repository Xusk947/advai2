# Kaggle Setup Guide for Pac-Man MARL 🤖🦇

Follow these steps to start training on Kaggle with GPU and Telegram notifications:

### 1. Create a New Notebook
- Set **Accelerator** to **GPU P100** or **GPU T4 x2**.
- Ensure **Internet on** is enabled in the sidebar.

### 2. Set Up Telegram Secrets (IMPORTANT)
- Go to **Add-ons -> Secrets**.
- Add two secrets:
  - `BOT_TOKEN`: Your Telegram bot token.
  - `CHAT_ID`: Your Telegram chat ID.

### 3. Preparation Cell
Copy and run this in the first cell to clone the repo and install dependencies:
```python
import os
from kaggle_secrets import UserSecretsClient

# 1. Setup Environment
user_secrets = UserSecretsClient()
os.environ['BOT_TOKEN'] = user_secrets.get_secret("BOT_TOKEN")
os.environ['CHAT_ID'] = user_secrets.get_secret("CHAT_ID")

# 2. Clone Repository
!rm -rf advai2
!git clone https://github.com/Xusk947/advai2.git
%cd advai2

# 3. Install Dependencies
!pip install gymnasium pygame opencv-python torch numpy requests
```

### 4. Training Cell
Run this to start the training:
```python
!python train.py
```

### 5. What to Expect
- **Immediately**: You should get a message "🚀 Обучение на Kaggle началось!" in Telegram.
- **Milestones**: At episodes 100, 500, 1000, 2000... you will receive:
  - 📊 Text Summary (Avg score, rewards for all 5 agents).
  - 📎 `stats.json` with full history.
  - 🎥 Replay Video of the simulation.
  - 📦 Model Weights (`.pth`) for each agent.
