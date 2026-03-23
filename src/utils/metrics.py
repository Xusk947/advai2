import json
import os
import numpy as np
from typing import Dict, List

def save_metrics(metrics_history: List[Dict], episode: int):
    results_path = "metrics/stats.json"
    os.makedirs("metrics", exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(metrics_history, f, indent=2)

def format_summary(metrics_history: List[Dict], window: int = 10) -> str:
    recent = metrics_history[-window:]
    if not recent: return "No data."
    
    # World metrics
    avg_score = np.mean([m["world"]["score"] for m in recent])
    avg_len = np.mean([m["world"]["length"] for m in recent])
    
    summary = f"📊 *Evaluation (Last {window} Eps)*\n"
    summary += f"🏆 Avg Score: {avg_score:.1f}\n"
    summary += f"⏱️ Avg Length: {avg_len:.1f}\n\n"
    
    # Agent metrics
    agents = recent[0]["agents"].keys()
    for name in agents:
        avg_rew = np.mean([m["agents"][name]["reward"] for m in recent])
        summary += f"• *{name.capitalize()}*: {avg_rew:+.1f} rew\n"
        
    return summary
