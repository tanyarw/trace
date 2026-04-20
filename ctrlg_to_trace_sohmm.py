import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from src.sohmm import SOHMM as TraceSOHMM

# Update these paths based on your actual model names and checkpoint numbers
NAME = "sohmm_gpt2-large_4096"
src_dir = Path(f"./models/{NAME}/checkpoint-200")
dst_dir = Path(f"./models/{NAME}_trace/checkpoint-200")
dst_dir.mkdir(parents=True, exist_ok=True)

# Load Ctrl-G config
config_path = src_dir / "config.json"
if not config_path.exists():
    raise FileNotFoundError(f"No config.json found in {src_dir}")

with open(config_path, "r") as f:
    cfg = json.load(f)

# Extract parameters from config
hidden_size = cfg.get("hidden_states", cfg.get("hidden_size"))
vocab_size = cfg["vocab_size"]
eos_token_id = cfg["eos_token_id"]

# Load Ctrl-G weights
state = None
if (src_dir / "model.safetensors").exists():
    state = load_file(str(src_dir / "model.safetensors"))
elif (src_dir / "pytorch_model.bin").exists():
    state = torch.load(src_dir / "pytorch_model.bin", map_location="cpu")
else:
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {src_dir}")

# Build TRACE SOHMM
# Note: SOHMM uses (hidden_size, vocab_size, eos_token_id) in __init__
trace_sohmm = TraceSOHMM(hidden_size=hidden_size, vocab_size=vocab_size, eos_token_id=eos_token_id)

# Map compatible parameters
with torch.no_grad():
    # SOHMM alpha_exp: P(z_{t+1} | z_{t-1}, z_t) -> [H, H, H]
    # SOHMM beta: P(x_t | z_t) -> [H, V] (log space)
    # SOHMM gamma: P(z_0, z_1) -> [H, H] (log space)

    trace_sohmm.alpha_exp.copy_(state["alpha_exp"])
    trace_sohmm.beta.copy_(state["beta"])

    # In SOHMM, gamma is already stored in log-space [H, H]
    # If Ctrl-G stores it in log-space, we can copy directly
    trace_sohmm.gamma.copy_(state["gamma"])

# Save TRACE-compatible checkpoint
trace_sohmm.save_pretrained(dst_dir)

print(f"Converted SOHMM checkpoint saved to: {dst_dir}")
