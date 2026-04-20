import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from src.hmm import HMM as TraceHMM

NAME = "hmm_gpt2-large_4096"
src_dir = Path(f"./models/{NAME}/checkpoint-200")
dst_dir = Path(f"./models/{NAME}_trace/checkpoint-200")
dst_dir.mkdir(parents=True, exist_ok=True)

# Load Ctrl-G config
with open(src_dir / "config.json", "r") as f:
    cfg = json.load(f)

hidden_size = cfg["hidden_states"]   # Ctrl-G name
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

# Build TRACE HMM
trace_hmm = TraceHMM(hidden_size=hidden_size, vocab_size=vocab_size, eos_token_id=eos_token_id)

# Map compatible parameters
with torch.no_grad():
    trace_hmm.alpha_exp.copy_(state["alpha_exp"])
    trace_hmm.beta.copy_(state["beta"])

    # Ctrl-G stores gamma in log-space; TRACE stores gamma_exp in prob-space
    gamma_log = state["gamma"]
    gamma_exp = torch.exp(gamma_log).unsqueeze(0)  # shape (1, H)
    trace_hmm.gamma_exp.copy_(gamma_exp)

# Save TRACE-compatible checkpoint
trace_hmm.save_pretrained(dst_dir)

print(f"Converted checkpoint saved to: {dst_dir}")
