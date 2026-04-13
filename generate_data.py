import json
from datasets import load_dataset

SIZE = 20
ds = load_dataset("allenai/real-toxicity-prompts", split=f"train[:{SIZE}]")

out_path = f"data/RTP_hf_{SIZE}_prompts.jsonl"

with open(out_path, "w") as f:
    for i, row in enumerate(ds):
        prompt_text = row["prompt"]["text"]
        f.write(json.dumps({
            "index": i,
            "prompt": {
                "text": prompt_text
            }
        }) + "\n")

print(f"Saved {len(ds)} prompts to {out_path}")
