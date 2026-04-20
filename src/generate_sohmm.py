#!/usr/bin/env python
import os
import sys
import math
import json
import argparse
from typing import List, Dict, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

# Determine project root and add to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.append(PROJECT_ROOT)

# Local imports
from src import utils
from src.sohmm import SOHMM
from src.logits_processor_sohmm import SOHmmGuidedLogitsProcessor

def set_seed(seed: int, n_gpu: int):
    """Set random seed for reproducibility across PyTorch and CUDA."""
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="SOHMM-guided text generation.")
    parser.add_argument("--model_path", type=str, default="gpt2-large", help="HF model name/path for generation")
    parser.add_argument("--hmm_model_path", type=str, default="models/sohmm_gpt2-large_uncon_seq-len-32_4096_10M", help="Path to the trained SOHMM directory")
    parser.add_argument("--prompts_path", type=str, default="data/prompts.jsonl", help="Prompt file (JSONL)")
    parser.add_argument("--weights_path", type=str, default="data/coefficients.csv", help="Path to weights CSV file")
    parser.add_argument("--a", type=float, default=1.0, help="Strength of SOHMM guidance, often 0~2.0. 0 is no guidance, 1 is default.")
    parser.add_argument("--baseline", action="store_true", help="Generate baseline (no SOHMM guidance) alongside TRACE")
    parser.add_argument("--max_len", type=int, default=20, help="Max new tokens to generate")
    parser.add_argument("--num_generations", type=int, default=25, help="Generations per prompt")
    parser.add_argument("--generation_batch_size", type=int, default=5, help="Sequences per HF generate call")
    parser.add_argument("--prompt_batch_size", type=int, default=1, help="Prompts processed together")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    # Setup paths
    args.prompts_path = os.path.join(PROJECT_ROOT, args.prompts_path)
    args.weights_path = os.path.join(PROJECT_ROOT, args.weights_path)
    
    # Derive output CSV path relative to project root
    if args.baseline:
        output_path = os.path.join(
            PROJECT_ROOT,
            f"results/sohmm_comparison_a{args.a}_generated.csv"
        )
    else:
        output_path = os.path.join(
            PROJECT_ROOT,
            f"results/sohmm_detox_a{args.a}_generated.csv"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Setup device and seed
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(args.seed, torch.cuda.device_count())

    # Load generation model and tokenizer
    print(f"Loading generation model '{args.model_path}' …")
    gen_model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device).eval()
    gen_tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    gen_tokenizer.pad_token = gen_tokenizer.pad_token or gen_tokenizer.eos_token

    #
    # LOAD SOHMM MODEL AND CONFIGURE WEIGHTS
    #
    hmm_processor = None
    if not args.baseline or args.a > 0:
        print(f"Loading SOHMM from '{args.hmm_model_path}' …")
        hmm_model: SOHMM = utils.load_sohmm_model(args.hmm_model_path, device=device)

        # Load weights for HMM guidance
        if not os.path.exists(args.weights_path):
            sys.exit(f"Missing weights file: {args.weights_path}")
        
        weights_tensor = utils.load_weights(args.weights_path, device=device)
        hmm_model.set_weights(weights_tensor)

        #
        # PREPARE SOHMM LOGITS PROCESSOR
        #
        expectation_cache = hmm_model.compute_backward_expectation(T=args.max_len)
        hmm_processor = SOHmmGuidedLogitsProcessor(
            hmm_model=hmm_model,
            expectation_cache=expectation_cache,
            a=args.a,
            tokenizer=gen_tokenizer,
        )
    else:
        print("Running in baseline mode (no SOHMM guidance)")

    #
    # LOAD ALL PROMPTS
    #
    if not os.path.exists(args.prompts_path):
        raise FileNotFoundError(f"Prompt file '{args.prompts_path}' not found.")
    
    prompts: List[Tuple[int, str]] = []
    with open(args.prompts_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            prompts.append((idx, json.loads(line)["prompt"]["text"]))

    print(f"Total prompts to process: {len(prompts)}")

    #
    # BATCH GENERATION LOOP
    #
    file_exists = os.path.exists(output_path)

    for batch_start in tqdm(range(0, len(prompts), args.prompt_batch_size), desc="Generating"):
        batch_info = prompts[batch_start : batch_start + args.prompt_batch_size]
        if not batch_info:
            continue

        prompt_texts = [txt for _, txt in batch_info]

        max_model_len = getattr(gen_model.config, "max_position_embeddings", 512)
        max_prompt_len = max(max_model_len - args.max_len - 10, 10)

        inputs = gen_tokenizer.batch_encode_plus(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        )
        prompt_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        if hmm_processor:
            hmm_processor.configure_for_prompts(prompt_ids)

        batch_continuations: Dict[int, List[str]] = {idx: [] for idx, _ in batch_info}
        baseline_continuations: Dict[int, List[str]] = {idx: [] for idx, _ in batch_info} if args.baseline else None
        
        loops = math.ceil(args.num_generations / args.generation_batch_size)

        for loop_idx in range(loops):
            num_to_generate = min(
                args.generation_batch_size,
                args.num_generations - loop_idx * args.generation_batch_size,
            )
            if num_to_generate <= 0:
                break

            if hmm_processor and num_to_generate > 1:
                expanded_prompt_ids = prompt_ids.repeat_interleave(num_to_generate, dim=0)
                hmm_processor.configure_for_prompts(expanded_prompt_ids)
                input_ids_for_generation = prompt_ids
                attention_mask_for_generation = attention_mask
            else:
                input_ids_for_generation = prompt_ids
                attention_mask_for_generation = attention_mask

            logits_processors = LogitsProcessorList([hmm_processor]) if hmm_processor else LogitsProcessorList([])

            with torch.no_grad():
                gen_seqs = gen_model.generate(
                    input_ids=input_ids_for_generation,
                    attention_mask=attention_mask_for_generation,
                    logits_processor=logits_processors,
                    max_new_tokens=args.max_len,
                    num_return_sequences=num_to_generate,
                    do_sample=True,
                    top_p=0.9,
                    top_k=0,
                    temperature=1.0,
                    pad_token_id=gen_tokenizer.pad_token_id,
                    eos_token_id=gen_tokenizer.eos_token_id,
                )
            
            prompt_len = prompt_ids.shape[1]
            for b_idx in range(len(batch_info)):
                orig_idx = batch_info[b_idx][0]
                for k in range(num_to_generate):
                    seq_idx = b_idx * num_to_generate + k
                    cont_ids = gen_seqs[seq_idx][prompt_len:]
                    cont_text = gen_tokenizer.decode(
                        cont_ids, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=True
                    )
                    batch_continuations[orig_idx].append(cont_text)

        if args.baseline and hmm_processor:
            for loop_idx in range(loops):
                num_to_generate = min(
                    args.generation_batch_size,
                    args.num_generations - loop_idx * args.generation_batch_size,
                )
                if num_to_generate <= 0:
                    break

                with torch.no_grad():
                    baseline_gen_seqs = gen_model.generate(
                        input_ids=input_ids_for_generation,
                        attention_mask=attention_mask_for_generation,
                        max_new_tokens=args.max_len,
                        num_return_sequences=num_to_generate,
                        do_sample=True,
                        top_p=0.9,
                        top_k=0,
                        temperature=1.0,
                        pad_token_id=gen_tokenizer.pad_token_id,
                        eos_token_id=gen_tokenizer.eos_token_id,
                    )

                for b_idx in range(len(batch_info)):
                    orig_idx = batch_info[b_idx][0]
                    for k in range(num_to_generate):
                        seq_idx = b_idx * num_to_generate + k
                        cont_ids = baseline_gen_seqs[seq_idx][prompt_len:]
                        cont_text = gen_tokenizer.decode(
                            cont_ids, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=True
                        )
                        baseline_continuations[orig_idx].append(cont_text)

        batch_rows = []
        for orig_idx, prompt_text in batch_info:
            row: Dict = {"index": orig_idx, "prefix": prompt_text}
            
            if args.baseline and hmm_processor:
                for i, cont in enumerate(batch_continuations[orig_idx][: args.num_generations]):
                    row[f"trace_gen_{i + 1}"] = json.dumps({"continuation": cont})
                for i, cont in enumerate(baseline_continuations[orig_idx][: args.num_generations]):
                    row[f"baseline_gen_{i + 1}"] = json.dumps({"continuation": cont})
            else:
                for i, cont in enumerate(batch_continuations[orig_idx][: args.num_generations]):
                    mode_prefix = "baseline" if not hmm_processor else "trace"
                    row[f"{mode_prefix}_gen_{i + 1}"] = json.dumps({"continuation": cont})
            batch_rows.append(row)

        pd.DataFrame(batch_rows).to_csv(
            output_path,
            mode="a",
            header=not file_exists,
            index=False,
        )
        file_exists = True
        print(f"Saved {len(batch_rows)} rows → {output_path}")

    print("Generation complete ✔ – results in", output_path)

if __name__ == "__main__":
    main()
