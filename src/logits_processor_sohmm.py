import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, PreTrainedTokenizer
from typing import List, Dict, Tuple, Any

# We'll use logsumexp directly for the 3D/4D tensors to maintain clarity
# but we can still import stable_mvm if needed for consistency.
from src.hmm import stable_mvm

@torch.compile
def logit_adjustment_so(
    log_alpha_prev: torch.Tensor,          # (B, H, H) - log P(x_{<t}, z_{t-2}, z_{t-1})
    log_A: torch.Tensor,                   # (1, H, H, H) - log P(z_t | z_{t-2}, z_{t-1})
    log_B: torch.Tensor,                   # (H, V) - log P(x_t | z_t)
    expectation_zm: torch.Tensor,          # (H, H) - backward expectation E[exp|z_{t-1}, z_t]
    product_generated_toxicity: torch.Tensor,  # (B,) - cumulative toxicity product 
    exp_weights: torch.Tensor,             # (V,) - token weights exp(w(x))
    scores: torch.Tensor,                  # (B, V) - original language model logits
    a: float,                              # guidance strength
    epsilon: float = 1e-12                 # numerical stability
) -> torch.Tensor:
    """
    Second-order implementation of logit adjustment.
    """
    # Step 1: Compute forward transition probabilities
    # log_p_zm_x_less_m: log P(x_{<t}, z_{t-1}, z_t) = log sum_{z_{t-2}} P(z_t | z_{t-2}, z_{t-1}) P(x_{<t}, z_{t-2}, z_{t-1})
    # alpha_prev is (B, i, j), A is (1, i, j, k)
    log_p_zm_x_less_m = torch.logsumexp(log_alpha_prev.unsqueeze(3) + log_A, dim=1)  # (B, j, k)

    # Step 2: Compute HMM token probabilities log P(x_t | x_{<t})
    # First, marginalize over z_{t-1} to get log P(x_{<t}, z_t)
    log_p_zt_marginal = torch.logsumexp(log_p_zm_x_less_m, dim=1)  # (B, k)
    # log_p_x: log P(x_t, x_{<t}) = log sum_{z_t} P(x_t | z_t) P(x_{<t}, z_t)
    log_p_x = torch.vmap(stable_mvm, in_dims=(None, 0))(log_B.T, log_p_zt_marginal)  # (B, V)

    # Step 3: Compute expected future toxicity for each token
    # log_expectation_zm_x_less_m: log (E[future|z_{t-1}, z_t] * P(x_{<t}, z_{t-1}, z_t))
    log_expectation_zm_x_less_m = log_p_zm_x_less_m + torch.log(expectation_zm.unsqueeze(0) + epsilon) # (B, j, k)
    
    # log_num: log sum_{z_{t-1}, z_t} E[future|z_{t-1}, z_t] P(x_t | z_t) P(x_{<t}, z_{t-1}, z_t)
    # = log sum_k P(x_t | z_t=k) * sum_j (E[|j, k] * P(x_{<t}, j, k))
    log_expectation_weighted_zt = torch.logsumexp(log_expectation_zm_x_less_m, dim=1) # (B, k)
    log_num = torch.vmap(stable_mvm, in_dims=(None, 0))(log_B.T, log_expectation_weighted_zt) # (B, V)
    
    # log_expectation_xm: log E[future | x_{<t}, x_t]
    log_expectation_xm = log_num - log_p_x # (B, V)
    
    # Step 4: Apply current toxicity product and token weights
    log_expectation_xm += torch.log(product_generated_toxicity.unsqueeze(1) + epsilon) + torch.log(exp_weights + epsilon)
    
    # Step 5: Apply sigmoid-logit scaling
    expectation_xm = torch.exp(log_expectation_xm)
    expectation_xm = torch.clamp(expectation_xm, epsilon, 1 - epsilon)
    logit_expectation_xm = torch.log(expectation_xm / (1 - expectation_xm + epsilon) + epsilon)
    logit_adjusted = a * logit_expectation_xm
    expectation_xm_adjusted = torch.sigmoid(logit_adjusted)
    
    # Step 6: Re-weight language model probabilities and normalize
    p_lm = torch.softmax(scores, dim=-1)
    p_adjusted = p_lm * expectation_xm_adjusted  
    p_adjusted = p_adjusted / (p_adjusted.sum(dim=-1, keepdim=True) + epsilon)
    adjusted_logits = torch.log(p_adjusted + epsilon)
    
    return adjusted_logits

class SOHmmGuidedLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 hmm_model: Any,
                 expectation_cache: torch.Tensor,
                 a: float = 1.0,
                 tokenizer: PreTrainedTokenizer | None = None,
                 epsilon: float = 1e-12):
        
        required_attrs = ['alpha_exp', 'beta', 'compute_forward_probability']
        for attr in required_attrs:
            if not hasattr(hmm_model, attr):
                raise AttributeError(f"SOHMM model missing required attribute: {attr}")
        
        self.hmm_model = hmm_model
        self._model_device = hmm_model.alpha_exp.device
        
        self.expectation_cache = expectation_cache.to(self._model_device)
        self.a = a
        self.epsilon = epsilon
        self.tokenizer = tokenizer

        # Second-order A is (H, H, H)
        self.log_A = torch.log(self.hmm_model.alpha_exp.to(self._model_device) + self.epsilon).unsqueeze(0)
        self.log_B = self.hmm_model.beta.to(self._model_device)
        self.exp_weights = self.hmm_model.exp_weights.squeeze(-1).to(self._model_device)

        self.log_alpha_prev = None
        self.product_generated_toxicity = None
        self.prompt_len = -1
        self.is_configured_for_prompt = False
        self.generation_step = 0

    def configure_for_prompts(self, prompt_batch: torch.Tensor):
        self.prompt_len = prompt_batch.shape[1]
        self.is_configured_for_prompt = True
        self.generation_step = 0
        
        # log_alpha_prev is (B, H, H)
        self.log_alpha_prev = self.hmm_model.compute_forward_probability(prompt_batch)
        
        initial_product = 1.0
        batch_size = prompt_batch.shape[0]
        self.product_generated_toxicity = torch.full((batch_size,), initial_product, 
                                                   device=self._model_device, dtype=torch.float32)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if not self.is_configured_for_prompt:
                raise RuntimeError("SOHmmGuidedLogitsProcessor.configure_for_prompts() must be called before generation.")
            
            current_seq_len = input_ids.shape[1]
            expected_seq_len = self.prompt_len + self.generation_step
            
            if current_seq_len > expected_seq_len:
                new_token_pos = expected_seq_len
                new_tokens = input_ids[:, new_token_pos]
                
                # Update forward state: P(x_{<t}, z_{t-1}, z_t)
                # First transition
                log_p_zm_x_less_m = torch.logsumexp(self.log_alpha_prev.unsqueeze(3) + self.log_A, dim=1)
                
                # Update toxicity product
                self.product_generated_toxicity *= self.exp_weights[new_tokens]
                
                # Emission for new token x_t at z_t
                emission = self.hmm_model.beta[:, new_tokens].transpose(0, 1) # (B, H)
                self.log_alpha_prev = log_p_zm_x_less_m + emission.unsqueeze(1) # (B, H, H)
                
                self.generation_step += 1
            
            if self.generation_step >= len(self.expectation_cache):
                return scores
            
            expectation_zm = self.expectation_cache[self.generation_step] # (H, H)
            
            adjusted_logits = logit_adjustment_so(
                log_alpha_prev=self.log_alpha_prev,
                log_A=self.log_A,
                log_B=self.log_B,
                expectation_zm=expectation_zm,
                product_generated_toxicity=self.product_generated_toxicity,
                exp_weights=self.exp_weights,
                scores=scores,
                a=self.a,
                epsilon=self.epsilon
            )
            
            return adjusted_logits
