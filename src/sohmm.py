import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

def matmul(A, B):
    return torch.matmul(A, B)

def calc_alpha_flow_2nd_order(pf, pp, cp):
    """
    Calculates the alpha flow for a second-order HMM.

    Expected Input Shapes:
    pf: [batch_size, hidden_states, hidden_states] -> (b, i, j)
    pp: [batch_size, hidden_states, hidden_states] -> (b, i, j)
    cp: [batch_size, hidden_states, hidden_states] -> (b, j, k)
    """

    ll = cp.amax(dim=-1)

    pp_exp = torch.exp(pp - ll.unsqueeze(1))

    cp_exp = torch.exp(cp - ll.unsqueeze(-1))

    ratio = pf / pp_exp
    ratio[pp_exp == 0.0] = 0.0

    # Contract over the batch dimension to get the flow for each state pair (i, j)
    af = torch.einsum('bij, bjk -> ijk', ratio, cp_exp)

    return af

class SOHMM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_size: int, vocab_size: int, eos_token_id: int, sep_token_id: int = None):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id

        # alpha_exp: P(z_{t+1} | z_{t-1}, z_t) -> [H, H, H]
        alpha_exp = torch.softmax(torch.randn(hidden_size, hidden_size, hidden_size), dim=2)
        # beta: P(x_t | z_t) -> [H, V] (stored in log space)
        beta = torch.log_softmax(torch.randn(hidden_size, vocab_size), dim=1)
        # gamma: P(z_0, z_1) -> [H, H] (stored in log space)
        gamma = torch.log_softmax(torch.randn(hidden_size, hidden_size).flatten(), dim=0).view(hidden_size, hidden_size)

        if sep_token_id is not None:
            ################# SEP TOKEN INITIALIZATION #################
            beta[-1, sep_token_id] = 1e10
            beta[:-1, sep_token_id] = -1e10
            beta = torch.log_softmax(beta, dim=-1)
            # Second order initialization for SEP
            alpha_exp[:, -1, -1] = 1e-10
            ################# SEP TOKEN INITIALIZATION #################
        else:
            ################# EOS TOKEN INITIALIZATION #################
            beta[-1, eos_token_id] = 1e10
            beta[:-1, eos_token_id] = -1e10
            beta = torch.log_softmax(beta, dim=-1)
            # If current state is EOS, next state is almost certainly EOS
            alpha_exp[:, -1, :] = 1e-10
            alpha_exp[:, -1, -1] = 1.0
            ################# EOS TOKEN INITIALIZATION #################

        self.alpha_exp = nn.Parameter(alpha_exp, requires_grad=True)
        self.beta = nn.Parameter(beta, requires_grad=True)
        self.gamma = nn.Parameter(gamma, requires_grad=True)

        # ------------------- Weight buffers (Set #1) -------------------------
        self.register_buffer('weights_tensor', torch.zeros(vocab_size, dtype=torch.float32))  
        self.register_buffer('exp_weights', torch.ones(vocab_size, dtype=torch.float32))      
        self.register_buffer('weighted_beta', torch.zeros(hidden_size, vocab_size, dtype=torch.float32))

        # Initialize to default
        self.set_weights(weights_tensor=torch.zeros(vocab_size))

    @property
    def pi(self):
        """Initial state distribution P(z_0, z_1)"""
        return torch.softmax(self.gamma.flatten(), dim=0).view(self.gamma.shape)
    
    @property  
    def log_B(self):
        """Log emission probabilities (beta)"""
        return self.beta
    
    @property
    def w(self):
        """Exponentiated weights for toxicity"""
        return self.exp_weights

    def set_weights(self, weights_tensor: torch.Tensor):
        """
        Set the weights and update weighted_beta accordingly.
        """
        if weights_tensor.shape != (self.vocab_size,):
            raise ValueError(
                f"weights_tensor must have shape ({self.vocab_size},), but got {weights_tensor.shape}"
            )

        self.weights_tensor.copy_(weights_tensor)  
        self.exp_weights.copy_(torch.exp(weights_tensor))  

        P_x_given_s = torch.exp(self.beta)  # (H, V)
        weighted_beta = P_x_given_s * self.exp_weights.unsqueeze(0)  # (H, V)
        self.weighted_beta.copy_(weighted_beta)

    def update_params(self, alpha_exp, beta, gamma):
        self.alpha_exp.data = alpha_exp
        self.beta.data = beta
        self.gamma.data = gamma

    # bottom-up circuit pass (backward pass for LL)
    def forward(self, input_ids):
        device = self.alpha_exp.device
        alpha_exp, beta = self.alpha_exp, self.beta
        gamma_exp = torch.softmax(self.gamma.flatten(), dim=0).view(self.gamma.shape)
        hidden_states = self.hidden_size
        batch_size, seq_len = input_ids.shape

        input_ids_ = torch.permute(input_ids, (1, 0)).contiguous()
        input_probs = beta[
            torch.arange(0, hidden_states, device=device)[None, :, None],
            input_ids_[:, None, :]].contiguous() # seq_len * hidden_states * batch_size
        input_probs *= (input_ids_ != -1)[:, None, :].expand(-1, hidden_states, -1)

        ys = []
        y = torch.zeros((hidden_states, hidden_states, batch_size), device=device) # (z_{t-1}, z_t, Batch)

        for t in range(seq_len-1, -1, -1):
            if t != seq_len - 1:
                y_max = y.amax(dim=0, keepdim=True).amax(dim=1, keepdim=True)
                y = torch.exp(y - y_max)
                y = torch.einsum('ijk, jkb -> ijb', alpha_exp, y)
                y = torch.log(y + 1e-12) + y_max

            y += input_probs[t, :, :].unsqueeze(0) # broadcast over z_{t-1}
            ys.append(y)

        y_max = y.amax(dim=0).amax(dim=0)
        y = torch.exp(y - y_max.unsqueeze(0).unsqueeze(0))

        y = torch.einsum('ij, ijb -> b', gamma_exp, y)
        y = torch.log(y + 1e-12) + y_max

        ys.append(y)

        return ys

    # top-down circuit pass (calculating flows)
    def backward(self, input_ids, probs, alpha_flow, beta_flow, gamma_flow):
        device = self.alpha_exp.device
        alpha_exp, beta = self.alpha_exp, self.beta
        gamma_exp = torch.softmax(self.gamma.flatten(), dim=0).view(self.gamma.shape)
        hidden_states, vocab_size = self.hidden_size, self.vocab_size
        batch_size, seq_len = input_ids.shape

        input_ids_ = torch.permute(input_ids, (1, 0)).contiguous()
        input_probs = beta[
            torch.arange(0, hidden_states, device=device)[None, :, None],
            input_ids_[:, None, :]].contiguous()
        input_probs *= (input_ids_ != -1)[:, None, :].expand(-1, hidden_states, -1)

        flows = []

        probs_root = torch.permute(probs[-2], (2, 0, 1)).contiguous()
        pf = gamma_exp.unsqueeze(0) * torch.exp(probs_root - probs[-1][:, None, None])
        flows.append(pf)

        gamma_flow.add_(torch.sum(pf, dim=0))

        for t in range(0, seq_len-1):
            layer_idx = seq_len - t - 1

            pp = probs[layer_idx] - input_probs[t, :, :].unsqueeze(0)
            cp = probs[layer_idx-1]

            pp_b = torch.permute(pp, (2, 0, 1)).contiguous()
            cp_b = torch.permute(cp, (2, 0, 1)).contiguous()

            alpha_flow.add_(calc_alpha_flow_2nd_order(pf, pp_b, cp_b))

            pp_max = pp_b.amax(dim=1, keepdim=True).amax(dim=2, keepdim=True)
            pp_exp = torch.exp(pp_b - pp_max)

            ratio = pf / (pp_exp + 1e-12)
            ratio[pp_exp == 0.0] = 0.0

            pf_unscaled = torch.einsum('bij, ijk -> bjk', ratio, alpha_exp)
            pf = pf_unscaled * torch.exp(cp_b - pp_max)

            flows.append(pf)

        flows = torch.stack(flows, dim=0)
        flows_zt = torch.sum(flows, dim=2) # marginalize down to current state for emissions
        input_ids_[input_ids_ == -1] = vocab_size
        input_ids_ = input_ids_[:, :, None].expand(-1, -1, hidden_states).view(seq_len * batch_size, hidden_states)

        beta_flow.scatter_add_(0, input_ids_, flows_zt.view(seq_len * batch_size, hidden_states))

    def loglikelihood(self, input_ids, batch_size):
        device = self.alpha_exp.device
        data_size = input_ids.shape[0]

        ll = torch.tensor([0.0], device=device)
        for batch_idx in range(0, data_size, batch_size):
            batch_size_ = min(batch_size, data_size - batch_idx)
            input_ids_batch = input_ids[batch_idx: batch_idx + batch_size_].to(device)
            probs_ = self.forward(input_ids_batch)
            ll += torch.sum(probs_[-1])

        return ll

    def compute_forward_probability(self, input_ids):
        """
        Compute forward probability for second-order HMM.
        Returns log P(x_0...x_t, z_{t-1}, z_t) of shape (B, H, H).
        """
        device = self.alpha_exp.device
        batch_size, m = input_ids.size()
        
        if m < 2:
            # Fallback for short sequences: return P(x0, z0, z1)
            # This is log P(z0, z1) + log P(x0 | z0)
            x0 = input_ids[:, 0]
            emission0 = self.beta[:, x0].transpose(0, 1) # (B, H)
            log_gamma = torch.log_softmax(self.gamma.flatten(), dim=0).view(self.gamma.shape)
            return log_gamma.unsqueeze(0) + emission0.unsqueeze(2)

        x0 = input_ids[:, 0]
        x1 = input_ids[:, 1]
        
        emission0 = self.beta[:, x0].transpose(0, 1)
        emission1 = self.beta[:, x1].transpose(0, 1)
        
        log_gamma = torch.log_softmax(self.gamma.flatten(), dim=0).view(self.gamma.shape)
        
        # alpha_1(z0, z1) = log P(z0, z1) + log P(x0|z0) + log P(x1|z1)
        alpha_prev = log_gamma.unsqueeze(0) + emission0.unsqueeze(2) + emission1.unsqueeze(1)
        
        log_A = torch.log(self.alpha_exp + 1e-12)
        
        for t in range(2, m):
            # alpha_t(z_t, z_{t+1}) = log P(x_{t+1}|z_{t+1}) + log sum_{z_{t-1}} P(z_{t+1}|z_{t-1},z_t) alpha_{t-1}(z_{t-1},z_t)
            alpha_curr = torch.logsumexp(alpha_prev.unsqueeze(3) + log_A.unsqueeze(0), dim=1)
            xt = input_ids[:, t]
            emission_t = self.beta[:, xt].transpose(0, 1)
            alpha_prev = alpha_curr + emission_t.unsqueeze(1)
            
        return alpha_prev

    def compute_backward_expectation(self, T: int) -> torch.Tensor:
        """
        Compute backward expectation for second-order HMM.
        B[t, z_{t-1}, z_t] = E[exp(sum_{i=t+1}^T w(x_i)) | z_{t-1}, z_t]
        """
        device = self.alpha_exp.device
        hidden_states = self.hidden_size
        
        B = torch.ones((T, hidden_states, hidden_states), dtype=torch.float32, device=device)
        weighted_emission_sum = torch.sum(self.weighted_beta, dim=1) # (H,)
        
        for t in reversed(range(T - 1)):
            # B[t, i, j] = sum_k A[i, j, k] * W[k] * B[t+1, j, k]
            temp = weighted_emission_sum.unsqueeze(0) * B[t+1] # (j, k)
            B[t] = torch.sum(self.alpha_exp * temp.unsqueeze(0), dim=2)
            
        return B
