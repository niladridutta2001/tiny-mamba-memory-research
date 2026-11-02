"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    conv_mode: str = 'fixed'
    ssm_decay_mode: str = 'baseline'
    fast_min_rate: float = 1.0
    slow_max_rate: float = 0.004
    slow_gate_mode: str = 'none'
    slow_gate_bias_init: float = -4.0
    erase_gate_bias_init: float = -4.0
    write_gate_bias_init: float = -2.0
    event_embedding_dim: int = 16
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.conv_mode not in {'fixed', 'adaptive', 'static_mix'}:
            raise ValueError(f"unknown conv_mode: {self.conv_mode}")
        if self.conv_mode != 'fixed' and self.d_conv != 8:
            raise ValueError("adaptive convolution modes require d_conv=8")
        if self.ssm_decay_mode not in {'baseline', 'fast_slow_unconstrained', 'fast_slow'}:
            raise ValueError(f"unknown SSM decay mode: {self.ssm_decay_mode}")
        if self.ssm_decay_mode != 'baseline' and self.d_state != 8:
            raise ValueError("partitioned SSM decay modes require d_state=8")
        if self.fast_min_rate <= 0 or self.slow_max_rate <= 0:
            raise ValueError("fast_min_rate and slow_max_rate must be positive")
        if self.ssm_decay_mode == 'fast_slow' and self.slow_max_rate >= self.fast_min_rate:
            raise ValueError("slow_max_rate must be smaller than fast_min_rate")
        if self.slow_gate_mode not in {
            'none', 'adaptive', 'static', 'decoupled', 'decoupled_tied',
            'decoupled_no_event', 'decoupled_no_fast'
        }:
            raise ValueError(f"unknown slow gate mode: {self.slow_gate_mode}")
        if self.slow_gate_mode != 'none' and self.ssm_decay_mode != 'fast_slow':
            raise ValueError("identity-default slow gates require fast_slow decay modes")
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper


    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x)) + x

        return output
            

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        if args.conv_mode != 'fixed':
            self.conv_router = nn.Linear(args.d_inner, 4)
            self.conv_router_anchor = nn.Parameter(torch.zeros(args.d_inner))
            nn.init.zeros_(self.conv_router.weight)
            with torch.no_grad():
                # Begin near width 4 without saturating softmax, so both the
                # adaptive and static controls can move to other widths.
                self.conv_router.bias.copy_(torch.tensor([-2.0, -2.0, 2.0, -2.0]))
            # Tap order follows Conv1d weights: oldest to current. Candidate
            # widths retain the rightmost 1, 2, 4, or 8 taps.
            masks = torch.zeros(4, 8)
            for index, width in enumerate((1, 2, 4, 8)):
                masks[index, 8 - width :] = 1.0
            self.register_buffer("conv_width_masks", masks)
            self.capture_conv_alpha = False
            self.captured_conv_alpha = None

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        if args.slow_gate_mode in {'adaptive', 'static'}:
            # Compact channel-wise router: one adaptive scalar per inner
            # channel, broadcast over the four slow SSM modes. The static
            # control uses the same parameters with an input-independent one.
            self.slow_gate_weight = nn.Parameter(torch.zeros(args.d_inner))
            self.slow_gate_bias = nn.Parameter(torch.full((args.d_inner,), args.slow_gate_bias_init))
            self.capture_slow_gate = False
            self.captured_slow_gate = None
        if args.slow_gate_mode.startswith('decoupled'):
            # Event-conditioned H8 router.  The tied control retains exactly the
            # same parameters but averages the two logits during computation.
            self.event_embedding = nn.Embedding(args.vocab_size, args.event_embedding_dim)
            self.fast_summary_weight = nn.Parameter(torch.full((args.d_inner, 4), 0.25))
            self.erase_u_weight = nn.Parameter(torch.zeros(args.d_inner))
            self.erase_fast_weight = nn.Parameter(torch.zeros(args.d_inner))
            self.erase_event = nn.Linear(args.event_embedding_dim, args.d_inner, bias=False)
            self.erase_bias = nn.Parameter(torch.full((args.d_inner,), args.erase_gate_bias_init))
            self.write_u_weight = nn.Parameter(torch.zeros(args.d_inner))
            self.write_fast_weight = nn.Parameter(torch.zeros(args.d_inner))
            self.write_event = nn.Linear(args.event_embedding_dim, args.d_inner, bias=False)
            self.write_bias = nn.Parameter(torch.full((args.d_inner,), args.write_gate_bias_init))
            nn.init.normal_(self.erase_event.weight, std=0.02)
            nn.init.normal_(self.write_event.weight, std=0.02)
            self.event_ids_by_position = None
            self.capture_decoupled_gates = False
            self.captured_erase_gate = None
            self.captured_write_gate = None

        baseline_rates = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner).float()
        if args.ssm_decay_mode == 'baseline':
            self.A_log = nn.Parameter(torch.log(baseline_rates))
        elif args.ssm_decay_mode == 'fast_slow_unconstrained':
            # Initialization-only control: exactly the same initial A as constrained M1,
            # but both groups subsequently follow unconstrained softplus rates.
            fast_rates = torch.tensor((1.1, 2.0, 3.0, 4.0)).view(1, 4).expand(args.d_inner, -1)
            slow_rates = args.slow_max_rate * torch.tensor((0.2, 0.4, 0.6, 0.8)).view(1, 4)
            slow_rates = slow_rates.expand(args.d_inner, -1)
            self.A_fast_theta = nn.Parameter(torch.log(torch.expm1(fast_rates)).clone())
            self.A_slow_theta = nn.Parameter(torch.log(torch.expm1(slow_rates)).clone())
        else:
            fast_initial = torch.arange(1, 5, dtype=torch.float32).view(1, 4).expand(args.d_inner, -1)
            fast_excess = (fast_initial - args.fast_min_rate).clamp_min(0.1)
            fast_theta = torch.log(torch.expm1(fast_excess))
            slow_fraction = torch.tensor((0.2, 0.4, 0.6, 0.8)).view(1, 4).expand(args.d_inner, -1)
            slow_theta = torch.logit(slow_fraction)
            self.A_fast_theta = nn.Parameter(fast_theta.clone())
            self.A_slow_theta = nn.Parameter(slow_theta.clone())
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        # Optional inference-time intervention. The default None preserves the
        # upstream implementation exactly. A (batch, length) tensor is set by
        # the task wrapper only for controlled delta-scaling experiments.
        self.delta_scale_by_position = None
        # Optional multiplicative gate on the discretized input write. Shape:
        # (batch, length, d_inner). The task wrapper supplies this only for the
        # key-routing model variants; None preserves upstream behavior.
        self.write_route_by_position = None
        # Optional diagnostic gate on the decay exponent only. A zero at a
        # position makes deltaA exactly one there while leaving deltaB_u
        # unchanged, separating shared attenuation from incoming writes.
        self.decay_scale_by_position = None
        # Optional diagnostic reset points for the local convolutional history.
        # Recurrent SSM state is not reset. Shape: (batch, length), boolean.
        self.conv_reset_by_position = None
        # Optional diagnostic mask applied only to C_t h_t. Recurrent updates
        # remain untouched, so this isolates each mode group's readout use.
        self.readout_mode_mask = None

    def decay_matrix(self) -> torch.Tensor:
        """Return the negative continuous-time decay matrix A (64, 8)."""
        if self.args.ssm_decay_mode == 'baseline':
            return -torch.exp(self.A_log.float())
        if self.args.ssm_decay_mode == 'fast_slow_unconstrained':
            fast = -F.softplus(self.A_fast_theta.float())
            slow = -F.softplus(self.A_slow_theta.float())
        else:
            fast = -(self.args.fast_min_rate + F.softplus(self.A_fast_theta.float()))
            slow = -self.args.slow_max_rate * torch.sigmoid(self.A_slow_theta.float())
        return torch.cat((fast, slow), dim=-1)

    def slow_update_gate(self, u: torch.Tensor) -> torch.Tensor | None:
        """Return (batch, length, channels) slow-update interpolation gates."""
        if self.args.slow_gate_mode == 'none' or self.args.slow_gate_mode.startswith('decoupled'):
            return None
        source = u if self.args.slow_gate_mode == 'adaptive' else torch.ones_like(u)
        return torch.sigmoid(source * self.slow_gate_weight + self.slow_gate_bias)

    def decoupled_slow_gates(
        self, u: torch.Tensor, fast_state: torch.Tensor, event_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute H8 erase/write gates after the current fast-state update."""
        fast_summary = (fast_state * self.fast_summary_weight.unsqueeze(0)).sum(dim=-1)
        event_feature = self.event_embedding(event_ids)
        # Fuse the two projections and channel-wise branches. This is
        # mathematically identical to evaluating erase/write separately, but
        # avoids several tiny CUDA launches inside every scan timestep.
        event_weight = torch.cat((self.erase_event.weight, self.write_event.weight), dim=0)
        event_terms = F.linear(event_feature, event_weight).view(-1, 2, self.args.d_inner)
        if self.args.slow_gate_mode == 'decoupled_no_event':
            event_terms = torch.zeros_like(event_terms)
        if self.args.slow_gate_mode == 'decoupled_no_fast':
            fast_summary = torch.zeros_like(fast_summary)
        u_weights = torch.stack((self.erase_u_weight, self.write_u_weight), dim=0)
        fast_weights = torch.stack((self.erase_fast_weight, self.write_fast_weight), dim=0)
        biases = torch.stack((self.erase_bias, self.write_bias), dim=0)
        logits = (
            u.unsqueeze(1) * u_weights.unsqueeze(0)
            + fast_summary.unsqueeze(1) * fast_weights.unsqueeze(0)
            + event_terms + biases.unsqueeze(0)
        )
        erase_logits, write_logits = logits.unbind(dim=1)
        if self.args.slow_gate_mode == 'decoupled_tied':
            tied = 0.5 * (erase_logits + write_logits)
            gate = torch.sigmoid(tied)
            return gate, gate
        return torch.sigmoid(erase_logits), torch.sigmoid(write_logits)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        conv_input = x
        if self.args.conv_mode == 'fixed':
            x = self.conv1d(conv_input)[:, :, :l]
        else:
            current = rearrange(conv_input, 'b d l -> b l d')
            if self.args.conv_mode == 'adaptive':
                router_input = current + self.conv_router_anchor
            else:
                router_input = self.conv_router_anchor.view(1, 1, -1).expand(b, l, -1)
            alpha = torch.softmax(self.conv_router(router_input), dim=-1)
            if self.capture_conv_alpha:
                self.captured_conv_alpha = alpha.detach()
            effective_mask = einsum(alpha, self.conv_width_masks, 'b l k, k j -> b l j')
            windows = F.pad(conv_input, (self.args.d_conv - 1, 0)).unfold(2, self.args.d_conv, 1)
            kernel = self.conv1d.weight[:, 0, :].view(1, self.args.d_inner, 1, self.args.d_conv)
            x = (windows * kernel * effective_mask.unsqueeze(1)).sum(dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias.view(1, -1, 1)
        if self.conv_reset_by_position is not None:
            if self.args.conv_mode != 'fixed':
                raise ValueError("convolution-history reset is defined only for fixed convolution")
            reset = self.conv_reset_by_position.to(device=x.device, dtype=torch.bool)
            # Recompute each suffix as a fresh convolutional segment. Beyond
            # d_conv-1 positions this equals the normal convolution exactly.
            for batch_index in range(b):
                reset_positions = torch.nonzero(reset[batch_index], as_tuple=False).flatten().tolist()
                for segment_index, start in enumerate(reset_positions):
                    stop = reset_positions[segment_index + 1] if segment_index + 1 < len(reset_positions) else l
                    segment = conv_input[batch_index : batch_index + 1, :, start:stop]
                    x[batch_index : batch_index + 1, :, start:stop] = self.conv1d(segment)[:, :, : stop - start]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        A = self.decay_matrix()
        (d_in, n) = A.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        if self.delta_scale_by_position is not None:
            scale = self.delta_scale_by_position.to(device=delta.device, dtype=delta.dtype)
            if scale.ndim == 2:
                scale = scale.unsqueeze(-1)
            delta = delta * scale
        
        slow_gate = self.slow_update_gate(x)
        if slow_gate is not None and self.capture_slow_gate:
            self.captured_slow_gate = slow_gate.detach()
        y = self.selective_scan(x, delta, A, B, C, D, slow_gate)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D, slow_gate=None):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        decay_delta = delta
        if self.decay_scale_by_position is not None:
            decay_scale = self.decay_scale_by_position.to(device=delta.device, dtype=delta.dtype)
            if decay_scale.ndim == 2:
                decay_scale = decay_scale.unsqueeze(-1)
            decay_delta = delta * decay_scale
        deltaA = torch.exp(einsum(decay_delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        if self.write_route_by_position is not None:
            route = self.write_route_by_position.to(device=deltaB_u.device, dtype=deltaB_u.dtype)
            deltaB_u = deltaB_u * route.unsqueeze(-1)
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        erase_gates = []
        write_gates = []
        decoupled = self.args.slow_gate_mode.startswith('decoupled')
        if decoupled and self.event_ids_by_position is None:
            raise RuntimeError("decoupled slow gates require event_ids_by_position")
        for i in range(l):
            candidate = deltaA[:, i] * x + deltaB_u[:, i]
            if decoupled:
                fast = candidate[:, :, :4]
                erase_gate, write_gate = self.decoupled_slow_gates(
                    u[:, i], fast, self.event_ids_by_position[:, i]
                )
                slow = (
                    (1.0 - erase_gate.unsqueeze(-1)) * x[:, :, 4:]
                    + write_gate.unsqueeze(-1) * deltaB_u[:, i, :, 4:]
                )
                x = torch.cat((fast, slow), dim=-1)
                erase_gates.append(erase_gate)
                write_gates.append(write_gate)
            elif slow_gate is None:
                x = candidate
            else:
                gate = slow_gate[:, i].unsqueeze(-1)
                fast = candidate[:, :, :4]
                slow = (1.0 - gate) * x[:, :, 4:] + gate * candidate[:, :, 4:]
                x = torch.cat((fast, slow), dim=-1)
            readout_state = x
            if self.readout_mode_mask is not None:
                mask = self.readout_mode_mask.to(device=x.device, dtype=x.dtype).view(1, 1, n)
                readout_state = x * mask
            y = einsum(readout_state, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        if decoupled and self.capture_decoupled_gates:
            self.captured_erase_gate = torch.stack(erase_gates, dim=1).detach()
            self.captured_write_gate = torch.stack(write_gates, dim=1).detach()
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
        
