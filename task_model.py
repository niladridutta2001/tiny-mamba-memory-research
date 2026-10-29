"""Five-class state tracker and hypothesis-driven routing variants."""

import math

import torch
import torch.nn as nn

from model import Mamba, ModelArgs


class StateTracker(nn.Module):
    VARIANTS = (
        "baseline",
        "key_routing",
        "tied_routing",
        "event_decay",
        "coupled_event_gate",
        "adaptive_kernel",
        "static_kernel_mix",
        "adaptive_kernel_only",
        "static_kernel_mix_only",
    )

    def __init__(self, variant: str = "baseline", route_init: float = 0.95, d_conv: int = 4) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown model variant {variant!r}; expected one of {self.VARIANTS}")
        if not 0.0 < route_init < 1.0:
            raise ValueError("route_init must lie strictly between zero and one")
        if d_conv < 1:
            raise ValueError("d_conv must be at least one")
        self.variant = variant
        self.d_conv = d_conv
        conv_mode = {
            "adaptive_kernel": "adaptive",
            "static_kernel_mix": "static_mix",
            "adaptive_kernel_only": "adaptive",
            "static_kernel_mix_only": "static_mix",
        }.get(variant, "fixed")
        if conv_mode != "fixed" and d_conv != 8:
            raise ValueError(f"{variant} requires d_conv=8")
        self.backbone = Mamba(
            ModelArgs(
                d_model=32,
                n_layer=1,
                d_state=8,
                expand=2,
                d_conv=d_conv,
                conv_mode=conv_mode,
                vocab_size=41,
            )
        )
        self.head = nn.Linear(32, 5)
        if variant in {"key_routing", "tied_routing"}:
            initial_logit = math.log(route_init / (1.0 - route_init))
            self.route_logits = nn.Parameter(torch.full((8, 64), initial_logit))
        elif variant in {"event_decay", "coupled_event_gate", "adaptive_kernel", "static_kernel_mix"}:
            # exp(tanh(0)) = 1, so both variants are exactly baseline-equivalent
            # before training and share the same 3x64 gate parameterization.
            self.event_gate = nn.Parameter(torch.zeros(3, 64))

    def routing_table(self) -> torch.Tensor | None:
        if self.variant not in {"key_routing", "tied_routing"}:
            return None
        if self.variant == "key_routing":
            logits = self.route_logits
        else:
            # Parameter-matched control: retain all 8x64 parameters, but tie
            # rows during computation so the write cannot depend on key ID.
            logits = self.route_logits.mean(dim=0, keepdim=True).expand_as(self.route_logits)
        return torch.sigmoid(logits)

    def write_routes(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        table = self.routing_table()
        if table is None:
            return None
        is_set = (input_ids >= 1) & (input_ids <= 32)
        key_indices = torch.div((input_ids - 1).clamp_min(0), 4, rounding_mode="floor").clamp_max(7)
        selected = table[key_indices]
        return torch.where(is_set.unsqueeze(-1), selected, torch.ones_like(selected))

    def event_retention_scale(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if self.variant not in {"event_decay", "coupled_event_gate", "adaptive_kernel", "static_kernel_mix"}:
            return None
        event_types = torch.zeros_like(input_ids)
        event_types = torch.where((input_ids >= 1) & (input_ids <= 32), 1, event_types)
        event_types = torch.where((input_ids >= 33) & (input_ids <= 40), 2, event_types)
        return torch.exp(torch.tanh(self.event_gate[event_types]))

    def hidden_states(
        self,
        input_ids: torch.Tensor,
        delta_scale: torch.Tensor | None = None,
        write_route_override: torch.Tensor | None = None,
        decay_scale: torch.Tensor | None = None,
        conv_reset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mixers = [layer.mixer for layer in self.backbone.layers]
        write_routes = write_route_override
        if write_routes is None:
            write_routes = self.write_routes(input_ids)
        event_scale = self.event_retention_scale(input_ids)
        if event_scale is not None:
            if self.variant in {"event_decay", "adaptive_kernel", "static_kernel_mix"} and decay_scale is None:
                decay_scale = event_scale
            elif self.variant == "coupled_event_gate" and delta_scale is None:
                delta_scale = event_scale
        for mixer in mixers:
            mixer.delta_scale_by_position = delta_scale
            mixer.write_route_by_position = write_routes
            mixer.decay_scale_by_position = decay_scale
            mixer.conv_reset_by_position = conv_reset
        try:
            hidden = self.backbone.embedding(input_ids)
            for layer in self.backbone.layers:
                hidden = layer(hidden)
            return self.backbone.norm_f(hidden)
        finally:
            for mixer in mixers:
                mixer.delta_scale_by_position = None
                mixer.write_route_by_position = None
                mixer.decay_scale_by_position = None
                mixer.conv_reset_by_position = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden_states(input_ids))

    def scaled_logits(self, input_ids: torch.Tensor, alpha: float, mode: str) -> torch.Tensor:
        if mode == "global":
            scale = torch.full_like(input_ids, alpha, dtype=torch.float32)
        elif mode == "noop_only":
            scale = torch.where(input_ids == 0, alpha, 1.0).float()
        elif mode == "set_only":
            is_set = (input_ids >= 1) & (input_ids <= 32)
            scale = torch.where(is_set, alpha, 1.0).float()
        else:
            raise ValueError(f"unknown delta scaling mode: {mode}")
        return self.head(self.hidden_states(input_ids, scale))
