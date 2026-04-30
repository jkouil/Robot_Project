from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn


class TeacherDictFeaturesExtractor(nn.Module):
    """Teacher encoder with scandots -> y_t and x_t concatenation before the recurrent core."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        proprio_hidden_dim: int = 128,
        scandot_hidden_dim: int = 128,
        command_hidden_dim: int = 32,
        privileged_hidden_dim: int = 32,
        fused_dim: int = 256,
        encoder_variant: str = "current",
        include_privileged: bool = True,
    ):
        encoder_variant = str(encoder_variant).lower()
        include_privileged = bool(include_privileged)
        proprio_dim = int(observation_space.spaces["proprio"].shape[0])
        scandot_space = observation_space.spaces.get("scandots")
        scandot_dim = 0
        if scandot_space is not None:
            scandot_dim = scandot_hidden_dim

        command_dim = 0
        command_space = observation_space.spaces.get("command")
        if command_space is not None:
            command_input_dim = int(command_space.shape[0])
            command_dim = command_input_dim if encoder_variant == "paper" else command_hidden_dim

        privileged_dim = 0
        privileged_space = observation_space.spaces.get("privileged")
        if privileged_space is not None and include_privileged:
            privileged_dim = privileged_hidden_dim

        recurrent_input_dim = proprio_dim + scandot_dim + command_dim + privileged_dim
        features_dim = fused_dim if encoder_variant == "current" else recurrent_input_dim
        super().__init__()
        self.observation_space = observation_space
        self._features_dim = features_dim

        self.encoder_variant = encoder_variant
        self.include_privileged = include_privileged
        self.proprio_identity = nn.Identity()
        self.scandot_encoder = None
        if scandot_space is not None:
            scandot_input_dim = int(scandot_space.shape[0])
            self.scandot_encoder = nn.Sequential(
                nn.Linear(scandot_input_dim, scandot_hidden_dim),
                nn.LayerNorm(scandot_hidden_dim),
                nn.SiLU(),
                nn.Linear(scandot_hidden_dim, scandot_hidden_dim),
                nn.SiLU(),
            )

        self.command_encoder = None
        if command_space is not None and self.encoder_variant != "paper":
            command_input_dim = int(command_space.shape[0])
            self.command_encoder = nn.Sequential(
                nn.Linear(command_input_dim, command_hidden_dim),
                nn.LayerNorm(command_hidden_dim),
                nn.SiLU(),
            )

        self.privileged_encoder = None
        if privileged_space is not None and self.include_privileged:
            privileged_input_dim = int(privileged_space.shape[0])
            self.privileged_encoder = nn.Sequential(
                nn.Linear(privileged_input_dim, privileged_hidden_dim),
                nn.LayerNorm(privileged_hidden_dim),
                nn.SiLU(),
            )

        self.fuser = None
        if self.encoder_variant == "current":
            self.fuser = nn.Sequential(
                nn.Linear(recurrent_input_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.SiLU(),
            )
        elif self.encoder_variant != "paper":
            raise ValueError(f"Unsupported teacher encoder variant: {encoder_variant}")

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        xt = self.proprio_identity(observations["proprio"])
        encoded = [xt]
        if self.scandot_encoder is not None and "scandots" in observations:
            encoded.append(self.scandot_encoder(observations["scandots"]))
        if "command" in observations:
            if self.command_encoder is None:
                encoded.append(observations["command"])
            else:
                encoded.append(self.command_encoder(observations["command"]))
        if self.privileged_encoder is not None and "privileged" in observations:
            encoded.append(self.privileged_encoder(observations["privileged"]))
        fused = torch.cat(encoded, dim=1)
        if self.fuser is None:
            return fused
        return self.fuser(fused)

    @property
    def features_dim(self) -> int:
        return int(self._features_dim)
