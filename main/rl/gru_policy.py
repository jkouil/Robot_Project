from __future__ import annotations

from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy


class ScandotGruPolicy(RecurrentMultiInputActorCriticPolicy):
    """GRU-based recurrent policy with the same public state interface as sb3-contrib LSTM policies."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )

        self.lstm_actor = nn.GRU(
            self.features_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            **self.lstm_kwargs,
        )
        if self.enable_critic_lstm:
            self.lstm_critic = nn.GRU(
                self.features_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        lstm: nn.GRU,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        hidden_state = lstm_states[0]
        n_seq = hidden_state.shape[1]
        features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        if th.all(episode_starts == 0.0):
            gru_output, hidden_state = lstm(features_sequence, hidden_state)
            gru_output = th.flatten(gru_output.transpose(0, 1), start_dim=0, end_dim=1)
            return gru_output, (hidden_state, th.zeros_like(hidden_state))

        gru_output = []
        for features_step, episode_start in zip(features_sequence, episode_starts, strict=True):
            hidden_state = (1.0 - episode_start).view(1, n_seq, 1) * hidden_state
            hidden, hidden_state = lstm(features_step.unsqueeze(dim=0), hidden_state)
            gru_output.append(hidden)
        gru_output = th.flatten(th.cat(gru_output).transpose(0, 1), start_dim=0, end_dim=1)
        return gru_output, (hidden_state, th.zeros_like(hidden_state))
