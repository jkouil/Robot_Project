from __future__ import annotations

from typing import Generator

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.utils import FloatSchedule

from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent import RecurrentPPO


class _ChunkedSequenceMixin:
    recurrent_sequence_length: int | None

    def _chunked_indices(self, batch_size: int) -> Generator[np.ndarray, None, None]:
        total_steps = self.buffer_size * self.n_envs
        sequence_length = int(self.recurrent_sequence_length or 0)
        if sequence_length <= 0:
            yield np.arange(total_steps)
            return

        chunks: list[np.ndarray] = []
        for env_idx in range(self.n_envs):
            env_offset = env_idx * self.buffer_size
            for start in range(0, self.buffer_size, sequence_length):
                end = min(start + sequence_length, self.buffer_size)
                chunks.append(np.arange(env_offset + start, env_offset + end))

        np.random.shuffle(chunks)
        batch_chunks: list[np.ndarray] = []
        batch_transition_count = 0
        for chunk in chunks:
            if batch_chunks and batch_transition_count + len(chunk) > batch_size:
                yield np.concatenate(batch_chunks)
                batch_chunks = []
                batch_transition_count = 0
            batch_chunks.append(chunk)
            batch_transition_count += len(chunk)
        if batch_chunks:
            yield np.concatenate(batch_chunks)

    def _chunk_env_change(self) -> np.ndarray:
        sequence_length = int(self.recurrent_sequence_length or 0)
        env_change = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        env_change[0, :] = 1.0
        if sequence_length > 0:
            env_change[::sequence_length, :] = 1.0
        return self.swap_and_flatten(env_change)


class PaperBpttRecurrentRolloutBuffer(_ChunkedSequenceMixin, RecurrentRolloutBuffer):
    def __init__(self, *args, recurrent_sequence_length: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recurrent_sequence_length = recurrent_sequence_length

    def get(self, batch_size: int | None = None):
        assert self.full, "Rollout buffer must be full before sampling from it"

        if not self.generator_ready:
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        env_change = self._chunk_env_change()
        for batch_inds in self._chunked_indices(batch_size):
            yield self._get_samples(batch_inds, env_change)


class PaperBpttRecurrentDictRolloutBuffer(_ChunkedSequenceMixin, RecurrentDictRolloutBuffer):
    def __init__(self, *args, recurrent_sequence_length: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recurrent_sequence_length = recurrent_sequence_length

    def get(self, batch_size: int | None = None):
        assert self.full, "Rollout buffer must be full before sampling from it"

        if not self.generator_ready:
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        env_change = self._chunk_env_change()
        for batch_inds in self._chunked_indices(batch_size):
            yield self._get_samples(batch_inds, env_change)


class PaperBpttRecurrentPPO(RecurrentPPO):
    def __init__(self, *args, recurrent_sequence_length: int, **kwargs):
        self.recurrent_sequence_length = int(recurrent_sequence_length)
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = (
            PaperBpttRecurrentDictRolloutBuffer
            if isinstance(self.observation_space, spaces.Dict)
            else PaperBpttRecurrentRolloutBuffer
        )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        lstm = self.policy.lstm_actor
        if not isinstance(self.policy, RecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )

        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            recurrent_sequence_length=self.recurrent_sequence_length,
        )

        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)
