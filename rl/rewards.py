from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardBreakdown:
    total: float
    survival_bonus: float
    command_tracking_reward: float
    absolute_work_penalty: float
    foot_jerk_penalty: float
    foot_drag_penalty: float
    collision_penalty: float
    stuck_penalty: float
    stair_riser_collision_penalty: float
    rear_leg_riser_collision_penalty: float
    step_completion_bonus: float
    rear_foot_step_bonus: float
    rear_follow_bonus: float
    rear_lag_penalty: float
    support_bonus: float
    multi_air_penalty: float
    low_speed_penalty: float
    fall_penalty: float


def compute_teacher_walk_reward(
    command_velocity: float,
    command_yaw_rate: float,
    forward_velocity: float,
    yaw_rate: float,
    actuator_torque: np.ndarray,
    joint_velocity: np.ndarray,
    foot_contact_forces: np.ndarray,
    previous_foot_contact_forces: np.ndarray,
    foot_horizontal_speeds: np.ndarray,
    collision_forces: np.ndarray,
    terminated: bool,
    weights: dict,
    *,
    stuck_penalty: float = 0.0,
    stair_riser_collision_penalty: float = 0.0,
    rear_leg_riser_collision_penalty: float = 0.0,
    step_completion_bonus: float = 0.0,
    rear_foot_step_bonus: float = 0.0,
    rear_follow_bonus: float = 0.0,
    rear_lag_penalty: float = 0.0,
    support_bonus: float = 0.0,
    multi_air_penalty: float = 0.0,
    low_speed_penalty: float = 0.0,
) -> RewardBreakdown:
    command_tracking_reward = float(
        weights["command_tracking"]
        * (
            command_velocity
            - abs(command_velocity - forward_velocity)
            - abs(command_yaw_rate - yaw_rate)
        )
    )
    absolute_work_penalty = float(
        weights["absolute_work"] * np.mean(np.abs(actuator_torque * joint_velocity))
    )

    foot_force_delta = np.linalg.norm(foot_contact_forces - previous_foot_contact_forces, axis=1)
    foot_jerk_penalty = float(weights["foot_jerk"] * np.sum(foot_force_delta))

    contact_force_threshold = float(weights["contact_force_threshold"])
    feet_in_contact = foot_contact_forces[:, 2] >= contact_force_threshold
    foot_drag_penalty = float(
        weights["foot_drag"] * np.sum(foot_horizontal_speeds[feet_in_contact])
    )

    collision_force_norms = np.linalg.norm(collision_forces, axis=1)
    collision_force_threshold = float(weights["collision_force_threshold"])
    collision_penalty = float(
        weights["collision"] * np.sum(collision_force_norms >= collision_force_threshold)
    )

    survival_bonus = float(weights["survival"] if not terminated else 0.0)
    fall_penalty = float(weights["fall"] if terminated else 0.0)

    total = (
        survival_bonus
        + command_tracking_reward
        + step_completion_bonus
        + rear_foot_step_bonus
        + rear_follow_bonus
        + support_bonus
        - absolute_work_penalty
        - foot_jerk_penalty
        - foot_drag_penalty
        - collision_penalty
        - stuck_penalty
        - stair_riser_collision_penalty
        - rear_leg_riser_collision_penalty
        - rear_lag_penalty
        - multi_air_penalty
        - low_speed_penalty
        - fall_penalty
    )
    return RewardBreakdown(
        total=total,
        survival_bonus=survival_bonus,
        command_tracking_reward=command_tracking_reward,
        absolute_work_penalty=absolute_work_penalty,
        foot_jerk_penalty=foot_jerk_penalty,
        foot_drag_penalty=foot_drag_penalty,
        collision_penalty=collision_penalty,
        stuck_penalty=float(stuck_penalty),
        stair_riser_collision_penalty=float(stair_riser_collision_penalty),
        rear_leg_riser_collision_penalty=float(rear_leg_riser_collision_penalty),
        step_completion_bonus=float(step_completion_bonus),
        rear_foot_step_bonus=float(rear_foot_step_bonus),
        rear_follow_bonus=float(rear_follow_bonus),
        rear_lag_penalty=float(rear_lag_penalty),
        support_bonus=float(support_bonus),
        multi_air_penalty=float(multi_air_penalty),
        low_speed_penalty=float(low_speed_penalty),
        fall_penalty=fall_penalty,
    )
