# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from dataclasses import dataclass, field

import draccus

from lerobot.common.constants import ACTION, OBS_ENV, OBS_IMAGE, OBS_IMAGES, OBS_ROBOT
from lerobot.configs.types import FeatureType, PolicyFeature


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    task: str = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "environment_state": OBS_ENV,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(type=FeatureType.VISUAL, shape=(384, 384, 3))
        elif self.obs_type == "environment_state_agent_pos":
            self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("xarm")
@dataclass
class XarmEnv(EnvConfig):
    task: str = "XarmLift-v0"
    fps: int = 15
    episode_length: int = 200
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(84, 84, 3)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("lowcostrobot")
@dataclass
class LowCostRobotEnv(EnvConfig):
    task: str = "PushCube-v0"
    fps: int = 25
    module: str = "gym_lowcostrobot"

    max_episode_steps: int = 50

    observation_mode: str = "state"
    action_mode: str = "joint"
    reward_type: str = "dense"
    block_gripper: bool = True
    distance_threshold: float = 0.05
    cube_xy_range: float = 0.3
    n_substeps: int = 20
    render_mode: str | None = None

    # only for PushCube-v0
    target_xy_range: float = 0.3
    simulation_timestep: float = 0.002
    robot_observation_mode: str = "joint"
    cube_vel: bool = False

    def __post_init__(self):
        if self.observation_mode in ['image', 'both'] or not self.block_gripper or self.reward_type == 'sparse': raise NotImplementedError

        action_shape = {"joint": 5, "ee": 3}[self.action_mode]
        action_shape += 0 if self.block_gripper else 1

        self.filter_keys, robot_state_numel = self.filter_map[self.robot_observation_mode]
        self.filter_keys += ['cube_pos'] if self.observation_mode == 'state' else ['image_front', 'image_top']
        self.filter_keys += ['target_pos'] if (self.task == "PushCube-v0" and self.observation_mode == 'state') else []
        self.filter_keys += ['cube_vel'] if self.cube_vel else []

        from gymnasium.wrappers import FilterObservation, FlattenObservation
        self.wrappers = [lambda env: FilterObservation(env, self.filter_keys),
                         lambda env: FlattenObservation(env),]

        env_state_numel = (3+3 if self.cube_vel else 3) + (3 if self.task == "PushCube-v0" else 0) # if self.observation_mode == 'state'
        self.state_numel = robot_state_numel + env_state_numel
        self.features = {
            "flattened_state": PolicyFeature(type=FeatureType.STATE, shape=(self.state_numel,)),
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_shape,)),
        }
        self.features_map = {
            "flattened_state": OBS_ROBOT,
            "action": ACTION,
        }

    @property
    def filter_map(self) -> dict[str | None, tuple[list[str], int]]:
        return {
            None: ([], 0),
            'joint': (['arm_qpos', 'arm_qvel'], 6+6),
            'ee': (['ee_xpos', 'ee_xvel'], 3+3),
            'all': (['arm_qpos', 'arm_qvel', 'ee_xpos', 'ee_xvel'], 6+6+3+3)
        }

    @property
    def gym_kwargs(self) -> dict:
        return {
            "max_episode_steps": self.max_episode_steps,
            "observation_mode": self.observation_mode,
            "action_mode": self.action_mode,
            "reward_type": self.reward_type,
            "block_gripper": self.block_gripper,
            "distance_threshold": self.distance_threshold,
            "cube_xy_range": self.cube_xy_range,
            "n_substeps": self.n_substeps,
            "render_mode": self.render_mode,
        } | ({
            "target_xy_range": self.target_xy_range,
            "simulation_timestep": self.simulation_timestep,
        } if self.task == "PushCube-v0" else {})
