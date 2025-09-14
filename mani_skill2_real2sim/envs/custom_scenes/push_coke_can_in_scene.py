from collections import OrderedDict
from typing import List, Optional

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.common import random_choice
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import vectorize_pose

from .base_env import CustomOtherObjectsInSceneEnv, CustomSceneEnv


class PushCokeCanInSceneEnv(CustomSceneEnv):
    obj: sapien.Actor  # target coke can to push

    def __init__(
        self,
        success_from_episode_stats: bool = True,
        **kwargs,
    ):
        self.model_id = None
        self.model_scale = None
        self.model_bbox_size = None

        self.obj = None
        self.obj_init_options = {}

        self.success_from_episode_stats = success_from_episode_stats
        self.obj_initial_pos = None
        self.obj_push_distance = 0.0
        self.episode_stats = None

        super().__init__(**kwargs)

        # Add same visual elements as grasp task - black table and gray walls
        builder = self._scene.create_actor_builder()
        builder.add_box_visual(half_size=[5, 5, 0.87], color=[0.05, 0.05, 0.05])
        dummy_table = builder.build_static()
        dummy_table.set_pose(sapien.Pose([0, 0, 0.019], [0, 0, 0, 1]))

        builder = self._scene.create_actor_builder()
        builder.add_box_visual(half_size=[2, 0.01, 2], color=[0.48, 0.48, 0.48])
        dummy_left = builder.build_static()
        dummy_left.set_pose(sapien.Pose([0, -0.28, 0], [0, 0, 0, 1]))

        builder = self._scene.create_actor_builder()
        builder.add_box_visual(half_size=[2, 0.01, 2], color=[0.48, 0.48, 0.48])
        dummy_right = builder.build_static()
        dummy_right.set_pose(sapien.Pose([0, 0.76, 0], [0, 0, 0, 1]))

    def _load_actors(self):
        self._load_arena_helper()
        self._load_model()
        self.obj.set_damping(0.1, 0.1)

    def _load_model(self):
        """Load the target coke can."""
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.obj_init_options = options.get("obj_init_options", {})

        # Set objects
        self.set_episode_rng(seed)
        model_scale = options.get("model_scale", None)
        model_id = options.get("model_id", None)
        reconfigure = options.get("reconfigure", False)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure

        options["reconfigure"] = reconfigure

        self.obj_push_distance = 0.0
        self.obj_initial_pos = None
        # episode-level info
        self._initialize_episode_stats()

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update(
            {
                "model_id": self.model_id,
                "model_scale": self.model_scale,
                "obj_init_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.obj.pose,
            }
        )
        return obs, info

    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(
            moved_towards_camera=False,
            success=False,
        )

    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow
        # Use same lighting as grasp task for consistency
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        # default lighting
        self._scene.add_directional_light(
            [0, 0, -1],
            [2.2, 2.2, 2.2],
            shadow=shadow,
            scale=5,
            shadow_map_size=2048,
        )
        self._scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self._scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _set_model(self, model_id, model_scale):
        """Set the model id and scale. If not provided, choose one randomly from self.model_ids."""
        reconfigure = False

        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            self.model_id = model_id
            reconfigure = True

        if model_scale is None:
            model_scales = self.model_db[self.model_id].get("scales")
            if model_scales is None:
                model_scale = 1.0
            else:
                model_scale = random_choice(model_scales, self._episode_rng)
        if model_scale != self.model_scale:
            self.model_scale = model_scale
            reconfigure = True

        model_info = self.model_db[self.model_id]
        if "bbox" in model_info:
            bbox = model_info["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_size = bbox_size * self.model_scale
        else:
            self.model_bbox_size = None

        return reconfigure

    def _initialize_actors(self):
        # Initialize the coke can in vertical position (laid down)
        obj_init_xy = self.obj_init_options.get("init_xy", None)
        if obj_init_xy is None:
            # Fixed position for consistent push task
            obj_init_xy = np.array([-0.25, 0.05])  # Fixed: X=-0.25, Y=0.05 (further from robot)
        obj_init_z = self.obj_init_options.get(
            "init_z", self.scene_table_height
        )
        obj_init_z = obj_init_z + 0.5  # let object fall onto the table

        # Always set to vertical/laid down orientation for push task
        obj_init_rot_quat = self.obj_init_options.get(
            "init_rot_quat", euler2quat(0, 0, np.pi / 2)  # laid vertically (90° around Z)
        )
        p = np.hstack([obj_init_xy, obj_init_z])
        q = obj_init_rot_quat

        # Rotate along z-axis
        if self.obj_init_options.get("init_rand_rot_z", False):
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = qmult(euler2quat(0, 0, ori), q)

        # Rotate along a random axis by a small angle
        if (
            init_rand_axis_rot_range := self.obj_init_options.get(
                "init_rand_axis_rot_range", 0.0
            )
        ) > 0:
            axis = self._episode_rng.uniform(-1, 1, 3)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            ori = self._episode_rng.uniform(0, init_rand_axis_rot_range)
            q = qmult(q, axangle2quat(axis, ori, True))
        self.obj.set_pose(sapien.Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later in _initialize_agent (in base_env.py)
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # Lock rotation around x and y to let the target object fall onto the table
        self.obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)

        # Unlock motion
        self.obj.lock_motion(0, 0, 0, 0, 0, 0)
        # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
        self.obj.set_pose(self.obj.pose)
        self.obj.set_velocity(np.zeros(3))
        self.obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.obj.velocity)
        ang_vel = np.linalg.norm(self.obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(1.5)

        # Record the object initial position for push distance calculation
        self.obj_initial_pos = self.obj.pose.p.copy()

    @property
    def obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.obj.pose.transform(self.obj.cmass_local_pose)

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=vectorize_pose(self.obj_pose),
                tcp_to_obj_pos=self.obj_pose.p - self.tcp.pose.p,
            )
        return obs

    def evaluate(self, **kwargs):
        # Success criteria: coke can moved downward (positive X direction)
        # Failure criteria: coke can moved too far left or right (Y direction)

        current_pos = self.obj.pose.p

        # Check if can has moved downward (positive X direction)
        moved_up = False
        failed_sideways = False
        failed_backward = False
        if self.obj_initial_pos is not None:
            x_displacement = current_pos[0] - self.obj_initial_pos[0]  # Positive when moved up
            y_displacement = abs(current_pos[1] - self.obj_initial_pos[1])  # Absolute sideways movement

            moved_up = x_displacement > 0.1  # 10cm threshold for downward movement
            failed_sideways = y_displacement > 0.02  # 2cm threshold for sideways movement (failure)
            failed_backward = x_displacement < -0.01  # 1cm threshold for backward movement (failure)

            # Track max displacement for stats
            self.obj_push_distance = max(self.obj_push_distance, x_displacement)
        else:
            x_displacement = 0.0
            y_displacement = 0.0

        # Success if can moved downward AND didn't move too far sideways AND didn't move backward
        success = moved_up and not failed_sideways and not failed_backward

        self.episode_stats["moved_towards_camera"] = (
            self.episode_stats.get("moved_towards_camera", False) or moved_up
        )
        self.episode_stats["success"] = (
            self.episode_stats["success"] or success
        )

        # Mark episode as failed if failure conditions are met
        if failed_sideways or failed_backward:
            self.episode_stats["success"] = False
            success = False

        if self.success_from_episode_stats:
            success = self.episode_stats["success"]

        return dict(
            moved_towards_camera=moved_up,
            x_displacement=x_displacement,
            y_displacement=y_displacement,
            failed_sideways=failed_sideways,
            failed_backward=failed_backward,
            push_distance=self.obj_push_distance,
            success=success,
            episode_stats=self.episode_stats,
        )

    def get_done(self, info: dict, **kwargs):
        # Episode terminates on success OR failure (sideways/backward movement)
        return bool(info["success"]) or bool(info.get("failed_sideways", False)) or bool(info.get("failed_backward", False))


# ---------------------------------------------------------------------------- #
# Custom Assets
# ---------------------------------------------------------------------------- #


@register_env("PushCokeCanInScene-v0", max_episode_steps=80)
class PushCokeCanInSceneEnv(
    PushCokeCanInSceneEnv, CustomOtherObjectsInSceneEnv
):
    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)

        self.obj = self._build_actor_helper(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=density,
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0,
            ),
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"push {obj_name} downward"
        return task_description


@register_env("PushSingleCokeCanInScene-v0", max_episode_steps=80)
class PushSingleCokeCanInSceneEnv(PushCokeCanInSceneEnv):
    def __init__(self, laid_vertically=True, **kwargs):
        # Remove orientation parameters that the parent class doesn't handle
        kwargs.pop("laid_vertically", None)
        kwargs.pop("upright", None)
        kwargs.pop("lr_switch", None)

        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["coke_can"]
        super().__init__(**kwargs)


@register_env("PushSingleOpenedCokeCanInScene-v0", max_episode_steps=80)
class PushSingleOpenedCokeCanInSceneEnv(PushCokeCanInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["opened_coke_can"]
        super().__init__(**kwargs)