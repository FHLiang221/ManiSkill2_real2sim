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

from .base_env import CustomSceneEnv, CustomOtherObjectsInSceneEnv
from .open_drawer_in_scene import OpenDrawerInSceneEnv, CloseDrawerInSceneEnv


class DrawerWithObjectsInSceneEnv(OpenDrawerInSceneEnv):
    """Base class for drawer tasks with distractor objects on top of the drawer"""

    def __init__(
        self,
        distractor_object_ids: List[str] = None,
        **kwargs,
    ):
        self.distractor_object_ids = distractor_object_ids or ["opened_coke_can", "sponge", "apple"]
        self.distractor_objs = []
        self.obj_init_options = {}

        super().__init__(**kwargs)

    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.contact_offset = 0.005  # avoid "false-positive" collisions
        return scene_config

    def _load_actors(self):
        # Load the drawer/cabinet scene
        self._load_arena_helper(add_collision=False)
        # Load distractor objects
        self._load_distractor_objects()

        # Set damping for all objects
        for obj in self.distractor_objs:
            obj.set_damping(0.1, 0.1)

    def _load_distractor_objects(self):
        """Load distractor objects that will be placed on the drawer"""
        for distractor_id in self.distractor_object_ids:
            distractor_density = self.model_db[distractor_id].get("density", 1000)
            distractor_obj = self._build_actor_helper(
                distractor_id,
                self._scene,
                scale=1.0,
                density=distractor_density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            distractor_obj.name = distractor_id
            self.distractor_objs.append(distractor_obj)

    def _initialize_actors(self):
        # Initialize the drawer position based on task type
        initial_drawer_qpos = self._get_initial_drawer_qpos()
        if initial_drawer_qpos is not None:
            qpos = [0.0] * self.art_obj.dof
            qpos[self.joint_idx] = initial_drawer_qpos
            self.art_obj.set_qpos(qpos)

        # Get drawer position to place objects on top
        drawer_link = self.drawer_obj
        drawer_pos = drawer_link.pose.p
        drawer_top_z = drawer_pos[2] + 0.4  # Place objects on top of drawer

        # Place distractor objects on drawer top
        distractor_positions = self._get_distractor_positions(drawer_pos[:2])
        for i, (distractor_obj, distractor_pos) in enumerate(zip(self.distractor_objs, distractor_positions)):
            distractor_init_z = drawer_top_z + 0.5  # Let objects fall onto drawer top
            distractor_p = np.array([distractor_pos[0], distractor_pos[1], distractor_init_z])
            distractor_q = [1, 0, 0, 0]

            # Add some random rotation for variety
            if self.obj_init_options.get("init_rand_rot_z", True):
                ori = self._episode_rng.uniform(0, 2 * np.pi)
                distractor_q = qmult(euler2quat(0, 0, ori), distractor_q)

            distractor_obj.set_pose(sapien.Pose(distractor_p, distractor_q))

        # Move robot away initially
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # Lock objects to prevent rolling while settling
        for obj in self.distractor_objs:
            obj.lock_motion(0, 0, 0, 1, 1, 0)

        self._settle(0.5)

        # Unlock objects
        for obj in self.distractor_objs:
            obj.lock_motion(0, 0, 0, 0, 0, 0)
            obj.set_pose(obj.pose)
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))

        self._settle(0.5)

        # Additional settling if objects are still moving
        total_vel = 0.0
        for obj in self.distractor_objs:
            total_vel += np.linalg.norm(obj.velocity)
        if total_vel > 1e-3:
            self._settle(1.5)

    def _get_initial_drawer_qpos(self):
        """Override in subclasses to set initial drawer position"""
        return 0.0  # Default: closed drawer

    def _get_distractor_positions(self, drawer_center_xy):
        """Generate positions for distractor objects on the drawer top"""
        positions = []
        for i in range(len(self.distractor_object_ids)):
            # Place distractors in a grid/circle pattern on drawer top
            if len(self.distractor_object_ids) <= 3:
                # Small circle for few objects
                angle = 2 * np.pi * i / len(self.distractor_object_ids)
                radius = 0.06 + self._episode_rng.uniform(0, 0.02)
                x = drawer_center_xy[0] + radius * np.cos(angle)
                y = drawer_center_xy[1] + radius * np.sin(angle)
            else:
                # Grid pattern for more objects
                grid_size = int(np.ceil(np.sqrt(len(self.distractor_object_ids))))
                row = i // grid_size
                col = i % grid_size
                x_offset = (col - grid_size/2) * 0.04
                y_offset = (row - grid_size/2) * 0.04
                x = drawer_center_xy[0] + x_offset + self._episode_rng.uniform(-0.01, 0.01)
                y = drawer_center_xy[1] + y_offset + self._episode_rng.uniform(-0.01, 0.01)

            positions.append([x, y])
        return positions

    def reset(self, seed=None, options=None):
        # Remove any existing distractor objects from previous episodes
        for obj in self.distractor_objs:
            if obj in self._scene.get_all_actors():
                self._scene.remove_actor(obj)
        self.distractor_objs = []

        if options is None:
            options = dict()
        options = options.copy()

        self.obj_init_options = options.get("obj_init_options", {})

        return super().reset(seed=seed, options=options)


class OpenDrawerWithObjectsInSceneEnv(DrawerWithObjectsInSceneEnv):
    """Open drawer task with distractor objects on top"""

    def _get_initial_drawer_qpos(self):
        """Start with closed drawer for opening task"""
        return 0.0


class CloseDrawerWithObjectsInSceneEnv(DrawerWithObjectsInSceneEnv, CloseDrawerInSceneEnv):
    """Close drawer task with distractor objects on top"""

    def _get_initial_drawer_qpos(self):
        """Start with open drawer for closing task"""
        return 0.2


# Registered environments for opening drawers with objects

@register_env("OpenDrawerWithObjectsCustomInScene-v0", max_episode_steps=113)
class OpenDrawerWithObjectsCustomInSceneEnv(OpenDrawerWithObjectsInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top", "middle", "bottom"]


@register_env("OpenTopDrawerWithObjectsCustomInScene-v0", max_episode_steps=113)
class OpenTopDrawerWithObjectsCustomInSceneEnv(OpenDrawerWithObjectsCustomInSceneEnv):
    drawer_ids = ["top"]


@register_env("OpenMiddleDrawerWithObjectsCustomInScene-v0", max_episode_steps=113)
class OpenMiddleDrawerWithObjectsCustomInSceneEnv(OpenDrawerWithObjectsCustomInSceneEnv):
    drawer_ids = ["middle"]


@register_env("OpenBottomDrawerWithObjectsCustomInScene-v0", max_episode_steps=113)
class OpenBottomDrawerWithObjectsCustomInSceneEnv(OpenDrawerWithObjectsCustomInSceneEnv):
    drawer_ids = ["bottom"]


# Registered environments for closing drawers with objects

@register_env("CloseDrawerWithObjectsCustomInScene-v0", max_episode_steps=113)
class CloseDrawerWithObjectsCustomInSceneEnv(CloseDrawerWithObjectsInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top", "middle", "bottom"]


@register_env("CloseTopDrawerWithObjectsCustomInScene-v0", max_episode_steps=113)
class CloseTopDrawerWithObjectsCustomInSceneEnv(CloseDrawerWithObjectsCustomInSceneEnv):
    drawer_ids = ["top"]


@register_env("CloseMiddleDrawerWithObjectsCustomInScene-v0", max_episode_steps=113)
class CloseMiddleDrawerWithObjectsCustomInSceneEnv(CloseDrawerWithObjectsCustomInSceneEnv):
    drawer_ids = ["middle"]


@register_env("CloseBottomDrawerWithObjectsCustomInScene-v0", max_episode_steps=113)
class CloseBottomDrawerWithObjectsCustomInSceneEnv(CloseDrawerWithObjectsCustomInSceneEnv):
    drawer_ids = ["bottom"]