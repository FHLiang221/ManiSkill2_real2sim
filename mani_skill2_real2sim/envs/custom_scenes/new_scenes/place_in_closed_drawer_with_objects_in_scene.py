from collections import OrderedDict
from typing import List, Optional

import numpy as np
import cv2
import sapien.core as sapien
from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
from transforms3d.euler import euler2quat
from mani_skill2_real2sim.utils.common import random_choice
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult
from mani_skill2_real2sim.utils.sapien_utils import (
    get_pairwise_contacts,
    compute_total_impulse,
)

from .base_env import CustomOtherObjectsInSceneEnv, CustomSceneEnv
from .place_in_closed_drawer_in_scene import PlaceObjectInClosedDrawerInSceneEnv


class PlaceObjectInClosedDrawerWithObjectsInSceneEnv(PlaceObjectInClosedDrawerInSceneEnv):
    """Place object in closed drawer task with distractor objects on top of the drawer"""

    def __init__(
        self,
        distractor_object_ids: List[str] = None,
        force_advance_subtask_time_steps: int = 100,
        **kwargs,
    ):
        self.distractor_object_ids = distractor_object_ids or ["sponge", "apple"]
        self.distractor_objs = []

        super().__init__(
            force_advance_subtask_time_steps=force_advance_subtask_time_steps,
            **kwargs
        )

    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.contact_offset = 0.005  # avoid "false-positive" collisions
        return scene_config

    def _load_actors(self):
        # Load base environment (arena + target object)
        super()._load_actors()
        # Load additional distractor objects
        self._load_distractor_objects()

        # Set damping for distractor objects
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
        # First initialize the main object as per the parent class
        super()._initialize_actors()

        # Now add distractor objects on the closed drawer
        if len(self.distractor_objs) > 0:
            # Get drawer position to place distractor objects on top
            # The drawer_link is set in the parent reset method
            drawer_link = get_entity_by_name(
                self.art_obj.get_links(), f"{self.drawer_id}_drawer"
            )
            drawer_pos = drawer_link.pose.p
            drawer_top_z = drawer_pos[2] + 0.4  # Place objects on top of closed drawer

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

            # Lock distractor objects to prevent rolling while settling
            for obj in self.distractor_objs:
                obj.lock_motion(0, 0, 0, 1, 1, 0)

            self._settle(0.5)

            # Unlock distractor objects
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

    def _get_distractor_positions(self, drawer_center_xy):
        """Generate positions for distractor objects on the drawer top"""
        positions = []
        for i in range(len(self.distractor_object_ids)):
            # Place distractors in a circle pattern on drawer top
            angle = 2 * np.pi * i / len(self.distractor_object_ids)
            radius = 0.06 + self._episode_rng.uniform(0, 0.02)
            x = drawer_center_xy[0] + radius * np.cos(angle)
            y = drawer_center_xy[1] + radius * np.sin(angle)
            positions.append([x, y])
        return positions

    def reset(self, seed=None, options=None):
        # Remove any existing distractor objects from previous episodes
        for obj in self.distractor_objs:
            if obj in self._scene.get_all_actors():
                self._scene.remove_actor(obj)
        self.distractor_objs = []

        return super().reset(seed=seed, options=options)


# Registered environments

@register_env("PlaceIntoClosedDrawerWithObjectsCustomInScene-v0", max_episode_steps=200)
class PlaceIntoClosedDrawerWithObjectsCustomInSceneEnv(
    PlaceObjectInClosedDrawerWithObjectsInSceneEnv, CustomOtherObjectsInSceneEnv
):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v1.json"
    drawer_ids = ["top", "middle", "bottom"]


@register_env("PlaceIntoClosedTopDrawerWithObjectsCustomInScene-v0", max_episode_steps=200)
class PlaceIntoClosedTopDrawerWithObjectsCustomInSceneEnv(PlaceIntoClosedDrawerWithObjectsCustomInSceneEnv):
    drawer_ids = ["top"]


@register_env("PlaceIntoClosedMiddleDrawerWithObjectsCustomInScene-v0", max_episode_steps=200)
class PlaceIntoClosedMiddleDrawerWithObjectsCustomInSceneEnv(
    PlaceIntoClosedDrawerWithObjectsCustomInSceneEnv
):
    drawer_ids = ["middle"]


@register_env("PlaceIntoClosedBottomDrawerWithObjectsCustomInScene-v0", max_episode_steps=200)
class PlaceIntoClosedBottomDrawerWithObjectsCustomInSceneEnv(
    PlaceIntoClosedDrawerWithObjectsCustomInSceneEnv
):
    drawer_ids = ["bottom"]