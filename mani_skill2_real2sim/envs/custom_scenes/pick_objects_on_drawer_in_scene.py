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
from .open_drawer_in_scene import OpenDrawerInSceneEnv


class PickObjectOnDrawerInSceneEnv(OpenDrawerInSceneEnv):
    """Base class for picking objects that are placed on top of a closed drawer with distractor objects"""

    def __init__(
        self,
        target_object_id: str = "opened_coke_can",
        distractor_object_ids: List[str] = None,
        require_lifting_obj_for_success: bool = True,
        success_from_episode_stats: bool = True,
        **kwargs,
    ):
        self.target_object_id = target_object_id
        self.distractor_object_ids = distractor_object_ids or ["sponge", "apple"]

        self.target_obj = None
        self.distractor_objs = []

        self.require_lifting_obj_for_success = require_lifting_obj_for_success
        self.success_from_episode_stats = success_from_episode_stats
        self.consecutive_grasp = 0
        self.lifted_obj = False
        self.obj_height_after_settle = None

        # Object initialization options
        self.obj_init_options = {}

        super().__init__(**kwargs)

    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.contact_offset = 0.005  # avoid "false-positive" collisions
        return scene_config

    def _load_actors(self):
        # Load the drawer/cabinet scene
        self._load_arena_helper(add_collision=False)
        # Load target and distractor objects
        self._load_objects()

        # Set damping for all objects
        self.target_obj.set_damping(0.1, 0.1)
        for obj in self.distractor_objs:
            obj.set_damping(0.1, 0.1)

    def _load_objects(self):
        """Load target object and distractor objects"""
        # Load target object
        target_density = self.model_db[self.target_object_id].get("density", 1000)
        self.target_obj = self._build_actor_helper(
            self.target_object_id,
            self._scene,
            scale=1.0,
            density=target_density,
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0,
            ),
            root_dir=self.asset_root,
        )
        self.target_obj.name = self.target_object_id

        # Load distractor objects
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
        # Initialize the drawer (keep it closed)
        self.art_obj.set_qpos([0.0] * self.art_obj.dof)

        # Get drawer position to place objects on top
        drawer_link = self.drawer_obj
        drawer_pos = drawer_link.pose.p
        drawer_top_z = drawer_pos[2] + 0.4  # Place objects on top of closed drawer

        # Place target object on drawer
        target_xy = self.obj_init_options.get("target_init_xy", None)
        if target_xy is None:
            target_xy = drawer_pos[:2] + self._episode_rng.uniform([-0.05, -0.05], [0.05, 0.05])

        target_init_z = drawer_top_z + 0.5  # Let object fall onto drawer top
        target_p = np.array([target_xy[0], target_xy[1], target_init_z])
        target_q = self.obj_init_options.get("target_init_rot_quat", [1, 0, 0, 0])

        # Apply random rotation if specified
        if self.obj_init_options.get("init_rand_rot_z", False):
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            target_q = qmult(euler2quat(0, 0, ori), target_q)

        self.target_obj.set_pose(sapien.Pose(target_p, target_q))

        # Place distractor objects on drawer top
        distractor_positions = self._get_distractor_positions(drawer_pos[:2], drawer_top_z)
        for i, (distractor_obj, distractor_pos) in enumerate(zip(self.distractor_objs, distractor_positions)):
            distractor_p = np.array([distractor_pos[0], distractor_pos[1], target_init_z])
            distractor_q = [1, 0, 0, 0]

            # Add some random rotation for variety
            if self.obj_init_options.get("init_rand_rot_z", True):
                ori = self._episode_rng.uniform(0, 2 * np.pi)
                distractor_q = qmult(euler2quat(0, 0, ori), distractor_q)

            distractor_obj.set_pose(sapien.Pose(distractor_p, distractor_q))

        # Move robot away initially
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # Lock objects to prevent rolling while settling
        self.target_obj.lock_motion(0, 0, 0, 1, 1, 0)
        for obj in self.distractor_objs:
            obj.lock_motion(0, 0, 0, 1, 1, 0)

        self._settle(0.5)

        # Unlock objects
        self.target_obj.lock_motion(0, 0, 0, 0, 0, 0)
        self.target_obj.set_pose(self.target_obj.pose)
        self.target_obj.set_velocity(np.zeros(3))
        self.target_obj.set_angular_velocity(np.zeros(3))

        for obj in self.distractor_objs:
            obj.lock_motion(0, 0, 0, 0, 0, 0)
            obj.set_pose(obj.pose)
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))

        self._settle(0.5)

        # Record object height after settling
        self.obj_height_after_settle = self.target_obj.pose.p[2]

        # Additional settling if objects are still moving
        total_vel = np.linalg.norm(self.target_obj.velocity)
        for obj in self.distractor_objs:
            total_vel += np.linalg.norm(obj.velocity)
        if total_vel > 1e-3:
            self._settle(1.5)

    def _get_distractor_positions(self, drawer_center_xy, drawer_top_z):
        """Generate positions for distractor objects around the target"""
        positions = []
        for i in range(len(self.distractor_object_ids)):
            # Place distractors in a circle around the drawer center
            angle = 2 * np.pi * i / len(self.distractor_object_ids)
            radius = 0.08 + self._episode_rng.uniform(0, 0.04)  # Small radius on drawer top
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

        if options is None:
            options = dict()
        options = options.copy()

        self.obj_init_options = options.get("obj_init_options", {})

        # Initialize episode stats
        self.consecutive_grasp = 0
        self.lifted_obj = False
        self.obj_height_after_settle = None
        self._initialize_episode_stats()

        return super().reset(seed=seed, options=options)

    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(
            n_lift_significant=0,
            consec_grasp=False,
            grasped=False,
        )

    @property
    def target_obj_pose(self):
        """Get the center of mass (COM) pose of target object."""
        return self.target_obj.pose.transform(self.target_obj.cmass_local_pose)

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                target_obj_pose=vectorize_pose(self.target_obj_pose),
                tcp_to_target_obj_pos=self.target_obj_pose.p - self.tcp.pose.p,
            )
        return obs

    def evaluate(self, **kwargs):
        # Check if target object is grasped
        is_grasped = self.agent.check_grasp(self.target_obj, max_angle=80)
        if is_grasped:
            self.consecutive_grasp += 1
        else:
            self.consecutive_grasp = 0
            self.lifted_obj = False

        # Check if object is lifted (not in contact with non-robot surfaces)
        contacts = self._scene.get_contacts()
        flag = True
        robot_link_names = [x.name for x in self.agent.robot.get_links()]

        for contact in contacts:
            actor_0, actor_1 = contact.actor0, contact.actor1
            other_obj_contact_actor_name = None
            if actor_0.name == self.target_obj.name:
                other_obj_contact_actor_name = actor_1.name
            elif actor_1.name == self.target_obj.name:
                other_obj_contact_actor_name = actor_0.name

            if other_obj_contact_actor_name is not None:
                contact_impulse = np.sum([point.impulse for point in contact.points], axis=0)
                if (other_obj_contact_actor_name not in robot_link_names) and (
                    np.linalg.norm(contact_impulse) > 1e-6
                ):
                    flag = False
                    break

        consecutive_grasp = self.consecutive_grasp >= 5
        diff_obj_height = self.target_obj.pose.p[2] - self.obj_height_after_settle
        self.lifted_obj = self.lifted_obj or (flag and (diff_obj_height > 0.10))
        lifted_object_significantly = self.lifted_obj and (diff_obj_height > 0.10)

        if self.require_lifting_obj_for_success:
            success = self.lifted_obj
        else:
            success = consecutive_grasp

        # Update episode stats
        self.episode_stats["n_lift_significant"] += int(lifted_object_significantly)
        self.episode_stats["consec_grasp"] = (
            self.episode_stats["consec_grasp"] or consecutive_grasp
        )
        self.episode_stats["grasped"] = self.episode_stats["grasped"] or is_grasped

        if self.success_from_episode_stats:
            success = success or (self.episode_stats["n_lift_significant"] >= 5)

        return dict(
            is_grasped=is_grasped,
            consecutive_grasp=consecutive_grasp,
            lifted_object=self.lifted_obj,
            lifted_object_significantly=lifted_object_significantly,
            success=success,
            episode_stats=self.episode_stats,
        )

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.target_object_id)
        return f"pick {obj_name}"


# Specific implementations for different target objects

@register_env("PickCokeCanOnClosedDrawerInScene-v0", max_episode_steps=10000)
class PickCokeCanOnClosedDrawerInSceneEnv(PickObjectOnDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top", "middle", "bottom"]

    def __init__(self, **kwargs):
        super().__init__(
            target_object_id="opened_coke_can",
            distractor_object_ids=["sponge", "apple"],
            **kwargs
        )


@register_env("PickSpongeOnClosedDrawerInScene-v0", max_episode_steps=10000)
class PickSpongeOnClosedDrawerInSceneEnv(PickObjectOnDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top", "middle", "bottom"]

    def __init__(self, **kwargs):
        super().__init__(
            target_object_id="sponge",
            distractor_object_ids=["opened_coke_can", "apple"],
            **kwargs
        )


@register_env("PickAppleOnClosedDrawerInScene-v0", max_episode_steps=10000)
class PickAppleOnClosedDrawerInSceneEnv(PickObjectOnDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top", "middle", "bottom"]

    def __init__(self, **kwargs):
        super().__init__(
            target_object_id="apple",
            distractor_object_ids=["opened_coke_can", "sponge"],
            **kwargs
        )