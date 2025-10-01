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
from transforms3d.quaternions import axangle2quat, qmult
from mani_skill2_real2sim.utils.sapien_utils import vectorize_pose

from .base_env import CustomOtherObjectsInSceneEnv, CustomSceneEnv
from .open_drawer_in_scene import OpenDrawerInSceneEnv


class PickObjectOnDrawerInSceneEnv(OpenDrawerInSceneEnv):
    """Pick an object placed on top of a closed drawer"""

    def __init__(
        self,
        **kwargs,
    ):
        self.model_id = None
        self.model_scale = None
        self.model_bbox_size = None
        self.obj = None
        self.obj_init_options = {}
        self.obj_height_after_settle = None

        super().__init__(**kwargs)

    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.contact_offset = 0.005  # avoid "false-positive" collisions
        return scene_config

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

    def _load_actors(self):
        super()._load_actors()
        self._load_model()
        self.obj.set_damping(0.1, 0.1)

    def _initialize_actors(self):
        # Keep the drawer closed (no opening needed for pick task)
        self.art_obj.set_qpos([0.0] * self.art_obj.dof)

        # The object will fall from a certain initial height
        obj_init_xy = self.obj_init_options.get("init_xy", None)
        if obj_init_xy is None:
            obj_init_xy = self._episode_rng.uniform([-0.10, -0.00], [-0.05, 0.1], [2])
        obj_init_z = self.obj_init_options.get("init_z", self.scene_table_height)
        obj_init_z = obj_init_z + 0.5  # let object fall onto the table
        obj_init_rot_quat = self.obj_init_options.get("init_rot_quat", [1, 0, 0, 0])
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

        # Record the object height after it settles
        self.obj_height_after_settle = self.obj.pose.p[2]

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        self.set_episode_rng(seed)

        # Set objects
        self.obj_init_options = options.get("obj_init_options", {})
        model_scale = options.get("model_scale", None)
        model_id = options.get("model_id", None)
        reconfigure = options.get("reconfigure", False)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure

        obs, info = super().reset(seed=self._episode_seed, options=options)
        return obs, info

    def _additional_prepackaged_config_reset(self, options):
        # Use prepackaged evaluation configs under visual matching setup
        overlay_ids = ["a0", "b0", "c0"]
        rgb_overlay_paths = [
            str(ASSET_DIR / f"real_inpainting/open_drawer_{i}.png") for i in overlay_ids
        ]
        robot_init_xs = [0.644, 0.652, 0.665]
        robot_init_ys = [-0.179, 0.009, 0.224]
        robot_init_rotzs = [-0.03, 0, 0]
        idx_chosen = self._episode_rng.choice(len(overlay_ids))

        options["robot_init_options"] = {
            "init_xy": [robot_init_xs[idx_chosen], robot_init_ys[idx_chosen]],
            "init_rot_quat": (
                sapien.Pose(q=euler2quat(0, 0, robot_init_rotzs[idx_chosen]))
                * sapien.Pose(q=[0, 0, 0, 1])
            ).q,
        }
        self.rgb_overlay_img = (
            cv2.cvtColor(cv2.imread(rgb_overlay_paths[idx_chosen]), cv2.COLOR_BGR2RGB)
            / 255
        )
        new_urdf_version = self._episode_rng.choice(
            [
                "",
                "recolor_tabletop_visual_matching_1",
                "recolor_tabletop_visual_matching_2",
                "recolor_cabinet_visual_matching_1",
            ]
        )
        if new_urdf_version != self.urdf_version:
            self.urdf_version = new_urdf_version
            self._configure_agent()
            return True
        return False

    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(
            is_grasped=False,
            lifted_10cm=False
        )

    @property
    def obj_pose(self):
        """Get the center of mass (COM) pose of the object."""
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
        # Check if object is grasped
        is_grasped = self.agent.check_grasp(self.obj, max_angle=80)
        self.episode_stats["is_grasped"] = self.episode_stats["is_grasped"] or is_grasped

        # Check if object is lifted 10cm above its settled position
        current_height = self.obj.pose.p[2]
        height_diff = current_height - self.obj_height_after_settle
        lifted_10cm = height_diff >= 0.10  # 10cm = 0.10m
        self.episode_stats["lifted_10cm"] = self.episode_stats["lifted_10cm"] or lifted_10cm

        # Success is when object is lifted 10cm
        success = lifted_10cm

        return dict(
            success=success,
            is_grasped=is_grasped,
            lifted_10cm=lifted_10cm,
            height_diff=height_diff,
            episode_stats=self.episode_stats
        )

    def get_language_instruction(self, **kwargs):
        model_name = self._get_instruction_obj_name(self.model_id)
        return f"pick {model_name}"


@register_env("PickObjectOnClosedDrawerInScene-v0", max_episode_steps=200)
class PickObjectOnClosedDrawerInSceneEnv(
    PickObjectOnDrawerInSceneEnv, CustomOtherObjectsInSceneEnv
):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v1.json"
    drawer_ids = ["top", "middle", "bottom"]


# Specific implementations for different drawers
@register_env("PickObjectOnClosedTopDrawerInScene-v0", max_episode_steps=200)
class PickObjectOnClosedTopDrawerInSceneEnv(PickObjectOnClosedDrawerInSceneEnv):
    drawer_ids = ["top"]


@register_env("PickObjectOnClosedMiddleDrawerInScene-v0", max_episode_steps=200)
class PickObjectOnClosedMiddleDrawerInSceneEnv(PickObjectOnClosedDrawerInSceneEnv):
    drawer_ids = ["middle"]


@register_env("PickObjectOnClosedBottomDrawerInScene-v0", max_episode_steps=200)
class PickObjectOnClosedBottomDrawerInSceneEnv(PickObjectOnClosedDrawerInSceneEnv):
    drawer_ids = ["bottom"]


# Specific object implementations
@register_env("PickCokeCanOnClosedDrawerInScene-v0", max_episode_steps=200)
class PickCokeCanOnClosedDrawerInSceneEnv(PickObjectOnClosedDrawerInSceneEnv):
    drawer_ids = ["top", "middle", "bottom"]

    def __init__(self, **kwargs):
    # Initialize attributes before calling super().__init__
    # Important: pass model_ids into super().__init__ so the very first
    # auto-reset during BaseEnv.__init__ uses the coke can, not a random model.
    self.distractor_ids = ["sponge", "apple"]
    self.distractor_objs = []

    super().__init__(model_ids=["opened_coke_can"], **kwargs)

    def _load_actors(self):
        super()._load_actors()
        # Load distractor objects
        self._load_distractor_objects()

    def _load_distractor_objects(self):
        """Load distractor objects (sponge and apple)"""
        # Clear existing distractor objects
        self.distractor_objs = []

        # Check if distractor_ids is defined (only for coke can env)
        if not hasattr(self, 'distractor_ids') or not self.distractor_ids:
            return

        for distractor_id in self.distractor_ids:
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
            distractor_obj.set_damping(0.1, 0.1)
            self.distractor_objs.append(distractor_obj)

    def _get_random_drop_locations(self, num_objects):
                """Generate random drop locations that are on the table with spacing.

                Notes:
                - Bounds chosen to match the single-object placement region used in
                    PickObjectOnDrawerInSceneEnv (keeps items on tabletop in the default scene).
                - Enforce a minimum spacing to reduce immediate collisions that can
                    push objects off the surface.
                """
                # Conservative tabletop bounds near the drawer area (w.r.t. world frame)
                # Keep consistent with single-object defaults: x in [-0.10,-0.05], y in [0.00,0.10]
                # Slightly expanded but still safely on-table
                x_min, x_max = -0.12, -0.03
                y_min, y_max = -0.02, 0.12

                min_dist = 0.06  # 6cm separation to reduce push-offs
                max_tries = 100
                drop_locations = []

                def far_enough(p):
                        for q in drop_locations:
                                if np.linalg.norm(np.array(p) - np.array(q)) < min_dist:
                                        return False
                        return True

                for _ in range(num_objects):
                        # rejection sampling with spacing
                        for _try in range(max_tries):
                                x = float(self._episode_rng.uniform(x_min, x_max))
                                y = float(self._episode_rng.uniform(y_min, y_max))
                                cand = [x, y]
                                if far_enough(cand):
                                        drop_locations.append(cand)
                                        break
                        else:
                                # Fallback without spacing if too crowded (should be rare for 2-3 objs)
                                x = float(self._episode_rng.uniform(x_min, x_max))
                                y = float(self._episode_rng.uniform(y_min, y_max))
                                drop_locations.append([x, y])

                return drop_locations

    def _initialize_actors(self):
        # Keep the drawer closed (no opening needed for pick task)
        self.art_obj.set_qpos([0.0] * self.art_obj.dof)

        # Generate random drop locations for all objects (target + distractors)
        all_objects = [self.obj] + (self.distractor_objs if hasattr(self, 'distractor_objs') and self.distractor_objs else [])
        drop_locations = self._get_random_drop_locations(len(all_objects))

        # Randomly shuffle objects to assign them to random drop locations
        shuffled_objects = all_objects.copy()
        self._episode_rng.shuffle(shuffled_objects)

        # Drop height
        drop_z = self.scene_table_height + 0.5

        # Move the robot far away to avoid collision
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # Place all objects at their assigned drop locations
        for obj, drop_xy in zip(shuffled_objects, drop_locations):
            p = np.hstack([drop_xy, drop_z])

            # Set appropriate orientation based on object type
            if obj == self.obj:  # Target coke can - upright
                q = euler2quat(np.pi / 2, 0, 0)
            else:  # Distractor objects - default orientation
                q = [1, 0, 0, 0]

            # Apply random Z rotation for variety
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = qmult(euler2quat(0, 0, ori), q)

            obj.set_pose(sapien.Pose(p, q))

        # Lock objects to prevent rolling while settling
        for obj in all_objects:
            obj.lock_motion(0, 0, 0, 1, 1, 0)

        self._settle(0.5)

        # Unlock objects
        for obj in all_objects:
            obj.lock_motion(0, 0, 0, 0, 0, 0)
            obj.set_pose(obj.pose)
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))

        self._settle(0.5)

        # Additional settling if objects are still moving
        total_vel = sum(np.linalg.norm(obj.velocity) for obj in all_objects)
        if total_vel > 1e-3:
            self._settle(1.5)

        # Record the target object height after it settles
        self.obj_height_after_settle = self.obj.pose.p[2]

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=vectorize_pose(self.obj_pose),
                tcp_to_obj_pos=self.obj_pose.p - self.tcp.pose.p,
            )
            # Add distractor object poses
            for i, distractor_obj in enumerate(self.distractor_objs):
                distractor_pose = distractor_obj.pose.transform(distractor_obj.cmass_local_pose)
                obs[f"distractor_{i}_pose"] = vectorize_pose(distractor_pose)
                obs[f"tcp_to_distractor_{i}_pos"] = distractor_pose.p - self.tcp.pose.p
        return obs

    def reset(self, seed=None, options=None):
    # Keep distractors across resets; they will be repositioned in _initialize_actors.
    # Reconfigure will recreate the scene and actors as needed.
    return super().reset(seed=seed, options=options)


@register_env("PickSpongeOnClosedDrawerInScene-v0", max_episode_steps=200)
class PickSpongeOnClosedDrawerInSceneEnv(PickObjectOnClosedDrawerInSceneEnv):
    drawer_ids = ["top", "middle", "bottom"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_ids = ["sponge"]


@register_env("PickAppleOnClosedDrawerInScene-v0", max_episode_steps=200)
class PickAppleOnClosedDrawerInSceneEnv(PickObjectOnClosedDrawerInSceneEnv):
    drawer_ids = ["top", "middle", "bottom"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_ids = ["apple"]