from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import cv2
import sapien.core as sapien
from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
from transforms3d.euler import euler2quat

from .base_env import CustomOtherObjectsInSceneEnv, CustomSceneEnv


class MultiObjectOpenDrawerInSceneEnv(CustomOtherObjectsInSceneEnv): # CustomSceneEnv):
    # drawer_ids: List[str]
    model_ids = [
        "opened_coke_can",
        "apple"
        "sponge",
    ]
    task_info: List[Tuple[str, str]]

    def __init__(
        self,
        light_mode: Optional[str] = None,
        camera_mode: Optional[str] = None,
        station_name: float = "mk_station",
        cabinet_joint_friction: float = 0.05,
        prepackaged_config: bool = False,
        target_drawer=False,
        require_lifting_obj_for_success: bool = True,
        success_from_episode_stats: bool = True,
        use_distractors: bool = True,
        **kwargs,
    ):
        self.light_mode = light_mode
        self.camera_mode = camera_mode
        self.station_name = station_name
        self.cabinet_joint_friction = cabinet_joint_friction
        self.episode_stats = None
        
        self.drawer_id = None
        self.obj_id = None
        self.task = None

        self.prepackaged_config = prepackaged_config
        if self.prepackaged_config:
            # use prepackaged evaluation configs (visual matching)
            kwargs.update(self._setup_prepackaged_env_init_config())
            
        self.model_id = None
        self.obj = None
        self.distractor_objs = []
        
        self.obj_init_options = {  # obj initialization options, v
            "opened_coke_can": {
                "init_xy": [-0.05, 0.03],
            },
            "apple": {
                "init_xy": [-0.05, 0.15],
            },
            "sponge": {
                "init_xy": [-0.05, -0.15],
            }
        }
        self.distractor_obj_init_options = {
            
        }
        
        # grasp scenarios (fixed at upright)
        self.orientation = "upright"
        # if upright:
        #     self.orientation = "upright"
        # elif laid_vertically:
        #     self.orientation = "laid_vertically"
        # elif lr_switch:
        #     self.orientation = "lr_switch"
        # else:
        #     self.orientation = None
        self.orientations_dict = {
            "upright": euler2quat(np.pi / 2, 0, 0),
            "laid_vertically": euler2quat(0, 0, np.pi / 2),
            "lr_switch": euler2quat(0, 0, np.pi),
        }
        
        obj_positions = [
            
        ]
        
        self.require_lifting_obj_for_success = require_lifting_obj_for_success
        self.success_from_episode_stats = success_from_episode_stats
        self.consecutive_grasp = 0
        self.lifted_obj = False
        self.obj_height_after_settle = None
        self.episode_stats = None
        
        self.use_distractors = use_distractors

        super().__init__(**kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "mk_station_recolor"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True

        return ret

    def _load_models(
        self,
        model_ids = None
    ):
        """Load the 3 objects on top of the drawer
        """
        model_ids = model_ids or self.model_ids
        assert model_ids is not None
        # assert len(model_ids) == 3, "Only support loading 3 objects on top of the drawer"
        self.model_ids = [
            "opened_coke_can",
            "apple",
            "sponge",
        ]

        # load the main object as the object, and everything else as distractors
        model_to_load = self.task[2] if self.task[0] == "object" else None

        self.distractor_objs = []
        for model_id in self.model_ids:
            if model_to_load == model_id:
                self.model_id = model_id
                
                density = self.model_db[self.model_id].get("density", 1000)

                self.obj = self._build_actor_helper(
                    model_id,
                    self._scene,
                    # scale=self.model_scale,
                    density=density,
                    physical_material=self._scene.create_physical_material(
                        static_friction=self.obj_static_friction,
                        dynamic_friction=self.obj_dynamic_friction,
                        restitution=0.0,
                    ),
                    root_dir=self.asset_root,
                )
                self.obj.name = model_id
            else:
                if not self.use_distractors:
                    continue
                
                distractor_obj = self._build_actor_helper(
                    model_id,
                    self._scene,
                    # scale=distractor_model_scale,
                    density=self.model_db[model_id].get(
                        "density", 1000
                    ),
                    physical_material=self._scene.create_physical_material(
                        static_friction=self.obj_static_friction,
                        dynamic_friction=self.obj_dynamic_friction,
                        restitution=0.0,
                    ),
                    root_dir=self.asset_root,
                )
                distractor_obj.name = model_id
                self.distractor_objs.append(distractor_obj)
                
        # HACK: if the object doesn't exist, set it accordingly
        if self.obj is None:
            self.obj = self.distractor_objs[0]
            self.distractor_objs = self.distractor_objs[1:]
            
            
    # def _get_default_scene_config(self):
    #     scene_config = super()._get_default_scene_config()
    #     scene_config.enable_pcm = True
    #     return scene_config

    def _initialize_agent(self):
        init_qpos = np.array(
            [
                -0.2639457174606611,
                0.0831913360274175,
                0.5017611504652179,
                1.156859026208673,
                0.028583671314766423,
                1.592598203487462,
                -1.080652960128774,
                0,
                0,
                -0.00285961,
                0.7851361,
            ]
        )
        if self.camera_mode == "variant":
            init_qpos[-2] += -0.025
            init_qpos[-1] += 0.008
        self.robot_init_options.setdefault("qpos", init_qpos)
        super()._initialize_agent()

    def _setup_lighting(self):
        if self.light_mode != "simple":
            return self._setup_lighting_legacy()

        self._scene.set_ambient_light([1.0, 1.0, 1.0])
        angle = 75
        self._scene.add_directional_light(
            [-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))], [1.0, 1.0, 1.0]
        )

    def _setup_lighting_legacy(self):
        # self.enable_shadow = True
        # super()._setup_lighting()

        direction = [-0.2, 0, -1]
        if self.light_mode == "vertical":
            direction = [-0.1, 0, -1]

        color = [1, 1, 1]
        if self.light_mode == "darker":
            color = [0.5, 0.5, 0.5]
        elif self.light_mode == "brighter":
            color = [2, 2, 2]

        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        # Only the first of directional lights can have shadow
        self._scene.add_directional_light(
            direction, color, shadow=True, scale=5, shadow_map_size=2048
        )
        self._scene.add_directional_light([-1, 1, -0.05], [0.5] * 3)
        self._scene.add_directional_light([-1, -1, -0.05], [0.5] * 3)

    def _load_actors(self):
        self._load_arena_helper(add_collision=False)
        self._load_models()
        
    def _initialize_actors(self):
        obj_init_options = self.obj_init_options.get(self.model_id, {})
        
        # The object will fall from a certain initial height
        obj_init_xy = obj_init_options.get("init_xy", None)
        if obj_init_xy is None:
            obj_init_xy = self._episode_rng.uniform(
                [-0.08, -0.02], [-0.02, 0.08], [2]
            )

        obj_init_z = obj_init_options.get(
            "init_z", self.scene_table_height
        )
        obj_init_z = obj_init_z + 0.5  # let object fall onto the table
        obj_init_rot_quat = obj_init_options.get(
            "init_rot_quat", [1, 0, 0, 0]
        )
        p = np.hstack([obj_init_xy, obj_init_z])
        q = obj_init_rot_quat

        # if a main target object exists, then fix its pose first
        if self.obj is not None:
            # Rotate along z-axis
            if obj_init_options.get("init_rand_rot_z", False):
                ori = self._episode_rng.uniform(0, 2 * np.pi)
                q = qmult(euler2quat(0, 0, ori), q)

            # Rotate along a random axis by a small angle
            if (
                init_rand_axis_rot_range := obj_init_options.get(
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

        if len(self.distractor_objs) > 0:
            # Set distractor objects
            for distractor_obj in self.distractor_objs:
                distractor_obj_init_options = (
                    self.obj_init_options.get(
                        distractor_obj.name, {}
                    )
                )

                distractor_init_xy = distractor_obj_init_options.get(
                    "init_xy", None
                )

                if distractor_init_xy is None:
                    while True:
                        distractor_init_xy = (
                            obj_init_xy
                            + self._episode_rng.uniform(-0.05, 0.05, [2])
                        )  # hardcoded for now
                        distractor_init_xy = np.clip(
                            distractor_init_xy, [-0.08, -0.02], [-0.02, 0.08]
                        )
                        if (
                            np.linalg.norm(distractor_init_xy - obj_init_xy)
                            > 0.03
                        ):
                            break

                p = np.hstack(
                    [distractor_init_xy, obj_init_z]
                )  # let distractor fall from the same height as the main object
                distractor_init_rot_quat = distractor_obj_init_options.get(
                    "init_rot_quat", None
                )
                q = (
                    obj_init_rot_quat
                    if distractor_init_rot_quat is None
                    else distractor_init_rot_quat
                )

                distractor_obj.set_pose(sapien.Pose(p, q))
                distractor_obj.set_velocity(np.zeros(3))
                distractor_obj.set_angular_velocity(np.zeros(3))
                # Lock rotation around x and y
                distractor_obj.lock_motion(1, 1, 0, 1, 1, 0)

                # debug
                # sim_steps = int(self.sim_freq * 0.5)
                # for _ in range(sim_steps):
                #     print(distractor_obj.pose)
                #     while True:
                #         self.render_human()
                #         sapien_viewer = self.viewer
                #         if sapien_viewer.window.key_down("0"):
                #             break
                #     self._scene.step()

                # Let distractor objects fall onto the table
                self._settle(0.5)

            # Unlock motion
            for distractor_obj in self.distractor_objs:
                distractor_obj.lock_motion(0, 0, 0, 0, 0, 0)
                distractor_obj.set_pose(distractor_obj.pose)
                distractor_obj.set_velocity(np.zeros(3))
                distractor_obj.set_angular_velocity(np.zeros(3))
                self._settle(0.5)

            lin_vel, ang_vel = 0.0, 0.0
            for distractor_obj in self.distractor_objs:
                lin_vel += np.linalg.norm(distractor_obj.velocity)
                ang_vel += np.linalg.norm(distractor_obj.angular_velocity)
            if lin_vel > 1e-3 or ang_vel > 1e-2:
                self._settle(1.5)

    def _load_articulations(self):
        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.295, 0, 0.017], [1, 0, 0, 0]))
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        # only keep this logged for open drawer tasks?
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        if self.task[0] == "drawer":
            self.drawer_obj = get_entity_by_name(
                self.art_obj.get_links(), f"{self.drawer_id}_drawer"
            )
            self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

    def reset(self, seed=None, options=None):
            # # remove distractor objects
            # for distractor_obj in self.distractor_objs:
            #     self._scene.remove_actor(distractor_obj)
            # print("Remove all distractor objects")
            # self.distractor_objs = []

        if options is None:
            options = dict()
        options = options.copy()
        
        # self.obj_init_options = options.get("obj_init_options", self.obj_init_options)
        # self.distractor_obj_init_options = options.get(
        #     "distractor_obj_init_options", {}
        # )

        reconfigure = options.get("reconfigure", False)
        self.set_episode_rng(seed)
        
        self.task = self.task_info[0]
        if self.task[0] == "drawer":
            self.drawer_id = self.task[2]

        if self.prepackaged_config:
            _reconfigure = self._additional_prepackaged_config_reset(options)
            reconfigure = reconfigure or _reconfigure

        options["reconfigure"] = reconfigure

        self._initialize_episode_stats()

        obs, info = super().reset(seed=self._episode_seed, options=options) # articulations are loaded here

        if self.task[0] == "drawer":
            self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

        # keep top drawer closed (do nothing)

        # open the bottom drawer
        bottom_drawer_joint_idx = self.joint_names.index(f"bottom_drawer_joint")
        tmp = [0.0] * self.art_obj.dof
        tmp[bottom_drawer_joint_idx] = 0.2
        self.art_obj.set_qpos(tmp)


        # set the object            
        obj_init_options = options.get("obj_init_options", None)
        if obj_init_options is None:
            obj_init_options = dict()
        obj_init_options = (
            obj_init_options.copy()
        )  # avoid modifying the original options

        orientation = None
        if obj_init_options.get("init_rot_quat", None) is None:
            if obj_init_options.get("orientation", None) is not None:
                orientation = obj_init_options["orientation"]
            else:
                orientation = self.orientation

            if orientation is not None:
                try:
                    obj_init_options["init_rot_quat"] = self.orientations_dict[
                        orientation
                    ]
                except KeyError as e:
                    if "standing" in orientation:
                        obj_init_options[
                            "init_rot_quat"
                        ] = self.orientations_dict["upright"]
                    elif "horizontal" in orientation:
                        obj_init_options[
                            "init_rot_quat"
                        ] = self.orientations_dict["lr_switch"]
                    else:
                        raise e
            else:
                orientation = self._episode_rng.choice(
                    list(self.orientations_dict.keys())
                )
                obj_init_options["init_rot_quat"] = self.orientations_dict[
                    orientation
                ]

        obs = self.get_obs()

        if self.task[0] == "drawer":
            info.update({
                "drawer_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.drawer_obj.pose,
                "cabinet_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.art_obj.pose,
            })
        elif self.task[0] == "object":
            info.update({"orientation": orientation})

        info.update(
            {
                "station_name": self.station_name,
                "light_mode": self.light_mode,
            }
        )
        return obs, info

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged evaluation configs under visual matching setup
        overlay_ids = ["a0", "a1", "a2", "b0", "b1", "b2", "c0", "c1", "c2"]
        rgb_overlay_paths = [
            str(ASSET_DIR / f"real_inpainting/open_drawer_{i}.png") for i in overlay_ids
        ]
        robot_init_xs = [0.644, 0.652, 0.665, 0.620, 0.630, 0.640, 0.650, 0.660, 0.670]
        robot_init_ys = [
            -0.179,
            -0.182,
            -0.203,
            0.009,
            0.009,
            0.035,
            0.224,
            0.222,
            0.222,
        ]
        robot_init_rotzs = [-0.03, -0.02, -0.06, 0, 0, 0, 0, -0.025, -0.025]
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
            n_lift_significant=0,
            consec_grasp=False,
            grasped=False,
            qpos=0.0
        )

    def evaluate(self, **kwargs):    
        if self.task[0] == "object":
            # evaluate the success of the task

            is_grasped = self.agent.check_grasp(self.obj, max_angle=80)
            if is_grasped:
                self.consecutive_grasp += 1
            else:
                self.consecutive_grasp = 0
                self.lifted_obj = False

            contacts = self._scene.get_contacts()
            flag = True
            robot_link_names = [x.name for x in self.agent.robot.get_links()]
            for contact in contacts:
                actor_0, actor_1 = contact.actor0, contact.actor1
                other_obj_contact_actor_name = None
                if actor_0.name == self.obj.name:
                    other_obj_contact_actor_name = actor_1.name
                elif actor_1.name == self.obj.name:
                    other_obj_contact_actor_name = actor_0.name
                if other_obj_contact_actor_name is not None:
                    # the object is in contact with an actor
                    contact_impulse = np.sum(
                        [point.impulse for point in contact.points], axis=0
                    )
                    if (other_obj_contact_actor_name not in robot_link_names) and (
                        np.linalg.norm(contact_impulse) > 1e-6
                    ):
                        # the object has contact with an actor other than the robot link, so the object is not yet lifted up
                        # print(other_obj_contact_actor_name, np.linalg.norm(contact_impulse))
                        flag = False
                        break

            consecutive_grasp = self.consecutive_grasp >= 5
            diff_obj_height = self.obj.pose.p[2] - self.obj_height_after_settle
            self.lifted_obj = self.lifted_obj or (flag and (diff_obj_height > 0.05))
            lifted_object_significantly = self.lifted_obj and (
                diff_obj_height > 0.05
            )

            if self.require_lifting_obj_for_success:
                success = self.lifted_obj
            else:
                success = consecutive_grasp

            self.episode_stats["n_lift_significant"] += int(
                lifted_object_significantly
            )
            self.episode_stats["consec_grasp"] = (
                self.episode_stats["consec_grasp"] or consecutive_grasp
            )
            self.episode_stats["grasped"] = (
                self.episode_stats["grasped"] or is_grasped
            )
            if self.success_from_episode_stats:
                # During evaluation, if policy puts down coke can in the end but has lifted it significantly before, it is still a success
                # However, if you want to perform RL training on this environment, make sure to turn off this option
                success = success or (self.episode_stats["n_lift_significant"] >= 5)

            res = dict(
                is_grasped=is_grasped,
                consecutive_grasp=consecutive_grasp,
                lifted_object=self.lifted_obj,
                lifted_object_significantly=lifted_object_significantly,
                success=success,
                episode_stats=self.episode_stats,
            )
            return res
        
        elif self.task[0] == "drawer":
            # get drawer joint qpos
            qpos = self.art_obj.get_qpos()[self.joint_idx]
            self.episode_stats["qpos"] = "{:.3f}".format(qpos)

            # determine success accordingly
            success = qpos >= 0.15 if self.task[1] == "open" else qpos <= 0.05
            return dict(success=success, qpos=qpos, episode_stats=self.episode_stats)
        else:
            raise NotImplementedError

    def get_language_instruction(self, **kwargs):
        if self.task[0] == "drawer":
            return f"{self.task[1]} {self.drawer_id} drawer"
        elif self.task[0] == "object":
            obj_name = self._get_instruction_obj_name(self.obj.name)
            task_description = f"pick {obj_name}"
            return task_description


@register_env("CustomMultiObjectOpenDrawerInSceneEnv-v0", max_episode_steps=113)
class CustomMultiObjectOpenDrawerInSceneEnv(MultiObjectOpenDrawerInSceneEnv):
    model_ids = [
        "opened_coke_can",
        "apple"
        "sponge",
    ]


# ---------------------------------------------------------------------------- #
# Pick place drawer tasks
# ---------------------------------------------------------------------------- #
@register_env("MultiObjectGraspSingleOpenedCokeCanInScene-v0", max_episode_steps=10000)
class PickCokeCanCustomInSceneEnv(CustomMultiObjectOpenDrawerInSceneEnv):
    # drawer_ids = ["top", "middle", "bottom"]
    task_info = [("object", "grasp", "opened_coke_can")]

@register_env("MultiObjectGraspSingleAppleInScene-v0", max_episode_steps=10000)
class PickCokeCanCustomInSceneEnv(CustomMultiObjectOpenDrawerInSceneEnv):
    # drawer_ids = ["top", "middle", "bottom"]
    task_info = [("object", "grasp", "apple")]

@register_env("MultiObjectGraspSingleSpongeInScene-v0", max_episode_steps=10000)
class PickCokeCanCustomInSceneEnv(CustomMultiObjectOpenDrawerInSceneEnv):
    # drawer_ids = ["top", "middle", "bottom"]
    task_info = [("object", "grasp", "sponge")]


# ---------------------------------------------------------------------------- #
# Open drawer tasks
# ---------------------------------------------------------------------------- #
@register_env("MultiObjectOpenTopDrawerCustomInScene-v0", max_episode_steps=10000)
class OpenTopDrawerCustomInSceneEnv(CustomMultiObjectOpenDrawerInSceneEnv):
    task_info = [("drawer", "open", "top")]
    # drawer_ids = ["top"]


# ---------------------------------------------------------------------------- #
# Close drawer tasks
# ---------------------------------------------------------------------------- #
@register_env("MultiObjectCloseBottomDrawerCustomInScene-v0", max_episode_steps=10000)
class CloseBottomDrawerCustomInSceneEnv(CustomMultiObjectOpenDrawerInSceneEnv):
    task_info = [("drawer", "close", "bottom")]
    # drawer_ids = ["bottom"]