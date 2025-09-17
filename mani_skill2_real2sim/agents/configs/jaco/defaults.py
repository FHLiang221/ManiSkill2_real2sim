from copy import deepcopy

import numpy as np

from mani_skill2_real2sim.agents.controllers import *
from mani_skill2_real2sim.sensors.camera import CameraConfig
from mani_skill2_real2sim.utils.sapien_utils import look_at


class JacoDefaultConfig:
    def __init__(self) -> None:
        self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/ada_description/robots_urdf/ada.urdf"

        # finger_min_patch_radius = 0.01  # used to calculate torsional friction
        # self.urdf_config = dict(
        #     _materials=dict(
        #         gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        #     ),
        #     link=dict(
        #         left_finger_link=dict(
        #             material="gripper",
        #             patch_radius=finger_min_patch_radius,
        #             min_patch_radius=finger_min_patch_radius,
        #         ),
        #         right_finger_link=dict(
        #             material="gripper",
        #             patch_radius=finger_min_patch_radius,
        #             min_patch_radius=finger_min_patch_radius,
        #         ),
        #     ),
        # )
        self.urdf_config = dict()

        self.arm_joint_names = [
            "j2n6s200_joint_1",
            "j2n6s200_joint_2",
            "j2n6s200_joint_3",
            "j2n6s200_joint_4",
            "j2n6s200_joint_5",
            "j2n6s200_joint_6",
        ]
        self.gripper_joint_names = [
            "j2n6s200_joint_finger_1",
            "j2n6s200_joint_finger_2",
        ]

        # self.gripper_force_limit = [20] * len(self.gripper_joint_names)
        # self.gripper_vel_limit = [1.0] * len(self.gripper_joint_names)

        # # Force control
        # self.arm_stiffness = [1e9] * len(self.arm_joint_names)
        # self.arm_damping = [1e3] * len(self.arm_joint_names)

        # self.gripper_stiffness = [1e9] * len(self.gripper_joint_names)
        # self.gripper_damping = [1e3] * len(self.gripper_joint_names)

        # self.arm_stiffness = [
        #     1169.7891719504198,
        #     730.0,
        #     808.4601346394447,
        #     1229.1299089624076,
        #     1272.2760456418862,
        #     1056.3326605132252,
        # ]
        # self.arm_damping = [
        #     330.0,
        #     180.0,
        #     152.12036565582588,
        #     309.6215302722146,
        #     201.04998711007383,
        #     269.51458932695414,
        # ]

        self.arm_stiffness = 1000
        self.arm_damping = 100

        self.arm_force_limit = [2400, 4800, 2400, 1200, 1200, 1200]
        self.arm_friction = 0.0
        self.arm_vel_limit = 0.1

        self.gripper_stiffness = 200
        self.gripper_damping = 100
        # self.gripper_stiffness = 1000
        # self.gripper_damping = 200
        # self.gripper_pid_stiffness = 1000
        # self.gripper_pid_damping = 200
        # self.gripper_pid_integral = 300
        self.gripper_force_limit = 120
        self.gripper_vel_limit = 0.1

        self.ee_link_name = "j2n6s200_end_effector"

    @property
    def controllers(self):
        _C = {}

        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_common_args = [
            self.arm_joint_names,
            -1.0,  # dummy limit, which is unused since normalize_action=False
            1.0,
            np.pi / 2,
            self.arm_stiffness,
            self.arm_damping,
            # self.arm_force_limit,
        ]
        arm_common_kwargs = dict(
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            normalize_action=False,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee", **arm_common_kwargs
        )
        arm_pd_ee_target_delta_pose = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee", use_target=True, **arm_common_kwargs
        )
        _C["arm"] = dict(
            arm_pd_ee_delta_pose=arm_pd_ee_delta_pose,
            arm_pd_ee_target_delta_pose=arm_pd_ee_target_delta_pose,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_common_args = [
            self.gripper_joint_names,
            0.1,
            1.6,
            self.gripper_stiffness,
            self.gripper_damping,
            # self.gripper_force_limit,
        ]
        gripper_common_kwargs = dict(
            normalize_action=True,
            drive_mode="force",
        )

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            *gripper_common_args, **gripper_common_kwargs
        )
        gripper_pd_joint_target_pos = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            use_target=True,
            clip_target=True,
            **gripper_common_kwargs,
        )
        gripper_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            *gripper_common_args, use_delta=True, **gripper_common_kwargs
        )
        gripper_pd_joint_target_delta_pos = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            use_delta=True,
            use_target=True,
            clip_target=True,
            **gripper_common_kwargs,
        )
        _C["gripper"] = dict(
            gripper_pd_joint_pos=gripper_pd_joint_pos,
            gripper_pd_joint_target_pos=gripper_pd_joint_target_pos,
            gripper_pd_joint_delta_pos=gripper_pd_joint_delta_pos,
            gripper_pd_joint_target_delta_pos=gripper_pd_joint_target_delta_pos,
        )

        controller_configs = {}
        for arm_controller_name in _C["arm"]:
            for gripper_controller_name in _C["gripper"]:
                c = {}
                c["arm"] = _C["arm"][arm_controller_name]
                c["gripper"] = _C["gripper"][gripper_controller_name]
                combined_name = (
                    arm_controller_name + "_" + gripper_controller_name
                )
                controller_configs[combined_name] = c

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        # Table width: about 36cm
        # Table height: 0.87; camera height: 0.65 above table
        # Arm root 0.75+0.37 from camera along x
        # Arm root 0.39 from camera along y
        camera_root = np.array([0.67, 0.21, 1.52])
        camera_aim = np.array([-0.32, 0.21, 1.07])

        return [
            CameraConfig(
                uid="3rd_view_camera",  # the camera used for real evaluation
                p=camera_root,
                q=look_at(eye=camera_root, target=camera_aim).q,
                width=640,
                height=480,
                # actor_uid is None since we are not mounting our camera
                fov=0.73,  # Realsense D435 FOV 69x42
            ),
        ]
