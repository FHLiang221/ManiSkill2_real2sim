import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2_real2sim.agents.base_agent import BaseAgent
from mani_skill2_real2sim.agents.configs.jaco import defaults
from mani_skill2_real2sim.utils.common import compute_angle_between
from mani_skill2_real2sim.utils.sapien_utils import (
    get_entity_by_name,
    get_pairwise_contact_impulse,
)


class Jaco(BaseAgent):
    _config: defaults.JacoDefaultConfig

    """
    Links: [Actor(name="world", id="2"), Actor(name="j2n6s200_link_base", id="3"), Actor(name="j2n6s200_link_1", id="4"), Actor(name="j2n6s200_link_2", id="5"), Actor(name="j2n6s200_link_3", id="6"), Actor(name="j2n6s200_link_4", id="7"), Actor(name="j2n6s200_link_5", id="8"), Actor(name="j2n6s200_link_6", id="9"), Actor(name="j2n6s200_end_effector", id="15"), Actor(name="j2n6s200_hand_tip", id="16"), Actor(name="j2n6s200_hand_base", id="10"), Actor(name="j2n6s200_link_finger_1", id="13"), Actor(name="j2n6s200_link_finger_tip_1", id="14"), Actor(name="j2n6s200_link_finger_2", id="11"), Actor(name="j2n6s200_link_finger_tip_2", id="12")]
    Active joints: ['j2n6s200_joint_1', 'j2n6s200_joint_2', 'j2n6s200_joint_3', 'j2n6s200_joint_4', 'j2n6s200_joint_5', 'j2n6s200_joint_6', 'j2n6s200_joint_finger_1', 'j2n6s200_joint_finger_2']

    Joint limits:
        [[      -inf        inf]
        [0.82030475 5.4628806 ]
        [0.33161256 5.951573  ]
        [      -inf        inf]
        [      -inf        inf]
        [      -inf        inf]
        [0.         2.        ]
        [0.         2.        ]]
    """

    @classmethod
    def get_default_config(cls):
        return defaults.JacoDefaultConfig()

    def __init__(
        self,
        scene,
        control_freq,
        control_mode=None,
        fix_root_link=True,
        config=None,
    ):
        if control_mode is None:  # if user did not specify a control_mode
            control_mode = (
                "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
            )
        super().__init__(
            scene,
            control_freq,
            control_mode=control_mode,
            fix_root_link=fix_root_link,
            config=config,
        )
        self.robot.set_qpos([-1.5, 2.93, 1, -2.09, 1.44, 1.32, 0.02, 0.02])


    def _after_init(self):
        super()._after_init()

        # ignore collision between j2n6s200_link_6 and j2n6s200_link_finger_1/2
        gripper_bar_link = get_entity_by_name(
            self.robot.get_links(), "j2n6s200_link_6"
        )
        left_finger_link = get_entity_by_name(
            self.robot.get_links(), "j2n6s200_link_finger_1"
        )
        right_finger_link = get_entity_by_name(
            self.robot.get_links(), "j2n6s200_link_finger_2"
        )
        for l in gripper_bar_link.get_collision_shapes():
            l.set_collision_groups(1, 1, 0b11, 0)
        for l in left_finger_link.get_collision_shapes():
            l.set_collision_groups(1, 1, 0b01, 0)
        for l in right_finger_link.get_collision_shapes():
            l.set_collision_groups(1, 1, 0b10, 0)

        self.base_link = [
            x for x in self.robot.get_links() if x.name == "j2n6s200_link_base"
        ][0]
        self.ee_link = [x for x in self.robot.get_links() if x.name == "j2n6s200_end_effector"][0]

        self.finger_right_joint = get_entity_by_name(
            self.robot.get_joints(), "j2n6s200_joint_finger_1"
        )
        self.finger_left_joint = get_entity_by_name(
            self.robot.get_joints(), "j2n6s200_joint_finger_2"
        )

        self.finger_right_link = get_entity_by_name(
            self.robot.get_links(), "j2n6s200_link_finger_1"
        )
        self.finger_left_link = get_entity_by_name(
            self.robot.get_links(), "j2n6s200_link_finger_2"
        )

    @property
    def gripper_closedness(self):
        finger_qpos = self.robot.get_qpos()[-2:]
        finger_qlim = self.robot.get_qlimits()[-2:]
        closedness_left = (finger_qlim[0, 1] - finger_qpos[0]) / (
            finger_qlim[0, 1] - finger_qlim[0, 0]
        )
        closedness_right = (finger_qlim[1, 1] - finger_qpos[1]) / (
            finger_qlim[1, 1] - finger_qlim[1, 0]
        )
        return np.maximum(np.mean([closedness_left, closedness_right]), 0.0)

    def get_fingers_info(self):
        finger_right_pos = self.finger_right_link.get_global_pose().p
        finger_left_pos = self.finger_left_link.get_global_pose().p

        finger_right_vel = self.finger_right_link.get_velocity()
        finger_left_vel = self.finger_left_link.get_velocity()

        return {
            "finger_right_pos": finger_right_pos,
            "finger_left_pos": finger_left_pos,
            "finger_right_vel": finger_right_vel,
            "finger_left_vel": finger_left_vel,
        }

    def check_grasp(
        self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=60
    ):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse_finger = get_pairwise_contact_impulse(
            contacts, self.finger_left_link, actor
        )
        rimpulse_finger = get_pairwise_contact_impulse(
            contacts, self.finger_right_link, actor
        )

        # direction to open the gripper
        ldirection_finger = (
            self.finger_left_link.pose.to_transformation_matrix()[:3, 1]
        )
        rdirection_finger = (
            self.finger_right_link.pose.to_transformation_matrix()[:3, 1]
        )

        # angle between impulse and open direction
        langle = compute_angle_between(ldirection_finger, limpulse_finger)
        rangle = compute_angle_between(-rdirection_finger, rimpulse_finger)

        lflag = (np.linalg.norm(limpulse_finger) >= min_impulse) and np.rad2deg(
            langle
        ) <= max_angle
        rflag = (np.linalg.norm(rimpulse_finger) >= min_impulse) and np.rad2deg(
            rangle
        ) <= max_angle

        return all([lflag, rflag])

    def check_contact_fingers(self, actor: sapien.ActorBase, min_impulse=1e-6):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse_finger = get_pairwise_contact_impulse(
            contacts, self.finger_left_link, actor
        )
        rimpulse_finger = get_pairwise_contact_impulse(
            contacts, self.finger_right_link, actor
        )

        return (
            np.linalg.norm(limpulse_finger) >= min_impulse,
            np.linalg.norm(rimpulse_finger) >= min_impulse,
        )

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(approaching, closing)
        T = np.eye(4)
        T[:3, :3] = np.stack([approaching, closing, ortho], axis=1)
        T[:3, 3] = center
        return Pose.from_transformation_matrix(T)

    @property
    def base_pose(self):
        return self.base_link.get_pose()

    @property
    def ee_pose(self):
        return self.ee_link.pose