"""
Schmalz ECBPi electric vacuum gripper.

Placeholder model: geometry is approximated with primitives (see
ecbpi_gripper.xml) since no vendor CAD was available. The gripper has a single
DOF standing in for the suction cup's bellows compliance (engaged vs idle) --
it has no functional grasp physics, matching how this codebase only ever
renders/masks the gripper rather than simulating an actual grip.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class ECBPiGripper(GripperModel):
    """
    1-DoF vacuum gripper (Schmalz ECBPi).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/ecbpi_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0])  # cup idle/relaxed

    @property
    def speed(self):
        return 0.01
