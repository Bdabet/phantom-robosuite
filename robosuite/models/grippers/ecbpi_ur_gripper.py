"""
Schmalz ECBPi electric vacuum gripper -- full "Robots set ECBPi UR" variant.

Unlike ECBPiGripper (ecbpi_gripper.py), which only has the bare pump housing
approximated with primitives, this uses the real flange+pump+valve+suction-cup
geometry converted from Schmalz's "ROB-SET ECBPi UR" CAD export. See
ecbpi_ur_gripper.xml for the mesh/joint layout.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class ECBPiURGripper(GripperModel):
    """
    1-DoF vacuum gripper (Schmalz ECBPi, full "Robots set ECBPi UR" geometry).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/ecbpi_ur_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0])  # cup idle/relaxed

    @property
    def speed(self):
        return 0.01
