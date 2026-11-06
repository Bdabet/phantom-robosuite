import xml.etree.ElementTree as ET

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import PhantomTableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, find_elements, new_element


class Phantom(SingleArmEnv):
    """
    This class corresponds to the stacking task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="frontview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        object_placements=None,
        direct_gripper_control=False,
        camera_pos=None,
        camera_quat_wxyz=None,
        camera_fov=None,
        camera_sensorsize=None,
        camera_principalpixel=None,
        camera_focalpixel=None,
        wrist_camera_pos=None,
        wrist_camera_quat_wxyz=None,
        wrist_camera_sensorsize=None,
        wrist_camera_principalpixel=None,
        wrist_camera_focalpixel=None,
        show_wrist_camera=False,
        wrist_camera_box_size=(0.09, 0.025, 0.025),
        grasp_object=None,
    ):

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.object_placements = object_placements
        self.camera_pos = camera_pos
        self.camera_quat_wxyz = camera_quat_wxyz
        self.camera_fov = camera_fov
        self.camera_sensorsize = camera_sensorsize
        self.camera_principalpixel = camera_principalpixel
        self.camera_focalpixel = camera_focalpixel
        self.wrist_camera_pos = wrist_camera_pos
        self.wrist_camera_quat_wxyz = wrist_camera_quat_wxyz
        self.wrist_camera_sensorsize = wrist_camera_sensorsize
        self.wrist_camera_principalpixel = wrist_camera_principalpixel
        self.wrist_camera_focalpixel = wrist_camera_focalpixel
        self.show_wrist_camera = show_wrist_camera
        self.wrist_camera_box_size = np.array(wrist_camera_box_size, dtype=float)

        # Optional object the gripper can "pick up". It has no collision geometry: it is
        # held by weld constraints that set_suction() swaps between the world and the
        # gripper. dict(type="box"|"cylinder", size=<half extents>, rgba=<4 floats>).
        self.grasp_object_cfg = dict(grasp_object) if grasp_object else None
        self.grasp_object = None

        # pdb.set_trace()

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            direct_gripper_control=direct_gripper_control,
        )

    def reset(self, object_placements=None):
        self.object_placements = object_placements
        return super().reset()

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = PhantomTableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.grasp_object = self._build_grasp_object()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.grasp_object] if self.grasp_object is not None else None,
        )
        if self.grasp_object is not None:
            self._add_grasp_welds()

        # Modify default frontview camera
        if self.camera_pos is not None:
            robot_base_pos = np.array([-0.56, 0, 0.912])
            mujoco_arena.set_camera(
                camera_name="frontview",
                pos=self.camera_pos + robot_base_pos,
                quat=self.camera_quat_wxyz,
                camera_attribs={"sensorsize": np.array2string(self.camera_sensorsize)[1:-1], 
                                "resolution": f"{self.camera_widths[0]} {self.camera_heights[0]}",
                                "principalpixel": np.array2string(self.camera_principalpixel)[1:-1],
                                "focalpixel": np.array2string(self.camera_focalpixel)[1:-1],}
            )

        # Attach an optional wrist camera to the eef ("flange") body, so its pos/quat
        # are local to the flange frame rather than world coordinates.
        if self.wrist_camera_pos is not None:
            eef_body = find_elements(
                root=self.robots[0].robot_model.root,
                tags="body",
                attribs={"name": self.robots[0].robot_model.eef_name},
                return_first=True,
            )
            camera_attribs = {
                "sensorsize": np.array2string(self.wrist_camera_sensorsize)[1:-1],
                "resolution": f"{self.camera_widths[0]} {self.camera_heights[0]}",
                "principalpixel": np.array2string(self.wrist_camera_principalpixel)[1:-1],
                "focalpixel": np.array2string(self.wrist_camera_focalpixel)[1:-1],
                "pos": array_to_string(self.wrist_camera_pos),
                "quat": array_to_string(self.wrist_camera_quat_wxyz),
            }
            wrist_camera = find_elements(root=eef_body, tags="camera", attribs={"name": "wristview"}, return_first=True)
            if wrist_camera is None:
                eef_body.append(new_element(tag="camera", name="wristview", **camera_attribs))
            else:
                for attrib, value in camera_attribs.items():
                    wrist_camera.set(attrib, value)

            # Optionally add a box standing in for the camera housing, so the wrist camera
            # is visible from other cameras (e.g. frontview). Box size is full (w, h, depth)
            # in the camera frame; it sits just behind the lens (MuJoCo cameras look along
            # -z) so it doesn't occlude the wrist view itself.
            if self.show_wrist_camera:
                robot_model = self.robots[0].robot_model
                geom_name = "wristview_box_vis"
                half_size = self.wrist_camera_box_size / 2
                cam_rot = T.quat2mat(T.convert_quat(np.array(self.wrist_camera_quat_wxyz), to="xyzw"))
                box_pos = np.array(self.wrist_camera_pos) + cam_rot @ np.array([0, 0, half_size[2]])
                box_attribs = {
                    "type": "box",
                    "size": array_to_string(half_size),
                    "pos": array_to_string(box_pos),
                    "quat": array_to_string(self.wrist_camera_quat_wxyz),
                    "rgba": "0.1 0.1 0.1 1",
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                    "mass": "0",
                }
                box = find_elements(root=eef_body, tags="geom",
                                    attribs={"name": robot_model.naming_prefix + geom_name}, return_first=True)
                if box is None:
                    eef_body.append(new_element(tag="geom", name=robot_model.naming_prefix + geom_name, **box_attribs))
                else:
                    for attrib, value in box_attribs.items():
                        box.set(attrib, value)
                # Register with the robot model so instance segmentation counts it as robot
                # (robot_mask); correct_naming() re-adds the prefix.
                if geom_name not in robot_model._visual_geoms:
                    robot_model._visual_geoms.append(geom_name)

    # ---- optional grasp object: kinematic "suction" via weld constraints ----------------
    # The gripper models here have no collision geometry, so there is no contact grasp.
    # Instead the object is always welded to something: the world while it rests, the
    # gripper's flange body while suction is on. Swapping which weld is active (rather
    # than writing qpos after env.step) keeps the attachment exact inside every physics
    # substep, so the frame rendered by env.step already shows the object where it is.
    WORLD_WELD = "grasp_world_weld"
    GRIP_WELD = "grasp_grip_weld"

    def _build_grasp_object(self):
        cfg = self.grasp_object_cfg
        if cfg is None:
            return None
        kind = cfg.get("type", "box")
        kwargs = dict(
            name="grasp_object",
            size=cfg.get("size"),
            rgba=cfg.get("rgba", (0.8, 0.3, 0.2, 1.0)),
            density=cfg.get("density", 100.0),
            obj_type="visual",  # no contact geoms: nothing may collide with it
        )
        if kind == "box":
            return BoxObject(**kwargs)
        if kind == "cylinder":
            return CylinderObject(**kwargs)
        raise ValueError(f"grasp_object type must be 'box' or 'cylinder', got {kind!r}")

    def _add_grasp_welds(self):
        obj = self.grasp_object
        body = find_elements(root=self.model.worldbody, tags="body",
                             attribs={"name": obj.root_body}, return_first=True)
        # Cancel gravity on the object so the (soft) welds do not sag under its weight.
        body.set("gravcomp", "1")
        # Visual-only geoms carry no mass and a free body needs some, so give it a token one.
        body.append(ET.Element("inertial", pos="0 0 0", mass="0.05", diaginertia="1e-4 1e-4 1e-4"))
        for name, body1, active in ((self.WORLD_WELD, "world", "true"), (self.GRIP_WELD, self._grasp_body(), "false")):
            self.model.equality.append(
                new_element(tag="weld", name=name, body1=body1, body2=obj.root_body, active=active))

    def _grasp_body(self):
        """
        Body the object is welded to while suction is on: the suction cup's own body if
        the gripper has one (ECBPiUR's cup_assembly, which the bellows pull up on engage,
        so the object rises with it), else the robot's flange body.
        """
        cup = [b for b in self.robots[0].gripper.bodies if b.endswith("cup_assembly")]
        return cup[0] if cup else self.robots[0].robot_model.eef_name

    def _eq_id(self, name):
        import mujoco
        return mujoco.mj_name2id(self.sim.model._model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def _set_weld_active(self, name, active):
        eq = self._eq_id(name)
        data = self.sim.data._data
        # mjData.eq_active exists from MuJoCo 3; earlier versions keep it on the model.
        if hasattr(data, "eq_active"):
            data.eq_active[eq] = int(active)
        else:
            self.sim.model._model.eq_active[eq] = int(active)

    def _set_weld_relpose(self, name, body1, body2):
        """Pin weld `name` to the current pose of body2 expressed in body1's frame."""
        import mujoco
        model, data = self.sim.model, self.sim.data
        b1, b2 = model.body_name2id(body1), model.body_name2id(body2)
        p1, q1 = data.body_xpos[b1].copy(), data.body_xquat[b1].copy()
        p2, q2 = data.body_xpos[b2].copy(), data.body_xquat[b2].copy()
        q1_inv = np.zeros(4)
        mujoco.mju_negQuat(q1_inv, q1)
        rel_pos, rel_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_rotVecQuat(rel_pos, p2 - p1, q1_inv)
        mujoco.mju_mulQuat(rel_quat, q1_inv, q2)
        eq = self._eq_id(name)
        # weld eq_data layout: anchor(3) | relative position(3) | relative quat wxyz(4) | torquescale
        model._model.eq_data[eq, 0:3] = 0.0
        model._model.eq_data[eq, 3:6] = rel_pos
        model._model.eq_data[eq, 6:10] = rel_quat

    def _zero_object_velocity(self):
        start, end = self.sim.model.get_joint_qvel_addr(self.grasp_object.joints[0])
        self.sim.data.qvel[start:end] = 0.0

    def set_object_pose(self, pos, quat_wxyz):
        """Teleport the grasp object (world frame) and leave it resting, suction off."""
        assert self.grasp_object is not None, "env was created without grasp_object"
        obj = self.grasp_object
        self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([pos, quat_wxyz]))
        self._zero_object_velocity()
        self.sim.forward()
        self._set_weld_relpose(self.WORLD_WELD, "world", obj.root_body)
        self._set_weld_active(self.GRIP_WELD, False)
        self._set_weld_active(self.WORLD_WELD, True)
        self.suction_on = False

    def set_suction(self, on):
        """
        Suction on: weld the object to the gripper at its current relative pose.
        Suction off: weld it to the world where it currently is. No-op if unchanged.
        """
        assert self.grasp_object is not None, "env was created without grasp_object"
        if bool(on) == getattr(self, "suction_on", False):
            return
        obj = self.grasp_object
        self.sim.forward()
        if on:
            self._set_weld_relpose(self.GRIP_WELD, self._grasp_body(), obj.root_body)
            self._set_weld_active(self.WORLD_WELD, False)
            self._set_weld_active(self.GRIP_WELD, True)
        else:
            self._zero_object_velocity()
            self._set_weld_relpose(self.WORLD_WELD, "world", obj.root_body)
            self._set_weld_active(self.GRIP_WELD, False)
            self._set_weld_active(self.WORLD_WELD, True)
        self.suction_on = bool(on)

    def get_object_pose(self):
        """(pos (3,), quat_xyzw (4,)) of the grasp object in the world frame."""
        bid = self.sim.model.body_name2id(self.grasp_object.root_body)
        return (self.sim.data.body_xpos[bid].copy(),
                T.convert_quat(self.sim.data.body_xquat[bid].copy(), to="xyzw"))

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

   
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        # A hard reset rebuilds the sim with the welds in their XML state (world weld on).
        self.suction_on = False

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # # Color the gripper visualization site according to its distance to the cube
        # if vis_settings["grippers"]:
        #     self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)

    def reward(self, action):
        return 0.0