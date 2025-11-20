try:
    import isaacgym
    from isaacgym import gymapi, gymtorch
except ImportError as e:
    from rich import print

    print(
        """[red][Isaac Gym Import Error]
  1. You need to install Isaac Gym, if not installed.
    - Download Isaac Gym following https://clvrai.github.io/furniture-bench/docs/getting_started/installation_guide_furniture_sim.html#download-isaac-gym
    - Then, pip install -e isaacgym/python
  2. If PyTorch was imported before furniture_bench, please import torch after furniture_bench.[/red]
"""
    )
    print()
    raise ImportError(e)


from typing import Union
from datetime import datetime
from pathlib import Path

import torch
import cv2
import gym
import numpy as np

import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C
from furniture_bench.envs.initialization_mode import Randomness, str_to_enum
from furniture_bench.controllers.osc import osc_factory
from furniture_bench.furniture import furniture_factory
from furniture_bench.sim_config import sim_config
from furniture_bench.config import ROBOT_HEIGHT, config
from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.envs.observation import (
    FULL_OBS,
    DEFAULT_VISUAL_OBS,
    DEFAULT_STATE_OBS,
)
from furniture_bench.robot.robot_state import ROBOT_STATE_DIMS
from furniture_bench.furniture.parts.part import Part


ASSET_ROOT = str(Path(__file__).parent.parent.absolute() / "assets")


class FurnitureSimEnv(gym.Env):
    """FurnitureSim base class."""

    def __init__(
        self,
        furniture: str,
        num_envs: int = 1,
        resize_img: bool = True,
        obs_keys=None,
        concat_robot_state: bool = False,
        manual_label: bool = False,
        manual_done: bool = False,
        headless: bool = False,
        compute_device_id: int = 0,
        graphics_device_id: int = 0,
        init_assembled: bool = False,
        np_step_out: bool = False,
        channel_first: bool = False,
        randomness: Union[str, Randomness] = "low",
        high_random_idx: int = 0,
        save_camera_input: bool = False,
        record: bool = False,
        max_env_steps: int = 3000,
        act_rot_repr: str = "quat",
        **kwargs,
    ):
        """
        Args:
            furniture (str): Specifies the type of furniture. Options are 'lamp', 'square_table', 'desk', 'drawer', 'cabinet', 'round_table', 'stool', 'chair', 'one_leg'.
            num_envs (int): Number of parallel environments.
            resize_img (bool): If true, images are resized to 224 x 224.
            obs_keys (list): List of observations for observation space (i.e., RGB-D image from three cameras, proprioceptive states, and poses of the furniture parts.)
            concat_robot_state (bool): Whether to return concatenated `robot_state` or its dictionary form in observation.
            manual_label (bool): If true, the environment reward is manually labeled.
            manual_done (bool): If true, the environment is terminated manually.
            headless (bool): If true, simulation runs without GUI.
            compute_device_id (int): GPU device ID used for simulation.
            graphics_device_id (int): GPU device ID used for rendering.
            init_assembled (bool): If true, the environment is initialized with assembled furniture.
            np_step_out (bool): If true, env.step() returns Numpy arrays.
            channel_first (bool): If true, color images are returned in channel first format [3, H, w].
            randomness (str): Level of randomness in the environment. Options are 'low', 'med', 'high'.
            high_random_idx (int): Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
            save_camera_input (bool): If true, the initial camera inputs are saved.
            record (bool): If true, videos of the wrist and front cameras' RGB inputs are recorded.
            max_env_steps (int): Maximum number of steps per episode (default: 3000).
            act_rot_repr (str): Representation of rotation for action space. Options are 'quat', 'axis', or 'rot_6d'.
        """
        super(FurnitureSimEnv, self).__init__()
        self.device = torch.device("cuda", compute_device_id)

        self.assemble_idx = 0
        # Track if robot 1 has retreated to safe position for round_table
        self.robot1_retreated = False
        
        # For round_table spatial assignment
        self.round_table_arm_assignments = {}  # {env_idx: {'arm1': 'part_name', 'arm2': 'part_name'}}
        self.round_table_fsm_stage = 0  # 0: not started, 1: parallel grasp, 2: leg assembly, 3: base assembly
        self.round_table_stage1_complete = {'arm1': False, 'arm2': False}  # Track stage 1 completion
        
        # Furniture for each environment (reward, reset).
        self.furnitures = [furniture_factory(furniture) for _ in range(num_envs)]

        if num_envs == 1:
            self.furniture = self.furnitures[0]
        else:
            self.furniture = furniture_factory(furniture)

        self.furniture.max_env_steps = max_env_steps
        for furn in self.furnitures:
            furn.max_env_steps = max_env_steps

        self.furniture_name = furniture
        self.num_envs = num_envs
        self.obs_keys = obs_keys or DEFAULT_VISUAL_OBS
        self.robot_state_keys = [
            k.split("/")[1] for k in self.obs_keys if k.startswith("robot_state")
        ]
        self.concat_robot_state = concat_robot_state
        self.pose_dim = 7
        self.resize_img = resize_img
        self.manual_label = manual_label
        self.manual_done = manual_done
        self.headless = headless
        self.move_neutral = False
        self.ctrl_started = False
        self.init_assembled = init_assembled
        self.np_step_out = np_step_out
        self.channel_first = channel_first
        self.from_skill = (
            0  # TODO: Skill benchmark should be implemented in FurnitureSim.
        )
        self.randomness = str_to_enum(randomness)
        self.high_random_idx = high_random_idx
        self.last_grasp = torch.tensor([-1.0] * num_envs, device=self.device)
        self.grasp_margin = 0.02 - 0.001  # To prevent repeating open an close actions.
        self.max_gripper_width = config["robot"]["max_gripper_width"][furniture]
        self.gripper_pos_control = kwargs.get("gripper_pos_control", False)

        self.save_camera_input = save_camera_input
        self.img_size = sim_config["camera"][
            "resized_img_size" if resize_img else "color_img_size"
        ]

        # Simulator setup.
        self.isaac_gym = gymapi.acquire_gym()
        self.sim = self.isaac_gym.create_sim(
            compute_device_id,
            graphics_device_id,
            gymapi.SimType.SIM_PHYSX,
            sim_config["sim_params"],
        )
        self._create_ground_plane()
        self._setup_lights()
        self.import_assets()
        self.create_envs()
        self.set_viewer()
        self.set_camera()
        self.acquire_base_tensors()

        self.isaac_gym.prepare_sim(self.sim)
        self.refresh()

        self.isaac_gym.refresh_actor_root_state_tensor(self.sim)

        self.init_ee_pos, self.init_ee_quat = self.get_ee_pose()

        gym.logger.set_level(gym.logger.INFO)

        self.record = record
        if self.record:
            record_dir = Path("sim_record") / datetime.now().strftime("%Y%m%d-%H%M%S")
            record_dir.mkdir(parents=True, exist_ok=True)
            self.video_writer = cv2.VideoWriter(
                str(record_dir / "video.mp4"),
                cv2.VideoWriter_fourcc(*"MP4V"),
                30,
                (self.img_size[1] * 2, self.img_size[0]),  # Wrist and front cameras.
            )

        if act_rot_repr != "quat" and act_rot_repr != "axis" and act_rot_repr != "rot_6d":
            raise ValueError(f"Invalid rotation representation: {act_rot_repr}")
        self.act_rot_repr = act_rot_repr

        self.robot_state_as_dict = kwargs.get("robot_state_as_dict", True)
        self.squeeze_batch_dim = kwargs.get("squeeze_batch_dim", False)

    def _create_ground_plane(self):
        """Creates ground plane."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.isaac_gym.add_ground(self.sim, plane_params)

    def _setup_lights(self):
        for light in sim_config["lights"]:
            l_color = gymapi.Vec3(*light["color"])
            l_ambient = gymapi.Vec3(*light["ambient"])
            l_direction = gymapi.Vec3(*light["direction"])
            self.isaac_gym.set_light_parameters(
                self.sim, 0, l_color, l_ambient, l_direction
            )

    def create_envs(self):
        table_pos = gymapi.Vec3(0.8, 0.8, 0.4)
        self.franka_pose = gymapi.Transform()
        self.franka2_pose = gymapi.Transform()  # Second robot on left side

        table_half_width = 0.015
        table_surface_z = table_pos.z + table_half_width
        # First robot (right side, original position)
        self.franka_pose.p = gymapi.Vec3(
            0.5 * -table_pos.x + 0.1, 0, table_surface_z + ROBOT_HEIGHT
        )
        # Second robot (left edge of table)
        self.franka2_pose.p = gymapi.Vec3(
        #    0.5 * -table_pos.x + 0.1, 0, table_surface_z + ROBOT_HEIGHT
           -0.1, 0.5 * table_pos.y - 0.1, table_surface_z + ROBOT_HEIGHT
        )

        self.franka_from_origin_mat = get_mat(
            [self.franka_pose.p.x, self.franka_pose.p.y, self.franka_pose.p.z],
            [0, 0, 0],
        )
        self.franka2_from_origin_mat = get_mat(
            [self.franka2_pose.p.x, self.franka2_pose.p.y, self.franka2_pose.p.z],
            [0, 0, 0],
        )
        self.base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]
        
        # Compute base_tag transformation for robot 2
        # base_tag is at the same world position, but we need transformation relative to robot 2
        base_tag_world = self.franka_from_origin_mat @ self.base_tag_from_robot_mat
        self.base_tag_from_robot2_mat = np.linalg.inv(self.franka2_from_origin_mat) @ base_tag_world

        franka_link_dict = self.isaac_gym.get_asset_rigid_body_dict(self.franka_asset)
        self.franka_ee_index = franka_link_dict["k_ee_link"]
        self.franka_base_index = franka_link_dict["panda_link0"]
        # Parts assets.
        # Create assets.
        self.part_assets = {}
        for part in self.furniture.parts:
            asset_option = sim_config["asset"][part.name]
            self.part_assets[part.name] = self.isaac_gym.load_asset(
                self.sim, ASSET_ROOT, part.asset_file, asset_option
            )
        # Create envs.
        num_per_row = int(np.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.envs = []
        self.env_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self.handles = {}
        self.ee_idxs = []
        self.ee_handles = []
        self.osc_ctrls = []

        self.base_idxs = []
        self.part_idxs = {}
        self.franka_handles = []
        
        # Second robot variables
        self.ee2_idxs = []
        self.ee2_handles = []
        self.osc2_ctrls = []
        self.base2_idxs = []
        self.franka2_handles = []
        self.last_grasp2 = torch.tensor([-1.0] * self.num_envs, device=self.device)
        # Store base_tag positions per environment for calculating obstacle position
        self.base_tag_positions = []
        self.table_surface_z = table_pos.z + table_half_width
        for i in range(self.num_envs):
            env = self.isaac_gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            # Add workspace (table).
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0.0, 0.0, table_pos.z)

            table_handle = self.isaac_gym.create_actor(
                env, self.table_asset, table_pose, "table", i, 0
            )
            table_props = self.isaac_gym.get_actor_rigid_shape_properties(
                env, table_handle
            )
            table_props[0].friction = sim_config["table"]["friction"]
            self.isaac_gym.set_actor_rigid_shape_properties(
                env, table_handle, table_props
            )

            self.base_tag_pose = gymapi.Transform()
            base_tag_pos = T.pos_from_mat(config["robot"]["tag_base_from_robot_base"])
            self.base_tag_pose.p = self.franka_pose.p + gymapi.Vec3(
                base_tag_pos[0], base_tag_pos[1], -ROBOT_HEIGHT
            )
            self.base_tag_pose.p.z = table_surface_z
            base_tag_handle = self.isaac_gym.create_actor(
                env, self.base_tag_asset, self.base_tag_pose, "base_tag", i, 0
            )
            # Store base_tag position for calculating obstacle target position
            self.base_tag_positions.append(gymapi.Vec3(
                self.base_tag_pose.p.x,
                self.base_tag_pose.p.y,
                self.base_tag_pose.p.z
            ))
            bg_pos = gymapi.Vec3(-0.8, 0, 0.75)
            bg_pose = gymapi.Transform()
            bg_pose.p = gymapi.Vec3(bg_pos.x, bg_pos.y, bg_pos.z)
            bg_handle = self.isaac_gym.create_actor(
                env, self.background_asset, bg_pose, "background", i, 0
            )
            # Obstacles removed - franka2 will hold the base instead
            # TODO: Make config
            # obstacle_pose = gymapi.Transform()
            # obstacle_pose.p = gymapi.Vec3(
            #     self.base_tag_pose.p.x + 0.37 + 0.01, 0.0, table_surface_z + 0.015
            # )
            # obstacle_pose.r = gymapi.Quat.from_axis_angle(
            #     gymapi.Vec3(0, 0, 1), 0.5 * np.pi
            # )

            # obstacle_handle = self.isaac_gym.create_actor(
            #     env, self.obstacle_front_asset, obstacle_pose, f"obstacle_front", i, 0
            # )
            # part_idx = self.isaac_gym.get_actor_rigid_body_index(
            #     env, obstacle_handle, 0, gymapi.DOMAIN_SIM
            # )
            # if self.part_idxs.get("obstacle_front") is None:
            #     self.part_idxs["obstacle_front"] = [part_idx]
            # else:
            #     self.part_idxs[f"obstacle_front"].append(part_idx)

            # for j, name in enumerate(["obstacle_right", "obstacle_left"]):
            #     y = -0.175 if j == 0 else 0.175
            #     obstacle_pose = gymapi.Transform()
            #     obstacle_pose.p = gymapi.Vec3(
            #         self.base_tag_pose.p.x + 0.37 + 0.01 - 0.075,
            #         y,
            #         table_surface_z + 0.015,
            #     )
            #     obstacle_pose.r = gymapi.Quat.from_axis_angle(
            #         gymapi.Vec3(0, 0, 1), 0.5 * np.pi
            #     )

            #     obstacle_handle = self.isaac_gym.create_actor(
            #         env, self.obstacle_side_asset, obstacle_pose, name, i, 0
            #     )
            #     part_idx = self.isaac_gym.get_actor_rigid_body_index(
            #         env, obstacle_handle, 0, gymapi.DOMAIN_SIM
            #     )
            #     if self.part_idxs.get(name) is None:
            #         self.part_idxs[name] = [part_idx]
            #     else:
            #         self.part_idxs[name].append(part_idx)
            # Add robot.
            franka_handle = self.isaac_gym.create_actor(
                env, self.franka_asset, self.franka_pose, "franka", i, 0
            )
            self.franka_num_dofs = self.isaac_gym.get_actor_dof_count(
                env, franka_handle
            )

            self.isaac_gym.enable_actor_dof_force_sensors(env, franka_handle)
            self.franka_handles.append(franka_handle)

            # Get global index of hand and base.
            self.ee_idxs.append(
                self.isaac_gym.get_actor_rigid_body_index(
                    env, franka_handle, self.franka_ee_index, gymapi.DOMAIN_SIM
                )
            )
            self.ee_handles.append(
                self.isaac_gym.find_actor_rigid_body_handle(
                    env, franka_handle, "k_ee_link"
                )
            )
            self.base_idxs.append(
                self.isaac_gym.get_actor_rigid_body_index(
                    env, franka_handle, self.franka_base_index, gymapi.DOMAIN_SIM
                )
            )
            # Set dof properties.
            franka_dof_props = self.isaac_gym.get_asset_dof_properties(
                self.franka_asset
            )
            franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            franka_dof_props["stiffness"][:7].fill(0.0)
            franka_dof_props["damping"][:7].fill(0.0)
            franka_dof_props["friction"][:7] = sim_config["robot"]["arm_frictions"]
            # Grippers
            if self.gripper_pos_control:
                franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
                franka_dof_props["stiffness"][7:].fill(200.0)
                franka_dof_props["damping"][7:].fill(60.0)
            else:
                franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_EFFORT)
                franka_dof_props["stiffness"][7:].fill(0)
                franka_dof_props["damping"][7:].fill(0)
                franka_dof_props["friction"][7:] = sim_config["robot"]["gripper_frictions"]
            franka_dof_props["upper"][7:] = self.max_gripper_width / 2

            self.isaac_gym.set_actor_dof_properties(
                env, franka_handle, franka_dof_props
            )
            # Set initial dof states
            franka_num_dofs = self.isaac_gym.get_asset_dof_count(self.franka_asset)
            self.default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
            self.default_dof_pos[:7] = np.array(
                config["robot"]["reset_joints"], dtype=np.float32
            )
            self.default_dof_pos[7:] = self.max_gripper_width / 2
            default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
            default_dof_state["pos"] = self.default_dof_pos
            self.isaac_gym.set_actor_dof_states(
                env, franka_handle, default_dof_state, gymapi.STATE_ALL
            )
            
            # Add second robot (left side of table)
            franka2_handle = self.isaac_gym.create_actor(
                env, self.franka_asset, self.franka2_pose, "franka2", i, 0
            )
            self.isaac_gym.enable_actor_dof_force_sensors(env, franka2_handle)
            self.franka2_handles.append(franka2_handle)

            # Get global index of hand and base for second robot
            self.ee2_idxs.append(
                self.isaac_gym.get_actor_rigid_body_index(
                    env, franka2_handle, self.franka_ee_index, gymapi.DOMAIN_SIM
                )
            )
            self.ee2_handles.append(
                self.isaac_gym.find_actor_rigid_body_handle(
                    env, franka2_handle, "k_ee_link"
                )
            )
            self.base2_idxs.append(
                self.isaac_gym.get_actor_rigid_body_index(
                    env, franka2_handle, self.franka_base_index, gymapi.DOMAIN_SIM
                )
            )
            
            # Add franka2 end-effector to part_idxs so it can be accessed like obstacles
            if self.part_idxs.get("franka2_ee") is None:
                self.part_idxs["franka2_ee"] = [self.ee2_idxs[i]]
            else:
                self.part_idxs["franka2_ee"].append(self.ee2_idxs[i])
            
            # Set dof properties for second robot (same as first)
            self.isaac_gym.set_actor_dof_properties(
                env, franka2_handle, franka_dof_props
            )
            # Set initial dof states for second robot
            self.isaac_gym.set_actor_dof_states(
                env, franka2_handle, default_dof_state, gymapi.STATE_ALL
            )
            
            # Add furniture parts.
            poses = []
            for part in self.furniture.parts:
                pos, ori = self._get_reset_pose(part)
                part_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
                part_pose = gymapi.Transform()
                part_pose.p = gymapi.Vec3(
                    part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3]
                )
                reset_ori = self.april_coord_to_sim_coord(ori)
                part_pose.r = gymapi.Quat(*T.mat2quat(reset_ori[:3, :3]))
                poses.append(part_pose)
                part_handle = self.isaac_gym.create_actor(
                    env, self.part_assets[part.name], part_pose, part.name, i, 0
                )
                self.handles[part.name] = part_handle

                part_idx = self.isaac_gym.get_actor_rigid_body_index(
                    env, part_handle, 0, gymapi.DOMAIN_SIM
                )
                # Set properties of part.
                part_props = self.isaac_gym.get_actor_rigid_shape_properties(
                    env, part_handle
                )
                part_props[0].friction = sim_config["parts"]["friction"]
                self.isaac_gym.set_actor_rigid_shape_properties(
                    env, part_handle, part_props
                )

                if self.part_idxs.get(part.name) is None:
                    self.part_idxs[part.name] = [part_idx]
                else:
                    self.part_idxs[part.name].append(part_idx)

            self.parts_handles = {}
            for part in self.furniture.parts:
                self.parts_handles[part.name] = self.isaac_gym.find_actor_index(
                    env, part.name, gymapi.DOMAIN_ENV
                )
        
        # print(f'Getting the separate actor indices for the frankas and the furniture parts (not the handles)')
        self.franka_actor_idx_all = []
        self.franka2_actor_idx_all = []
        self.part_actor_idx_all = []  # global list of indices, when resetting all parts
        self.part_actor_idx_by_env = {}  # allow to access part indices based on environment indices
        for env_idx in range(self.num_envs):
            self.franka_actor_idx_all.append(self.isaac_gym.find_actor_index(self.envs[env_idx], 'franka', gymapi.DOMAIN_SIM))
            self.franka2_actor_idx_all.append(self.isaac_gym.find_actor_index(self.envs[env_idx], 'franka2', gymapi.DOMAIN_SIM))
            self.part_actor_idx_by_env[env_idx] = []
            for part in self.furnitures[env_idx].parts:
                part_actor_idx = self.isaac_gym.find_actor_index(self.envs[env_idx], part.name, gymapi.DOMAIN_SIM)
                self.part_actor_idx_all.append(part_actor_idx)
                self.part_actor_idx_by_env[env_idx].append(part_actor_idx)

        self.franka_actor_idxs_all_t = torch.tensor(self.franka_actor_idx_all, device=self.device, dtype=torch.int32)
        self.franka2_actor_idxs_all_t = torch.tensor(self.franka2_actor_idx_all, device=self.device, dtype=torch.int32)
        self.part_actor_idxs_all_t = torch.tensor(self.part_actor_idx_all, device=self.device, dtype=torch.int32)

    def _get_reset_pose(self, part: Part):
        """Get the reset pose of the part.

        Args:
            part: The part to get the reset pose.
        """
        if self.init_assembled:
            if part.name == "chair_seat":
                # Special case handling for chair seat since the assembly of chair back is not available from initialized pose.
                part.reset_pos = [[0, 0.16, -0.035]]
                part.reset_ori = [rot_mat([np.pi, 0, 0], hom=True)]
            attached_part = False
            attach_to = None
            for assemble_pair in self.furniture.should_be_assembled:
                if part.part_idx == assemble_pair[1]:
                    attached_part = True
                    attach_to = self.furniture.parts[assemble_pair[0]]
                    break
            if attached_part:
                attach_part_pos = self.furniture.parts[attach_to.part_idx].reset_pos[0]
                attach_part_ori = self.furniture.parts[attach_to.part_idx].reset_ori[0]
                attach_part_pose = get_mat(attach_part_pos, attach_part_ori)
                if part.default_assembled_pose is not None:
                    pose = attach_part_pose @ part.default_assembled_pose
                    pos = pose[:3, 3]
                    ori = T.to_hom_ori(pose[:3, :3])
                else:
                    pos = (
                        attach_part_pose
                        @ self.furniture.assembled_rel_poses[
                            (attach_to.part_idx, part.part_idx)
                        ][0][:4, 3]
                    )
                    pos = pos[:3]
                    ori = (
                        attach_part_pose
                        @ self.furniture.assembled_rel_poses[
                            (attach_to.part_idx, part.part_idx)
                        ][0]
                    )
                part.reset_pos[0] = pos
                part.reset_ori[0] = ori
            pos = part.reset_pos[self.from_skill]
            ori = part.reset_ori[self.from_skill]
        else:
            pos = part.reset_pos[self.from_skill]
            ori = part.reset_ori[self.from_skill]
        return pos, ori

    def set_viewer(self):
        """Create the viewer."""
        self.enable_viewer_sync = True
        self.viewer = None

        if not self.headless:
            self.viewer = self.isaac_gym.create_viewer(
                self.sim, gymapi.CameraProperties()
            )
            # Point camera at middle env.
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.62)
            middle_env = self.envs[0]
            self.isaac_gym.viewer_camera_look_at(
                self.viewer, middle_env, cam_pos, cam_target
            )

    def set_camera(self):
        self.camera_handles = {}
        self.camera_obs = {}

        def create_camera(name, i):
            env = self.envs[i]
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = self.img_size[0]
            camera_cfg.height = self.img_size[1]
            camera_cfg.near_plane = 0.001
            camera_cfg.far_plane = 2.0
            camera_cfg.horizontal_fov = 40.0 if self.resize_img else 69.4
            self.camera_cfg = camera_cfg

            if name == "wrist":
                if self.resize_img:
                    camera_cfg.horizontal_fov = 55.0  # Wide view.
                camera = self.isaac_gym.create_camera_sensor(env, camera_cfg)
                transform = gymapi.Transform()
                transform.p = gymapi.Vec3(-0.04, 0, -0.05)
                transform.r = gymapi.Quat.from_axis_angle(
                    gymapi.Vec3(0, 1, 0), np.radians(-70.0)
                )
                self.isaac_gym.attach_camera_to_body(
                    camera, env, self.ee_handles[i], transform, gymapi.FOLLOW_TRANSFORM
                )
            elif name == "front":
                camera = self.isaac_gym.create_camera_sensor(env, camera_cfg)
                cam_pos = gymapi.Vec3(0.90, -0.00, 0.65)
                cam_target = gymapi.Vec3(-1, -0.00, 0.3)
                self.isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
                self.front_cam_pos = np.array([cam_pos.x, cam_pos.y, cam_pos.z])
                self.front_cam_target = np.array(
                    [cam_target.x, cam_target.y, cam_target.z]
                )
            elif name == "rear":
                camera = self.isaac_gym.create_camera_sensor(env, camera_cfg)
                transform = gymapi.Transform()
                transform.p = gymapi.Vec3(
                    self.franka_pose.p.x + 0.08, 0, self.franka_pose.p.z + 0.2
                )
                transform.r = gymapi.Quat.from_axis_angle(
                    gymapi.Vec3(0, 1, 0), np.radians(35.0)
                )
                self.isaac_gym.set_camera_transform(camera, env, transform)
            return camera

        camera_names = {"1": "wrist", "2": "front", "3": "rear"}
        for env_idx, env in enumerate(self.envs):
            for k in self.obs_keys:
                if k.startswith("color"):
                    camera_name = camera_names[k[-1]]
                    render_type = gymapi.IMAGE_COLOR
                elif k.startswith("depth"):
                    camera_name = camera_names[k[-1]]
                    render_type = gymapi.IMAGE_DEPTH
                else:
                    continue
                if camera_name not in self.camera_handles:
                    self.camera_handles[camera_name] = []
                # Only when the camera handle for the current environment does not exist.
                if len(self.camera_handles[camera_name]) <= env_idx:
                    self.camera_handles[camera_name].append(create_camera(camera_name, env_idx))
                handle = self.camera_handles[camera_name][env_idx]
                tensor = gymtorch.wrap_tensor(
                    self.isaac_gym.get_camera_image_gpu_tensor(
                        self.sim, env, handle, render_type
                    )
                )
                if k not in self.camera_obs:
                    self.camera_obs[k] = []
                self.camera_obs[k].append(tensor)

    def import_assets(self):
        self.base_tag_asset = self._import_base_tag_asset()
        self.background_asset = self._import_background_asset()
        self.table_asset = self._import_table_asset()
        self.obstacle_front_asset = self._import_obstacle_front_asset()
        self.obstacle_side_asset = self._import_obstacle_side_asset()
        self.franka_asset = self._import_franka_asset()

    def acquire_base_tensors(self):
        # Get rigid body state tensor
        _rb_states = self.isaac_gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        _root_tensor = self.isaac_gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor)
        self.root_pos = self.root_tensor.view(self.num_envs, -1, 13)[..., 0:3]
        self.root_quat = self.root_tensor.view(self.num_envs, -1, 13)[..., 3:7]

        _forces = self.isaac_gym.acquire_dof_force_tensor(self.sim)
        _forces = gymtorch.wrap_tensor(_forces)
        # With two robots, we have 18 DOFs per environment
        self.forces = _forces.view(self.num_envs, 18)

        # Get DoF tensor
        _dof_states = self.isaac_gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(
            _dof_states
        )  # (num_dofs, 2), 2 for pos and vel.
        # With two robots, we have 18 DOFs per environment (9 per robot)
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, 18)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, 18)
        # Get jacobian tensor for first robot
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.isaac_gym.acquire_jacobian_tensor(self.sim, "franka")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        # jacobian entries corresponding to franka hand
        self.jacobian_eef = self.jacobian[
            :, self.franka_ee_index - 1, :, :7
        ]  # -1 due to finxed base link.
        # Get jacobian tensor for second robot
        _jacobian2 = self.isaac_gym.acquire_jacobian_tensor(self.sim, "franka2")
        self.jacobian2 = gymtorch.wrap_tensor(_jacobian2)
        self.jacobian_eef2 = self.jacobian2[
            :, self.franka_ee_index - 1, :, :7
        ]
        # Prepare mass matrix tensor for first robot
        # For franka, tensor shape is (num_envs, 7 + 2, 7 + 2), 2 for grippers.
        _massmatrix = self.isaac_gym.acquire_mass_matrix_tensor(self.sim, "franka")
        self.mm = gymtorch.wrap_tensor(_massmatrix)
        # Mass matrix for second robot
        _massmatrix2 = self.isaac_gym.acquire_mass_matrix_tensor(self.sim, "franka2")
        self.mm2 = gymtorch.wrap_tensor(_massmatrix2)

    def april_coord_to_sim_coord(self, april_coord_mat):
        """Converts AprilTag coordinate to simulator base_tag coordinate."""
        return self.april_to_sim_mat @ april_coord_mat

    def sim_coord_to_april_coord(self, sim_coord_mat):
        return self.sim_to_april_mat @ sim_coord_mat

    @property
    def april_to_sim_mat(self):
        return self.franka_from_origin_mat @ self.base_tag_from_robot_mat

    @property
    def sim_to_april_mat(self):
        return torch.tensor(
            np.linalg.inv(self.base_tag_from_robot_mat)
            @ np.linalg.inv(self.franka_from_origin_mat),
            device=self.device,
        )

    @property
    def sim_to_robot_mat(self):
        return torch.tensor(self.franka_from_origin_mat, device=self.device)

    @property
    def april_to_robot_mat(self):
        return torch.tensor(self.base_tag_from_robot_mat, device=self.device)
    
    @property
    def april_to_robot2_mat(self):
        """Transformation from AprilTag to robot 2 base coordinate."""
        # Compute transformation from AprilTag to robot 2 base
        # base_tag is positioned relative to robot 1, we need to compute it relative to robot 2
        return torch.tensor(self.base_tag_from_robot2_mat, device=self.device)

    @property
    def robot_to_ee_mat(self):
        return torch.tensor(rot_mat([np.pi, 0, 0], hom=True), device=self.device)

    @property
    def action_space(self):
        # Action space to be -1.0 to 1.0.
        if self.act_rot_repr == "quat":
            pose_dim = 7
        elif self.act_rot_repr == "rot_6d":
            pose_dim = 9
        else: # axis
            pose_dim = 6

        low = np.array([-1] * pose_dim + [-1], dtype=np.float32)
        high = np.array([1] * pose_dim + [1], dtype=np.float32)

        low = np.tile(low, (self.num_envs, 1))
        high = np.tile(high, (self.num_envs, 1))

        return gym.spaces.Box(low, high, (self.num_envs, pose_dim + 1))
    
    @property
    def action_dimension(self):
        return self.action_space.shape[-1]

    @property
    def observation_space(self):
        low, high = -np.inf, np.inf
        parts_poses = self.furniture.num_parts * self.pose_dim
        img_size = reversed(self.img_size)
        img_shape = (3, *img_size) if self.channel_first else (*img_size, 3)

        obs_dict = {}
        robot_state = {}
        robot_state_dim = 0
        for k in self.obs_keys:
            if k.startswith("robot_state"):
                obs_key = k.split("/")[1]
                obs_shape = (ROBOT_STATE_DIMS[obs_key],)
                robot_state_dim += ROBOT_STATE_DIMS[obs_key]
                robot_state[obs_key] = gym.spaces.Box(low, high, obs_shape)
            elif k.startswith("color"):
                obs_dict[k] = gym.spaces.Box(0, 255, img_shape)
            elif k.startswith("depth"):
                obs_dict[k] = gym.spaces.Box(0, 255, img_size)
            elif k == "parts_poses":
                obs_dict[k] = gym.spaces.Box(low, high, parts_poses)
            else:
                raise ValueError(f"FurnitureSim does not support observation ({k}).")

        if robot_state:
            if self.concat_robot_state:
                obs_dict["robot_state"] = gym.spaces.Box(low, high, (robot_state_dim,))
            else:
                obs_dict["robot_state"] = gym.spaces.Dict(robot_state)

        return gym.spaces.Dict(obs_dict)

    @torch.no_grad()
    def step(self, action):
        """Robot takes an action.

        Args:
            action:
                (num_envs, 8): End-effector delta in [x, y, z, qx, qy, qz, qw, gripper] if self.act_rot_repr == "quat".
                (num_envs, 10): End-effector delta in [x, y, z, 6D rotation, gripper] if self.act_rot_repr == "rot_6d".
                (num_envs, 7): End-effector delta in [x, y, z, ax, ay, az, gripper] if self.act_rot_repr == "axis".
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(device=self.device)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        # Clip the action to be within the action space.
        low = torch.from_numpy(self.action_space.low).to(device=self.device)
        high = torch.from_numpy(self.action_space.high).to(device=self.device)
        action = torch.clamp(action, low, high)

        sim_steps = int(
            1.0
            / config["robot"]["hz"]
            / sim_config["sim_params"].dt
            / sim_config["sim_params"].substeps
            + 0.1
        )
        if not self.ctrl_started:
            self.init_ctrl()
        # Set the goal for both robots
        ee_pos, ee_quat = self.get_ee_pose()
        ee2_pos, ee2_quat = self.get_ee2_pose()

        for env_idx in range(self.num_envs):
            if self.act_rot_repr == "quat":
                action_quat = action[env_idx][3:7]
            elif self.act_rot_repr == "rot_6d":
                import pytorch3d.transforms as pt
                # Create "actions" dataset.
                rot_6d = action[:, 3:9]
                rot_mat = pt.rotation_6d_to_matrix(rot_6d)
                quat = pt.matrix_to_quaternion(rot_mat)
                action_quat = quat[env_idx]
            else:
                action_quat = C.axisangle2quat(action[env_idx][3:6])

            furniture = self.furnitures[env_idx] if hasattr(self, 'furnitures') else self.furniture
            
            # Determine control strategy based on furniture type
            if self.furniture_name == "round_table":
                # New spatial assignment logic with 3 stages
                # Stage 1: Parallel grasp
                # Stage 2: Leg assembly + retreat
                # Stage 3: Base assembly
                
                if self.round_table_fsm_stage == 1:
                    # Stage 1: Both robots grasp in parallel
                    # Robot 1 executes action from get_assembly_action
                    self.osc_ctrls[env_idx].set_goal(
                        action[env_idx][:3] + ee_pos[env_idx],
                        C.quat_multiply(ee_quat[env_idx], action_quat).to(self.device),
                    )
                    
                    # Robot 2 executes stored action
                    if hasattr(self, 'round_table_arm2_action'):
                        if self.act_rot_repr == "quat":
                            action2_quat = self.round_table_arm2_action[3:7]
                        elif self.act_rot_repr == "rot_6d":
                            import pytorch3d.transforms as pt
                            rot_6d = self.round_table_arm2_action[3:9]
                            rot_mat = pt.rotation_6d_to_matrix(rot_6d.unsqueeze(0))
                            quat = pt.matrix_to_quaternion(rot_mat)
                            action2_quat = quat[0]
                        else:
                            action2_quat = C.axisangle2quat(self.round_table_arm2_action[3:6])
                        
                        self.osc2_ctrls[env_idx].set_goal(
                            self.round_table_arm2_action[:3] + ee2_pos[env_idx],
                            C.quat_multiply(ee2_quat[env_idx], action2_quat).to(self.device),
                        )
                
                elif self.round_table_fsm_stage == 2:
                    # Stage 2: Leg assembly (one arm active) or retreat
                    if self.leg_holding_arm == 1:
                        # Robot 1 is active
                        self.osc_ctrls[env_idx].set_goal(
                            action[env_idx][:3] + ee_pos[env_idx],
                            C.quat_multiply(ee_quat[env_idx], action_quat).to(self.device),
                        )
                        # Robot 2 stays at current position
                        self.osc2_ctrls[env_idx].set_goal(
                            ee2_pos[env_idx],
                            ee2_quat[env_idx],
                        )
                    else:  # leg_holding_arm == 2
                        # Robot 2 is active
                        self.osc2_ctrls[env_idx].set_goal(
                            action[env_idx][:3] + ee2_pos[env_idx],
                            C.quat_multiply(ee2_quat[env_idx], action_quat).to(self.device),
                        )
                        # Robot 1 stays at current position
                        self.osc_ctrls[env_idx].set_goal(
                            ee_pos[env_idx],
                            ee_quat[env_idx],
                        )
                
                elif self.round_table_fsm_stage == 3:
                    # Stage 3: Base assembly
                    base_holding_arm = 2 if self.leg_holding_arm == 1 else 1
                    
                    if base_holding_arm == 1:
                        # Robot 1 is active
                        self.osc_ctrls[env_idx].set_goal(
                            action[env_idx][:3] + ee_pos[env_idx],
                            C.quat_multiply(ee_quat[env_idx], action_quat).to(self.device),
                        )
                        # Robot 2 maintains safe position
                        if self.leg_holding_arm == 2:
                            safe_pos = torch.tensor([0.3, 0.2, 0.25], device=self.device)
                            safe_quat = torch.tensor([0, 0.707, 0, 0.707], device=self.device)
                            self.osc2_ctrls[env_idx].set_goal(
                                safe_pos,
                                safe_quat,
                            )
                    else:  # base_holding_arm == 2
                        # Robot 2 is active
                        self.osc2_ctrls[env_idx].set_goal(
                            action[env_idx][:3] + ee2_pos[env_idx],
                            C.quat_multiply(ee2_quat[env_idx], action_quat).to(self.device),
                        )
                        # Robot 1 maintains safe position
                        if self.leg_holding_arm == 1:
                            safe_pos = torch.tensor([0.3, -0.2, 0.25], device=self.device)
                            safe_quat = torch.tensor([0, 0.707, 0, 0.707], device=self.device)
                            self.osc_ctrls[env_idx].set_goal(
                                safe_pos,
                                safe_quat,
                            )
            elif self.furniture_name == "lamp":
                # For lamp: original logic with robot 1 active and robot 2 as obstacle/holder
                self.osc_ctrls[env_idx].set_goal(
                    action[env_idx][:3] + ee_pos[env_idx],
                    C.quat_multiply(ee_quat[env_idx], action_quat).to(self.device),
                )
                # Second robot - move to obstacle position first, then grasp lamp base when it arrives
                lamp_base_part = None
                for part in furniture.parts:
                    if part.name == "lamp_base":
                        lamp_base_part = part
                        break
                
                # Calculate obstacle target position from base_tag position
                base_tag_pos = self.base_tag_positions[env_idx]
                obstacle_target_world = torch.tensor([
                    base_tag_pos.x + 0.37 + 0.01,
                    0.0,
                    self.table_surface_z + 0.015
                ], device=self.device)
                
                # Get franka2's base position in world coordinates
                franka2_base_world = self.rb_states[self.base2_idxs[env_idx], :3]
                
                # Set orientation: gripper parallel to ground (horizontal)
                gripper_ori_ground = torch.tensor([0.707, 0, 0, 0.707], device=self.device)  # Quaternion for 90 deg rotation around x-axis
                
                if lamp_base_part is not None:
                    # Check if lamp_base is in push state (franka is pushing base forward)
                    if lamp_base_part._state in ["push", "push_x"]:
                        # When franka is pushing, franka2 should be at obstacle position ready to grasp
                        obstacle_target_rel = obstacle_target_world - franka2_base_world
                        self.osc2_ctrls[env_idx].set_goal(
                            obstacle_target_rel,
                            gripper_ori_ground,
                        )
                    elif lamp_base_part._state == "release":
                        # When base is released, franka2 should grasp it
                        # Get lamp_base position in world coordinates
                        if "lamp_base" in self.part_idxs and len(self.part_idxs["lamp_base"]) > env_idx:
                            lamp_base_idx = self.part_idxs["lamp_base"][env_idx]
                            lamp_base_pos = self.rb_states[lamp_base_idx, :3]
                            # Position franka2's gripper to grasp the base (slightly above and in front)
                            grasp_target = lamp_base_pos.clone()
                            grasp_target[2] += 0.02  # Slightly above
                            grasp_target_rel = grasp_target - franka2_base_world
                            self.osc2_ctrls[env_idx].set_goal(
                                grasp_target_rel,
                                gripper_ori_ground,
                            )
                        else:
                            # Fallback: move to obstacle position
                            obstacle_target_rel = obstacle_target_world - franka2_base_world
                            self.osc2_ctrls[env_idx].set_goal(
                                obstacle_target_rel,
                                gripper_ori_ground,
                            )
                    else:
                        # Before push state: move franka2 to obstacle position (serving as obstacle)
                        obstacle_target_rel = obstacle_target_world - franka2_base_world
                        self.osc2_ctrls[env_idx].set_goal(
                            obstacle_target_rel,
                            gripper_ori_ground,
                        )
                else:
                    # No lamp_base part found, move to obstacle position
                    obstacle_target_rel = obstacle_target_world - franka2_base_world
                    self.osc2_ctrls[env_idx].set_goal(
                        obstacle_target_rel,
                        gripper_ori_ground,
                    )
            else:
                # For other furniture types: only use robot 1
                self.osc_ctrls[env_idx].set_goal(
                    action[env_idx][:3] + ee_pos[env_idx],
                    C.quat_multiply(ee_quat[env_idx], action_quat).to(self.device),
                )
                # Keep robot 2 at neutral
                self.osc2_ctrls[env_idx].set_goal(
                    ee2_pos[env_idx],
                    ee2_quat[env_idx],
                )

        for _ in range(sim_steps):
            self.refresh()

            pos_action = torch.zeros_like(self.dof_pos)
            torque_action = torch.zeros_like(self.dof_pos)
            grip_action = torch.zeros((self.num_envs, 2))  # Two grippers
            for env_idx in range(self.num_envs):
                grasp = action[env_idx, -1]
                furniture = self.furnitures[env_idx] if hasattr(self, 'furnitures') else self.furniture
                
                if self.furniture_name == "round_table":
                    # New spatial assignment logic with 3 stages
                    if self.round_table_fsm_stage == 1:
                        # Stage 1: Both robots control their grippers (parallel grasp)
                        # Robot 1
                        if (
                            torch.sign(grasp) != torch.sign(self.last_grasp[env_idx])
                            and torch.abs(grasp) > self.grasp_margin
                        ):
                            grip_sep1 = self.max_gripper_width if grasp < 0 else 0.0
                            self.last_grasp[env_idx] = grasp
                        else:
                            if self.last_grasp[env_idx] < 0:
                                grip_sep1 = self.max_gripper_width
                            else:
                                grip_sep1 = 0.0
                        grip_action[env_idx, 0] = grip_sep1
                        
                        # Robot 2 - use stored action
                        if hasattr(self, 'round_table_arm2_action'):
                            grasp2 = self.round_table_arm2_action[-1]
                            if (
                                torch.sign(grasp2) != torch.sign(self.last_grasp2[env_idx])
                                and torch.abs(grasp2) > self.grasp_margin
                            ):
                                grip_sep2 = self.max_gripper_width if grasp2 < 0 else 0.0
                                self.last_grasp2[env_idx] = grasp2
                            else:
                                if self.last_grasp2[env_idx] < 0:
                                    grip_sep2 = self.max_gripper_width
                                else:
                                    grip_sep2 = 0.0
                            grip_action[env_idx, 1] = grip_sep2
                        else:
                            grip_action[env_idx, 1] = self.max_gripper_width
                    
                    elif self.round_table_fsm_stage == 2:
                        # Stage 2: Only leg-holding arm is active
                        if self.leg_holding_arm == 1:
                            # Robot 1 is active (assembling leg)
                            if (
                                torch.sign(grasp) != torch.sign(self.last_grasp[env_idx])
                                and torch.abs(grasp) > self.grasp_margin
                            ):
                                grip_sep = self.max_gripper_width if grasp < 0 else 0.0
                                self.last_grasp[env_idx] = grasp
                            else:
                                if self.last_grasp[env_idx] < 0:
                                    grip_sep = self.max_gripper_width
                                else:
                                    grip_sep = 0.0
                            grip_action[env_idx, 0] = grip_sep
                            # Robot 2 holds base - MUST keep gripper CLOSED
                            grip_action[env_idx, 1] = 0.0  # Closed to hold base
                        else:  # leg_holding_arm == 2
                            # Robot 2 is active (assembling leg)
                            if (
                                torch.sign(grasp) != torch.sign(self.last_grasp2[env_idx])
                                and torch.abs(grasp) > self.grasp_margin
                            ):
                                grip_sep = self.max_gripper_width if grasp < 0 else 0.0
                                self.last_grasp2[env_idx] = grasp
                            else:
                                if self.last_grasp2[env_idx] < 0:
                                    grip_sep = self.max_gripper_width
                                else:
                                    grip_sep = 0.0
                            grip_action[env_idx, 1] = grip_sep
                            # Robot 1 holds base - MUST keep gripper CLOSED
                            grip_action[env_idx, 0] = 0.0  # Closed to hold base
                    
                    elif self.round_table_fsm_stage == 3:
                        # Stage 3: Only base-holding arm is active
                        base_holding_arm = 2 if self.leg_holding_arm == 1 else 1
                        
                        if base_holding_arm == 1:
                            # Robot 1 is active
                            if (
                                torch.sign(grasp) != torch.sign(self.last_grasp[env_idx])
                                and torch.abs(grasp) > self.grasp_margin
                            ):
                                grip_sep = self.max_gripper_width if grasp < 0 else 0.0
                                self.last_grasp[env_idx] = grasp
                            else:
                                if self.last_grasp[env_idx] < 0:
                                    grip_sep = self.max_gripper_width
                                else:
                                    grip_sep = 0.0
                            grip_action[env_idx, 0] = grip_sep
                            # Robot 2 gripper open (in retreat)
                            grip_action[env_idx, 1] = self.max_gripper_width
                        else:  # base_holding_arm == 2
                            # Robot 2 is active
                            if (
                                torch.sign(grasp) != torch.sign(self.last_grasp2[env_idx])
                                and torch.abs(grasp) > self.grasp_margin
                            ):
                                grip_sep = self.max_gripper_width if grasp < 0 else 0.0
                                self.last_grasp2[env_idx] = grasp
                            else:
                                if self.last_grasp2[env_idx] < 0:
                                    grip_sep = self.max_gripper_width
                                else:
                                    grip_sep = 0.0
                            grip_action[env_idx, 1] = grip_sep
                            # Robot 1 gripper open (in retreat)
                            grip_action[env_idx, 0] = self.max_gripper_width
                    else:
                        # Default: both grippers open
                        grip_action[env_idx, 0] = self.max_gripper_width
                        grip_action[env_idx, 1] = self.max_gripper_width
                elif self.furniture_name == "lamp":
                    # For lamp: original logic
                    # First robot gripper
                    if (
                        torch.sign(grasp) != torch.sign(self.last_grasp[env_idx])
                        and torch.abs(grasp) > self.grasp_margin
                    ):
                        grip_sep = self.max_gripper_width if grasp < 0 else 0.0
                        self.last_grasp[env_idx] = grasp
                    else:
                        if self.last_grasp[env_idx] < 0:
                            grip_sep = self.max_gripper_width
                        else:
                            grip_sep = 0.0
                    grip_action[env_idx, 0] = grip_sep
                    
                    # Second robot gripper - close when grasping lamp base
                    lamp_base_part = None
                    for part in furniture.parts:
                        if part.name == "lamp_base":
                            lamp_base_part = part
                            break
                    
                    # Close gripper when in release state (base is being released by franka)
                    if lamp_base_part is not None and lamp_base_part._state == "release":
                        # Close gripper to grasp
                        grip_action[env_idx, 1] = 0.0  # Closed
                    else:
                        # Keep gripper open or maintain current state
                        current_gripper2 = self.dof_pos[env_idx, 16:17] + self.dof_pos[env_idx, 17:18]
                        grip_action[env_idx, 1] = current_gripper2
                else:
                    # For other furniture: only control robot 1
                    if (
                        torch.sign(grasp) != torch.sign(self.last_grasp[env_idx])
                        and torch.abs(grasp) > self.grasp_margin
                    ):
                        grip_sep = self.max_gripper_width if grasp < 0 else 0.0
                        self.last_grasp[env_idx] = grasp
                    else:
                        if self.last_grasp[env_idx] < 0:
                            grip_sep = self.max_gripper_width
                        else:
                            grip_sep = 0.0
                    grip_action[env_idx, 0] = grip_sep
                    # Robot 2 stays open
                    grip_action[env_idx, 1] = self.max_gripper_width

                # First robot control
                state_dict = {}
                ee_pos, ee_quat = self.get_ee_pose()
                state_dict["ee_pose"] = C.pose2mat(
                    ee_pos[env_idx], ee_quat[env_idx], self.device
                ).t()  # OSC expect column major
                state_dict["joint_positions"] = self.dof_pos[env_idx][:7]
                state_dict["joint_velocities"] = self.dof_vel[env_idx][:7]
                state_dict["mass_matrix"] = self.mm[env_idx][
                    :7, :7
                ].t()  # OSC expect column major
                state_dict["jacobian"] = self.jacobian_eef[
                    env_idx
                ].t()  # OSC expect column major
                torque_action[env_idx, :7] = self.osc_ctrls[env_idx](state_dict)[
                    "joint_torques"
                ]

                # Second robot control - keep it stationary
                state_dict2 = {}
                ee2_pos, ee2_quat = self.get_ee2_pose()
                state_dict2["ee_pose"] = C.pose2mat(
                    ee2_pos[env_idx], ee2_quat[env_idx], self.device
                ).t()
                state_dict2["joint_positions"] = self.dof_pos[env_idx][9:16]  # DOFs 9-15 for second robot
                state_dict2["joint_velocities"] = self.dof_vel[env_idx][9:16]
                state_dict2["mass_matrix"] = self.mm2[env_idx][
                    :7, :7
                ].t()
                state_dict2["jacobian"] = self.jacobian_eef2[
                    env_idx
                ].t()
                torque_action[env_idx, 9:16] = self.osc2_ctrls[env_idx](state_dict2)[
                    "joint_torques"
                ]
                # Keep franka2's gripper stationary (maintain current state)
                # Don't update gripper based on action

                # Gripper actions
                if self.gripper_pos_control:
                    pos_action[env_idx, 7:9] = grip_action[env_idx, 0]
                    pos_action[env_idx, 16:18] = grip_action[env_idx, 1]
                else:
                    # Robot 1 gripper torque
                    if grip_action[env_idx, 0] > 0:
                        torque_action[env_idx, 7:9] = sim_config["robot"]["gripper_torque"]
                    else:
                        torque_action[env_idx, 7:9] = -sim_config["robot"]["gripper_torque"]
                    
                    # Robot 2 gripper torque
                    if grip_action[env_idx, 1] > 0:
                        torque_action[env_idx, 16:18] = sim_config["robot"]["gripper_torque"]
                    else:
                        torque_action[env_idx, 16:18] = -sim_config["robot"]["gripper_torque"]
            
            # Apply actions
            if self.gripper_pos_control:
                self.isaac_gym.set_dof_position_target_tensor(
                    self.sim, gymtorch.unwrap_tensor(pos_action)
                )
            self.isaac_gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(torque_action)
            )

            # Update viewer
            if not self.headless:
                self.isaac_gym.draw_viewer(self.viewer, self.sim, False)
                self.isaac_gym.sync_frame_time(self.sim)

        self.isaac_gym.end_access_image_tensors(self.sim)

        obs = self._get_observation()
        self.env_steps += 1

        return (
            obs,
            self._reward(),
            self._done(),
            {"obs_success": True, "action_success": True},
        )

    def _reward(self):
        """Reward is 1 if two parts are assembled."""
        rewards = torch.zeros(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )

        if self.manual_label:
            # Return zeros since the reward is manually labeled by data_collector.py.
            return rewards

        # Don't have to convert to AprilTag coordinate since the reward is computed with relative poses.
        parts_poses, founds = self._get_parts_poses(sim_coord=True) 
        for env_idx in range(self.num_envs):
            env_parts_poses = parts_poses[env_idx].cpu().numpy()
            env_founds = founds[env_idx].cpu().numpy()
            rewards[env_idx] = self.furnitures[env_idx].compute_assemble(
                env_parts_poses, env_founds
            )

        if self.np_step_out:
            return rewards.cpu().numpy()

        return rewards

    def _get_parts_poses(self, sim_coord=False):
        """Get furniture parts poses in the AprilTag frame.
        
        Args:
            sim_coord: If True, return the poses in the simulator coordinate. Otherwise, return the poses in the AprilTag coordinate.

        Returns:
            parts_poses: (num_envs, num_parts * pose_dim). The poses of all parts in the AprilTag frame.
            founds: (num_envs, num_parts). Always 1 since we don't use AprilTag for detection in simulation.
        """
        parts_poses = torch.zeros(
            (self.num_envs, len(self.furniture.parts) * self.pose_dim),
            dtype=torch.float32,
            device=self.device,
        )
        founds = torch.ones(
            (self.num_envs, len(self.furniture.parts)),
            dtype=torch.float32,
            device=self.device,
        )
        if sim_coord:
            # Return the poses in the simulator coordinate.
            for part_idx in range(len(self.furniture.parts)):
                part = self.furniture.parts[part_idx]
                rb_idx = self.part_idxs[part.name]
                part_pose = self.rb_states[rb_idx, :7]
                parts_poses[
                    :, part_idx * self.pose_dim : (part_idx + 1) * self.pose_dim
                ] = part_pose[:, : self.pose_dim]

            return parts_poses, founds

        for env_idx in range(self.num_envs):
            for part_idx in range(len(self.furniture.parts)):
                part = self.furniture.parts[part_idx]
                rb_idx = self.part_idxs[part.name][env_idx]
                part_pose = self.rb_states[rb_idx, :7]
                # To AprilTag coordinate.
                part_pose = torch.concat(
                    [
                        *C.mat2pose(
                            self.sim_coord_to_april_coord(
                                C.pose2mat(
                                    part_pose[:3], part_pose[3:7], device=self.device
                                )
                            )
                        )
                    ]
                )
                parts_poses[
                    env_idx, part_idx * self.pose_dim : (part_idx + 1) * self.pose_dim
                ] = part_pose
        return parts_poses, founds

    def _save_camera_input(self):
        """Saves camera images to png files for debugging."""
        root = "sim_camera"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        Path(root).mkdir(exist_ok=True)

        for cam, handles in self.camera_handles.items():
            self.isaac_gym.write_camera_image_to_file(
                self.sim,
                self.envs[0],
                handles[0],
                gymapi.IMAGE_COLOR,
                f"{root}/{timestamp}_{cam}_sim.png",
            )

            self.isaac_gym.write_camera_image_to_file(
                self.sim,
                self.envs[0],
                handles[0],
                gymapi.IMAGE_DEPTH,
                f"{root}/{timestamp}_{cam}_sim_depth.png",
            )

    def _read_robot_state(self):
        # First robot state
        joint_positions = self.dof_pos[:, :7]
        joint_velocities = self.dof_vel[:, :7]
        joint_torques = self.forces[:, :9]  # First robot's forces (9 DOFs)
        ee_pos, ee_quat = self.get_ee_pose()
        for q in ee_quat:
            if q[3] < 0:
                q *= -1
        ee_pos_vel = self.rb_states[self.ee_idxs, 7:10]
        ee_ori_vel = self.rb_states[self.ee_idxs, 10:]
        gripper_width = self.gripper_width()

        robot_state_dict = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_torques": joint_torques,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "ee_pos_vel": ee_pos_vel,
            "ee_ori_vel": ee_ori_vel,
            "gripper_width": gripper_width,
        }
        return {k: robot_state_dict[k] for k in self.robot_state_keys}

    def refresh(self):
        self.isaac_gym.simulate(self.sim)
        self.isaac_gym.fetch_results(self.sim, True)
        self.isaac_gym.step_graphics(self.sim)

        # Refresh tensors.
        self.isaac_gym.refresh_dof_state_tensor(self.sim)
        self.isaac_gym.refresh_dof_force_tensor(self.sim)
        self.isaac_gym.refresh_rigid_body_state_tensor(self.sim)
        self.isaac_gym.refresh_jacobian_tensors(self.sim)
        self.isaac_gym.refresh_mass_matrix_tensors(self.sim)
        self.isaac_gym.render_all_camera_sensors(self.sim)
        self.isaac_gym.start_access_image_tensors(self.sim)

    def init_ctrl(self):
        # Positional and velocity gains for robot control.
        kp = torch.tensor(sim_config["robot"]["kp"], device=self.device)
        kv = (
            torch.tensor(sim_config["robot"]["kv"], device=self.device)
            if sim_config["robot"]["kv"] is not None
            else torch.sqrt(kp) * 2.0
        )

        ee_pos, ee_quat = self.get_ee_pose()
        for env_idx in range(self.num_envs):
            self.osc_ctrls.append(
                osc_factory(
                    real_robot=False,
                    ee_pos_current=ee_pos[env_idx],
                    ee_quat_current=ee_quat[env_idx],
                    init_joints=torch.tensor(
                        config["robot"]["reset_joints"], device=self.device
                    ),
                    kp=kp,
                    kv=kv,
                    mass_matrix_offset_val=[0.0, 0.0, 0.0],
                    position_limits=torch.tensor(
                        config["robot"]["position_limits"], device=self.device
                    ),
                    joint_kp=10,
                )
            )
        
        # Initialize controllers for second robot
        ee2_pos, ee2_quat = self.get_ee2_pose()
        for env_idx in range(self.num_envs):
            self.osc2_ctrls.append(
                osc_factory(
                    real_robot=False,
                    ee_pos_current=ee2_pos[env_idx],
                    ee_quat_current=ee2_quat[env_idx],
                    init_joints=torch.tensor(
                        config["robot"]["reset_joints"], device=self.device
                    ),
                    kp=kp,
                    kv=kv,
                    mass_matrix_offset_val=[0.0, 0.0, 0.0],
                    position_limits=torch.tensor(
                        config["robot"]["position_limits"], device=self.device
                    ),
                    joint_kp=10,
                )
            )
        self.ctrl_started = True

    def get_ee_pose(self):
        """Gets end-effector pose in world coordinate."""
        hand_pos = self.rb_states[self.ee_idxs, :3]
        hand_quat = self.rb_states[self.ee_idxs, 3:7]
        base_pos = self.rb_states[self.base_idxs, :3]
        base_quat = self.rb_states[self.base_idxs, 3:7]  # Align with world coordinate.
        return hand_pos - base_pos, hand_quat

    def get_ee2_pose(self):
        """Gets end-effector pose for second robot in world coordinate."""
        hand_pos = self.rb_states[self.ee2_idxs, :3]
        hand_quat = self.rb_states[self.ee2_idxs, 3:7]
        base_pos = self.rb_states[self.base2_idxs, :3]
        base_quat = self.rb_states[self.base2_idxs, 3:7]  # Align with world coordinate.
        return hand_pos - base_pos, hand_quat

    def gripper_width(self):
        # Return first robot's gripper width (for backward compatibility)
        return self.dof_pos[:, 7:8] + self.dof_pos[:, 8:9]

    def _done(self) -> bool:
        dones = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        if self.manual_done:
            return dones
        for env_idx in range(self.num_envs):
            timeout = self.env_steps[env_idx] > self.furniture.max_env_steps
            if self.furnitures[env_idx].all_assembled() or timeout:
                dones[env_idx] = 1
                if timeout:
                    gym.logger.warn(f"[env] env_idx: {env_idx} timeout")
        if self.np_step_out:
            dones = dones.cpu().numpy().astype(bool)
        return dones

    def _get_color_obs(self, color_obs):
        color_obs = torch.stack(color_obs)[..., :-1]  # RGBA -> RGB
        if self.channel_first:
            color_obs = color_obs.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return color_obs

    def get_front_projection_view_matrix(self):
        cam_pos = self.front_cam_pos
        cam_target = self.front_cam_target
        width = self.img_size[0]
        height = self.img_size[1]
        near_plane = self.camera_cfg.near_plane
        far_plane = self.camera_cfg.far_plane
        horizontal_fov = self.camera_cfg.horizontal_fov

        # Compute aspect ratio
        aspect_ratio = width / height
        # Convert horizontal FOV from degrees to radians and calculate focal length
        fov_rad = np.radians(horizontal_fov)
        f = 1 / np.tan(fov_rad / 2)
        # Construct the projection matrix
        # fmt: off
        P = np.array(
            [
                [f / aspect_ratio, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far_plane + near_plane) / (near_plane - far_plane), (2 * far_plane * near_plane) / (near_plane - far_plane)],
                [0, 0, -1, 0],
            ]
        )
        # fmt: on

        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v

        forward = normalize(cam_target - cam_pos)
        up = np.array([0, 1, 0])
        right = normalize(np.cross(up, forward))
        # Recompute Up Vector
        up = np.cross(forward, right)

        # Construct the View Matrix
        # fmt: off
        V = np.matrix(
            [
                [right[0], right[1], right[2], -np.dot(right, cam_pos)],
                [up[0], up[1], up[2], -np.dot(up, cam_pos)],
                [forward[0], forward[1], forward[2], -np.dot(forward, cam_pos)],
                [0, 0, 0, 1],
            ]
        )
        # fmt: on

        return P, V

    def _get_observation(self):
        robot_state = self._read_robot_state()
        color_obs = {
            k: self._get_color_obs(v)
            for k, v in self.camera_obs.items()
            if "color" in k
        }
        depth_obs = {
            k: torch.stack(v) for k, v in self.camera_obs.items() if "depth" in k
        }

        if self.np_step_out:
            robot_state = {k: v.cpu().numpy() for k, v in robot_state.items()}
            color_obs = {k: v.cpu().numpy() for k, v in color_obs.items()}
            depth_obs = {k: v.cpu().numpy() for k, v in depth_obs.items()}

        if robot_state and self.concat_robot_state:
            if self.np_step_out:
                robot_state = np.concatenate(list(robot_state.values()), -1)
            else:
                robot_state = torch.cat(list(robot_state.values()), -1)

        if self.record:
            record_images = []
            for k in sorted(color_obs.keys()):
                img = color_obs[k][0]
                if not self.np_step_out:
                    img = img.cpu().numpy().copy()
                if self.channel_first:
                    img = img.transpose(0, 2, 3, 1)
                record_images.append(img.squeeze())
            stacked_img = np.hstack(record_images)
            self.video_writer.write(cv2.cvtColor(stacked_img, cv2.COLOR_RGB2BGR))

        obs = {}
        if (
            isinstance(robot_state, (np.ndarray, torch.Tensor)) or robot_state
        ):  # Check if robot_state is empty.
            if self.robot_state_as_dict:
                obs["robot_state"] = robot_state
            else:
                obs.update(robot_state)  # Flatten the dict.
        for k in self.obs_keys:
            if k == "parts_poses":
                parts_poses, _ = self._get_parts_poses()  # Part poses in AprilTag coordinate.
                if self.np_step_out:
                    parts_poses = parts_poses.cpu().numpy()
                obs["parts_poses"] = parts_poses
            elif k.startswith("color"):
                obs[k] = color_obs[k]
            elif k.startswith("depth"):
                obs[k] = depth_obs[k]

        if self.squeeze_batch_dim:
            for k, v in obs.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        obs[k][kk] = vv.squeeze(0)
                else:
                    obs[k] = v.squeeze(0)
        return obs

    def get_observation(self):
        return self._get_observation()

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError
        return self._get_observation()["color_image2"]

    def is_success(self):
        return [{"task": self.furnitures[env_idx].all_assembled()} for env_idx in range(self.num_envs)]

    def reset(self):
        # can also reset the full set of robots/parts, without applying torques and refreshing
        # self._reset_franka_all()
        # self._reset_parts_all()
        for i in range(self.num_envs):
            # if using ._reset_*_all(), can set reset_franka=False and reset_parts=False in .reset_env
            self.reset_env(i)  

            # apply zero torque across the board and refresh in between each env reset (not needed if using ._reset_*_all())
            torque_action = torch.zeros_like(self.dof_pos)
            self.isaac_gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(torque_action)
            )
            self.refresh()
        
        self.furniture.reset()

        self.refresh()
        self.assemble_idx = 0
        self.robot1_retreated = False  # Reset retreat flag
        
        # Reset round_table specific variables
        self.round_table_fsm_stage = 0
        self.round_table_arm_assignments = {}
        self.round_table_stage1_complete = {'arm1': False, 'arm2': False}
        if hasattr(self, 'round_table_arm2_action'):
            delattr(self, 'round_table_arm2_action')

        if self.save_camera_input:
            self._save_camera_input()

        return self._get_observation()

    def reset_to(self, state):
        """Reset to a specific state.

        Args:
            state: List of observation dictionary for each environment.
        """
        for i in range(self.num_envs):
            self.reset_env_to(i, state[i])

    def reset_env(self, env_idx, reset_franka=True, reset_parts=True):
        """Resets the environment. **MUST refresh in between multiple calls
        to this function to have changes properly reflected in each environment.
        Also might want to set a zero-torque action via .set_dof_actuation_force_tensor
        to avoid additional movement**

        Args:
            env_idx: Environment index.
            reset_franka: If True, then reset the franka for this env
            reset_parts: If True, then reset the part poses for this env
        """
        self.furnitures[env_idx].reset()
        if self.randomness == Randomness.LOW and not self.init_assembled:
            self.furnitures[env_idx].randomize_init_pose(
                self.from_skill, pos_range=[-0.015, 0.015], rot_range=15
            )

        if self.randomness == Randomness.MEDIUM:
            self.furnitures[env_idx].randomize_init_pose(self.from_skill)
        elif self.randomness == Randomness.HIGH:
            self.furnitures[env_idx].randomize_high(self.high_random_idx)
        
        if reset_franka:
            self._reset_franka(env_idx)
            self._reset_franka2(env_idx)  # Reset second robot
        if reset_parts:
            self._reset_parts(env_idx)
        self.env_steps[env_idx] = 0
        self.move_neutral = False
        self.robot1_retreated = False  # Reset retreat flag
        
        # Reset round_table specific variables
        self.round_table_fsm_stage = 0
        if env_idx in self.round_table_arm_assignments:
            del self.round_table_arm_assignments[env_idx]
        self.round_table_stage1_complete = {'arm1': False, 'arm2': False}
        if hasattr(self, 'round_table_arm2_action'):
            delattr(self, 'round_table_arm2_action')

    def reset_env_to(self, env_idx, state):
        """Reset to a specific state. **MUST refresh in between multiple calls
        to this function to have changes properly reflected in each environment.
        Also might want to set a zero-torque action via .set_dof_actuation_force_tensor
        to avoid additional movement**

        Args:
            env_idx: Environment index.
            state: A dict containing the state of the environment.
        """
        self.furnitures[env_idx].reset()
        dof_pos = np.concatenate(
            [
                state["robot_state"]["joint_positions"],
                np.array([state["robot_state"]["gripper_width"] / 2] * 2),
            ],
        )
        self._reset_franka(env_idx, dof_pos)
        self._reset_parts(env_idx, state["parts_poses"])
        self.env_steps[env_idx] = 0
        self.move_neutral = False

    def _update_franka_dof_state_buffer(self, dof_pos=None):
        """
        Sets internal tensor state buffer for Franka actor 
        """
        # Low randomness only.
        if self.from_skill >= 1:
            dof_pos = torch.from_numpy(self.default_dof_pos)
            ee_pos = torch.from_numpy(
                self.furniture.furniture_conf["ee_pos"][self.from_skill]
            )
            ee_quat = torch.from_numpy(
                self.furniture.furniture_conf["ee_quat"][self.from_skill]
            )
            dof_pos = self.robot_model.inverse_kinematics(ee_pos, ee_quat)
        else:
            dof_pos = self.default_dof_pos if dof_pos is None else dof_pos
        
        # Views for self.dof_states (used with set_dof_state_tensor* function)
        self.dof_pos[:, 0 : self.franka_num_dofs] = torch.tensor(
            dof_pos, device=self.device, dtype=torch.float32
        )
        self.dof_vel[:, 0 : self.franka_num_dofs] = torch.tensor(
            [0] * len(self.default_dof_pos), device=self.device, dtype=torch.float32
        )

    def _reset_franka(self, env_idx, dof_pos=None):
        """
        Resets Franka actor within a single env. If calling multiple times,
        need to refresh in between calls to properly register individual env changes, 
        and set zero torques on frankas across all envs to prevent the reset arms
        from moving while others are still being reset
        """
        self._update_franka_dof_state_buffer(dof_pos=dof_pos)
        
        # Update a single actor 
        actor_idx = self.franka_actor_idxs_all_t[env_idx].reshape(1, 1)
        self.isaac_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(actor_idx),
            len(actor_idx),
        )

    def _reset_franka2(self, env_idx, dof_pos=None):
        """
        Resets second Franka actor within a single env.
        """
        self._update_franka_dof_state_buffer(dof_pos=dof_pos)
        
        # Update second robot's DOF states (DOFs 9-17 for this env)
        # Note: DOF states are organized as (env_idx * num_dofs_per_env + robot_offset + dof_idx)
        # For now, we'll update the DOF states for the second robot
        actor_idx = self.franka2_actor_idxs_all_t[env_idx].reshape(1, 1)
        self.isaac_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(actor_idx),
            len(actor_idx),
        )

    def _reset_franka_all(self, dof_pos=None):
        """
        Resets all Franka actors across all envs
        """
        self._update_franka_dof_state_buffer(dof_pos=dof_pos)

        # Update all actors across envs at once
        self.isaac_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(self.franka_actor_idxs_all_t),
            len(self.franka_actor_idxs_all_t),
        )
        
        # Also reset all second robots
        self.isaac_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(self.franka2_actor_idxs_all_t),
            len(self.franka2_actor_idxs_all_t),
        )

    def _reset_parts(self, env_idx, parts_poses=None, skip_set_state=False):
        """Resets furniture parts to the initial pose.

        Args:
            env_idx (int): The index of the environment.
            parts_poses (np.ndarray): The poses of the parts. If None, the parts will be reset to the initial pose.
        """
        for part_idx, part in enumerate(self.furnitures[env_idx].parts):
            # Use the given pose.
            if parts_poses is not None:
                part_pose = parts_poses[part_idx * 7 : (part_idx + 1) * 7]

                pos = part_pose[:3]
                ori = T.to_homogeneous(
                    [0, 0, 0], T.quat2mat(part_pose[3:])
                )  # Dummy zero position.
            else:
                pos, ori = self._get_reset_pose(part)

            part_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
            part_pose = gymapi.Transform()
            part_pose.p = gymapi.Vec3(
                part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3]
            )
            reset_ori = self.april_coord_to_sim_coord(ori)
            part_pose.r = gymapi.Quat(*T.mat2quat(reset_ori[:3, :3]))
            idxs = self.parts_handles[part.name]
            idxs = torch.tensor(idxs, device=self.device, dtype=torch.int32)

            self.root_pos[env_idx, idxs] = torch.tensor(
                [part_pose.p.x, part_pose.p.y, part_pose.p.z], device=self.device
            )
            self.root_quat[env_idx, idxs] = torch.tensor(
                [part_pose.r.x, part_pose.r.y, part_pose.r.z, part_pose.r.w],
                device=self.device,
            )

        if skip_set_state:
            # Set the value for the root state tensor, but don't call isaac gym function yet (useful when resetting all at once)
            # If skip_set_state == True, then must self.refresh() to register the isaac set_actor_root_state* function
            return

        # Reset root state for actors in a single env
        part_actor_idxs = torch.tensor(self.part_actor_idx_by_env[env_idx], device=self.device, dtype=torch.int32)
        self.isaac_gym.get_sim_actor_count(self.sim)
        self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(part_actor_idxs),
            len(part_actor_idxs),
        )

    def _reset_parts_all(self, parts_poses=None):
        """Resets ALL furniture parts to the initial pose.

        Args:
            parts_poses (np.ndarray): The poses of the parts. If None, the parts will be reset to the initial pose.
        """
        for env_idx in range(self.num_envs):
            self._reset_parts(env_idx, parts_poses=parts_poses, skip_set_state=True)

        # Reset root state for actors across all envs
        self.isaac_gym.get_sim_actor_count(self.sim)
        self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(self.part_actor_idxs_all_t),
            len(self.part_actor_idxs_all_t),
        )

    def _import_base_tag_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        base_asset_file = "furniture/urdf/base_tag.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, base_asset_file, asset_options
        )

    def _import_obstacle_front_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        obstacle_asset_file = "furniture/urdf/obstacle_front.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, obstacle_asset_file, asset_options
        )

    def _import_obstacle_side_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        obstacle_asset_file = "furniture/urdf/obstacle_side.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, obstacle_asset_file, asset_options
        )

    def _import_background_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        background_asset_file = "furniture/urdf/background.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, background_asset_file, asset_options
        )

    def _import_table_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset_file = "furniture/urdf/table.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, table_asset_file, asset_options
        )

    def _import_franka_asset(self):
        self.franka_asset_file = (
            "franka_description_ros/franka_description/robots/franka_panda.urdf"
        )
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, self.franka_asset_file, asset_options
        )

    def _assign_round_table_parts_by_proximity(self, env_idx=0):
        """Assign RoundTableLeg and RoundTableBase to arms based on spatial proximity.
        
        Returns:
            dict: {'arm1': 'part_name', 'arm2': 'part_name'}
        """
        # Get arm positions
        ee1_pos, _ = self.get_ee_pose()
        ee2_pos, _ = self.get_ee2_pose()
        ee1_pos = ee1_pos[env_idx]
        ee2_pos = ee2_pos[env_idx]
        
        # Get part positions
        leg_idx = self.part_idxs['round_table_leg'][env_idx]
        base_idx = self.part_idxs['round_table_base'][env_idx]
        
        leg_pos = self.rb_states[leg_idx, :3]
        base_pos = self.rb_states[base_idx, :3]
        
        # Calculate distances
        arm1_to_leg = torch.norm(ee1_pos - leg_pos)
        arm1_to_base = torch.norm(ee1_pos - base_pos)
        arm2_to_leg = torch.norm(ee2_pos - leg_pos)
        arm2_to_base = torch.norm(ee2_pos - base_pos)
        
        # Assign based on closest distance
        # We need to ensure each part is assigned to exactly one arm
        if arm1_to_leg + arm2_to_base < arm1_to_base + arm2_to_leg:
            # Arm 1 takes leg, Arm 2 takes base
            assignment = {'arm1': 'round_table_leg', 'arm2': 'round_table_base'}
        else:
            # Arm 1 takes base, Arm 2 takes leg
            assignment = {'arm1': 'round_table_base', 'arm2': 'round_table_leg'}
        
        print(f"************* Part Assignment *************")
        print(f"Arm 1: {assignment['arm1']}")
        print(f"Arm 2: {assignment['arm2']}")
        print(f"Distances - Arm1->Leg: {arm1_to_leg:.3f}, Arm1->Base: {arm1_to_base:.3f}")
        print(f"            Arm2->Leg: {arm2_to_leg:.3f}, Arm2->Base: {arm2_to_base:.3f}")
        
        return assignment
    
    def _get_round_table_assembly_action(self):
        """Get assembly action for round_table with spatial assignment logic."""
        env_idx = 0  # Only support one environment for now
        
        # Stage 0: Initialize and assign parts based on proximity
        if self.round_table_fsm_stage == 0:
            self.round_table_arm_assignments[env_idx] = self._assign_round_table_parts_by_proximity(env_idx)
            self.round_table_fsm_stage = 1
            print("************* Stage 0->1: Parts assigned, starting parallel grasp *************")
        
        # Stage 1: Parallel grasp - both arms pick up their assigned parts
        if self.round_table_fsm_stage == 1:
            # Get both arms' poses
            ee1_pos, ee1_quat = self.get_ee_pose()
            ee2_pos, ee2_quat = self.get_ee2_pose()
            gripper1_width = self.gripper_width()
            gripper2_width = self.dof_pos[:, 16:17] + self.dof_pos[:, 17:18]
            
            ee1_pos, ee1_quat = ee1_pos.squeeze(), ee1_quat.squeeze()
            ee2_pos, ee2_quat = ee2_pos.squeeze(), ee2_quat.squeeze()
            
            # Get the parts assigned to each arm
            arm1_part_name = self.round_table_arm_assignments[env_idx]['arm1']
            arm2_part_name = self.round_table_arm_assignments[env_idx]['arm2']
            
            # Get the part objects
            arm1_part = None
            arm2_part = None
            for part in self.furniture.parts:
                if part.name == arm1_part_name:
                    arm1_part = part
                elif part.name == arm2_part_name:
                    arm2_part = part
            
            # Check if both arms have completed grasping
            arm1_done = arm1_part.pre_assemble_done if arm1_part else False
            arm2_done = arm2_part.pre_assemble_done if arm2_part else False
            
            if arm1_done and arm2_done:
                # Stage 1 complete, move to stage 2
                self.round_table_fsm_stage = 2
                print("************* Stage 1->2: Parallel grasp complete, starting leg assembly *************")
                # Determine which arm holds the leg for stage 2
                self.leg_holding_arm = 1 if arm1_part_name == 'round_table_leg' else 2
                
                # Reset leg part state for assembly phase (skip grasp states)
                leg_part = None
                for part in self.furniture.parts:
                    if part.name == 'round_table_leg':
                        leg_part = part
                        break
                if leg_part:
                    # Reset to assembly starting state (after grasp)
                    leg_part._state = "move_center"
                    # Keep prev_pose - it's needed for move_center state
                
                return self._get_round_table_assembly_action()  # Recursively call for stage 2
            
            # Generate actions for both arms
            goal_pos1, goal_ori1, gripper1, skill_complete1 = arm1_part.pre_assemble(
                ee1_pos, ee1_quat, gripper1_width,
                self.rb_states, self.part_idxs,
                self.sim_to_april_mat, self.april_to_robot_mat
            )
            
            goal_pos2, goal_ori2, gripper2, skill_complete2 = arm2_part.pre_assemble(
                ee2_pos, ee2_quat, gripper2_width,
                self.rb_states, self.part_idxs,
                self.sim_to_april_mat, self.april_to_robot2_mat
            )
            
            # Store arm2 action for step() to use
            delta_pos2 = goal_pos2 - ee2_pos
            delta_quat2 = C.quat_mul(C.quat_conjugate(ee2_quat), goal_ori2)
            self.round_table_arm2_action = torch.concat([delta_pos2, delta_quat2, gripper2])
            
            # Return arm1 action
            delta_pos1 = goal_pos1 - ee1_pos
            delta_quat1 = C.quat_mul(C.quat_conjugate(ee1_quat), goal_ori1)
            action1 = torch.concat([delta_pos1, delta_quat1, gripper1])
            
            return action1.unsqueeze(0), max(skill_complete1, skill_complete2)
        
        # Stage 2: Arm holding leg assembles it to top, then retreats
        if self.round_table_fsm_stage == 2:
            if self.leg_holding_arm == 1:
                ee_pos, ee_quat = self.get_ee_pose()
                gripper_width = self.gripper_width()
                april_to_robot = self.april_to_robot_mat
            else:
                ee_pos, ee_quat = self.get_ee2_pose()
                gripper_width = self.dof_pos[:, 16:17] + self.dof_pos[:, 17:18]
                april_to_robot = self.april_to_robot2_mat
            
            ee_pos, ee_quat = ee_pos.squeeze(), ee_quat.squeeze()
            
            # Get leg part
            leg_part = None
            for part in self.furniture.parts:
                if part.name == 'round_table_leg':
                    leg_part = part
            
            # Check if assembly is complete
            leg_pose = C.to_homogeneous(
                self.rb_states[self.part_idxs['round_table_leg']][0][:3],
                C.quat2mat(self.rb_states[self.part_idxs['round_table_leg']][0][3:7]),
            )
            top_pose = C.to_homogeneous(
                self.rb_states[self.part_idxs['round_table_top']][0][:3],
                C.quat2mat(self.rb_states[self.part_idxs['round_table_top']][0][3:7]),
            )
            rel_pose = torch.linalg.inv(top_pose) @ leg_pose
            assembled_rel_poses = self.furniture.assembled_rel_poses[(0, 1)]
            
            # Debug: print assembly progress every 50 steps
            if self.env_steps[0] % 50 == 0:
                pos_error = torch.norm(rel_pose[:3, 3] - torch.tensor(assembled_rel_poses[0][:3, 3], device=self.device))
                print(f"[Stage 2 Debug] Leg assembly progress - Position error: {pos_error:.4f}m, Current state: {leg_part._state}")
            
            if self.furniture.assembled(rel_pose.cpu().numpy(), assembled_rel_poses):
                if not self.robot1_retreated:
                    # Start retreat for leg-holding arm
                    print("************* Stage 2: Leg assembled, starting retreat *************")
                    # Generate retreat action
                    if self.leg_holding_arm == 1:
                        safe_pos = torch.tensor([0.3, -0.2, 0.25], device=self.device)
                    else:
                        safe_pos = torch.tensor([0.3, 0.2, 0.25], device=self.device)
                    
                    safe_quat = torch.tensor([0, 0.707, 0, 0.707], device=self.device)
                    
                    if torch.norm(ee_pos - safe_pos) < 0.03:
                        self.robot1_retreated = True
                        self.round_table_fsm_stage = 3
                        print("************* Stage 2->3: Retreat complete, starting base assembly *************")
                        return self._get_round_table_assembly_action()
                    
                    delta_pos = safe_pos - ee_pos
                    delta_quat = C.quat_mul(C.quat_conjugate(ee_quat), safe_quat)
                    gripper = torch.tensor([-1], dtype=torch.float32, device=self.device)
                    action = torch.concat([delta_pos, delta_quat, gripper])
                    return action.unsqueeze(0), 0
            
            # Assemble leg to top
            goal_pos, goal_ori, gripper, skill_complete = leg_part.fsm_step(
                ee_pos, ee_quat, gripper_width,
                self.rb_states, self.part_idxs,
                self.sim_to_april_mat, april_to_robot,
                'round_table_top'
            )
            
            delta_pos = goal_pos - ee_pos
            delta_quat = C.quat_mul(C.quat_conjugate(ee_quat), goal_ori)
            action = torch.concat([delta_pos, delta_quat, gripper])
            
            return action.unsqueeze(0), skill_complete
        
        # Stage 3: Arm holding base assembles it to leg
        if self.round_table_fsm_stage == 3:
            base_holding_arm = 2 if self.leg_holding_arm == 1 else 1
            
            if base_holding_arm == 1:
                ee_pos, ee_quat = self.get_ee_pose()
                gripper_width = self.gripper_width()
                april_to_robot = self.april_to_robot_mat
            else:
                ee_pos, ee_quat = self.get_ee2_pose()
                gripper_width = self.dof_pos[:, 16:17] + self.dof_pos[:, 17:18]
                april_to_robot = self.april_to_robot2_mat
            
            ee_pos, ee_quat = ee_pos.squeeze(), ee_quat.squeeze()
            
            # Get base part and reset its state for assembly
            base_part = None
            first_call_stage3 = False
            for part in self.furniture.parts:
                if part.name == 'round_table_base':
                    base_part = part
                    # Reset to assembly starting state if coming from pre_assemble
                    if base_part._state == "done":
                        # Base is at safe position, need to move to assembly position
                        base_part._state = "move_to_assembly"
                        first_call_stage3 = True
                        # Keep prev_pose (safe position)
            
            # Check if assembly is complete
            leg_pose = C.to_homogeneous(
                self.rb_states[self.part_idxs['round_table_leg']][0][:3],
                C.quat2mat(self.rb_states[self.part_idxs['round_table_leg']][0][3:7]),
            )
            base_pose = C.to_homogeneous(
                self.rb_states[self.part_idxs['round_table_base']][0][:3],
                C.quat2mat(self.rb_states[self.part_idxs['round_table_base']][0][3:7]),
            )
            rel_pose = torch.linalg.inv(leg_pose) @ base_pose
            assembled_rel_poses = self.furniture.assembled_rel_poses[(1, 2)]
            
            if self.furniture.assembled(rel_pose.cpu().numpy(), assembled_rel_poses):
                print("************* Stage 3: Base assembled, assembly complete! *************")
                self.round_table_fsm_stage = 4
                return torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=self.device).unsqueeze(0), 1
            
            # Assemble base to leg
            goal_pos, goal_ori, gripper, skill_complete = base_part.fsm_step(
                ee_pos, ee_quat, gripper_width,
                self.rb_states, self.part_idxs,
                self.sim_to_april_mat, april_to_robot,
                'round_table_leg'
            )
            
            delta_pos = goal_pos - ee_pos
            delta_quat = C.quat_mul(C.quat_conjugate(ee_quat), goal_ori)
            action = torch.concat([delta_pos, delta_quat, gripper])
            
            return action.unsqueeze(0), skill_complete
        
        # All stages complete
        return torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=self.device).unsqueeze(0), 1
    
    def get_assembly_action(self) -> torch.Tensor:
        """Scripted furniture assembly logic.

        Returns:
            Tuple (action for the assembly task, skill complete mask)
        """
        print ('************* Hello get_assembly_action *************')
        assert self.num_envs == 1  # Only support one environment for now.
        if self.furniture_name not in ["one_leg", "cabinet", "lamp", "round_table"]:
            raise NotImplementedError("[one_leg, cabinet, lamp, round_table] are supported for scripted agent")

        # Handle round_table with new spatial assignment logic
        if self.furniture_name == "round_table":
            return self._get_round_table_assembly_action()
        
        if self.assemble_idx >= len(self.furniture.should_be_assembled):
            return torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=self.device)

        # For round_table, handle robot 1 retreat before robot 2 starts
        if self.furniture_name == "round_table" and self.assemble_idx == 1 and not self.robot1_retreated:
            # Robot 1 needs to retreat to safe position before robot 2 starts
            ee_pos, ee_quat = self.get_ee_pose()
            ee_pos, ee_quat = ee_pos.squeeze(), ee_quat.squeeze()
            
            # Safe retreat position: high and away from work area
            safe_pos = torch.tensor([0.3, -0.2, 0.25], device=self.device)
            safe_quat = torch.tensor([0, 0.707, 0, 0.707], device=self.device)  # Pointing up
            
            # Check if robot 1 has reached safe position
            pos_error = torch.norm(ee_pos - safe_pos)
            if pos_error < 0.03:  # 3cm threshold
                self.robot1_retreated = True
                print("************* Robot 1 retreated to safe position *************")
            
            # Generate action to move to safe position
            delta_pos = safe_pos - ee_pos
            delta_quat = C.quat_mul(C.quat_conjugate(ee_quat), safe_quat)
            gripper = torch.tensor([-1], dtype=torch.float32, device=self.device)  # Open gripper
            action = torch.concat([delta_pos, delta_quat, gripper])
            return action.unsqueeze(0), 0
        
        # For round_table, determine which robot should be active
        if self.furniture_name == "round_table":
            # assembly_idx 0: (0,1) - robot 1 assembles leg to top
            # assembly_idx 1: (1,2) - robot 2 assembles base to leg
            active_robot = 1 if self.assemble_idx == 0 else 2
            
            # Get the appropriate end-effector pose and gripper width
            if active_robot == 1:
                ee_pos, ee_quat = self.get_ee_pose()
                gripper_width = self.gripper_width()
                april_to_robot = self.april_to_robot_mat
            else:  # active_robot == 2
                ee_pos, ee_quat = self.get_ee2_pose()
                gripper_width = self.dof_pos[:, 16:17] + self.dof_pos[:, 17:18]
                april_to_robot = self.april_to_robot2_mat
            ee_pos, ee_quat = ee_pos.squeeze(), ee_quat.squeeze()
        else:
            # For lamp and other furniture, use robot 1
            ee_pos, ee_quat = self.get_ee_pose()
            gripper_width = self.gripper_width()
            ee_pos, ee_quat = ee_pos.squeeze(), ee_quat.squeeze()
            april_to_robot = self.april_to_robot_mat

        if self.move_neutral:
            if ee_pos[2] <= 0.15 - 0.01:
                print ('************* move_neutral True *************')
                gripper = torch.tensor([-1], dtype=torch.float32, device=self.device)
                goal_pos = torch.tensor(
                    [ee_pos[0], ee_pos[1], 0.15], device=self.device
                )
                delta_pos = goal_pos - ee_pos
                delta_quat = torch.tensor([0, 0, 0, 1], device=self.device)
                action = torch.concat([delta_pos, delta_quat, gripper])
                return action.unsqueeze(0), 0
            else:
                self.move_neutral = False
        
        part_idx1, part_idx2 = self.furniture.should_be_assembled[self.assemble_idx]

        part1 = self.furniture.parts[part_idx1]
        part1_name = self.furniture.parts[part_idx1].name
        part1_pose = C.to_homogeneous(
            self.rb_states[self.part_idxs[part1_name]][0][:3],
            C.quat2mat(self.rb_states[self.part_idxs[part1_name]][0][3:7]),
        )
        part2 = self.furniture.parts[part_idx2]
        part2_name = self.furniture.parts[part_idx2].name
        part2_pose = C.to_homogeneous(
            self.rb_states[self.part_idxs[part2_name]][0][:3],
            C.quat2mat(self.rb_states[self.part_idxs[part2_name]][0][3:7]),
        )
        rel_pose = torch.linalg.inv(part1_pose) @ part2_pose
        assembled_rel_poses = self.furniture.assembled_rel_poses[(part_idx1, part_idx2)]
        if self.furniture.assembled(rel_pose.cpu().numpy(), assembled_rel_poses):
            self.assemble_idx += 1
            self.move_neutral = True
            return (
                torch.tensor(
                    [0, 0, 0, 0, 0, 0, 1, -1], dtype=torch.float32, device=self.device
                ).unsqueeze(0),
                1,
            )  # Skill complete is always 1 when assembled.
        if not part1.pre_assemble_done:
            print ('************* pre-assemble part1 *************')
            goal_pos, goal_ori, gripper, skill_complete = part1.pre_assemble(
                ee_pos,
                ee_quat,
                gripper_width,
                self.rb_states,
                self.part_idxs,
                self.sim_to_april_mat,
                april_to_robot,
            )
        elif not part2.pre_assemble_done:
            print ('************* pre-assemble part2 *************')
            goal_pos, goal_ori, gripper, skill_complete = part2.pre_assemble(
                ee_pos,
                ee_quat,
                gripper_width,
                self.rb_states,
                self.part_idxs,
                self.sim_to_april_mat,
                april_to_robot,
            )
        else:
            # print ('************* last step *************')
            goal_pos, goal_ori, gripper, skill_complete = self.furniture.parts[
                part_idx2
            ].fsm_step(
                ee_pos,
                ee_quat,
                gripper_width,
                self.rb_states,
                self.part_idxs,
                self.sim_to_april_mat,
                april_to_robot,
                self.furniture.parts[part_idx1].name,
            )

        delta_pos = goal_pos - ee_pos

        # Scale translational action.
        delta_pos_sign = delta_pos.sign()
        delta_pos = torch.abs(delta_pos) * 2
        for i in range(3):
            if delta_pos[i] > 0.03:
                delta_pos[i] = 0.03 + (delta_pos[i] - 0.03) * np.random.normal(1.5, 0.1)
        delta_pos = delta_pos * delta_pos_sign

        # Clamp too large action.
        max_delta_pos = 0.11 + 0.01 * torch.rand(3, device=self.device)
        max_delta_pos[2] -= 0.04
        delta_pos = torch.clamp(delta_pos, min=-max_delta_pos, max=max_delta_pos)

        delta_quat = C.quat_mul(C.quat_conjugate(ee_quat), goal_ori)
        # Add random noise to the action.
        if (
            self.furniture.parts[part_idx2].state_no_noise()
            and np.random.random() < 0.50
        ):
            delta_pos = torch.normal(delta_pos, 0.005)
            delta_quat = C.quat_multiply(
                delta_quat,
                torch.tensor(
                    T.axisangle2quat(
                        [
                            np.radians(np.random.normal(0, 5)),
                            np.radians(np.random.normal(0, 5)),
                            np.radians(np.random.normal(0, 5)),
                        ]
                    ),
                    device=self.device,
                ),
            ).to(self.device)
        action = torch.concat([delta_pos, delta_quat, gripper])
        return action.unsqueeze(0), skill_complete

    def assembly_success(self):
        print ('************* assembly_success *************')
        return self._done().squeeze()

    def __del__(self):
        if not self.headless:
            self.isaac_gym.destroy_viewer(self.viewer)
        self.isaac_gym.destroy_sim(self.sim)

        if self.record:
            self.video_writer.release()


class FurnitureSimFullEnv(FurnitureSimEnv):
    """FurnitureSim environment with all available observations."""

    def __init__(self, **kwargs):
        super().__init__(obs_keys=FULL_OBS, **kwargs)


class FurnitureSimStateEnv(FurnitureSimEnv):
    """FurnitureSim environment with state observations."""

    def __init__(self, **kwargs):
        obs_keys = DEFAULT_STATE_OBS
        super().__init__(obs_keys=obs_keys, concat_robot_state=True, **kwargs)
