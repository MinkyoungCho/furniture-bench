# run python -m furniture_bench.scripts.collect_data --furniture one_leg --is-sim --input-device xbox

import gym
import numpy as np
import pygame
import time

from furniture_bench.device.device_interface import DeviceInterface
from furniture_bench.data.collect_enum import CollectEnum
import furniture_bench.utils.transform as T

class XboxInterface(DeviceInterface):
    """
    Continuous 6-DoF + gripper teleop using an Xbox controller.
    Uses pygame for real-time analog stick polling.
    """

    INIT_POS_DELTA = 0.01
    INIT_ROT_DELTA = 0.13
    MAX_POS_DELTA = 0.1
    MAX_ROT_DELTA = 0.2

    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No Xbox controller detected!")

        self.js = pygame.joystick.Joystick(0)
        self.js.init()

        # internal control state
        self.reset()

    def reset(self):
        self.pos_delta = XboxInterface.INIT_POS_DELTA
        self.rot_delta = XboxInterface.INIT_ROT_DELTA

        self.pos = np.zeros(3)
        self.last_pos = self.pos.copy()

        self.ori = np.zeros(3)
        self.last_ori = self.ori.copy()

        # -1 or +1 consistent with KeyboardInterface
        self.grasp = np.array([-1])

        self.key_enum = CollectEnum.DONE_FALSE
        self.prev_buttons = np.zeros(16)  # track edge events
        time.sleep(0.2)

    # -------- Utility --------
    def _button_pressed(self, idx):
        """Detect rising edge of a button."""
        cur = self.js.get_button(idx)
        prev = self.prev_buttons[idx]
        self.prev_buttons[idx] = cur
        return cur == 1 and prev == 0

    # -------- Main Input Update --------
    def _update_from_controller(self):
        pygame.event.pump()  # refresh controller state

        # ---------- Left stick (axes 0,1) → X/Y ----------
        lx = self.js.get_axis(0)   # left stick left/right
        ly = self.js.get_axis(1)   # left stick up/down

        d_x = -ly * self.pos_delta     # forward/back maps to +/- X
        d_y = lx * self.pos_delta      # left/right maps to +/- Y

        # ---------- Bumpers → Z ----------
        lb = self.js.get_button(4)
        rb = self.js.get_button(5)

        d_z = (rb - lb) * self.pos_delta

        # ---------- Right stick (axes 3,4) → Pitch/Yaw ----------
        rx = self.js.get_axis(3)   # yaw
        ry = self.js.get_axis(4)   # pitch

        d_pitch = -ry * self.rot_delta
        d_yaw = rx * self.rot_delta

        # ---------- Triggers → Roll ----------
        # Many Xbox drivers use axes 2 (LT) and 5 (RT) in range [-1,1]
        lt = (self.js.get_axis(2) + 1) / 2   # 0 → 1
        rt = (self.js.get_axis(5) + 1) / 2   # 0 → 1
        d_roll = (rt - lt) * self.rot_delta

        # ---------- Apply ----------
        self.pos += np.array([d_x, d_y, d_z])
        self.ori += np.array([d_roll, d_pitch, d_yaw])

        # ---------- Gripper ----------
        if self._button_pressed(0):   # A
            self.grasp = np.array([-1])   # close
        if self._button_pressed(1):   # B
            self.grasp = np.array([1])    # open

        # ---------- Speed Scale ----------
        hat_x, hat_y = self.js.get_hat(0)

        if hat_y == -1:   # D-pad down
            self.pos_delta *= 0.9
            self.rot_delta *= 0.9
        elif hat_y == 1:  # D-pad up
            self.pos_delta *= 1.1
            self.rot_delta *= 1.1

        # clamp
        self.pos_delta = np.clip(self.pos_delta, 0.001, XboxInterface.MAX_POS_DELTA)
        self.rot_delta = np.clip(self.rot_delta, 0.001, XboxInterface.MAX_ROT_DELTA)

    # -------- Output (same API as KeyboardInterface) --------
    def get_action(self, use_quat=True):
        self._update_from_controller()

        dpos = self.pos - self.last_pos
        dori = self.ori - self.last_ori

        self.last_pos = self.pos.copy()
        self.last_ori = self.ori.copy()

        if use_quat:
            dquat = T.mat2quat(T.euler2mat(dori))
            action = np.concatenate([dpos, dquat, self.grasp])
        else:
            action = np.concatenate([dpos, dori, self.grasp])

        key = self.key_enum
        self.key_enum = CollectEnum.DONE_FALSE
        return action, key

    def print_usage(self):
        print("========== Xbox Controller Usage ==========")
        print("Left Stick:  X/Y translation")
        print("LB / RB:     Z translation")
        print("Right Stick: Pitch / Yaw rotation")
        print("LT / RT:     Roll rotation")
        print("A:           Close gripper")
        print("B:           Open gripper")
        print("D-pad Up/Down: Adjust speed scaling")
        print("===========================================")

    def close(self):
        pygame.quit()