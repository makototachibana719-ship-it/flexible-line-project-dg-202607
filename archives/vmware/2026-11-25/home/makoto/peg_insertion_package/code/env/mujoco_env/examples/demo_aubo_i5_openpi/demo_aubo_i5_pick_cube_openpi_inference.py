#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPI 远程推理接入 MuJoCo (agv_auboi5)

运行方式：
1) 先在 openpi 环境启动 server:
   uv run scripts/serve_policy.py policy:checkpoint \
     --policy.config=<policy_config_name> \
     --policy.dir=path/to/your/checkpoint

2) 本地转发远程端口（假设 server 在远程机器
   ssh -L 8000:localhost:8000 user@remote_host

3) 在 mujoco-env 环境运行本脚本：
   python examples/demo_aubo_i5_pick_cube_openpi_inference.py \
   --prompt "pick up the green cube and put it into basket" \
   --infer-interval 10 \
   --host localhost \
   --port 8000

"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import mujoco

# 无头模式下设置 MuJoCo 渲染后端（必须在 import viewer 前）
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
        print("无头模式: 设置 MUJOCO_GL=egl")

try:
    import cv2
except Exception as e:
    raise RuntimeError("需要 opencv-python (cv2) 用于 resize。请先安装 opencv-python。") from e

from mujoco import viewer as mj_viewer
import matplotlib.pyplot as plt

try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mujoco_env import make_sim_env
from mujoco_env.mujoco_env.tasks.auboi5_agv.aubo_i5_agv_config import AuboI5AGVConfig
from mujoco_env.mujoco_env.tasks.pick_cube.pick_cube import PickCubeTask
from mujoco_env.mujoco_env.robot_config.aubo_i5 import AuboI5Robot

# openpi-client
from openpi_client import websocket_client_policy,image_tools

def preprocess_image(img, size: int):
    if img is None or img.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    img = image_tools.resize_with_pad(img, size, size)
    img = image_tools.convert_to_uint8(img)
    return img

class ManualGripperControl:
    """Keyboard override: press O to request gripper open once."""

    def __init__(self):
        self._listener = None
        self._open_requested = False

    def start(self):
        if pynput_keyboard is None:
            print("[WARN] pynput not available, manual gripper override disabled")
            return

        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char in ('o', 'O'):
                    self._open_requested = True
                    print("[manual] gripper open requested (O)")
            except Exception:
                pass

        self._listener = pynput_keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    def consume_open(self):
        if self._open_requested:
            self._open_requested = False
            return True
        return False

    def stop(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None



# -------------------------
# 1) 相机采集
# -------------------------
class CameraCapture:
    """相机图像采集器：capture 返回 RGB uint8"""

    def __init__(self, env, camera_name, h=480, w=640, ax=None):
        self.env = env
        self.camera_name = camera_name
        self.h = h
        self.w = w
        self.ax = ax
        self.im_artist = None

        self.renderer = None
        self.camera_id = -1
        self._init_renderer()

    def _init_renderer(self):
        try:
            self.renderer = mujoco.Renderer(self.env.model, height=self.h, width=self.w)
            self.camera_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)

            if self.camera_id == -1:
                # 兼容 wrist_cam 名字
                if self.camera_name != "wrist_cam":
                    cid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
                    if cid != -1:
                        self.camera_id = cid
                        self.camera_name = "wrist_cam"

            if self.camera_id != -1:
                print(f"相机 '{self.camera_name}' 初始化成功")
            else:
                print(f"警告: 找不到相机 '{self.camera_name}'")
                print("可用相机列表:")
                for i in range(self.env.model.ncam):
                    nm = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                    print(f"  - {nm}")

        except Exception as e:
            print(f"渲染器初始化失败: {e}")

    def capture(self):
        if self.renderer is None or self.camera_id == -1:
            return None
        try:
            self.renderer.update_scene(self.env.data, camera=self.camera_id)
            img = self.renderer.render()  # RGB uint8
            return img
        except Exception as e:
            print(f"图像捕获失败({self.camera_name}): {e}")
            return None

    def display(self, image):
        if image is None or image.size == 0 or self.ax is None:
            return
        if self.im_artist is None:
            self.im_artist = self.ax.imshow(image)
            self.ax.set_title(self.camera_name)
            self.ax.axis("off")
        else:
            self.im_artist.set_data(image)


# -------------------------
# 2) 数据采集器
# -------------------------
class DataCollector:
    """
    保存每帧数据到 episode_XXXX/:
    - states.npy: (N, 8) [x,y,z,qw,qx,qy,qz, gripper_open]   (注意：1=open, 0=close)
    - actions.npy: (N, 7) [dx,dy,dz,droll,dpitch,dyaw, gripper_open]
    - main_images.npy / wrist_images.npy: (N,512,512,3) uint8
    """
    LEROBOT_IMAGE_SIZE = (224, 224)

    def __init__(self, save_dir, save_images=True, fps=50):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.task = ""
        self.save_images = save_images
        self.fps = fps

    def reset(self, task=""):
        self.frames = []
        self.task = task

    def add_frame(self, joint_angles, gripper_pos, gripper_quat_wxyz, gripper_open,
                  main_image=None, wrist_image=None):
        frame_data = {
            "gripper_pos": np.array(gripper_pos, dtype=np.float32),
            "gripper_quat": np.array(gripper_quat_wxyz, dtype=np.float32),
            "gripper_state": float(gripper_open),  # 1=open, 0=close
        }

        H, W = self.LEROBOT_IMAGE_SIZE
        if self.save_images:
            if main_image is not None:
                frame_data["main_image"] = preprocess_image(main_image, H)
            if wrist_image is not None:
                frame_data["wrist_image"] = preprocess_image(wrist_image, H)

        self.frames.append(frame_data)

    def get_next_episode_id(self):
        existing_dirs = list(self.save_dir.glob("episode_*"))
        ids = []
        for d in existing_dirs:
            try:
                ids.append(int(d.name.split("_")[1]))
            except Exception:
                pass
        return 0 if not ids else max(ids) + 1

    @staticmethod
    def quat_to_euler_wxyz(q):
        # q = [w,x,y,z]
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw], dtype=np.float32)

    def compute_delta_actions(self, states):
        n = len(states)
        actions = np.zeros((n, 7), dtype=np.float32)
        for i in range(n):
            if i < n - 1:
                pos_curr = states[i, :3]
                pos_next = states[i + 1, :3]
                delta_pos = pos_next - pos_curr

                q_curr = states[i, 3:7]
                q_next = states[i + 1, 3:7]
                e_curr = self.quat_to_euler_wxyz(q_curr)
                e_next = self.quat_to_euler_wxyz(q_next)
                delta_e = e_next - e_curr
                delta_e = np.arctan2(np.sin(delta_e), np.cos(delta_e))

                actions[i, :3] = delta_pos
                actions[i, 3:6] = delta_e
            else:
                actions[i, :6] = 0.0
            actions[i, 6] = states[i, 7]
        return actions

    def save_episode(self):
        import json

        if not self.frames:
            print("警告: 没有数据可保存")
            return None

        eid = self.get_next_episode_id()
        ed = self.save_dir / f"episode_{eid:04d}"
        ed.mkdir(parents=True, exist_ok=True)

        n = len(self.frames)
        states = np.zeros((n, 8), dtype=np.float32)
        for i, f in enumerate(self.frames):
            states[i, :3] = f["gripper_pos"]
            states[i, 3:7] = f["gripper_quat"]
            states[i, 7] = f["gripper_state"]

        actions = self.compute_delta_actions(states)
        np.save(ed / "states.npy", states)
        np.save(ed / "actions.npy", actions)

        H, W = self.LEROBOT_IMAGE_SIZE
        if self.save_images and "main_image" in self.frames[0]:
            main_images = np.stack([f.get("main_image", np.zeros((H, W, 3), np.uint8)) for f in self.frames])
            wrist_images = np.stack([f.get("wrist_image", np.zeros((H, W, 3), np.uint8)) for f in self.frames])
            np.save(ed / "main_images.npy", main_images)
            np.save(ed / "wrist_images.npy", wrist_images)

        meta = {
            "task": self.task,
            "fps": self.fps,
            "n_frames": n,
            "image_size": [H, W],
            "state_dim": 8,
            "action_dim": 7,
        }
        with open(ed / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Episode {eid} 已保存: {ed} ({n} 帧)")
        self.frames = []
        return ed


# -------------------------
# 3) 去除关节限位
# -------------------------
def disable_mujoco_joint_limits(env, joint_names):
    """
    不“关闭”limited，而是把关节范围扩大到极大：
    - jnt_limited = 1  (让 IK 读取到有效 lower/upper)
    - jnt_range   = [-BIG, BIG]
    这样对 IK 来说是“无限关节范围”，同时不会触发 lower>=upper 的警告。
    """
    BIG = 1e6
    for name in joint_names:
        try:
            jid = env.model.joint(name).id
            env.model.jnt_limited[jid] = 1
            env.model.jnt_range[jid, 0] = -BIG
            env.model.jnt_range[jid, 1] =  BIG
        except Exception:
            pass
    mujoco.mj_forward(env.model, env.data)



def relax_controller_joint_limits(controller):
    """
    尝试放宽 controller/IK 内部关节限位（不同实现属性名不同，做 best-effort）。
    若 controller 还有更深一层 clamp，需要按实际类名再补属性。
    """
    BIG = 1e6

    pairs = [
        ("qpos_min", "qpos_max"),
        ("joint_lower", "joint_upper"),
        ("lower_limits", "upper_limits"),
        ("min_qpos", "max_qpos"),
    ]
    for lo_name, hi_name in pairs:
        if hasattr(controller, lo_name) and hasattr(controller, hi_name):
            try:
                lo = np.asarray(getattr(controller, lo_name), dtype=np.float32)
                hi = np.asarray(getattr(controller, hi_name), dtype=np.float32)
                lo[...] = -BIG
                hi[...] =  BIG
                setattr(controller, lo_name, lo)
                setattr(controller, hi_name, hi)
            except Exception:
                pass

    if hasattr(controller, "joint_limits"):
        try:
            jl = getattr(controller, "joint_limits")
            if isinstance(jl, (list, tuple)) and len(jl) > 0:
                setattr(controller, "joint_limits", [(-BIG, BIG) for _ in range(len(jl))])
        except Exception:
            pass




def clamp_joint_qpos(env, qpos_adr, dof_adr, value):
    # Hard lock joint angle by clamping qpos/qvel each step.
    if qpos_adr is None or value is None:
        return
    try:
        env.data.qpos[qpos_adr] = float(value)
        if dof_adr is not None:
            env.data.qvel[dof_adr] = 0.0
        mujoco.mj_forward(env.model, env.data)
    except Exception:
        pass


# -------------------------
# 4) delta 积分
# -------------------------
def quat_wxyz_to_euler(q):
    return DataCollector.quat_to_euler_wxyz(q)

def euler_to_quat_wxyz(e):
    roll, pitch, yaw = e
    cr = np.cos(roll * 0.5); sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5); sy = np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)

def integrate_delta_pose(pose7_wxyz, delta6, lock_yaw=True):
    """
    pose7=[x,y,z,qw,qx,qy,qz], delta6=[dx,dy,dz,droll,dpitch,dyaw]
    lock_yaw: 如果为 True，忽略 delta_yaw，保持 yaw 不变（锁定 wrist3 旋转）
    """
    pos = pose7_wxyz[:3]
    quat = pose7_wxyz[3:7]
    dpos = delta6[:3]
    drot = delta6[3:6].copy()

    if lock_yaw:
        drot[2] = 0.0  # 忽略 yaw 增量

    new_pos = (pos + dpos).astype(np.float32)

    e = quat_wxyz_to_euler(quat)
    e_new = (e + drot).astype(np.float32)
    e_new = np.arctan2(np.sin(e_new), np.cos(e_new)).astype(np.float32)  # wrap
    new_quat = euler_to_quat_wxyz(e_new)

    return np.concatenate([new_pos, new_quat]).astype(np.float32)


# -------------------------
# 5) 环境初始化
# -------------------------
def create_env(render: bool):
    robot_config = AuboI5AGVConfig.get_config()
    init_qpos = AuboI5Robot.INIT_QPOS

    task = PickCubeTask(
        robot_config=robot_config,
        scene_name="pick_object",
        include_image=False,
        image_size=(512, 512),
        include_depth=False,
    )

    ee_site_name = "grip_site"
    env = make_sim_env(
        task=task,
        control_dt=0.02,
        physics_dt=0.002,
        max_episode_steps=1000000,
        render_mode="human" if render else "rgb_array",
        ee_site_name=ee_site_name,
        use_target_as_ctrl=True,
        ik_regularization=1e-4,
        ik_radius=0.05,
    )

    # 隐藏 grip_site
    try:
        sid = env.model.site("grip_site").id
        env.model.site_rgba[sid][3] = 0.0
    except Exception:
        pass

    obs, info = env.reset()

    arm_joint_names = ["shoulder_joint", "upperArm_joint", "foreArm_joint",
                       "wrist1_joint", "wrist2_joint", "wrist3_joint"]
    for i, name in enumerate(arm_joint_names):
        try:
            joint_id = env.model.joint(name).id
            qpos_adr = env.model.jnt_qposadr[joint_id]
            env.data.qpos[qpos_adr] = init_qpos[i]
            qvel_adr = env.model.jnt_dofadr[joint_id]
            env.data.qvel[qvel_adr] = 0
        except Exception:
            pass

    # 去除 MuJoCo joint limits
    disable_mujoco_joint_limits(env, arm_joint_names)

    # 预缓存 ids
    qpos_addrs = []
    for name in arm_joint_names:
        try:
            joint_id = env.model.joint(name).id
            qpos_addrs.append(env.model.jnt_qposadr[joint_id])
        except Exception:
            pass

    grip_site_id = env.model.site("grip_site").id
    ee_body_id = env.model.body("end_effector").id

    # 锁定第6关节 (wrist3_joint) - 通过在物理步进前覆盖 ctrl[5] 实现
    wrist3_joint_name = "wrist3_joint"
    wrist3_joint_id = env.model.joint(wrist3_joint_name).id
    wrist3_qpos_adr = env.model.jnt_qposadr[wrist3_joint_id]
    wrist3_dof_adr = env.model.jnt_dofadr[wrist3_joint_id]
    wrist3_init_angle = float(env.data.qpos[wrist3_qpos_adr])
    print(f"[init] 锁定第6关节 ({wrist3_joint_name}) 初始角度: {wrist3_init_angle:.6f} rad (ctrl[5]覆盖方法)")

    cached_ids = {
        "qpos_addrs": qpos_addrs,
        "grip_site_id": grip_site_id,
        "ee_body_id": ee_body_id,
        "wrist3_qpos_adr": wrist3_qpos_adr,
        "wrist3_dof_adr": wrist3_dof_adr,
        "wrist3_init_angle": wrist3_init_angle,
    }

    mujoco.mj_forward(env.model, env.data)

    # 尝试放宽 controller/IK 内部 joint limits
    # relax_controller_joint_limits(env.controller)

    env.controller.reset()

    return env, ee_site_name, cached_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", type=str, default="pick up the green cube and put it into basket")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=99999)
    parser.add_argument("--infer-interval", type=int, default=10, help="50Hz 下每隔多少步推理一次（默认10 => 5Hz）")
    parser.add_argument("--record", action="store_true", help="是否保存 episode 数据")
    parser.add_argument("--data-dir", type=str, default=None, help="保存数据的目录（默认: <repo>/data）")
    parser.add_argument("--reset-err", type=float, default=0.10, help="actual-expected 误差超过该值则重置 expected_pose（单位：米）")
    parser.add_argument("--drop-chunk-on-reset", action="store_true", help="重置 expected_pose 后是否丢弃当前 chunk 并重新推理")
    args = parser.parse_args()

    render = not args.no_render
    if not render:
        import matplotlib
        matplotlib.use("Agg")

    # 连接远程 policy server
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[client] connected to ws://{args.host}:{args.port}")

    # 创建环境
    env, ee_site_name, cached_ids = create_env(render=render)
    print("仿真环境创建成功")

    # 相机
    cameras = {}
    if render:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.05)
        ax_main, ax_wrist = axes[0], axes[1]
        plt.ion()
        plt.show()
    else:
        ax_main = None
        ax_wrist = None

    cameras["Main"] = CameraCapture(env, "front_view_cam", h=480, w=640, ax=ax_main)
    cameras["Wrist"] = CameraCapture(env, "wrist_cam", h=480, w=640, ax=ax_wrist)

    # 数据采集器（可选）
    data_collector = None
    if args.record:
        save_dir = Path(args.data_dir) if args.data_dir else (Path(__file__).resolve().parent.parent / "data")
        data_collector = DataCollector(save_dir, save_images=True, fps=50)
        data_collector.reset(task=args.prompt)
        print(f"[record] ON, save_dir={save_dir}")
    else:
        print("[record] OFF")

    # 夹爪状态：1=open / 0=close
    manual_ctrl = ManualGripperControl()
    manual_ctrl.start()
    gripper_open = 1.0

    cached_chunk = None  # (50,7)
    chunk_idx = 0
    chunk_len = 0

    # 初始化 expected_pose = 当前实际位姿（wxyz）
    grip_pos0 = env.data.site_xpos[cached_ids["grip_site_id"]].copy().astype(np.float32)
    grip_quat0 = env.data.xquat[cached_ids["ee_body_id"]].copy().astype(np.float32)
    expected_pose = np.concatenate([grip_pos0, grip_quat0]).astype(np.float32)

    from contextlib import nullcontext
    viewer_ctx = mj_viewer.launch_passive(env.model, env.data) if render else nullcontext()

    with viewer_ctx as viewer:
        for step in range(args.max_steps):
            if viewer is not None and not viewer.is_running():
                break

            # 当前实际
            grip_pos = env.data.site_xpos[cached_ids["grip_site_id"]].copy().astype(np.float32)
            grip_quat = env.data.xquat[cached_ids["ee_body_id"]].copy().astype(np.float32)  # wxyz

            # --- 转为 LeRobot/OpenPI 训练格式 [x,y,z, ax,ay,az, g, g] ---
            import math
            def quat2axisangle(quat):
                q = quat / (np.linalg.norm(quat) + 1e-12)
                w, x, y, z = q
                if abs(w) > 0.999999:
                    return np.zeros(3, dtype=np.float32)
                angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
                sin_half = math.sqrt(max(1.0 - w*w, 1e-12))
                axis = np.array([x, y, z], dtype=np.float64) / sin_half
                return (axis * angle).astype(np.float32)

            grip_quat_infer = grip_quat.copy()
            # 半球固定
            if grip_quat_infer[0] > 0:
                grip_quat_infer = -grip_quat_infer
            aa = quat2axisangle(grip_quat_infer)
            state8_input = np.concatenate([grip_pos, aa, [gripper_open, gripper_open]], axis=0).astype(np.float32)
            # -----------------------------------------------------------

            # 相机图像
            main_img = cameras["Main"].capture()
            wrist_img = cameras["Wrist"].capture()
            if main_img is None:
                main_img = np.zeros((480, 640, 3), dtype=np.uint8)
            if wrist_img is None:
                wrist_img = np.zeros((480, 640, 3), dtype=np.uint8)

            # 推理
            IMG_SIZE = 224
            if cached_chunk is None or chunk_idx >= chunk_len:
                obs = {
                    "observation/state": state8_input.astype(np.float32),
                    "observation/image": preprocess_image(main_img, IMG_SIZE),
                    "observation/wrist_image": preprocess_image(wrist_img, IMG_SIZE),
                    "prompt": args.prompt,
                }
                out = client.infer(obs)
                cached_chunk = np.asarray(out["actions"], dtype=np.float32)  # (50,7)

                chunk_len = min(args.infer_interval, cached_chunk.shape[0])
                if chunk_len <= 0:
                    print("[error] empty action chunk from server")
                    break
                chunk_idx = 0

                a0 = cached_chunk[0]
                print(
                    f"[infer] step={step:05d} "
                    f"actual_pos={grip_pos.round(3)} "
                    f"delta_pos={a0[:3].round(4)} "
                    f"delta_rot={a0[3:6].round(4)} "
                    f"gr_raw={float(a0[6]):.3f}"
                )

            # 取 action
            a = cached_chunk[chunk_idx]
            chunk_idx += 1

            # 更新 expected_pose
            expected_pose = integrate_delta_pose(expected_pose, a[:6])
            gripper_open = 1.0 if float(a[6]) > 0.6 else 0.0
            if manual_ctrl.consume_open():
                gripper_open = 1.0

            # 发给环境：target=expected_pose
            gripper_val = 1.0 - gripper_open  # 0=开,1=关
            env_action = np.concatenate([expected_pose, [gripper_val]])

            try:
                # 手动执行 step 流程，在物理步进前覆盖第6关节控制信号
                env_action_clipped = np.clip(env_action, env.action_space.low, env.action_space.high)
                env._set_action(env_action_clipped)
                
                # 覆盖第6关节控制信号为初始角度（在物理步进前）
                env.data.ctrl[5] = cached_ids["wrist3_init_angle"]
                
                # 执行物理步进
                for _ in range(env._n_substeps):
                    mujoco.mj_step(env.model, env.data)
                
                env._elapsed_steps += 1
                
                # 获取观测和奖励
                obs = env._get_obs()
                reward = env._compute_reward(obs)
                terminated = env._check_terminated(obs)
                truncated = env._elapsed_steps >= env.max_episode_steps
                info = env._get_info()
            except Exception as e:
                # IK/step 报错：重置 expected_pose 到实际，避免下一步继续发散
                print(f"[IK ERROR] env.step failed: {repr(e)} -> reset expected_pose to actual")
                act_pos = env.data.site_xpos[cached_ids["grip_site_id"]].copy().astype(np.float32)
                act_quat = env.data.xquat[cached_ids["ee_body_id"]].copy().astype(np.float32)
                expected_pose = np.concatenate([act_pos, act_quat]).astype(np.float32)
                if args.drop_chunk_on_reset:
                    cached_chunk = None
                    chunk_idx = 0
                    chunk_len = 0
                continue

            # 观测 actual vs expected
            site_id = env.model.site("grip_site").id
            actual_xyz = env.data.site_xpos[site_id].copy()
            expected_xyz = expected_pose[:3].copy()
            err = float(np.linalg.norm(actual_xyz - expected_xyz))
            print(f"[track] expected={expected_xyz.round(3)} actual={actual_xyz.round(3)} err={err:.4f}")

            if err > float(args.reset_err):
                print(f"[WARN] tracking err {err:.4f} > reset_err {args.reset_err:.4f} -> reset expected_pose to actual")
                act_quat = env.data.xquat[cached_ids["ee_body_id"]].copy().astype(np.float32)
                expected_pose = np.concatenate([actual_xyz.astype(np.float32), act_quat]).astype(np.float32)
                if args.drop_chunk_on_reset:
                    cached_chunk = None
                    chunk_idx = 0
                    chunk_len = 0

            if step % 20 == 0:
                print("[info keys]", list(info.keys())[:30])

            # 记录
            if data_collector is not None:
                joint_angles = [env.data.qpos[adr] for adr in cached_ids["qpos_addrs"]]
                data_collector.add_frame(
                    joint_angles=joint_angles,
                    gripper_pos=grip_pos,
                    gripper_quat_wxyz=grip_quat,
                    gripper_open=gripper_open,
                    main_image=main_img,
                    wrist_image=wrist_img,
                )

            # 相机显示
            if render and (step % 6 == 0):
                cameras["Main"].display(main_img)
                cameras["Wrist"].display(wrist_img)
                plt.draw()
                plt.pause(0.001)

            if viewer is not None:
                viewer.sync()

            time.sleep(env.control_dt)

    manual_ctrl.stop()

    # 保存 episode
    if data_collector is not None:
        data_collector.save_episode()

    if render:
        plt.close("all")
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
