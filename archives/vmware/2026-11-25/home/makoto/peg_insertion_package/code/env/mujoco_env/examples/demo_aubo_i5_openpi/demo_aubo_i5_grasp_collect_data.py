#!/usr/bin/env python3
"""
RRT* 路径规划与 VLA 数据采集演示

Usage:
  python examples/demo_aubo_i5_grasp_collect_data.py \
    --object green_cube \
    --record-every 1 \
    --image-size 224 \
    --max-episode 100 \
    --no-render \
    --record 
"""

import os
import sys

# 在无头模式下设置 MuJoCo 渲染后端 (必须在 import mujoco 之前)
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
        print("无头模式: 设置 MUJOCO_GL=egl")

import time
import argparse
import numpy as np
import mujoco
from pathlib import Path
from mujoco import viewer as mj_viewer
from dm_robotics.transformations import transformations as tr

try:
    from pynput import keyboard  # noqa: F401
except ImportError:
    pass

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mujoco_env import make_sim_env
from mujoco_env.mujoco_env.tasks.auboi5_agv.aubo_i5_agv_config import AuboI5AGVConfig
from mujoco_env.mujoco_env.tasks.pick_cube.pick_cube import PickCubeTask
from mujoco_env.mujoco_env.robot_config.aubo_i5 import AuboI5Robot
from mujoco_env.mujoco_env.utils import KeyboardIO
from mujoco_env.mujoco_env.planners import RRT
import mujoco_env.mujoco_env.utils.transform as T
import matplotlib.pyplot as plt
try:
    from openpi_client import image_tools
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "未找到openpi_client模块。\n"
        "请在环境中执行命名安装包：pip install openpi_client"
    )

# 配置 Matplotlib: 移除工具栏
plt.rcParams["toolbar"] = "None"


class WristCameraCapture:
    """腕上/固定相机图像采集器"""

    def __init__(self, env, camera_name, ax=None):
        self.env = env
        self.camera_name = camera_name
        self.renderer = None
        self.camera_id = -1
        self.ax = ax
        self.im_artist = None
        self._init_renderer()

    def _init_renderer(self):
        """初始化渲染器"""
        try:
            self.renderer = mujoco.Renderer(self.env.model, height=480, width=640)

            self.camera_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
            )

            if self.camera_id == -1:
                name = "wrist_cam"
                cid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                if cid != -1:
                    self.camera_id = cid
                    self.camera_name = name
                    print(f"相机 '{name}' 初始化成功")

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
        """捕获图像"""
        if self.renderer is None or self.camera_id == -1:
            return None
        try:
            self.renderer.update_scene(self.env.data, camera=self.camera_id)
            return self.renderer.render()
        except Exception as e:
            print(f"图像捕获失败: {e}")
            return None

    def display(self, image, window_name="Camera"):
        """显示图像"""
        if image is not None and getattr(image, "size", 0) > 0:
            if self.ax is not None:
                try:
                    if self.im_artist is None:
                        self.im_artist = self.ax.imshow(image)
                        self.ax.set_title(self.camera_name)
                        self.ax.axis("off")
                    else:
                        self.im_artist.set_data(image)
                except Exception as e:
                    print(f"DEBUG: Display error for {self.camera_name}: {e}")
        else:
            print(f"DEBUG: Cannot display, image is invalid for {self.camera_name}")


class DataCollector:
    """数据采集器 (LeRobot 兼容格式)

    episode_XXXX/:
    - states.npy:  (N, 8) [x, y, z, qw, qx, qy, qz, gripper]
    - actions.npy: (N, 7) [dx, dy, dz, droll, dpitch, dyaw, gripper]
    - images.npz:  main_images, wrist_images (可选)
    - metadata.json
    """

    DEFAULT_IMAGE_SIZE = (224, 224)

    def __init__(
        self,
        save_dir,
        save_images=True,
        fps=50,
        image_size=None,
        record_every=5,
        image_format="npz",
        debug_every=0,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.task = ""
        self.save_images = save_images
        self.base_fps = int(fps)
        self.record_every = int(max(1, record_every))
        self.fps = int(max(1, round(self.base_fps / self.record_every)))
        self.image_size = (
            tuple(image_size) if image_size is not None else self.DEFAULT_IMAGE_SIZE
        )
        assert image_format in ("npz", "npy"), "image_format must be 'npz' or 'npy'"
        self.image_format = image_format
        self.debug_every = int(max(0, debug_every))

    def reset(self, task=""):
        self.frames = []
        self.task = task

    def add_frame(
        self,
        joint_angles,
        gripper_pos,
        gripper_quat,
        gripper_state,
        main_image=None,
        wrist_image=None,
    ):
        frame_data = {
            "gripper_pos": np.array(gripper_pos),
            "gripper_quat": np.array(gripper_quat),  # [qw, qx, qy, qz]
            "gripper_state": float(gripper_state),  # 0=close, 1=open
        }

        if self.save_images:
            H, W = self.image_size
            if main_image is not None:
                img = image_tools.resize_with_pad(main_image, H, W)
                img = image_tools.convert_to_uint8(img)
                frame_data["main_image"] = img
            if wrist_image is not None:
                img = image_tools.resize_with_pad(wrist_image, H, W)
                img = image_tools.convert_to_uint8(img)
                frame_data["wrist_image"] = img

        self.frames.append(frame_data)

        if self.debug_every > 0:
            idx = len(self.frames) - 1
            if idx % self.debug_every == 0:
                pos = frame_data["gripper_pos"].round(4)
                quat = frame_data["gripper_quat"].round(4)
                g = frame_data["gripper_state"]
                print(f"[debug] frame={idx} pos={pos} quat={quat} gripper_state={g:.1f}")

    def get_next_episode_id(self):
        existing_dirs = list(self.save_dir.glob("episode_*"))
        ids = []
        for d in existing_dirs:
            try:
                ids.append(int(d.name.split("_")[1]))
            except Exception:
                pass
        return 0 if not ids else (max(ids) + 1)

    @staticmethod
    def quat_to_euler(quat):
        w, x, y, z = quat
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
        return np.array([roll, pitch, yaw])

    def compute_delta_actions(self, states):
        n = len(states)
        actions = np.zeros((n, 7), dtype=np.float32)
        for i in range(n):
            if i < n - 1:
                pos_curr = states[i, :3]
                pos_next = states[i + 1, :3]
                actions[i, :3] = pos_next - pos_curr

                quat_curr = states[i, 3:7]
                quat_next = states[i + 1, 3:7]
                e_curr = self.quat_to_euler(quat_curr)
                e_next = self.quat_to_euler(quat_next)
                de = e_next - e_curr
                de = np.arctan2(np.sin(de), np.cos(de))
                actions[i, 3:6] = de
            else:
                actions[i, :6] = 0.0

            actions[i, 6] = states[i, 7]
        return actions

    def save_episode(self):
        import json

        if not self.frames:
            print("警告: 没有数据可保存")
            return None

        episode_id = self.get_next_episode_id()
        episode_dir = self.save_dir / f"episode_{episode_id:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        n_frames = len(self.frames)
        states = np.zeros((n_frames, 8), dtype=np.float32)
        for i, f in enumerate(self.frames):
            states[i, :3] = f["gripper_pos"]
            states[i, 3:7] = f["gripper_quat"]
            states[i, 7] = f["gripper_state"]

        actions = self.compute_delta_actions(states)

        np.save(episode_dir / "states.npy", states)
        np.save(episode_dir / "actions.npy", actions)

        if self.save_images and ("main_image" in self.frames[0] or "wrist_image" in self.frames[0]):
            H, W = self.image_size
            main_images = np.stack(
                [f.get("main_image", np.zeros((H, W, 3), dtype=np.uint8)) for f in self.frames]
            )
            wrist_images = np.stack(
                [f.get("wrist_image", np.zeros((H, W, 3), dtype=np.uint8)) for f in self.frames]
            )

            if self.image_format == "npz":
                np.savez_compressed(
                    episode_dir / "images.npz",
                    main_images=main_images,
                    wrist_images=wrist_images,
                )
            else:
                np.save(episode_dir / "main_images.npy", main_images)
                np.save(episode_dir / "wrist_images.npy", wrist_images)

        metadata = {
            "task": self.task,
            "fps": self.fps,
            "n_frames": n_frames,
            "image_size": list(self.image_size),
            "record_every": self.record_every,
            "base_fps": self.base_fps,
            "image_format": self.image_format,
            "state_dim": 8,
            "action_dim": 7,
        }
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Episode {episode_id} 已保存: {episode_dir} ({n_frames} 帧)")
        self.frames = []
        return episode_dir


# 放置成功判定阈值（米）
PLACE_XY_TOL = 0.05
PLACE_Z_TOL = 0.05


def is_object_placed(env, body_name, target_pos, xy_tol=PLACE_XY_TOL, z_tol=PLACE_Z_TOL):
    try:
        obj_pos = env.data.body(body_name).xpos.copy()
    except Exception:
        print(f"警告: 找不到物体 '{body_name}'，放置判定失败")
        return False

    xy_dist = np.linalg.norm(obj_pos[:2] - target_pos[:2])
    z_dist = abs(obj_pos[2] - target_pos[2])
    ok = (xy_dist <= xy_tol) and (z_dist <= z_tol)
    print(
        f"[place_check] obj={obj_pos.round(3)} target={target_pos.round(3)} "
        f"xy_dist={xy_dist:.3f} z_dist={z_dist:.3f} ok={ok}"
    )
    return ok


# 随机初始化物体位置的范围（半径，单位米）
RANDOMIZE_CUBE_RANGE = 0.05


def randomize_object_xy(env, body_name, max_delta=RANDOMIZE_CUBE_RANGE):
    try:
        body_id = env.model.body(body_name).id
    except Exception:
        print(f"警告: 找不到物体 '{body_name}'，无法随机初始化位置")
        return None

    if env.model.body_jntnum[body_id] <= 0:
        print(f"警告: 物体 '{body_name}' 无关节，无法移动")
        return None

    jnt_adr = env.model.body_jntadr[body_id]
    if env.model.jnt_type[jnt_adr] != mujoco.mjtJoint.mjJNT_FREE:
        print(f"警告: 物体 '{body_name}' 不是自由关节，无法随机移动")
        return None

    qpos_adr = env.model.jnt_qposadr[jnt_adr]
    qvel_adr = env.model.jnt_dofadr[jnt_adr]
    base_pos = env.data.qpos[qpos_adr : qpos_adr + 3].copy()

    delta_xy = np.random.uniform(-max_delta, max_delta, size=2)
    new_pos = base_pos.copy()
    new_pos[0] += delta_xy[0]
    new_pos[1] += delta_xy[1]

    env.data.qpos[qpos_adr : qpos_adr + 3] = new_pos
    env.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
    return new_pos


# 轨迹历史
trajectory_history = []


def clear_trajectory():
    global trajectory_history
    trajectory_history = []


def add_trajectory_point(pos):
    global trajectory_history
    trajectory_history.append(pos.copy())


def render_trajectory(viewer):
    if viewer is None or not hasattr(viewer, "user_scn"):
        return
    viewer.user_scn.ngeom = 0
    for point in trajectory_history:
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        gid = viewer.user_scn.ngeom
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[gid],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.005, 0, 0],
            pos=np.array(point),
            mat=np.eye(3).flatten(),
            rgba=[0.0, 1.0, 0.0, 1.0],
        )
        viewer.user_scn.ngeom += 1


trajectory_frame_counter = 0


def maybe_record_frame(
    env,
    data_collector,
    cached_ids,
    cameras,
    gripper_val,
    step_idx,
    step_total,
    record_every=1,
):
    """每 record_every 步记录一帧，并确保最后一步记录。"""
    if data_collector is None or cached_ids is None:
        return

    re = int(max(1, record_every))
    should_record = (step_idx % re == 0) or (step_idx == step_total - 1)
    if not should_record:
        return

    joint_angles = [env.data.qpos[adr] for adr in cached_ids["qpos_addrs"]]
    gripper_pos = env.data.site_xpos[cached_ids["grip_site_id"]].copy()
    gripper_quat = env.data.xquat[cached_ids["ee_body_id"]].copy()
    gripper_state = 1.0 - float(gripper_val)  # 0=close, 1=open（与你原逻辑一致）

    main_img = None
    wrist_img = None
    if cameras and data_collector.save_images:
        if "Main" in cameras:
            main_img = cameras["Main"].capture()
        if "Wrist" in cameras:
            wrist_img = cameras["Wrist"].capture()

    data_collector.add_frame(
        joint_angles,
        gripper_pos,
        gripper_quat,
        gripper_state,
        main_image=main_img,
        wrist_image=wrist_img,
    )


def step_and_record(
    env,
    action,
    viewer=None,
    cameras=None,
    render=True,
    data_collector=None,
    cached_ids=None,
    gripper_val_for_state=0.0,
    step_idx=0,
    step_total=1,
    record_every=1,
    do_camera_render=True,
    camera_render_stride=8,
):
    """统一封装：env.step + 录制 + viewer.sync + 相机显示"""
    obs, reward, terminated, truncated, info = env.step(action)

    maybe_record_frame(
        env,
        data_collector,
        cached_ids,
        cameras,
        gripper_val_for_state,
        step_idx,
        step_total,
        record_every=record_every,
    )

    if cameras and do_camera_render and (step_idx % int(max(1, camera_render_stride)) == 0):
        for cam_name, cam_obj in cameras.items():
            img = cam_obj.capture()
            if render:
                cam_obj.display(img, f"Camera: {cam_name}")
        if render:
            plt.pause(0.001)

    if viewer is not None:
        viewer.sync()

    return obs, reward, terminated, truncated, info


def move_linear(
    env,
    start_pose_7d,
    end_pose_7d,
    steps=20,
    viewer=None,
    gripper_val=0.0,
    record_trajectory=True,
    cameras=None,
    data_collector=None,
    cached_ids=None,
    render=True,
    record_every=5,
):
    """笛卡尔空间直线运动（全局录制版）"""
    global trajectory_frame_counter

    for i in range(steps):
        if viewer is not None and not viewer.is_running():
            return False

        alpha = (i + 1) / steps
        interpolated_pose = start_pose_7d + alpha * (end_pose_7d - start_pose_7d)
        action = np.concatenate([interpolated_pose, [gripper_val]])

        step_and_record(
            env,
            action,
            viewer=viewer,
            cameras=cameras,
            render=render,
            data_collector=data_collector,
            cached_ids=cached_ids,
            gripper_val_for_state=gripper_val,
            step_idx=i,
            step_total=steps,
            record_every=record_every,
            do_camera_render=True,
            camera_render_stride=6,
        )

        if record_trajectory:
            trajectory_frame_counter += 1
            if trajectory_frame_counter % 6 == 0:
                trajectory_frame_counter = 0
                add_trajectory_point(env.data.site("grip_site").xpos.copy())

        if viewer is not None and record_trajectory:
            render_trajectory(viewer)

        if viewer is None and (not cameras):
            time.sleep(env.control_dt)

    return True


def plan_path_rrt(start_pos, goal_pos, env=None):
    print(f"  RRT* 规划: 从 {start_pos[:3].round(3)} 到 {goal_pos[:3].round(3)}")

    z_min = min(start_pos[2], goal_pos[2]) - 0.05
    z_max = max(start_pos[2], goal_pos[2]) + 0.1

    play_area = [0.1, 1.0, -0.65, 0.65, z_min, z_max]

    rrt = RRT(
        start=start_pos[:3],
        goal=goal_pos[:3],
        play_area=play_area,
        expand_dis=0.04,
        goal_sample_rate=20,
        max_iter=1000,
        sim=None,  # 碰撞检测接口暂不兼容
    )

    path = rrt.planning(animation=False)
    if path is None:
        print("RRT* 未找到路径")
        return None
    return path[::-1]


def execute_path(
    env,
    path,
    current_quat,
    viewer=None,
    steps_per_segment=10,
    gripper_val=0.0,
    record_trajectory=True,
    cameras=None,
    render=True,
    data_collector=None,
    cached_ids=None,
    record_every=5,
):
    """执行 RRT* 路径（透传采集器）"""
    print(f"  开始执行路径，共 {len(path)} 个路径点...")
    print(f"  如果是本地渲染，需要按R重设仿真，如果是推理不需要管")
    current_pose = np.concatenate([path[0], current_quat])

    for i, waypoint in enumerate(path[1:], 1):
        target_pose = np.concatenate([waypoint, current_quat])
        ok = move_linear(
            env,
            current_pose,
            target_pose,
            steps=steps_per_segment,
            viewer=viewer,
            gripper_val=gripper_val,
            record_trajectory=record_trajectory,
            cameras=cameras,
            data_collector=data_collector,
            cached_ids=cached_ids,
            render=render,
            record_every=record_every,
        )
        if not ok:
            print(f"  路径执行中断于段 {i}/{len(path)-1}")
            return False
        current_pose = target_pose

    return True


def hold_pose(
    env,
    pose_7d,
    gripper_val,
    steps=0,
    viewer=None,
    cameras=None,
    render=True,
    data_collector=None,
    cached_ids=None,
    record_every=5,
    camera_stride=8,
):
    """在同一位姿保持 steps 步（也录制）"""
    for i in range(steps):
        if viewer is not None and not viewer.is_running():
            return False
        action = np.concatenate([pose_7d, [gripper_val]])
        step_and_record(
            env,
            action,
            viewer=viewer,
            cameras=cameras,
            render=render,
            data_collector=data_collector,
            cached_ids=cached_ids,
            gripper_val_for_state=gripper_val,
            step_idx=i,
            step_total=steps,
            record_every=record_every,
            do_camera_render=True,
            camera_render_stride=camera_stride,
        )
        if viewer is None and render:
            time.sleep(env.control_dt)
    return True


def reset_robot_joints(env, arm_joint_names, init_qpos):
    for i, name in enumerate(arm_joint_names):
        try:
            joint_id = env.model.joint(name).id
            adr = env.model.jnt_qposadr[joint_id]
            env.data.qpos[adr] = init_qpos[i]
            env.data.qvel[env.model.jnt_dofadr[joint_id]] = 0
        except Exception:
            pass


def main():
    OBJECT_MAP = {
        "green_cube": "green_cube",
        "red_cube": "red_cube",
        "milk": "milk",
        "soup": "alphabet_soup",
        "bbq": "bbq_sauce",
        "cookie": "cookies",
    }
    BODY_TO_NL = {
        "green_cube": "green cube",
        "red_cube": "red cube",
        "milk": "milk",
        "alphabet_soup": "alphabet soup",
        "bbq_sauce": "bbq sauce",
        "cookies": "cookie box",
    }

    parser = argparse.ArgumentParser(description="RRT* 抓取演示")
    parser.add_argument("--no-render", action="store_true", help="禁用MuJoCo渲染窗口")
    parser.add_argument(
        "--object",
        "-o",
        type=str,
        default="green_cube",
        choices=list(OBJECT_MAP.keys()),
        help=f'要抓取的物体: {", ".join(OBJECT_MAP.keys())} (默认: green_cube)',
    )
    parser.add_argument("--trajectory", "-t", action="store_true", help="启用轨迹渲染")
    parser.add_argument("--record", "-r", action="store_true", help="记录 VLA 训练数据")
    parser.add_argument("--debug-every", type=int, default=0, help="调试输出频率: 0=不输出，其余为每多少帧输出一次")

    parser.add_argument(
        "--max-episode",
        type=int,
        default=-1,
        help="Max saved episodes; <=0 means no limit",
    )

    # === 控制数据体积 ===
    parser.add_argument("--record-every", type=int, default=1, help="每多少个 env.step 记录一帧（全局）")
    parser.add_argument("--img-size", type=int, default=224, help="记录图像的边长 (默认 224)")
    parser.add_argument(
        "--image-format",
        type=str,
        default="npz",
        choices=["npz", "npy"],
        help="图像保存格式：npz=压缩(推荐)，npy=不压缩(很大)",
    )
    args = parser.parse_args()

    GRIPPER_OPEN = 0.0
    GRIPPER_CLOSED = 1.0

    render = not args.no_render
    if not render:
        import matplotlib
        matplotlib.use("Agg")

    target_object = OBJECT_MAP[args.object]
    enable_trajectory = args.trajectory

    print(f"\n目标物体: {args.object} ({target_object})")
    print("=" * 60)
    print("RRT* 路径规划抓取演示")
    print("=" * 60)

    keyboard_recorder = None
    env = None

    try:
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
        print("仿真环境创建成功")

        # 隐藏 grip_site (设置为全透明)
        try:
            site_id_tmp = env.model.site("grip_site").id
            env.model.site_rgba[site_id_tmp][3] = 0.0
            print("已隐藏 grip_site")
        except Exception:
            pass

        obs, info = env.reset()

        arm_joint_names = [
            "shoulder_joint",
            "upperArm_joint",
            "foreArm_joint",
            "wrist1_joint",
            "wrist2_joint",
            "wrist3_joint",
        ]
        reset_robot_joints(env, arm_joint_names, init_qpos)

        randomize_object_xy(env, target_object, max_delta=RANDOMIZE_CUBE_RANGE)

        # 预缓存 ids
        cached_ids = None
        if args.record:
            qpos_addrs = []
            for name in arm_joint_names:
                try:
                    joint_id = env.model.joint(name).id
                    qpos_addrs.append(env.model.jnt_qposadr[joint_id])
                except Exception:
                    pass

            grip_site_id = env.model.site("grip_site").id
            ee_body_id = env.model.body("end_effector").id

            cached_ids = {
                "qpos_addrs": qpos_addrs,
                "grip_site_id": grip_site_id,
                "ee_body_id": ee_body_id,
            }
            print(
                f"已预缓存 {len(qpos_addrs)} 个关节 qpos 地址, grip_site_id={grip_site_id}, ee_body_id={ee_body_id}"
            )

        mujoco.mj_forward(env.model, env.data)
        env.controller.reset()

        # 初始末端位姿
        site_id = env.model.site(ee_site_name).id
        initial_pos = env.data.site_xpos[site_id].copy()
        initial_mat = env.data.site_xmat[site_id].reshape(3, 3).copy()
        initial_quat = tr.mat_to_quat(initial_mat)
        initial_pose_7d = np.concatenate([initial_pos, initial_quat])

        print(f"   初始 TCP: {initial_pos.round(3)}")

        # 键盘控制器（保留）
        print("初始化键盘控制器...")
        if KeyboardIO is not None:
            keyboard_recorder = KeyboardIO()
            keyboard_recorder.start()
            # 默认夹爪打开（flag=0）
            try:
                keyboard_recorder._gripper_flag = 0
            except Exception:
                pass
        else:
            print("警告: KeyboardIO 不可用 (可能缺少 pynput)，键盘控制将失效")
            keyboard_recorder = None

        # 相机
        cameras = {}
        need_cameras = render or args.record
        if need_cameras:
            print("初始化相机...")
            try:
                if render:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.05)
                    ax_main, ax_wrist = axes[0], axes[1]
                    plt.ion()
                    plt.show()
                else:
                    ax_main = None
                    ax_wrist = None

                cameras["Main"] = WristCameraCapture(env, "front_view_cam", ax=ax_main)
                cameras["Wrist"] = WristCameraCapture(env, "wrist_cam", ax=ax_wrist)
            except Exception as e:
                print(f"相机初始化部分失败: {e}")
        else:
            print("无头模式且未开启记录，跳过相机初始化")

        # DataCollector
        data_collector = None
        task_instruction = ""
        if args.record:
            data_save_dir = Path(__file__).parent.parent / "data"
            base_fps = int(round(1.0 / float(env.control_dt)))  # 50
            data_collector = DataCollector(
                data_save_dir,
                save_images=True,
                fps=base_fps,
                image_size=(args.img_size, args.img_size),
                record_every=args.record_every,
                image_format=args.image_format,
                debug_every=args.debug_every,
            )
            task_object = BODY_TO_NL.get(target_object, target_object.replace("_", " "))
            task_instruction = f"pick up the {task_object} and put it into basket"
            data_collector.reset(task_instruction)
            print(f"VLA 数据采集已开启: {task_instruction}")
            if args.max_episode > 0:
                next_id = data_collector.get_next_episode_id()
                if next_id >= args.max_episode:
                    print(f"Reached max-episode={args.max_episode}, exiting")
                    return 0
        else:
            print("VLA 数据采集已关闭 (使用 --record 或 -r 开启)")

        print("按 ESC 退出, 按 R 重新规划路径")

        from contextlib import nullcontext
        viewer_ctx = mj_viewer.launch_passive(env.model, env.data) if render else nullcontext()

        with viewer_ctx as viewer:
            path_executed = False
            path = None
            step_counter = 0

            while True:
                if viewer is not None and not viewer.is_running():
                    break

                # 键盘输入
                if keyboard_recorder:
                    if keyboard_recorder.exit_flag:
                        print("收到退出信号")
                        break

                    if keyboard_recorder.reset_flag:
                        keyboard_recorder.reset_flag = False

                        obs, info = env.reset()
                        clear_trajectory()
                        reset_robot_joints(env, arm_joint_names, init_qpos)
                        randomize_object_xy(env, target_object, max_delta=RANDOMIZE_CUBE_RANGE)
                        mujoco.mj_forward(env.model, env.data)
                        env.controller.reset()

                        if data_collector:
                            data_collector.reset(task_instruction)

                        path_executed = False
                        path = None
                        continue

                # 规划：目标点 = 物体上方 15cm
                if not path_executed and path is None:
                    grip_pos = env.data.site("grip_site").xpos.copy()
                    try:
                        object_pos = env.data.body(target_object).xpos.copy()
                        print(f"{args.object} 位置: {object_pos.round(3)}")
                    except Exception:
                        print(f"找不到物体: {target_object}")
                        break

                    target_pos = object_pos.copy()
                    target_pos[2] += 0.15  # 先到上方 15cm

                    print("\n开始 RRT* 规划...")
                    print(f"  DEBUG: 传入起点: {grip_pos}")
                    print(f"  DEBUG: 传入终点(上方15cm): {target_pos}")

                    path = plan_path_rrt(grip_pos, target_pos, env)
                    if path is None:
                        print("RRT* 规划失败，请按 R 重置环境后重试")
                        path_executed = True  # 防止死循环

                # 执行任务（全程录制）
                if not path_executed and path is not None:
                    clear_trajectory()

                    # 1) 执行 RRT 到“上方15cm”（夹爪打开）
                    success = execute_path(
                        env,
                        path,
                        initial_quat,
                        viewer,
                        gripper_val=GRIPPER_OPEN,
                        record_trajectory=enable_trajectory,
                        cameras=cameras,
                        render=render,
                        data_collector=data_collector,
                        cached_ids=cached_ids,
                        record_every=args.record_every,
                    )

                    # 2) 上方稳定（录制，夹爪打开）
                    above_pose_7d = np.concatenate([path[-1], initial_quat])
                    if success:
                        success = hold_pose(
                            env,
                            above_pose_7d,
                            GRIPPER_OPEN,
                            steps=20,
                            viewer=viewer,
                            cameras=cameras,
                            render=render,
                            data_collector=None,
                            cached_ids=cached_ids,
                            record_every=args.record_every,
                            camera_stride=8,
                        )

                    # 3) 下压 15cm 到物体中心（录制，夹爪保持打开）
                    if success:
                        grasp_pos = above_pose_7d[:3].copy()
                        grasp_pos[2] -= 0.15
                        grasp_pose_7d = np.concatenate([grasp_pos, initial_quat])
                        success = move_linear(
                            env,
                            above_pose_7d,
                            grasp_pose_7d,
                            steps=30,
                            viewer=viewer,
                            gripper_val=GRIPPER_OPEN,
                            record_trajectory=enable_trajectory,
                            cameras=cameras,
                            data_collector=data_collector,
                            cached_ids=cached_ids,
                            render=render,
                            record_every=args.record_every,
                        )
                    else:
                        grasp_pose_7d = None

                    # 4) 抓取（录制，夹爪关闭）
                    if success and grasp_pose_7d is not None:
                        success = hold_pose(
                            env,
                            grasp_pose_7d,
                            GRIPPER_CLOSED,
                            steps=20,
                            viewer=viewer,
                            cameras=cameras,
                            render=render,
                            data_collector=None,
                            cached_ids=cached_ids,
                            record_every=args.record_every,
                            camera_stride=8,
                        )

                    # 5) 上抬 15cm（录制，夹爪关闭）
                    if success and grasp_pose_7d is not None:
                        lift_pose_7d = grasp_pose_7d.copy()
                        lift_pose_7d[2] += 0.15
                        success = move_linear(
                            env,
                            grasp_pose_7d,
                            lift_pose_7d,
                            steps=30,
                            viewer=viewer,
                            gripper_val=GRIPPER_CLOSED,
                            record_trajectory=enable_trajectory,
                            cameras=cameras,
                            data_collector=data_collector,
                            cached_ids=cached_ids,
                            render=render,
                            record_every=args.record_every,
                        )
                    else:
                        lift_pose_7d = None

                    # 6) 搬运到放置位置（录制，夹爪关闭）
                    place_pos = np.array([0.9, 0.2, 0.592])
                    place_pose_7d = np.concatenate([place_pos, initial_quat])
                    if success and lift_pose_7d is not None:
                        success = move_linear(
                            env,
                            lift_pose_7d,
                            place_pose_7d,
                            steps=50,
                            viewer=viewer,
                            gripper_val=GRIPPER_CLOSED,
                            record_trajectory=enable_trajectory,
                            cameras=cameras,
                            data_collector=data_collector,
                            cached_ids=cached_ids,
                            render=render,
                            record_every=args.record_every,
                        )

                    # 7) 下降 15cm（录制，夹爪关闭）
                    lower_pose_7d = place_pose_7d.copy()
                    lower_pose_7d[2] -= 0.15
                    if success:
                        success = move_linear(
                            env,
                            place_pose_7d,
                            lower_pose_7d,
                            steps=30,
                            viewer=viewer,
                            gripper_val=GRIPPER_CLOSED,
                            record_trajectory=enable_trajectory,
                            cameras=cameras,
                            data_collector=data_collector,
                            cached_ids=cached_ids,
                            render=render,
                            record_every=args.record_every,
                        )

                    # 8) 松开（录制，夹爪打开）
                    if success:
                        success = hold_pose(
                            env,
                            lower_pose_7d,
                            GRIPPER_OPEN,
                            steps=20,
                            viewer=viewer,
                            cameras=cameras,
                            render=render,
                            data_collector=None,
                            cached_ids=cached_ids,
                            record_every=args.record_every,
                            camera_stride=8,
                        )

                    # 9) 抬高 15cm（录制，不画轨迹）
                    retreat_pose_7d = lower_pose_7d.copy()
                    retreat_pose_7d[2] += 0.15
                    if success:
                        success = move_linear(
                            env,
                            lower_pose_7d,
                            retreat_pose_7d,
                            steps=30,
                            viewer=viewer,
                            gripper_val=GRIPPER_OPEN,
                            record_trajectory=False,
                            cameras=cameras,
                            data_collector=data_collector,
                            cached_ids=cached_ids,
                            render=render,
                            record_every=args.record_every,
                        )

                    # 10) 回 home（录制，不画轨迹）
                    home_pose_7d = np.concatenate([initial_pos, initial_quat])
                    if success:
                        success = move_linear(
                            env,
                            retreat_pose_7d,
                            home_pose_7d,
                            steps=30,
                            viewer=viewer,
                            gripper_val=GRIPPER_OPEN,
                            record_trajectory=False,
                            cameras=cameras,
                            data_collector=data_collector,
                            cached_ids=cached_ids,
                            render=render,
                            record_every=args.record_every,
                        )

                    # 保存/丢弃
                    if data_collector:
                        target_obj_pos = lower_pose_7d[:3].copy()
                        if success and is_object_placed(env, target_object, target_obj_pos):
                            if args.max_episode > 0:
                                next_id = data_collector.get_next_episode_id()
                                if next_id >= args.max_episode:
                                    print(f"Reached max-episode={args.max_episode}, skip save and exit")
                                    return 0
                            data_collector.save_episode()
                            if args.max_episode > 0:
                                next_id = data_collector.get_next_episode_id()
                                if next_id >= args.max_episode:
                                    print(f"Reached max-episode={args.max_episode}, exiting")
                                    return 0
                        else:
                            print("未放到目标位置或流程失败，本轮数据不保存")
                            # 丢弃：直接 reset 清空本轮缓存
                            data_collector.reset(task_instruction)

                        # 下一轮准备
                        data_collector.reset(task_instruction)

                    # 无头模式：自动 reset 继续收集
                    if not render:
                        print("无渲染模式：Episode 完成，自动重置继续收集...")

                        obs, info = env.reset()
                        clear_trajectory()
                        reset_robot_joints(env, arm_joint_names, init_qpos)
                        randomize_object_xy(env, target_object, max_delta=RANDOMIZE_CUBE_RANGE)
                        mujoco.mj_forward(env.model, env.data)
                        env.controller.reset()

                        path_executed = False
                        path = None
                        continue

                    # 渲染模式：home 稳定（也录制，允许 reset）
                    if keyboard_recorder:
                        try:
                            keyboard_recorder._gripper_flag = 0
                        except Exception:
                            pass

                    for i in range(999):
                        if keyboard_recorder and keyboard_recorder.reset_flag:
                            break
                        action = np.concatenate([home_pose_7d, [GRIPPER_OPEN]])
                        step_and_record(
                            env,
                            action,
                            viewer=viewer,
                            cameras=cameras,
                            render=render,
                            data_collector=None,
                            cached_ids=cached_ids,
                            gripper_val_for_state=GRIPPER_OPEN,
                            step_idx=i,
                            step_total=999,
                            record_every=args.record_every,
                            do_camera_render=True,
                            camera_render_stride=20,
                        )

                    path_executed = True
                    if not success:
                        print("\n任务执行失败，允许重试")
                        path = None

                # 手动控制（可选：也录制）
                if keyboard_recorder:
                    pos_offset = keyboard_recorder.get_end_pos_offset()
                    rot_offset = keyboard_recorder.get_end_rot_offset()

                    grip_pos = env.data.site("grip_site").xpos.copy()
                    current_pose = np.concatenate([grip_pos + pos_offset, initial_quat])

                    curr_mat = T.quat_2_mat(initial_quat)
                    new_mat = curr_mat @ rot_offset
                    current_pose[3:] = T.mat_2_quat(new_mat)

                    gripper_val = GRIPPER_CLOSED if keyboard_recorder.gripper_flag else GRIPPER_OPEN
                    action = np.concatenate([current_pose, [gripper_val]])

                    step_and_record(
                        env,
                        action,
                        viewer=viewer,
                        cameras=cameras,
                        render=render,
                        data_collector=data_collector,
                        cached_ids=cached_ids,
                        gripper_val_for_state=gripper_val,
                        step_idx=step_counter,
                        step_total=10**9,
                        record_every=args.record_every,
                        do_camera_render=True,
                        camera_render_stride=10,
                    )

                step_counter += 1

            print("演示结束")

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if keyboard_recorder:
            try:
                keyboard_recorder.stop()
            except Exception:
                pass
        if render:
            try:
                plt.close("all")
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
