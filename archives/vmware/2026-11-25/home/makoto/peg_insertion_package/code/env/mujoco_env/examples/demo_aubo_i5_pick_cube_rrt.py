#!/usr/bin/env python3
"""
Aubo i5 抓取方块演示 (RRT/纯仿真版)

基于 PickCubeTask 任务和 AuboI5Config 配置。
演示机器人在仿真环境中保持初始位姿，去除了键盘控制和示波器。

作者: Liu Gang
日期: 2026-01-03
"""

import sys
import time
import argparse
import random
import numpy as np
import mujoco
import cv2
from pathlib import Path
from mujoco import viewer as mj_viewer
from dm_robotics.transformations import transformations as tr


class CubeManager:
    """Cube管理器类，用于管理场景中的cube加载和显示"""
    
    def __init__(self):
        """初始化Cube管理器"""
        # 所有可用的cube名称
        self.all_cube_names = [
            "cube_red_1", "cube_red_2", "cube_red_3",
            "cube_green_1", "cube", "cube_green_3",
            "cube_blue_1", "cube_blue_2", "cube_blue_3"
        ]
        
        # 几何体名称映射
        self.geom_name_mapping = {
            "cube_red_1": "cube_geom_red_1",
            "cube_red_2": "cube_geom_red_2", 
            "cube_red_3": "cube_geom_red_3",
            "cube_green_1": "cube_geom_green_1",
            "cube": "cube_geom",  # 特殊处理名为"cube"的方块
            "cube_green_3": "cube_geom_green_3",
            "cube_blue_1": "cube_geom_blue_1",
            "cube_blue_2": "cube_geom_blue_2",
            "cube_blue_3": "cube_geom_blue_3"
        }
        
        self.selected_cube_names = []
    
    def select_random_cubes(self, count):
        """从所有cube中随机选择指定数量的cube
        
        Args:
            count: 要选择的cube数量（1-9）
            
        Returns:
            list: 选择的cube名称列表
        """
        if count >= len(self.all_cube_names):
            self.selected_cube_names = self.all_cube_names.copy()
        else:
            self.selected_cube_names = random.sample(self.all_cube_names, count)
        return self.selected_cube_names
    
    def hide_unselected_cubes(self, env):
        """隐藏未选择的cube
        
        Args:
            env: MuJoCo环境实例
        """
        unselected_cubes = set(self.all_cube_names) - set(self.selected_cube_names)
        
        for cube_name in unselected_cubes:
            try:
                geom_name = self.geom_name_mapping[cube_name]
                geom_id = env.model.geom(geom_name).id
                
                # 设置几何体为完全透明
                env.model.geom_rgba[geom_id][3] = 0.0
                
                # 设置不参与碰撞检测
                env.model.geom_contype[geom_id] = 0
                env.model.geom_conaffinity[geom_id] = 0
                
                print(f"✅ 成功隐藏cube: {cube_name}")
                
            except Exception as e:
                print(f"警告: 无法隐藏cube '{cube_name}': {e}")
    
    def get_selected_cubes(self):
        """获取当前选择的cube列表
        
        Returns:
            list: 选择的cube名称列表
        """
        return self.selected_cube_names
    
    def get_cube_count_info(self):
        """获取cube数量信息
        
        Returns:
            str: cube数量信息字符串
        """
        return f"{len(self.selected_cube_names)}/{len(self.all_cube_names)}"
    
    def print_cube_info(self):
        """打印cube信息"""
        print(f"   - 加载cube数量: {self.get_cube_count_info()}")
        print(f"   - 加载cube列表: {self.selected_cube_names}")
    
    def setup_cubes(self, env, cube_count):
        """设置cube的完整流程
        
        Args:
            env: MuJoCo环境实例
            cube_count: 要加载的cube数量
        """
        self.select_random_cubes(cube_count)
        self.hide_unselected_cubes(env)
        self.print_cube_info()

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 使用新的模块结构
from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_pick_cube_env
from mujoco_env.mujoco_env.robot_config.aubo_i5 import AuboI5Robot
from mujoco_env.mujoco_env.planners.rrt import RRT
import mujoco_env.mujoco_env.utils.transform as T
from mujoco_env.mujoco_env.utils.camera import StereoCamera
from mujoco_env.mujoco_env.utils.recorder import LowDimSaver, ImageSaver
from datetime import datetime
from pathlib import Path


class DataRecorder:
    """数据记录器类"""
    
    def __init__(self, env, arm_joint_names, site_id, camera_system, image_saver, lowdim_saver, gripper, rewards=None, record_data=True):
        """初始化数据记录器
        
        Args:
            env: 仿真环境
            arm_joint_names: 机械臂关节名称列表
            site_id: 末端执行器site ID
            camera_system: 相机系统实例
            image_saver: 图像保存器
            lowdim_saver: 低维数据保存器
            gripper: 夹爪管理器实例
            rewards: 奖励计算器实例
            record_data: 是否启用数据记录
        """
        self.env = env
        self.arm_joint_names = arm_joint_names
        self.site_id = site_id
        self.camera_system = camera_system
        self.image_saver = image_saver
        self.lowdim_saver = lowdim_saver
        self.gripper = gripper
        self.rewards = rewards
        self.record_data = record_data
        self.current_cube_name = None
    
    def set_current_cube(self, cube_name):
        """设置当前目标cube名称"""
        self.current_cube_name = cube_name
    
    def record_current_state(self):
        """记录当前机器人状态和图像数据"""
        if not self.record_data:
            return  # 未启用数据记录，跳过
        
        timestamp = time.time()
        
        # 1. 获取机械臂关节数据
        joint_qpos = []
        joint_qvel = []
        for name in self.arm_joint_names:
            try:
                joint_id = self.env.model.joint(name).id
                qpos_adr = self.env.model.jnt_qposadr[joint_id]
                qvel_adr = self.env.model.jnt_dofadr[joint_id]
                joint_qpos.append(self.env.data.qpos[qpos_adr])
                joint_qvel.append(self.env.data.qvel[qvel_adr])
            except Exception as e:
                print(f"Warning: Could not get joint {name} data: {e}")
                joint_qpos.append(0.0)
                joint_qvel.append(0.0)
        
        # 2. 获取夹爪开合程度
        gripper_opening = self.gripper.get_opening()
        
        # 3. 获取末端位置和姿态
        ee_pos = self.env.data.site_xpos[self.site_id].copy()
        ee_mat = self.env.data.site_xmat[self.site_id].reshape(3, 3).copy()
        ee_quat = tr.mat_to_quat(ee_mat)
        
        # 4. 获取夹爪尖端位置
        gripper_tip_pos = self.gripper.get_tip_pos_from_ee(ee_pos)
        
        # 5. 获取当前目标cube的位置
        target_cube_pos = get_current_cube_pos(self.env, self.current_cube_name) if self.current_cube_name else np.array([0.0, 0.0, 0.0])
        
        # 6. 计算到目标的距离
        distance_to_target = self.gripper.get_distance_to_target(ee_pos, target_cube_pos)
        
        # 7. 计算奖励
        reward = self.rewards.get_reward(ee_pos, target_cube_pos) if self.rewards else 0.0
        
        # 8. 获取相机图像和深度图
        rgb_image = self.camera_system.stereo_cam.get_camera_image(self.env.model, self.env.data, self.camera_system.stereo_cam.camera1_name)
        depth_map = None
        if hasattr(self.camera_system.stereo_cam, 'compute_depth_map'):
            depth_map, _, _, _ = self.camera_system.stereo_cam.compute_depth_map(self.env.model, self.env.data)
        
        # 9. 保存图像
        image_filenames = self.image_saver.save_image(rgb_image=rgb_image, depth_image=depth_map, timestamp=timestamp)
        
        # 10. 保存低维数据
        self.lowdim_saver.save_data(
            timestamp=timestamp,
            joint_qpos=joint_qpos,
            joint_qvel=joint_qvel,
            gripper_opening=gripper_opening,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            gripper_tip_pos=gripper_tip_pos,
            target_cube_pos=target_cube_pos,
            distance_to_target=distance_to_target,
            reward=reward,
            image_filenames=image_filenames
        )


class CameraSystem:
    """相机系统管理类"""
    
    def __init__(self, env, global_camera_name="sideview"):
        """初始化相机系统
        
        Args:
            env: 仿真环境
            global_camera_name: 全局相机名称
        """
        self.env = env
        self.global_camera_name = global_camera_name
        
        # 创建深度相机实例
        self.stereo_cam = StereoCamera(
            env.model, 
            camera1_name="rgb_camera_1", 
            camera2_name="rgb_camera_2", 
            pic_width=224, 
            pic_height=224,
            stereo_algorithm='BM'
        )
    
    def update_camera_display(self, record_current_state=None, show_image=False, frame_count=0):
        """更新相机显示窗口（支持无头模式）
        
        Args:
            record_current_state: 记录状态的函数（可选）
            show_image: 是否显示图像（仅在有显示器时使用）
            frame_count: 当前帧计数
            
        Returns:
            左相机图像（numpy数组）
        """
        # 获取手眼相机画面
        hand_cam_image = self.stereo_cam.get_camera_image(self.env.model, self.env.data, self.stereo_cam.camera1_name)
        
        # 获取全局相机画面
        global_cam_image = self.stereo_cam.get_camera_image(self.env.model, self.env.data, self.global_camera_name)
        
        # 直接拼接图像，不添加任何文字标签
        combined_image = np.hstack([hand_cam_image, global_cam_image])
        
        # 显示图像（仅在有显示器时）
        if show_image:
            cv2.imshow('Realsense D435i - 左相机画面 + 全局视图 (448x224)', combined_image)
            cv2.waitKey(1)
        
        # 记录当前状态（如果提供了记录函数）
        if record_current_state:
            record_current_state()
        
        return hand_cam_image


class Gripper:
    """夹爪管理器类"""
    
    def __init__(self, env, gripper_type="robotiq_2f_85"):
        """初始化夹爪管理器
        
        Args:
            env: MuJoCo仿真环境
            gripper_type: 夹爪类型
        """
        self.env = env
        self.gripper_type = gripper_type
        
        if gripper_type == "robotiq_2f_85":
            self.driver_joint = "robotiq_2f_85_right_driver_joint"
            self.min_pos = 0.0
            self.max_pos = 0.8
            self.tip_offset = np.array([0.08, 0, 0])    # 用于识别夹爪尖端位置
            self.joint_names = [
                "robotiq_2f_85_right_driver_joint",
                "robotiq_2f_85_left_driver_joint",
                "robotiq_2f_85_right_follower_joint",
                "robotiq_2f_85_left_follower_joint",
                "robotiq_2f_85_right_spring_link_joint",
                "robotiq_2f_85_left_spring_link_joint"
            ]
        else:
            self.driver_joint = "gripper_driver_joint"
            self.min_pos = 0.0
            self.max_pos = 0.05
            self.tip_offset = np.array([0.05, 0, 0])
            self.joint_names = ["gripper_driver_joint"]
    
    def get_opening(self):
        try:
            joint_id = self.env.model.joint(self.driver_joint).id
            qpos_adr = self.env.model.jnt_qposadr[joint_id]
            joint_pos = self.env.data.qpos[qpos_adr]
            
            normalized = (joint_pos - self.min_pos) / (self.max_pos - self.min_pos)
            return max(0.0, min(1.0, normalized))
        except Exception as e:
            print(f"Warning: Could not get gripper opening: {e}")
            return 0.0
    
    def open(self):
        return 0.0
    
    def close(self):
        return 1.0
    
    def set_target(self, opening):
        return max(0.0, min(1.0, opening))
    
    def is_open(self):
        return self.get_opening() < 0.1
    
    def is_closed(self):
        return self.get_opening() > 0.9
    
    def get_action(self, target_opening=None):
        if target_opening is None:
            return self.get_opening()
        return self.set_target(target_opening)
    
    def get_tip_pos_from_ee(self, ee_pos):
        return ee_pos + self.tip_offset
    
    def reset_open(self):
        """将所有夹爪关节重置为完全打开状态"""
        for joint_name in self.joint_names:
            try:
                joint_id = self.env.model.joint(joint_name).id
                qpos_adr = self.env.model.jnt_qposadr[joint_id]
                self.env.data.qpos[qpos_adr] = 0.0
                qvel_adr = self.env.model.jnt_dofadr[joint_id]
                self.env.data.qvel[qvel_adr] = 0.0
            except Exception:
                pass
    
    def get_distance_to_target(self, ee_pos, target_pos):
        return np.linalg.norm(ee_pos - target_pos)
    
    
class SimWrapper:
    """包装环境以适配 RRT 规划器"""
    def __init__(self, env):
        self.env = env
        self.mj_model = env.model
        # RRT 需要 controller.ik(pos, quat)
        class ControllerWrapper:
            def __init__(self, controller):
                self.controller = controller
            def ik(self, pos, quat):
                return self.controller.solve_ik(pos, quat)
        self.controller = ControllerWrapper(env.controller)

    def set_joint_qpos(self, qpos):
        self.env.set_joint_qpos(qpos)

    def forward(self):
        mujoco.mj_forward(self.env.model, self.env.data)

    def render(self):
        pass

    def get_geom_id(self, names):
        if isinstance(names, list):
            return [self.env.get_geom_id(name) for name in names]
        return self.env.get_geom_id(names)

    def is_contact(self, g1, g2, verbose=0):
        return self.env.is_contact(g1, g2, verbose=bool(verbose))

    def save_state(self):
        return self.env.save_state()

    def load_state(self, state):
        self.env.load_state(state)


class Rewards():
    def __init__(self, env, gripper):
        self.env = env
        self.gripper = gripper
    
    def get_reward(self, ee_pos, target_pos):
        distance = self.gripper.get_distance_to_target(ee_pos, target_pos)
        return -distance
        

def sample_action(action_space, target_pose=None, gripper_action=0.0):
    """
    返回固定的目标动作
    """
    if target_pose is None:
        return np.zeros(action_space.shape)
        
    # 构造动作：[pos(3), quat(4), gripper(1)]
    action = np.concatenate([target_pose, [gripper_action]])
    return action


def create_rrt_planner(sim, start, goal, play_area):
    """创建 RRT 规划器的辅助函数
    
    Args:
        sim: 仿真包装器实例
        start: 起点位置 (3D)
        goal: 目标位置 (3D)
        play_area: 规划区域限制 [x_min, x_max, y_min, y_max, z_min, z_max]
        
    Returns:
        配置好的 RRT 规划器实例
    """
    rrt = RRT(
        start=start,
        goal=goal,
        play_area=play_area,
        sim=sim,
        expand_dis=0.03,
        goal_sample_rate=20,
        max_iter=500
    )
    def aubo_is_collide(sim, node):
        if sim is None or node is None: return False
        for x, y, z in zip(node.path_x, node.path_y, node.path_z):
            if z < 0.42: return True
        return False
    rrt.is_collide = aubo_is_collide
    return rrt


def get_current_cube_pos(env, cube_name):
    """获取指定cube的当前位置"""
    try:
        cube_id = env.model.body(cube_name).id
        return env.data.xpos[cube_id].copy()
    except Exception as e:
        raise RuntimeError(f"无法获取物块 {cube_name} 的位置: {e}")


def move_to(env, site_id, initial_quat, camera_system, data_recorder, render, pos, gripper_val, steps=20, viewer=None):
    """直线移动到目标位置"""
    curr_pos = env.data.site_xpos[site_id].copy()
    for s in range(steps):
        frac = (s + 1) / steps
        interp_pos = curr_pos + frac * (pos - curr_pos)
        target_pose = np.concatenate([interp_pos, initial_quat])
        action = sample_action(env.action_space, target_pose=target_pose, gripper_action=gripper_val)
        env.step(action)
        
        camera_system.update_camera_display(data_recorder.record_current_state, show_image=render)
        
        if viewer is not None:
            viewer.sync()
            time.sleep(env.control_dt)


def execute_path(env, path, gripper_val, initial_quat, render, camera_system, record_current_state, steps_per_segment=12, viewer=None):
    """执行路径轨迹"""
    if path is None:
        return
    # RRT 返回的路径是 [goal, ..., start]，需要反转
    reversed_path = list(reversed(path))
    for i in range(len(reversed_path) - 1):
        start_wp = np.array(reversed_path[i])
        end_wp = np.array(reversed_path[i + 1])
        # 在两个 waypoint 之间线性插值
        for s in range(steps_per_segment):
            frac = (s + 1) / steps_per_segment
            interp_pos = start_wp + frac * (end_wp - start_wp)
            target_pose = np.concatenate([interp_pos, initial_quat])
            action = sample_action(env.action_space, target_pose=target_pose, gripper_action=gripper_val)
            env.step(action)
            
            camera_system.update_camera_display(record_current_state, show_image=render)
            
            if viewer is not None:
                viewer.sync()
                time.sleep(env.control_dt)




def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Aubo i5 抓取方块演示')
    parser.add_argument('--no-render', action='store_true', help='禁用MuJoCo渲染窗口')
    parser.add_argument('--steps', type=int, default=1000, help='每个 episode 的运行步数')
    parser.add_argument('--episodes', type=int, default=5, help='演示的 episode 数量')
    parser.add_argument('--cube-count', type=int, default=9, choices=range(1, 10), 
                       help='指定加载的cube数量（1-9，默认9个）')
    parser.add_argument('--record-data', action='store_true', help='记录机器人状态和图像数据')
    parser.add_argument('--no-timestamp-folder', action='store_true', help='不创建带时间戳的子文件夹，直接保存在outputs文件夹下')
    args = parser.parse_args()

    render = not args.no_render

    print("="*60)
    print("Aubo i5 抓取方块演示 (Cartesian IK + 初始位姿保持)")
    print(f"模式: {'渲染' if render else '无渲染'}, Episode数: {args.episodes}, 每Episode步数: {args.steps}")
    print("="*60)
    
    try:
        # === 获取初始关节配置 ===
        init_qpos = AuboI5Robot.INIT_QPOS
        print(f"初始关节配置 (INIT): {init_qpos}")
        
        # === 创建仿真环境 ===
        print("\n🌍 创建仿真环境...")
        
        env = make_aubo_i5_pick_cube_env(
            scene_name="cubes",  # 使用抓取场景
            include_image=False,    # 可视化不需要图像观测
            image_size=(128, 128),
            include_depth=False,
            use_ft_sensor=False,
            control_dt=0.02,    # 控制时间步长，决定画面帧率
            physics_dt=0.002,    # 物理时间步长，用于mujoco内部计算
            max_episode_steps=args.steps,
            render_mode="human" if render else "rgb_array",
        )
        
        # 获取机器人配置信息用于显示
        robot_config = env.task.robot_config
        print(f"✅ 仿真环境创建成功")
        print(f"   - 机器人: {robot_config.name}")
        print(f"   - 夹爪: {robot_config.gripper_name}")
        print(f"   - 控制器: {robot_config.controller_type}")
        print(f"   - 任务: {env.task.name}")
        print(f"   - 场景: {env.task.scene_name}")
        print(f"   - 帧率: {1.0 / env.control_dt:.2f} Hz")


        # 创建相机系统
        camera_system = CameraSystem(env)
        
        # 创建夹爪管理器
        gripper = Gripper(env)
        
        # 确定末端执行器site名称
        ee_site_name = "grip_site" if robot_config.gripper_name else "attachment_site"
        print(f"末端执行器Site: {ee_site_name}")
        print(f"   - 动作空间: {env.action_space}")
        
        # === 4. 运行 Episodes ===
        # 获取机器人关节名
        arm_joint_names = ["shoulder_joint", "upperArm_joint", "foreArm_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"]
        
        # 数据记录相关初始化
        record_data = args.record_data
        experiment_dir = None
        image_saver = None
        lowdim_saver = None
        
        if record_data:
            # 创建输出目录（根目录的outputs文件夹）
            project_root = Path(__file__).resolve().parent.parent  # 项目根目录
            output_root = project_root / "outputs"  # 根目录下的outputs文件夹
            output_root.mkdir(exist_ok=True)  # 确保outputs文件夹存在
            
            # 根据参数决定是否创建带时间戳的子文件夹
            if args.no_timestamp_folder:
                # 直接使用outputs文件夹
                experiment_dir = output_root
                print(f"\n📁 实验结果将直接保存到: {experiment_dir}")
            else:
                # 创建每次实验的结果文件夹：outputs_YYYYMMDD_HHMMSS
                current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                experiment_dir = output_root / f"outputs_{current_time}"  # 实验结果文件夹
                experiment_dir.mkdir(exist_ok=True)
                print(f"\n📁 实验结果将保存到: {experiment_dir}")
        
        # 创建Cube管理器并设置cube
        cube_manager = CubeManager()
        cube_manager.setup_cubes(env, args.cube_count)
        
        for episode in range(args.episodes):
            print(f"\n🚀 开始 Episode {episode + 1}/{args.episodes}...")
            
            # 初始化当前Episode的数据记录器
            if record_data:
                # 创建当前Episode的目录
                episode_dir = experiment_dir / f"episode_{episode}"
                episode_dir.mkdir(exist_ok=True)
                print(f"   📁 Episode {episode} 数据将保存到: {episode_dir}")
                
                # 初始化当前Episode的图像保存器
                image_saver = ImageSaver(output_dir=str(episode_dir), rgb_prefix="rgb", depth_prefix="depth", save_rgb=True, save_depth=True)
                
                # 初始化当前Episode的低维数据保存器
                lowdim_saver = LowDimSaver(
                    filename=str(episode_dir / "robot_state.csv"),
                    joint_names=arm_joint_names,  # 传递关节名称列表
                    save_fields=[
                        "joint_pos",  # 关节位置
                        "joint_vel",  # 关节速度
                        "gripper_opening",  # 夹爪开合程度
                        "ee_pos",     # 末端位置
                        "ee_quat",     # 末端姿态
                        "gripper_tip_pos",  # 夹爪尖端位置
                        "target_cube_pos",  # 目标cube位置
                        "distance_to_target",  # 到目标的距离
                        "reward"  # 奖励值
                    ]
                )
            
            # 1. 环境重置
            obs, info = env.reset()
            
            # 2. 强制设置机器人到初始关节配置 (确保稳定性)
            for i, name in enumerate(arm_joint_names):
                try:
                    joint_id = env.model.joint(name).id
                    qpos_adr = env.model.jnt_qposadr[joint_id]
                    env.data.qpos[qpos_adr] = init_qpos[i]
                    qvel_adr = env.model.jnt_dofadr[joint_id]
                    env.data.qvel[qvel_adr] = 0
                except Exception as e:
                    print(f"Warning: Could not set joint {name}: {e}")
            
            # 3. 推进仿真以更新运动学状态
            mujoco.mj_forward(env.model, env.data)
            
            # 4. 重置控制器状态
            env.controller.reset()
            
            # 5. 获取初始末端位姿，作为 Cartesian IK 的目标
            site_id = env.model.site(ee_site_name).id
            initial_pos = env.data.site_xpos[site_id].copy()
            initial_mat = env.data.site_xmat[site_id].reshape(3, 3).copy()
            initial_quat = tr.mat_to_quat(initial_mat)  # [w, x, y, z]
            
            current_target_pose = np.concatenate([initial_pos, initial_quat])
            
            print(f"   Episode {episode+1} 重置完成. 初始 TCP: {initial_pos}")
            
            # --- RRT 任务逻辑 ---
            sim_wrapper = SimWrapper(env)
            
            # 使用Cube管理器选择的cube列表
            cube_names = cube_manager.get_selected_cubes()
            drop_pos_center = np.array([0.3, 0.3, 0.55])
            safe_height = 0.6
            approach_height = 0.55
            
            # 创建DataRecorder实例
            rewards_calculator = Rewards(env, gripper)
            data_recorder = DataRecorder(
                env=env,
                arm_joint_names=arm_joint_names,
                site_id=site_id,
                camera_system=camera_system,
                image_saver=image_saver,
                lowdim_saver=lowdim_saver,
                gripper=gripper,
                rewards=rewards_calculator,
                record_data=record_data
            )
            
            def run_rrt_demo(viewer=None):
                """运行 RRT 搬运演示（支持渲染和无头模式）
                
                Args:
                    viewer: MuJoCo 查看器实例（仅在渲染模式下提供）
                """
                for idx, cube_name in enumerate(cube_names):
                    print(f"   正在搬运: {cube_name}")
                    
                    # 设置当前目标cube
                    data_recorder.set_current_cube(cube_name)
                    
                    # 计算当前方块的错开放置点 (3x3 网格在篮子内)
                    row = idx // 3
                    col = idx % 3
                    offset_x = (row - 1) * 0.06  # 间距 5cm
                    offset_y = (col - 1) * 0.06
                    current_drop_pos = drop_pos_center + np.array([offset_x, offset_y, 0])
                    
                    # 1. 获取方块位置
                    try:
                        cube_pos = env.data.body(cube_name).xpos.copy()
                    except Exception:
                        print(f"   Warning: Could not find cube {cube_name}, skipping.")
                        continue

                    # 2. 规划到方块上方
                    start_pos = env.data.site_xpos[site_id].copy()
                    goal_pos = cube_pos.copy()
                    goal_pos[2] = approach_height
                    
                    print(f"      规划 RRT 路径到 {cube_name} 上方...")
                    rrt = create_rrt_planner(
                        sim_wrapper,
                        start_pos, 
                        goal_pos, 
                        [0.1, 0.8, -0.5, 0.5, 0.4, 0.8]
                    )
                    path = rrt.planning(animation=False)
                    if path:
                        execute_path(env, path, 0.0, initial_quat, render, camera_system, data_recorder.record_current_state, viewer=viewer)
                    else:
                        print("      无法找到 RRT 路径，尝试直线移动")
                        move_to(env, site_id, initial_quat, camera_system, data_recorder, render, goal_pos, 0.0, viewer=viewer)

                    # 3. 下降并抓取
                    print("      下降抓取...")
                    pick_pos = cube_pos.copy()
                    pick_pos[2] = cube_pos[2] - 0.01 # 稍微深入一点确保抓稳
                    move_to(env, site_id, initial_quat, camera_system, data_recorder, render, pick_pos, 0.0, viewer=viewer)

                    # 闭合夹爪
                    for _ in range(10):
                        action = sample_action(env.action_space, target_pose=np.concatenate([pick_pos, initial_quat]), gripper_action=1.0)
                        env.step(action)
                        
                        camera_system.update_camera_display(data_recorder.record_current_state, show_image=render)
                        
                        if viewer is not None:
                            viewer.sync()   
                            time.sleep(env.control_dt)
                    
                    # 4. 提升
                    print("      提升...")
                    up_pos = pick_pos.copy()
                    up_pos[2] = approach_height
                    move_to(env, site_id, initial_quat, camera_system, data_recorder, render, up_pos, 1.0, viewer=viewer)

                    # 5. 规划到目标位置上方
                    print(f"      规划 RRT 路径到目标位置上方...")
                    start_pos = env.data.site_xpos[site_id].copy()
                    goal_drop_above = current_drop_pos.copy()
                    goal_drop_above[2] = approach_height
                    
                    rrt = create_rrt_planner(sim_wrapper, start_pos, goal_drop_above, [0.1, 0.8, -0.5, 0.5, 0.5, 0.7])
                    path = rrt.planning(animation=False)
                    if path:
                        execute_path(env, path, 1.0, initial_quat, render, camera_system, data_recorder.record_current_state, viewer=viewer)
                    else:
                        move_to(env, site_id, initial_quat, camera_system, data_recorder, render, goal_drop_above, 1.0, viewer=viewer)

                    # 6. 下降并放下
                    print("      放下...")
                    move_to(env, site_id, initial_quat, camera_system, data_recorder, render, current_drop_pos, 1.0, viewer=viewer)
                    
                    # 松开夹爪
                    for _ in range(10):
                        action = sample_action(env.action_space, target_pose=np.concatenate([current_drop_pos, initial_quat]), gripper_action=0.0)
                        env.step(action)
                        
                        camera_system.update_camera_display(data_recorder.record_current_state, show_image=render)
                        
                        if viewer is not None:
                            viewer.sync()
                            time.sleep(env.control_dt)
                    
                    # 7. 回到安全高度
                    print("      回到安全高度...")
                    final_up = current_drop_pos.copy()
                    final_up[2] = safe_height
                    move_to(env, site_id, initial_quat, camera_system, data_recorder, render, final_up, 0.0, viewer=viewer)
                    
                    # 8. 重置夹爪关节角度到完全打开状态
                    print("      重置夹爪...")
                    gripper.reset_open()
                    mujoco.mj_forward(env.model, env.data)


            # 执行 RRT 演示
            # 初始化帧计数器
            frame_count = 0
            
            original_update_method = camera_system.update_camera_display
            
            def update_camera_display_with_count(record_current_state=None, show_image=False, frame_count_param=0, _orig=original_update_method):
                nonlocal frame_count
                result = _orig(record_current_state, show_image, frame_count)
                frame_count += 1
                return result
            
            camera_system.update_camera_display = update_camera_display_with_count
            
            if render:
                with mj_viewer.launch_passive(env.model, env.data) as viewer:
                    # 创建相机显示窗口
                    cv2.namedWindow('Realsense D435i - 左相机画面 + 全局视图 (448x224)', cv2.WINDOW_NORMAL)
                    run_rrt_demo(viewer=viewer)
                    # 关闭相机窗口
                    cv2.destroyAllWindows()
            else:
                # 无头模式：直接执行 RRT 演示，无需 viewer
                run_rrt_demo(viewer=None)
            
            print(f"✅ Episode {episode+1} 演示结束")
            
            # 关闭当前Episode的数据记录器
            if record_data:
                lowdim_saver.close()
                print(f"   ✅ Episode {episode} 数据记录完成")
        
        print("\n🎉 所有演示任务完成")
        env.close()
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
