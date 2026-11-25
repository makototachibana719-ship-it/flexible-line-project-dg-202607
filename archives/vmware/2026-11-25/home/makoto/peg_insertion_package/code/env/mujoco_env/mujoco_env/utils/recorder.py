
import queue
import time
import threading
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
from pathlib import Path

class DataSaver:
    """通用数据保存器 (CSV)"""

    def __init__(self, filename: str, headers: List[str]):
        """
        初始化数据保存器

        Args:
            filename: 保存文件名
            headers: CSV头部列表
        """
        self.filename = filename
        self.file = None
        self.headers = headers

        try:
            self.file = open(filename, 'w')
            # 写入CSV头部
            self.file.write(",".join(headers) + "\n")
            print(f"✅ 数据将保存到: {filename}")
        except Exception as e:
            print(f"❌ 创建数据文件失败: {e}")
            self.file = None

    def save_data(self, *args):
        """保存一行数据"""
        if self.file is None:
            return

        try:
            # 将所有参数转换为字符串并用逗号连接
            # 处理列表/数组类型的参数
            formatted_args = []
            for arg in args:
                if hasattr(arg, '__iter__') and not isinstance(arg, str):
                    formatted_args.extend([f"{x:.6f}" if isinstance(x, float) else str(x) for x in arg])
                else:
                    formatted_args.append(f"{arg:.6f}" if isinstance(arg, float) else str(arg))
            
            line = ",".join(formatted_args) + "\n"
            self.file.write(line)
            self.file.flush()  # 立即写入磁盘
        except Exception as e:
            print(f"保存数据失败: {e}")

    def close(self):
        """关闭文件"""
        if self.file:
            self.file.close()
            print(f"✅ 数据已保存到: {self.filename}")


class DataLogger:
    """实时数据记录和打印器 (线程安全)"""

    def __init__(self, print_interval=0.1, formatter=None):
        """
        初始化数据记录器

        Args:
            print_interval: 打印间隔时间（秒）
            formatter: 自定义格式化函数，接收数据字典返回字符串。如果为None，则使用默认格式。
        """
        self.data_queue = queue.Queue()
        self.running = False
        self.print_interval = print_interval
        self.last_print_time = 0
        self.formatter = formatter
        self.print_thread = None

    def log_data(self, data_dict):
        """记录数据"""
        if self.running:
            self.data_queue.put(data_dict)

    def start(self):
        """启动打印线程"""
        self.running = True
        self.print_thread = threading.Thread(target=self._print_loop)
        self.print_thread.daemon = True
        self.print_thread.start()

    def _print_loop(self):
        """实时打印数据循环"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_print_time >= self.print_interval:
                    # 获取最新数据
                    data_items = []
                    while not self.data_queue.empty():
                        data_items.append(self.data_queue.get())

                    if data_items:
                        # 只打印最新的数据
                        latest_data = data_items[-1]
                        
                        if self.formatter:
                            print(f"\r{self.formatter(latest_data)}", end='', flush=True)
                        else:
                            # 默认打印
                            print(f"\rData: {latest_data}", end='', flush=True)

                        self.last_print_time = current_time

                time.sleep(0.01)  # 小延迟避免CPU占用过高

            except Exception as e:
                print(f"\n数据打印错误: {e}")
                break

    def stop(self):
        """停止数据记录"""
        self.running = False
        if self.print_thread:
            self.print_thread.join(timeout=1.0)


class ImageSaver:
    """图像保存器，支持RGB和深度图像"""
    def __init__(self, output_dir: str, save_rgb: bool = True, save_depth: bool = False, rgb_prefix: str = "rgb", depth_prefix: str = "depth"):
        """
        初始化图像保存器
        
        Args:
            output_dir: 输出目录
            save_rgb: 是否保存RGB图像
            save_depth: 是否保存深度图像
            rgb_prefix: RGB图像文件名前缀
            depth_prefix: 深度图像文件名前缀
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.save_rgb = save_rgb
        self.save_depth = save_depth
        self.rgb_prefix = rgb_prefix
        self.depth_prefix = depth_prefix
        self.counter = 0
        
        # 创建子目录
        if save_rgb:
            self.rgb_dir = self.output_dir / "rgb"
            self.rgb_dir.mkdir(exist_ok=True)
        if save_depth:
            self.depth_dir = self.output_dir / "depth"
            self.depth_dir.mkdir(exist_ok=True)
        
        print(f"✅ 图像保存器初始化完成: {self.output_dir}")
        print(f"   - RGB图像: {'启用' if save_rgb else '禁用'} (前缀: {rgb_prefix})")
        print(f"   - 深度图像: {'启用' if save_depth else '禁用'} (前缀: {depth_prefix})")
    
    def save_image(self, rgb_image=None, depth_image=None, timestamp=None):
        """
        保存图像到文件
        
        Args:
            rgb_image: RGB图像数据
            depth_image: 深度图像数据
            timestamp: 时间戳
            
        Returns:
            dict: 保存的图像文件名
        """
        try:
            timestamp = timestamp if timestamp is not None else time.time()
            filenames = {}
            
            # 保存RGB图像
            if self.save_rgb and rgb_image is not None:
                rgb_filename = f"{self.rgb_prefix}_{self.counter:06d}_{timestamp:.3f}.jpg"
                rgb_path = self.rgb_dir / rgb_filename
                cv2.imwrite(str(rgb_path), rgb_image)
                filenames['rgb'] = self._get_relative_path(rgb_path)
            
            # 保存深度图像（PNG格式，支持float32，单位：米）
            if self.save_depth and depth_image is not None:
                depth_filename = f"{self.depth_prefix}_{self.counter:06d}_{timestamp:.3f}.png"
                depth_path = self.depth_dir / depth_filename
                # 转换为float32以支持PNG编码器
                depth_to_save = depth_image.astype(np.float32)
                cv2.imwrite(str(depth_path), depth_to_save)
                filenames['depth'] = self._get_relative_path(depth_path)
            
            self.counter += 1
            return filenames
        except Exception as e:
            print(f"❌ 保存图像失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _get_relative_path(self, path: Path) -> str:
        """获取相对于项目根目录的路径"""
        project_root = Path(__file__).resolve().parent.parent
        try:
            return str(path.relative_to(project_root))
        except ValueError:
            return str(path)


class LowDimSaver:
    """低维数据保存器，用于存储机械臂状态数据（支持动态关节数量）"""
    def __init__(self, filename: str, joint_names: list, save_fields: list):
        """
        初始化低维数据保存器
        
        Args:
            filename: 保存文件名
            joint_names: 机械臂关节名称列表（动态适应关节数量）
            save_fields: 要保存的字段列表，可选值：
                        - joint_pos: 关节位置
                        - joint_vel: 关节速度
                        - joint_acc: 关节加速度
                        - ee_pos: 末端位置
                        - ee_quat: 末端姿态（四元数）
                        - ee_vel: 末端速度
                        - gripper_opening: 夹爪开合程度
                        - gripper_tip_pos: 夹爪尖端位置
                        - target_cube_pos: 目标cube位置
                        - distance_to_target: 到目标的距离
        """
        self.filename = filename
        self.joint_names = joint_names  # 关节名称列表
        self.joint_count = len(joint_names)  # 动态关节数量
        self.save_fields = save_fields
        self.file = None
        
        # 生成CSV头部
        headers = ["timestamp"]
        
        # 关节数据 (动态数量)
        if "joint_pos" in save_fields:
            headers.extend([f"{name}_pos" for name in joint_names])
        if "joint_vel" in save_fields:
            headers.extend([f"{name}_vel" for name in joint_names])
        if "joint_acc" in save_fields:
            headers.extend([f"{name}_acc" for name in joint_names])
        
        # 夹爪数据
        if "gripper_opening" in save_fields:
            headers.append("gripper_opening")
        
        # 末端数据
        if "ee_pos" in save_fields:
            headers.extend(["ee_pos_x", "ee_pos_y", "ee_pos_z"])
        if "ee_quat" in save_fields:
            headers.extend(["ee_quat_w", "ee_quat_x", "ee_quat_y", "ee_quat_z"])
        if "ee_vel" in save_fields:
            headers.extend(["ee_vel_x", "ee_vel_y", "ee_vel_z"])
        
        # 夹爪尖端位置
        if "gripper_tip_pos" in save_fields:
            headers.extend(["gripper_tip_pos_x", "gripper_tip_pos_y", "gripper_tip_pos_z"])
        
        # 目标cube位置
        if "target_cube_pos" in save_fields:
            headers.extend(["target_cube_pos_x", "target_cube_pos_y", "target_cube_pos_z"])
        
        # 到目标的距离
        if "distance_to_target" in save_fields:
            headers.append("distance_to_target")
        
        # 奖励值
        if "reward" in save_fields:
            headers.append("reward")
        
        # 图像关联字段
        headers.extend(["rgb_filename", "depth_filename"])
        
        try:
            self.file = open(filename, 'w')
            self.file.write(",".join(headers) + "\n")
            print(f"✅ 低维数据将保存到: {filename}")
            print(f"   - 机械臂关节数量: {self.joint_count}")
            print(f"   - 保存字段: {', '.join(save_fields)}")
        except Exception as e:
            print(f"❌ 创建低维数据文件失败: {e}")
            import traceback
            traceback.print_exc()
            self.file = None
    
    def save_data(self, timestamp, joint_qpos=None, joint_qvel=None, joint_qacc=None, 
                  gripper_opening=None,
                  ee_pos=None, ee_quat=None, ee_vel=None,
                  gripper_tip_pos=None, target_cube_pos=None, distance_to_target=None,
                  reward=None,
                  image_filenames=None):
        """
        保存一行低维数据
        
        Args:
            timestamp: 时间戳
            joint_qpos: 关节位置 (动态长度)
            joint_qvel: 关节速度 (动态长度)
            joint_qacc: 关节加速度 (动态长度)
            gripper_opening: 夹爪开合程度 (0=完全打开, 1=完全闭合)
            ee_pos: 末端位置 (3个值)
            ee_quat: 末端姿态四元数 (4个值)
            ee_vel: 末端速度 (3个值)
            gripper_tip_pos: 夹爪尖端位置 (3个值)
            target_cube_pos: 目标cube位置 (3个值)
            distance_to_target: 到目标的距离
            reward: 奖励值
            image_filenames: 图像文件名字典
        """
        if self.file is None:
            return
        
        try:
            data = [f"{timestamp:.6f}"]
            
            # 关节数据 (动态处理)
            if "joint_pos" in self.save_fields and joint_qpos is not None:
                data.extend([f"{x:.6f}" for x in joint_qpos[:self.joint_count]])
            if "joint_vel" in self.save_fields and joint_qvel is not None:
                data.extend([f"{x:.6f}" for x in joint_qvel[:self.joint_count]])
            if "joint_acc" in self.save_fields and joint_qacc is not None:
                data.extend([f"{x:.6f}" for x in joint_qacc[:self.joint_count]])
            
            # 夹爪数据
            if "gripper_opening" in self.save_fields and gripper_opening is not None:
                data.append(f"{gripper_opening:.6f}")
            
            # 末端数据
            if "ee_pos" in self.save_fields and ee_pos is not None:
                data.extend([f"{x:.6f}" for x in ee_pos])
            if "ee_quat" in self.save_fields and ee_quat is not None:
                data.extend([f"{x:.6f}" for x in ee_quat])
            if "ee_vel" in self.save_fields and ee_vel is not None:
                data.extend([f"{x:.6f}" for x in ee_vel])
            
            # 夹爪尖端位置
            if "gripper_tip_pos" in self.save_fields and gripper_tip_pos is not None:
                data.extend([f"{x:.6f}" for x in gripper_tip_pos])
            
            # 目标cube位置
            if "target_cube_pos" in self.save_fields and target_cube_pos is not None:
                data.extend([f"{x:.6f}" for x in target_cube_pos])
            
            # 到目标的距离
            if "distance_to_target" in self.save_fields and distance_to_target is not None:
                data.append(f"{distance_to_target:.6f}")
            
            # 奖励值
            if "reward" in self.save_fields and reward is not None:
                data.append(f"{reward:.6f}")
            
            # 图像关联
            image_filenames = image_filenames or {}
            data.append(image_filenames.get("rgb", ""))
            data.append(image_filenames.get("depth", ""))
            
            line = ",".join(data) + "\n"
            self.file.write(line)
            self.file.flush()
        except Exception as e:
            print(f"❌ 保存低维数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    def close(self):
        """关闭文件"""
        if self.file:
            self.file.close()
            print(f"✅ 低维数据已保存到: {self.filename}")

