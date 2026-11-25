"""
数据集结构如下图所示

dataset.hdf5
└── data/
    ├── env_args                    # 环境配置参数 (JSON格式的字符串)
    ├── total                       # 所有episode的总样本数
    ├── episode_0/
    │   ├── obs/
    │   │   ├── robot0_eye_in_hand_image       # RGB图像 (T, H, W, 3), uint8
    │   │   ├── robot0_eef_pos                 # 末端位置 (T, 3), float64
    │   │   ├── robot0_eef_quat                # 末端四元数 (T, 4), float64
    │   │   ├── robot0_gripper_qpos            # 夹爪位置 (T, 1), float64
    │   │   └── target_position                # 目标位置 (T, 3), float64
    │   ├── next_obs/
    │   │   └── (与 obs 结构相同)
    │   ├── actions                            # 动作 (T, 8), float64
    │   ├── rewards                            # 奖励 (T,), float64
    │   ├── states                             # 状态 (T, 11), float64
    │   ├── dones                              # 终止标志 (T,), int64
    │   ├── num_samples                        # 当前episode的时间步数
    │   └── model_file                         # MuJoCo模型文件路径
    └── episode_1/
        └── ...
"""

from tkinter import YES
from typing import Any, Dict, List, Optional, Tuple
import h5py
import numpy as np
import json
import os
from pathlib import Path
import pandas as pd
import cv2
from tqdm import tqdm


class MakeDataset:
    """
    数据集创建类，用于构建符合RoboMimic格式的数据集
    只接受外部提供的数据，不包含任何随机数据生成功能
    """
    
    def __init__(self, robot_name: str = "Panda", camera_name: str = "robot0_eye_in_hand", compression: str = 'gzip'):
        self.robot_name = robot_name
        self.camera_name = camera_name
        self.env_args = self._default_env_args()
        
        self.blosc_available = False
        try:
            import hdf5plugin
            self.blosc_available = True
        except ImportError:
            print("提示: 未安装hdf5plugin，将使用gzip压缩。如需blosc压缩，执行: pip install hdf5plugin")
        
        self.compression = compression
        self.compression_opts = None
        if compression == 'gzip':
            self.compression_opts = 5
        elif compression == 'blosc':
            if self.blosc_available:
                import hdf5plugin
                self.compression = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
            else:
                print(f"  警告: blosc不可用，回退到gzip压缩")
                self.compression = 'gzip'
                self.compression_opts = 5
    
    def _default_env_args(self) -> Dict:
        """
        设置默认环境参数
        """
        return {
            "env_name": "Lift",
            "type": 1,  # 1=robosuite, 2=robomimic
            "env_kwargs": {
                "has_renderer": False,
                "has_offscreen_renderer": True,
                "ignore_done": True,
                "use_object_obs": False,
                "use_camera_obs": True,
                "control_freq": 20,
                "controller_configs": {
                    "type": "OSC_POSE",
                    "input_max": 1,
                    "input_min": -1,
                    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                    "kp": 150,
                    "damping_ratio": 1,
                    "impedance_mode": "fixed",
                    "kp_limits": [0, 300],
                    "damping_ratio_limits": [0, 10],
                    "uncouple_pos_ori": True,
                    "control_delta": True,
                    "interpolation": "linear",
                    "ramp_ratio": 0.2
                },
                "robots": [self.robot_name],
                "camera_depths": False,
                "camera_heights": 224,
                "camera_widths": 224,
                "reward_shaping": False,
                "camera_names": [self.camera_name],
                "render_gpu_device_id": 0
            }
        }
    
    def create_episode(self, obs: Dict, actions: np.ndarray, rewards: np.ndarray, states: np.ndarray, 
                      dones: np.ndarray = None) -> Dict:
        """
        构建完整的episode数据
        
        Args:
            obs: 观测数据字典
            actions: 动作数据
            rewards: 奖励数据
            states: 状态数据
            dones: 终止标志数据，如果为None则自动生成
            
        Returns:
            完整的episode数据字典
        """
        T = len(rewards)
        
        return {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'states': states,
            'dones': dones,
            'num_samples': T
        }
    
    def create_hdf5_file(self, episodes_data: List[Dict], output_path: str) -> None:
        """
        创建HDF5文件并写入数据
        
        Args:
            episodes_data: 所有episode数据列表
            output_path: 输出文件路径
        """
        print(f"\n开始创建 HDF5 文件: {output_path}")
        print(f"压缩方式: {self.compression}, 压缩等级: {self.compression_opts}")
        
        chunks = None
        
        with h5py.File(output_path, 'w') as f:
            data_grp = f.create_group('data')
            data_grp.attrs['env_args'] = json.dumps(self.env_args)
            
            total_samples = 0
            
            for ep_idx, episode in tqdm(enumerate(episodes_data), desc="写入Episodes", total=len(episodes_data)):
                ep_name = f'episode_{ep_idx}'
                ep_group = data_grp.create_group(ep_name)
                
                obs_grp = ep_group.create_group('obs')
                for key, value in episode['obs'].items():
                    chunks = None
                    # 对图像数据分块
                    if key.endswith('_image') and len(value.shape) >= 3:
                        chunk_h = min(16, value.shape[1])
                        chunk_w = min(16, value.shape[2])
                        chunks = (1, chunk_h, chunk_w, value.shape[3])
                    obs_grp.create_dataset(key, data=value, compression=self.compression, compression_opts=self.compression_opts, chunks=chunks)
                
                # 创建 next_obs 组（obs 向后移动一步，最后一个时间步重复）
                next_obs_grp = ep_group.create_group('next_obs')
                for key, value in episode['obs'].items():
                    if key.endswith('_image') and len(value.shape) >= 3:
                        next_value = np.concatenate([value[1:], value[-1:]], axis=0)
                    else:
                        next_value = np.concatenate([value[1:], value[-1:]], axis=0)
                    chunks = None
                    if key.endswith('_image') and len(next_value.shape) >= 3:
                        chunk_h = min(16, next_value.shape[1])
                        chunk_w = min(16, next_value.shape[2])
                        chunks = (1, chunk_h, chunk_w, next_value.shape[3])
                    next_obs_grp.create_dataset(key, data=next_value, compression=self.compression, compression_opts=self.compression_opts, chunks=chunks)
                
                ep_group.create_dataset('actions', data=episode['actions'], compression=self.compression, compression_opts=self.compression_opts)        
                ep_group.create_dataset('rewards', data=episode['rewards'], compression=self.compression, compression_opts=self.compression_opts)        
                ep_group.create_dataset('states', data=episode['states'], compression=self.compression, compression_opts=self.compression_opts)
                ep_group.create_dataset('dones', data=episode['dones'], compression=self.compression, compression_opts=self.compression_opts)
                
                # 使用传入的num_samples
                ep_group.attrs['num_samples'] = np.int64(episode['num_samples'])
                
                T = episode['num_samples']
                total_samples += T
                print(f"  已添加 {ep_name}: {T} 个时间步")
            
            data_grp.attrs['total'] = np.int64(total_samples)
        
        print(f"\n✅ Dataset created successfully: {output_path}")
        print(f"   Episodes: {len(episodes_data)}, Total samples: {total_samples}")
        print(f"   Compression: {self.compression}")


def create_random_dataset(output_path: str, num_demos: int = 2, image_height: int = 224, image_width: int = 224, compression: str = 'gzip') -> str:
    """
    使用随机数据创建符合标准 RoboMimic 格式的 HDF5 数据集，用于测试验证

    Args:
        output_path: 输出数据集文件路径
        num_demos: 演示数量
        image_height: 图像高度
        image_width: 图像宽度
        compression: 压缩选项 ('gzip', 'blosc')
        
    Returns:
        生成的数据集路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dataset_maker = MakeDataset(compression=compression)
    
    np.random.seed(42)
    episodes_data = []
    
    for i in range(num_demos):
        print(f"生成演示数据 {i+1}/{num_demos}...")
        
        T = np.random.randint(50, 70)
        
        eef_pos = np.random.randn(T, 3).astype(np.float64)
        eef_quat = np.random.randn(T, 4).astype(np.float64)
        gripper_qpos = np.random.randn(T, 1).astype(np.float64)
        target_position = np.random.randn(T, 3).astype(np.float64)
        
        # 动作数据 (T, 8) - 包含位置(3)、四元数(4)、夹爪(1)
        actions = np.concatenate([eef_pos, eef_quat, gripper_qpos], axis=1)
        
        # 奖励数据 (T,)
        rewards = np.random.randn(T).astype(np.float64)
        
        # 观测数据
        obs = {
            'robot0_eye_in_hand_image': np.random.randint(0, 255, (T, image_height, image_width, 3), dtype=np.uint8),
            'robot0_eef_pos': eef_pos,
            'robot0_eef_quat': eef_quat,
            'robot0_gripper_qpos': gripper_qpos,
            'target_position': target_position
        }
        
        # 状态数据 (T, 11) - 包含位置(3)、四元数(4)、夹爪(1)、目标位置(3)
        states = np.concatenate([eef_pos, eef_quat, gripper_qpos, target_position], axis=1)
        
        # 使用create_episode方法创建完整的episode数据
        episode = dataset_maker.create_episode(obs, actions, rewards, states)
        episodes_data.append(episode)
    
    dataset_maker.create_hdf5_file(episodes_data, output_path)
    
    return output_path


def create_real_dataset(input_dir: str, output_path: str, sample_rate: int = 1, compression: str = 'gzip') -> str:
    """
    从真实数据创建符合标准 RoboMimic 格式的 HDF5 数据集
    
    Args:
        input_dir: 输入数据目录
        output_path: 输出数据集文件路径
        sample_rate: 采样率，每隔 sample_rate 个时间步采样一次
        compression: 压缩选项 ('gzip', 'blosc')
        
    Returns:
        生成的数据集路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset_maker = MakeDataset(robot_name="Aubo_i5", camera_name="rgb_camera_1")
    
    print(f"从 {input_dir} 加载真实数据...")
    episodes_data = []
    
    project_root = Path(__file__).parent.parent.parent
    
    episode_dirs = sorted([d for d in Path(input_dir).glob("episode_*") if d.is_dir()])
    for episode_dir in tqdm(episode_dirs, desc="处理 Episodes"):
        csv_path = episode_dir / "robot_state.csv"
        if not csv_path.exists():
            print(f"  跳过 {episode_dir.name}: 缺少 robot_state.csv")
            continue
        
        # 读取CSV数据
        df = pd.read_csv(csv_path)
        
        # 按采样率减少数据量
        df = df.iloc[::sample_rate].reset_index(drop=True)
        T = len(df)
        print(f"  原始时间步: {len(df)*sample_rate}, 采样后时间步: {T}")
        
        csv_dir = episode_dir
        rgb_images = []
        
        for _, row in df.iterrows():
            # 读取RGB图像
            rgb_path = project_root / row["rgb_filename"]
            if rgb_path.exists():
                rgb_img = cv2.imread(str(rgb_path))
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                rgb_images.append(rgb_img)
            else:
                raise FileNotFoundError(f"RGB 图像文件不存在: {rgb_path}")
            
        
        # 转换为numpy数组
        rgb_images = np.array(rgb_images)
        
        target_position = np.zeros((T, 3)).astype(np.float64)
        if 'target_cube_pos_x' in df.columns:
            target_position = df[['target_cube_pos_x', 'target_cube_pos_y', 'target_cube_pos_z']].values.astype(np.float64)
        
        rewards = np.zeros(T).astype(np.float64)
        if 'reward' in df.columns:
            rewards = df['reward'].values.astype(np.float64)
        
        eef_pos = df[['ee_pos_x', 'ee_pos_y', 'ee_pos_z']].values.astype(np.float64)
        eef_quat = df[['ee_quat_w', 'ee_quat_x', 'ee_quat_y', 'ee_quat_z']].values.astype(np.float64)
        gripper_qpos = df[['gripper_opening']].values.astype(np.float64)
        actions = np.concatenate([eef_pos, eef_quat, gripper_qpos], axis=1)
        states = np.concatenate([eef_pos, eef_quat, gripper_qpos, target_position], axis=1)
        
        # 生成终止标志 (T,) - 最后一个时间步为1，其余为0
        dones = np.zeros(T, dtype=np.int64)
        dones[-1] = 1
        
        obs = {
            'robot0_eye_in_hand_image': rgb_images,
            'robot0_eef_pos': eef_pos,
            'robot0_eef_quat': eef_quat,
            'robot0_gripper_qpos': gripper_qpos,
            'target_position': target_position
        }
        
        
        episode = dataset_maker.create_episode(obs, actions, rewards, states, dones)
        episodes_data.append(episode)
    
    print(f"加载完成，共 {len(episodes_data)} 个 episodes")
    dataset_maker.create_hdf5_file(episodes_data, output_path)
    
    return output_path


if __name__ == '__main__':
    """
    使用方法：
    # 随机数据模式（测试用）
    python mujoco_env/utils/make_datasets.py --mode random --output-path datasets/random_dataset.hdf5 --num-demos 2
    
    # 真实数据模式
    python mujoco_env/utils/make_datasets.py --mode real --input-dir outputs --output-path datasets/real_dataset.hdf5
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='创建 RoboMimic 格式数据集')
    parser.add_argument('--mode', type=str, choices=['random', 'real'], required=True, help='数据集创建模式')
    parser.add_argument('--input-dir', type=str, help='输入数据目录(仅real模式需要)')
    parser.add_argument('--output-path', type=str, required=True, help='输出数据集文件路径')
    parser.add_argument('--num-demos', type=int, default=2, help='演示数量(random模式需要)')
    parser.add_argument('--sample-rate', type=int, default=1, help='采样率(real模式需要)')
    parser.add_argument('--compression', type=str, choices=['gzip', 'blosc'], default='gzip', help='压缩方式')
    
    args = parser.parse_args()
    
    if args.mode == 'random':
        create_random_dataset(output_path=args.output_path, num_demos=args.num_demos, compression=args.compression)
    elif args.mode == 'real':
        create_real_dataset(input_dir=args.input_dir, output_path=args.output_path, sample_rate=args.sample_rate, compression=args.compression)
    else:
        raise ValueError(f"无效的模式: {args.mode}。请选择 'random' 或 'real'。")
        
