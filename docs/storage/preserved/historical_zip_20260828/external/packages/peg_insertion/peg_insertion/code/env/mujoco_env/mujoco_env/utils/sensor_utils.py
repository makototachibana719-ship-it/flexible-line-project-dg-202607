
import numpy as np

class ForceSensorMonitor:
    """力传感器数据监控器"""

    def __init__(self, model, data, force_sensor_name="end_force_sensor", torque_sensor_name="end_torque_sensor"):
        """
        初始化力传感器监控器

        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            force_sensor_name: 力传感器名称
            torque_sensor_name: 力矩传感器名称
        """
        self.model = model
        self.data = data
        self.force_sensor_name = force_sensor_name
        self.torque_sensor_name = torque_sensor_name

        # 获取传感器ID
        try:
            self.force_sensor_id = model.sensor(force_sensor_name).id
            self.torque_sensor_id = model.sensor(torque_sensor_name).id
            # print(f"✅ 力传感器初始化成功")
            # print(f"   - 力传感器ID: {self.force_sensor_id}")
            # print(f"   - 力矩传感器ID: {self.torque_sensor_id}")
        except Exception as e:
            raise RuntimeError(f"无法找到力传感器: {e}")

    def get_force_data(self):
        """获取力传感器数据 [Fx, Fy, Fz]"""
        return self.data.sensordata[self.force_sensor_id:self.force_sensor_id + 3].copy()

    def get_torque_data(self):
        """获取力矩传感器数据 [Tx, Ty, Tz]"""
        return self.data.sensordata[self.torque_sensor_id:self.torque_sensor_id + 3].copy()

    def get_wrench_data(self):
        """获取6维力/力矩数据 [Fx, Fy, Fz, Tx, Ty, Tz]"""
        return np.concatenate([self.get_force_data(), self.get_torque_data()])

    def get_sensor_info(self):
        """获取传感器信息"""
        force_sensor = self.model.sensor(self.force_sensor_id)
        torque_sensor = self.model.sensor(self.torque_sensor_id)

        return {
            'force_name': force_sensor.name,
            'force_type': force_sensor.type,
            'torque_name': torque_sensor.name,
            'torque_type': torque_sensor.type
        }

def perform_collision_detection(sensor_monitor, baseline_force, baseline_torque,
                              force_threshold=2.0, torque_threshold=0.5):
    """
    执行碰撞检测

    Args:
        sensor_monitor: 传感器监控器
        baseline_force: 基准力数据
        baseline_torque: 基准力矩数据
        force_threshold: 力检测阈值 (N)
        torque_threshold: 力矩检测阈值 (Nm)

    Returns:
        tuple: (is_collision, force_change, torque_change)
    """
    current_force = sensor_monitor.get_force_data()
    current_torque = sensor_monitor.get_torque_data()

    force_change = np.abs(current_force - baseline_force)
    torque_change = np.abs(current_torque - baseline_torque)

    is_collision = np.any(force_change > force_threshold) or np.any(torque_change > torque_threshold)

    return is_collision, force_change, torque_change

