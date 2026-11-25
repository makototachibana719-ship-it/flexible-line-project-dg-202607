import numpy as np
import cv2
import mujoco


class StereoCamera:
    """
    双目深度相机封装类,将两个相同相机封装成一个双目相机，使用OpenCV计算深度图
    
    立体匹配算法选择：
    - 'BM': StereoBM (块匹配算法，速度快)
    - 'SGBM': StereoSGBM (半全局块匹配算法，精度高)
    """
    def __init__(self, mj_model, camera1_name, camera2_name, pic_width=640, pic_height=480, 
                 stereo_algorithm='BM'):
        self.camera1_name, self.camera2_name = camera1_name, camera2_name
        self.width, self.height = int(pic_width), int(pic_height)
        self.mj_model = mj_model
        self.stereo_algorithm = stereo_algorithm
        
        # 根据选择的算法创建匹配器
        if stereo_algorithm == 'BM':
            self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        elif stereo_algorithm == 'SGBM':
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=64,
                blockSize=7,
                P1=8 * 3 * 7**2,   # 控制视差连续性
                P2=32 * 3 * 7**2,  # 控制视差连续性
                disp12MaxDiff=1,
                preFilterCap=63,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32
            )
        else:
            raise ValueError(f"Unknown stereo algorithm: {stereo_algorithm}. Use 'BM' or 'SGBM'")
        
        self._renderer = mujoco.Renderer(mj_model, height=self.height, width=self.width)
        self._extract_camera_params(mj_model)

    def _extract_camera_params(self, mj_model):
        """从MuJoCo模型提取相机参数
        
        计算水平和垂直焦距以提高精度：
        - fovx（水平视场角）用于计算水平焦距 fx
        - fovy（垂直视场角）用于计算垂直焦距 fy
        这样可以处理非正方形像素（宽高比不等于1）的情况
        """
        c1_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera1_name)
        c2_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera2_name)
        self.fovy = float(mj_model.cam_fovy[c1_id])  # 垂直视场角
        
        # 计算水平视场角（fovx）
        # 使用宽高比和垂直视场角计算水平视场角
        aspect_ratio = self.width / self.height
        self.fovx = 2 * np.arctan(np.tan(np.radians(self.fovy) / 2) * aspect_ratio)
        self.fovx = np.degrees(self.fovx)
        
        b1_id, b2_id = mj_model.cam_bodyid[c1_id], mj_model.cam_bodyid[c2_id]
        self.baseline = abs(np.linalg.norm(mj_model.cam_pos[c1_id] - mj_model.cam_pos[c2_id])) if b1_id >= 0 else 0
        
        # 分别计算水平和垂直焦距
        self.fx = self.width / (2 * np.tan(np.radians(self.fovx) / 2))  # 水平焦距
        self.fy = self.height / (2 * np.tan(np.radians(self.fovy) / 2))  # 垂直焦距
        self.focal_length = self.fx  # 保留向后兼容性
        
        self.cx, self.cy = self.width / 2.0, self.height / 2.0
        # 使用 fx 和 fy 分别设置内参矩阵（处理非正方形像素）
        self.camera_intrinsic_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)
        self.inv_focal_length = 1.0 / self.focal_length
        self.cx_f, self.cy_f, self.fx_f, self.fy_f = float(self.cx), float(self.cy), float(self.fx), float(self.fy)
    
    def get_camera_image(self, mj_model, mj_data, camera_name):
        """获取单个相机图像"""
        self._renderer.update_scene(mj_data, camera=camera_name)
        return cv2.cvtColor(self._renderer.render(), cv2.COLOR_RGB2BGR)

    def get_stereo_images(self, mj_model, mj_data):
        """获取双相机彩色图像"""
        left_image = self.get_camera_image(mj_model, mj_data, self.camera1_name)
        right_image = self.get_camera_image(mj_model, mj_data, self.camera2_name)
        return left_image, right_image
    
    def get_stereo_gray_images(self, mj_model, mj_data):
        """获取双相机灰度图像"""
        left_image, right_image = self.get_stereo_images(mj_model, mj_data)
        return cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)   
    
    def compute_disparity(self, left_gray, right_gray):
        """计算单相机的视差图"""
        disparity_raw = self.stereo.compute(left_gray, right_gray).astype(np.float32)
        # SGBM算法输出16位视差，需除以16转换为浮点亚像素精度
        disparity = disparity_raw / 16.0 if self.stereo_algorithm == 'SGBM' else disparity_raw
        # 改进：使用中值滤波减少噪声
        if self.stereo_algorithm == 'BM':
            # BM算法噪声较大，应用中值滤波
            disparity = cv2.medianBlur(disparity.astype(np.uint8), 5).astype(np.float32)
        disparity[disparity <= 0] = 0.1  # 处理无效视差值
        return disparity
    
    def compute_depth_from_disparity(self, disparity):
        """从视差计算深度图"""
        depth = (self.fx * self.baseline) / disparity  # depth = (fx * baseline) / disparity，使用水平焦距
        # 改进：限制深度范围，避免异常值
        depth = np.clip(depth, 0.01, 10.0)  # 深度范围：1cm到10m
        return depth
    
    def compute_depth_map(self, mj_model, mj_data):
        """计算双相机深度图
        
        Returns:
            depth_map: 深度图
            disparity: 视差图
            left_image: 左相机图像
            right_image: 右相机图像
        """
        left_image, right_image = self.get_stereo_images(mj_model, mj_data)
        left_gray, right_gray = self.get_stereo_gray_images(mj_model, mj_data)
        disparity = self.compute_disparity(left_gray, right_gray)
        depth_map = self.compute_depth_from_disparity(disparity)
        return depth_map, disparity, left_image, right_image
    
    def get_depth_at_point(self, depth_map, point_2d):
        """获取2D点处的深度值"""
        if depth_map is None or point_2d is None: return None
        x, y = int(point_2d[0]), int(point_2d[1])
        return depth_map[y, x] if (0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0] and depth_map[y, x] > 0) else None
    
    def depth_to_colormap(self, depth_map, max_depth=2.0):
        """将深度图转换为像素值归一化的彩色热力图用于可视化
        
        Args:
            depth_map: 深度图 (numpy array, float64, 单位：米)
            max_depth: 可视化的最大深度值（米），超过此值的像素显示为红色
            
        Returns:
            color_map: 彩色热力图 (numpy array, uint8, BGR格式)
        """
        if depth_map is None:
            return None
        
        depth_display = np.clip(depth_map, 0, max_depth)
        depth_normalized = cv2.normalize(depth_display, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        return color_map
    

