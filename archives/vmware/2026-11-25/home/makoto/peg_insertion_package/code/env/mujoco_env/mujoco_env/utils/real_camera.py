import numpy as np
import queue
import threading
import time

try:
    import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
except ImportError:
    class MockRS:
        class context:
            def __init__(self):
                # Mock all possible devices used in configs
                self.devices = [
                    MockRS.device("130322274175"),
                    MockRS.device("128422272758"),
                    MockRS.device("127122270572")
                ]
        class device:
            def __init__(self, serial):
                self.serial = serial
            def get_info(self, info):
                return self.serial
        class camera_info:
            serial_number = 0
        class pipeline:
            def start(self, config): return None
            def stop(self): pass
            def wait_for_frames(self): return MockRS.frames()
        class config:
            def enable_device(self, serial): pass
            def enable_stream(self, stream, w, h, fmt, fps): pass
            def disable_all_streams(self): pass
        class stream:
            color = 0
            depth = 1
        class format:
            bgr8 = 0
            z16 = 1
        class align:
            def __init__(self, stream): pass
            def process(self, frames): return frames
        class frames:
            def get_color_frame(self): return MockRS.frame()
            def get_depth_frame(self): return MockRS.frame()
        class frame:
            def is_video_frame(self): return True
            def is_depth_frame(self): return True
            def get_data(self): return np.zeros((480, 640, 3), dtype=np.uint8)

    rs = MockRS()


class RSCapture:
    def get_device_serial_numbers(self):
        if rs is None:
            raise ImportError("pyrealsense2 is not installed")
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False):
        if rs is None:
            raise ImportError("pyrealsense2 is not installed")
        self.name = name
        assert serial_number in self.get_device_serial_numbers()
        self.serial_number = serial_number
        self.depth = depth
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
        self.profile = self.pipe.start(self.cfg)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()

        if color_frame.is_video_frame():
            image = np.asarray(color_frame.get_data())
            if self.depth and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((image, depth), axis=-1)
            else:
                return True, image
        else:
            return False, None

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()


class VideoCapture:
    def __init__(self, cap, name=None):
        if name is None:
            name = cap.name
        self.name = name
        self.q = queue.Queue()
        self.cap = cap
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.enable = True
        self.t.start()

        # read frames as soon as they are available, keeping only most recent one

    def _reader(self):
        while self.enable:
            time.sleep(0.01)
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get(timeout=5)

    def close(self):
        self.enable = False
        self.t.join()
        self.cap.close()
