import datetime
import glob
import itertools
import math
import os

import cv2
import joblib
import numpy as np
import torchvision
from pytube import YouTube

from phalp.utils import get_pylogger
from phalp.utils.io import IO_Manager

log = get_pylogger(__name__)


class Trace_IO_Manager(IO_Manager):
    """
    Class used for loading and saving videos.
    """

    def __init__(self, cfg):
        super(Trace_IO_Manager, self).__init__(cfg)
        self.source_path = self.cfg.video.source
        # find a proper video name based on the source path
        self.video_name = self.source_path.split("/")[-1].split(".")[0]
        self.frame_extractor = None

    def get_frames_from_source(self):
        if not os.path.exists(self.source_path):
            log.error(f"!!!!! Video file does not exist: {self.source_path}")
            raise Exception(f"Video file does not exist: {self.source_path}")

        self.frame_extractor = TraceFrameExtractor(self.source_path)
        log.info("Number of frames: " + str(self.frame_extractor.n_frames))

        io_data = {
            "list_of_frames": self.frame_extractor.interpolations,
            "additional_data": {},
            "video_name": self.video_name,
        }

        return io_data

    def read_frame(self, frame_path):
        frame = None
        # frame path can be either a path to an image or a list of [video_path, frame_id in pts]
        if isinstance(frame_path, tuple):
            frame = torchvision.io.read_video(
                frame_path[0],
                pts_unit="pts",
                start_pts=frame_path[1],
                end_pts=frame_path[1] + 1,
            )[0][0]
            frame = frame.numpy()[:, :, ::-1]
        elif isinstance(frame_path, str):
            frame = cv2.imread(frame_path)
        elif isinstance(frame_path, int):
            frame = self.frame_extractor.read_frame(frame_path)
        else:
            raise Exception("Invalid frame path")

        return frame

    @staticmethod
    def read_from_video_pts(video_path, frame_pts):
        frame = torchvision.io.read_video(
            video_path, pts_unit="pts", start_pts=frame_pts, end_pts=frame_pts + 1
        )[0][0]
        frame = frame.numpy()[:, :, ::-1]
        return frame

    def reset(self):
        self.video = None

    def save_video(self, video_path, rendered_, f_size, t=0):
        if t == 0:
            self.video = {
                "video": cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.output_fps,
                    frameSize=f_size,
                ),
                "path": video_path,
            }
        if self.video is None:
            raise Exception("Video is not initialized")
        self.video["video"].write(rendered_)

    def close_video(self):
        if self.video is not None:
            self.video["video"].release()
            if self.cfg.video.useffmpeg:
                ret = os.system(
                    "ffmpeg -hide_banner -loglevel error -y -i {} {}".format(
                        self.video["path"],
                        self.video["path"].replace(".mp4", "_compressed.mp4"),
                    )
                )
                # Delete if successful
                if ret == 0:
                    os.system("rm {}".format(self.video["path"]))
            self.video = None


class TraceFrameExtractor:
    """
    Class used for extracting frames from a video file.
    """

    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
        # 元のフレームを30fpsで計算し直した場合の1Fごとの該当フレーム数
        self.interpolations = np.round(np.arange(0, n_frames, fps / 30)).astype(
            np.int32
        ).tolist()
        self.n_frames = len(self.interpolations)
        self.fps = 30

    def read_frame(self, frame_id):
        self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.vid_cap.read()
        if not ret:
            raise Exception("Failed to read frame")
        return np.array(frame)

    def get_video_duration(self):
        duration = self.n_frames / self.fps
        print(f"Duration: {datetime.timedelta(seconds=duration)}")

    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        print(
            f"Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images."
        )

    def extract_frames(
        self,
        every_x_frame,
        img_name,
        dest_path=None,
        img_ext=".jpg",
        start_frame=1000,
        end_frame=2000,
    ):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)

        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
                print(f"Created the following directory: {dest_path}")

        frame_cnt = 0
        img_cnt = 0
        while self.vid_cap.isOpened():
            success, image = self.vid_cap.read()
            if not success:
                break
            if (
                frame_cnt % every_x_frame == 0
                and frame_cnt >= start_frame
                and (frame_cnt < end_frame or end_frame == -1)
            ):
                img_path = os.path.join(
                    dest_path, "".join([img_name, "%06d" % (img_cnt + 1), img_ext])
                )
                cv2.imwrite(img_path, image)
                img_cnt += 1
            frame_cnt += 1
        self.vid_cap.release()
        cv2.destroyAllWindows()
