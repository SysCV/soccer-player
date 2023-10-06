#!/usr/bin/env python
"""
BlueRov video capture class
"""

import atexit
import cv2
import gi
import numpy as np
import threading
import sys
import signal

from isaacgymenvs.go1_deploy.utils.fisheye import Converter
from ultralytics import YOLO

gi.require_version("Gst", "1.0")
from gi.repository import Gst


class ObjectDetector:
    def __init__(self, detection_model, converter):
        self.detection_model = detection_model
        self.converter = converter
        self.frame = None
        self.detection_result = None
        self.lock = threading.Lock()
        self.stop_thread = False

        # Start a separate thread for detection
        self.thread = threading.Thread(target=self._detect_thread)
        self.thread.daemon = (
            True  # Allow the program to exit even if this thread is running
        )

        atexit.register(self.stop)
        self.thread.start()

    def detect(self, frame):
        # Set the current frame for detection
        with self.lock:
            self.frame = frame.copy()

    def get_detection_result(self):
        # Get the latest detection result
        with self.lock:
            return self.detection_result

    def _detect_thread(self):
        while not self.stop_thread:
            if self.frame is not None:
                # Perform object detection on the current frame
                pinhole_frame = self.converter.fish_to_pinhole(self.frame)
                # cv2.imshow("origin-backend", self.frame)
                # cv2.imshow("pinhole-backend", pinhole_frame)
                print("loop running ... ")
                cv2.waitKey(10)
                results = self.detection_model.predict(
                    pinhole_frame, show=True, stream=False, device="cuda:1"
                )

                # for r in results:
                #     im_array = r.plot()  # plot a BGR numpy array of predictions
                #     cv2.imshow("result", im_array)
                # cv2.waitKey(1)

                # Store the detection result
                # with self.lock:
                #     self.detection_result = results
                # print(results.)

    def stop(self):
        self.stop_thread = True
        self.thread.join()


class Video:
    """BlueRov video capture class constructor

    Attributes:
        port (int): Video UDP port
        video_codec (string): Source h264 parser
        video_decode (string): Transform YUV (12bits) to BGR (24bits)
        video_pipe (object): GStreamer top-level pipeline
        video_sink (object): Gstreamer sink element
        video_sink_conf (string): Sink configuration
        video_source (string): Udp source ip and port
        latest_frame (np.ndarray): Latest retrieved video frame
    """

    def __init__(self, port=9200):
        """Summary

        Args:
            port (int, optional): UDP port
        """

        Gst.init(None)

        self.port = port
        self.latest_frame = self._new_frame = None

        # [Software component diagram](https://www.ardusub.com/software/components.html)
        # UDP video stream (:9200)
        self.video_source = "udpsrc port={}".format(self.port)
        # [Rasp raw image](http://picamera.readthedocs.io/en/release-0.7/recipes2.html#raw-image-capture-yuv-format)
        # Cam -> CSI-2 -> H264 Raw (YUV 4-4-4 (12bits) I420)
        self.video_codec = (
            "! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264"
        )
        # Python don't have nibble, convert YUV nibbles (4-4-4) to OpenCV standard BGR bytes (8-8-8)
        self.video_decode = (
            "! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert"
        )
        # Create a sink to get data
        self.video_sink_conf = (
            "! appsink emit-signals=true sync=false max-buffers=2 drop=true"
        )

        self.video_pipe = None
        self.video_sink = None

        self.run()

        self.model = YOLO(
            "/home/gymuser/IsaacGymEnvs-main/isaacgymenvs/dataset/best_93.pt"
        )
        self.converter = Converter()
        # self.detector = ObjectDetector(self.model, self.converter)

    def start_gst(self, config=None):
        """ Start gstreamer pipeline and sink
        Pipeline description list e.g:
            [
                'videotestsrc ! decodebin', \
                '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                '! appsink'
            ]

        Args:
            config (list, optional): Gstreamer pileline description list
        """

        if not config:
            config = [
                "videotestsrc ! nvv4l2decoder",
                "! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert",
                "! appsink drop=1",
            ]

        print("Gstreamer pipeline: {}".format(" ".join(config)))
        command = " ".join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name("appsink0")

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array

        Args:
            sample (TYPE): Description

        Returns:
            TYPE: Description
        """
        buf = sample.get_buffer()
        caps_structure = sample.get_caps().get_structure(0)
        array = np.ndarray(
            (caps_structure.get_value("height"), caps_structure.get_value("width"), 3),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8,
        )
        return array

    def frame(self):
        """Get Frame

        Returns:
            np.ndarray: latest retrieved image frame
        """
        if self.frame_available():
            self.latest_frame = self._new_frame
            # reset to indicate latest frame has been 'consumed'
            self._new_frame = None
        return self.latest_frame

    def dectection_result(self):
        pinhole_frame = self.converter.fish_to_pinhole(self.frame())
        result = self.model.predict(
            pinhole_frame, show=False, stream=True, device="cuda:0", verbose=False
        )
        return result

    def frame_available(self):
        """Check if a new frame is available

        Returns:
            bool: true if a new frame is available
        """
        return self._new_frame is not None

    def run(self):
        """Get frame to update _new_frame"""

        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf,
            ]
        )

        self.video_sink.connect("new-sample", self.callback)

    def callback(self, sink):
        sample = sink.emit("pull-sample")
        self._new_frame = self.gst_to_opencv(sample)

        return Gst.FlowReturn.OK


if __name__ == "__main__":
    # Create the video object
    # Add port= if is necessary to use a different one
    video = Video()
    # model = YOLO("./dataset/best_93.pt")

    print("Initialising stream...")
    waited = 0
    while not video.frame_available():
        waited += 1
        print("\r  Frame not available (x{})".format(waited), end="")
        cv2.waitKey(30)
    print('\nSuccess!\nStarting streaming - press "q" to quit.')

    convert = Converter()
    while True:
        # Wait for the next frame to become available
        if video.frame_available():
            # Only retrieve and display a frame if it's new
            frame = video.frame()
            # cv2.imshow("origin", frame)
            # print(frame.shape)
            # cv2.waitKey(0)
            # Convert fisheye to equirectangular
            pinhole_frame = convert.fish_to_pinhole(frame)

            import os

            # Create a directory to save the frames
            if not os.path.exists("saved_frames"):
                os.makedirs("saved_frames")

            # Initialize a counter to keep track of the frame number
            frame_num = 0
            save_frames = False

            while True:
                # Wait for the next frame to become available
                if video.frame_available():
                    # Only retrieve and display a frame if it's new
                    frame = video.frame()
                    # Convert fisheye to equirectangular

                    cv2.imshow("origin", frame)

                    # Display the frame
                    # pinhole_frame = convert.fish_to_pinhole(frame)
                    # cv2.imshow("frame", pinhole_frame)

                    results = video.dectection_result()

                    for r in results:
                        print(r.boxes)
                        im_array = r.plot()  # plot a BGR numpy array of predictions
                        cv2.imshow("result", im_array)

                    # Save the frame as a picture if save_frames is True
                    if save_frames:
                        print("saving...", frame_num)
                        cv2.imwrite(
                            "./saved_frames/frame{}.jpg".format(frame_num),
                            pinhole_frame,
                        )
                        frame_num += 1

                    # Wait for a key press and check if the user wants to quit or save frames
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        # Start saving frames after a key press
                        save_frames = True
                        print("press saving")
                    elif key == ord("x"):
                        # Stop saving frames after a second key press
                        save_frames = False
                        print("press stop")

            # Clean up
            cv2.destroyAllWindows()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
