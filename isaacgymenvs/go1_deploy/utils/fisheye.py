import cv2
import numpy as np
import math
import time


class Converter:
    def __init__(self, width=928, height=800, pinhole_width=640, fov=110, rot_X=0.0):
        self.width = width
        self.height = height
        self.pinhole_width = pinhole_width
        self.fov = fov

        self.equirectWidth = pinhole_width
        self.equirectHeight = pinhole_width
        self.equirectImage = np.zeros(
            (self.equirectHeight, self.equirectWidth, 3), dtype=np.uint8
        )

        self.output_FOV = self.fov
        self.output_focal_length = (
            0.5 * self.equirectWidth / math.tan(self.output_FOV * math.pi / 180 / 2)
        )
        self.input_K = np.array(
            [
                [3.7513105535330789e02, 1.1310698962271699e00, 4.3923707519180203e02],
                [0.0, 3.8594489809074821e02, 4.0432640383775964e02],
                [0.0, 0.0, 1.0],
            ]
        )
        self.input_D = np.array(
            [
                -2.1254604151820813e-01,
                2.8620423345986648e-02,
                1.7361499056536282e-03,
                -8.2752772562614830e-05,
            ],
            dtype=np.float32,
        )
        self.output_K = np.array(
            [
                [self.output_focal_length, 0, self.equirectWidth / 2],
                [0, self.output_focal_length, self.equirectHeight / 2],
                [0, 0, 1],
            ]
        )
        self.mapX, self.mapY = cv2.fisheye.initUndistortRectifyMap(
            self.input_K,
            self.input_D,
            np.array([rot_X, 0.0, 0.0]),
            self.output_K,
            (self.equirectWidth, self.equirectHeight),
            cv2.CV_32FC1,
        )

    def fish_to_pinhole(self, img_in):
        """
        Converts a fisheye image to a pinhole image using the precomputed mapX and mapY.

        Args:
            img_in (numpy.ndarray): The fisheye image to be converted.

        Returns:
            numpy.ndarray: The pinhole image after conversion.
        """
        assert img_in.shape[0] == self.height and img_in.shape[1] == self.width

        self.equirectImage = cv2.remap(
            img_in,
            self.mapX,
            self.mapY,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        return self.equirectImage


if __name__ == "__main__":
    # Step 1: Read the Fisheye Image
    fisheyeImage = cv2.imread("./resized_image-2.jpg")

    # resized_image = cv2.resize(fisheyeImage, (464, 400))

    # # Save the resized image
    # cv2.imwrite("resized_image-3.jpg", resized_image)

    # cv2.imshow("Fisheye Image", fisheyeImage)
    # cv2.waitKey(0)

    # Parameters

    width = fisheyeImage.shape[1]
    height = fisheyeImage.shape[0]

    print(width, height)

    assert width == 464 and height == 400

    # fov = 120.0  # Field of view in degrees
    # FOV = fov * math.pi / 180.0
    # focal_length = 0.5 * width / math.tan(FOV / 2)

    # Step 3: Create the Equirectangular Image
    equirectWidth = 640
    equirectHeight = 640
    equirectImage = np.zeros((equirectHeight, equirectWidth, 3), dtype=np.uint8)

    output_FOV = 110.0
    output_focal_length = 0.5 * equirectWidth / math.tan(output_FOV * math.pi / 180 / 2)

    # input_K = np.array(
    #     [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]]
    # )

    input_K = np.array(
        [
            [1.8756552767665394e02, 5.6553494811358496e-01, 2.1961853759590102e02],
            [0.0, 1.9297244904537411e02, 2.0216320191887982e02],
            [0.0, 0.0, 1.0],
        ]
    )

    output_K = np.array(
        [
            [output_focal_length, 0, equirectWidth / 2],
            [0, output_focal_length, equirectHeight / 2],
            [0, 0, 1],
        ]
    )

    # Step 4: Perform the Conversion
    print("K size:", K.shape)
    D = np.array(
        [
            -2.1254604151820813e-01,
            2.8620423345986648e-02,
            1.7361499056536282e-03,
            -8.2752772562614830e-05,
        ],
        dtype=np.float32,
    )
    # new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
    #     K, D, (width, height), 1, (equirectWidth, equirectHeight), 0
    # )

    for i in range(1):
        # calc time for processing

        mapX, mapY = cv2.fisheye.initUndistortRectifyMap(
            input_K,
            D,
            np.array([0.0, 0.0, 0.0]),
            output_K,
            (equirectWidth, equirectHeight),
            cv2.CV_32FC1,
        )

        # map with X rotation
        mapX1, mapY1 = cv2.fisheye.initUndistortRectifyMap(
            input_K,
            D,
            np.array([1.0, 0.0, 0.0]),
            output_K,
            (equirectWidth, equirectHeight),
            cv2.CV_32FC1,
        )

        mapX2, mapY2 = cv2.fisheye.initUndistortRectifyMap(
            input_K,
            D,
            np.array([0.0, -1.0, 0.0]),
            output_K,
            (equirectWidth, equirectHeight),
            cv2.CV_32FC1,
        )

        mapX3, mapY3 = cv2.fisheye.initUndistortRectifyMap(
            input_K,
            D,
            np.array([0.0, 0.0, 1.0]),
            output_K,
            (equirectWidth, equirectHeight),
            cv2.CV_32FC1,
        )

        start_time = time.time()

        equirectImage = cv2.remap(
            fisheyeImage,
            mapX,
            mapY,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        rotImage = cv2.remap(
            fisheyeImage,
            mapX1,
            mapY1,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        # map with Y rotation

        rotImage2 = cv2.remap(
            fisheyeImage,
            mapX2,
            mapY2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        # map with Z rotation

        rotImage3 = cv2.remap(
            fisheyeImage,
            mapX3,
            mapY3,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        # end time
        end_time = time.time()
        print("Time taken in seconds : ", (end_time - start_time))

    cv2.imshow("Equirectangular Image", equirectImage)
    cv2.waitKey(0)

    # Step 5: Save the Equirectangular Image
    # cv2.imwrite("equirectangular_image.jpg", equirectImage)
