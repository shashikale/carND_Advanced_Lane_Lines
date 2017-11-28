import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Ref: Use of lectures notes  

# Chessboard numbers of internal corners (nx,ny)
chessboard_size = (9,6)


def calibrate_camera_matrix(path):
    """Returns camera calibration matrix for a given
    chessboard images 9x6
    """

    object_points = []  # 3d point in real world space
    img_points = []     # 2d points in image plane.

    images = glob.glob(path)
    total_image_count = len(images)

    image_count = 1
    fig = plt.figure()


    # Make a list of calibration images
    for filename in images:
        img = cv2.imread(filename)
        nx, ny = chessboard_size
	
	# Find the chessboard corners
        retval, corners = cv2.findChessboardCorners(img, (nx, ny))
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0: nx, 0: ny].T.reshape(-1, 2)
        
        
        if retval == True:
            object_points.append(objp)
            img_points.append(corners)

            ax = fig.add_subplot(math.ceil(total_image_count / 2), 2, image_count)
            chessboard_with_corners = cv2.drawChessboardCorners(img, (nx, ny), corners, retval)
            chessboard_with_corners = cv2.cvtColor(chessboard_with_corners, cv2.COLOR_BGR2RGB)
            ax.imshow(chessboard_with_corners)
            ax.axis('off')
            image_count += 1


    return cv2.calibrateCamera(object_points, img_points, img.shape[0:2], None, None), fig


# Function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image

def undistort_image(img, cameraMatrix, distCoeffs):
    """Returns undistorted image using given
    object points and image points
    """
    return cv2.undistort(img, cameraMatrix, distCoeffs)

# Helper function to write the image
def write_image(img, file):
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file, bgr_img)



    



