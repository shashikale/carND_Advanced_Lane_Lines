import cv2
import numpy as np
from utils import cameraCalibration
from utils.line import *
from utils.binary_Threshold import *

# DEFINE A CLASS FOR IMAGE PROCESSING PIPELINE

class Pipeline:
    line = None
    Xform_Matrix = None
    Minv = None
    cameraMat = None
    distCoeffs = None
    
    
    @staticmethod
    def set_param(l, trans_mat, minv, cammat, distcoeff):
        Pipeline.line = l
        Pipeline.Xform_Matrix = trans_mat
        Pipeline.Minv = minv
        Pipeline.cameraMat = cammat
        Pipeline.distCoeffs = distcoeff

    @staticmethod
    def pipeline(img):
        line, Xform_Matrix, Minv, cameraMat, distCoeffs = Pipeline.line, Pipeline.Xform_Matrix, Pipeline.Minv, Pipeline.cameraMat, Pipeline.distCoeffs
        if (line is None or Xform_Matrix is None or Minv is None or cameraMat is None or distCoeffs is None):
            raise NotImplementedError
            
        img_size = (img.shape[1], img.shape[0])
        width, height = img_size
        img = cameraCalibration. undistort_image(np.copy(img), cameraMat, distCoeffs)

	# use cv2.warpPerspective() to warp the image
        binary_warped = cv2.warpPerspective(Sobel_Grad_dir_gradient(img),Xform_Matrix, (width, height))

	# Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Choose the number of sliding windows
        nwindows = 9
	# Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
	# Set the width of the windows +/- margin
        margin = 100
	# Set minimum number of pixels found to recenter window
        minpix = 50

        # Skip the first frame processing
        if not line.first_frame_processed:
            histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
	
	    # Find the peak of the left and right halves of the histogram
	    # These will be the starting point for the left and right lines

            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
	    # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
	    
	    # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

		# Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

		# Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
		

                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 

            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            line.update_coeff_polyfit(left_fit, right_fit)

            line.first_frame_processed = True
     
        else:
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            left_fit = line.left_fit
            right_fit = line.right_fit

            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) 
            + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) 
            + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            
            # Fit a second order polynomial to each
            line.update_coeff_polyfit(np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2))
            left_fit = line.left_fit
            right_fit = line.right_fit

        # Generate x and y values for plotting   
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 

        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        
        ## Add Radius of Curvature
        cv2.putText(result,'Curve radius: %.2fm' % line.curvature,(20,40), cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),2,cv2.LINE_AA)
        
        ## Add distance from center
        position_from_center = line.get_position_from_center()
        if position_from_center < 0:
            text = 'left'
        else:
            text = 'right'
        cv2.putText(result,'Distance From Center: %.2fm %s' % (np.absolute(position_from_center), text),(20,80), cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),2,cv2.LINE_AA)
        
        return result

 


