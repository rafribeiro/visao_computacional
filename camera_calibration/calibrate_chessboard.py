
import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.png')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (6,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (6,7), corners2, ret)
        cv2.imshow(fname, img)
        cv2.waitKey(500)

print ("Found corners in {} images".format(len(objpoints)))

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Up to this point, the code was copied from the OpenCV tutorial on Camera Calibration (https://docs.opencv.org/4.5.0/dc/dbb/tutorial_py_calibration.html)

#The following was coded by me (Rafael O. Ribeiro)
print("Intrinsic Matrix (K):\n",mtx,'\n')
print("Distortion Coefficients:\n",dist,'\n\n')

for fname, rvec, tvec in zip(images, rvecs, tvecs):
    r, _ = cv2.Rodrigues(rvec) #convert the rotation vector into a rotation matrix
    print("Extrinsic parameters for image {}:\n".format(fname))
    print("R - Rotation matrix:\n", r,'\n')
    print("Translation vector:\n",tvec,'\n')
    
    ext = cv2.hconcat((r, tvec)) #compose the matrix of extrinsic parameters [R|t]
    print('Extrinsic Matrix:\n', ext,'\n')
    
    #Display the axes as referenced by the image
    img = cv2.imread(fname)
    img = cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, 3, thickness=2)
    cv2.imshow(fname,img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


