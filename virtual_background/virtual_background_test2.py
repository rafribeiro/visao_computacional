import cv2
from time import sleep
import numpy as np

frame_rate = 30 # camera fps
cam_id = 0 
time_bg = 2 # how many seconds capturing background 

vid = cv2.VideoCapture(cam_id)
vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 10)

for i in range(0,4):
	print('Background capture will start in {} seconds'.format(3-i), end='\r')
	sleep(1)

# Capture virtual background
for i in range(0,time_bg*frame_rate+1):
	ret, frame = vid.read() # grab frame from camera
	if i == 0:
		(h, w) = (frame.shape[0], frame.shape[1])
		avgR = np.zeros((h, w)) # initialize each color channel
		avgG = np.zeros((h, w))
		avgB = np.zeros((h, w))
	B, G, R = cv2.split(frame.astype('float')) # split captured frame into color channels
	avgR = (avgR*i + R)/(i+1) # accumulate averages for each color channel
	avgG = (avgG*i + G)/(i+1)
	avgB = (avgB*i + B)/(i+1)

# Merge averaged color channels into one image
background = cv2.merge([avgB, avgG, avgR]).astype('uint8')

bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# Display captured background for 2 seconds
cv2.imshow('Background', background)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# Load image to be used as virtual background
virtual = cv2.resize(cv2.imread('rio-de-janeiro.png'),(w,h))

vB, vG, vR = cv2.split(virtual)

tau = 60 # threshold

kernel = np.ones((3,3), np.uint8) # kernel for dilate/erode
while True:
	ret, frame = vid.read()
	
	# Convert frame to grayscale
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Create mask based on threshold of the difference of grayscale frame and background
	mask = np.where(abs(frame_gray-bg_gray)>tau, 255, 0).astype('uint8')

	# Some processing on the mask to remove some noise / "holes" in the mask
	mask = cv2.dilate(mask,kernel, iterations=10)
	mask = cv2.erode(mask,kernel, iterations=20)
	mask = cv2.dilate(mask,kernel, iterations=10)
	
	# Obtain an inverted mask
	inv_mask = cv2.bitwise_not(mask)
	
	cv2.imshow('Mask', mask)
	cv2.imshow('Inverted_mask', inv_mask)
	
	# Obtain foreground based on the mask
	frameB, frameG, frameR = cv2.split(frame)
	fg_B = cv2.bitwise_and(frameB, frameB, mask = mask)
	fg_G = cv2.bitwise_and(frameG, frameG, mask = mask)
	fg_R = cv2.bitwise_and(frameR, frameR, mask = mask)
	
	# Obtain the virtual background based on the inverted mask
	virtual_B = cv2.bitwise_and(vB, vB, mask = inv_mask)
	virtual_G = cv2.bitwise_and(vG, vG, mask = inv_mask)
	virtual_R = cv2.bitwise_and(vR, vR, mask = inv_mask)

	# Join color channels of foreground and virtual background
	fg = cv2.merge([fg_B, fg_G, fg_R]).astype('uint8')
	v_bg = cv2.merge([virtual_B,virtual_G,virtual_R]).astype('uint8')

	# Compute final frame
	final = fg + v_bg
	
	# Display frame with virtual background
	cv2.imshow('Virtual Background, thershold = {}'.format(tau), final)

	if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to stop
		break
vid.release()
cv2.destroyAllWindows()