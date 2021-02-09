import cv2
from time import sleep
import numpy as np

frame_rate = 30 # fps
cam_id = 0 
time_bg = 2 # how many seconds capturing background 

vid = cv2.VideoCapture(cam_id)

# Capture virtual background
for i in range(0,4):
	print('Background capture will start in {} seconds'.format(3-i), end='\r')
	sleep(1)

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

# merge averaged color channels into one image
background = cv2.merge([avgB, avgG, avgR]).astype('uint8')

bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# display captured background for 2 seconds
cv2.imshow('Background', background)
cv2.waitKey(2000)

# load image to be used as virtual background
virtual = cv2.resize(cv2.imread('rio-de-janeiro.png'),(w,h))

vB, vG, vR = cv2.split(virtual)

tau = 10 # threshold

vid = cv2.VideoCapture(cam_id)

cnt = 0

while True:
	ret, frame = vid.read()
	cnt += 1

	frameB, frameG, frameR = cv2.split(frame)
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	mask = np.where(abs(frame_gray-bg_gray)>tau, 255, 0).astype('uint8') # create mask based on threshold of the difference 
	inv_mask = cv2.bitwise_not(mask)
	
	if cnt % 200 == 0: # increase thresold every 200 frames - to find ideal threshold
		tau += 10

	fg_B = cv2.bitwise_and(frameB, frameB, mask = mask)
	fg_G = cv2.bitwise_and(frameG, frameG, mask = mask)
	fg_R = cv2.bitwise_and(frameR, frameR, mask = mask)
	
	virtual_B = cv2.bitwise_and(vB, vB, mask = inv_mask)
	virtual_G = cv2.bitwise_and(vG, vG, mask = inv_mask)
	virtual_R = cv2.bitwise_and(vR, vR, mask = inv_mask)

	fg = cv2.merge([fg_B, fg_G, fg_R]).astype('uint8')
	v_bg = cv2.merge([virtual_B,virtual_G,virtual_R]).astype('uint8')

	final = fg + v_bg
	
	cv2.imshow('Virtual Background, tau = {}'.format(tau), final)

	if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to stop
		break

vid.release()

cv2.destroyAllWindows()
