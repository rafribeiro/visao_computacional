import cv2
from time import sleep
import numpy as np

vid = cv2.VideoCapture(0)

# Capture virtual background

for i in range(0,5):
	print('Background capture will start in {} seconds'.format(5-i), end='\r')
	sleep(1)
print('Background capture will start in 0 seconds')

t = 3 # how many seconds capturing background
avgR = np.zeros((480, 640)) # initialize each color channel
avgG = np.zeros((480, 640))
avgB = np.zeros((480, 640))
cnt = 0
for i in range(0,30*t):
	ret, frame = vid.read() # grab frame from camera
	cnt += 1
	B, G, R = cv2.split(frame.astype('float')) # split captured frame into color channels

	avgR = (avgR*(cnt-1) + R)*(1./cnt) # accumulate averages for each color channel
	avgG = (avgG*(cnt-1) + G)*(1./cnt)
	avgB = (avgB*(cnt-1) + B)*(1./cnt)

# merge averaged color channels into one image
background = cv2.merge([avgB, avgG, avgR]).astype('uint8')

# display captured background for 2 seconds
cv2.imshow('Background', background)
cv2.waitKey(2000)

# load image to be used as virtual background
virtual = cv2.resize(cv2.imread('rio-de-janeiro.png'),(640,480))

vB, vG, vR = cv2.split(virtual.astype('float'))

tau = 10 # threshold

cnt = 0
while True:
	ret, frame = vid.read()
	cnt += 1

	fB, fG, fR = cv2.split(frame.astype('float'))

	if cnt % 150 == 0: # increase thresold every 150 frames - to find ideal threshold
		tau += 10

	# compute pixel value according to threshold
	finalB = np.where(abs(fB-avgB) < tau, vB, fB)
	finalG = np.where(abs(fG-avgG) < tau, vG, fG)
	finalR = np.where(abs(fR-avgR) < tau, vR, fR)
	
	final = cv2.merge([finalB, finalG, finalR]).astype('uint8')
	
	cv2.imshow('Virtual Background, tau = {}'.format(tau), final)

	if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to stop
		break

vid.release()

cv2.destroyAllWindows()
