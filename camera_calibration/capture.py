import cv2
import os

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
i=0
while True:
	ret, frame = cap.read()
	cv2.imshow('Webcam', frame)
	key = cv2.waitKey(1)
	if key & 0xFF == ord('q'): # press 'q' to stop
		break
	if key & 0xFF == ord('c'): # press 'c' to capture frame
		if ret:
			i += 1
			cv2.imwrite('img_' + str(i) + '.png', frame)
cap.release()
cv2.destroyAllWindows()
