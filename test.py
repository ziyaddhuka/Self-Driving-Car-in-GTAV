import cv2
import numpy as np

arr = np.load('training_data-196.npy')

for i in range(0,len(arr)):
	print(arr[i][1])
	print(i)
	cv2.imshow('img',arr[i][0])
	# name0 = "D:\\Avasyu\\project\\data\\pygta5\\class0\\"+str(i+1500)+".jpg"
	# name1 = "D:\\Avasyu\\project\\data\\pygta5\\class1\\"+str(i+1500)+".jpg"
	# if arr[i][1] == [1,0]:
	# 	cv2.imwrite(name0,arr[i][0])
	# else:
	# 	cv2.imwrite(name1,arr[i][0])
	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break


