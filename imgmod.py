import numpy as np
import cv2
#img = cv2.imread('messi5.jpg')




############################TRANSLATION###############################
def translate(img,num):
        rows= img.shape[0]
        cols=img.shape[1]
        M = np.float32([[1,0,10],[0,1,10]])
        dst = cv2.warpAffine(img,M,(cols,rows))
        return dst
        cv2.imwrite("C:\\Python36\\ModifiedMNIST\\"+str(num)+".png",dst);

###############################ROTATION#########################################
def rotate(img,num):
        rows= img.shape[0]
        cols=img.shape[1]
        R= cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        rt = cv2.warpAffine(img,R,(cols,rows))
        return rt
        cv2.imwrite("C:\\Python36\\ModifiedMNIST\\"+str(num)+".png",rt);
        


#############################AFFINE###########################################
def affine(img):
        rows= img.shape[0]
        cols=img.shape[1]
        pts1 = np.float32([[5,5],[2,5],[5,2]])
        pts2 = np.float32([[1,10],[2,5],[10,2]])
        M = cv2.getAffineTransform(pts1,pts2)
        aff = cv2.warpAffine(img,M,(cols,rows))
        cv2.imwrite("test_affine.png",aff);

        
path="C:\\Python27\\Lib\\idlelib\\Test\\test_"

for i in range(1,10001,1):
	
		
	if i<10:
		
		
		imgnum="000"+str(i)
	elif i<100:
		imgnum="00"+str(i)
	elif i<1000:
		
		imgnum="0"+str(i)
	else:
		imgnum=""+str(i)
		

	img=cv2.imread(path+imgnum+".png")
	
	if i%2==0:
		
		translate(img,i)
		print(newi.shape)
		newi=newi.reshape((28,28))
	elif i%2==1:
		rotate(img,i)
		
	
	
