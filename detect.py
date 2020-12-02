import numpy as np
import os
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math

arr = os.listdir("F:\\GreenParking")

arrX = [1,-1,1,0,-1,0,1,-1]
arrY = [1,-1,0,1,0,-1,-1,1]

#chuyển về ảnh xám	
def rgb2gray(rgb):
	(h, w, d) = rgb.shape
	img_gray = []
	for i in range(0,h):
		gray = []
		for j in range(0,w):
			r = rgb[i,j,0]
			g = rgb[i,j,1]
			b = rgb[i,j,2]
			tmp = 0.2989 * r + 0.5870 * g + 0.1140 * b
			gray.append(tmp)
		img_gray.append(gray)
	return img_gray
	
#threshold lấy ngưỡng để chuyển thành ảnh đen trắng
def threshold(gray, x):
	(h, w) = gray.shape
	thres = []
	for i in range(0,h):
		line = []
		for j in range(0,w):
			if (gray[i,j] >= x):
				tmp = 255
			else:
				tmp = 0
			line.append(tmp)
		thres.append(line)
	thres = np.uint8(thres)
	return thres

#Bộ lọc dilate làm dày cạnh
def dilate3(binary):
	check = []
	(h,w) = binary.shape
	for i in range(0,h):
		line = []
		for j in range(0,w):
			line.append(0)
		check.append(line)

	for i in range(1,h-1):
		for j in range(1,w-1):
			if (binary[i,j] == 0):
				for k in range(0,8):
					ii = 0
					jj = 0
					ii = i + arrX[k]
					jj = j + arrY[k]
					if (binary[ii,jj] == 255) and (check[ii][jj] == 0):
						binary[i,j] = 255
						check[i][j] = 1
	return binary


#convolution/nhân ma trận
def convolution(image, kernel):
    if len(image.shape) == 3:
        image = rgb2gray(image)

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

    return output

#Bộ lọc sobel (theo chiều dọc)
filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

def sobel_edge_detection(image, filter):

	#nhân ma trận hình ảnh với ma trận sobel chiều dọc
    new_image_x = convolution(image, filter) 

    #nhân ma trận hình ảnh với ma trận chiều ngang
    new_image_y = convolution(image, np.flip(filter.T, axis=0)) 

    #Kết hợp cạnh ngang và dọc
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    gradient_magnitude *= 255 / gradient_magnitude.max()

    return gradient_magnitude

#Tính khoảng cách giữa 2 điểm
def distance(a,b):
		x = round(math.sqrt(pow((a[0]-b[0]),2) + pow((a[1]-b[1]),2)))
		return x

#Process
for apart in arr:
	print("processing "+apart)
	im = cv2.imread("F:\\GreenParking\\"+apart)
	output = "F:\\VUONG\\MCN\\output\\"+apart

	#chuyển thành ảnh xám
	im_gray = rgb2gray(im)

	#chuyển thành ảnh đen trắng
	im_gray = threshold(im_gray,150)
	
	#lọc cạnh sobel
	sobel_image = sobel_edge_detection(im_gray,filter)

	#dùng ngưỡng để chuyển thành đen trắng
	sobel_image = threshold(sobel_image,150)

	#làm dày các cạnh
	dilated_image = dilate3(sobel_image)

	#tìm các hình học
	contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	screenCnt = None
	NumberPlateCnt = []

	#Với mỗi cạnh được phát hiện trong ảnh
	for c in contours:

		#tính độ dài cạnh
	    peri = cv2.arcLength(c, True)

	    #kiểm tra xem với độ sai lệch 0.09*peri có hình học kín nào thỏa mãn
	    approx = cv2.approxPolyDP(c, 0.09 * peri, True) 

	    #nếu hình đó có 4 cạnh
	    if len(approx) == 4:
	    	len_contour = []

	    	#tính khoảng cách từ 1 đỉnh đến các đỉnh còn lại
	    	for i in range(1,4):
	    		tmp = distance(approx[0][0], approx[i][0])
	    		len_contour.append(tmp)

	    	#sắp xếp các chiều dài tính được để lấy 2 cạnh ngắn nhất
	    	len_contour = sorted(len_contour)

	    	#tính tỉ lệ giữa các cạnh phù hợp kích thước biển số xe máy
	    	tmp = len_contour[0]/len_contour[1]
	    	if (tmp >= 0.6) and (tmp <= 1):
	    		#chiều dài rộng trong một hình ảnh thỏa mãn
	    		if (len_contour[0] > 30) and (len_contour[1] < 90):
	    			NumberPlateCnt.append(approx)

	#vẽ khung lên hình ảnh
	for screenCnt in NumberPlateCnt:
		cv2.drawContours(im, [screenCnt], 0, (0,255,0), 3)
	cv2.imwrite(output, im)