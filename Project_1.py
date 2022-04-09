# -*- coding: utf-8 -*-
"""
@author: mehme
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt # for showing the histogram
    
def vertical_flip(img):

    height,width=img.shape
    same_image=[] # copy image

    for j in range(height): # take all rows
       same_image.append(img[j,:])
       
    new_image=[]  
    for a in range(0,len(same_image),1): # reverse all rows and write to a new list
         new_image.append(same_image[a][::-1]) 
    
    vertical = np.reshape(np.array(new_image), (375, 1242)) # reshaping
    return vertical

def horizontal_flip(img):
    height,width=img.shape
    same_image=[] # copy image
    
    
    for j in range(width): # take all columns
       same_image.append(img[:,j])
       
    new_image=[]  
    for a in range(0,len(same_image)): # reverse all columns and write to a new list
         new_image.append(same_image[a][::-1])
    
    List_flatten = []
    
    for j in range(height):  # flating the array according to pixel order
      for i in range (width):
        List_flatten.append(new_image[i][j])
    
    horizontal = np.reshape(np.array(List_flatten), (375, 1242)) # reshaping the flatten array
    return horizontal


#rotates the image 90 deg counterclockwise 
def rotate_90_counter(height, width, i, j):
    # reassign pixels
    i_new = -j + width - 1
    j_new = i
    return i_new, j_new

def rotate_90(height, width, i, j):
    # reassign pixels
    i_new = j 
    j_new = -i + height - 1
    return i_new, j_new


def negative(im):
    height=len(im)
    width = len(im[0])
    for row in range(height):
        for col in range(width):
            im[row][col] = 255 - im[row][col] # process of making negative 
    return im


def gamma_transformation(img):
    print("Please write gamma value \n")
    gamma = float(input()) # take gamme value
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8')# gamma process
    return gamma_corrected


def histogram_comp(img):
	image_height = img.shape[0]
	image_width = img.shape[1]
	
	histogram = np.zeros([256], np.int32) # pixel values are between 0-255 so there are 256 values.
	
	for x in range(0, image_height):
		for y in range(0, image_width):
			histogram[img[x,y]] +=1 # counter the histogram value which has "img[x,y]" value
	return histogram

def plot_histogram(Histogram):
	plt.figure()  # creating empty figure
	plt.title("Histogram Of The Image")
	plt.xlabel("Intensity Level")
	plt.ylabel("Intensity Frequency")
	plt.xlim([0, 256]) # x label limit
	plt.plot(Histogram) # plotting figure


def main():
    
  # read image  
  print("Please write absolute path of the image e.g. F:/yedek/Ders PDF/4.S/II/Image Processing/Homework1/image2.png \n")
  path = input()
  image_raw = cv2.imread(path) #read image
  gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY) # converts all images to GRAYSCALE
  image = np.copy(gray)

  cv2.imshow("Source image",image) # shows real image
  cv2.waitKey()
      
  vertical = vertical_flip(image)
  cv2.imshow("vertical",vertical)
  cv2.waitKey()
 
    
  horizontal = horizontal_flip(image)
  cv2.imshow("horizontal",horizontal)
  cv2.waitKey()
  
  rotate_90_counter_img = np.zeros([image.shape[1], image.shape[0]], dtype=image.dtype) # empty array that has dimensions of source image
  for i in range(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            newPoint = rotate_90_counter(image.shape[0], image.shape[1], i, j)
            rotate_90_counter_img[newPoint[0], newPoint[1]]=image[i,j] # assign values to empty array              
  cv2.imshow("90 deg counterclockwise",rotate_90_counter_img)
  cv2.waitKey()
  
  rotate_90_img = np.zeros([image.shape[1], image.shape[0]], dtype=image.dtype) # empty array that has dimensions of source image
  for i in range(image.shape[0]-1):
         for j in range(image.shape[1]-1):
             newPoint = rotate_90(image.shape[0], image.shape[1], i, j)
             rotate_90_img[newPoint[0], newPoint[1]]=image[i,j] # assign values to empty array              
  cv2.imshow("90 deg clockwise",rotate_90_img)
  cv2.waitKey()

  negative_img = negative(image)
  cv2.imshow("negative",negative_img)
  cv2.waitKey()

  gamma_trans = gamma_transformation(image)
  cv2.imshow("gamma transformation",gamma_trans)
  cv2.waitKey()
  
  histogram = histogram_comp(image)
  plot_histogram(histogram)

if __name__ == "__main__":
    main()