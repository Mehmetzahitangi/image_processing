import cv2
import numpy as np
import pylab as plt


def filtered(n):

  img_filter = np.array([(1, 2, 1), (2, 4, 2), (1, 2, 1)]) * (1/16)
 
  
  m = np.zeros([n,n])

  # convolution process with N*N filter (except 3*3)
  if n !=3 :
      
      m[0][0] = img_filter[0][0]
      m[0][int(np.floor(n/2))] = img_filter[0][1]
      m[0][n-1] = img_filter[0][2]
              
      m[int(np.floor(m.shape[0]/2))][0] = img_filter[1][0]
      m[int(np.floor(m.shape[0]/2))][int(np.floor(m.shape[0]/2))] = img_filter[1][1]
      m[int(np.floor(m.shape[0]/2))][n-1] = img_filter[1][2]
              
      m[n-1][0] = img_filter[2][0]
      m[n-1][int(np.floor(m.shape[0]/2))] = img_filter[2][1]
      m[n-1][n-1] = img_filter[2][2]
      

  #convolution process with 3*3 filter 
  elif n == 3:
      m = img_filter   
  return m

def my_conv(img, kernel):
  h = img.shape[0] 
  w = img.shape[1]
  
  # The result size of a convolution
  num_missing_pixels = kernel.shape[0]-1
  conv_size_x = img.shape[0]-num_missing_pixels
  conv_size_y = img.shape[1]-num_missing_pixels
  
  
  filter_img = np.empty([conv_size_x,conv_size_y])
  
  filter_ch1 = np.empty([conv_size_x,conv_size_y])
  
  kernel_size = kernel.shape[0]
  img_channel = img
  for i in range(h - num_missing_pixels):
     for j in range(w - num_missing_pixels):
         im_region = img_channel[i:(i + kernel_size), j:(j + kernel_size)]
         filter_img[i,j] = np.sum(im_region*kernel)

          
  filter_ch1 = filter_img


  return filter_ch1




def imhist(im):

	m, n = im.shape
	hist = [0.0] * 256
	for i in range(m):
		for j in range(n):
			hist[im[i, j]]+=1
	return np.array(hist)/(m*n)

def cumsum(h):

	return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):

	hist_original = imhist(im)
	cumm_df = np.array(cumsum(hist_original)) 
	sk = np.uint8(255 * cumm_df) 
	s1, s2 = im.shape
	new_hist_im = np.zeros_like(im)

	for i in range(0, s1):
		for j in range(0, s2):
			new_hist_im[i, j] = sk[im[i, j]]
	new_hist = imhist(new_hist_im)

	return new_hist_im , hist_original, new_hist

def main():
    
  # read image  
  print("Please write absolute path of the image e.g. F:/yedek/Ders PDF/4.S/II/Image Processing/Homework2/image2.png \n")
  path = input()
  
  img_raw = cv2.imread(path,0)
  
  new_img, h, new_h = histeq(img_raw)

  #plotting histograms
  plt.subplot(121)
  plt.imshow(img_raw)
  plt.title('original image')


  plt.subplot(122)
  plt.imshow(new_img)
  plt.title('hist. equalized image')
  plt.show()
    

  fig = plt.figure()
  fig.add_subplot(221)
  plt.plot(h)
  plt.title('Original')
    
  fig.add_subplot(222)
  plt.plot(new_h)
  plt.title('histogram of equalized image') 
  plt.show()
   
  
  kernel_x= filtered(int(5))  
  img = my_conv(img_raw, kernel_x)  
  

  im1 = cv2.blur(img_raw,(5,5))  
  cv2.imshow('filtered image with built-in function', im1)
  cv2.imshow("filtered image with my own algorithm",img/255)
  cv2.waitKey()
  cv2.destroyAllWindows()


if __name__ == "__main__":
    main()