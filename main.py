import cv2
import matplotlib.pyplot as plt
'''Here by default, OpenCV reads the image in BGR format. If you want to display it correctly using Matplotlib,
 you need to convert it to RGB format. However, if you want to display the image as it is without conversion, 
 you can directly use the BGR format with Matplotlib, but it may not display the colors correctly.'''
image = cv2.imread('data/ROAD.png')
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')
plt.show()
#BGR to RGB conversion
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title("Satellite Image")
plt.axis('off')
plt.show()
#Converting the image into grayscale
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image,cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()
'''As we will be using the Canny edge detection algorithm, we need to apply
 a Gaussian blur to the image to reduce noise and improve edge detection results.
  Because the Canny edge detection algorithm is sensitive to noise, applying a Gaussian blur
    helps to smooth the image and reduce the impact of noise on edge detection.'''
'''Without applying Gaussian blur, the Canny edge detection algorithm may produce more
    false edges due to noise in the image.'''
gaussian_blur = cv2.GaussianBlur(gray_image, (5,5), 0)
plt.imshow(gaussian_blur, cmap='gray')
plt.title("Gaussian Blurred Image")
plt.axis('off')
plt.show()
edges = cv2.Canny(gaussian_blur, 50, 150)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edges")
plt.axis('off')
plt.show()
'''Now as  we have done the canny edge detection, we have to do shape processing dilation and erosion to get the better result.'''
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
dilated_edges=cv2.dilate(edges,kernel,iterations=1)
plt.imshow(dilated_edges,cmap='gray')
plt.title("Dilated Edges")
plt.axis('off')
plt.show()