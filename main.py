import cv2
import matplotlib.pyplot as plt
'''Here by default, OpenCV reads the image in BGR format. If you want to display it correctly using Matplotlib,
 you need to convert it to RGB format. However, if you want to display the image as it is without conversion, 
 you can directly use the BGR format with Matplotlib, but it may not display the colors correctly.'''
image = cv2.imread('data/ROAD.png')
#plt.imshow(image)
#plt.title("Original Image")
#plt.axis('off')
#plt.show()
#BGR to RGB conversion
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#plt.imshow(image_rgb)
#plt.title("Satellite Image")
#plt.axis('off')
#plt.show()
#Converting the image into grayscale
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#plt.imshow(gray_image,cmap='gray')
#plt.title("Grayscale Image")
#plt.axis('off')
#plt.show()
'''As we will be using the Canny edge detection algorithm, we need to apply
 a Gaussian blur to the image to reduce noise and improve edge detection results.
  Because the Canny edge detection algorithm is sensitive to noise, applying a Gaussian blur
    helps to smooth the image and reduce the impact of noise on edge detection.'''
'''Without applying Gaussian blur, the Canny edge detection algorithm may produce more
    false edges due to noise in the image.'''
gaussian_blur = cv2.GaussianBlur(gray_image, (5,5), 0)
#plt.imshow(gaussian_blur, cmap='gray')
#plt.title("Gaussian Blurred Image")
#plt.axis('off')
#plt.show()
edges = cv2.Canny(gaussian_blur, 50, 150)
#plt.imshow(edges, cmap='gray')
#plt.title("Canny Edges")
#plt.axis('off')
#plt.show()
'''Now as  we have done the canny edge detection, we have to do shape processing dilation and erosion to get the better result.'''
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
dilated_edges=cv2.dilate(edges,kernel,iterations=1)
#plt.imshow(dilated_edges,cmap='gray')
#plt.title("Dilated Edges")
#plt.axis('off')
#plt.show()

'''Plotting all the images at once'''
plt.figure(figsize=(12,8))

# Original RGB
plt.subplot(2,3,1)
plt.imshow(image_rgb)
plt.title("Original RGB")
plt.axis('off')

# Grayscale
plt.subplot(2,3,2)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale")
plt.axis('off')

# Gaussian Blur
plt.subplot(2,3,3)
plt.imshow(gaussian_blur, cmap='gray')
plt.title("Gaussian Blur")
plt.axis('off')

# Canny Edge Detection
plt.subplot(2,3,4)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge")
plt.axis('off')

# Dilated Edges basically expands the edges to make them more visible and connected, which can be useful for further processing or visualization.
plt.subplot(2,3,5)
plt.imshow(dilated_edges, cmap='gray')
plt.title("Dilated Edges")
plt.axis('off')

plt.tight_layout()
plt.show()
#plt.savefig('screenshots/combined_pipeline.png')

'''contour algorithm used after dilation to find contours in the dilated image. As dilated image has continuous boundaries
 than canny edge image,it helps to find contours more effectively.
 Contour basically means the outer boundary of the object. It is used to find the shape of the object in the image.'''

'''First parameter is the input image, second parameter is the contour retrieval mode, third parameter is the contour approximation method.
   The contour retrieval mode specifies how the contours are retrieved from the image and gives importance
   to outer boundaries only and ignores inside boundaries. The contour approximation method specifies
   how the contours are approximated like basically storage optimization(stores essential points) 
   and the [0] at the end is used to get the contours from the output of the function.'''

contours=cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
#Drawing contours on the original image
contour_image=image_rgb.copy()
''' here -1 means that we want to draw all the contours, (0,255,0) is the color of the contour in RGB format and 
    2 is the thickness of the contour line.'''
cv2.drawContours(contour_image, contours, -1, (0,255,0), 2)
plt.imshow(contour_image)
plt.title("Contours on Original Image")
plt.axis('off')
plt.show()
for contour in contours:
    area=cv2.contourArea(contour)
    if area>100: # Filter out small contours based on area
        x,y,w,h=cv2.boundingRect(contour)
        '''Draw bounding box around the contour (x,y) is the top-left corner of the bounding box and 
        (x+w,y+h) is the bottom-right corner of the bounding box. (255,0,0) is the color of the bounding box in RGB format and
        2 is the thickness of the bounding box line.'''
        cv2.rectangle(image_rgb,(x,y),(x+w,y+h),(255,0,0),2)
plt.imshow(image_rgb)
plt.title("Bounding Boxes on Original Image")
plt.axis('off')
plt.show()