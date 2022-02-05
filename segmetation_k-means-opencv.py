import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Data/BSE-image-of-charnockite.jpg")


# Convert MxNx3 image into Kx3 where K=MxN acording to opencv documentation
img2 = img.reshape((-1,3))  #-1 reshape means, in this case MxN

#We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
img2 = np.float32(img2)

#Define criteria, number of clusters and apply k-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Number of clusters
k = 4

attempts = 10

#other flags needed
#Two options, cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS

ret,label,center=cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

center = np.uint8(center) 

#Next, we have to access the labels to regenerate the clustered image
reshaped = center[label.flatten()]
reshaped2 = res.reshape((img.shape)) #Reshape labels to the size of original image
cv2.imwrite("Data/segmented.jpg", reshaped2)


#Now let us visualize the output result
figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img2)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(res2)
plt.title('Segmented Image when K = %i' % k), plt.xticks([]), plt.yticks([])
plt.show()
