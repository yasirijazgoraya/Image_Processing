import cv2 as cv2
from matplotlib import pyplot as plt
import numpy as np
# create figure
fig = plt.figure(figsize=(10, 7))
rows = 4
columns = 3


img=cv2.imread("C://Users\SB00845233\Desktop\image_dataset\data\LinuxLogo.jpg")


fig.add_subplot(rows, columns, 1)
# showing image
plt.imshow(img)
plt.axis('off')
plt.title("Real image")

#dilation 

"""
dilation used to remove the noise
"""

#dilation = cv2.dilate(img, kernel, iterations=1)


kernal = np.ones((5,5), np.uint8)  



dilation = cv2.dilate(img, kernal, iterations=1)

fig.add_subplot(rows, columns, 2)
# showing image
plt.imshow(dilation)
plt.axis('off')
plt.title("dilation image")


"""
Erosion used to add pixel
"""



img_erosion = cv2.erode(img, kernal, iterations=1)

fig.add_subplot(rows, columns, 3)
# showing image
plt.imshow(img_erosion)
plt.axis('off')
plt.title("Erosion image")


"""
 Opening
Opening is just another name of erosion followed by dilation. It is useful in removing noise,
 as we explained above. Here we use the function,

"""

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernal)
fig.add_subplot(rows, columns, 4)
# showing image
plt.imshow(opening)
plt.axis('off')
plt.title("opening image")


"""
4. Closing
Closing is reverse of Opening, Dilation followed by Erosion.
 It is useful in closing small holes inside the foreground objects, or small black points on the object.
"""

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernal)
fig.add_subplot(rows, columns, 5)
# showing image
plt.imshow(closing)
plt.axis('off')
plt.title("closing image")


"""
Morphological Gradient
It is the difference between dilation and erosion of an image.

"""
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernal)

fig.add_subplot(rows, columns, 6)
# showing image
plt.imshow(gradient)
plt.axis('off')
plt.title("gradient image")


"""
Top Hat
It is the difference between input image and Opening of the image.
 Below example is done for a 9x9 kernel.
"""
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernal)

fig.add_subplot(rows, columns, 7)
# showing image
plt.imshow(tophat)
plt.axis('off')
plt.title("top_hat image")


"""
 Black Hat
It is the difference between the closing of the input image and input image.

"""
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernal)
fig.add_subplot(rows, columns, 8)
# showing image
plt.imshow(blackhat)
plt.axis('off')
plt.title("black_hat image")

