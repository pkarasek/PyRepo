import cv2
import numpy as np
def noVal(val):
    pass
identity_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
edge_kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
#gaussian_kernel = np.array([[1,2,1],[2,4,2],[1,2,1]] , np.float32) / 16
gaussian_kernel1 = cv2.getGaussianKernel(3, 0)
gaussian_kernel2 = cv2.getGaussianKernel(5, 0)
emboss_kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
multi_kernel = np.array([[-4,8,0],[0,-80,0],[0,8,-4]])*2
kernels = [identity_kernel, gaussian_kernel1, gaussian_kernel2, multi_kernel, emboss_kernel, sharpen_kernel, edge_kernel]
color_original = cv2.imread('test.jpg')
color_modified = color_original.copy()
gray_original = cv2.cvtColor(color_original, cv2.COLOR_BGR2GRAY)
gray_modified = gray_original.copy()
cv2.namedWindow('pKaras - Filter Application')
cv2.createTrackbar('Kontrast', 'pKaras - Filter Application', 1, 100, noVal)
cv2.createTrackbar('Jasnosc', 'pKaras - Filter Application', 50, 100, noVal)
cv2.createTrackbar('Rodzaj Filtru', 'pKaras - Filter Application', 0, len(kernels)-1, noVal)
cv2.createTrackbar('Monochrom', 'pKaras - Filter Application', 0, 1, noVal)
counter = 1
while True:
    grayscale = cv2.getTrackbarPos('Monochrom', 'pKaras - Filter Application')
    if grayscale == 0:
        cv2.imshow('pKaras - Filter Application', color_modified)
    else:
        cv2.imshow('pKaras - Filter Application', gray_modified)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('s'):
        if grayscale == 0:
            cv2.imwrite('output%d.png'% counter, color_modified)
        else:
            cv2.imwrite('output%d.png'% counter, gray_modified)
    counter += 1
    contrast = cv2.getTrackbarPos('Kontrast', 'pKaras - Filter Application')
    brightness = cv2.getTrackbarPos('Jasnosc', 'pKaras - Filter Application')
    kernel = cv2.getTrackbarPos('Rodzaj Filtru','pKaras - Filter Application')
    if(grayscale == 0):
        color_modified = cv2.filter2D(color_original, -1, kernels[kernel])
        color_modified = cv2.addWeighted(color_modified,contrast, np.zeros(color_original.shape, dtype=color_original.dtype), 0, brightness-50)
    else:
        gray_modified = cv2.filter2D(gray_original, -1, kernels[kernel])
        gray_modified = cv2.addWeighted(gray_modified,contrast, np.zeros(gray_original.shape, dtype=gray_original.dtype), 0, brightness-50)
    color_modified = cv2.filter2D(color_original, -1, kernels[kernel])
    color_modified = cv2.addWeighted(color_modified,contrast, np.zeros(color_original.shape, dtype=color_original.dtype), 0, brightness-50)
cv2.destroyAllWindows()
