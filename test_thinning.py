import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import random
from ransac_map_merge import RANSAC_Map_Merging
from skimage.transform import radon, rescale


def _color_change(input_map):
    r, c = np.shape(input_map)
    for i in range(r):
        for j in range(c):
            if input_map[i, j]  > 150:
                input_map[i, j] = 0
            else:
                input_map[i, j] = 1
    return input_map

def _transform_state(input_map):
    r, c = np.shape(input_map)
    for i in range(r):
        for j in range(c):
            if input_map[i, j]  < 150:
                input_map[i, j] = 255
            else:
                input_map[i, j] = 0
    return input_map



def _thinning(img):
    
    r, c = np.shape(img)
    img_tmp = np.ones((r+2, c+2))
    img_tmp[1:r+1, 1:c+1] = img    
    for i in range(r):
        for j in range(c):
            x = i+1
            y = j+1
            if img_tmp[x,y] == 1:
               continue 
            p2 = img_tmp[x, y+1]
            p3 = img_tmp[x+1, y+1]
            p4 = img_tmp[x+1, y]
            p5 = img_tmp[x+1, y-1]
            p6 = img_tmp[x, y-1]
            p7 = img_tmp[x-1, y-1]
            p8 = img_tmp[x-1, y]
            p9 = img_tmp[x-1, y+1]
            B = p2+p3+p4+p5+p6+p7+p8+p9
            if  B <= 6 and B >= 2:
                pass
            else:
                continue 
            
            
            # if p2*p4*p6 == 0 and p4*p6*p8 == 0:
            #     pass
            # else:
            #     continue
            
            if p2*p4*p8 == 0 and p2*p6*p8 == 0:
                pass
            else:
                continue
            
            pts = [p2,p3,p4,p5,p6,p7,p8,p9]
            count = 0
            for i in range(7):
                if pts[i]== 0:
                    if pts[i+1]== 1:
                        count +=1
            if count == 1:
                img_tmp[x,y] = 1
                # print("delete")
            else:
                continue
    # for i in range(r):
    #     for j in range(c):
    #         x = i+1
    #         y = j+1
    #         if img_tmp[x,y] == 1:
    #            continue 
    #         p2 = img_tmp[x, y+1]
    #         p3 = img_tmp[x+1, y+1]
    #         p4 = img_tmp[x+1, y]
    #         p5 = img_tmp[x+1, y-1]
    #         p6 = img_tmp[x, y-1]
    #         p7 = img_tmp[x-1, y-1]
    #         p8 = img_tmp[x-1, y]
    #         p9 = img_tmp[x-1, y+1]
    #         B = p2+p3+p4+p5+p6+p7+p8+p9
    #         if  B <= 6 and B >= 2:
    #             pass
    #         else:
    #             continue 
            
            
    #         # if p2*p4*p6 == 0 and p4*p6*p8 == 0:
    #         #     pass
    #         # else:
    #         #     continue
            
    #         if p2*p4*p8 == 0 and p2*p6*p8 == 0:
    #             pass
    #         else:
    #             continue
            
    #         pts = [p2,p3,p4,p5,p6,p7,p8,p9]
    #         count = 0
    #         for i in range(7):
    #             if pts[i]== 0:
    #                 if pts[i+1]== 1:
    #                     count +=1
    #         if count == 1:
    #             img_tmp[x,y] = 1
    #             # print("delete")
    #         else:
    #             continue
            
    return img_tmp[1:r+1,1:c+1]

def _radon_find_theta(image):
    image = rescale(image, scale=1.0, mode='reflect', multichannel=False)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    x = np.where(sinogram==np.amax(sinogram))
    yaw = x[1][0]/sinogram.shape[1]*180
    return yaw, sinogram

def _rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If the center of the rotation is not defined, use the center of the image.
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img_rot = cv2.warpAffine(image, M, (w, h))
    r, c = np.shape(img_rot)
    for i in range(r):
        for j in range(c):
            if img_rot[i, j]  > 0:
                img_rot[i, j] = 255
            else:
                img_rot[i, j] = 0
    return img_rot

def _dilate_image(img, L=1):
    img_r, img_c = np.shape(img)
    img_tmp = np.ones((img_r+2*L, img_c+2*L))
    pts = np.where(img==255)
    _, length = np.shape(pts)
    for i in range(length):
        x = pts[0][i]+L
        y = pts[1][i]+L
        img_tmp[x-L:x+L+1, y-L:y+L+1] = 255
    # new_img = img_tmp[L:L+img_r, L:L+img_c]   
    new_img = img_tmp     
    return new_img.astype(np.uint8)

if __name__ == "__main__":
    img1 = cv2.imread('Gazebo_test_map//sigma_difference//0426_map_01.png')
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img1 = _transform_state(img1)

    if 1:
        fig, ax = plt.subplots()
        ax.imshow(img1, cmap='gray')
        ax.set_title("OGM - Original Image")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()


    dtheta_1, sino_1 = _radon_find_theta(img1)
    img1 = _rotate_image(img1, 270-dtheta_1)
    img1 = _dilate_image(img1, L=1)

    if 1:
        fig, ax = plt.subplots()
        ax.imshow(img1, cmap='gray')
        ax.set_title("OGM - Rotated Image")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()

    img1 = _color_change(img1)

    if 1:
        fig, ax = plt.subplots()
        ax.imshow(img1, cmap='gray')
        ax.set_title("OGM - Color changed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()

    img2 = img1
    for i in range(5):
        img1 = _thinning(img1)

    if 1:
        fig, ax = plt.subplots()
        ax.imshow(img2-img1, cmap='gray')
        ax.set_title("OGM - Thinning Difference")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()

    if 1:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM - Thinning")
        ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM - Dilation")
        ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()

    plt.show()

