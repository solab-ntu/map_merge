# -*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import random
import copy
from skimage.transform import radon, rescale
import matplotlib.gridspec as gridspec


def _transform_state(input_map):
    r, c = np.shape(input_map)
    for i in range(r):
        for j in range(c):
            if input_map[i, j]  < 150:
                input_map[i, j] = 255
            else:
                input_map[i, j] = 0
    return input_map


def _dilate_image(img, L=1):
    img_r, img_c = np.shape(img)
    img_tmp = np.zeros((img_r+2*L, img_c+2*L))
    pts = np.where(img==255)
    _, length = np.shape(pts)
    for i in range(length):
        x = pts[0][i]+L
        y = pts[1][i]+L
        img_tmp[x-L:x+L+1, y-L:y+L+1] = 255
    # new_img = img_tmp[L:L+img_r, L:L+img_c]  
    new_img = img_tmp    
    return new_img.astype(np.uint8)

def _erode_number(img, L=1):
    img_r, img_c = np.shape(img)
    img_tmp = np.zeros((img_r+2*L,img_c+2*L))
    img_tmp[L:L+img_r, L:L+img_c] = img
    new_img = np.zeros((img_r, img_c))
    pts = np.where(img==255)
    _, length = np.shape(pts)
    # B1 = np.ones((2*L+1,2*L+1))*255
    for i in range(length):
        x = pts[0][i]+L
        y = pts[1][i]+L
        
        window = img_tmp[x-L:x+L+1, y-L:y+L+1]   
        number = len(np.where(window==255)[0])

        if number/(2*L+1)**2 > 0.7:
            new_img[x-L, y-L] = 255
    return new_img.astype(np.uint8)

def _erode_blur(img, L=1):
    img_r, img_c = np.shape(img)
    img_tmp = np.zeros((img_r+2*L,img_c+2*L))
    img_tmp[L:L+img_r, L:L+img_c] = img
    new_img = np.zeros((img_r, img_c))
    pts = np.where(img==255)
    _, length = np.shape(pts)

    x, y = np.mgrid[-1:2*L, -1:2*L]
    kernel = np.exp(-(x**2+y**2))
    kernel = kernel / kernel.sum()
    

    for i in range(length):
        x = pts[0][i]+L
        y = pts[1][i]+L
        
        window = img_tmp[x-L:x+L+1, y-L:y+L+1]   
        ratio = (window * kernel).sum()/255.0

        if ratio > 0.7:
            new_img[x-L, y-L] = 255
    return new_img.astype(np.uint8)

def _erode_image(img, L=1):
    """
    基本的侵蝕。
    """
    img_r, img_c = np.shape(img)
    img_tmp = np.zeros((img_r+2*L,img_c+2*L))
    img_tmp[L:L+img_r, L:L+img_c] = img
    new_img = np.zeros((img_r, img_c))
    pts = np.where(img==255)
    _, length = np.shape(pts)
    B1 = np.ones((2*L+1,2*L+1))*255
    for i in range(length):
        x = pts[0][i]+L
        y = pts[1][i]+L
        
        window = img_tmp[x-L:x+L+1, y-L:y+L+1]   
        flag = window==B1
        if flag.all():
            new_img[x-L, y-L] = 255
    return new_img.astype(np.uint8)

def _erode_reverse(img, L=1):
    """
    直接找到邊界的侵蝕。
    """
    img_r, img_c = np.shape(img)
    img_tmp = np.zeros((img_r+2*L,img_c+2*L))
    img_tmp[L:L+img_r, L:L+img_c] = img
    new_img = np.zeros((img_r, img_c))
    pts = np.where(img==255)
    _, length = np.shape(pts)
    B1 = np.ones((2*L+1,2*L+1))*255
    for i in range(length):
        x = pts[0][i]+L
        y = pts[1][i]+L
        
        window = img_tmp[x-L:x+L+1, y-L:y+L+1]   
        flag = window==B1
        if not flag.all():
            new_img[x-L, y-L] = 255
    return new_img.astype(np.uint8)

def _erode_corner(img, L=1, mode=0):
    img_r, img_c = np.shape(img)
    img_tmp = np.zeros((img_r+2*L,img_c+2*L))
    img_tmp[L:L+img_r, L:L+img_c] = img
    new_img = np.zeros((img_r, img_c))
    pts = np.where(img==255)
    _, length = np.shape(pts)
    
    B1 = np.zeros((2*L+1,2*L+1))
    B1[L:2*L+1, L:2*L+1]=255
    
    B2 = np.zeros((2*L+1,2*L+1))
    B2[0:L+1, L:2*L+1]=255

    B3 = np.zeros((2*L+1,2*L+1))
    B3[L:2*L+1, 0:L+1]=255

    B4 = np.zeros((2*L+1,2*L+1))
    B4[0:L+1, 0:L+1]=255

    for i in range(length):
        x = pts[0][i]+L
        y = pts[1][i]+L
        
        window = img_tmp[x-L:x+L+1, y-L:y+L+1]   
        f1 = window==B1
        f2 = window==B2
        f3 = window==B3
        f4 = window==B4

        if mode == 1:
            if f1[L:2*L+1, L:2*L+1].all():
                new_img[x-L, y-L] = 255
        elif mode == 2:
            if f2[0:L+1, L:2*L+1].all():
                new_img[x-L, y-L] = 255
        elif mode == 3:
            if f3[L:2*L+1, 0:L+1].all():
                new_img[x-L, y-L] = 255
        elif mode == 4:
            if f4[0:L+1, 0:L+1].all():
                new_img[x-L, y-L] = 255
        else:
            if f1[L:2*L+1, L:2*L+1].all() or f2[0:L+1, L:2*L+1].all() or f4[0:L+1, 0:L+1].all() or f3[L:2*L+1, 0:L+1].all():
                new_img[x-L, y-L] = 255
    return new_img.astype(np.uint8)

def _erode_cross_1(img, L=1):
    img_r, img_c = np.shape(img)
    img_tmp = np.zeros((img_r+2*L,img_c+2*L))
    img_tmp[L:L+img_r, L:L+img_c] = img
    new_img = np.zeros((img_r, img_c))
    pts = np.where(img==255)
    _, length = np.shape(pts)
    
    B1 = np.zeros((2*L+1,2*L+1))
    B1[L:2*L+1, L]=255
    
    B2 = np.zeros((2*L+1,2*L+1))
    B2[0:L+1, L]=255

    B3 = np.zeros((2*L+1,2*L+1))
    B3[L, 0:L+1]=255

    B4 = np.zeros((2*L+1,2*L+1))
    B4[L, L:2*L+1]=255

    for i in range(length):
        x = pts[0][i]+L
        y = pts[1][i]+L
        
        window = img_tmp[x-L:x+L+1, y-L:y+L+1]   
        f1 = window==B1
        f2 = window==B2
        f3 = window==B3
        f4 = window==B4
        if f1[L:2*L+1, L].all() or f2[0:L+1, L].all() or f4[L, L:2*L+1].all() or f3[L, 0:L+1].all():
            new_img[x-L, y-L] = 255
    return new_img.astype(np.uint8)

def _erode_line(img, L=1):
    img_r, img_c = np.shape(img)
    img_tmp = np.zeros((img_r+2*L,img_c+2*L))
    img_tmp[L:L+img_r, L:L+img_c] = img
    new_img = np.zeros((img_r, img_c))
    pts = np.where(img==255)
    _, length = np.shape(pts)
    
    B1 = np.zeros((2*L+1,2*L+1))
    # B1[L,:]=255
    B1[:,L]=255
    for i in range(length):
        x = pts[0][i]+L
        y = pts[1][i]+L
        
        window = img_tmp[x-L:x+L+1, y-L:y+L+1]   
        flag = window==B1
        # if flag[L,:].all():
        if flag[:,L].all():
            new_img[x-L, y-L] = 255
    return new_img.astype(np.uint8)

def _erode_test(img, L=1):
    img_r, img_c = np.shape(img)
    img_tmp = np.zeros((img_r+2*L,img_c+2*L))
    img_tmp[L:L+img_r, L:L+img_c] = img
    new_img = np.zeros((img_r, img_c))
    pts = np.where(img==255)
    _, length = np.shape(pts)
    
    B1 = np.ones((2*L+1,2*L+1))*255
    
    for i in range(length):
        x = pts[0][i]+L
        y = pts[1][i]+L
        
        window = img_tmp[x-L:x+L+1, y-L:y+L+1]   
        flag = window==B1
        if flag.all():
            new_img[x-L, y-L] = 255
            img_tmp[x-L:x+L+1, y-L:y+L+1] = 0
    return new_img.astype(np.uint8)


def _erode_cross(img, L=1):
    img_r, img_c = np.shape(img)
    img_tmp = np.zeros((img_r+2*L,img_c+2*L))
    img_tmp[L:L+img_r, L:L+img_c] = img
    new_img = np.zeros((img_r, img_c))
    pts = np.where(img==255)
    _, length = np.shape(pts)
    
    B1 = np.zeros((2*L+1,2*L+1))
    B1[L,:]=255
    B1[:,L]=255
    for i in range(length):
        x = pts[0][i]+L
        y = pts[1][i]+L
        
        window = img_tmp[x-L:x+L+1, y-L:y+L+1]   
        flag = window==B1
        if flag[L,:].all() and flag[:,L].all():
            new_img[x-L, y-L] = 255
    return new_img.astype(np.uint8)

def _radon_find_theta(image):
    image = rescale(image, scale=1.0, mode='reflect', multichannel=False)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    x = np.where(sinogram==np.amax(sinogram))
    yaw = x[1][0]/sinogram.shape[1]*180
    return yaw

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


def _image_boundary_follow(image, L=10):
    img_r, img_c = np.shape(image)
    img = np.zeros((img_r+2*L, img_c+2*L))
    img[L:img_r+L, L:img_c+L] = image
    new_img = np.zeros((img_r+2*L, img_c+2*L))    
    pts = np.where(img==255)
    _, length = np.shape(pts)

    direction = None
    times = 0
    x = pts[0][0]
    y = pts[1][0]
    # 先用有限次數的迭代
    while times < 100000 :
        times += 1
        img[x, y] = 0    
        new_img[x-1:x+2, y-1:y+2] = 255

        window = img[x-L:x+L+1, y-L:y+L+1]
        window_pts = np.where(window==255)
        

        if window_pts[0].shape[0]!=0:
            
            index = np.argmin( abs(window_pts[0] - L) + abs(window_pts[1] - L) )
            dx = window_pts[0][index] - L
            dy = window_pts[1][index] - L

            abs_dx = abs(dx)
            abs_dy = abs(dy)


            if abs_dx > abs_dy: # go down
                img[x+dx, y+dy] = 0
                if dx > 0:
                    new_img[x:x+dx, y-1:y+2] = 255
                else:
                    new_img[x+dx:x, y-1:y+2] = 255
                x = x + dx
                y = y
                direction = 'vertical'

            elif abs_dy > abs_dx: # go right
                img[x+dx, y+dy] = 0
                if dy > 0:
                    new_img[x-1:x+2, y:y+dy] = 255
                else: 
                    new_img[x-1:x+2, y+dy:y] = 255
                x = x
                y = y + dy
                direction = 'horizontal'


            else:
                print("here", x, y)
                if direction == 'vertical':
                    if dx > 0:
                        new_img[x:x+dx, y-1:y+2] = 255
                    else:
                        new_img[x+dx:x, y-1:y+2] = 255
                    x = x + dx
                    y = y

                elif direction == 'horizontal':
                    if dy > 0:
                        new_img[x-1:x+2, y:y+dy] = 255
                    else: 
                        new_img[x-1:x+2, y+dy:y] = 255
                    x = x
                    y = y + dy

                else:
                    pts = np.where(img==255)
                    x = pts[0][0]
                    y = pts[1][0]
                
        else:
            # change to next segment.
            print("change~", x,y)
            pts = np.where(img==255)
            _, length = np.shape(pts)
            if length == 0:
                print("break")
                break
            
            
            x = pts[0][0]
            y = pts[1][0]
            
    
    return new_img.astype(np.uint8)

    


def _iterate_process(img, L=1):
    img_this = img
    img_last = np.zeros(np.shape(img))
    times = 0
    while not (img_last==img_this).all():
        img_last = copy.copy(img_this)
        img_e = _erode_blur(img=img_this, L=L)
        img_this = _dilate_image(img=img_e, L=L)

        times += 1
        if times > 20:
            print("too much times...")
            break
    
    return img_this

if __name__ == "__main__":
    
    # img1 = cv2.imread('Gazebo_test_map//tesst2.png')
    img1 = cv2.imread('Gazebo_test_map//test0223_2.png')

    ## grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    ## binary
    img1 = _transform_state(img1)


    # plt.figure()
    # plt.imshow(img1, cmap='gray')
    # plt.title('binary')

    ## dilate
    img1 = _dilate_image(img1, L=2)
    
    # plt.figure()
    # plt.imshow(img1, cmap='gray')
    # plt.title('dilate')

    ## rotate
    dtheta_1 = _radon_find_theta(img1)
    img1 = _rotate_image(img1, 90-dtheta_1)
    
    plt.figure()
    plt.imshow(img1, cmap='gray')
    plt.title('rotation')
    plt.tight_layout()

    # img_rev = _erode_reverse(img1)
    # plt.figure()
    # plt.imshow(img_rev, cmap='gray')
    # plt.title('reverse')
    # plt.tight_layout()
    
    # img_it = _iterate_process(img1, L=1)
    img_it = _erode_test(img1, L=2)
    
    
    plt.figure()
    plt.imshow(img_it, cmap='gray')
    plt.title('erode_number')
    
    
    img_tt = _image_boundary_follow(img_it, L=20)
    plt.figure()
    plt.imshow(img_tt, cmap='gray')
    plt.tight_layout()

    # img_e = _erode_image(img1, L=1) 
    # plt.figure()
    # plt.subplot(211), plt.imshow(img_e, cmap='gray')
    # plt.title('erode')
    # plt.subplot(212), plt.imshow(img1 - img_e, cmap='gray')
    # plt.tight_layout()

    # img_this = np.zeros((np.shape(img1)))
    # img_last = 1
    # times = 0
    # while not (img_this == img_last).all():
    #     times += 1
    #     img_last = copy.copy(img_this)
    #     img_this = _dilate_image(_erode_image(img1, L=1), 1)
    #     if (img_this == img_last).all():
    #         break
    #     if times > 20:
    #         print("break mech")
    #         break

    # plt.figure()
    # plt.imshow(img_this, cmap='gray')

    # img_corner = _erode_corner(img1, L=1)
    # plt.figure()
    # plt.subplot(211), plt.imshow(img_corner, cmap='gray')
    # plt.title('erode_corner1')
    # plt.subplot(212), plt.imshow(img1 - img_corner, cmap='gray')
    # plt.tight_layout()

    # img_cross_1 = _erode_cross_1(img1, L=1)
    # plt.figure()
    # plt.subplot(211), plt.imshow(img_cross_1, cmap='gray')
    # plt.title('erode_cross1')
    # plt.subplot(212), plt.imshow(img1 - img_cross_1, cmap='gray')
    # plt.tight_layout()

    # img_tmp = np.zeros((np.shape(img1)))
    # for i in range(5):
    #     img_corner = _erode_corner(img1, L=1, mode=i)
    #     plt.figure()
    #     plt.subplot(211), plt.imshow(img_corner, cmap='gray')
    #     plt.title('erode_corner_mode_%i'%i)
    #     plt.subplot(212), plt.imshow(img1 - img_corner, cmap='gray')
    #     plt.tight_layout()
    #     img_this = _erode_cross_1(img1 - img_corner)
    #     img_tmp += img_this

    # r, c = np.shape(img_tmp)
    # for i in range(r):
    #     for j in range(c):
    #         if img_tmp[i,j] != 0:
    #             img_tmp[i,j] = 255

    # plt.figure()
    # plt.imshow(img_tmp -(img1 -  img_e), cmap='gray')

    plt.show()