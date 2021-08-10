# -*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import random
import copy
import matplotlib.gridspec as gridspec
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
from ransac_map_merge import MAP_Function, RANSAC_Map_Merging
from itertools import combinations, permutations
from icp_map_merge import ScanMatching
from sklearn.cluster import KMeans
import csv
from icp_map_merge import ScanMatching

def _merging_map(dx, dy, dtheta, dr, map1, map2):
    
    map2 = cv2.resize(map2, (0, 0), fx=1/dr, fy=1/dr, interpolation=cv2.INTER_CUBIC)

    r1, c1 = np.shape(map1)
    r2, c2 = np.shape(map2)
    extension = max(r2,c2)
    map_temp = np.ones(( r1 + extension*2 , c1 + extension*2 )) * 0
    map_temp[extension:extension+r1 , extension:extension+c1] = map1

    dx = dx/dr
    dy = dy/dr
    # 下面這步是為了先將Map1 和Map2的原點貼齊，否則Map2一開始會在擴充地圖的左上角。
    dx = dx + extension
    dy = dy + extension
    for row in range(r2):
        for col in range(c2):

            new_x = int( (math.cos(dtheta)*row - math.sin(dtheta)*col + dx))
            new_y = int( (math.sin(dtheta)*row + math.cos(dtheta)*col + dy))

            if new_x >= extension and new_x <= extension + r1 and new_y >= extension and new_y <= extension + c1:
                # Overlap area, use Entropy filter to determine whether adopting mixed probability or not.
                prob1 = (map_temp[new_x, new_y]+1)/257.0  # -> equals to map1 prob
                prob2 = (map2[row, col]+1)/257.0          # -> map2 prob
                
                prob = MAP_Function()._entropy_filter(prob1, prob2)
                map_temp[new_x, new_y] = prob*257-1

            else:
                map_temp[new_x, new_y] = map2[row, col]

    return map_temp

def _modify_map_size(merged_map):
    """
    After merging maps, size of the merged map is very large due to the extension.\n
    This function is designed to reduce parts of extension areas.\n
    """
    pos_x_white, pos_y_white = np.where(merged_map == 255)
    # pos_x_black, pos_y_black = np.where(merged_map == 0)

    pos_x_M = np.amax(pos_x_white)
    pos_x_m = np.amin(pos_x_white)
    pos_y_M = np.amax(pos_y_white)
    pos_y_m = np.amin(pos_y_white)

    reduced_map = merged_map[pos_x_m-5:pos_x_M+5, pos_y_m-5:pos_y_M+5]

    return reduced_map


if __name__ == "__main__":

    img1 = cv2.imread('test1.png')
    img2 = cv2.imread('test2.png')

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    x1, y1 = np.where(img1 == 255)
    pts1 = np.vstack((x1,y1)).T
    x2, y2 = np.where(img2 == 255)
    pts2 = np.vstack((x2,y2)).T

    SM = ScanMatching()
    _, pts, [x, y, yaw] = SM.icp(   reference_points=pts1, 
                                    points=pts2, 
                                    max_iterations=5000, 
                                    distance_threshold=10, 
                                    convergence_translation_threshold=1e-5, 
                                    convergence_rotation_threshold=1e-3, 
                                    point_pairs_threshold=2000, 
                                    verbose=False)
    print(x,y,yaw)
    map_f = _merging_map(dx=x, dy=y, dtheta=yaw, dr=1, map1=img1, map2=img2)

    r, c = np.shape(map_f)
    for i in range(r):
        for j in range(c):
            if map_f[i, j]  > 10:
                map_f[i, j] = 255
            else:
                map_f[i, j] = 0
    
    # map_f = map_f.astype(np.uint8)
    map_f = _modify_map_size(merged_map=map_f)

    # fig, ax = plt.subplots(3, 1)
    # ax[0].imshow(map_f, cmap='gray')
    # ax[1].imshow(img1, cmap='gray')
    # ax[2].imshow(img2, cmap='gray')
   
    fig = plt.figure() 
    gs = gridspec.GridSpec(2, 4) 
    

    ax1 = plt.subplot(gs[:,1:]) 
    ax1.imshow(map_f, cmap='gray')

    ax2 = plt.subplot(gs[0, 0]) 
    ax2.imshow(img1, cmap='gray')

    ax3 = plt.subplot(gs[1:, 0]) 
    ax3.imshow(img2, cmap='gray')
    fig.tight_layout()
    

    # x = []
    # y = []
    # for i in range(100):
    #     x.append([1.414*i,0])
    #     y.append([i,i])
    # pts1 = np.array(x)
    # pts2 = np.array(y)
    # _, pts, [x, y, yaw] = SM.icp(   reference_points=pts1, 
    #                                 points=pts2, 
    #                                 max_iterations=5000, 
    #                                 distance_threshold=10, 
    #                                 convergence_translation_threshold=1e-2, 
    #                                 convergence_rotation_threshold=1e-2, 
    #                                 point_pairs_threshold=5, 
    #                                 verbose=False)
    # c = math.cos(yaw)
    # s = math.sin(yaw)
    # rot = np.array([[c, -s],
    #                 [s, c]])
    # pts3 = np.dot(pts2, rot.T)
    # pts3[:, 0] += x
    # pts3[:, 1] += y
    # print(x,y,yaw/3.14*180)
    # plt.scatter(pts1[:,0],pts1[:,1],s=5, c='blue')    
    # plt.scatter(pts2[:,0],pts2[:,1],s=5, c='black')  
    # plt.scatter(pts3[:,0],pts3[:,1],s=2, c='red') 

    plt.show()

