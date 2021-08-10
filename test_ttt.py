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


if __name__ == "__main__":
    ratio = 3
    s = np.array([70, 60])
    yaw = 0.523598776

    num = 90

    pts = []
    for i in range(num):
        pts.append([
            random.randint(1, 100),
            random.randint(1, 100)
        ]) 
    
    pts1 = np.array(pts)
    pts2 = RANSAC_Map_Merging()._rotation(yaw).dot(pts1.T).T*ratio + s
    
    pts1_e, pts2_e = [],[]
    for i in range(100-num):
        pts1_e.append([i,i])
        pts2_e.append([random.randint(1, 10),random.randint(1, 10)])
    pts1 = np.vstack((pts1, np.array(pts1_e)))
    pts2 = np.vstack((pts2, np.array(pts2_e)))

    relation, value, inlier_point_match = RANSAC_Map_Merging()._ransac_find_all(pts_set_1=pts2, pts_set_2=pts1, sigma=10, max_iter=1000)
    print(relation)

    dx = relation[0]
    dy = relation[1]
    dyaw = relation[2]
    dr = relation[3]

    pts3 = (RANSAC_Map_Merging()._rotation(dyaw).dot(pts2.T).T + np.array([dx[0], dy[0]]))/dr

    fig, ax = plt.subplots()    
    ax.scatter(pts1[:,0], pts1[:,1], s=10, c='blue')
    ax.scatter(pts2[:,0], pts2[:,1], s=5, c='green')
    ax.scatter(pts3[:,0], pts3[:,1], s=2, c='red')
    fig.tight_layout()

    plt.show()