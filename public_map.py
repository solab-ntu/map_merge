# -*- coding:utf-8 -*-
from operator import le
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import random
import copy
import matplotlib.gridspec as gridspec
from numpy.lib.polynomial import polyint
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
from ransac_map_merge import MAP_Function, RANSAC_Map_Merging
from itertools import combinations, count, permutations
from icp_map_merge import ScanMatching
from sklearn.cluster import KMeans
from bresenham import bresenham

def _enlarge_image(img, L=1):
    img_r, img_c = np.shape(img)
    # new_img = np.ones((img_r+2*L, img_c+2*L))*128
    new_img = np.zeros((img_r+2*L, img_c+2*L))
    new_img[L:L+img_r, L:L+img_c] = img
    return new_img.astype(np.uint8)

def _simple_rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If the center of the rotation is not defined, use the center of the image.
    if center is None:
        (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(128))


def _simple_transform_state(input_map):
    r, c = np.shape(input_map)
    for i in range(r):
        for j in range(c):
            if input_map[i, j]  < 100:
                input_map[i, j] = 0
            elif input_map[i, j]  > 240:
                input_map[i, j] = 255
            else:
                input_map[i, j] = 128
    return input_map


def _do_bf_match(img1_data, img2_data):
    img1, kp1, des1 = img1_data
    img2, kp2, des2 = img2_data
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.5*n.distance:
        # if 1:
            good.append([m])

    img3 = np.empty((600,600))
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, img3, flags=2) #採用cv2.drawMatchesKnn()函數，在最佳匹配的點之間繪製直線
    return img3

def _sift_feature(img):
    siftDetector = cv2.xfeatures2d.SIFT_create()
    key_points, descriptor = siftDetector.detectAndCompute(img, None)
    kp_list = []
    des_list = []
    # print(len(key_points), np.shape(descriptor))
    for i in range(len(key_points)):
        kp = key_points[i]
        y = round(kp.pt[0])
        x = round(kp.pt[1])
        if img[x,y] == 255:
        # if 1:
            kp_list.append(kp)
            des_list.append(descriptor[i,:])
    des_list = np.array(des_list)
    img = cv2.drawKeypoints(img,
                            outImage=img,
                            keypoints=kp_list, 
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color= (255, 0, 0))
    return img, kp_list, des_list

def _patch_defect(img):
    r, c = np.shape(img)
    pts = np.where(img==0) # black pixel
    _, length = np.shape(pts)
    for i in range(length):
        x = pts[0][i]
        y = pts[1][i]
        if x-1 < 0 or y-1 < 0 or x+1 >= r or y+1 >= c:
            pass
        else:
            if img[x-1,y]==255 and img[x+1,y]==255:
                img[x,y] = 255
            elif img[x,y-1]==255 and img[x,y+1]==255:
                img[x,y] = 255
    return img.astype(np.uint8)

def _map_merge_1(map1, map2, match_pairs):
    RMM = RANSAC_Map_Merging()
    MF = MAP_Function()
    
    pts_1, pts_2 = [], []
    for data1, data2 in match_pairs:
        for i in range(len(data1)):
            pts_1.append([
                data1[i][1][0],
                data1[i][1][1]
                ])
        for i in range(len(data2)):
            pts_2.append([
                data2[i][1][0],
                data2[i][1][1]
                ])
    pts1 = np.array(pts_1)
    pts2 = np.array(pts_2)

    relation, value = RMM._ransac_find_rotation_translation(pts_set_1=pts2, pts_set_2=pts1, sigma=0.5, max_iter=5000)
    print("- Inlier Percent: %f"%value)
    # Because the coordinates between the maps and the SIFT features are different:
    # SIFT Features:    Right: +x, Down:  +y
    # Maps:             Down:  +x, Right: +y
    # Hence the dx and dy should be changed.
    dx = relation[1]
    dy = relation[0]
    dyaw = relation[2]
    print("- (x, y, t): (%f, %f, %f)"%(dx,dy,dyaw))

    # index, agr, dis = RMM._similarity_index(x=[dy, dx, dyaw], map1=map_ref, map2=map_align)
    # print("Similarity Index: %f\nAgree Number: %f\nDisargee Number: %f"%(index, agr, dis))
    index, agr, dis = RMM._similarity_index_2(x=[dx, dy, dyaw], map1=map1, map2=map2)
    print("- Similarity Index: %f\n- Agree Number: %f\n- Disargee Number: %f"%(index, agr, dis))
    
    map_merged = MF._merging_map(dx=dx, dy=dy, dtheta=dyaw, map1=map1, map2=map2)
    map_ref = map_merged.astype(np.uint8)
    map_ref = MF._modify_map_size(merged_map=map_ref)

    return map_ref


def _map_merge(map_list, feature_option='sift'):
    RMM = RANSAC_Map_Merging()
    MF = MAP_Function()
    map_ref = map_list[0]
    for i in range(len(map_list)-1):
        map_align = map_list[i+1]

        
        if feature_option == "orb":
            orb = cv2.ORB_create()
            key_points_1, descriptor_1 = orb.detectAndCompute(map_ref, None)
            key_points_2, descriptor_2 = orb.detectAndCompute(map_align, None)
        
        elif feature_option == "surf":
            surf = cv2.xfeatures2d.SURF_create(400)
            key_points_1, descriptor_1 = surf.detectAndCompute(map_ref, None)
            key_points_2, descriptor_2 = surf.detectAndCompute(map_align, None)
        else:
            siftDetector = cv2.xfeatures2d.SIFT_create()
            key_points_1, descriptor_1 = siftDetector.detectAndCompute(map_ref, None)
            key_points_2, descriptor_2 = siftDetector.detectAndCompute(map_align, None)

        # bf = cv2.BFMatcher()
        # matches = bf.knnMatch(descriptor_1, descriptor_2, k=2)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(descriptor_1, descriptor_2, k=2)

        

        good = []
        good_pair = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
                good_pair.append([m])
        
        pts_1, pts_2 = [], []
        for i in good:
            query_idx = i.queryIdx
            train_idx = i.trainIdx

            pts_1.append([
                key_points_1[query_idx].pt[0],
                key_points_1[query_idx].pt[1],
                ])
            pts_2.append([
                key_points_2[train_idx].pt[0],
                key_points_2[train_idx].pt[1],
                ])
        
        img_match = np.empty((600,600))
        img_match = cv2.drawMatchesKnn(map_ref, key_points_1, map_align, key_points_2, good_pair, img_match, flags=2)

        pts1 = np.array(pts_1)
        pts2 = np.array(pts_2)

        relation, value = RMM._ransac_find_rotation_translation(pts_set_1=pts2, pts_set_2=pts1, sigma=0.5, max_iter=5000)
        print("- Inlier Percent: %f"%value)
        # Because the coordinates between the maps and the SIFT features are different:
        # SIFT Features:    Right: +x, Down:  +y
        # Maps:             Down:  +x, Right: +y
        # Hence the dx and dy should be changed.
        dx = relation[1]
        dy = relation[0]
        dyaw = relation[2]
        print("- (x, y, t): (%f, %f, %f)"%(dx,dy,dyaw))

        # index, agr, dis = RMM._similarity_index(x=[dy, dx, dyaw], map1=map_ref, map2=map_align)
        # print("Similarity Index: %f\nAgree Number: %f\nDisargee Number: %f"%(index, agr, dis))
        index, agr, dis = RMM._similarity_index_2(x=[dx, dy, dyaw], map1=map_ref, map2=map_align)
        print("- Similarity Index: %f\n- Agree Number: %f\n- Disargee Number: %f"%(index, agr, dis))
        
        map_merged = MF._merging_map(dx=dx, dy=dy, dtheta=dyaw, map1=map_ref, map2=map_align)
        map_ref = map_merged.astype(np.uint8)
        map_ref = MF._modify_map_size(merged_map=map_ref)

    return map_ref, img_match

def _submap_merge(maps_to_be_merged, sub_img, feature_option='sift'):
    RMM = RANSAC_Map_Merging()
    MF = MAP_Function()
    map_ref = maps_to_be_merged[0]
    map_align = maps_to_be_merged[1]
    img1 = sub_img[0]
    img2 = sub_img[1]

    if feature_option == "orb":
        orb = cv2.ORB_create()
        key_points_1, descriptor_1 = orb.detectAndCompute(img1, None)
        key_points_2, descriptor_2 = orb.detectAndCompute(img2, None)
    
    elif feature_option == "surf":
        surf = cv2.xfeatures2d.SURF_create(400)
        key_points_1, descriptor_1 = surf.detectAndCompute(img1, None)
        key_points_2, descriptor_2 = surf.detectAndCompute(img2, None)
    else:
        siftDetector = cv2.xfeatures2d.SIFT_create()
        key_points_1, descriptor_1 = siftDetector.detectAndCompute(img1, None)
        key_points_2, descriptor_2 = siftDetector.detectAndCompute(img2, None)

    # FLANN parameters
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)   # or pass empty dictionary
    # flann = cv2.FlannBasedMatcher(index_params,search_params)
    # matches = flann.knnMatch(descriptor_1, descriptor_2, k=2)

    # good = []
    # good_pair = []
    # for m, n in matches:
    #     # if m.distance < 0.75*n.distance:
    #     if 1:
    #         good.append(m)
    #         good_pair.append([m])
    
    # pts_1, pts_2 = [], []
    # for i in good:
    #     query_idx = i.queryIdx
    #     train_idx = i.trainIdx

    #     pts_1.append([
    #         key_points_1[query_idx].pt[0],
    #         key_points_1[query_idx].pt[1],
    #         ])
    #     pts_2.append([
    #         key_points_2[train_idx].pt[0],
    #         key_points_2[train_idx].pt[1],
    #         ])

    pts_1, pts_2 = [], []
    for kp1 in key_points_1:
        for kp2 in key_points_2: 
            pts_1.append([
                kp1.pt[0],
                kp1.pt[1]
            ])
            pts_2.append([
                kp2.pt[0],
                kp2.pt[1]
            ])
            
    if len(pts_1) <2:
        return None, 0

        
    pts1 = np.array(pts_1)
    pts2 = np.array(pts_2)
    

    relation, value, _ = RMM._ransac_find_rotation_translation(pts_set_1=pts2, pts_set_2=pts1, sigma=0.5, max_iter=5000)
    # print("- Inlier Percent: %f"%value)

    # Because the coordinates between the maps and the SIFT features are different:
    # SIFT Features:    Right: +x, Down:  +y
    # Maps:             Down:  +x, Right: +y
    # Hence the dx and dy should be changed.
    dx = relation[1]
    dy = relation[0]
    dyaw = relation[2]
    # print("- (x, y, t): (%f, %f, %f)"%(dx,dy,dyaw))

    index, agr, dis = RMM._similarity_index_2(x=[dx, dy, dyaw], map1=map_ref, map2=map_align)
    # print("- Similarity Index: %f\n- Agree Number: %f\n- Disargee Number: %f"%(index, agr, dis))
    
    map_merged = MF._merging_map(dx=dx, dy=dy, dtheta=dyaw, map1=map_ref, map2=map_align)
    map_merged = map_merged.astype(np.uint8)
    map_merged = MF._modify_map_size(merged_map=map_merged)

    return map_merged, index

def _sub_kp_merge(maps_to_be_merged, sub_kp_list):
    RMM = RANSAC_Map_Merging()
    MF = MAP_Function()
    map_ref = maps_to_be_merged[0]
    map_align = maps_to_be_merged[1]
    
    pts_1, pts_2 = [], []
    for kp1 in sub_kp_list[0]:
        for kp2 in sub_kp_list[1]: 
            pts_1.append([
                kp1[0],
                kp1[1]
            ])
            pts_2.append([
                kp2[0],
                kp2[1]
            ])
            
    if len(pts_1) <2:
        return None, 0

    pts1 = np.array(pts_1)
    pts2 = np.array(pts_2)
    

    relation, value, inlier_point_match = RMM._ransac_find_rotation_translation(pts_set_1=pts2, pts_set_2=pts1, sigma=5, max_iter=1000)
    # print("- Inlier Percent: %f"%value)

    # Because the coordinates between the maps and the SIFT features are different:
    # SIFT Features:    Right: +x, Down:  +y
    # Maps:             Down:  +x, Right: +y
    # Hence the dx and dy should be changed.
    dx = relation[1]
    dy = relation[0]
    dyaw = relation[2]
    # print("- (x, y, t): (%f, %f, %f)"%(dx,dy,dyaw))

    """似乎有錯誤"""
    pts1 = []
    pts2 = []
    for pt in inlier_point_match:
        pts1 += [pt[0]]
        pts2 += [pt[1]]
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    # print(pts1.shape, pts2.shape)
    ddyaw, _, dt = _fit_model(pts2, pts1)
    ddx = dt[1]
    ddy = dt[0]
    # print(dyaw-ddyaw, dx - ddx, dy-ddy)
    # Index 用改過的relation，合併用沒改過的relation，感覺很怪？這邊得除錯一下。
    index, agr, dis = RMM._similarity_index_2(x=[dx, dy, dyaw], map1=map_ref, map2=map_align)
    """"""

    # index, agr, dis = RMM._similarity_index_2(x=[dx, dy, dyaw], map1=map_ref, map2=map_align)
    # print("- Similarity Index: %f\n- Agree Number: %f\n- Disargee Number: %f"%(index, agr, dis))
    
    map_merged = MF._merging_map(dx=dx, dy=dy, dtheta=dyaw, map1=map_ref, map2=map_align)
    map_merged = map_merged.astype(np.uint8)
    map_merged = MF._modify_map_size(merged_map=map_merged)

    return map_merged, index, (dx, dy, dyaw)



def _rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If the center of the rotation is not defined, use the center of the image.
    if center is None:
        (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)
 
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    img_rot = cv2.warpAffine(image, M, (nW, nH))
    r, c = np.shape(img_rot)
    for i in range(r):
        for j in range(c):
            if img_rot[i, j]  > 0:
                img_rot[i, j] = 255
            else:
                img_rot[i, j] = 0
    return img_rot



def _transform_state(input_map):
    r, c = np.shape(input_map)
    for i in range(r):
        for j in range(c):
            if input_map[i, j]  < 150:
                input_map[i, j] = 255
            else:
                input_map[i, j] = 0
    return input_map

def feature_detect_match(img1, img2, method="sift"):
    if method == "sift":
        siftDetector = cv2.xfeatures2d.SIFT_create()
        key_points_1, descriptor_1 = siftDetector.detectAndCompute(img1, None)
        key_points_2, descriptor_2 = siftDetector.detectAndCompute(img2, None)
    elif method == "surf":
        surf = cv2.xfeatures2d.SURF_create(400)
        key_points_1, descriptor_1 = surf.detectAndCompute(img1, None)
        key_points_2, descriptor_2 = surf.detectAndCompute(img2, None)
    elif method == "orb":        
        orb = cv2.ORB_create()
        key_points_1, descriptor_1 = orb.detectAndCompute(img1, None)
        key_points_2, descriptor_2 = orb.detectAndCompute(img2, None)
    elif method == "fast":        
        fast = cv2.FastFeatureDetector_create()
        key_points_1 = fast.detect(img1, None)
        key_points_2 = fast.detect(img2, None)
        br = cv2.BRISK_create()
        key_points_1, descriptor_1 = br.compute(img1,  key_points_1)
        key_points_2, descriptor_2 = br.compute(img2,  key_points_2)
    else:
        print("Method selection error, please input 'sift', 'surf', or 'orb'.")
        return
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor_1, descriptor_2, k=2)
    good = []
    for m, n in matches:
        # if 1:
        if m.distance < 0.5*n.distance:
            good.append([m])

    img3 = np.empty((600,600))
    img3 = cv2.drawMatchesKnn(img1, key_points_1, img2, key_points_2, good, img3, flags=2) #採用cv2.drawMatchesKnn()函數，在最佳匹配的點之間繪製直線
    return img3

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

def _blur_image(img, L=1):
    img_r, img_c = np.shape(img)
    img_tmp = np.zeros((img_r+2*L,img_c+2*L))
    img_tmp[L:L+img_r, L:L+img_c] = img
    new_img = np.zeros((img_r, img_c))
    pts = np.where(img==255)
    _, length = np.shape(pts)
    
    for i in range(length):
        x = pts[0][i]+L
        y = pts[1][i]+L
        window = img_tmp[x-L:x+L+1, y-L:y+L+1]   
        _, num = np.shape(np.where(window==255))
        new_img[x, y] = round(num/((2*L+1)**2) * 255)
    return new_img.astype(np.uint8)

def _erode_image(img, L=1):
    img_r, img_c = np.shape(img)
    img_tmp = np.zeros((img_r+2*L,img_c+2*L))
    img_tmp[L:L+img_r, L:L+img_c] = img
    new_img = np.zeros((img_r+2*L,img_c+2*L))
    pts = np.where(img==255)
    _, length = np.shape(pts)
    B1 = np.ones((2*L+1,2*L+1))*255
    for i in range(length):
        x = pts[0][i]+L
        y = pts[1][i]+L
        if 1:
            window = img_tmp[x-L:x+L+1, y-L:y+L+1]   
            flag = window==B1
            if flag.all():
                new_img[x, y] = 255
    return new_img.astype(np.uint8)

def _radon_find_theta(image):
    image = rescale(image, scale=1.0, mode='reflect', multichannel=False)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    x = np.where(sinogram==np.amax(sinogram))
    yaw = x[1][0]/sinogram.shape[1]*180
    return yaw, sinogram


def _plot_map(map_list, map_merged):        
    L = len(map_list)
    plt.figure()
    gs=gridspec.GridSpec(L, 2)
    for i in range(L):
        x = plt.subplot(gs[i, 0])
        plt.imshow(map_list[i], cmap='gray')
        x.set_title('Orginal Map: No.%i'%(i+1))
    x = plt.subplot(gs[:,1])
    plt.imshow(map_merged, cmap='gray')
    x.set_title('Merging Result')
    plt.tight_layout() 
    plt.show()
    return

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
        flag = (window==B1).all()
        if flag:
            new_img[x-L, y-L] = 255
            img_tmp[x-L:x+L+1, y-L:y+L+1] = 0
            # for a in range(3):
            #     for b in range(3):
            #         if a == 1 and b == 1:
            #             continue
            #         cx = x-1+a
            #         cy = y-1+b
            #         window_check = img_tmp[cx-L:cx+L+1, cy-L:cy+L+1]
            #         contradict_flag = (window_check==B1).all()
            #         if contradict_flag:
            #             break
            #     if contradict_flag:
            #         break
            # if not contradict_flag:
            #     new_img[x-L, y-L] = 255
            #     img_tmp[x-L:x+L+1, y-L:y+L+1] = 0
    return new_img.astype(np.uint8)


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

    img_list = []
    now_img = np.zeros((img_r+2*L, img_c+2*L))

    # 先用有限次數的迭代
    while times < 100000 :
        times += 1
        img[x, y] = 0    
        new_img[x-1:x+2, y-1:y+2] = 255
        now_img[x-1:x+2, y-1:y+2] = 255

        window = img[x-L:x+L+1, y-L:y+L+1]
        window_pts = np.where(window==255)
        
        # Window內有白點。
        if window_pts[0].shape[0]!=0:
            # 曼哈頓距離最近者
            index = np.argmin( abs(window_pts[0] - L) + abs(window_pts[1] - L) )
            dx = window_pts[0][index] - L
            dy = window_pts[1][index] - L

            abs_dx = abs(dx)
            abs_dy = abs(dy)

            # 上下移動
            if abs_dx > abs_dy: 
                img[x+dx, y+dy] = 0
                if dx > 0:
                    new_img[x:x+dx, y-1:y+2] = 255
                    now_img[x:x+dx, y-1:y+2] = 255
                else:
                    new_img[x+dx:x, y-1:y+2] = 255
                    now_img[x+dx:x, y-1:y+2] = 255
                x = x + dx
                y = y
                direction = 'vertical'

            # 左右移動
            elif abs_dy > abs_dx:
                img[x+dx, y+dy] = 0
                if dy > 0:
                    new_img[x-1:x+2, y:y+dy] = 255
                    now_img[x-1:x+2, y:y+dy] = 255
                else: 
                    new_img[x-1:x+2, y+dy:y] = 255
                    now_img[x-1:x+2, y+dy:y] = 255
                x = x
                y = y + dy
                direction = 'horizontal'

            # 45度移動抉擇，跟隨先前移動方向。
            else:
                if direction == 'vertical':
                    if dx > 0:
                        new_img[x:x+dx, y-1:y+2] = 255
                        now_img[x:x+dx, y-1:y+2] = 255
                    else:
                        new_img[x+dx:x, y-1:y+2] = 255
                        now_img[x+dx:x, y-1:y+2] = 255
                    x = x + dx
                    y = y

                elif direction == 'horizontal':
                    if dy > 0:
                        new_img[x-1:x+2, y:y+dy] = 255
                        now_img[x-1:x+2, y:y+dy] = 255
                    else: 
                        new_img[x-1:x+2, y+dy:y] = 255
                        now_img[x-1:x+2, y+dy:y] = 255
                    x = x
                    y = y + dy

                # 沒有先前移動方向，直接選取下一個點。
                else:
                    pts = np.where(img==255)
                    x = pts[0][0]
                    y = pts[1][0]
        
        # Window內沒白點。
        else:
            # change to next segment.
            pts = np.where(img==255)
            _, length = np.shape(pts)
            if length == 0:
                print("break")
                break
            
            img_list.append(now_img.astype(np.uint8))
            now_img = np.zeros((img_r+2*L, img_c+2*L))
            
            x = pts[0][0]
            y = pts[1][0]
            
    
    return new_img.astype(np.uint8), img_list


def _image_boundary_follow_test(image, image_sp, L=10):
    # 複製興趣點到新樣板上。
    img_r, img_c = np.shape(image_sp)
    img = np.zeros((img_r+2*L, img_c+2*L))
    img[L:img_r+L, L:img_c+L] = image_sp

    # 複製確認用膨脹樣板至新樣板上。
    img_check = np.zeros((img_r+2*L, img_c+2*L))
    img_check[L:img_r+L, L:img_c+L] = image

    # 新建輸出樣板。
    new_img = np.zeros((img_r+2*L, img_c+2*L))    

    # 列出所有興趣點，並從最左上的點開始。
    pts = np.where(img==255)
    _, length = np.shape(pts)
    idx = np.argmin( abs(pts[0]) + abs(pts[1]) )
    x = pts[0][idx]
    y = pts[1][idx]

    # 紀錄最初始點。
    icx = copy.copy(x)
    icy = copy.copy(y)

    # 這個之後可以刪除
    img_list = []
    now_img = np.zeros((img_r+2*L, img_c+2*L))

    # 初始條件
    best_v = None
    best_pts = None

    # 先用有限次數的迭代
    times = 0
    while times < 100000 :
        times += 1
        # 目前興趣點歸零，並紀錄於輸出樣板上。
        img[x, y] = 0    
        new_img[x, y] = 255
        now_img[x, y] = 255
        

        # 如果沒有初始修正斜率，以下列方式計算。
        if best_v == None:
            # # 建立修正用視窗，與搜尋視窗內下一個興趣點。
            # window = img[x-L:x+L+1, y-L:y+L+1] 
            # window_check = img_check[x-L:x+L+1, y-L:y+L+1]
            # window_pts = np.where(window==255)
            
            # # 視窗內若具有其他興趣點。
            # if window_pts[0].shape[0]!=0:
            #     # 選擇曼哈頓距離最近者作為修正點。
            #     index = np.argmin( abs(window_pts[0] - L) + abs(window_pts[1] - L) )
            #     pt_x = window_pts[0][index]
            #     pt_y = window_pts[1][index]
            #     dx = pt_x-x
            #     dy = pt_y-y

            #     if dx < 0:
            #         if dy < 0:
            #             w_idx = 0
            #         else:
            #             w_idx = 2
            #     else:
            #         if dy < 0:
            #             w_idx = 1
            #         else:
            #             w_idx = 3

            # w1 = img[x-L:x+1, y-L:y+1]
            # w2 = img[x:x+L+1, y-L:y+1]
            # w3 = img[x-L:x+1, y:y+L+1]
            # w4 = img[x:x+L+1, y:y+L+1]

            # all_window = [w1, w2, w3, w4]
            # window_used = all_window[w_idx]

            # 選擇要開始的四個象限視窗，以最多佔據格點的視窗作為第一順序。
            w1_check = img_check[x-L:x+1, y-L:y+1]
            w2_check = img_check[x:x+L+1, y-L:y+1]
            w3_check = img_check[x-L:x+1, y:y+L+1]
            w4_check = img_check[x:x+L+1, y:y+L+1]
            all_window_check = [w1_check, w2_check, w3_check, w4_check]
            w_idx = np.argmax([np.sum(w1_check), np.sum(w2_check), np.sum(w3_check), np.sum(w4_check)])
            window_used_check = all_window_check[w_idx]

            # fig, ax = plt.subplots(2, 2)
            # ax[0,0].imshow(w1_check, cmap='gray')
            # ax[1,0].imshow(w2_check, cmap='gray')
            # ax[0,1].imshow(w3_check, cmap='gray')
            # ax[1,1].imshow(w4_check, cmap='gray')
            # plt.show()

            # 列出初始直線，以向量方式紀錄：best_v。
            best_score = 0 
            if w_idx == 0:
                for i in range(L+1):
                    score = 0
                    pts = list(bresenham(L,L,i,0))
                    for pt in pts:
                        score += window_used_check[pt]
                    if score > best_score:
                        best_score = score
                        best_pt = (i,0)
                        # best_pts = pts
                for j in range(L):
                    score = 0
                    pts = list(bresenham(L,L,0,i))
                    for pt in pts:
                        score += window_used_check[pt]
                    if score > best_score:
                        best_score = score
                        best_pt = (0,i)
                        # best_pts = pts
                best_v = (best_pt[0]-L, best_pt[1]-L)


            elif w_idx ==1:
                for i in range(L+1):
                    score = 0
                    pts = list(bresenham(0,L,i,0))
                    for pt in pts:
                        score += window_used_check[pt]
                    if score > best_score:
                        best_score = score
                        best_pt = (i,0)
                        # best_pts = pts
                for j in range(L):
                    score = 0
                    pts = list(bresenham(0,L,L,i))
                    for pt in pts:
                        score += window_used_check[pt]
                    if score > best_score:
                        best_score = score
                        best_pt = (L,i)
                        # best_pts = pts
                best_v = (best_pt[0]-0, best_pt[1]-L)


            elif w_idx ==2:
                for i in range(L+1):
                    score = 0
                    pts = list(bresenham(L,0,i,L))
                    for pt in pts:
                        score += window_used_check[pt]
                    if score > best_score:
                        best_score = score
                        best_pt = (i,L)
                        # best_pts = pts
                for j in range(L):
                    score = 0
                    pts = list(bresenham(L,0,0,i))
                    for pt in pts:
                        score += window_used_check[pt]
                    if score > best_score:
                        best_score = score
                        best_pt = (0,i)
                        # best_pts = pts
                best_v = (best_pt[0]-L, best_pt[1]-0)

            elif w_idx ==3:
                for i in range(L+1):
                    score = 0
                    pts = list(bresenham(0,0,i,L))
                    for pt in pts:
                        score += window_used_check[pt]
                    if score > best_score:
                        best_score = score
                        best_pt = (i,L)
                        # best_pts = pts
                for j in range(L):
                    score = 0
                    pts = list(bresenham(0,0,L,i))
                    for pt in pts:
                        score += window_used_check[pt]
                    if score > best_score:
                        best_score = score
                        best_pt = (L,i)
                        # best_pts = pts
                best_v = (best_pt[0]-0, best_pt[1]-0)

        # 建立修正用視窗，與搜尋視窗內下一個興趣點。
        window = img[x-L:x+L+1, y-L:y+L+1] 
        window_check = img_check[x-L:x+L+1, y-L:y+L+1]
        window_pts = np.where(window==255)
        
        # 視窗內若具有其他興趣點。
        if window_pts[0].shape[0]!=0:
            # 選擇曼哈頓距離最近者作為修正點。
            index = np.argmin( abs(window_pts[0] - L) + abs(window_pts[1] - L) )
            pt_x = window_pts[0][index]
            pt_y = window_pts[1][index]

            # 確認與最近點是否進行修正（連線佔有的區域幾乎為佔據區域）
            line_pts = list(bresenham(L,L,pt_x,pt_y))
            line_pts.pop(0)
            count = 0
            for line_pt in line_pts:
                if not window_check[line_pt] == 255:
                    count += 1

            if count/len(line_pts) > 0.5:
                correction_flag = False
                # print(count/len(line_pts))
            else:
                correction_flag = True



            # 若進行修正：
            if correction_flag:
                # 確認修正點
                best_pts = list(bresenham(L,L,best_v[0]+L, best_v[1]+L))
                best_pts.pop(0)
                best_pts = np.array(best_pts)
                # print("\n",abs(best_pts[:,0]-pt_x)+abs(best_pts[:,1]-pt_y))
                correction_idx = np.argmin(abs(best_pts[:,0]-pt_x)+abs(best_pts[:,1]-pt_y))
                correction_pt_x = best_pts[correction_idx][0]
                correction_pt_y = best_pts[correction_idx][1]
                correction_pts = list(bresenham(L,L,correction_pt_x, correction_pt_y))
            
                # （目前問題出在這）
                count = 0
                for check_pt in correction_pts:
                    if window_check[check_pt] == 255:
                        # check_flag = False
                        count += 1  
                    else:
                        pass
                        check_flag = True
                if count/len(correction_pts) > 0.1:
                    check_flag = True
                else:
                    check_flag = True

                if check_flag:
                    dx = pt_x - L
                    dy = pt_y - L
                    img[x+dx, y+dy] = 0
                    # best_v = None
                    for check_pt in correction_pts:
                        new_img[x+check_pt[0]-L, y+check_pt[1]-L] = 255
                    x = x+check_pt[0]-L
                    y = y+check_pt[1]-L
                    
            else:
                pts = np.where(img==255)
                x = pts[0][0]
                y = pts[1][0]  
                best_v = None
        # Window內沒白點。
        else:
            
            # change to next segment.
            best_v = None
            pts = np.where(img==255)
            _, length = np.shape(pts)
            if length == 0:
                print("break: (%i,%i)"%(x,y))
                break
            
            print("change to next point: (%i,%i)"%(x,y))
            img_list.append(now_img.astype(np.uint8))
            now_img = np.zeros((img_r+2*L, img_c+2*L))
            

            x = pts[0][0]
            y = pts[1][0]
            
            if abs(x-icx) <= L and abs(y-icy) <= L:
                x = icx
                y = icy                
            
            # new_img[x-2:x+3,y-2:y+3]=255
            
    
    return new_img.astype(np.uint8), img#img_list


def _image_boundary_follow_test_1(image, image_sp, L=10):
    img_r, img_c = np.shape(image_sp)
    img = np.zeros((img_r+2*L, img_c+2*L))
    img_check = np.zeros((img_r+2*L, img_c+2*L))
    img[L:img_r+L, L:img_c+L] = image_sp
    img_check[L:img_r+L, L:img_c+L] = image
    new_img = np.zeros((img_r+2*L, img_c+2*L))    
    
    pts = np.where(img==255)
    _, length = np.shape(pts)

    direction = None
    times = 0
    x = pts[0][0]
    y = pts[1][0]

    img_list = []
    now_img = np.zeros((img_r+2*L, img_c+2*L))

    # 先用有限次數的迭代
    while times < 100000 :
        times += 1
        img[x, y] = 0    
        new_img[x, y] = 255
        now_img[x, y] = 255
        
        #建立檢查範圍
        window = img[x-L:x+L+1, y-L:y+L+1] 
        window_check = img_check[x-L:x+L+1, y-L:y+L+1]
        window_pts = np.where(window==255)
        
        # Window內有白點。
        if window_pts[0].shape[0]!=0:
            # 曼哈頓距離最近者
            index = np.argmin( abs(window_pts[0] - L) + abs(window_pts[1] - L) )
            dx = window_pts[0][index] - L
            dy = window_pts[1][index] - L

            abs_dx = abs(dx)
            abs_dy = abs(dy)

            # 中心點到最近點
            line_pts = list(bresenham(L,L,window_pts[0][index],window_pts[1][index]))
            line_pts.pop(0)
            for line_pt in line_pts:
                if not window_check[line_pt] == 255:
                    # 興趣點連線矛盾，不予以修正。
                    correction_flag = False
                    break
                else:
                    # 可嘗試修正
                    correction_flag = True
            if correction_flag:
                for line_pt in line_pts:
                    cx = line_pt[0] - L + x
                    cy = line_pt[1] - L + y
                    new_img[cx, cy] = 255
                    now_img[cx, cy] = 255
                x = cx
                y = cy
            else:
                pts = np.where(img==255)
                x = pts[0][0]
                y = pts[1][0]       
        # Window內沒白點。
        else:
            # change to next segment.
            pts = np.where(img==255)
            _, length = np.shape(pts)
            if length == 0:
                print("break")
                break
            
            img_list.append(now_img.astype(np.uint8))
            now_img = np.zeros((img_r+2*L, img_c+2*L))
            
            x = pts[0][0]
            y = pts[1][0]
            
    
    return new_img.astype(np.uint8), img_list

def _image_boundart_follow_test_2(image, image_sp, L=10):
    r, c =np.shape(image_sp)
    img_sp_tmp = np.zeros((r+2*L, c+2*L))
    img_sp_tmp[L:r+L, L:c+L] = image_sp

    r, c =np.shape(image)
    img_tmp = np.zeros((r+2*L, c+2*L))
    img_tmp[L:r+L, L:c+L] = image


    # 列出所有興趣點，並從最左上的點開始。
    pts = np.where(img_sp_tmp==255)
    idx = np.argmin( abs(pts[0]) + abs(pts[1]) )
    x = pts[0][idx]
    y = pts[1][idx]

    # 紀錄最初始點。
    icx = copy.copy(x)
    icy = copy.copy(y)
    vector_set = []
    vector_set_all = []
    pts_set = []
    pts_set_all = []

    times = 0
    while times < 10000:
        # 如果沒有興趣點，結束while迴圈。
        break_flag = np.where(img_sp_tmp==255)
        if break_flag[0].shape[0] == 0:
            break
    
        # 目前起點
        img_sp_tmp[x,y] = 0
        window = img_sp_tmp[x-L:x+L+1, y-L:y+L+1]
        # pts = np.where(img_sp_tmp==255)
        window_pts = np.where(window==255)

        # 目前視窗內有興趣點
        if not window_pts[0].shape[0] == 0:

            # 目前終點
            next_idx = np.argmin( abs(window_pts[0]-L) + abs(window_pts[1]-L) )
            next_x = window_pts[0][next_idx]-L+x
            next_y = window_pts[1][next_idx]-L+y

            # 向量差異，s:起點 e:終點
            now_vector = [next_x - x, next_y - y]
            now_pts_s = [x, y]
            now_pts_e = [next_x, next_y]

            # 確認連線大部分皆為佔據
            line_pts = list(bresenham(now_pts_s[0], now_pts_s[1], now_pts_e[0], now_pts_e[1]))
            count = 0
            for pt in line_pts:
                if img_tmp[pt] == 255:
                    count += 1
            if count/len(line_pts) > 0.75:
                # vector_set 已有東西
                if vector_set:
                    vector_test = np.array(vector_set)
                    vector_test = np.mean(vector_test, axis=0)
                    
                    # 改用角度判斷， +-5 deg 作為評判標準。
                    diff_angle = (math.atan2(vector_test[1], vector_test[0]) - math.atan2(now_vector[1], now_vector[0]))/math.pi*180
                    # print(diff_angle)
                    # 足夠相似，加入vector_set
                    if abs(diff_angle) < 20:
                        vector_set.append(now_vector)
                        pts_set.append(now_pts_s)
                        pts_set.append(now_pts_e)
                    # 不夠相似，換到下一個vector_set
                    else:
                        vector_set_all.append(vector_set)
                        vector_set = [now_vector]
                        pts_set_all.append(pts_set)
                        pts_set = [now_pts_s, now_pts_e]

                # vector_set 內無東西，直接存
                else:
                    vector_set.append(now_vector)
                    pts_set.append(now_pts_s)
                    pts_set.append(now_pts_e)

            x = next_x
            y = next_y

        else:
            pts = np.where(img_sp_tmp==255)
            if pts[0].shape[0] == 0:
                break
            
            # idx = np.argmin( abs(pts[0]-x) + abs(pts[1]-y) )
            
            # 換至下一個起點
            idx = np.argmin( abs(pts[0]) + abs(pts[1]) )
            x = pts[0][idx]
            y = pts[1][idx]

            # if abs(x-icx) <= L and abs(y-icy) <= L:
            #     x = icx
            #     y = icy     
            


    new_img = np.zeros((r+2*L, c+2*L))
    for pi in pts_set_all:
        line_pts = list(bresenham(pi[0][0], pi[0][1], pi[-1][0], pi[-1][1]))
        for pt in line_pts:
            new_img[pt] = 255
    return vector_set_all, pts_set_all, new_img[L:L+r, L:L+c]





def _compute_triangle_vertex(pts):
    """
    pts: Nx2 list
    """
    return list(combinations(pts,3))
    
def _compute_triangle_side_vector(vertex):
    """
    vertex: tuple, contain 3 points.
    """
    a, b, c = vertex
    
    s1 = ( (a[0]-b[0])**2 + (a[1]-b[1])**2 )**0.5
    s2 = ( (b[0]-c[0])**2 + (b[1]-c[1])**2 )**0.5
    s3 = ( (c[0]-a[0])**2 + (c[1]-a[1])**2 )**0.5

    # Long to short.
    side_set = sorted([(s1, c), (s2, a), (s3, b)], reverse = True, key = lambda s: s[0])
    vector = [1, side_set[1][0]/side_set[0][0], side_set[2][0]/side_set[0][0]]
    return vector, side_set


def _triangle_vectors(pts):
    """
    pts: Nx2 points list.
    """
    triangles = _compute_triangle_vertex(pts)
    data = []
    for vertex in triangles:
        tri_vector, tri_data = _compute_triangle_side_vector(vertex)
        data.append([
            tri_vector,
            tri_data
        ])
    return data

def _euclidean_dist(vec1, vec2):
    
    return ( (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2 + (vec1[2] - vec2[2])**2 )**0.5

def _get_keypoint(key_point):
    pts = []
    for kp_i in key_point:
        x = round(kp_i.pt[0])
        y = round(kp_i.pt[1])
        if [x , y] not in pts:
            pts.append([
                x,
                y            
            ])
    return pts
        
def _convert_cartesian_to_polar(pts):
    """
    pts: Nx2 list
    """
    polar_list = []
    for pt in pts:
        rho = np.sqrt(pt[0]**2 + pt[1]**2)
        phi = np.arctan2(pt[1], pt[0])
        polar_list.append([
            rho, phi
        ])
    return polar_list

def _kmeans_classify(pts, k_clusters=3):

    point_list = []
    for i in range(k_clusters):
        point_list.append([])
    kmeans = KMeans(n_clusters=k_clusters)
    kmeans.fit(pts)
    group = kmeans.predict(pts)
    for i in range(len(group)):
        group_i = group[i]
        point_list[group_i].append([
            pts[i,0],
            pts[i,1]
        ])
        
    return point_list

def _find_hough_circles(img):
    # param2 < 25
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=10, param2=20, minRadius=0, maxRadius=50)
    circles = np.uint16(np.around(circles))
    return circles

def _fit_model(pts1, pts2):

    ux1, uy1 = np.mean(pts1, axis=0)
    ux2, uy2 = np.mean(pts2, axis=0)

    dpts1 = pts1 - np.array([ux1, uy1])
    dpts2 = pts2 - np.array([ux2, uy2])
    
    eucli_dist_1 = np.sum(dpts1**2, axis=1)**0.5
    eucli_dist_2 = np.sum(dpts2**2, axis=1)**0.5

    # ratio = np.mean(eucli_dist_2/eucli_dist_1)
    ratio = 1

    theta_1 = np.arctan2(dpts1[:,1], dpts1[:,0])
    theta_2 = np.arctan2(dpts2[:,1], dpts2[:,0])

    error_theta = np.mean(theta_1 - theta_2)

    pts1_trans = rot(pts=pts1.T, theta=error_theta, ratio=ratio).T
    dt = np.mean(pts2 - pts1_trans, axis=0)

    return error_theta/np.pi*180.0, ratio, dt

def rot(pts, theta, ratio):
    
    R = np.array([
        [np.cos(theta) , np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])*ratio

    return R.dot(pts)


if __name__ == "__main__":
    
    """
    Show Original Image Process
    """
    Step_param = 6




    if Step_param >= 1:
        """
        1 - load images
        """
        # img1 = cv2.imread('alg_test_maps//hand1.png')
        # img2 = cv2.imread('alg_test_maps//hand1.png')

        img1 = cv2.imread('Gazebo_test_map//tesst1.png')
        img2 = cv2.imread('Gazebo_test_map//test0223_2.png')

        # img1 = cv2.imread('alg_test_maps//small_test1.png')
        # img2 = cv2.imread('alg_test_maps//small_test3.png')

    if Step_param >= 2:
        """
        2 - grayscale
        """
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        if 0:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Original Image")
            ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Original Image")
            fig.tight_layout()

    if Step_param >= 3:
        """
        3 - binary image
        """
        img1 = _transform_state(img1)
        img2 = _transform_state(img2)
        
        if 0:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Binary Image")
            ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Binary Image")
            fig.tight_layout()
        

    if Step_param >= 6:
        """
        6 - special erode: find center of each mask.
        
            膨脹、侵蝕等操作會增加地圖影像的大小，增加長、寬各 2L 格。
        """
        L = 2
        img1 = _dilate_image(img1, L=L)
        img2 = _dilate_image(img2, L=L)

        if 1:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Extract Interest Point")
            ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Extract Interest Point")
            fig.tight_layout()

        # img1 = _erode_image(img1, L=1)
        # img2 = _erode_image(img2, L=1)


        # if 1:
        #     fig, ax = plt.subplots(1, 2)
        #     ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Extract Interest Point")
        #     ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Extract Interest Point")
        #     fig.tight_layout()

        img1_sp = _erode_test(img1, L=L)
        img2_sp = _erode_test(img2, L=L)

        if 1:
            img11 = _dilate_image(img1_sp, L=1)
            img22 = _dilate_image(img2_sp, L=1)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img11, cmap='gray'), ax[0].set_title("OGM 1 - Extract Interest Point")
            ax[1].imshow(img22, cmap='gray'), ax[1].set_title("OGM 2 - Extract Interest Point")
            fig.tight_layout()


        """
        test2
        """
        for i in range(3):
            v1, p1, n1 = _image_boundart_follow_test_2(image=img1, image_sp=img1_sp)
            v2, p2, n2 = _image_boundart_follow_test_2(image=img2, image_sp=img2_sp)

            # v1 = np.array(vector1)
            # v2 = np.array(vector2)


            if 1:
                # fig, ax = plt.subplots(1, 2)
                # for i in range(len(v1)):
                #     vi = np.array(v1[i])
                #     pi = np.array(p1[i])
                #     # ax[0].scatter(vi[:,0], vi[:,1], s=1)
                #     ax[0].plot(pi[:,0], pi[:,1])

                # for i in range(len(v2)):
                #     vi = np.array(v2[i])
                #     pi = np.array(p2[i])
                #     # ax[0].scatter(vi[:,0], vi[:,1], s=1)
                #     ax[1].plot(pi[:,0], pi[:,1])
                # fig.tight_layout()

                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(n1, cmap='gray')
                ax[1].imshow(n2, cmap='gray')
                fig.tight_layout()


                img1 = _dilate_image(n1, L=2)
                img2 = _dilate_image(n2, L=2)
            
                img1_sp = _erode_test(img1, L=2)
                img2_sp = _erode_test(img2, L=2)

                img11 = _dilate_image(img1_sp, L=1)
                img22 = _dilate_image(img2_sp, L=1)
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(img11, cmap='gray')
                ax[1].imshow(img22, cmap='gray')
                fig.tight_layout()


    if Step_param >= 7:
        """
        7 - boundary_follow: to correct the location.

            邊緣追蹤也會增加地圖影像，增加長、寬各 2L 格。
        """
        # img1, img1_list = _image_boundary_follow(img1_sp, L=20)
        # img2, img2_list = _image_boundary_follow(img2_sp, L=20)
        img1 = _erode_image(img1, L=L-1)
        img2 = _erode_image(img2, L=L-1)
        img1_sp = _enlarge_image(img1_sp, L=L-1)
        img2_sp = _enlarge_image(img2_sp, L=L-1)

        print("---No.1---")
        img1b, img1_list = _image_boundary_follow_test(img1, img1_sp, L=40)
        print("---No.2---")
        img2b, img2_list = _image_boundary_follow_test(img2, img2_sp, L=40)

        if 1:
            img11 = _dilate_image(img1b, L=1)
            img22 = _dilate_image(img2b, L=1)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img11, cmap='gray'), ax[0].set_title("OGM 1 - Correction")
            ax[1].imshow(img22, cmap='gray'), ax[1].set_title("OGM 2 - Correction")
            # ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
            # ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
            fig.tight_layout()

        if 0:
            img11 = _dilate_image(img1_list, L=0)
            img22 = _dilate_image(img2_list, L=0)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img11, cmap='gray'), ax[0].set_title("OGM 1 - Correction")
            ax[1].imshow(img22, cmap='gray'), ax[1].set_title("OGM 2 - Correction")
            # ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
            # ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
            fig.tight_layout()

    if Step_param >= 8:
        """
        8 - feature detection
        """
        img1_1, kp1_1, des1_1 = _sift_feature(img1b)
        img2_1, kp2_1, des2_1 = _sift_feature(img2b)
        
        if 1:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img1_1, cmap='gray'), ax[0].set_title("OGM 1 - SIFT Feature")
            ax[1].imshow(img2_1, cmap='gray'), ax[1].set_title("OGM 2 - SIFT Feature")
            ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
            ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
            fig.tight_layout()



    plt.show()
