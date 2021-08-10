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


def _icp_final_correction(img1, img2, pre_X):
    pre_x, pre_y, pre_yaw, pre_scale = pre_X    
    # 先對img2進行縮放
    img3 = cv2.resize(img2, (0, 0), fx=1/pre_scale, fy=1/pre_scale, interpolation=cv2.INTER_CUBIC)
    r,c = np.shape(img3)
    for i in range(r):
        for j in range(c):
            if img3[i,j] > 20:
                img3[i,j] = 255
            else:
                img3[i,j] = 0
    # 重新計算tf
    dx = pre_x/pre_scale
    dy = pre_y/pre_scale
    dyaw = pre_yaw


    x1, y1 = np.where(img1 == 255)
    pts1 = np.vstack((x1,y1)).T
    x3, y3 = np.where(img3 == 255)
    pts3 = np.vstack((x3,y3)).T
    # 第一次變換
    pts3 = (RANSAC_Map_Merging()._rotation(-dyaw).dot(pts3.T).T + np.array([dx, dy]))

    # 用來檢測icp輸入點是否已初步對齊
    plt.figure()
    plt.scatter(pts1[:,0], pts1[:,1], s=5, c='black')
    plt.scatter(pts3[:,0], pts3[:,1], s=2, c='red')

    SM = ScanMatching()
    _, _, [x, y, yaw] = SM.icp(   reference_points=pts1, 
                                    points=pts3, 
                                    max_iterations=5000, 
                                    distance_threshold=10, 
                                    convergence_translation_threshold=1e-5, 
                                    convergence_rotation_threshold=1e-3, 
                                    point_pairs_threshold=300, 
                                    verbose=False)


    # 用2階段轉換，或者之後把兩個轉換整理在一起：
    # 1st: pre_X
    # 2nd: x, y, yaw

    c = math.cos(yaw)
    s = math.sin(yaw)
    integrate_yaw = yaw + pre_yaw
    integrate_x =  (c*dx - s*dy + x)
    integrate_y =  (s*dx + c*dy + y)
    # 再把tf從img3轉換為img2
    # （皆參考img1）
    final_scale = pre_scale
    final_x = integrate_x * final_scale
    final_y = integrate_y * final_scale
    final_yaw = integrate_yaw
    print("1st Transformation: %f, %f, %f, %f"%(pre_x, pre_y, pre_yaw, pre_scale))
    print("2nd Transformation: %f, %f, %f"%(x, y, yaw)) 
    print("Final Transformation: %f, %f, %f, %f"%(final_x, final_y, final_yaw, final_scale))

    map_fused = _merging_map(dx=final_x, dy=final_y, dtheta=final_yaw, dr=final_scale, map1=img1, map2=img2)
    r, c = np.shape(map_fused)
    for i in range(r):
        for j in range(c):
            if map_fused[i, j]  > 10:
                map_fused[i, j] = 255
            else:
                map_fused[i, j] = 0
    map_fused = _modify_map_size(merged_map=map_fused)



    return map_fused, (final_x, final_y, final_yaw, final_scale)

def _enlarge_image(img, L=1, pixel_value=0):
    img_r, img_c = np.shape(img)
    new_img = np.ones((img_r+2*L, img_c+2*L))*pixel_value
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
    

    # relation, value, inlier_point_match = RMM._ransac_find_rotation_translation(pts_set_1=pts2, pts_set_2=pts1, sigma=5, max_iter=1000)
    relation, value, inlier_point_match = RMM._ransac_find_all(pts_set_1=pts2, pts_set_2=pts1, sigma=5, max_iter=1000)
    # print("- Inlier Percent: %f"%value)

    # Because the coordinates between the maps and the SIFT features are different:
    # SIFT Features:    Right: +x, Down:  +y
    # Maps:             Down:  +x, Right: +y
    # Hence the dx and dy should be changed.
    dx = relation[1]
    dy = relation[0]
    dyaw = relation[2]
    dr = relation[3]
    # print("- (x, y, t): (%f, %f, %f)"%(dx,dy,dyaw))
    # map_align = cv2.resize(map_align, (0, 0), fx=dr, fy=dr, interpolation=cv2.INTER_NEAREST)
    
    """
    這裡出錯了，出錯原因是因為ratio的值很怪，所以要去確認ratio。
    我猜問題出在比例小的圖片sift擷取的特徵點。
    """
    
    print(dr)
    if dr == 0:
        return 0, 0, (0,0,0)
    map_align = _scale_image(image=map_align, ratio=dr)


    r, c = np.shape(map_align)
    for i in range(r):
        for j in range(c):
            if map_align[i, j]  > 100:
                map_align[i, j] = 255
            else:
                map_align[i, j] = 0

    index, agr, dis = RMM._similarity_index_2(x=[dx, dy, dyaw], map1=map_ref, map2=map_align)
    """"""

    # index, agr, dis = RMM._similarity_index_2(x=[dx, dy, dyaw], map1=map_ref, map2=map_align)
    # print("- Similarity Index: %f\n- Agree Number: %f\n- Disargee Number: %f"%(index, agr, dis))
    
    map_merged = MF._merging_map(dx=dx, dy=dy, dtheta=dyaw, map1=map_ref, map2=map_align)
    map_merged = map_merged.astype(np.uint8)
    map_merged = MF._modify_map_size(merged_map=map_merged)

    return map_merged, index, (dx, dy, dyaw)


def _scale_image(image, ratio):
    image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return image


def _rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If the center of the rotation is not defined, use the center of the image.
    if center is None:
        (cX, cY) = (w // 2, h // 2)
    else:
        (cX, cY) = center
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

# def _transform_state(input_map, state_range=[230, 100], gray=255):
#     """
#     Probabilities in OGMs will be transformed into 3 values: 0, 128, 255. \n
#     Occupied Space:   0\n
#     Unknown  Space: 128\n
#     Free     Space: 255
#     """
#     r, c = np.shape(input_map)
#     for i in range(r):
#         for j in range(c):

#             if input_map[i, j]  > state_range[0]:  # White >> Free Space
#                 input_map[i, j] = 255
            
#             elif input_map[i, j] < state_range[1]: # Black >> Occupied Space
#                 input_map[i, j] = 0
            
#             else:                       # Gray  >> Unknown Space
#                 input_map[i, j] = gray
#     return input_map

def _transform_state(input_map):
    r, c = np.shape(input_map)
    for i in range(r):
        for j in range(c):
            if input_map[i, j]  < 100:
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
    new_img = np.zeros((img_r, img_c))
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
        flag = window==B1
        if flag.all():
            new_img[x-L, y-L] = 255
            img_tmp[x-L:x+L+1, y-L:y+L+1] = 0
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
    
    """
    Show Original Image Process
    """
    Original_Process_state = True

    """
    1 - load images
    """
    img1 = cv2.imread('Gazebo_test_map//tesst1.png')
    # img2 = cv2.imread('Gazebo_test_map//test0223_2.png')

    # img2 = cv2.imread('Gazebo_test_map//sigma_difference//0426_map_01.png')
    # img2 = cv2.imread('Gazebo_test_map//sigma_difference//0426_map_005.png')

    img2 = cv2.imread('Gazebo_test_map//ratio_difference//0503_r01.png')
    # img2 = cv2.imread('Gazebo_test_map//ratio_difference//0503_r02.png')


    """
    2 - grayscale
    """
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    if 0:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Original Image")
        ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Original Image")
        ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()

    if Original_Process_state:
        img_gray_1 = copy.copy(img1)
        img_gray_2 = copy.copy(img2)


    """
    3 - binary image
    """
    img1 = _transform_state(img1)
    img2 = _transform_state(img2)
    
    if 0:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Binary Image")
        ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Binary Image")
        ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()
    
    if Original_Process_state:
        img_gray_1 = _simple_transform_state(img_gray_1)
        img_gray_2 = _simple_transform_state(img_gray_2)
        if 0:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img_gray_1, cmap='gray'), ax[0].set_title("OGM 1")
            ax[1].imshow(img_gray_2, cmap='gray'), ax[1].set_title("OGM 2")
            ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
            ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
            fig.tight_layout()


    """
    4 - radon and rotate
    """
    dtheta_1, sino_1 = _radon_find_theta(img1)
    dtheta_2, sino_2 = _radon_find_theta(img2)

    if 0:
        dx1 = -0.5 * 180.0 / max(img1.shape)
        dy1 = 0.5 / sino_1.shape[0]
        dx2 = -0.5 * 180.0 / max(img2.shape)
        dy2 = 0.5 / sino_2.shape[0]

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(sino_1, cmap='gray', extent=(-dx1, 180.0+dx1, -dy1, sino_1.shape[0] + dy1)), ax[0].set_title("OGM 1 - Radon Transform")
        ax[1].imshow(sino_2, cmap='gray', extent=(-dx2, 180.0+dx2, -dy2, sino_2.shape[0] + dy2)), ax[1].set_title("OGM 2 - Radon Transform")
        ax[0].set_xlabel("Projection angle (deg)"), ax[0].set_ylabel("Projection position (pixels)")
        ax[1].set_xlabel("Projection angle (deg)"), ax[1].set_ylabel("Projection position (pixels)")
        fig.tight_layout()

    img1 = _rotate_image(img1, 90-dtheta_1)
    img2 = _rotate_image(img2, 90-dtheta_2)

    if 0:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Rotated Image")
        ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Rotated Image")
        ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()

    if Original_Process_state:
        img_gray_1 = _simple_rotate_image(img_gray_1, 90-dtheta_1)
        img_gray_2 = _simple_rotate_image(img_gray_2, 90-dtheta_2)
        if 0:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img_gray_1, cmap='gray'), ax[0].set_title("OGM 1 - Rotated Image")
            ax[1].imshow(img_gray_2, cmap='gray'), ax[1].set_title("OGM 2 - Rotated Image")
            ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
            ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
            fig.tight_layout()


    """
    Closing
    """
    img1 = _dilate_image(img1, L=1)
    img1 = _erode_image(img1, L=1)
    img2 = _dilate_image(img2, L=1)
    img2 = _erode_image(img2, L=1)

    if Original_Process_state:
        img_gray_1 = _enlarge_image(img_gray_1, L=1, pixel_value=128)
        img_gray_2 = _enlarge_image(img_gray_2, L=1, pixel_value=128)

    """
    5 - special erode: find center of each mask.
    
        膨脹、侵蝕等操作會增加地圖影像的大小，增加長、寬各 2L 格。
    """
    img1 = _dilate_image(img1, L=2)
    img1 = _erode_test(img1, L=2)

    img2 = _dilate_image(img2, L=1)
    img2 = _erode_test(img2, L=1)

    if 0:
        img11 = _dilate_image(img1, L=1)
        img22 = _dilate_image(img2, L=1)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img11, cmap='gray'), ax[0].set_title("OGM 1 - Extract Interest Point")
        ax[1].imshow(img22, cmap='gray'), ax[1].set_title("OGM 2 - Extract Interest Point")
        ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()

    if Original_Process_state:
        img_gray_1 = _enlarge_image(img_gray_1, L=2, pixel_value=128)
        img_gray_2 = _enlarge_image(img_gray_2, L=1, pixel_value=128)
        if 0:
            img11 = _dilate_image(img1, L=1)
            img22 = _dilate_image(img2, L=1)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img_gray_1, cmap='gray'), ax[0].set_title("OGM 1 - Extract Interest Point")
            ax[1].imshow(img_gray_2, cmap='gray'), ax[1].set_title("OGM 2 - Extract Interest Point")
            ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
            ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
            fig.tight_layout()


    """
    6 - boundary_follow: to correct the location.

        邊緣追蹤也會增加地圖影像，增加長、寬各 2L 格。
    """
    img1, img1_list = _image_boundary_follow(img1, L=20)
    img2, img2_list = _image_boundary_follow(img2, L=20)

    if 0:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Correction")
        ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Correction")
        ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()

    if Original_Process_state:
        img_gray_1 = _enlarge_image(img_gray_1, L=20, pixel_value=128)
        img_gray_2 = _enlarge_image(img_gray_2, L=20, pixel_value=128)
        if 0:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img_gray_1, cmap='gray'), ax[0].set_title("OGM 1 - Correction")
            ax[1].imshow(img_gray_2, cmap='gray'), ax[1].set_title("OGM 2 - Correction")
            ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
            ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
            fig.tight_layout()

    """
    7 - feature detection
    """
    img1_1, kp1_1, des1_1 = _sift_feature(img1)
    img2_1, kp2_1, des2_1 = _sift_feature(img2)
    
    if 0:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img1_1, cmap='gray'), ax[0].set_title("OGM 1 - SIFT Feature")
        ax[1].imshow(img2_1, cmap='gray'), ax[1].set_title("OGM 2 - SIFT Feature")
        ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()


    # """
    # 8 - k means clustering
    # """


    # kp_point_1 = np.array(_get_keypoint(kp1_1))
    # kp_point_2 = np.array(_get_keypoint(kp2_1))

    # sub_pts_1 = _kmeans_classify(pts=kp_point_1, k_clusters=2)
    # sub_pts_2 = _kmeans_classify(pts=kp_point_2, k_clusters=2)
    # dr = 0
    # time = 0
    # while dr < 0.4:
    #     print("No. %i"%(time+1))
    #     if time > 50:
    #         break
    #     time += 1
    #     for raw_pts1 in sub_pts_1:
    #         for raw_pts2 in sub_pts_2:

    #             # pts1 = np.array(raw_pts1)
    #             # pts2 = np.array(raw_pts2)
                
    #             # print(pts1.shape)
    #             # print(pts2.shape)

    #             # 建立配對關係
    #             pts1, pts2 = [], []
    #             for kp1 in raw_pts1:
    #                 for kp2 in raw_pts2: 
    #                     pts1.append([
    #                         kp1[0],
    #                         kp1[1]
    #                     ])
    #                     pts2.append([
    #                         kp2[0],
    #                         kp2[1]
    #                     ])
                        
    #             pts1 = np.array(pts1)
    #             pts2 = np.array(pts2)
                
    #             l1 = len(raw_pts1)
    #             l2 = len(raw_pts2)

    #             # 計算變換關係
    #             relation, value, inlier_point_match = RANSAC_Map_Merging()._ransac_find_all(pts_set_1=pts2, pts_set_2=pts1, sigma=5, max_iter=l1*l2)
    #             dx = relation[1]
    #             dy = relation[0]
    #             dyaw = relation[2]
    #             dr = relation[3]
                
    #             # if not dr > 0.4:
    #             #     continue
    #             # print( np.shape(inlier_point_match))
    #             # print(relation)
    #             t1 , t2 = [],[]
    #             for i in inlier_point_match:
    #                 t2 += [i[0]]
    #                 t1 += [i[1]]

    #             if dr > 0.4:
    #                 # dx = 0
    #                 # dy = 0
    #                 # dyaw = 0
    #                 # dr = 0.5
    #                 map_f = _merging_map(dx=dx, dy=dy, dtheta=dyaw, dr=dr, map1=img1, map2=img2)

    #                 r, c = np.shape(map_f)
    #                 for i in range(r):
    #                     for j in range(c):
    #                         if map_f[i, j]  > 10:
    #                             map_f[i, j] = 255
    #                         else:
    #                             map_f[i, j] = 0
                    

    #                 # map_f = map_f.astype(np.uint8)
    #                 map_f = _modify_map_size(merged_map=map_f)
    #                 fig, ax = plt.subplots(3, 1)
    #                 ax[0].imshow(map_f, cmap='gray')
    #                 ax[1].imshow(img1, cmap='gray')
    #                 ax[2].imshow(img2, cmap='gray')
    #                 print("Success")
    #                 print("dx, dy, dyaw, dr = %f, %f, %f, %f"%(dx, dy, dyaw, dr))
    #             else:
    #                 print("Fail")


    """
    10 - ICP, final correction
    """

    # 這邊用函數取代：_icp_final_correction

    """先前案例之變換：tesst1 - 0503_r01"""
    dx, dy, dyaw, dr = 257.247974, 244.393586, -3.133371, 0.494268


    # img1 = _erode_image(img1, L=1)
    # img2 = _erode_image(img2, L=1)
    img1 = _transform_state(copy.copy(img_gray_1))
    img2 = _transform_state(copy.copy(img_gray_2))
    # img1 = _dilate_image(img1, L=1)
    # img2 = _dilate_image(img2, L=1)
    # img_gray_1 = _enlarge_image(img_gray_1, L=1, pixel_value=128)
    # img_gray_2 = _enlarge_image(img_gray_2, L=1, pixel_value=128)

    
    map_fused, final_tf = _icp_final_correction(img1=img1, img2=img2, pre_X=(dx, dy, dyaw, dr))
    plt.figure()
    plt.imshow(map_fused, cmap='gray')

    if Original_Process_state:
        # dx, dy, dyaw, dr= final_tf
        # img_gray_merged = MAP_Function()._merging_map_ratio(dx=dx, dy=dy, dtheta=dyaw, dr=dr, map1=img_gray_1, map2=img_gray_2)
        # img_gray_merged = img_gray_merged.astype(np.uint8)
        # img_gray_merged = MAP_Function()._modify_map_size(merged_map=img_gray_merged)
        # cv2.imwrite("yonlin_merging_1.png", img_gray_merged) 

        """
        TEST - start
        """
        dx, dy, dyaw, dr= final_tf
        dx = dx/dr
        dy = dy/dr
        img_gray_3 = cv2.resize(img_gray_2, (0, 0), fx=1/dr, fy=1/dr, interpolation=cv2.INTER_CUBIC)
        row, col = np.shape(img_gray_3)
        for i in range(row):
            for j in range(col):
                if img_gray_3[i,j] >= 192:
                    img_gray_3[i,j] = 255
                elif img_gray_3[i,j] <= 64:
                    img_gray_3[i,j] = 0
                else:
                    img_gray_3[i,j] = 128

        img_gray_merged = MAP_Function()._merging_map(dx=dx, dy=dy, dtheta=dyaw, map1=img_gray_1, map2=img_gray_3)
        img_gray_merged = img_gray_merged.astype(np.uint8)
        img_gray_merged = MAP_Function()._modify_map_size(merged_map=img_gray_merged)
        """
        TEST - end
        """


        _, agr, dis, map_v = RANSAC_Map_Merging()._similarity_index_2(x=[dx, dy, dyaw], map1=img_gray_1, map2=img_gray_3)
        ai = (agr)/(agr+dis)
        print("Acceptance Index: %f"%ai)

        fig, ax = plt.subplots(3, 1)
        ax[0].imshow(img_gray_merged, cmap='gray')
        ax[1].imshow(img_gray_1, cmap='gray')
        ax[2].imshow(img_gray_2, cmap='gray')

        ax[0].set_title("OGM - Merging Result")
        ax[1].set_title("OGM 1")
        ax[2].set_title("OGM 2")
        fig.tight_layout()

        plt.figure()
        ax = plt.gca()      
        plt.margins(0, 0)        
        ax.imshow(map_v)
        ax.xaxis.set_ticks_position('top') 
        ax.set_xlim(795,1220)
        ax.set_ylim(775,1195)
        ax.invert_yaxis()
        if 1:
            plt.savefig("match_visualization_tesst1_0503r01.png", bbox_inches='tight')


    plt.show()