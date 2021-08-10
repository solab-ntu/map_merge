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

def _find_max_rectangle(img):

    white_pixel = np.where(img == 255)

    if white_pixel[0].size == 0:
        return 0

    min_x = np.amin(white_pixel[0])
    max_x = np.amax(white_pixel[0])
    min_y = np.amin(white_pixel[1])
    max_y = np.amax(white_pixel[1])

    return (max_x - min_x)*(max_y - min_y)

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
    index, agr, dis = RMM._similarity_index_2(x=[ddx, ddy, ddyaw], map1=map_ref, map2=map_align)
    """"""

    # index, agr, dis = RMM._similarity_index_2(x=[dx, dy, dyaw], map1=map_ref, map2=map_align)
    # print("- Similarity Index: %f\n- Agree Number: %f\n- Disargee Number: %f"%(index, agr, dis))
    
    map_merged = MF._merging_map(dx=dx, dy=dy, dtheta=dyaw, map1=map_ref, map2=map_align)
    map_merged = map_merged.astype(np.uint8)
    map_merged = MF._modify_map_size(merged_map=map_merged)

    return map_merged, index

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

if __name__ == "__main__":
    
    """
    1 - load images
    """
    img2 = cv2.imread('Gazebo_test_map//tesst2.png')
    # img2 = cv2.imread('Gazebo_test_map//test0223_1.png')

    img1 = cv2.imread('Gazebo_test_map//sigma_difference//0426_map_01.png')
    # img2 = cv2.imread('Gazebo_test_map//sigma_difference//0426_map_005.png')

    # img1 = cv2.imread('Gazebo_test_map//ratio_difference//0503_r01.png')
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

    """
    3 - binary image
    """
    img1 = _transform_state(img1)
    img2 = _transform_state(img2)
    
    if 1:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Binary Image")
        ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Binary Image")
        # ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
        # ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()


    """
    4 - dilate : help radon tf.
    """
    # img1 = _dilate_image(img1, L=1)
    # img2 = _dilate_image(img2, L=1)
    
    # if 0:
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Dilated Image")
    #     ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Dilated Image")
    #     ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
    #     ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
    #     fig.tight_layout()

    """
    5 - radon and rotate
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

    if 1:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Rotated Image")
        ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Rotated Image")
        ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()



    """
    6 - Dilate
    """
    num_1 = np.sum(img1==255)
    num_2 = np.sum(img2 == 255)
    img1 = _dilate_image(img1, L=2)
    img2 = _dilate_image(img2, L=2)

    if 1:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Dilated Image")
        ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Dilated Image")
        ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()


    """
    7 - erode : change the boundary width to 1
    """

        
    
    
    ori_rec_1 = _find_max_rectangle(img1)
    rec_1 = _find_max_rectangle(img1)
    while not rec_1/ori_rec_1 < 0.9:
        img_tmp = _erode_image(img=img1, L=1)
        rec_1 = _find_max_rectangle(img_tmp)
        if not rec_1/ori_rec_1 < 0.9:
            img1 = img_tmp
    print(np.sum(img1==255)/num_1)

    
    ori_rec_2 = _find_max_rectangle(img2)
    rec_2 = _find_max_rectangle(img2)
    while not rec_2/ori_rec_2 < 0.9:
        img_tmp = _erode_image(img=img2, L=2)
        rec_2 = _find_max_rectangle(img_tmp)
        if not rec_2/ori_rec_2 < 0.9:
            img2 = img_tmp
    print(np.sum(img2==255)/num_2)

    if 1:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Eroded Image")
        ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Eroded Image")
        ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()


    # """
    # 8 - special erode: find center of each mask.
    # """
    # img1 = _dilate_image(img1, L=2)
    # img2 = _dilate_image(img2, L=2)
    # img1 = _erode_test(img1, L=2)
    # img2 = _erode_test(img2, L=2)

    # if 1:
    #     img11 = _dilate_image(img1, L=1)
    #     img22 = _dilate_image(img2, L=1)
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(img11, cmap='gray'), ax[0].set_title("OGM 1 - Extract Interest Point")
    #     ax[1].imshow(img22, cmap='gray'), ax[1].set_title("OGM 2 - Extract Interest Point")
    #     ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
    #     ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
    #     fig.tight_layout()

    # """
    # 9 - boundary_follow: to correct the location.
    # """
    # img1, img1_list = _image_boundary_follow(img1, L=20)
    # img2, img2_list = _image_boundary_follow(img2, L=20)

    # if 1:
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(img1, cmap='gray'), ax[0].set_title("OGM 1 - Correction")
    #     ax[1].imshow(img2, cmap='gray'), ax[1].set_title("OGM 2 - Correction")
    #     ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
    #     ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
    #     fig.tight_layout()


    # """
    # 10 - feature detection
    # """
    # img1_1, kp1_1, des1_1 = _sift_feature(img1)
    # img2_1, kp2_1, des2_1 = _sift_feature(img2)
    
    # if 0:
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(img1_1, cmap='gray'), ax[0].set_title("OGM 1 - SIFT Feature")
    #     ax[1].imshow(img2_1, cmap='gray'), ax[1].set_title("OGM 2 - SIFT Feature")
    #     ax[0].get_xaxis().set_visible(False), ax[0].get_yaxis().set_visible(False)
    #     ax[1].get_xaxis().set_visible(False), ax[1].get_yaxis().set_visible(False)
    #     fig.tight_layout()
    
    # """
    # 11 - k means clustering
    # """
    
    # kp_point_1 = np.array(_get_keypoint(kp1_1))
    # kp_point_2 = np.array(_get_keypoint(kp2_1))

    # sub_pts_1 = _kmeans_classify(pts=kp_point_1, k_clusters=2)
    # sub_pts_2 = _kmeans_classify(pts=kp_point_2, k_clusters=2)

    # best_result = None
    # best_index = 0
    # for pts_list_1 in sub_pts_1:
    #     for pts_list_2 in sub_pts_2:
    #         map_merged, index = _sub_kp_merge(maps_to_be_merged=[img1, img2], sub_kp_list=[pts_list_1, pts_list_2])
    #         # if index > best_index:
    #         if 1:
    #             best_index = index
    #             best_result = map_merged
    #             best_sub1 = np.array(pts_list_1)
    #             best_sub2 = np.array(pts_list_2)
    
    
    #             fig, ax = plt.subplots(1, 3)
    #             ax[0].imshow(best_result, cmap='gray')
    #             ax[1].imshow(img1, cmap='gray')
    #             ax[2].imshow(img2, cmap='gray')

    #             ax[1].scatter(best_sub1.T[0], best_sub1.T[1], s=3)
    #             ax[2].scatter(best_sub2.T[0], best_sub2.T[1], s=3)
    #             ax[0].set_title("OGM - Merging Result")
    #             ax[1].set_title("OGM 1 - Clustering Keypoints")
    #             ax[2].set_title("OGM 2 - Clustering Keypoints")
    #             fig.tight_layout()


    plt.show()