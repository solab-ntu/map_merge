# -*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import random
import copy
import matplotlib.gridspec as gridspec


class RANSAC_Map_Merging():

    def __init__(self):
        pass

    def _cal_distance(self, x1, y1, x2, y2):
        return ((x1-x2)**2 + (y1-y2)**2)**0.5

    def _rotation(self, theta):
        T = np.array([
        [math.cos(theta), math.sin(theta)],
        [-math.sin(theta), math.cos(theta)]
        ])
        return T

    def _ransac_find_scale(self, pts_set_1, pts_set_2, sigma, max_iter=1000):
        """

        - pts_set_1: Nx2 ndarray\n
        - pts_set_2: Nx2 ndarray\n
        - sigma: error threshold\n
        - max_iter: Maximum times of the iteration
        
        """
        length, _ = np.shape(pts_set_1)
        best_ratio = 0
        total_inlier, pre_total_inlier = 0, 0

        for i in range(max_iter):
            
            index = random.sample(range(length), 2)
            x11 = pts_set_1[index[0], 0]
            y11 = pts_set_1[index[0], 1]
            x12 = pts_set_1[index[1], 0]
            y12 = pts_set_1[index[1], 1]
            
            x21 = pts_set_2[index[0], 0]
            y21 = pts_set_2[index[0], 1]
            x22 = pts_set_2[index[1], 0]
            y22 = pts_set_2[index[1], 1]

            dist_1 = self._cal_distance(x11, y11, x12, y12)
            dist_2 = self._cal_distance(x21, y21, x22, y22)

            mean_1 = [(x11+x12)/2, (y11+y12)/2]
            mean_2 = [(x21+x22)/2, (y21+y22)/2]

            ratio = dist_1 / dist_2
            
            for j in range(length):
                x1j = pts_set_1[j,0]
                y1j = pts_set_1[j,1]
                x2j = pts_set_2[j,0]
                y2j = pts_set_2[j,1]

                dist_1_j = self._cal_distance(mean_1[0], mean_1[1], x1j, y1j)
                dist_2_j = self._cal_distance(mean_2[0], mean_2[1], x2j, y2j)

                ratio_j = dist_1_j / dist_2_j

                if abs(ratio - ratio_j) < sigma:

                    total_inlier += 1

            if total_inlier > pre_total_inlier:
                
                pre_total_inlier = total_inlier
                best_ratio = ratio
            
            total_inlier = 0
        return best_ratio

    def _ransac_find_rotation_translation(self, pts_set_1, pts_set_2, sigma=0.1, max_iter=1000):
        """
        - pts_set_1: Nx2 ndarray. \n
        - pts_set_2: Nx2 ndarray. \n
        - sigma: error threshold. \n
        - max_iter: Maximum times of the iteration
        """
        length, _ = np.shape(pts_set_1)
        best_relation = None
        total_inlier, pre_total_inlier = 0, 0

        for i in range(max_iter):
            
            index = random.sample(range(length), 2)
            x11 = pts_set_1[index[0], 0]
            y11 = pts_set_1[index[0], 1]
            x12 = pts_set_1[index[1], 0]
            y12 = pts_set_1[index[1], 1]
            
            x21 = pts_set_2[index[0], 0]
            y21 = pts_set_2[index[0], 1]
            x22 = pts_set_2[index[1], 0]
            y22 = pts_set_2[index[1], 1]

            v1 = [x11- x12, y11 -y12]
            u1 = [x21- x22, y21 -y22]

            dtheta =  math.atan2(v1[1], v1[0]) - math.atan2(u1[1], u1[0])
            ds = np.array([[x21],[y21]]) - self._rotation(dtheta).dot(np.array([[x11], [y11]]))
            

            for j in range(length):
                x1j = pts_set_1[j,0]
                y1j = pts_set_1[j,1]
                x2j = pts_set_2[j,0]
                y2j = pts_set_2[j,1]

                pt1j = np.array([
                                    [x1j],
                                    [y1j]
                                ])
                pt2j = np.array([
                                    [x2j],
                                    [y2j]
                                ])
                ptj_error = pt2j - self._rotation(dtheta).dot(pt1j) - ds

                if self._cal_distance(x1=0, y1=0, x2=ptj_error[0], y2=ptj_error[1]) < sigma:

                    total_inlier += 1

            if total_inlier > pre_total_inlier:
                
                pre_total_inlier = total_inlier
                dx, dy = ds[0], ds[1]
                best_relation = [dx, dy, dtheta]
            
            total_inlier = 0
        return best_relation, pre_total_inlier/length


    def _similarity_index(self, x, map1, map2):
        """
        Variable: x {dx, dy, dtheta} (float). \n
        Input Map Layers: (ndarray).
        
        Note: this method needs to be debugged.
        """
        # extract map layers
        map1_black_layer = MAP_Function()._extract_layer(map_input=map1, layer_type="black")
        map1_white_layer = MAP_Function()._extract_layer(map_input=map1, layer_type="white")
        map2_black_layer = MAP_Function()._extract_layer(map_input=map2, layer_type="black")
        map2_white_layer = MAP_Function()._extract_layer(map_input=map2, layer_type="white")

        dx = x[0]
        dy = x[1]
        dtheta = x[2]
        
        agr = 0.
        dis = 0.

        r1, c1 = np.shape(map1_black_layer)
        r2, c2 = np.shape(map2_black_layer)
        bound = max(r1, c1, r2, c2)

        new_map1_black_tmp = np.zeros((5*bound, 5*bound))
        new_map1_white_tmp = np.zeros((5*bound, 5*bound))
        new_map2_black_tmp = np.zeros((5*bound, 5*bound))
        new_map2_white_tmp = np.zeros((5*bound, 5*bound))

        new_map1_black_tmp[2*bound:2*bound+r1, 2*bound:2*bound+c1] = map1_black_layer
        new_map1_white_tmp[2*bound:2*bound+r1, 2*bound:2*bound+c1] = map1_white_layer
        new_map2_black_tmp[2*bound:2*bound+r2, 2*bound:2*bound+c2] = map2_black_layer
        new_map2_white_tmp[2*bound:2*bound+r2, 2*bound:2*bound+c2] = map2_white_layer


        new_map2_black_tmp = MAP_Function()._rotate_image(image=new_map2_black_tmp, angle=dtheta, center=(2*bound, 2*bound))
        new_map2_white_tmp = MAP_Function()._rotate_image(image=new_map2_white_tmp, angle=dtheta, center=(2*bound, 2*bound))
        new_map2_black_tmp = MAP_Function()._translate_image(image=new_map2_black_tmp, x=dx, y=dy)
        new_map2_white_tmp = MAP_Function()._translate_image(image=new_map2_white_tmp, x=dx, y=dy)

        dis = (new_map1_black_tmp * new_map2_white_tmp).sum() + (new_map1_white_tmp * new_map2_black_tmp).sum()
        agr = (new_map1_black_tmp * new_map2_black_tmp).sum() + (new_map1_white_tmp * new_map2_white_tmp).sum()
        plt.figure()
        plt.imshow(new_map1_white_tmp * new_map2_white_tmp, cmap='gray')
        
        if agr == 0 and dis ==0:
            return 0, 0, 0
        else:
            agr = agr*1.0
            dis = dis*1.0
            return  100.0*(agr-dis)/(agr+dis), agr, dis

    def _similarity_index_2(self, x, map1, map2):
        """
        Variable: x {dx, dy, dtheta} (float). \n
        Input Map Layers: (ndarray).
        """
        agr, dis = 0, 0

        r1, c1 = np.shape(map1)
        r2, c2 = np.shape(map2)
        extension = max(r2, c2)
        map_temp = np.ones(( r1 + extension*2 , c1 + extension*2 )) * 128
        map_temp[extension:extension+r1 , extension:extension+c1] = map1

        # 下面這步是為了先將Map1 和Map2的原點貼齊，否則Map2一開始會在擴充地圖的左上角。
        dx = x[0] + extension
        dy = x[1] + extension
        dtheta = x[2]
        for row in range(r2):
            for col in range(c2):

                new_x = int( math.cos(dtheta)*row - math.sin(dtheta)*col + dx)
                new_y = int( math.sin(dtheta)*row + math.cos(dtheta)*col + dy)

                if new_x >= extension and new_x <= extension + r1 and new_y >= extension and new_y <= extension + c1:
                    # Overlap area
                    if map_temp[new_x, new_y] >= 225:
                        if map2[row, col] >= 255: 
                            agr += 1
                        elif map2[row, col] <= 30:
                            dis += 1
                    elif map_temp[new_x, new_y] <= 30:
                        if map2[row, col] >= 255: 
                            dis += 1
                        elif map2[row, col] <= 30:
                            agr += 1

        return (agr - dis)/(agr + dis), agr, dis


class MAP_Function():
    def __init__(self):
        pass

    def _transform_state(self, input_map, state_range=[230, 100], gray=255):
        """
        Probabilities in OGMs will be transformed into 3 values: 0, 128, 255. \n
        Occupied Space:   0\n
        Unknown  Space: 128\n
        Free     Space: 255
        """

        r, c = np.shape(input_map)
        for i in range(r):
            for j in range(c):

                if input_map[i, j]  > state_range[0]:  # White >> Free Space
                    input_map[i, j] = 255
                
                elif input_map[i, j] < state_range[1]: # Black >> Occupied Space
                    input_map[i, j] = 0
                
                else:                       # Gray  >> Unknown Space
                    input_map[i, j] = gray

        return input_map

    def _entropy_filter(self, prob1, prob2):
        """
        This filter is used for selecting a prob with the least entropy. \n
        Input : probability in range (0,1); (prob1, prob2). \n
        Output: probability in range (0,1); (prob1, prob2, or merged prob).
        """


        # calculate merged prob.
        prob_merged = (prob1 + prob2)/2
        # Compute entropy for each prob.
        H1 = -prob1 * math.log(prob1) - (1-prob1) * math.log(1-prob1)
        H2 = -prob2 * math.log(prob2) - (1-prob2) * math.log(1-prob2)
        Hm = -prob_merged * math.log(prob_merged) - (1-prob_merged) * math.log(1-prob_merged)

        H_min = min(H1, H2, Hm)

        if H_min == H1:
            return prob1
        elif H_min == H2:
            return prob2
        else:
            return prob_merged

    def _merging_map(self, dx, dy, dtheta, map1, map2):
        
        r1, c1 = np.shape(map1)
        r2, c2 = np.shape(map2)
        extension = max(r2, c2)
        map_temp = np.ones(( r1 + extension*2 , c1 + extension*2 )) * 128
        map_temp[extension:extension+r1 , extension:extension+c1] = map1

        # 下面這步是為了先將Map1 和Map2的原點貼齊，否則Map2一開始會在擴充地圖的左上角。
        dx = dx + extension
        dy = dy + extension
        for row in range(r2):
            for col in range(c2):

                new_x = int( math.cos(dtheta)*row - math.sin(dtheta)*col + dx)
                new_y = int( math.sin(dtheta)*row + math.cos(dtheta)*col + dy)

                if new_x >= extension and new_x <= extension + r1 and new_y >= extension and new_y <= extension + c1:
                    # Overlap area, use Entropy filter to determine whether adopting mixed probability or not.
                    prob1 = (map_temp[new_x, new_y]+1)/257.0  # -> equals to map1 prob
                    prob2 = (map2[row, col]+1)/257.0          # -> map2 prob
                    
                    prob = self._entropy_filter(prob1, prob2)
                    map_temp[new_x, new_y] = prob*257-1

                else:
                    map_temp[new_x, new_y] = map2[row, col]

        return map_temp


    def _translate_image(self, image, x, y):

        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return shifted

    def _rotate_image(self, image, angle, center=None, scale=1.0):
        """
        angle unit: rad
        """
        (h, w) = image.shape[:2]
        # If the center of the rotation is not defined, use the center of the image.
        if center is None:
            center = (w / 2, h / 2)
        
        theta = angle/math.pi*180
        M = cv2.getRotationMatrix2D(center, theta, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated


    def _dilate_image(self, img, L=1):
        
        img_r, img_c = np.shape(img)
        new_img = np.ones((img_r, img_c))*255
        black_pts = np.where(img==0)
        _, length = np.shape(black_pts)
        for i in range(length):
            x = black_pts[0][i]
            y = black_pts[1][i]
            window = img[x-L:x+L+1, y-L:y+L+1]
            number = len(np.where(window==0)[0])
            if number > 1:
                new_img[x-L:x+L+1, y-L:y+L+1] = 0
            else:
                new_img[x,y] = 255
        return new_img.astype(np.uint8)

    def _find_contour(self, img1, img2):
        r,c = np.shape(img1)
        new_img = np.ones((r,c))*255
        for i in range(r):
            for j in range(c):
                if img1[i,j]!=img2[i,j]:
                    new_img[i,j]=0
        return new_img.astype(np.uint8)

    def _series_merging_map(self, map_set, feature_option="sift"):
        """
        Please finish the state transformation before using this function.\n
        Input:\n
        - map_list: List, stores the map series.
        """
        print(" --- Start ---")
        # Transform state into 3 specified values

        
        map_list = []
        map_list_ogm = []
        for i in range(len(map_set)):
            current_map = cv2.cvtColor(map_set[i], cv2.COLOR_RGB2GRAY)
            current_map_binary = MF._transform_state(input_map=current_map)
            # current_map_binary = MF._find_contour(img1=MF._dilate_image(current_map_binary, L=2), img2=MF._dilate_image(current_map_binary, L=1))
            current_map_binary = MF._dilate_image(current_map_binary, L=1)
            map_list.append(current_map_binary)
            
        for i in range(len(map_set)):
            current_map = cv2.cvtColor(map_set[i], cv2.COLOR_RGB2GRAY)
            current_map_ogm = MF._transform_state(input_map=current_map, gray=128)
            map_list_ogm.append(current_map_ogm)

        map_ref = map_list[0]
        map_ref_ogm = map_list_ogm[0]
        for i in range(len(map_list)-1):
            map_align = map_list[i+1]
            map_align_ogm = map_list_ogm[i+1]

            
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

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptor_1, descriptor_2, k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)
            
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
            
            pts1 = np.array(pts_1)
            pts2 = np.array(pts_2)

            relation, value = RMM._ransac_find_rotation_translation(pts_set_1=pts2, pts_set_2=pts1, sigma=1, max_iter=5000)
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
            
            map_merged = MF._merging_map(dx=dx, dy=dy, dtheta=dyaw, map1=map_ref_ogm, map2=map_align_ogm)
            map_ref = map_merged.astype(np.uint8)
            map_ref = MF._modify_map_size(merged_map=map_ref)

        return map_ref


    def _modify_map_size(self, merged_map):
        """
        After merging maps, size of the merged map is very large due to the extension.\n
        This function is designed to reduce parts of extension areas.\n
        """
        pos_x_white, pos_y_white = np.where(merged_map == 255)
        pos_x_black, pos_y_black = np.where(merged_map == 0)

        pos_x_M = np.amax(np.hstack((pos_x_black, pos_x_white)))
        pos_x_m = np.amin(np.hstack((pos_x_black, pos_x_white)))
        pos_y_M = np.amax(np.hstack((pos_y_black, pos_y_white)))
        pos_y_m = np.amin(np.hstack((pos_y_black, pos_y_white)))

        reduced_map = merged_map[pos_x_m-5:pos_x_M+5, pos_y_m-5:pos_y_M+5]

        return reduced_map

    def _plot_map(self, map_list, map_merged):
                
        # L = len(map_list)
        # plt.figure()
        # for i in range(L):
        #     plt.subplot(L, 1, i+1), plt.imshow(map_list[i], cmap='gray')
        # plt.suptitle("Original Maps")

        # plt.figure()
        # plt.title("Merging Result")
        # plt.imshow(map_merged, cmap='gray')
        # plt.show()

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


    def _extract_layer(self, map_input=None, layer_type="white"):
        
        if layer_type == "white":
            target_data = 225
            weight = 1
            target_data_pos = np.where(map_input >= target_data)

        elif layer_type == "black":
            target_data = 30
            weight = 5
            target_data_pos = np.where(map_input <= target_data)

        else:
            print("Argument Error: layer_type should be white or black.")
            return 

        r, c = np.shape(map_input)
        map_layer = np.zeros((r,c))
        map_layer[target_data_pos] = weight

        return map_layer


if __name__ == "__main__":
    

    MF = MAP_Function()
    RMM = RANSAC_Map_Merging()

    map1 = cv2.imread('Gazebo_test_map//tesst1.png')
    # map2 = cv2.imread('Gazebo_test_map//tesst1.png')

    # map1 = cv2.imread('Gazebo_test_map//test0223_1.png')
    map2 = cv2.imread('Gazebo_test_map//test0223_1.png')
    map_list = [map1, map2]


    map_merged = MF._series_merging_map(map_set=map_list, feature_option="sift")
    # map_reduced = MF._modify_map_size(merged_map=map_merged)

    MF._plot_map(map_list=map_list, map_merged=map_merged)




