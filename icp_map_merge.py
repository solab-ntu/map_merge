# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from sklearn.neighbors import NearestNeighbors



class ScanMatching():

    def __init__(self):
        pass


    def euclidean_distance(self, point1, point2):
        """
        Euclidean distance between two points.
        :param point1: the first point as a tuple (a_1, a_2, ..., a_n)
        :param point2: the second point as a tuple (b_1, b_2, ..., b_n)
        :return: the Euclidean distance
        """
        a = np.array(point1)
        b = np.array(point2)

        return np.linalg.norm(a - b, ord=2)


    def point_based_matching(self, point_pairs):
        """
        This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
        by F. Lu and E. Milios.

        :param point_pairs: the matched point pairs [((x1, y1), (x1', y1')), ..., ((xi, yi), (xi', yi')), ...]
        :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points
        """

        x_mean = 0
        y_mean = 0
        xp_mean = 0
        yp_mean = 0
        n = len(point_pairs)

        if n == 0:
            return None, None, None

        for pair in point_pairs:

            (x, y), (xp, yp) = pair

            x_mean += x
            y_mean += y
            xp_mean += xp
            yp_mean += yp

        x_mean /= n
        y_mean /= n
        xp_mean /= n
        yp_mean /= n

        s_x_xp = 0
        s_y_yp = 0
        s_x_yp = 0
        s_y_xp = 0
        for pair in point_pairs:

            (x, y), (xp, yp) = pair

            s_x_xp += (x - x_mean)*(xp - xp_mean)
            s_y_yp += (y - y_mean)*(yp - yp_mean)
            s_x_yp += (x - x_mean)*(yp - yp_mean)
            s_y_xp += (y - y_mean)*(xp - xp_mean)

        rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
        translation_x = xp_mean - (x_mean*math.cos(rot_angle) - y_mean*math.sin(rot_angle))
        translation_y = yp_mean - (x_mean*math.sin(rot_angle) + y_mean*math.cos(rot_angle))

        return rot_angle, translation_x, translation_y


    def icp(self, reference_points, points, max_iterations=2000, distance_threshold=1000, convergence_translation_threshold=1,
            convergence_rotation_threshold=1e-3, point_pairs_threshold=10, verbose=False):
        """
        An implementation of the Iterative Closest Point algorithm that matches a set of M 2D points to another set
        of N 2D (reference) points.

        :param reference_points: the reference point set as a numpy array (N x 2) \n
        :param points: the point that should be aligned to the reference_points set as a numpy array (M x 2) \n
        :param max_iterations: the maximum number of iteration to be executed \n
        :param distance_threshold: the distance threshold between two points in order to be considered as a pair \n
        :param convergence_translation_threshold: the threshold for the translation parameters (x and y) for the
                                                transformation to be considered converged \n
        :param convergence_rotation_threshold: the threshold for the rotation angle (in rad) for the transformation
                                                to be considered converged \n
        :param point_pairs_threshold: the minimum number of point pairs the should exist \n
        :param verbose: whether to print informative messages about the process (default: False) \n
        :return: the transformation history as a list of numpy arrays containing the rotation (R) and translation (T)
                transformation in each iteration in the format [R | T] and the aligned points as a numpy array M x 2 \n
        """

        transformation_history = []

        x, y, yaw = 0, 0, 0

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)
        
        for iter_num in range(max_iterations):
            if verbose:
                print('------ iteration', iter_num, '------')

            closest_point_pairs = []  # list of point correspondences for closest point rule

            distances, indices = nbrs.kneighbors(points)
            for nn_index in range(len(distances)):
                if distances[nn_index][0] < distance_threshold:
                    closest_point_pairs.append((points[nn_index], reference_points[indices[nn_index][0]]))

            # if only few point pairs, stop process
            if verbose:
                print('number of pairs found:', len(closest_point_pairs))
            if len(closest_point_pairs) < point_pairs_threshold:
                if verbose:
                    print('No better solution can be found (very few point pairs)!')
                break

            # compute translation and rotation using point correspondences
            closest_rot_angle, closest_translation_x, closest_translation_y = self.point_based_matching(closest_point_pairs)
            if closest_rot_angle is not None:
                if verbose:
                    print('Rotation:', math.degrees(closest_rot_angle), 'degrees')
                    print('Translation:', closest_translation_x, closest_translation_y)
            if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
                if verbose:
                    print('No better solution can be found!')
                break

            # transform 'points' (using the calculated rotation and translation)
            c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
            rot = np.array([[c, -s],
                            [s, c]])
            aligned_points = np.dot(points, rot.T)
            aligned_points[:, 0] += closest_translation_x
            aligned_points[:, 1] += closest_translation_y

            # update 'points' for the next iteration
            points = aligned_points

            # update transformation history
            transformation_history.append(np.vstack((np.hstack( (rot, np.array([[closest_translation_x], [closest_translation_y]]) )), np.array([0,0,1]))))

            yaw += closest_rot_angle
            x += closest_translation_x
            y += closest_translation_y

            # check convergence
            if (abs(closest_rot_angle) < convergence_rotation_threshold) \
                    and (abs(closest_translation_x) < convergence_translation_threshold) \
                    and (abs(closest_translation_y) < convergence_translation_threshold):
                if verbose:
                    print('Converged!')
                break
        print("No. %i"%iter_num)
        return transformation_history, points, [x, y, yaw]




def transform_state(input_map):
    """
    Probabilities in OGMs will be transformed into 3 values: 0, 128, 255. \n
    Occupied Space:   0\n
    Unknown  Space: 128\n
    Free     Space: 255
    """

    r, c = np.shape(input_map)
    for i in range(r):
        for j in range(c):

            if input_map[i, j]  > 230:  # White >> Free Space
                input_map[i, j] = 255
            
            elif input_map[i, j] < 100: # Black >> Occupied Space
                input_map[i, j] = 0
            
            else:                       # Gray  >> Unknown Space
                input_map[i, j] = 128

    return input_map

def entropy_filter(prob1, prob2):
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

def merging_map(dx, dy, dtheta, map1, map2):
    
    r1, c1 = np.shape(map1)
    r2, c2 = np.shape(map2)
    extension = max(r2, c2)
    map_temp = np.zeros(( r1 + extension*2 , c1 + extension*2 ))
    map_temp[extension:extension+r1 , extension:extension+c1] = map1

    # 下面這步是為了先將Map1 和Map2的原點貼齊，否則Map2一開始會在擴充地圖的左上角。
    dx = dx + extension
    dy = dy + extension
    for row in range(r2):
        for col in range(c2):

            new_x = int( math.cos(dtheta)*row + math.sin(dtheta)*col) + dx
            new_y = int(-math.sin(dtheta)*row + math.cos(dtheta)*col) + dy

            if new_x >= extension and new_x <= extension + r1 and new_y >= extension and new_y <= extension + c1:
                # Overlap area, use Entropy filter to determine whether adopting mixed probability or not.
                prob1 = (map_temp[new_x, new_y]+1)/257  # -> equals to map1 prob
                prob2 = (map2[row, col]+1)/257          # -> map2 prob
                
                prob = entropy_filter(prob1, prob2)
                map_temp[new_x, new_y] = prob*257-1

            else:
                map_temp[new_x, new_y] = map2[row, col]

    return map_temp




if __name__ == "__main__":
    
    # Read map image:
    img1 = cv2.imread('map4.1.png')
    map1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    map1 = transform_state(map1)

    img2 = cv2.imread('map4.2.1.png')
    map2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    map2 = transform_state(map2)

    

    siftDetector = cv2.xfeatures2d.SIFT_create()
    key_points_1, descriptor_1 = siftDetector.detectAndCompute(map1, None)
    key_points_2, descriptor_2 = siftDetector.detectAndCompute(map2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor_1, descriptor_2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.01*n.distance:
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
    
    SM = ScanMatching()
    _, pts, [x, y, yaw] = SM.icp(   reference_points=pts2, 
                                    points=pts1, 
                                    max_iterations=5000, 
                                    distance_threshold=5000, 
                                    convergence_translation_threshold=100, 
                                    convergence_rotation_threshold=1e-1, 
                                    point_pairs_threshold=10, 
                                    verbose=False)
    # print(x, y, yaw)

    # map_merged = merging_map(dx=x, dy=y, dtheta=yaw, map1=map1, map2=map2)
    
    
    
    plt.figure()
    p1 = plt.scatter(pts1[:,0], pts1[:,1], c='black', s=10)
    p2 = plt.scatter(pts2[:,0], pts2[:,1], c='blue', s=5)
    p3 = plt.scatter(pts[:,0], pts[:,1], c='red', s=1)
    plt.legend([p1, p2, p3], ['map_1', 'map_2', 'alignment'], loc='lower right', scatterpoints=1)
    
    # plt.figure()
    # plt.subplot(2,1,1), plt.imshow(map1, cmap='gray'), plt.title("Original Map")
    # plt.subplot(2,1,2), plt.imshow(map2, cmap='gray')
    
    # plt.figure()
    # plt.title("Merging Result")
    # plt.imshow(map_merged, cmap='gray')
    plt.show()
