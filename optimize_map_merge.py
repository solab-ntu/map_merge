import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from scipy.optimize import basinhopping
from scipy import sparse
import time


def objective_function(x, map1_black_layer, map1_white_layer, map2_black_layer, map2_white_layer):
    """
    Variable: x {dx, dy, dtheta} (float). \n
    Input Map Layers: (ndarray).
    """
    dx = int(x[0]*1000)
    dy = int(x[1]*1000)
    dtheta = x[2]*360
    
    agr = 0.
    dis = 0.

    # Q: bound?
    map2_black_layer = rotate(image=map2_black_layer, angle=dtheta, center=(2*bound, 2*bound))
    map2_white_layer = rotate(image=map2_white_layer, angle=dtheta, center=(2*bound, 2*bound))

    map2_black_layer = translate(image=map2_black_layer, x=dx, y=dy)
    map2_white_layer = translate(image=map2_white_layer, x=dx, y=dy)

    # map1_black_layer = sparse.csr_matrix(map1_black_layer)
    # map2_black_layer = sparse.csr_matrix(map2_black_layer)
    # map1_white_layer = sparse.csr_matrix(map1_white_layer)
    # map2_white_layer = sparse.csr_matrix(map2_white_layer)


    dis = (map1_black_layer * map2_white_layer).sum() + (map1_white_layer * map2_black_layer).sum()
    agr = (map1_black_layer * map2_black_layer).sum() + (map1_white_layer * map2_white_layer).sum()

    print("dis: %f and agr: %f"%(dis,agr))
    # extension = max(r2, c2)
    # map_temp = np.zeros(( r1 + extension*2 , c1 + extension*2 ))
    # map_temp[extension:extension+r1 , extension:extension+c1] = map1

    # for row in range(r2):
    #     for col in range(c2):
            
    #         state = map2[row, col]
    #         if state == 128: # unknown state
    #             pass
    #         else:
    #             new_x = int(+math.cos(dtheta)*row + math.sin(dtheta)*col + dx)
    #             new_y = int(-math.sin(dtheta)*row + math.cos(dtheta)*col + dy)

    #             if new_x >= extension and new_x <= extension + r1 and new_y >= extension and new_y <= extension + c1:
    #                 if map_temp[new_x, new_y] != 128:
    #                     if state == map_temp[new_x, new_y]:
    #                         agr += 1
    #                     else:
    #                         dis += 1

    if agr == 0 and dis ==0:
        return 0
    else:
        return  -agr/(agr+dis)

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

def extract_layer(map_input=None, bound=None, layer_type="white"):
    

    if bound == None:
        return print("Argument Error: Need a Parameter(bound) for the layer map.")

    if layer_type == "white":
        target_data = 255
        weight = 1

    elif layer_type == "black":
        target_data = 0
        weight = 5
    else:
        return print("Argument Error: layer_type should be white or black.")

    r, c = np.shape(map_input)
    map_tmp = np.zeros((5*bound, 5*bound))
    map_layer = np.zeros((r,c))

    target_data_pos = np.where(map_input==target_data)
    map_layer[target_data_pos] = 1*weight

    map_tmp[2*bound:2*bound+r, 2*bound:2*bound+c] = map_layer

    return map_tmp

def translate(image, x, y):

    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return shifted

def rotate(image, angle, center=None, scale=1.0):

    (h, w) = image.shape[:2]
 
    # If the center of the rotation is not defined, use the center of the image.
    if center is None:
        center = (w / 2, h / 2)
 
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
 
    return rotated

if __name__ == "__main__":
    
    # Read map image:
    img1 = cv2.imread('map4.1.png')
    map1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    map1 = transform_state(map1)

    img2 = cv2.imread('map4.1.png')
    map2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    map2 = transform_state(map2)


    # t0 = time.time()

    # Objective Fcn test:
    
    r1, c1 = np.shape(map1)
    r2, c2 = np.shape(map2)

    bound = max(r1, c1, r2, c2)

    map1_white_layer = extract_layer(map_input=map1, bound=bound, layer_type="white")
    map1_black_layer = extract_layer(map_input=map1, bound=bound, layer_type="black")
    map2_white_layer = extract_layer(map_input=map2, bound=bound, layer_type="white")
    map2_black_layer = extract_layer(map_input=map2, bound=bound, layer_type="black")

    # print(map1_white_layer.sum(),map1_black_layer.sum())

    # pos = [0,0,0]
    # for i in range(2):
    #     prob = objective_function(  x=pos,    
    #                                 map1_black_layer=map1_black_layer,
    #                                 map1_white_layer=map1_white_layer,
    #                                 map2_black_layer=map2_black_layer,
    #                                 map2_white_layer=map2_white_layer)
    #     print(prob)
    #     pos = [i*0.1,i*0.1,0.045]
    # print(time.time() - t0)
    
    if 1:
        x0 = [0, 0, 0]
        # Fixed parameters will be called or passed into the obective function by this "args" setup.
        minimizer_kwargs = {"method":"L-BFGS-B", "args":(map1_black_layer, map1_white_layer, map2_black_layer, map2_white_layer)}#, "jac":True}
        ret = basinhopping(objective_function, x0,  minimizer_kwargs=minimizer_kwargs, niter=50)
        print(ret)
    if 0:
        # r1, c1 = np.shape(map1)
        # r2, c2 = np.shape(map2)

        # extension = max(r2, c2)
        # map_temp = np.zeros(( r1 + extension*2 , c1 + extension*2 ))
        # map_temp[extension:extension+r1 , extension:extension+c1] = map1
        # map_temp[extension:extension+r2 , extension:extension+c2] = map2

        map_merged = merging_map(1914, 1733, -0.4848*math.pi, map1, map2)
        plt.figure()
        plt.imshow(map_merged, cmap='gray')
        plt.figure()
        plt.subplot(1,2,1), plt.imshow(map1, cmap='gray')
        plt.subplot(1,2,2), plt.imshow(map2, cmap='gray')
        plt.show()


