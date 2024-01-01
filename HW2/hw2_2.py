import cv2
import numpy as np
import copy
import os
import matplotlib.pyplot as plt

# mouse callback function
def mouse_callback(event, x, y, flags, param):
    
    global corner_list
    if event == cv2.EVENT_LBUTTONDOWN:  
        if(len(corner_list)<4):
            corner_list.append((x,y))

# a) Implement a function that estimates the homography matrix H that maps a set of interest points to a new set of interest points. Describe your implementation. 
def Find_Homography(world,camera):
    '''
    given corresponding point and return the homagraphic matrix 
    '''

    A = []
    for p1,p2 in zip(world, camera):
        row1 = [p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]]
        row2 = [0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]]
        A.append(row1)
        A.append(row2)
    
    A = np.array(A)
    
    U, S, Vt = np.linalg.svd(A)
    # print(U.shape, S.shape, Vt.shape)

    # row of V that corresponds to the smallest singular value of S (that is eigenvector of ATA with smallest eigenvalue)
    H = Vt.T[:,-1].reshape(3, 3)

    print("homography matrix:")
    print(H)
    print("least square constraint for SVD:")
    print(f"norm(H): {np.linalg.norm(H)}")
    print(f"norm(AH)^2(minimized to 0) : {np.linalg.norm(np.dot(A,H.flatten()))**2}")

    return H

def create_region(corner_points):
    '''
    given 4 corner points, create a box region that contains them(meshgrid)
    '''
    
    min_x, min_y = corner_points.min(axis=0)
    max_x, max_y = corner_points.max(axis=0)

    # X,Y are all the point coordinates within the red box region 
    X, Y = np.meshgrid(np.arange(min_x, max_x+1), np.arange(min_y, max_y+1))
    return X, Y

def bilinear(img, pt):
    x1 = int(pt[0])
    x2 = int(pt[0]) + 1
    y1 = int(pt[1])
    y2 = int(pt[1]) + 1

    a = pt[0] - x1
    b= pt[1] - y1

    new_pt = (1-a)*(1-b)*img[y1, x1] + a*(1-b)*img[y1, x2] + b*(1-a)*img[y2, x1] + a*b*img[y2, x2]
    
    return new_pt

def intersection(line1, line2):
    '''
    Finds the intersection of two lines, i.e. vanishing line
    '''
    p1, q1 = line1
    p2, q2 = line2

    p1 = np.hstack((p1,1))
    q1 = np.hstack((q1,1))
    p2 = np.hstack((p2,1))
    q2 = np.hstack((q2,1))

    intersection = np.cross(np.cross(p1, q1), np.cross(p2, q2))

    intersection = intersection[:-1]/intersection[-1]
    intersection = intersection.astype(int)
    return intersection

if __name__=="__main__":
    
    img_src = cv2.imread("assets/post.png") 
    src_H,src_W,channels=img_src.shape

    corners_src = np.array([[0, 0],[src_W-1, 0],[src_W-1, src_H-1],[0, src_H-1]])
    # print(H,W)
    file_path="./output"
    img_tar = cv2.imread("assets/display.jpg") 
    
    cv2.namedWindow("Interative window")
    cv2.setMouseCallback("Interative window", mouse_callback)
    cv2.setMouseCallback("Interative window", mouse_callback)
    
    corner_list=[]
    while True:
        fig=img_tar.copy()
        key = cv2.waitKey(1) & 0xFF
        
        
        if(len(corner_list)==4):

            corner_list = np.array(corner_list)

            # order corner points from left top, right top, right down, left down
            # sort by y ascending first (y1 then y2)
            corner_list = corner_list[np.argsort(corner_list[:,-1])]
            # for first 2, sort by x ascending (x1,y1 then x2,y1)
            corner_list[:2] = corner_list[np.argsort(corner_list[:2, 0])]
            # for last 2, sort by x descending (x2,y2 then x1,y2)
            corner_list[2:] = corner_list[2:][np.argsort(corner_list[2:, 0])[::-1]]

            print(f"selected corner points (order: left top -> right top -> right down -> left down):")
            print(corner_list)
            

            # H = Find_Homography(corners_src, corner_list)

            # X, Y = create_region(corners_src)
            # points = np.column_stack((X.flatten(), Y.flatten()))
            # homo_points = np.column_stack([points, np.ones(len(points))])
            # print(homo_points)
            
            # output_points = np.dot(H, homo_points.T).T
            # output_points = np.array([point[:-1]/point[-1] for point in output_points])


            
            # inverse homography mapping
            H = Find_Homography(corner_list, corners_src)

            # create all coordinates within red box region on screen image
            X, Y = create_region(corner_list)
            points = np.column_stack((X.flatten(), Y.flatten()))
            
            # change to homogenous
            homo_points = np.column_stack([points, np.ones(len(points))])
            # print(homo_points.shape)

            # homography matrix maps screen image pixel to CV image(since inverse mapping)
            output_points = np.dot(H, homo_points.T).T
            
            # change back from homogeneous to 2D
            output_points = output_points[:,:2]/output_points[:,[-1]]
            print("mapped points of display pixels on CV image: ")
            print(output_points)

            # bi-linear interpolation
            for point_src, point_target in zip(points, output_points):
                if point_target[0]<img_src.shape[1]-1 and point_target[1]<img_src.shape[0]-1 and point_target[1]>=0 and point_target[0]>=0:
                    # image[y, x]
                    fig[point_src[1], point_src[0]] = bilinear(img_src,point_target)

            # draw four corresponding straight lines(green lines)
            fig = cv2.line(fig, corner_list[0], corner_list[1], (0, 255, 0), 2)
            fig = cv2.line(fig, corner_list[1], corner_list[2], (0, 255, 0), 2) 
            fig = cv2.line(fig, corner_list[2], corner_list[3], (0, 255, 0), 2) 
            fig = cv2.line(fig, corner_list[3], corner_list[0], (0, 255, 0), 2) 
            
            # compute vanishing point using monitor's top and bottom parallel lines
            intersect = intersection((corner_list[0], corner_list[1]), (corner_list[3], corner_list[2]))
            print(f"vanishing point: {intersect}")
            cv2.circle(fig,intersect, 5,(0,255,0),3)
            

            # drawing red box region
            min_x, min_y = corner_list.min(axis=0)
            max_x, max_y = corner_list.max(axis=0)
            fig = cv2.line(fig, (min_x, min_y), (max_x, min_y), (0, 0, 255), 2)
            fig = cv2.line(fig, (max_x, min_y), (max_x, max_y), (0, 0, 255), 2) 
            fig = cv2.line(fig, (max_x, max_y), (min_x, max_y), (0, 0, 255), 2) 
            fig = cv2.line(fig, (min_x, max_y), (min_x, min_y), (0, 0, 255), 2) 

            break

            
        # quit 
        if key == ord("q"):
            break
    
        # reset the corner_list
        if key == ord("r"):
            corner_list=[]
        # show the corner list
        if key == ord("p"):
            print(corner_list)
        cv2.imshow("Interative window", fig)

    cv2.imshow("Interative window", fig)
    cv2.imwrite(os.path.join(file_path,"homography.png"),fig)
    print("Press Enter to close the window:")
    while True:
        k = cv2.waitKey(0) & 0xFF
        break
    cv2.destroyAllWindows()