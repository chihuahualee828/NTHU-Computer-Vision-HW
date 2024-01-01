import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from tqdm.auto import tqdm

def sift(image1, image2, ratio=0.7, threshold_d=None):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # sort by keypoints and descriptors of image1 by keypoint response
    # keypoints1, descriptors1 = zip(*sorted(zip(keypoints1, descriptors1), key=lambda pair: pair[0].response, reverse=True))

    print("matching...")
    matches = []
    #  for each 128D vector in desc1, find a match in desc2
    for i, desc1 in enumerate(tqdm(descriptors1)): 
        # i used as queryIdx(first image descriptor vector index)

        # j used as trainIdx(second image descriptor vector index)
        # Calculate distance between desc1 and desc2, L2 norm
        distances = [(j, np.linalg.norm(desc1 - desc2)) for j, desc2 in enumerate(descriptors2)]
        distances.sort(key = lambda x:x[1]) # sort by distance
        best_match, second_best_match = distances[0], distances[1]
        
        if not threshold_d:
        # pass the ratio test to get matched
            if best_match[1] < ratio * second_best_match[1]:
                matches.append((i, best_match[0], best_match[1])) # image index pair
        else:
            if best_match[1] < threshold_d:
                matches.append((i, best_match[0], best_match[1])) # image index pair

    matches.sort(key = lambda x:x[2])
    print(f"number of matches: {len(matches)}")
    matches = [(cv2.DMatch(i, j, 0)) for i,j,d in matches] # convert to DMatch object, imgIdx = 0
    
    matched_points = [(keypoints1[match.queryIdx].pt, keypoints2[match.trainIdx].pt) for match in matches]
    # matched_image = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches[:60], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return np.asarray(matched_points), matched_image

def compute_homography(matches):
    '''
    given corresponding point and return the homagraphic matrix 
    '''
    matches = list(matches)
    A = []
    for p1,p2 in matches:
        row1 = [p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]]
        row2 = [0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]]
        A.append(row1)
        A.append(row2)
    
    A = np.array(A)
    
    U, S, Vt = np.linalg.svd(A)
    # print(U.shape, S.shape, Vt.shape)

    # row of V that corresponds to the smallest singular value of S (that is eigenvector of ATA with smallest eigenvalue)
    H = Vt.T[:,-1].reshape(3, 3)
    return H

def ransac_homography(matches, iterations, threshold, k=4):
    
    def select_seed(matches, k):
        '''
        randomly select matched points 
        '''
        # k = 4 because homography requires 4
        index = random.sample(range(len(matches)), k)
        # print(f"random: {index}")
        points = [matches[i] for i in index]
        return np.array(points)
    def count_inliers(matches, H, threshold):
        '''count inliers for computed Homography'''
        inliers = 0
        num_points = len(matches)
        pts1 = np.hstack((matches[:, 0, :], np.ones((num_points, 1))))
        pts2 = matches[:, 1, :]
        pts2_estimate = np.zeros((num_points, 2))
        temp = np.dot(H, pts1.T).T
        pts2_estimate = temp[:, :-1] / temp[:, -1][:, None]
        # Compute error
        errors = np.linalg.norm(pts2 - pts2_estimate , axis=1) ** 2
        inliers = np.where(errors<threshold)[0]
        return inliers
    
    best_H = 0
    max_inliers = 0
    best_inliers = 0
    best_inliers_indices = 0
    for i in range(iterations):
        # print(f"iterations: {i}")
        random_matches = select_seed(matches, k)
        H = compute_homography(random_matches)
        inliers = count_inliers(matches, H, threshold)
        # print(f"number of inliers: {len(inliers)}")
        if len(inliers) >= max_inliers:
            max_inliers = len(inliers)
            best_inliers = matches[inliers]
            best_inliers_indices = inliers
            best_H = H
        # print("==============")

    print(f"best H: {best_H}")
    print(f"max number of inliers: {max_inliers}")
    print(f"inliers: {best_inliers_indices}")
    # recompute H using inliers
    H = compute_homography(best_inliers)
    return H



def points_match(img, keypoints1, keypoints2, color=(0, 0, 255), radius=5, mode=None):
    '''draw colored points and match them on image1 and image2'''
    color2 = (color[1], color[2], color[0])
    color3 = (color[2], color[1], color[0])
    for kp1,kp2 in zip(keypoints1, keypoints2):
        pt1 = tuple(map(int, kp1))
        cv2.circle(img, pt1, radius, color, -1)
        pt2 = tuple(map(int, kp2))
        cv2.circle(img, pt2, radius, color2, -1)
        if mode:
            cv2.arrowedLine(img, pt1, pt2, color3, 1)
        else:
            cv2.line(img, pt1, pt2, color3, 1)

def draw_box(img, corners1, corners2, color=(255, 0, 0)):
    cv2.line(img, corners1[0], corners1[1], color, 5)
    cv2.line(img, corners1[1], corners1[2], color, 5) 
    cv2.line(img, corners1[2], corners1[3], color, 5) 
    cv2.line(img, corners1[3], corners1[0], color, 5)

    cv2.line(img, corners2[0], corners2[1], color, 5)
    cv2.line(img, corners2[1], corners2[2], color, 5) 
    cv2.line(img, corners2[2], corners2[3], color, 5) 
    cv2.line(img, corners2[3], corners2[0], color, 5)

def draw_and_save(img, save_path):
    '''plot and save image'''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.imsave(save_path, img)
    plt.draw()
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()

def concat_image(img1, img2):
    max_height = max(img1.shape[0], img2.shape[0])
    # Create black images to pad the smaller image
    padded_image1 = np.zeros((max_height, img1.shape[1], 3), dtype=np.uint8)
    padded_image2 = np.zeros((max_height, img2.shape[1], 3), dtype=np.uint8)
    padded_image1[0:img1.shape[0], 0:img1.shape[1], :] = img1
    padded_image2[0:img2.shape[0], 0:img2.shape[1], :] = img2

    return np.concatenate((padded_image1, padded_image2), axis=1)

def estimate_points(pts1, H):
    pts1 = np.hstack((pts1, np.ones((len(pts1), 1))))
    temp = np.dot(H, pts1.T).T
    pts2_estimate = temp[:, :-1] / temp[:, -1][:, None]
    return pts2_estimate

if __name__=='__main__':
    # read image img subfolder
    img_path = os.path.dirname(os.path.abspath(__file__))

    image1 = cv2.imread("1-book1.jpg")
    image2 = cv2.imread("1-book2.jpg")
    image3 = cv2.imread("1-book3.jpg")
    input_image = cv2.imread("1-image.jpg")

    matches1, matched_image1 = sift(image1, input_image, 0.8)
    draw_and_save(matched_image1, os.path.join(img_path,"output/2(a)sift_match1.jpg"))

    H1 = ransac_homography(matches1, 1000, 50)
    pts1 = matches1[:, 0, :]
    pts2_estimate = estimate_points(pts1, H1)
    pts2_original = [(pt[0], pt[1]) for pt in matches1[:, 1, :]]
    corner_points = np.array([[97,219],[996,224],[985,1329],[121,1350]])
    corner_estimate = estimate_points(corner_points, H1)

    matched_image1 = concat_image(image1, input_image)
    points_match(matched_image1, pts1, tuple(map(lambda p: (int(p[0] + image1.shape[1]), int(p[1])), pts2_estimate)))
    draw_box(matched_image1, corner_points, tuple(map(lambda p: (int(p[0] + image1.shape[1]), int(p[1])), corner_estimate)))
    draw_and_save(matched_image1, os.path.join(img_path,"output/2(b)sift_match1_ransac.jpg"))

    dv_image1 = input_image.copy()
    points_match(dv_image1, pts2_original, pts2_estimate, mode=1)
    draw_and_save(dv_image1, os.path.join(img_path,"output/2(b)dv1.jpg"))

    # ---------------------------------------------------------------
    matches2, matched_image2 = sift(image2, input_image, 0.75)
    draw_and_save(matched_image2, os.path.join(img_path,"output/2(a)sift_match2.jpg"))
    # H2, mask = cv2.findHomography(matches2[:, 0, :], matches2[:, 1, :], cv2.RANSAC, 5.0)

    H2 = ransac_homography(matches2, 1000, 50)
    pts1 = matches2[:, 0, :]
    pts2_estimate = estimate_points(pts1, H2)
    pts2_original = [(pt[0], pt[1]) for pt in matches2[:, 1, :]]
    corner_points = np.array([[67,122],[1042,112],[1042,1342],[78,1349]])
    corner_estimate = estimate_points(corner_points, H2)

    matched_image2 = concat_image(image2, input_image)
    points_match(matched_image2, pts1, tuple(map(lambda p: (int(p[0] + image2.shape[1]), int(p[1])), pts2_estimate)))
    draw_box(matched_image2, corner_points, tuple(map(lambda p: (int(p[0] + image2.shape[1]), int(p[1])), corner_estimate)))
    draw_and_save(matched_image2, os.path.join(img_path,"output/2(b)sift_match2_ransac.jpg"))

    dv_image2 = input_image.copy()
    points_match(dv_image2, pts2_original, pts2_estimate, mode=1)
    draw_and_save(dv_image2, os.path.join(img_path,"output/2(b)dv2.jpg"))

    # ---------------------------------------------------------------
    matches3, matched_image3 = sift(image3, input_image, threshold_d=110)
    draw_and_save(matched_image3, os.path.join(img_path,"output/2(a)sift_match3.jpg"))

    H3 = ransac_homography(matches3, 1000, 50)
    pts1 = matches3[:, 0, :]
    pts2_estimate = estimate_points(pts1, H3)
    pts2_original = [(pt[0], pt[1]) for pt in matches3[:, 1, :]]
    corner_points = np.array([[121,191],[987,181],[979,1391],[132,1400]])
    corner_estimate = estimate_points(corner_points, H3)

    matched_image3 = concat_image(image3, input_image)
    points_match(matched_image3, pts1, tuple(map(lambda p: (int(p[0] + image3.shape[1]), int(p[1])), pts2_estimate)))
    draw_box(matched_image3, corner_points, tuple(map(lambda p: (int(p[0] + image3.shape[1]), int(p[1])), corner_estimate)))
    draw_and_save(matched_image3, os.path.join(img_path,"output/2(b)sift_match3_ransac.jpg"))

    dv_image3 = input_image.copy()
    points_match(dv_image3, pts2_original, pts2_estimate, mode=1)
    draw_and_save(dv_image3, os.path.join(img_path,"output/2(b)dv3.jpg"))