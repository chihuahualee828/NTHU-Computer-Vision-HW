import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def clusterKeypoints(keypoints, k):
    points = [keypoint.pt for keypoint in keypoints]
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(points)
    cluster_ids = kmeans.labels_

    for keypoint, cluster_id in zip(keypoints, cluster_ids):
        keypoint.class_id = int(cluster_id)

def sift(image1, image2, k):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    # keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    
    segments = np.array_split(image2, k)

    keypoints2, descriptors2 = [], []
    for i in range(k):
        start_height = i * segments[i].shape[0]
        end_height = (i+1) * segments[i].shape[0]

        black_mask = np.zeros_like(image2)
        black_mask[start_height:end_height, :] = segments[i]

        keypoints_segment, descriptor_segment = sift.detectAndCompute(black_mask, None)
        # assign class_id so annotate the group
        for each in keypoints_segment:
            each.class_id = i

        # combine all segment's keypoints and desciprtors
        keypoints2.extend(keypoints_segment)
        descriptors2.append(descriptor_segment)
    
    keypoints2, descriptors2 = tuple(keypoints2), np.vstack(descriptors2)
    # keypoints2, descriptors2  = keypoints2_1+keypoints2_2+keypoints2_3, np.vstack([descriptors2_1,descriptors2_2,descriptors2_3])
    print(descriptors1.shape) # sift descriptor, 128 dimen vector
    print(descriptors2.shape)
    
    # # cluster keypoints into k(number of objects) groups
    # clusterKeypoints(keypoints1, k)
    # clusterKeypoints(keypoints2, k)
    # keypoints1_0 = [keypoint for keypoint in keypoints2 if keypoint.class_id == 2]

    # sift_image = cv2.drawKeypoints(image2, keypoints2 , image2)
    # plt.imshow(sift_image, cmap='gray')
    # plt.show()
    
    # sort by keypoints and descriptors of image1 by keypoint response
    keypoints1, descriptors1 = zip(*sorted(zip(keypoints1, descriptors1), key=lambda pair: pair[0].response, reverse=True))

    print("matching...")
    ratio = 0.7
    matches = []
    #  for each 128D vector in desc1, find a match in desc2
    for i, desc1 in enumerate(descriptors1): 
        # i used as queryIdx(first image descriptor vector index)
        distances = []
        for j, desc2 in enumerate(descriptors2):
            # j used as trainIdx(second image descriptor vector index)
            # Calculate distance between desc1 and desc2, L2 norm
            distance = np.linalg.norm(desc1 - desc2)
            distances.append((j, distance))
        distances.sort(key = lambda x:x[1]) # sort by distance
        best_match, second_best_match = distances[0], distances[1]
        
        # pass the ratio test to get matched
        if best_match[1] < ratio * second_best_match[1]:
            matches.append((i, best_match[0])) # image index pair

    matches = [(cv2.DMatch(i, j, 0)) for i,j in matches] # convert to DMatch object, imgIdx = 0
    
    # find top 20 matches for each object
    matches_obj = []
    for object in range(k):
        matches_obj.extend([match for match in matches if keypoints2[match.trainIdx].class_id == object][:20])

    # matched_image = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches[:60], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches_obj, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return matched_image

def draw_and_save(matched_image, save_path):
    # Plot matching result on the images
    plt.figure('(c) sift')
    plt.imshow(matched_image, cmap='gray')
    plt.imsave(save_path, matched_image, cmap='gray')
    plt.draw()
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()



if __name__=='__main__':
    # read image img subfolder
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "img")
    # read image as grayscale (if usage of cv2.IMREAD_GRAYSCALE is not allowed, use NTSC formula to convert RGB into grayscale value)
    #image1 = cv2.imread(os.path.join(img_path,"hw1-3-1.jpg"), cv2.IMREAD_GRAYSCALE)
    #image1 = cv2.imread(os.path.join(img_path,"hw1-3-2.jpg"), cv2.IMREAD_GRAYSCALE)
    # use this when path contains chinese char
    image1 = cv2.imdecode(np.fromfile(os.path.join(img_path,"hw1-3-1.jpg"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imdecode(np.fromfile(os.path.join(img_path,"hw1-3-2.jpg"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    
    matched_image = sift(image1, image2, 3)
    draw_and_save(matched_image, os.path.join(img_path,"sift_match.jpg"))

    scaled_image = cv2.resize(image1, (2 * image1.shape[1], 2 * image1.shape[0]))
    matched_image = sift(scaled_image, image2, 3)
    draw_and_save(matched_image, os.path.join(img_path,"sift_match_2xscaled.jpg"))