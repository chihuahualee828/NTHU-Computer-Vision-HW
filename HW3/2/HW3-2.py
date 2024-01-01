import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from collections import defaultdict
from tqdm.auto import tqdm

def k_means_init(points, k, random_seed=None):
    '''
    Randomly choose k data points as centroids
    '''
    # random.seed(random_seed)
    centroid_indices = random.sample(range(len(points)), k)
    centroids = points[centroid_indices]
    return centroids

def group(points, centroids):
    # distances of each pixel to each centroid
    distances = np.sum((points[:, None] - centroids) ** 2, axis=2)
    # index of closest centroid for each pixel
    closest_centroids = np.argmin(distances, axis=1)
    clusters = {centroid_index: [] for centroid_index in range(len(centroids))}
    colors = {centroid_index: centroids[centroid_index] for centroid_index in range(len(centroids))}
    # assign each pixel to the closest cluster
    for i, centroid_index in enumerate(closest_centroids):
        clusters[centroid_index].append(i)

    return clusters, colors 

def k_means(points, k, threshold=1, random_seed=42, mode=None):
    points = points.astype(np.float32)
    if not mode:
        centroids = k_means_init(points, k)
    else:
        centroids = k_means_pp_init(points, k)
    clusters, colors = group(points, centroids)

    if not mode:
        min_wcss = float("inf")
        best_clusters = 0
        best_colors = 0
        best_centroids = 0
        for i in tqdm(range(49)):
            centroids = k_means_init(points, k)
            clusters, colors = group(points, centroids)
            wcss = kmeans_objective(points, centroids, clusters)
            if wcss < min_wcss:
                min_wcss = wcss
                best_clusters = clusters
                best_colors = colors
                best_centroids = centroids
        centroids, clusters, colors = best_centroids, best_clusters, best_colors
        
    fixed_centroids = [np.array([])]*k
    converges = False
    while True:
        for i, (key, value) in enumerate(clusters.items()):
            new_centroid = np.mean(points[value], axis=0)
            if np.linalg.norm(centroids[i] - new_centroid) > threshold:
                centroids[i] = new_centroid
            else:
                if fixed_centroids[i].size == 0:
                    fixed_centroids[i] = centroids[i]
                # print(f"fixed_centroids: {len([each for each in fixed_centroids if each.size!=0])}")
                converges = all(each.size!=0 for each in fixed_centroids)
                if converges:
                    break
        if converges:
            break
        centroids = [each if each.size!=0 else centroids[i] for i, each in enumerate(fixed_centroids)]
        clusters, colors = group(points, centroids)
    # total = 0
    # for each in clusters:
    #     print(len(clusters[each]))
    #     total += len(clusters[each])
    # print(total)

    wcss = kmeans_objective(points, centroids, clusters)
    return clusters, colors, wcss

def segmentation(img, clusters, colors=None):
    img_segemented = img.copy()
    k = len(clusters)
    print(len(clusters))
    i = 0
    for key, value in clusters.items():
        if colors:
            color = colors[key]
        if isinstance(clusters, defaultdict):
            color = key
            key = clusters[key]
        img_segemented[np.unravel_index(value, img.shape[:2])] = color
        i+=1
    return img_segemented

def k_means_pp_init(data, k, random_seed=None):
    data = data.astype(np.float32)
    centroids = []
    c1_index = random.sample(range(len(data)), 1)
    c1 = data[c1_index]
    centroids.append(c1[0])
    for i in range(k-1):
        clusters, colors = group(data,centroids)

        max_distances = 0
        furthest_centroid = 0
        for i, (key, cluster) in enumerate(clusters.items()):
            distances_to_ci = np.linalg.norm((data[cluster][:, None] - centroids[i]), axis=2)
            temp = np.max(distances_to_ci)
            temp_index = np.argmax(distances_to_ci)
            if temp > max_distances:
                max_distances = temp
                furthest_centroid = cluster[temp_index]

        # furthest_centroid = np.argmax(np.max(distances, axis=1))
        # print(distances[furthest_centroid])
        # print(furthest_centroid)
        # print(data[furthest_centroid])
        centroids.append(data[furthest_centroid])
    # print(centroids)
    return centroids
    
def kmeans_objective(points, centroids, clusters):
    wcss = 0
    for key, cluster in clusters.items():
        centroid = centroids[key]
        points_of_ci = points[cluster]
        distances_to_ci = np.sum(np.linalg.norm(centroid - points_of_ci, axis=1)**2)
        wcss += distances_to_ci
    # print(f"Within-Cluster Sum of Square: {wcss}")

    return wcss

def mean_shift(img, bandwidth, c=1, spatial=False):
    # if spatial and bandwidth2 is None:
    #     raise ValueError("Bandwidth2 is required when add in spatial.")

    MIN_DISTANCE = 1
    if spatial:
        m, n, _ = img.shape
        # Create x and y coordinate grids
        x, y = np.meshgrid(np.arange(n), np.arange(m))
        img_copy = np.zeros((m, n, 5), dtype=np.uint8)
        img_copy[:, :, :3] = img
        img_copy[:, :, 3] = x
        img_copy[:, :, 4] = y
        img = img_copy
    img_faltten = img.reshape((-1,img.shape[-1]))
    img_faltten = img_faltten.astype(np.float32)

    shifting = np.array([True] * img_faltten.shape[0])
    while np.sum(shifting)>0:
        sum_distance = 0
        for i in tqdm(range(0, len(img_faltten))):
            if not shifting[i]:
                continue
            centroid = img_faltten[i]

            # if not spatial:
            distances = np.linalg.norm(img_faltten - centroid, axis=1)
            filtered_pixels = img_faltten[distances <= bandwidth]
            # distances_squared = np.sum((filtered_pixels - centroid)**2, axis=1)
            distances = distances[distances <= bandwidth]
            uniform_kernel = np.where(distances <= bandwidth, c, 0)
            # n_kernel = np.exp(-distances_squared / (2 * bandwidth**2))

            mx = np.sum(filtered_pixels * uniform_kernel[:, np.newaxis], axis=0) / np.sum(uniform_kernel)
            # else:
            #     distances = np.linalg.norm(img_faltten - centroid, axis=1)
            #     distances_s = np.linalg.norm(img_faltten[:, 3:] - centroid[3:], axis=1)
            #     distances_r = np.linalg.norm(img_faltten[:, :3] - centroid[:3], axis=1)
            #     filtered_pixels = img_faltten[(distances_r <= bandwidth) & (distances_s <= bandwidth2)]
            #     distances = distances[(distances_r <= bandwidth) & (distances_s <= bandwidth2)]
            #     uniform_kernel = np.where(distances, c, 0)
            #     mx = np.sum(filtered_pixels * uniform_kernel[:, np.newaxis], axis=0) / np.sum(uniform_kernel)
                
            dist = np.linalg.norm(mx - centroid)
            sum_distance += dist
            img_faltten[i] = mx
            if dist < MIN_DISTANCE:
                shifting[i] = False
        print(np.sum(shifting))
        
    img_faltten = img_faltten.astype(np.uint8)
    return img_faltten.reshape((img.shape[0],img.shape[1], img.shape[-1]))


def group_by_mode(arr):
    groups = defaultdict(list)
    arr_flatten = arr.reshape((-1, 3))
    for i in range(len(arr_flatten)):
        value = tuple(arr_flatten[i])
        groups[value].append(i)

    return groups

def draw_and_save(img, path):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.imsave(path, img)
    plt.draw()
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()

def draw_rgb(img, img2, path):
    img_flatten = img.reshape((-1,3))
    img_flatten2 = img2.reshape((-1,3))
    # Create a 3D scatter plot
    fig = plt.figure(dpi=300)
    axs = fig.subplots(1,3) 

    axs[0].set_title('Before:')
    axs[0].axis('off')
    axs[0] = fig.add_subplot(131, projection='3d')
    axs[0].scatter(img_flatten[:, 0], img_flatten[:, 1], img_flatten[:, 2], s=10, c=img_flatten / 255.0, marker='o', linewidth=0.5 )
    axs[0].set_xlabel('Red')
    axs[0].set_ylabel('Green')
    axs[0].set_zlabel('Blue')
    axs[0].view_init(elev=20, azim=30)

    axs[1].set_title('After(mode):')
    axs[1].axis('off')
    axs[1] = fig.add_subplot(132, projection='3d')   
    axs[1].scatter(img_flatten2[:, 0], img_flatten2[:, 1], img_flatten2[:, 2], s=10, c=img_flatten2 / 255.0, marker='o', linewidth=0.5 )
    axs[1].set_xlabel('Red')
    axs[1].set_ylabel('Green')
    axs[1].set_zlabel('Blue')
    axs[1].view_init(elev=20, azim=30)

    axs[2].set_title('After(segmentation):')
    axs[2].axis('off')
    axs[2] = fig.add_subplot(133, projection='3d')   
    axs[2].scatter(img_flatten[:, 0], img_flatten[:, 1], img_flatten[:, 2], s=10, c=img_flatten2 / 255.0, marker='o', linewidth=0.5 )
    axs[2].set_xlabel('Red')
    axs[2].set_ylabel('Green')
    axs[2].set_zlabel('Blue')
    axs[2].view_init(elev=20, azim=30)
    
    plt.savefig(path)

    # Show the plot
    plt.show(block=False)
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()

if __name__=='__main__':
    img_path = os.path.dirname(os.path.abspath(__file__))
    image1 = cv2.imread("2-image.jpg")
    image2 = cv2.imread("2-masterpiece.jpg")
    image1_flatten = image1.reshape((-1,3))
    image2_flatten = image2.reshape((-1,3))

    print("=======2(a)=======")
    # 2(a) 2-image
    clusters, colors, wcss = k_means(image1_flatten, 5)
    print(f"2-image kmeans best wcss(k=5): {wcss}")
    image1_segemented = segmentation(image1, clusters, colors)
    draw_and_save(image1_segemented, os.path.join(img_path,"output/2(a)2-image_kmeans_k5.jpg"))

    clusters, colors, wcss = k_means(image1_flatten, 10)
    print(f"2-image kmeans best wcss(k=10): {wcss}")
    image1_segemented = segmentation(image1, clusters, colors)
    draw_and_save(image1_segemented, os.path.join(img_path,"output/2(a)2-image_kmeans_k10.jpg"))

    clusters, colors, wcss = k_means(image1_flatten, 15)
    print(f"2-image kmeans best wcss(k=15): {wcss}")
    image1_segemented = segmentation(image1, clusters, colors)
    draw_and_save(image1_segemented, os.path.join(img_path,"output/2(a)2-image_kmeans_k15.jpg"))


    # 2(a) masterpiece
    clusters, colors, wcss = k_means(image2_flatten, 5)
    print(f"masterpiece kmeans best wcss(k=5): {wcss}")
    image2_segemented = segmentation(image2, clusters, colors)
    draw_and_save(image2_segemented, os.path.join(img_path,"output/2(a)masterpiece_kmeans_k5.jpg"))

    clusters, colors, wcss = k_means(image2_flatten, 10)
    print(f"masterpiece kmeans best wcss(k=10): {wcss}")
    image2_segemented = segmentation(image2, clusters, colors)
    draw_and_save(image2_segemented, os.path.join(img_path,"output/2(a)masterpiece_kmeans_k10.jpg"))

    clusters, colors, wcss = k_means(image2_flatten, 15)
    print(f"masterpiece kmeans best wcss(k=15): {wcss}")
    image2_segemented = segmentation(image2, clusters, colors)
    draw_and_save(image2_segemented, os.path.join(img_path,"output/2(a)masterpiece_kmeans_k15.jpg"))


    print("=======2(b)=======")
    # 2(b) 2-image
    clusters, colors, wcss = k_means(image1_flatten, 5, mode=1)
    print(f"2-image kmeans++ wcss(k=5): {wcss}")
    image1_segemented = segmentation(image1, clusters, colors)
    draw_and_save(image1_segemented, os.path.join(img_path,"output/2(b)2-image_kmeans++_k5.jpg"))

    clusters, colors, wcss = k_means(image1_flatten, 10, mode=1)
    print(f"2-image kmeans++ wcss(k=10): {wcss}")
    image1_segemented = segmentation(image1, clusters, colors)
    draw_and_save(image1_segemented, os.path.join(img_path,"output/2(b)2-image_kmeans++_k10.jpg"))

    clusters, colors, wcss = k_means(image1_flatten, 15, mode=1)
    print(f"2-image kmeans++ wcss(k=15): {wcss}")
    image1_segemented = segmentation(image1, clusters, colors)
    draw_and_save(image1_segemented, os.path.join(img_path,"output/2(b)2-image_kmeans++_k15.jpg"))


    # 2(b) masterpiece
    image2_flatten = image2.reshape((-1,3))
    clusters, colors, wcss = k_means(image2_flatten, 5, mode=1)
    print(f"masterpiece kmeans++ wcss(k=5): {wcss}")
    image2_segemented = segmentation(image2, clusters, colors)
    draw_and_save(image2_segemented, os.path.join(img_path,"output/2(b)masterpiece_kmeans++_k5.jpg"))  

    clusters, colors, wcss = k_means(image2_flatten, 10, mode=1)
    print(f"masterpiece kmeans++ wcss(k=10): {wcss}")
    image2_segemented = segmentation(image2, clusters, colors)
    draw_and_save(image2_segemented, os.path.join(img_path,"output/2(b)masterpiece_kmeans++_k10.jpg"))

    clusters, colors, wcss = k_means(image2_flatten, 15, mode=1)
    print(f"masterpiece kmeans++ wcss(k=15): {wcss}")
    image2_segemented = segmentation(image2, clusters, colors)
    draw_and_save(image2_segemented, os.path.join(img_path,"output/2(b)masterpiece_kmeans++_k15.jpg"))



    print("=======2(c)=======")
    # mean shift rgb on image1
    image1_resized = cv2.resize(image1, (image1.shape[1]//4, image1.shape[0]//4), interpolation=cv2.INTER_AREA)
    image1_segemented = mean_shift(image1_resized.copy(), 50, 1)
    print(f"number of clusters: {np.unique(image1_segemented.reshape((-1,3)), axis=0).shape[0]}")
    draw_and_save(image1_segemented, os.path.join(img_path,"output/2(c)2-image_msrgb_h50.jpg"))
    draw_rgb(cv2.cvtColor(image1_resized, cv2.COLOR_BGR2RGB), cv2.cvtColor(image1_segemented, cv2.COLOR_BGR2RGB), os.path.join(img_path,"output/2(c)2-image_rgb_cube.jpg"))

    image2_resized = cv2.resize(image2, (image2.shape[1]//6, image2.shape[0]//6), interpolation=cv2.INTER_AREA)
    image2_segemented = mean_shift(image2_resized.copy(), 50, 1)
    print(f"number of clusters: {np.unique(image2_segemented.reshape((-1,3)), axis=0).shape[0]}")
    draw_and_save(image2_segemented, os.path.join(img_path,"output/2(c)masterpiece_msrgb_h50.jpg"))
    draw_rgb(cv2.cvtColor(image2_resized, cv2.COLOR_BGR2RGB), cv2.cvtColor(image2_segemented, cv2.COLOR_BGR2RGB), os.path.join(img_path,"output/2(c)masterpiece_rgb_cube.jpg"))

    print("=======2(d)=======")
    # mean shift rgb+xy on image1
    image1_segemented = mean_shift(image1_resized.copy(), 50, 1, True)
    print(f"number of clusters: {np.unique(image1_segemented.reshape((-1,5)), axis=0).shape[0]}")
    draw_and_save(image1_segemented[:,:,:3], os.path.join(img_path,"output/2(d)2-image_msxy.jpg"))

    image2_segemented = mean_shift(image2_resized.copy(), 50, 1, True)
    print(f"number of clusters: {np.unique(image2_segemented.reshape((-1,5)), axis=0).shape[0]}")
    draw_and_save(image2_segemented[:,:,:3], os.path.join(img_path,"output/2(d)masterpiece_msxy.jpg"))
    

    print("=======2(e)=======")
    # mean shift rgb on image1 with different bandwidth
    image1_segemented = mean_shift(image1_resized.copy(), 5, 1)
    print(f"number of clusters: {np.unique(image1_segemented.reshape((-1,3)), axis=0).shape[0]}")
    draw_and_save(image1_segemented, os.path.join(img_path,"output/2(e)2-image_msrgb_h5.jpg"))

    image1_segemented = mean_shift(image1_resized.copy(), 25, 1)
    print(f"number of clusters: {np.unique(image1_segemented.reshape((-1,3)), axis=0).shape[0]}")
    draw_and_save(image1_segemented, os.path.join(img_path,"output/2(e)2-image_msrgb_h25.jpg"))

    image1_segemented = mean_shift(image1_resized.copy(), 75, 1)
    print(f"number of clusters: {np.unique(image1_segemented.reshape((-1,3)), axis=0).shape[0]}")
    draw_and_save(image1_segemented, os.path.join(img_path,"output/2(e)2-image_msrgb_h75.jpg"))

    image2_segemented = mean_shift(image2_resized.copy(), 5, 1)
    print(f"number of clusters: {np.unique(image2_segemented.reshape((-1,3)), axis=0).shape[0]}")
    draw_and_save(image2_segemented, os.path.join(img_path,"output/2(e)masterpiece_msrgb_h5.jpg"))

    image2_segemented = mean_shift(image2_resized.copy(), 25, 1)
    print(f"number of clusters: {np.unique(image2_segemented.reshape((-1,3)), axis=0).shape[0]}")
    draw_and_save(image2_segemented, os.path.join(img_path,"output/2(e)masterpiece_msrgb_h25.jpg"))

    image2_segemented = mean_shift(image2_resized.copy(), 75, 1)
    print(f"number of clusters: {np.unique(image2_segemented.reshape((-1,3)), axis=0).shape[0]}")
    draw_and_save(image2_segemented, os.path.join(img_path,"output/2(e)masterpiece_msrgb_h75.jpg"))

