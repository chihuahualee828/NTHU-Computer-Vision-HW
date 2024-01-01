import cv2
import numpy as np
import matplotlib.pyplot as plt


def lls_eight_point(pts1, pts2):

    # m x 9 (where m>=8)
    A = []
    for i in range(pts1.shape[0]):
        row = [ pts1[i][0]*pts2[i][0], pts1[i][0]*pts2[i][1], pts1[i][0],
               pts1[i][1]*pts2[i][0], pts1[i][1]*pts2[i][1], pts1[i][1],
               pts2[i][0], pts2[i][1], 1 ]
        A.append(row)

    A = np.array(A)

    U, S, Vt = np.linalg.svd(A)

    # get last row of V
    F = Vt.T[:,-1].reshape(3, 3)

    # compute rank 2 F UD'Vt
    Uf, Sf, Vft = np.linalg.svd(F)
    print(Uf.shape, Sf.shape, Vft.shape)

    # D -> D'
    Sf[-1] = 0

    F = np.dot(np.dot(Uf,np.diag(Sf)),Vft)
    print("Fundamental matrix(wo normalized):")
    print(F)
    print("least square constraint for SVD:")
    print(f"norm(F): {np.linalg.norm(F)}")
    print(f"norm(AF)^2(minimized to 0) : {np.linalg.norm(np.dot(A,F.flatten()))**2}")
    # print(np.sum(np.dot(A,F.flatten())**2))

    return F

def normalizled_eight_point(pts1, pts2):

    # translation(move origin to the center) and scale
    T = np.array([[2/image1.shape[1], 0, -1], [0, 2/image1.shape[0], -1], [0, 0, 1]])
    # m x 9 (where m>=8)
    A = []
    for i in range(pts1.shape[0]):
        pt1_homo = np.hstack((pts1[i],1))
        pt1_hat = np.dot(T, pt1_homo)[:2]
        pt2_homo = np.hstack((pts2[i],1))
        pt2_hat = np.dot(T, pt2_homo)[:2]
        row = [ pt1_hat[0]*pt2_hat[0], pt1_hat[0]*pt2_hat[1], pt1_hat[0],
               pt1_hat[1]*pt2_hat[0], pt1_hat[1]*pt2_hat[1], pt1_hat[1],
               pt2_hat[0], pt2_hat[1], 1 ]
        
        A.append(row)

    A = np.array(A)

    U, S, Vt = np.linalg.svd(A)

    # get last row of V
    F = Vt.T[:,-1].reshape(3, 3)

    # compute rank 2 F UD'Vt
    Uf, Sf, Vft = np.linalg.svd(F)
    print(Uf.shape, Sf.shape, Vft.shape)

    # D -> D'
    Sf[-1] = 0

    F = np.dot(np.dot(Uf,np.diag(Sf)),Vft)

    F = np.dot(np.dot(T.T, F), T)
    print("Fundamental matrix(normalized):")
    print(F)
    # print("least square constraint for SVD:")
    # print(f"norm(F): {np.linalg.norm(F)}")
    # print(f"norm(AF)^2(minimized to 0) : {np.linalg.norm(np.dot(A,F.flatten()))**2}")

    return F

# image2(偏) should be viewed from right angle, so it's right view
# image1(正) should be viewed from left, left view
# so epilines for image1(left) should be Lleft = F^T*pright(which is pt2)
def compute_line(pts1, pts2, F):
    epi_lines_1 = []
    epi_lines_2 = []

    # l = Fp, l' =F^Tp'
    for i in range(pts1.shape[0]):
        pt1_homo = np.hstack((pts1[i],1))
        pt2_homo = np.hstack((pts2[i],1))
        epi_lines_1.append(np.dot(F, pt2_homo))
        epi_lines_2.append(np.dot(F.T, pt1_homo))
    
    epi_lines_1 = np.array(epi_lines_1)
    epi_lines_2 = np.array(epi_lines_2)
    
    return epi_lines_1, epi_lines_2

def drawlines(image,lines,pts1):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    img = image.copy()
    total_distance = 0
    total_points = len(pts1)
    colors = [(int(255 * (total_points - i) / total_points), 0, int(255 * i / total_points)) for i in range(total_points)]

    for r, pt1, color in zip(lines, pts1, colors):
        
        # we draw lines from leftmost to rightmost, so x0 = 0
        x0,y0 = 0, int(-r[2]/r[1])
        x1,y1 = img.shape[1], int(-(r[2]+r[0]*img.shape[1])/r[1])

        # distance of current correspondance point to its epipolar line
        total_distance += abs(np.cross(np.array([x1,y1])-np.array([x0,y0]), pt1 - np.array([x0,y0]))) / np.linalg.norm(np.array([x1,y1]) - np.array([x0,y0]))

        img = cv2.line(img, (x0,y0), (x1,y1), color,1)
        img = cv2.circle(img,(int(pt1[0]), int(pt1[1])),5,color,-1)

    avg_distance = total_distance/len(pts1)

    return img, avg_distance

if __name__=="__main__":

    pt_txt1 = open("assets/pt_2D_1.txt")
    pt_txt2 = open("assets/pt_2D_2.txt")
    image1 = cv2.imread("assets/image1.jpg")
    image2 = cv2.imread("assets/image2.jpg")
    
    print(image1.shape)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    pts1 = []
    pts2 = []

    for line in zip(pt_txt1.readlines()[1:], pt_txt2.readlines()[1:]):
        x1, y1 = map(float, line[0].strip().split())
        x2, y2 = map(float, line[1].strip().split())

        pts1.append([x1,y1])
        pts2.append([x2,y2])

    
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)

    print(pts1.shape)


    F = lls_eight_point(pts1, pts2)
    
    epi_lines_1, epi_lines_2 = compute_line(pts1, pts2, F)

    image_l, avg_distance = drawlines(image1,epi_lines_1,pts1)
    print(f"image left - average distance: {avg_distance}")
    image_r, avg_distance = drawlines(image2,epi_lines_2,pts2)
    print(f"image right - average distance: {avg_distance}")

    fig = plt.figure('(c) epipolar lines for (a)unnormalized:', figsize=(8,6))
    plt.suptitle('(c) epipolar lines for (a)unnormalized:', fontsize=16, y=0.95)

    axs = fig.subplots(1,2) 
    axs[0].set_title('image1-left view:')
    axs[0].imshow(image_l)
    axs[0].set_axis_off()
    
    axs[1].set_title('image2-right view:')
    axs[1].imshow(image_r)
    axs[1].set_axis_off()

    plt.imsave("output/wo_normalized_img1.png",image_l)
    plt.imsave("output/wo_normalized_img2.png",image_r)
    plt.draw()
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()

    

    F_n = normalizled_eight_point(pts1, pts2)
    epi_lines_1_n, epi_lines_2_n = compute_line(pts1, pts2, F_n)

    image_l, avg_distance = drawlines(image1,epi_lines_1_n,pts1)
    print(f"image left - average distance: {avg_distance}")
    image_r, avg_distance = drawlines(image2,epi_lines_2_n,pts2)
    print(f"image right - average distance: {avg_distance}")

    fig = plt.figure('(c) epipolar lines for (b)normalized:', figsize=(8,6))
    plt.suptitle('(c) epipolar lines for (b)normalized:', fontsize=16, y=0.95)

    axs = fig.subplots(1,2) 
    axs[0].set_title('image1-left view:')
    axs[0].imshow(image_l)
    axs[0].set_axis_off()
    
    axs[1].set_title('image2-right view:')
    axs[1].imshow(image_r)
    axs[1].set_axis_off()

    plt.imsave("output/normalized_img1.png",image_l)
    plt.imsave("output/normalized_img2.png",image_r)
    plt.draw()
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()
