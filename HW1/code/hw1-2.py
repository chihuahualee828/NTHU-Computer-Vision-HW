import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# (a)(i)Grayscale and Gaussian Smooth(1 image)
def gaussian_smooth(image, sigma=1, kernel_size=3):

    gaussian_1D_filter = cv2.getGaussianKernel(kernel_size,sigma)
    gaussian_2D_filter = np.outer(gaussian_1D_filter,gaussian_1D_filter)
    output_image = cv2.filter2D(image, -1, kernel=gaussian_2D_filter)

    return output_image

# def gaussian_smooth_test(image, output_image, sigma=1, kernel_size=(3,3)):

#     levels = 2**(image.dtype.itemsize*8)
#     # this opencv function is used only for double checking
#     output_image2 = cv2.GaussianBlur(image, kernel_size, sigma)
#     print(output_image2)
#     # compute mean absolute difference of two images:
#     mean_abs_diff = np.mean(np.abs(output_image.astype(np.int16)-output_image2.astype(np.int16)))
#     similarity = ((levels-1 - mean_abs_diff) / (levels-1)) * 100
#     print(f"The MAD similarity between the self-implementation and default GaussianBlur() function is {similarity:.4f}%.")

    

# (a)(ii.)Intensity Gradient (Sobel operator)(2 image) 
def sobel_operator(image):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    sobel_y = sobel_x.T
    sobel_y[[0, 2]] = sobel_y[[2, 0]]

    image = image.astype(np.float32)
    gradient_x = cv2.filter2D(image, -1, kernel=sobel_x)
    gradient_y = cv2.filter2D(image, -1, kernel=sobel_y)
    
    return gradient_x, gradient_y

# (iii.)Structure Tensor(1 image)
def structure_tensor(image, window_size=3, k = 0.04):
    
    epsilon = 1e-6
    response_array = np.zeros_like(image).astype(np.float32)
    # response_array = cv2.cornerHarris(image, 3, 3, 0.04)
    dx, dy = sobel_operator(image)
    dx2 = dx*dx
    dy2 = dy*dy
    dxy = dx*dy
    # dxy == dyx
    print("calculating structure tensor and response...")
    offset = int(window_size/2)
    for y in range(offset, response_array.shape[0]-offset):
        for x in range(offset, response_array.shape[1]-offset):

            # sum of structure tensor within window
            Ix2 = np.sum(dx2[y-offset:y+1+offset, x-offset:x+1+offset])
            Iy2 = np.sum(dy2[y-offset:y+1+offset, x-offset:x+1+offset])
            IxIy = np.sum(dxy[y-offset:y+1+offset, x-offset:x+1+offset])
            h_tensor_window = np.array([[Ix2, IxIy], [IxIy, Iy2]])

            # different approach
            # h_tensor_window = np.zeros((2,2))
            # for i in range(-offset, offset+1):
            #     for j in range(-offset, offset+1):
            #         Ix, Iy = (dx[y+j, x+i], dy[y+j, x+i])
            #         a = np.array([[Ix], [Iy]])

            #         h_tensor = np.dot(a, a.T)
            #         h_tensor_window += h_tensor

            
            # two ways of calculating Harris response using local structure tensor:
            # response = np.linalg.det(h_tensor_window)-k*(h_tensor_window.trace()**2)
            # Harris operator:
            response = np.linalg.det(h_tensor_window)/(h_tensor_window.trace()+epsilon)
            response_array[y, x] = response

    return response_array
    
def nms(response_array, nms_window_size = 3):
    offset = int(nms_window_size/2)
    for y in range(offset, response_array.shape[0]-offset):
        for x in range(offset, response_array.shape[1]-offset):
            # Find local maxima in a window
            window = response_array[y-offset:y+1+offset, x-offset:x+1+offset]
            local_max = np.max(window)
            for row in window:
                for i,each in enumerate(row):
                    if each<local_max:
                        row[i] = 0
    return response_array

# do thresholding on corner response 
def response_thresholding(response_array, threshold):
    response_array = np.where((response_array > threshold*response_array.max()), 255, 0)
    return response_array


# (a)
def harris_corner_detector(image, sigma=3, kernel_size=3, window_size=3, threshold=0.1):
    # (i.) 
    gaussian_blurred_image = gaussian_smooth(image, sigma, kernel_size)
    # gaussian_smooth_test(image, gaussian_blurred_image, 3, (3,3))
    gradient_image_x, gradient_image_y = sobel_operator(gaussian_blurred_image)
    response_array = structure_tensor(image, window_size)
    harris_response_image = response_thresholding(response_array, threshold)

    response_array_nms = nms(response_array, nms_window_size=5)
    harris_response_nms_image = response_thresholding(response_array_nms, threshold)

    return gaussian_blurred_image, gradient_image_x, gradient_image_y, harris_response_image, harris_response_nms_image

    

# draw images and save
def draw_and_save_steps(*images, save_path):

    # save images to folder:
    plt.imsave(os.path.join(save_path, '2-(a)(i.)_gaussian_blur.jpg'), images[0], cmap='gray')
    plt.imsave(os.path.join(save_path, '2-(a)(ii.)_gradient_x.jpg'), images[1], cmap='gray')
    plt.imsave(os.path.join(save_path, '2-(a)(ii.)_gradient_y.jpg'), images[2], cmap='gray')
    plt.imsave(os.path.join(save_path, '2-(a)(iii.)_corner_response.jpg'), images[3], cmap='gray')
    plt.imsave(os.path.join(save_path, '2-(a)(iv.)_corner_response_nms.jpg'), images[-1], cmap='gray')

    plt.figure('(a)(i.) Gaussian Smooth')
    plt.imshow(images[0], cmap='gray')
    plt.suptitle('(a)(i.) Gaussian blur:', fontsize=16, y=0.95)
    plt.draw()
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()


    fig = plt.figure('(a)(ii.) Intensity Gradient (Sobel operator)', figsize=(8,6))
    axs = fig.subplots(1,2) 
    axs[0].set_title('x:')
    axs[0].imshow(images[1], cmap='gray')
    axs[0].set_axis_off()
    
    axs[1].set_title('y:')
    axs[1].imshow(images[2], cmap='gray')
    axs[1].set_axis_off()
    plt.suptitle('(a)(ii.) Show gradient intensity map of x and y direction:', fontsize=16, y=0.95)
    plt.draw()
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()


    plt.figure('(a)(iii.) Harris Response', figsize=(8,6))
    plt.imshow(images[3], cmap='gray')
    plt.suptitle('(a)(iii.) Show the corner response R after thresholding the response:', fontsize=16, y=0.95)
    plt.draw()
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()

    plt.figure('(a)(iv.) Harris Response NMS', figsize=(8,6))
    plt.imshow(images[-1], cmap='gray')
    plt.suptitle('(a)(iv.) Harris Response NMS:', fontsize=16, y=0.95)
    plt.draw()
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()


def draw_and_save_overlaid(image, overlay, window_size, threshold, save_path, title = "(b)(ii.)"):

    plt.figure(f'{title} Corner Points Overlaid', figsize=(8,6))
    plt.imshow(image, cmap='gray')
    overlay = np.column_stack(np.where(overlay > 0))
    # Create a scatter plot to draw the points on the image
    plt.scatter(overlay[:, 1], overlay[:, 0], s=1, c='red')
    plt.suptitle(f'{title} Shows the original image(grayscale) with corner points overlaid.', fontsize=16, y=0.95)
    if window_size and threshold:
        plt.figtext(0.5, 0.02, f"window_size={window_size}, threshold={threshold}", ha="center", fontsize=12)
    plt.savefig(save_path, dpi=300)
    plt.draw()
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()


def draw_response_array(response_array, window_size, threshold, save_path, question):

    if question == 1:
        plt.figure('(c)(i.) Harris Response', figsize=(8,6))
        plt.suptitle('(c)(i.)Try a different window size in computing the structure tensor H of each pixel.', fontsize=16, y=0.95)
    else:
        plt.figure('(c)(ii.) Harris Response', figsize=(8,6))
        plt.suptitle('(c)(ii.)Try a different threshold in thresholding corner response.', fontsize=16, y=0.95)

    if window_size and threshold:
        plt.figtext(0.5, 0.02, f"window_size={window_size}, threshold={threshold}", ha="center", fontsize=12)
    plt.imshow(response_array, cmap='gray')
    plt.savefig(save_path, dpi=300)
    plt.draw()
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()

if __name__=='__main__':
    # read image img subfolder
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "img", "hw1-2.jpg")
    save_path = os.path.dirname(img_path) # images save location
    
    # read image as grayscale (if usage of cv2.IMREAD_GRAYSCALE is not allowed, use NTSC formula to convert RGB into grayscale value)
    #image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # use this when path contains chinese char
    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    window_size = 3
    threshold = 0.1
    # (a) implement Harris corner detector
    output_images = harris_corner_detector(image, sigma=3, kernel_size=3, window_size=window_size, threshold=threshold)

    # # (b)(i.) Show images after each step in (a.)
    draw_and_save_steps(*output_images, save_path=save_path)
    # # (b)(ii.) Shows the original image(grayscale) with corner points overlaid.
    draw_and_save_overlaid(image, output_images[-1], window_size, threshold, save_path=os.path.join(save_path, "2-(b)(ii.)_corner_response_overlay.jpg"))
    
    # (c)(i.) Try a different window size in computing the structure tensor H of each pixel.
    print("(c)(i.)")
    window_size = 7
    output_images = harris_corner_detector(image, sigma=3, kernel_size=3, window_size=window_size, threshold=threshold)
    draw_response_array(output_images[-2], window_size=7, threshold=0.1, save_path=os.path.join(save_path, "2-(c)(i.)_corner_response.jpg"), question=1)
    draw_and_save_overlaid(image, output_images[-1], window_size=7, threshold=0.1, save_path=os.path.join(save_path, "2-(c)(i.)_corner_response_overlay.jpg"), title="(c)(i.)")
    
    # (c)(ii.) Try a different threshold in thresholding corner response.
    print("(c)(ii.)")
    threshold = 0.5
    output_images = harris_corner_detector(image, sigma=3, kernel_size=3, window_size=window_size, threshold=threshold)
    draw_response_array(output_images[-2], window_size, threshold, save_path=os.path.join(save_path, "2-(c)(ii.)_corner_response.jpg"), question=2)
    draw_and_save_overlaid(image, output_images[-1], window_size, threshold, save_path=os.path.join(save_path, "2-(c)(ii.)_corner_response_overlay.jpg"), title="(c)(ii.)")
