import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def form_histogram(image, levels):
    # create an empty array of length G(gray-levels, 256 here) and form histogram of 
    # the distribution of pixel value of the image
    histo_array = [0]*levels

    # scan through every pixel and count the frequency for each intensity gp
    for row in image:
        for gp in row:
            histo_array[gp] += 1
    return histo_array



def histo_equalize(image):
    
    # =================================
    # (a.)(10%)grayscale the image and implement histogram equalization and explain how do you implement step by step (you can not use cv2.equalizeHist() or other similar functions.)
    # =================================

    image_size = image.shape[0]*image.shape[1]
    levels = 2**(image.dtype.itemsize*8)
    # get the histogram distribution of pixel value:
    histo_array = form_histogram(image, levels)
    
    # check if total equals number of pixels
    print("area of histogram equals image size:", sum(histo_array)==image_size)
    
    # T[p]
    def t_function(i: int, histo_c_array):
        return round(((levels-1)/image_size)*histo_c_array[i])

    # form cumulative histogram:
    histo_c_array = histo_array.copy()
    for i, freq in enumerate(histo_c_array):
        if i != 0:
            histo_c_array[i] = freq + histo_c_array[i - 1]

    
    # histgoram equalization, rescan image every pixel intensity and apply T[gp]
    output_image = image.copy()
    for row in output_image:
        for i, gp in enumerate(row):
            row[i] = t_function(gp,histo_c_array)
    

    # =================================
    #  (b.)(1 image)show the image before and after histogramequalization
    # =================================

    # plot 2 images side by side in one image
    fig = plt.figure('(b)',dpi=150)
    axs = fig.subplots(1,2) 
    axs[0].set_title('Before:')
    # # since default matplotlib RGB, we have to specify it as grayscale, 
    # if we want to show colored opencv image, convert to cv2.COLOR_BGR2RGB
    axs[0].imshow(image, cmap='gray') 
    axs[0].axis('off')
    axs[1].set_title('After:')
    axs[1].imshow(output_image, cmap='gray')
    axs[1].axis('off')
    
    plt.suptitle('(b)before and after histogram equalization', fontsize=16, y=0.95)
    plt.savefig(os.path.join(save_path, "1-b.jpg"))
    plt.draw()
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()

    # =================================
    #  (c.)(10%)(1 image)plot the distribution of pixel value before and after histogram equalization
    # =================================
    
    # get the histogram distribution of pixel value after equalized:
    equalized_histo_array = form_histogram(output_image, levels)

    # plot 2 histograms side by side in one image
    fig = plt.figure('(c)', dpi=150,figsize=(8, 6))
    axs = fig.subplots(1,2) 
    axs[0].set_title('Before:')
    axs[0].hist(range(len(histo_array)), bins=len(histo_array), weights=histo_array, edgecolor='blue')
    axs[0].set_xlabel('Intensity')
    axs[0].set_ylabel('Frequency')
    
    axs[1].set_title('After:')
    axs[1].hist(range(len(equalized_histo_array)), bins=len(equalized_histo_array), weights=equalized_histo_array, edgecolor='blue')
    axs[1].set_xlabel('Intensity')
    axs[1].set_ylabel('Frequency')

    plt.suptitle('(c)before and after histogram equalization', fontsize=16, y=0.95)

    axs[0].yaxis.set_label_position("left")
    plt.savefig(os.path.join(save_path, "1-c.jpg"))
    plt.draw()
    print("Press Enter to close the window:")
    plt.waitforbuttonpress(0)
    plt.close()

    return output_image


# this is just for testing,
# to see if my implementation works as expected
def histo_equalize_test(image, output_image):
    levels = 2**(image.dtype.itemsize*8)
    # this opencv function is used only for double checking
    output_image2 = cv2.equalizeHist(image)

    # compute mean absolute difference of two images:
    mean_abs_diff = np.mean(np.abs(output_image.astype(np.int16)-output_image2.astype(np.int16)))
    similarity = ((levels-1 - mean_abs_diff) / (levels-1)) * 100
    print(f"The MAD similarity between the self-implementation and default equalizeHist() function is {similarity:.4f}%.")



if __name__=='__main__':

    # read image img subfolder
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "img", "hw1-1.jpg")
    save_path = os.path.join(os.path.dirname(img_path)) # images save location
    
    # read image as grayscale
    #image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # use this when path contains chinese char
    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    # do histogram equalization here
    output_image = histo_equalize(image)

    # check if my implementation works as expected
    # histo_equalize_test(image, output_image)