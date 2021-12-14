import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    img_ht = len(img);
    img_w = len(img[0]);
    #Setup pad spacing
    kernel_ht = len(kernel)//2;
    kernel_w = len(kernel[0])//2;
    #sets up pad_img and output for rgb images
    if(img.ndim == 3):
        pad_img = np.pad(img, ((kernel_ht,kernel_ht),(kernel_w,kernel_w), (0,0)), mode='constant');
        output = np.zeros(((img_ht + 2*kernel_ht, img_w + 2*kernel_w, 3)));
    #sets up pad_img and output for grayscale images
    else:
        pad_img = np.pad(img, ((kernel_ht, kernel_ht),(kernel_w,kernel_w)), mode = 'constant')
        output = np.zeros((img_ht + 2*kernel_ht, img_w + 2*kernel_w));
    #loops through the kernel
    for i in range(-kernel_ht, kernel_ht+1):
       for j in range(-kernel_w, kernel_w+1):
           #shifts the input to match the current kernel index
           temp_img = np.roll(np.roll(pad_img,-i,axis = 0),-j, axis = 1);
           output = np.add(output,(kernel[i+kernel_ht][j+kernel_w]*temp_img));

    if(img.ndim == 3):
        return output[kernel_ht:img_ht+kernel_ht,kernel_w:img_w+kernel_w,:];
    else:
        return output[kernel_ht:img_ht+kernel_ht,kernel_w:img_w+kernel_w];

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    #Cross correlate the image with a reversed kernel
    return cross_correlation_2d(img, np.flip(np.flip(kernel,0),1));

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height may be different, but sigma applies to
    both dimensions. Normalize the kernel so it sums to 1.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    kernel_w=width//2;
    kernel_h=height//2;
    kernel = np.zeros((height,width));
    for x in range(-kernel_h, kernel_h+1):
        for y in range(-kernel_w, kernel_w+1):
            #Gaussian Formula
            kernel[x+kernel_h][y+kernel_w] = (1.0/(2.0*np.pi*sigma**2.0))*np.exp(-((x**2.0+y**2.0)/(2.0*sigma**2.0)));
    sum = np.sum(kernel);
    kernel = kernel/sum;
    return kernel;


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return cross_correlation_2d(img,gaussian_blur_kernel_2d(sigma,size,size));

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return img-low_pass(img,sigma,size);

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

def construct_laplacian(img, levels):
    """ Construct a Laplacian pyramid for the image img with `levels` levels.
    Returns a python list; the first `levels`-1 elements are high-pass images
    each one half the size of the previous; the last one is the remaining
    low-pass image.
    Precondition: img has dimensions HxWxC, and H and W are each divisible
    by 2**(levels-1) """
    h, w, c = img.shape
    f = 2**(levels-1)
    assert h % f == 0 and w % f == 0
    pyr = [];
    #setup blur filter
    gaussian = np.array([[0,0,0,0,0],[0,0,0,0,0],[0.0625,0.25,0.375,0.25,0.0625],[0,0,0,0,0],[0,0,0,0,0]]);
    gaussian = cross_correlation_2d(gaussian,gaussian.T)
    image = np.copy(img);
    for level in range(levels):
        #save low pass image on last level
        if(level == levels-1):
            pyr.append(downsample_img);
        #save high pass image
        else:
            #blur the image
            blur_img = cross_correlation_2d(image,gaussian);
            #downsample the image
            downsample_img = blur_img[0::2,0::2];
            #setup upsample of the downsample image
            upsample_img = np.zeros(((2*len(downsample_img),2*len(downsample_img[0]),3)));
            #put black pixels between pixel values
            upsample_img[::2,::2] = downsample_img;
            #blur image and even out brightness
            upsample_img = cross_correlation_2d(upsample_img,gaussian)*4;
            pyr.append(image - upsample_img);
            image = downsample_img;
    return pyr;

def reconstruct_laplacian(pyr, weights=None):
    """ Given a laplacian pyramid, reconstruct the original image.
    `pyr` is a list of pyramid levels following the spec produced by
        `construct_laplacian`
    `weights` is either None or a list of floats whose length is len(pyr)
        If weights is not None, scale each level of the pyramid by its
        corresponding value in the weights list while adding it into the 
        reconstruction.
    """
    #copy and reverse the list without affecting the original
    pyr_copy = pyr.copy();
    pyr_copy.reverse();
    img = pyr_copy[0];
    #setup blur filter
    gaussian = np.array([[0,0,0,0,0],[0,0,0,0,0],[0.0625,0.25,0.375,0.25,0.0625],[0,0,0,0,0],[0,0,0,0,0]]);
    gaussian = cross_correlation_2d(gaussian,gaussian.T)
    #rebuild original image
    for level in range(1,len(pyr_copy)):
        #upsample image
        z_img = np.zeros(((2*len(img),2*len(img[0]),3)));
        z_img[::2,::2] = img;
        img = cross_correlation_2d(z_img, gaussian)*4;
        #apply weight value if necessary
        if not (weights == None):
            img += weights[level]*pyr_copy[level];
        else:
            img += pyr_copy[level];
    return img;
