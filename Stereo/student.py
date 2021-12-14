import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- 3 x 3 camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    output = np.zeros((len(points),len(points[0]), 2)) #Setup the output array

    # Fill in output array with projected values
    for y in range(len(points)):
        for x in range(len(points[0])):
            point = np.copy(points[y][x])	#Setup point with x and y values
            point = np.append(point,1)		#Set z = 1
            point = np.transpose(point)		#Reshape point to work with dot products
            camera_point = np.dot(Rt,point)	#Get the camera coordinates
            pixel = np.dot(K,camera_point)	#Get the pixel coordinates
            pixel /= pixel[2]			#Normalize pixel values
            output[y][x] = [pixel[0],pixel[1]]
    return output

def unproject_corners_impl(K, width, height, depth, Rt):
    """
    Unproject the corners of a height-by-width image
    into world coordinates at depth d.
    Input:
        K -- 3x3 camera intrinsics calibration matrix
        width -- width of the image in pixels
        height -- height of the image in pixels
        depth -- depth the 3D point is unprojected to
        Rt -- 3 x 4 camera extrinsics calibration matrix
    Output:
        points -- 2 x 2 x 3 array of 3D positions of the image corners
    """
    #Setup output value and inverse values of Rt and K
    output = np.zeros((2,2,3))
    Rt_square = np.vstack((Rt,[0,0,0,1]))
    Rt_inv = np.linalg.inv(Rt_square)
    K_inv = np.linalg.inv(K)

    """
    For each corner:
    1. Convert point to camera coordinates
    2. Move to depth value
    3. Convert to world coordinates
    4. Normalize world coordinates

    Note: vstack is used to add another dimension to the camera coordinates after moving
    to depth value in order to make it compatable to use the dot product with Rt_inv
    """

    #top_left
    point = [[0.0],[0.0],[1.0]]
    cam = np.dot(K_inv,point)
    dep = depth*cam
    dep = np.vstack((dep,[1]))
    world = np.dot(Rt_inv,dep)
    world=world/world[3]
    output[0][0] = [world[0],world[1],world[2]]

    #top_right
    point = [[float(width-1)],[0.0],[1.0]]
    cam = np.dot(K_inv,point)
    dep = depth*cam
    dep = np.vstack((dep,[1]))
    world = np.dot(Rt_inv,dep)
    world = world/world[3]
    output[0][1] = [world[0],world[1],world[2]]

    #bottom_left
    point = [[0.0],[float(height-1)],[1.0]]
    cam = np.dot(K_inv,point)
    dep = depth*cam
    dep = np.vstack((dep,[1]))
    world = np.dot(Rt_inv,dep)
    world=world/world[3]
    output[1][0] = [world[0],world[1],world[2]]

    #bottom_right
    point = [[float(width-1)],[float(height-1)],[1.0]]
    cam = np.dot(K_inv,point)
    dep = depth*cam
    dep = np.vstack((dep,[1]))
    world = np.dot(Rt_inv,dep)
    world=world/world[3]
    output[1][1] = [world[0],world[1],world[2]]

    return output

def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are flattened into vectors with the default numpy row
    major order.  For example, the following 3D numpy array with shape
    2 (channels) x 2 (height) x 2 (width) patch...

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    gets unrolled using np.reshape into a vector in the following order:

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer side length of (square) NCC patch region; must be odd
    Output:
        normalized -- height x width x (channels * ncc_size**2) array
    """
    h, w, c = image.shape

    normalized = np.zeros((h, w, c, ncc_size, ncc_size), dtype=np.float32)

    k = ncc_size // 2 # half-width of the patch size

    # the following code fills in `normalized`, which can be thought
    # of as a height-by-width image where each pixel is
    #   a channels-by-ncc_size-by-ncc_size array

    for i in range(ncc_size):
        for j in range(ncc_size):
            # i, j is the top left corner of the the patch
            # so the (i=0,j=0) is the pixel in the top-left corner of the patch
            # which is an offset of (-k, -k) from the center pixel

            # example: image is 10x10x3; ncc_size is 3
            # the image pixels for the top left of all patches come from
            # (0, 0) thru (7, 7) because (7, 7) is an offset of -1, -1 
            # (corresponding to i=0, j=0) from the bottom-right-most patch, 
            # which is centered at (8, 8)
            # generalizing to patch halfsize k, it's (h-2k, w-2k)
            # generalizing to any offset into the patch,
            #   the top left will be i, j
            #   the bottom right will be h-2k + i, w-2k + j
            normalized[k:h-k, k:w-k, :, i, j] = image[i:h-2*k+i, j:w-2*k+j, :]

    # For each patch, subtract out its per-channel mean
    # Then divide the patch by its (not-per-channel) vector norm.
    # Patches with norm < 1e-6 should become all zeros.

    # Subtract per-channel-mean from each patch
    normalized = normalized - np.mean(normalized,axis=(3,4))[:,:,:,None,None]

    # Unroll patches into vectors
    vector_array = np.reshape(normalized[:,:],(h,w,-1))

    # Normalize each vector if norm is large enough
    vector_array[np.linalg.norm(vector_array) >= 1e-6] /= (np.linalg.norm(vector_array, axis = 2)[:,:,None])

    # Setup output
    answer = np.zeros((len(vector_array),len(vector_array[0]),len(vector_array[0][0])))

    # Fill in output with patches that don't extend past the boundary of the input
    answer[k:-k,k:-k]=vector_array[k:-k,k:-k]

    return answer


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    return np.nansum((image1*image2),axis=2)
