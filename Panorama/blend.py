import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that that takes an image and a
       transform, and computes the bounding box of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         maxX: int for the maximum X value of a corner
         maxY: int for the maximum Y value of a corner
    """
    #TODO 8 determine the outputs for this method.
    top_left = np.dot(M,[[0],[0],[1]])
    top_left = top_left/top_left[2]
    top_right = np.dot(M, [[0],[len(img[0])-1],[1]])
    top_right = top_right/top_right[2]
    bottom_left = np.dot(M, [[len(img)-1],[0],[1]])
    bottom_left = bottom_left/bottom_left[2]
    bottom_right = np.dot(M, [[len(img)-1],[len(img[0])-1],[1]])
    bottom_right = bottom_right/bottom_right[2]

    minY = min(top_left[1],top_right[1],bottom_left[1],bottom_right[1])
    maxY = max(top_left[1],top_right[1],bottom_left[1],bottom_right[1])
    minX = min(top_left[0],top_right[0],bottom_left[0],bottom_right[0])
    maxX = max(top_left[0],top_right[0],bottom_left[0],bottom_right[0])
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # convert input image to floats
    img = img.astype(np.float64) / 255.0
    inverseM = np.linalg.inv(M)
    inverseM = inverseM/inverseM[2][2]
    # BEGIN TODO 10: Fill in this routine
    for y in range(len(acc)):
        for x in range(len(acc[0])):
            #try:
            values = np.dot(inverseM, [[x],[y],[1]])
            #values = values.astype(np.float64)
            values = values/values[2]
            sourceY = values[1]
            sourceX = values[0]

            upper_x = math.ceil(sourceX)
            lower_x = math.floor(sourceX)
            lower_y = math.floor(sourceY)
            upper_y = math.ceil(sourceY)
            if(upper_y < len(img) and lower_y >= 0 and upper_x < len(img[0]) and lower_x >= 0):
                #print((sourceX,[lower_x,upper_x],[img[lower_y][lower_x][0],img[lower_y][upper_x][0]]))
                #print([lower_x,upper_x,lower_y,upper_y])
                r_lower_interp = np.interp(sourceX,[lower_x,upper_x],[img[lower_y][lower_x][0],img[lower_y][upper_x][0]])[0]
                r_upper_interp = np.interp(sourceX,[lower_x,upper_x],[img[upper_y][lower_x][0],img[upper_y][upper_x][0]])[0]
                r_bilinterp = np.interp(sourceY,[lower_y,upper_y],[r_lower_interp,r_upper_interp])[0]

                g_lower_interp = np.interp(sourceX,[lower_x,upper_x],[img[lower_y][lower_x][1],img[lower_y][upper_x][1]])[0]
                g_upper_interp = np.interp(sourceX,[lower_x,upper_x],[img[upper_y][lower_x][1],img[upper_y][upper_x][1]])[0]
                g_bilinterp = np.interp(sourceY,[lower_y,upper_y],[g_lower_interp,g_upper_interp])[0]

                b_lower_interp = np.interp(sourceX,[lower_x,upper_x],[img[lower_y][lower_x][2],img[lower_y][upper_x][2]])[0]
                b_upper_interp = np.interp(sourceX,[lower_x,upper_x],[img[upper_y][lower_x][2],img[upper_y][upper_x][2]])[0]
                b_bilinterp = np.interp(sourceY,[lower_y,upper_y],[b_lower_interp,b_upper_interp])[0]
                bilinterp = [r_bilinterp,g_bilinterp,b_bilinterp,1]
                if(bilinterp[0] != 0 or bilinterp[1] !=0 or bilinterp[2] != 0):
                    if(sourceX < blendWidth):
                        bilinterp = np.multiply(bilinterp,((sourceX/blendWidth)[0]))
                    elif(sourceX > len(img[0])-blendWidth):
                        bilinterp = np.multiply(bilinterp,(((len(img[0])-sourceX)/blendWidth)[0]))
                    acc[y][x] += bilinterp
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11: fill in this routine
    img = np.copy(acc)
    #img[img[:,:,3] > 0] = img/img[:,:,3]
    #img = np.where(img[:,:,3] > 0, img/img[:,:,3])
    #img = np.any(img != [:,:,:,0])
    for y in range(len(img)):
        for x in range(len(img[0])):
            if(img[y][x][3] != 0):
                img[y][x] = img[y][x]/img[y][x][3]
            else:
                img[y][x] = [0,0,0,0]
    # END TODO
    return (img * 255).astype(np.uint8)


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and returns useful information about the
       accumulated image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and
             transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all
             tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all
             tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = sys.maxsize
    minY = sys.maxsize
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        # this can (should) use the code you wrote for TODO 8
        temp_minX,temp_minY,temp_maxX,temp_maxY = imageBoundingBox(img, M)
        minX = min(temp_minX,minX)
        minY = min(temp_minY,minY)
        maxX = max(temp_maxX,maxX)
        maxY = max(temp_maxY,maxY)
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    #print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    """ Computes parameters for drift correction.
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         translation: transformation matrix so that top-left corner of accumulator image is origin
         width: Width of each image(assumption: all input images have same width)
       OUTPUT:
         x_init, y_init: coordinates in acc of the top left corner of the
            panorama with half the left image cropped out to match the right side
         x_final, y_final: coordinates in acc of the top right corner of the
            panorama with half the right image cropped out to match the left side
    """
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final

def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    if is360:
        # BEGIN TODO 12
        # 497P: you aren't required to do this. 360 mode won't work.

        # 597P: fill in appropriate entries in A to trim the left edge and
        # to take out the vertical drift:
        #   Shift it left by the correct amount
        #   Then handle the vertical drift - using a shear in the Y direction
        # Note: warpPerspective does forward mapping which means A is an affine
        # transform that maps accumulator coordinates to final panorama coordinates
        raise Exception("TODO 12 in blend.py not implemented")
        # END TODO 12

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

