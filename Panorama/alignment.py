import math
import random

import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
            KeyPoint.pt holds a tuple of pixel coordinates (x, y)
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    #BEGIN TODO 2
    # Construct the A matrix that will be used to compute the homography
    # based on the given set of matches among feature sets f1 and f2.

    # Create A as a zero array
    A = np.zeros((2*len(matches),9))
    # Loop through each match to fill out A
    for match_num in range(len(matches)):
        match = matches[match_num]
        # Feature1 contains (x,y) from image 1
        feature1 = f1[match.queryIdx].pt
        # Feature2 contains (x,y) from image 2
        feature2 = f2[match.trainIdx].pt
        # Setup x and y values
        x1 = feature1[0]
        y1 = feature1[1]
        x2 = feature2[0]
        y2 = feature2[1]
        # Fill A with appropriate values
        A[match_num*2] = [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2]
        A[(match_num*2)+1] = [0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2]

    #END TODO

    if A_out is not None:
        A_out[:] = A

    x = minimizeAx(A) # find x that minimizes ||Ax||^2 s.t. ||x|| = 1


    H = np.eye(3) # create the homography matrix

    #BEGIN TODO 3
    #Fill the homography H with the correct values
    index = 0
    for i in range(3):
        for j in range(3):
            H[i][j] = x[index]
            index += 1

    #END TODO

    return H

def minimizeAx(A):
    """ Given an n-by-m array A, return the 1-by-m vector x that minimizes
    ||Ax||^2 subject to ||x|| = 1.  This turns out to be the right singular
    vector of A corresponding to the smallest singular value."""
    return np.linalg.svd(A)[2][-1,:]

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslate) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call computeHomography.
    #This function should also call getInliers and, at the end,
    #least_squares_fit.
    inlier_indices = []
    inlier_count = 0
    data = []
    # Run RANSAC algorithm k times
    for k in range(nRANSAC):

        # Check motion model
        if (m == eTranslate):
            # Pick a random feature match
            data = [matches[random.randint(0,len(matches)-1)]]

        elif (m == eHomography):
            # Pick four random feature matches
            rnums = random.sample(range(0,len(matches)),4)
            data = [matches[rnums[0]], matches[rnums[1]], matches[rnums[2]], matches[rnums[3]]]
        # Compute inter-image transformation matrix
        tempM = computeHomography(f1,f2,data)

        # Count the number of inliers in the inter-image transformation matrix
        temp_inlier_indices = getInliers(f1, f2, matches, tempM, RANSACthresh)
        temp_inlier_count = len(temp_inlier_indices)
        # Update M if there are more inliers
        if (inlier_count < temp_inlier_count):
            inlier_indices = temp_inlier_indices
            inlier_count = temp_inlier_count

    #END TODO
    return leastSquaresFit(f1, f2, matches, m, inlier_indices)

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        # Determine if the ith matched feature f1[matches[i].queryIdx], when
        # transformed by M, is within RANSACthresh of its match in f2.
        # If so, append i to inliers
        feature1 = f1[matches[i].queryIdx].pt
        point = [feature1[0],feature1[1],1]
        point = np.dot(M,point)
        point = point/point[2]
        feature2 = f2[matches[i].trainIdx].pt
        x_distance = (point[0]-feature2[0])**2
        y_distance = (point[1]-feature2[1])**2
        if (math.sqrt(x_distance+y_distance) <= RANSACthresh):
            inlier_indices.append(i)

        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if (m == eTranslate):
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        #BEGIN TODO 6 :Compute the average translation vector over all inliers.
        # Fill in the appropriate entries of M to represent the average
        # translation transformation.
        for i in range(len(inlier_indices)):
            feature1 = f1[matches[inlier_indices[i]].queryIdx].pt
            feature2 = f2[matches[inlier_indices[i]].trainIdx].pt
            u = u + (feature2[0]-feature1[0])
            v = v + (feature2[1]-feature1[1])
        u = u/len(inlier_indices)
        v = v/len(inlier_indices)
        M[0][2] = u
        M[1][2] = v
        #END TODO

    elif (m == eHomography):
        #BEGIN TODO 7
        #Compute a homography M using all inliers. This should call
        # computeHomography.
        inliers = []
        for i in range(len(inlier_indices)):
            inliers.append(matches[inlier_indices[i]])
        M = computeHomography(f1,f2,inliers)
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M

