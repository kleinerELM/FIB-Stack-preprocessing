#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################
# Automated Alignment and data preparation for FIB/SEM
# image stacks
#
# © 2020 Florian Kleiner
#   Bauhaus-Universität Weimar
#   Finger-Institut für Baustoffkunde
#
# programmed using python 3.7, gnuplot 5.2,
# Fiji/ImageJ 1.52k
# don't forget to install PIL (pip install Pillow)
#
#########################################################

import cv2
import os, sys
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
import csv
import multiprocessing

#remove root windows
root = tk.Tk()
root.withdraw()

translation = []
error_list = []

def is_x_near_y( x, y ):
    return (x < (y+0.02) and x > (y-0.02))

def homography_is_translation(h):
    if not (
        is_x_near_y( h[0,0], 1 ) and
        is_x_near_y( h[0,1], 0 ) and
        is_x_near_y( h[1,1], 1 ) and
        is_x_near_y( h[1,0], 0 )
    ):
        #print('  WARNING: Homography is not only a translation! ', h)
        return False

    if len(h) == 3:
        if not (
            is_x_near_y( h[2,0], 0 ) and
            is_x_near_y( h[2,1], 0 ) and
            is_x_near_y( h[2,2], 1 )
        ):
            #print('  WARNING: Homography is not only a translation! ', h)
            return False

    return True

def get_translation_area( translation ):
    _pos = 0
    _min = 0
    _max = 0
    for x in np.nditer(translation):
        _pos += x
        if _pos < _min: _min = _pos
        if _pos > _max: _max = _pos

    return _min, _max

def get_homography_from_translation(t_x, t_y):
    return np.array([[1,0,t_x],[0,1,t_y],[0,0,1]])


def extract_translation_from_homography(h):
    # only extract x and y movement
    # https://stackoverflow.com/questions/25658443/calculating-scale-rotation-and-translation-from-homography-matrix
    t_x = h[0,2]
    t_y = h[1,2]
    return t_x, t_y

# t_x, t_y, h_new = extract_translation_from_homography(h)
MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.4
def get_image_homography(im1, im2, mask=None, filename=''):
    # Convert images to grayscale
    if len(im1.shape) == 3:
      im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
      im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb  = False
    sift = True
    # Detect features and compute descriptors.
        #feature_detector = cv2.SURF_create()
    if orb:
        feature_detector = cv2.ORB_create(nfeatures=MAX_FEATURES)
        keypoints1, descriptors1 = feature_detector.detectAndCompute(im1, mask)
        keypoints2, descriptors2 = feature_detector.detectAndCompute(im2, mask)
        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)
        print(matches)
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)
        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

    if sift:
        feature_detector = cv2.SIFT_create(nfeatures=MAX_FEATURES, nOctaveLayers = 3, contrastThreshold = 0.04, edgeThreshold = 5, sigma = 1.6)
        keypoints1, descriptors1 = feature_detector.detectAndCompute(im1, mask)
        keypoints2, descriptors2 = feature_detector.detectAndCompute(im2, mask)
        # Match features.
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1,descriptors2,k=2)

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        points1 = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        points2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0) # <- wtf is the 5.0 - some method??
    #h, mask = cv2.estimateAffinePartial2D(points1, points2)# cv2.RANSAC)

    if not homography_is_translation(h) and filename != '':
        # Draw top matches
        imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        cv2.imwrite(os.path.dirname(os.path.realpath(__file__)) + os.sep + filename + "_matches.tif", imMatches)

    return h, mask

def alignImages(im1, im2, mask=None):
    print( "  aligning image using OpenCV2", flush=True )

    h, _ = get_image_homography(im1, im2, mask)

    # Use homography
    height, width = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

def denoiseNLMCV2( image ):
    t1 = time.time()
    print( "  denoising image using OpenCV2", flush=True )
    denoised = np.zeros(image.shape, np.uint8) # empty image
    cv2.fastNlMeansDenoising( image,
                            denoised,
                            h=15,
                            templateWindowSize=7,
                            searchWindowSize=(15+1)
                            )
    print( ", took %f s" % (time.time() - t1) )

    return denoised

def get_centered_mask( ref_img, mask_size=0.7, mask_values=255):
    mask = np.zeros(ref_img.shape, np.uint8) # empty mask
    # dimensions of inner square
    m_height = int(mask_size*ref_img.shape[0])
    m_width = int(mask_size*ref_img.shape[1])
    # padding of inner square
    m_v_pad = int((ref_img.shape[0]-m_height)/2)
    m_h_pad = int((ref_img.shape[1]-m_width)/2)
    # fill inner square with ones
    mask[m_v_pad:m_height+m_v_pad, m_h_pad:m_width+m_h_pad] = np.full((m_height, m_width), mask_values, np.uint8) #np.ones((m_height, m_width), np.uint8)
    return mask

# errorhandling of the homography process
# and extraction of the results
def image_processing_thread(filename, im1, im2, mask, mask_full):
    gauss_kernel = (5, 3)

    im_1_denoised = im1#cv2.GaussianBlur(im1, gauss_kernel, cv2.BORDER_DEFAULT)
    im_2_denoised = im2#cv2.GaussianBlur(im2, gauss_kernel, cv2.BORDER_DEFAULT)

    #im_1_denoised = cv2.medianBlur(im1, 5)
    #im_2_denoised = cv2.medianBlur(im2, 5)

    #im_1_denoised = cv2.equalizeHist(im_1_denoised)
    #im_2_denoised = cv2.equalizeHist(im_2_denoised)
    #im_1_denoised = cv2.equalizeHist(im1)
    #im_2_denoised = cv2.equalizeHist(im2)

    error = None
    h, _ = get_image_homography(im_1_denoised, im_2_denoised, mask, filename=filename + '_a')
    if not homography_is_translation(h):
        print('  WARNING: Homography is not only a translation! Retrying full image')
        print(h)
        h, _ = get_image_homography(im_1_denoised, im_2_denoised, mask_full, filename=filename + '_b')

        print(h)
        if not homography_is_translation(h):
            error = [filename, 'ERROR']
            print('  WARNING: Homography REALLY is not only a translation! ')
        else:
            error = [filename, 'WARNING']
    t_x, t_y = extract_translation_from_homography(h)
    result = [filename, t_x, t_y]
    print('  done processing {}'.format(filename))
    return result, error

#singlethreaded processing
def process_translation_of_folder_singlecore(folder, images, loaded_images):
    print('processing image stack singlethreaded:')
    global translation
    global error_list

    im1 = None
    im2 = None
    mask = None
    mask_full = None
    last_filename = ''
    for i, filename in enumerate( images ):
        file_path = folder + os.sep + filename
        #print( " processing {} ({} / {}):".format(filename, i+1, len(images)) )
        im2 = im1
        im1 = loaded_images[i]
        if not im2 is None:
            translation_line, error_list_line = image_processing_thread(filename, im1, im2, mask, mask_full)

            if error_list_line is not None:  error_list.append( error_list_line )
            translation.append(translation_line)
        else:
            mask      = get_centered_mask(im1, mask_size = 1)#0.8)
            mask_full = get_centered_mask(im1, mask_size = 1)

        last_filename = filename

    return translation, error_list

# multithreaded processing
def store_result(result):
    global translation
    global error_list

    result = list(result)
    translation.append(result[0])
    if not result[1] is None:  error_list.append( result[1] )

def process_translation_of_folder_multicore(folder, images, loaded_images):
    print('processing image stack multithreaded:')
    global translation
    global error_list

    im1 = None
    im2 = None
    mask = None
    mask_full = None

    coreCount = multiprocessing.cpu_count()
    processCount = (coreCount - 1) if coreCount > 1 else 1
    pool = multiprocessing.Pool(processCount)

    for i, image in enumerate( images ):
        #print( " processing {} ({} / {})".format(image, i+1, len(images)) )
        im2 = im1
        im1 = loaded_images[i]
        if not im2 is None:
            #translation_line, error_list_line = image_processing_thread(filename, im1, im2, mask, mask_full)
            pool.apply_async(image_processing_thread, args=(image, im1, im2, mask, mask_full), callback = store_result)
        else:
            mask      = get_centered_mask(im1, mask_size = 0.75)
            mask_full = get_centered_mask(im1, mask_size = 1)

    pool.close()
    pool.join()

    # results come in unsorted -> sort
    return translation, error_list

def get_image_in_canvas(image, canvas,
                        width, height,
                        x_min, y_min,
                        x_t, y_t):
    #print(int(-x_min+x_t), int(width-x_min+x_t), int(-y_min+y_t), int(height-y_min+y_t), canvas.shape, image.shape)
    #print()
    canvas[int(-x_min+x_t):int(width-x_min+x_t), int(-y_min+y_t):int(height-y_min+y_t)] = image

    return canvas

# main process function
def process_translation_of_folder(folder=None, multicore = True):
    global translation
    global error_list

    if folder is None:
        folder = filedialog.askdirectory(title='Please select the image / working directory')

    images = []
    for file in os.listdir(folder):
        if ( file.endswith(".tif") or file.endswith(".TIF")):
            images.append( file )

    print('loading {} images...'.format(len(images)))
    loaded_images = []
    for image in images:
        loaded_images.append( cv2.imread(folder + os.sep + image, cv2.IMREAD_GRAYSCALE) )


    print("processing {} images...".format(len(images)))
    if multicore:
        process_translation_of_folder_multicore( folder, images, loaded_images)
    else:
        process_translation_of_folder_singlecore(folder, images, loaded_images)

    print("processing basic data...")
    width, height = loaded_images[0].shape
    translation = sorted(translation)

    #save results
    write_list_to_csv(translation, folder + os.sep + 'translations.csv', ['file', 'transl_x', 'transl_y'])

    if len(error_list) > 0:
        write_list_to_csv(sorted(error_list),  folder + os.sep + 'error_list.csv',   ['file_b', 'serverity'])

    if len(translation) == 0:
        sys.exit('ERROR: no translation data found!')

    arr = np.array(translation)
    b   = np.delete(arr, 0, axis=1).astype(np.float)

    x_translation = np.pad(b[:,0], (1, 0), 'constant')  # np.pad() adds a 0 for the first image
    y_translation = np.pad(b[:,1], (1, 0), 'constant')

    x_min, x_max = get_translation_area( x_translation )
    y_min, y_max = get_translation_area( y_translation )
    print(x_min, x_max)
    print(y_min, y_max)
    print('translating images..')
    aligned_images = np.zeros(( len(images),
                                int(width - x_min + x_max),
                                int(height - y_min + y_max)
                              ), np.uint8)

    x_t = 0
    y_t = 0
    for i in range(len(x_translation)):
        x_t += x_translation[i]
        y_t += y_translation[i]
        aligned_images[i] = get_image_in_canvas(loaded_images[i], aligned_images[i], width, height, x_min, y_min, x_t, y_t)


    print('sucessfull')
    #print(aligned_images.shape)

    return translation, error_list, aligned_images


def write_list_to_csv( list, filename, columns=None ):
    with open(filename, 'w', newline ='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        if not columns is None:
            write.writerow(columns)
        write.writerows(list)
    print('saved {}'.format(filename))

if __name__ == '__main__':

    # Read reference image
    refFilename = os.path.dirname(os.path.realpath(__file__)) + os.sep + "SEM Image - SliceImage - 001.tif"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = os.path.dirname(os.path.realpath(__file__)) + os.sep + "SEM Image - SliceImage - 002.tif"
    print("Reading image to align : ", imFilename);
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = os.path.dirname(os.path.realpath(__file__)) + os.sep + "aligned.tif"
    print("Saving aligned image : ", outFilename);
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n",  h)

    print("-------")
    print("DONE!")