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
# programmed using python 3.7
# don't forget to install PIL (pip install Pillow)
#
#########################################################

import os, sys, math, cv2, time, re
import numpy as np
import tkinter as tk
from tkinter import filedialog
import csv
import multiprocessing

from scipy.ndimage import shift
import pandas as pd
import tifffile as tif
from PIL import Image

from skimage.filters import threshold_multiotsu
from skimage.segmentation import slic, mark_boundaries
from skimage import color

import bm3d

home_dir = os.path.dirname(os.path.realpath(__file__))
# import tiff_scaling script
ts_path = os.path.dirname( home_dir ) + '/tiff_scaling/'
ts_file = 'extract_tiff_scaling'
if ( os.path.isdir( ts_path ) and os.path.isfile( ts_path + ts_file + '.py' ) or os.path.isfile( home_dir + ts_file + '.py' ) ):
    if ( os.path.isdir( ts_path ) ): sys.path.insert( 1, ts_path )
    import extract_tiff_scaling as es
else:
    programInfo()
    print( 'missing ' + ts_path + ts_file + '.py!' )
    print( 'download from https://github.com/kleinerELM/tiff_scaling' )
    sys.exit()

#remove root windows
root = tk.Tk()
root.withdraw()

translation = []
error_list = []


def programInfo():
    print("#########################################################")
    print("# A Script to align FIB-Stacks                          #")
    print("#                                                       #")
    print("# © 2021 Florian Kleiner                                #")
    print("#   Bauhaus-Universität Weimar                          #")
    print("#   F. A. Finger-Institut für Baustoffkunde             #")
    print("#                                                       #")
    print("#########################################################")
    print()

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
    return h[1,2], h[0,2] # x, y

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.4
def get_image_homography_ORB(im1, im2, mask=None, filename=''):
    # Convert images to grayscale
    if len(im1.shape) == 3:
      im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
      im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect features and compute descriptors.
        #feature_detector = cv2.SURF_create()
    feature_detector = cv2.ORB_create(nfeatures=MAX_FEATURES)
    keypoints1, descriptors1 = feature_detector.detectAndCompute(im1, mask)
    keypoints2, descriptors2 = feature_detector.detectAndCompute(im2, mask)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score

    matches = sorted(matches, key = lambda x:x.distance, reverse=False)
    #matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    if len(points1) == 0 or len(points2) == 0: print("ERROR: no points found!!!")
    #h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0) # <- wtf is the 5.0 - some method??
    h, mask = cv2.estimateAffinePartial2D(points1, points2)# cv2.RANSAC)

    if not homography_is_translation(h) and filename != '':
        print("ERROR: the homography found is no translation!", h)
        # Draw top matches
        #imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        #cv2.imwrite(os.path.dirname(os.path.realpath(__file__)) + os.sep + filename + "_matches.tif", imMatches)

    return h, mask

def get_image_homography_SIFT(im1, im2, mask=None, filename=''):
    # Convert images to grayscale
    if len(im1.shape) == 3:
      im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
      im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb  = True
    sift = False#True
    # Detect features and compute descriptors.
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
    if len(points1) == 0 or len(points2) == 0: print("ERROR: no points found!!!")
    #print(points1, points2)
    #h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0) # <- wtf is the 5.0 - some method??
    h, mask = cv2.estimateAffinePartial2D(points1, points2)# cv2.RANSAC)

    if not homography_is_translation(h) and filename != '':
        print("ERROR: the homography found is no translation!", h)
        # Draw top matches
        #imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        #cv2.imwrite(os.path.dirname(os.path.realpath(__file__)) + os.sep + filename + "_matches.tif", imMatches)

    return h, mask

def alignImages(im1, im2, mask=None):
    print( "  aligning image using OpenCV2", flush=True )

    h, _ = get_image_homography_ORB(im1, im2, mask)
    #h, _ = get_image_homography_SIFT(im1, im2, mask)

    # Use homography
    height, width = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

def alignSecondaryImageSet():
    #height, width = im2.shape
    #im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg

def denoiseNLMCV2( image ):
    t1 = time.time()
    print( "  denoising image using NLM in OpenCV2", flush=True )
    denoised = np.zeros(image.shape, np.uint8) # empty image
    cv2.fastNlMeansDenoising( image,
                            denoised,
                            h=15,
                            templateWindowSize=7,
                            searchWindowSize=(15+1)
                            )
    print( ", took %f s" % (time.time() - t1) )

    return denoised

def denoiseBM3D( image ):
    t1 = time.time()
    print( "  denoising image using BM3D", flush=True )
    denoised = np.zeros(image.shape, np.uint8) # empty image
    cv2.fastNlMeansDenoising( image,
                            denoised,
                            h=15,
                            templateWindowSize=7,
                            searchWindowSize=(15+1)
                            )

    denoised = bm3d.bm3d(image, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.ALL_STAGES )

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
def get_image_translation(filename, im1, im2, mask, mask_full):
    error = None
    h, _ = get_image_homography_ORB(im1, im2, mask, filename=filename + '_a')
    #h, _ = get_image_homography_SIFT(im1, im2, mask, filename=filename + '_a')
    if not homography_is_translation(h):
        print('  WARNING: Homography for {} is not only a translation! Retrying full image'.format(filename), h)
        h, _ = get_image_homography_ORB(im1, im2, mask_full, filename=filename + '_b')
        #h, _ = get_image_homography_SIFT(im1, im2, mask, filename=filename + '_b')

        if not homography_is_translation(h):
            error = [filename, 'ERROR']
            print('  WARNING: Homography for {} REALLY is not only a translation! '.format(filename))
        else:
            error = [filename, 'WARNING']
    t_x, t_y = extract_translation_from_homography(h)

    return [filename, t_x, t_y], error

def image_processing_thread(filename, im1, im2, mask, mask_full, eq_hist):
    # preprocess the images fortranslation
    gauss_kernel = (3, 3)
    #gauss_kernel = (7, 7)
    im_1_denoised = cv2.GaussianBlur(im1, gauss_kernel, cv2.BORDER_DEFAULT)
    im_2_denoised = cv2.GaussianBlur(im2, gauss_kernel, cv2.BORDER_DEFAULT)
    # im_1_denoised = cv2.medianBlur(im1, 5)
    # im_2_denoised = cv2.medianBlur(im2, 5)

    if eq_hist:
        im_1_denoised = cv2.equalizeHist(im_1_denoised)
        im_2_denoised = cv2.equalizeHist(im_2_denoised)

    return get_image_translation(filename, im_1_denoised, im_2_denoised, mask, mask_full)

# singlethreaded processing
def process_translation_of_folder_singlecore(images, loaded_images, mask_size=0.9, eq_hist=True ):
    print('processing image stack singlethreaded:')
    global translation
    global error_list

    im1 = None
    im2 = None
    mask = None
    mask_full = None
    for i, filename in enumerate( images ):
        print( " processing {} ({} / {}):".format(filename, i+1, len(images)) )
        im2 = im1
        im1 = loaded_images[i]
        if not im2 is None:
            translation_line, error_list_line = image_processing_thread(filename, im1, im2, mask, mask_full, eq_hist)

            if error_list_line is not None:  error_list.append( error_list_line )
            #translation.append(translation_line)
            translation[images.index(translation_line[0])] = translation_line
        else:
            mask      = get_centered_mask(im1, mask_size=mask_size)
            mask_full = get_centered_mask(im1, mask_size=1)

    return translation, error_list

# multithreaded processing
def store_result(result):
    global translation
    global error_list
    global images

    result = list(result)
    translation[images.index(result[0][0])] = result[0]
    if not result[1] is None:  error_list.append( result[1] )

def process_translation_of_folder_multicore(images, loaded_images, mask_size=0.9, eq_hist=True):
    print('processing image stack ({} images {}) multithreaded:'.format(len(images), len(loaded_images)))
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
        print( " processing {} ({} / {})".format(image, i+1, len(images)) )
        im2 = im1
        im1 = loaded_images[i]
        if not im2 is None:
            #translation_line, error_list_line = image_processing_thread(filename, im1, im2, mask, mask_full)
            pool.apply_async(image_processing_thread, args=(image, im1, im2, mask, mask_full, eq_hist), callback = store_result)
        else:
            mask      = get_centered_mask(im1, mask_size=mask_size)
            mask_full = get_centered_mask(im1, mask_size=1)

    pool.close()
    pool.join()

    # results come in unsorted -> sort
    return translation, error_list

def get_find_border(arr, threshold, flip=False):
    if flip: arr = np.flip(arr)
    for i, val in enumerate( arr ):
        if val > threshold:
            return i
    print( 'ERROR: no value above the threshold of {} found'.format(threshold) )
    print(arr)

def auto_crop_stack( image_stack, threshold=10 ):
    print(' trying to auto crop image stack (threshold={})'.format(threshold))
    image_cnt, height, width = image_stack.shape
    z_mean = np.mean(image_stack, axis = 0)
    x_mean = np.mean(z_mean, axis = 0)
    y_mean = np.mean(z_mean, axis = 1)
    pad_left   =get_find_border(x_mean, threshold)
    pad_right  =get_find_border(x_mean, threshold, flip=True)
    pad_top    =get_find_border(y_mean, threshold)
    pad_bottom =get_find_border(y_mean, threshold, flip=True)
    print('identified padding (x_l: {}, x_r:{}, y_l: {}, y_r: {})'.format(pad_left, pad_right, pad_top, pad_bottom))
    cropped_stack = image_stack[0:image_cnt, pad_top:height-pad_bottom, pad_left:width-pad_right]
    return cropped_stack

# wrapper for the multiprocessing of the image arrangement and the NLM processing
def get_image_in_canvas_mp_wrapper(i, image, canvas, x_min, y_min, x_t, y_t, do_nlm):
    canvas = get_image_in_canvas(image, canvas, x_min, y_min, x_t, y_t)
    if do_nlm:
        #canvas = denoiseNLMCV2(canvas)
        canvas = denoiseBM3D(canvas)
    return i, canvas

# result processing after multithreaded processing
aligned_images = None
def update_image_list(result):
    global aligned_images
    print('asdf')
    result = list(result)
    aligned_images[result[0]] = result[1]

# apply the translation to the image and place it on an empty canvas
def get_image_in_canvas(image, canvas,
                        x_min, y_min,
                        x_t, y_t):
    d_x = -x_min+x_t
    d_y = -y_min+y_t
    skimageshift = False
    if skimageshift:
        im =  np.zeros(( int(image.shape[0] + math.ceil(d_x)),
                         int(image.shape[1] + math.ceil(d_y))
                       ), np.uint8)
        im[0:image.shape[0], 0:image.shape[1]] = image
        #print(canvas.shape, im.shape, d_x, d_y)
        canvas[0:im.shape[0], 0:im.shape[1]] = shift(im, shift=(d_x, d_y), mode='constant')
    else:
        #print(int(round(d_x)),':',int(round(image.shape[0]+d_x)), int(round(d_y)),':',int(round(image.shape[1]+d_y)))
        canvas[int(round(d_x)):int(round(image.shape[0]+d_x)), int(round(d_y)):int(round(image.shape[1]+d_y))] = image

    return canvas

# do_nlm      : bool | will apply non local mean filter if True
# x and y seems to be swappedin parts
def create_3D_stack(translation, loaded_images, do_nlm=False, first_x_offset=None, first_y_offset=None):
    global aligned_images

    print("Create 3D image stack with corrected image translation")
    width, height = loaded_images[0].shape
    im_cnt = len(loaded_images)

    if len(translation) == 0:
        sys.exit('  - ERROR: no translation data found!')

    #if len(translation)+1 != len(loaded_images):
    #    sys.exit('  - ERROR: translation matrix ({}) does not match the image stack size ({})'.format( len(translation)+1, len(loaded_images)))

    arr = np.array(translation)
    b   = np.delete( np.delete(arr, 0, axis=1).astype(np.float) , 0, axis=0)
    #print(b)
    # np.pad() adds a 0 for the first image
    x_translation = np.pad(b[:,0], (1, 0), 'constant')# if first_x_offset <= 0 else np.concatenate(([float(first_x_offset)],b[:,0]))
    y_translation = np.pad(b[:,1], (1, 0), 'constant')# if first_y_offset <= 0 else np.concatenate(([float(first_y_offset)],b[:,1]))
    #print(x_translation, y_translation)
    if not first_x_offset is None:
        print('found first_x_offset: {}'.format(first_x_offset))
        #x_translation = b[:,0]
        #x_translation[0] += float(first_x_offset)
        y_translation = b[:,1]
        y_translation[0] += float(first_x_offset)
    if not first_y_offset is None:
        print('found first_y_offset: {}'.format(first_y_offset))
        #y_translation = b[:,1]
        #y_translation[0] += float(first_y_offset)
        x_translation = b[:,0]
        x_translation[0] += float(first_y_offset)

    x_min, x_max = get_translation_area( x_translation )
    y_min, y_max = get_translation_area( y_translation )
    #print(x_min, x_max)
    #print(y_min, y_max)

    print('  - allocating 3D image space..')
    aligned_images = np.zeros(( im_cnt,
                                math.ceil(width - x_min + x_max),
                                math.ceil(height - y_min + y_max)
                              ), np.uint8)
    print('  - translating and denoising images..')

    coreCount = multiprocessing.cpu_count()
    processCount = (coreCount - 1) if coreCount > 1 else 1
    #pool = multiprocessing.Pool(processCount)

    #print(np.mean(aligned_images))
    if im_cnt > len(y_translation) or im_cnt > len(x_translation): print( 'ERROR! There are more images ({}) than translations ({} and {})!'.format( im_cnt, len(y_translation),len(x_translation) ) )
    else: print( 'image count  ({}) equals translations ({} and {})!'.format( im_cnt, len(y_translation),len(x_translation) ) )
    x_t = 0
    y_t = 0
    for i in range(im_cnt):
        x_t += x_translation[i]
        y_t += y_translation[i]
        #print(i, x_t, y_t)
        i, aligned_images[i] = get_image_in_canvas_mp_wrapper(i, loaded_images[i], aligned_images[i], x_min, y_min, x_t, y_t, do_nlm)
        #pool.apply_async(get_image_in_canvas_mp_wrapper, args=(i, loaded_images[i], aligned_images[i], x_min, y_min, x_t, y_t, do_nlm), callback = update_image_list)

    #pool.close()
    #pool.join()

    #print(np.mean(aligned_images))
    return aligned_images

#def has_n_digit_numbers(inputString, n=3):#
#    needle = ''
#    for i in range(n):
#        needle += r'\d'
#    return re.search(needle, inputString)

def has_1_digit_numbers(inputString):
    s = re.search(r'\d', inputString.split()[-1])
    if s: return s.group(0)
    else: return False

def has_2_digit_numbers(inputString):
    s = re.search(r'\d\d', inputString.split()[-1])
    if s: return s.group(0)
    else: return False

def has_3_digit_numbers(inputString):
    s = re.search(r'\d\d\d', inputString.split()[-1])
    if s: return s.group(0)
    else: return False

def has_4_digit_numbers(inputString):
    s = re.search(r'\d\d\d\d', inputString.split()[-1])
    if s: return s.group(0)
    else: return False

def has_5_digit_numbers(inputString):
    s = re.search(r'\d\d\d\d\d', inputString.split()[-1])
    if s: return s.group(0)
    else: return False

def load_image_set(folder, limit=[]):
    # the image numbering only uses 3 digits in basic settings
    # if there are more than 1000 images, the order is messed up
    # therefore in the following the script tries to check if there are
    # "malformatted" filenames
    image_paths = {}
    image_path_list = []
    for file in os.listdir(folder):
        if ( file.endswith(".tif") or file.endswith(".TIF")):
            d = has_5_digit_numbers(file)
            if d: image_paths[int(d)] = file
            else:
                d = has_4_digit_numbers(file)
                if d: image_paths[int(d)] = file
                else:
                    d = has_3_digit_numbers(file)
                    if d: image_paths[int(d)] = file
                    else:
                        d = has_2_digit_numbers(file)
                        if d: image_paths[int(d)] = file
                        else:
                            d = has_1_digit_numbers(file)
                            if d: image_paths[int(d)] = file
                            else: image_paths[len(image_paths)+1] = file

    # make sure the order is correct:
    image_paths = {key:image_paths[key] for key in sorted(image_paths.keys())}
    image_path_list = list(image_paths.values())

    if isinstance( limit, list ) and len(limit) == 2: image_path_list = image_path_list[ limit[0] : limit[1] ]
    print('loading {} images...'.format(len(image_path_list)))

    loaded_images = []
    for image in image_path_list:
        loaded_images.append( cv2.imread(folder + os.sep + image, cv2.IMREAD_GRAYSCALE) )

    return image_path_list, loaded_images

def load_translation_csv( translation_csv, expected_image_count ):
    translation = []
    if os.path.isfile(translation_csv):
        print( "Found existing translation csv, loading...")
        with open(translation_csv) as csv_file:
            csv_reader = csv.reader(csv_file)
            for i, row in enumerate(csv_reader):
                if i > 0:
                    translation.append([row[0],float(row[1]),float(row[2])])
        if len(translation) != expected_image_count-1:
            print("{} contains {} lines, while {} lines were expected".format(translation_csv, len(translation), expected_image_count-1))

    return translation

def save_translation_csv( translation, translation_csv):
    print('saving translation matrix to {}'.format(translation_csv))
    write_list_to_csv(translation, translation_csv, ['file', 'transl_x', 'transl_y'])

def check_folder_structure(folder):
    eds_elements = {}
    if os.path.basename(os.path.normpath(folder)) == 'EDS Export':
        BSE_image_path = ''
        for file in os.listdir(folder):
            if len(file) <= 2:
                eds_elements[file] = folder + file + os.sep
            elif file == 'Images':
                BSE_image_path = folder + file + os.sep

        print('identified {} elements'.format(len(eds_elements)))
        folder = BSE_image_path

    return folder, eds_elements

# main process function
# do_nlm      : bool | will apply non local mean filter if True
# mask_size   : float 0.0 - 1.0 | defines the size of the mask, which defines the area used for feature detection
# eq_hist     : bool | improve histogram for feature detection
# crop_thresh : in 0 - 255 | brightness value used for the auto cropper (disable with 0)
def process_translation_of_folder(folder=None, multicore = True, do_nlm=False, mask_size=0.9, eq_hist=True, crop_thresh=0, limit=False ):
    global translation
    global error_list
    global aligned_images

    if folder is None:
        folder = filedialog.askdirectory(title='Please select the image / working directory')

    # load images
    images, loaded_images = load_image_set( folder, limit )
    im_cnt = len(loaded_images)

    # load scaling
    #scaling = es.getImageJScaling( images[0], folder, verbose=True )
    scaling = es.autodetectScaling( images[0], folder, verbose=True )
    scaling['y'] = scaling['y']/math.cos(38) #fix distortion due to FIB-geometry
    # load translation table
    translation_csv = folder + os.sep + 'translations.csv'
    translation = load_translation_csv( translation_csv, im_cnt )

    # process translation table
    if len(translation) != im_cnt-1:
        translation = []
        for f in images:
            translation.append([f, 0.0, 0.0])
        print("processing {} images...".format(im_cnt))

        loaded_images_resized = []
        f = 1000/loaded_images[1].shape[0]
        if f < .95:
            print('  reducing image size for translation calculations')
            for i, image in enumerate( loaded_images ):
                loaded_images_resized.append( cv2.resize(image, None, fx=f, fy=f) )

            if multicore:
                process_translation_of_folder_multicore( images, loaded_images_resized, mask_size, eq_hist )
            else:
                process_translation_of_folder_singlecore( images, loaded_images_resized, mask_size, eq_hist )

            for i, translation_line in enumerate(translation):
                translation[i][1] = translation_line[1]/f
                translation[i][2] = translation_line[2]/f
        else:
            if multicore:
                process_translation_of_folder_multicore( images, loaded_images, mask_size, eq_hist )
            else:
                process_translation_of_folder_singlecore( images, loaded_images, mask_size, eq_hist )

        print("processing basic data...")
        #translation = sorted( translation )

        #save results
        save_translation_csv( translation, translation_csv)

        if len(error_list) > 0:
            write_list_to_csv(sorted(error_list),  folder + os.sep + 'error_list.csv',   ['file_b', 'serverity'])

    # align images
    aligned_images = create_3D_stack(translation, loaded_images, do_nlm)

    print('saving images..')
    save_path = folder + 'aligned' + os.sep
    if not os.path.isdir(save_path): os.makedirs(save_path)
    stack_fn = save_path + "aligned_stack_({}).tif".format(im_cnt)
    tif.imsave(stack_fn, aligned_images, bigtiff=True)
    print('saved "{}"'.format(stack_fn))
    if crop_thresh > 0:
        aligned_images = auto_crop_stack( aligned_images, threshold=crop_thresh )
        cropped_fn = save_path + 'cropped.tif'
        tif.imsave(cropped_fn, aligned_images, bigtiff=True)
        print('saved "{}"'.format(cropped_fn))

    print('sucessfull')

    return translation, error_list, aligned_images, loaded_images, scaling

def write_list_to_csv( list, filename, columns=None ):
    with open(filename, 'w', newline ='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        if not columns is None:
            write.writerow(columns)
        write.writerows(list)
    print('saved {}'.format(filename))

def get_axis_correction_list( correction_dict, z_slice_count ):
    z_list = list(correction_dict) # get the keys from the dict (z coordinate)

    min_xy = min(correction_dict.values())
    for z, xy in correction_dict.items():
        correction_dict[z]= xy - min_xy

    #print(y_correction_dict)
    #print('-'*20)
    next_index = 0
    correction_list = []
    last_value = 0
    for z, xy in correction_dict.items():
        index = z_list.index(z)
        next_index = index + 1
        if next_index < len(z_list):
            #print(z_list[index], z_list[next_index])
            start_pos = z_list[index] if z_list[index] >= 0 else 0
            end_pos = z_list[next_index] if z_list[next_index] < z_slice_count else z_slice_count
            index_range = end_pos-start_pos

            #print(start_pos, end_pos ,':')
            for pos in range(start_pos, end_pos):
                result = correction_dict[z_list[index]]+((correction_dict[z_list[next_index]]-correction_dict[z_list[index]])/index_range*(pos-start_pos))
                correction_list.append(result - last_value)
                #print(last_y, result)
                last_value = result

        # if the last item in z_list is smaller than z_slice_count-1
        elif z_list[index] < z_slice_count:
            for pos in range(z_list[index], z_slice_count):
                # fill with the last available correction
                correction_list.append(correction_dict[z_list[index]])

    return correction_list

#########################################################
#
# EDS functions
#
#########################################################
def get_full_img_translation( translation, search_pos, verbose = False ):
    x_t = 0
    y_t = 0
    min_x = 0
    min_y = 0
    if verbose: print('~'*20)
    for step in translation:
        x_t += step[2]
        y_t += step[1]
        if min_x > x_t: min_x = x_t
        if min_y > y_t: min_y = y_t

    x_t = 0
    y_t = 0
    for pos, step in enumerate(translation[0:search_pos]):
        if verbose: print(pos, (x_t, y_t), (min_x-x_t, min_y-y_t))
        x_t += step[2]
        y_t += step[1]

    if verbose:
        print(pos+1, (min_x, x_t), (min_y, y_t), (min_x-x_t, min_y-y_t))
        print('this translation', step)

    return (min_x, x_t), (min_y, y_t)

# preprocess the raw eds images - denoising, contrast enhancing and segmentation
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(1,1))
def preprocess_eds_image(img):
    _, tmp = cv2.threshold( clahe.apply(cv2.medianBlur(img, 9)) , 50, 255, cv2.THRESH_TOZERO)#THRESH_TRUNC)#THRESH_BINARY
    tmp = clahe.apply( tmp )

    return clahe.apply( tmp )

# process an EDS element image stack
# eds_elements      - List of elements. Eg: ['Ca','O','Si']
# selected_element  - string as in eds_elements - Eg: 'Ca'
# se_translation    - Translation array [[filename, t_x, t_y],[...],...] eg from the SE dataset
# eds_x_offset
# eds_y_offset
def process_element(eds_elements, selected_element, se_translation,  eds_x_offset, eds_y_offset, limit=[]):
    if selected_element in eds_elements.keys():
        print('-'*20)
        print('selected element is {}'.format(selected_element))
    else:
        print('ERROR: element {} not found in the dataset'.format(selected_element))

    #eds_each_nth_slice = 10

    folder = eds_elements[selected_element]
    images, loaded_images = load_image_set(folder, limit)
    print(images)

    image_numbering = []
    for i, image in enumerate(loaded_images):
        image_numbering.append( int(images[i].split(' ')[-1].split('.')[0]) )

    #sort lists
    images = [x for _, x in sorted(zip(image_numbering, images))]
    loaded_images = [x for _, x in sorted(zip(image_numbering, loaded_images))]
    corrected_images = loaded_images
    if len(images) > 1:
        shapes = []
        shape_counts = []
        image_numbering = []
        for i, image in enumerate(loaded_images):
            shape = image.shape
            if not shape in shapes:
                shapes.append(shape)
                shape_counts.append(0)
            shape_counts[-1] +=1

        if len(shapes) > 1:
            print( ' Found multiple shapes!' )
            for i, shape in enumerate(shapes):
                print('  found {} images with this shape:'.format(shape_counts[i]), shape)

        # TODO select the indended stack?
        ignore_first_n_images = 18#shape_counts[0]-1 if len(shapes) > 1 else 0
        ignore_last_n_images = shape_counts[-2] if len(shapes) == 3 else 0
        x_offset = 0
        y_offset = 0

        # correcting position of the eds image relative to the SE images
        if ignore_first_n_images > 0:
            t_x, t_y  = get_full_img_translation(se_translation, ignore_first_n_images)
            print('  - eds x translation: {} - {} + {} '.format(eds_x_offset, t_x[0], t_x[1]))
            print('  - eds y translation: {} - {} + {} '.format(eds_y_offset, t_y[0], t_y[1] ))
            x_offset = eds_x_offset - t_x[0] + t_x[1]
            y_offset = eds_y_offset - t_y[0] + t_y[1]
            print('  - final translation {} and {}'.format( x_offset, y_offset ) )
            print( - t_y[0] + t_y[1] )

        if ignore_last_n_images == 0:
            img_stack = loaded_images[ignore_first_n_images:]
            # select images to be displayed
            selected_translation = se_translation[ignore_first_n_images-1 :]
            #selected_translation[0] = [selected_translation[0][0], selected_translation[0][1]+eds_x_offset, selected_translation[0][2]+eds_y_offset]
        else:
            loaded_images[ignore_first_n_images : ignore_last_n_images]
            # select images to be displayed
            selected_translation = se_translation[ignore_first_n_images-1 : ignore_last_n_images]
            #selected_translation[0] = [selected_translation[0][0], selected_translation[0][1]+eds_x_offset, selected_translation[0][2]+eds_y_offset]

        print(selected_translation[0])
        #selected_translation[0] = [selected_translation[0][0], -50, -10]#y_offset]
        #selected_translation[0] = [selected_translation[0][0], selected_translation[0][1]+y_offset, selected_translation[0][2]+x_offset]
        #print(selected_translation[0])

        # process eds images to reduce noise
        print(' denoising images and enhance contrast')
        for i, img in enumerate(img_stack):
            img_stack[i] = preprocess_eds_image(img)
            #_, img_stack[i] = cv2.threshold( clahe.apply(cv2.medianBlur(img, 9)) , 50, 255, cv2.THRESH_TOZERO)#THRESH_TRUNC)
            #img_stack[i] = clahe.apply(img_stack[i])

        #corrected_images = create_3D_stack(selected_translation, img_stack, do_nlm=False, first_x_offset =49, first_y_offset =471 )
        corrected_images = create_3D_stack(selected_translation, img_stack, do_nlm=False, first_x_offset=x_offset, first_y_offset=y_offset )#52 )#x_offset = 471  y_offset should be 49-52??

        cs = corrected_images.shape
        ecs = (ignore_first_n_images + cs[0] + ignore_last_n_images, cs[1], cs[2])

        if len(shapes) > 1:
            if ignore_first_n_images > 1:
                temp = np.zeros(ecs)

                if ignore_last_n_images == 0:
                    temp[ignore_first_n_images:,:,:] = corrected_images
                else:
                    temp[ignore_first_n_images:-ignore_last_n_images,:,:] = corrected_images

                corrected_images = temp

    return corrected_images

#########################################################
#
# processing of EDX data using superpixels via SLIC
#
#########################################################
elements_SLIC      = {}
elements_mean_SLIC = {}

# find superpixel
def get_SLIC_segments( img_clahe, element=None, i=None, n_segments=500, compactness=0.2 ):
    segments_slic = slic(cv2.medianBlur(cv2.equalizeHist(img_clahe), ksize=5), n_segments=n_segments, compactness=compactness, sigma=1, start_label=1)

    # get mean color of superpixel
    return [segments_slic, element, i ]

def SLIC_segment_processing(result):
    global elements_SLIC

    segments_slic = result[0]
    element       = result[1]
    i             = result[2]

    elements_SLIC[element][i]      = segments_slic

# get mean color of the superpixel
def get_SLIC_mean_Image( img_clahe, segments_slic, element=None, i=None, threshold=50 ):
    mean_slic = cv2.equalizeHist(np.uint8( color.rgb2gray( color.label2rgb(segments_slic, img_clahe, kind='avg', bg_label=0) ) * 255))

    #remove very low values
    mean_slic[mean_slic<threshold] = 0

    return [mean_slic, element, i ]

def SLIC_mean_processing(result):
    global elements_mean_SLIC

    mean_slic     = result[0]
    element       = result[1]
    i             = result[2]

    elements_mean_SLIC[element][i] = mean_slic

#########################################################
#
# This is propably outdated.
#
#########################################################
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