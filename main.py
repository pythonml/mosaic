import ctypes
import re
import numpy as np
from numpy import ctypeslib
import os
import cv2
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
from scipy.spatial.distance import euclidean
from tqdm import tqdm

IMG_DIR = "images"
RATIO = 10

def resize(im, tile_row, tile_col):
    shape_row = im.shape[0]
    shape_col = im.shape[1]
    shrink_ratio = min(shape_row/tile_row, shape_col/tile_col)
    resized = cv2.resize(im, (int(shape_col/shrink_ratio)+1, int(shape_row/shrink_ratio)+1), interpolation=cv2.INTER_CUBIC)
    result = resized[:tile_row, :tile_col,:]
    return result

def img_distance(im1, im2):
    if im1.shape != im2.shape:
        msg = "shapes are different {} {}".format(im1.shape, im2.shape)
        raise Exception(msg)
    array1 = im1.flatten()
    array2 = im2.flatten()
    dist = euclidean(array1, array2)
    return dist

def load_all_images(tile_row, tile_col):
    img_dir = IMG_DIR
    filenames = os.listdir(img_dir)
    result = []
    print(len(filenames))
    for filename in tqdm(filenames):
        if not re.search(".jpg", filename, re.I):
            continue
        try:
            filepath = os.path.join(img_dir, filename)
            im = cv2.imread(filepath)
            row = im.shape[0]
            col = im.shape[1]
            im = resize(im, tile_row, tile_col)
            result.append(np.array(im))
        except Exception as e:
            msg = "error with {} - {}".format(filepath, str(e))
            print(msg)
    return np.array(result, dtype=np.uint8)

def find_closest_image(q, shared_tile_images, tile_images_shape, shared_result, img_shape, tile_row, tile_col):
    tile_images_array = np.frombuffer(shared_tile_images, dtype=np.uint8)
    tile_images = tile_images_array.reshape(tile_images_shape)
    while True:
        [row, col, im_roi] = q.get()
        print(row)
        min_dist = float("inf")
        min_img = None
        for im in tile_images:
            dist = img_distance(im_roi, im)
            if dist < min_dist:
                min_dist = dist
                min_img = im
        im_res = np.frombuffer(shared_result, dtype=np.uint8).reshape(img_shape)
        im_res[row:row+tile_row,col:col+tile_col,:] = min_img
        q.task_done()

def get_tile_row_col(shape):
    if shape[0] >= shape[1]:
        return [120, 90]
    else:
        return [90, 120]

def generate_mosaic(infile, outfile):
    img = cv2.imread(infile)
    tile_row, tile_col = get_tile_row_col(img.shape)
    img_shape = list(img.shape)
    img_shape[0] = int(img_shape[0]/tile_row) * tile_row * RATIO
    img_shape[1] = int(img_shape[1]/tile_col) * tile_col * RATIO
    img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)
    print(img_shape)
    im_res = np.zeros(img_shape, np.uint8)
    tile_images = load_all_images(tile_row, tile_col)
    shared_tile_images = mp.sharedctypes.RawArray(ctypes.c_ubyte, len(tile_images.flatten()))
    tile_images_shape = tile_images.shape
    np.copyto(np.frombuffer(shared_tile_images, dtype=np.uint8).reshape(tile_images_shape), tile_images)
    shared_result = mp.sharedctypes.RawArray(ctypes.c_ubyte, len(im_res.flatten()))

    q = mp.JoinableQueue()
    for i in range(5):
        p = mp.Process(target=find_closest_image,
            args=(q, shared_tile_images, tile_images_shape, shared_result, img_shape, tile_row, tile_col),
            daemon=True)
        p.start()
        print("started process")

    for row in range(0, img_shape[0], tile_row):
        for col in range(0, img_shape[1], tile_col):
            roi = img[row:row+tile_row,col:col+tile_col,:]
            q.put([row, col, roi])

    q.join()
    cv2.imwrite(outfile, np.frombuffer(shared_result, dtype=np.uint8).reshape(img_shape))

if __name__ == "__main__":
    generate_mosaic("test.jpg", "out.jpg")
