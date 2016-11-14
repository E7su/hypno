#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import pandas
import pylab
import numpy as np
from skimage import img_as_float
from skimage.io import imread, imsave
from sklearn.cluster import KMeans

from answer import create_answer_file

# Download parrots.jpg picture. Convert the image, bringing all the values ​​in the interval from 0 to 1.
# You can use the function img_as_float of skimage module.
image = img_as_float(imread('parrots.jpg'))
pylab.imshow(image)  # show image
w, h, d = image.shape

# After these actions the variable image will contain numpy-array with size n * m * 3,
# where n and m correspond to the dimensions of the image, and 3 represents RGB representation format.
# Create a matrix-signs objects:
# each pixel is characterized by three coordinates - the intensity values ​​in the RGB space.
pixels = pandas.DataFrame(np.reshape(image, (w * h, d)), columns=['R', 'G', 'B'])


# Run the algorithm K-Means with parameters init = 'k-means ++' and random_state = 241.
# After clustering all of the pixels classified into one cluster, try to fill in two ways:
# the median and the average color of the cluster.
def cluster(pixels, n_clusters=8):
    print 'Clustering: ' + str(n_clusters)

    pixels = pixels.copy()
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)
    pixels['cluster'] = model.fit_predict(pixels)

    # ------------------------//mean//------------------------
    means = pixels.groupby('cluster').mean().values
    mean_pixels = [means[c] for c in pixels['cluster'].values]
    mean_image = np.reshape(mean_pixels, (w, h, d))
    imsave('mean_parrots_' + str(n_clusters) + '.jpg', mean_image)

    # ---------------------//medians//------------------------
    medians = pixels.groupby('cluster').median().values
    median_pixels = [medians[c] for c in pixels['cluster'].values]
    median_image = np.reshape(median_pixels, (w, h, d))
    imsave('median_parrots_' + str(n_clusters) + '.jpg', median_image)

    return mean_image, median_image


# Measure the quality of the resulting segmentation using metrics PSNR (Peak signal-to-noise ratio).
def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return 10 * math.log10(float(1) / mse)


# Find the minimum number of clusters in which the PSNR value is higher than 20 (you can see no more than 20 clusters,
# but do not forget to consider both ways of filling a pixel cluster).
# This number will be the answer to this exercize.
for n in xrange(1, 21):
    mean_image, median_image = cluster(pixels, n)
    psnr_mean, psnr_median = psnr(image, mean_image), psnr(image, median_image)
    print psnr_mean, psnr_median

    if psnr_mean > 20 or psnr_median > 20:
        create_answer_file(1, n)
        break
