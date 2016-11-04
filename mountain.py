#!/usr/bin/python
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
import sys
import random


# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale, 0, filtered_y)
    return filtered_y ** 2


# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range(max(y - thickness / 2, 0), min(y + thickness / 2, image.size[1] - 1)):
            image.putpixel((x, t), color)
    return image


# main program
#
(input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]

input_filename = "test_images/mountain.jpg"

# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
imsave('edges.jpg', edge_strength)

# You'll need to add code here to figure out the results! For now,
col_len = 0
row_len = len(edge_strength)
for row in edge_strength:
    col_len = len(row)
    break

print col_len, row_len
############################################
# Question 1 of the code
edge_list = []
for col in range(0, col_len):
    col_list = [edge_strength[row][col] for row in range(0, row_len)]
    for state_index, intensity in enumerate(col_list):
        if intensity == max(col_list):
            edge_list.append(state_index)
            break

ridge = edge_list


# Question 1 ends here and ridge is the answer
##############################################

# Calculate P(S_i|S_i-1) or P(S_i|S_i+1)
def trans_prob(curr, n_curr, cur_col_list):
    return (curr + n_curr) / (2.0 * len(cur_col_list))


# Calculate P(W|S_i)
def emis_prob(index, col):
    return (col[index] + (len(col) - index)) / (255.0 * 255 * len(col))


# Calculate P(S_i|S_i+1,S_i-1,W_i)
def posterior_prob(trans1, trans2, emis):
    return trans1 * trans2 * emis


# Gibb's sampling
for col in range(0, col_len):
    prob_array = []
    col_list = [int(edge_strength[row][col]) for row in range(0, row_len)]
    for i, row in enumerate(col_list):
        # for 1st column there is no previous state
        if col == 0:
            trans_prob_post = trans_prob(i, ridge[col + 1], col_list)
            print "transition prob is ",trans_prob_post
            emis_probs = emis_prob(i, col_list)
            prob_array.append(posterior_prob(1, trans_prob_post, emis_probs))

        # for last column there is no next state
        elif col == col_len - 1:
            trans_prob_prev = trans_prob(i, ridge[col - 1], col_list)
            emis_probs = emis_prob(i, col_list)
            prob_array.append(posterior_prob(1, trans_prob_prev, emis_probs))

        else:
            trans_prob_prev = trans_prob(i, ridge[col - 1], col_list)
            trans_prob_post = trans_prob(row, ridge[col + 1], col_list)
            emis_probs = emis_prob(i, col_list)
            prob_array.append(posterior_prob(trans_prob_prev, trans_prob_post, emis_probs))

        print prob_array

# ridge = [edge_strength.shape[0] / 2] * edge_strength.shape[1]

# output answer
imsave(output_filename, draw_edge(input_image, ridge, (255, 0, 0), 5))
