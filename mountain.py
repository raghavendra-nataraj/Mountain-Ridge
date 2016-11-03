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


# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    imsave('b.jpg', grayscale)
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale, 0, filtered_y)
    imsave('c.jpg', filtered_y ** 2)
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

# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
# print edge_strength
imsave('edges.jpg', edge_strength)

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.

row_len = len(edge_strength)
for row in edge_strength:
    col_len = len(row)
    break

# print row_len, col_len
edge_list = []
for col in range(0, col_len):
    a = [edge_strength[row][col] for row in range(0, row_len)]
    for x, y in enumerate(a):
        if y == max(a):
            edge_list.append(x)
            break

ridge = edge_list
# ridge = [edge_strength.shape[0] / 2] * edge_strength.shape[1]

# output answer
imsave(output_filename, draw_edge(input_image, ridge, (255, 0, 0), 5))
