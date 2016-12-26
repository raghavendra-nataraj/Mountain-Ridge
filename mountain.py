#!/usr/bin/python
#
# Mountain ridge finder

# Your names and user ids:
#   Sarvothaman Madhavan    -   madhavas
#   Raghavendra Nataraj     -   natarajr
#   Prateek Srivastava      -   pratsriv

# Based on skeleton code by D. Crandall, Oct 2016
#
'''
Read Me
Output Color :
		Simplified				-	red
		MCMC					-	blue
		MCMC with user input			-	green

Emission Probabilty : It is the edge strength normalised over the column.(Assigning higher rows with more probability is misleading, so i have used uniform prior probability)
Transision probabitlty : It is the transition between row. Equal rows get higher probability and rows far away get less probability. I have normalised over all values of column. 

In simplified we have taken only the max of the edge strength, because P(S) is constant. (i.e Apriori, each row is 
equally likely to contain the mountain ridge)
In MCMC we define the probability based on edge strength and the transition from next column and previous column. 

If more importance(Weight) is given to edge strength, then images  in which the moutains are clear get good results but images in which mountains are a little out of scope(farther or a little faded i.e. having less intensity), get bad results. 
If more importance(Weight) is given to transistion probability then a single pixel's wrong calculation propogates over to the next points. So the results goes bad for clear mountains. 

The user input does not help to a great extent in finding the ridges. It helps in finding a few points but it normalizes as we progress thereby giving results very similar to MCMC. 

Note : Time for each image takes around 45 seconds to complete all three methodologies.
'''
from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
from copy import deepcopy
import sys
import math
import random
import operator
from collections import Counter

global col_len, row_len
helper_list = {}


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

def draw_edge1(image,x, y, color, thickness):
    for t in range(max(y - thickness / 2, 0), min(y + thickness / 2, image.size[1] - 1)):
        image.putpixel((x, t), color)
    return image

# main program
#
(input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]

input_filename = sys.argv[1]
x_axis = int(sys.argv[4])
y_axis = int(sys.argv[3])

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

# print col_len, row_len
############################################
# Question 1 of the code
def simple():
    edge_list = []
    for col in range(0, col_len):
        col_list = [edge_strength[row][col] for row in range(0, row_len)]
        for state_index, intensity in enumerate(col_list):
            if intensity == max(col_list):
                edge_list.append(state_index)
                break
    return edge_list


def hmm_viterbi():
    viterbi = {}
    viterbi[0] = {}
    col_list = [int(edge_strength[row][0]) for row in range(0, row_len)]
    length = len(col_list)
    for row in range(0,row_len):
        viterbi[0][row] = (float(emis_p(0,row)),[row])
    for col in range(1,col_len):
        viterbi[col] = {}
        col_list = [int(edge_strength[row][col]) for row in range(0, row_len)]
        for row in range(0,row_len):
            results =  []
            for row_k in range(0,row_len):
                tmpval = viterbi[col-1][row_k][0] + trans_p(row_k,row)
                results.append((tmpval,viterbi[col-1][row_k][1]))
            value,ridgev = min(results,key=operator.itemgetter(0))
            ridgetmp = deepcopy(ridgev)
            ridgetmp.append(row)
            viterbi[col][row] = ((value + emis_p(col,row)),ridgetmp)
    results =  []
    for row in range(0,row_len):
        results.append(viterbi[col_len-1][row])
    value,ridgev = min(results,key=operator.itemgetter(0))
    return ridgev


def hmm_viterbi_user():
    viterbi = {}
    viterbi[0] = {}
    col_list = [int(edge_strength[row][0]) for row in range(0, row_len)]
    length = len(col_list)
    for row in range(0,row_len):
        viterbi[0][row] = (float(emis_pu(0,row)),[row])
    for col in range(1,col_len):
        viterbi[col] = {}
        col_list = [int(edge_strength[row][col]) for row in range(0, row_len)]
        for row in range(0,row_len):
            results =  []
            for row_k in range(0,row_len):
                tmpval = viterbi[col-1][row_k][0] + trans_pu(row_k,row)
                results.append((tmpval,viterbi[col-1][row_k][1]))
            value,ridgev = min(results,key=operator.itemgetter(0))
            ridgetmp = deepcopy(ridgev)
            ridgetmp.append(row)
            viterbi[col][row] = ((value + emis_pu(col,row)),ridgetmp)
    results =  []
    for row in range(0,row_len):
        results.append(viterbi[col_len-1][row])
    value,ridgev = min(results,key=operator.itemgetter(0))
    return ridgev

# print col_len
ridge = [0] * col_len
# print ridge

# print max(ridge),len(ridge)

# Question 1 ends here and ridge is the answer
##############################################

def trans_p(curr, n_curr):
    return t_prob[curr][n_curr]


# Calculate P(W|S_i)
def emis_p(col,row):
    return e_prob[col][row]

def trans_pu(curr, n_curr):
    return t_prob_u[curr][n_curr]


# Calculate P(W|S_i)
def emis_pu(col,row):
    return e_prob_u[col][row]


# Calculate P(S_i|S_i-1) or P(S_i|S_i+1)
def trans_prob(curr, n_curr, length):
    weight = 20
    return (length - abs(curr - n_curr))**weight

# Calculate P(W|S_i)
def emis_prob(index, col):
    return (col[index])* ((len(col)-index))


def trans_prob_user(curr, n_curr, length):
    weight = 20
    return (((1) *  (length - abs(y_axis - curr))**weight))


# Calculate P(W|S_i)
def emis_prob_user(index, col):
    return (col[index]**(0.90))* ((len(col)-abs(index-y_axis))**1.5)


e_prob = {}
t_prob = {}
e_prob_u = {}
t_prob_u = {}

for row in range(0,row_len):
    t_prob[row]={}
    t_prob_u[row]={}
    row_sum = 0
    row_sumu = 0
    for rown in range(0,row_len):
        t_prob[row][rown] = trans_prob(row,rown,row_len)
        row_sum+=t_prob[row][rown]
        t_prob_u[row][rown] = trans_prob_user(row,rown,row_len)
        row_sumu+=t_prob_u[row][rown]
    for rown in range(0,row_len):
        t_prob[row][rown] = math.log(1/(t_prob[row][rown]/float(row_sum)))
        t_prob_u[row][rown] = math.log(1/(t_prob_u[row][rown]/float(row_sumu)))    

# Gibb's sampling
def calculate_prob():
    for col in range(0, col_len):
        e_prob[col] = {}
        e_prob_u[col] = {}
        col_list = [int(edge_strength[row][col]) for row in range(0, row_len)]
        length = len(col_list)
        sum_emis = float(0.0)
        sum_emisu = float(0.0)
        for i, row in enumerate(col_list):
            emis_probs = emis_prob(i, col_list)
            e_prob[col][i] = emis_probs
            sum_emis += emis_probs
            
            emis_probsu = emis_prob_user(i, col_list)
            e_prob_u[col][i] = emis_probsu
            sum_emisu += emis_probsu
        for index, row in enumerate(e_prob[col]):
            e_prob[col][index]+=0.00000001
            e_prob[col][index] = math.log(1/(e_prob[col][index]/float(sum_emis)))
            e_prob_u[col][index]+=0.00000001
            e_prob_u[col][index] = math.log(1/(e_prob_u[col][index]/float(sum_emisu)))
    # print posterior_prob_list[0]


# ridge = [edge_strength.shape[0] / 2] * edge_strength.shape[1]
#imsave(output_filename, draw_edge(input_image, ridge, (i % 255, x % 255, (i + x) % 255), 5))
calculate_prob()
sim_ridge = simple()
imsave(output_filename, draw_edge(input_image, sim_ridge, (255, 0,0), 5))

hmm_ridge = hmm_viterbi()
imsave(output_filename, draw_edge(input_image, hmm_ridge, (0, 255,0), 5))

hmm_usr_ridge = hmm_viterbi_user()
imsave(output_filename, draw_edge(input_image, hmm_usr_ridge, (0, 0,255), 5))

'''
# print random_roll(100),row_len
imsave(output_filename,draw_edge1(input_image,x_axis, y_axis, (0, 0,255), 5))
'''
