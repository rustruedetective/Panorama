from sys import argv
import glob, os
from PIL import Image
from pylab import *
import numpy as np


linearH = array([[1,0,0],[0,1,0],[0,0,1]])


def fileList(dirN, formN):
    """
    Parameters:Sub folder, format of files to look for
    Returns:a list of all file names in that directory of particular format
    """
    lis = []
    for file in glob.glob("./" + dirN + "/" + "*." + formN):         
        lis.append(Image.open(file).convert("RGB"))
    return lis


def show2Images(img1, img2, pointsCollect=-1, x=8):
    """
    Parameters: First image to plot, second image to plot, if want to collect points here pointscollect=1
    or to show images here points to collect=0, finally the number of points to collect from two images
    Return:plots two images at a time or returns points selected on image or do nothing except loading
    images in buffer
    """
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img1)
    axarr[1].imshow(img2)
    if pointsCollect == 0:
        show()
    elif pointsCollect == 1:
        return pointsCollector(x)
    elif pointsCollect == -1:
        pass


def pointsCollector(x):
    """
    Parameters: no of points to collect
    Return: a list of points selected by us on image
    """
    lis = ginput(x)
    show()
    return lis


def pointsAssign(value=1):
    """
    Parameters: selector for which predefined coordinates to select
    Return: the selected points as an array
    """
    p = array([
        [1,1,1],
        [2,2,1],
        [3,3,1],
        [4,4,1],
        [1,1,1],
        [2,2,1],
        [3,3,1],
        [4,4,1]
        ])
    p2 = array([
        [100,100,1],
        [200,100,1],
        [200,200,1],
        [100,200,1],
        [200,100,1],
        [300,100,1],
        [300,200,1],
        [200,200,1]
        ])
    p3 = array([
        [175,263,1],
        [173,425,1],
        [370,383,1],
        [330,523,1],
        [192,102,1],
        [195,267,1],
        [394,220,1],
        [353,360,1]
        ])
    if value == 1:
        return p
    elif value == 2:
        return p2
    elif value == 3:
        return p3


def arrayAssign(points, value=1):
    """
    Parameters: selector for which predefined A arrays to select
    Return: the selected A array
    """
    p = array([
        [0,0,0,           -points[0][0],-points[0][1],-1,   points[0][0]*points[4][1],points[0][1]*points[4][1],points[4][1]],
        [points[0][0],points[0][1],1,   0,0,0,           -points[0][0]*points[4][0],-points[0][1]*points[4][0],-points[4][0]],
        [0,0,0,           -points[1][0],-points[1][1],-1,   points[1][0]*points[5][1],points[1][1]*points[5][1],points[5][1]],
        [points[1][0],points[1][1],1,   0,0,0,           -points[1][0]*points[5][0],-points[1][1]*points[5][0],-points[5][0]],
        [0,0,0,           -points[2][0],-points[2][1],-1,   points[2][0]*points[6][1],points[2][1]*points[6][1],points[6][1]],
        [points[2][0],points[2][1],1,   0,0,0,           -points[2][0]*points[6][0],-points[2][1]*points[6][0],-points[6][0]],
        [0,0,0,           -points[3][0],-points[3][1],-1,   points[3][0]*points[7][1],points[3][1]*points[7][1],points[7][1]],
        [points[3][0],points[3][1],1,   0,0,0,           -points[3][0]*points[7][0],-points[3][1]*points[7][0],-points[7][0]]
    ]).astype(int)
    p2 = array([
        [0,0,0,           points[0][0],points[0][1],1,   -points[0][0]*points[4][1],-points[0][1]*points[4][1],-points[4][1]],
        [points[0][0],points[0][1],1,   0,0,0,           -points[0][0]*points[4][0],-points[0][1]*points[4][0],-points[4][0]],
        [0,0,0,           points[1][0],points[1][1],1,   -points[1][0]*points[5][1],-points[1][1]*points[5][1],-points[5][1]],
        [points[1][0],points[1][1],1,   0,0,0,           -points[1][0]*points[5][0],-points[1][1]*points[5][0],-points[5][0]],
        [0,0,0,           points[2][0],points[2][1],1,   -points[2][0]*points[6][1],-points[2][1]*points[6][1],-points[6][1]],
        [points[2][0],points[2][1],1,   0,0,0,           -points[2][0]*points[6][0],-points[2][1]*points[6][0],-points[6][0]],
        [0,0,0,           points[3][0],points[3][1],1,   -points[3][0]*points[7][1],-points[3][1]*points[7][1],-points[7][1]],
        [points[3][0],points[3][1],1,   0,0,0,           -points[3][0]*points[7][0],-points[3][1]*points[7][0],-points[7][0]]
    ]).astype(int)
    p3 = array([
        [points[0][0],points[0][1],1,   0,0,0,           -points[0][0]*points[4][0],-points[0][1]*points[4][0],-points[4][0]],
        [0,0,0,           -points[0][0],-points[0][1],-1,   points[0][0]*points[4][1],points[0][1]*points[4][1],points[4][1]],
        [points[1][0],points[1][1],1,   0,0,0,           -points[1][0]*points[5][0],-points[1][1]*points[5][0],-points[5][0]],
        [0,0,0,           -points[1][0],-points[1][1],-1,   points[1][0]*points[5][1],points[1][1]*points[5][1],points[5][1]],
        [points[2][0],points[2][1],1,   0,0,0,           -points[2][0]*points[6][0],-points[2][1]*points[6][0],-points[6][0]],
        [0,0,0,           -points[2][0],-points[2][1],-1,   points[2][0]*points[6][1],points[2][1]*points[6][1],points[6][1]],
        [points[3][0],points[3][1],1,   0,0,0,           -points[3][0]*points[7][0],-points[3][1]*points[7][0],-points[7][0]],
        [0,0,0,           -points[3][0],-points[3][1],-1,   points[3][0]*points[7][1],points[3][1]*points[7][1],points[7][1]]
    ]).astype(int)
    p4 = array([
        [points[0][0],points[0][1],1,   0,0,0,           -points[0][0]*points[4][0],-points[0][1]*points[4][0],-points[4][0]],
        [0,0,0,           points[0][0],points[0][1],1,   -points[0][0]*points[4][1],-points[0][1]*points[4][1],-points[4][1]],
        [points[1][0],points[1][1],1,   0,0,0,           -points[1][0]*points[5][0],-points[1][1]*points[5][0],-points[5][0]],
        [0,0,0,           points[1][0],points[1][1],1,   -points[1][0]*points[5][1],-points[1][1]*points[5][1],-points[5][1]],
        [points[2][0],points[2][1],1,   0,0,0,           -points[2][0]*points[6][0],-points[2][1]*points[6][0],-points[6][0]],
        [0,0,0,           points[2][0],points[2][1],1,   -points[2][0]*points[6][1],-points[2][1]*points[6][1],-points[6][1]],
        [points[3][0],points[3][1],1,   0,0,0,           -points[3][0]*points[7][0],-points[3][1]*points[7][0],-points[7][0]],
        [0,0,0,           points[3][0],points[3][1],1,   -points[3][0]*points[7][1],-points[3][1]*points[7][1],-points[7][1]]
    ]).astype(int)
    if value == 1:
        return p
    elif value == 2:
        return p2
    elif value == 3:
        return p3
    elif value == 4:
        return p4


def compH(arr):
    """
    Parameters: the A matrix for which to compute H matrix
    Return: the H matrix (array type)
    """
    u, s, v = linalg.svd(arr)
    h = v[:][8]
    h = h/h[8]
    return array((h[0:3], h[3:6], h[6:9]))


def newCoordinates(h, mat):
    """
    Parameters: the H transform, the matrix to apply the H transform on
    Return: an array containing new coordinates of each pixel of transformed matrix
    """
    newCoords = zeros(mat.shape)
    for x in range(newCoords.shape[0]):
        for y in range(newCoords.shape[1]):
            newCoords[x][y] = np.round(dot( h, array([x,y,1]) )).astype(int)
    return newCoords


def newEdges(newCoords):
    """
    Parameters: an array containing new coordinates of each pixel of transformed matrix
    Return: the maximum and minimum x and y coordinates the transformed matrix will go to
    """
    maxX = 0
    maxY = 0
    minX = 0
    minY = 0
    for x in range(newCoords.shape[0]):
        for y in range(newCoords.shape[1]):
            if newCoords[x][y][0] > maxX:
                maxX = newCoords[x][y][0].astype(int)
            if newCoords[x][y][0] < minX :
                minX = newCoords[x][y][0].astype(int)
            if newCoords[x][y][1] > maxY:
                maxY = newCoords[x][y][1].astype(int)
            if newCoords[x][y][1] < minY:
                minY = newCoords[x][y][1].astype(int)
    return [maxX, minX, maxY, minY]


def newMat(h, mat):
    """
    Parameters: the H transform, the matrix to apply the H transform on
    Return: a tuple containing:-
    1. an array containing new coordinates of each pixel of transformed matrix
    2. the maximum and minimum x and y coordinates the transformed matrix will go to
    """
    newArray = newCoordinates(h,mat)
    edges = newEdges(newArray)
    return (newArray, edges)
    

def panoEdges(newEdges, oldPano):
    """
    Parameters: an array of the maximum and minimum x and y coordinates the transformed matrix will go to,
    and the old panorama/image array
    Return: an array of the maximum and minimum x and y coordinates the new panorama
    """
    newMaxX = newEdges[0]
    newMaxY = newEdges[2]
        
    if newEdges[0] < (oldPano.shape[0]-1):
        newMaxX = (oldPano.shape[0]-1)
    if newEdges[2] < (oldPano.shape[1]-1):
        newMaxY = (oldPano.shape[1]-1)
    return [newMaxX, newEdges[1], newMaxY, newEdges[3]]


def panoCoords(panoEdges):
    """
    Parameters: an array of the maximum and minimum x and y coordinates the new panorama
    Return: a blank panorama array specified by its maximum points/coordinates
    """
    return zeros( (panoEdges[0]-panoEdges[1]+1, panoEdges[2]-panoEdges[3]+1, 3) )


def newPano(newCoords, edges, mat, oldPano):
    """
    Parameters:
    1. an array containing new coordinates of each pixel of transformed matrix
    2. the maximum and minimum x and y coordinates the transformed matrix will go to
    3. the matrix to which we apply the H transform
    4. the old panorama/image array
    Return: new panorama image fully stitched
    """
    panE = panoEdges(edges, oldPano)
    panC = panoCoords(panE)
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            panC[ newCoords[x][y][0].astype(int)-edges[1] ][newCoords[x][y][1].astype(int)-edges[3] ] = mat[x][y]

    for x in range(oldPano.shape[0]):
        for y in range(oldPano.shape[1]):
            panC[ x - edges[1], y - edges[3]] = oldPano[x][y]
    return panC


#1.making a list of images in folder
images = fileList(argv[1], argv[2])


#2.showing the images of interest
#show2Images(images[0], images[1])


#3.collect points
#points = pointsCollector(8)
#OR auto points assigning
points = pointsAssign(3)


#4.make the A array
a = arrayAssign(points,3)


#5.Solving for H
h = compH(a)


#6.Getting new array data
newC, newE = newMat(h, array(images[0]) )


#7.Making new panorama
panC = newPano(newC, newE, array(images[0]), array(images[1]))


imshow(panC.astype('uint8'))
show()