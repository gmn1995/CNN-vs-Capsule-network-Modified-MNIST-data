# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 08:19:38 2016

@author: aviza
"""

from PIL import Image
import os, sys
import numpy
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import random
# import cv2

digit = "0"

path = "/home/avishkar/deep_learning/CSE_676/CapsNet-Keras/mnist_png/testing/" + digit + "/"
target_path = "/home/avishkar/deep_learning/CSE_676/CapsNet-Keras/mnist_png/rotate_test_data/" + digit + "/"

dirs = os.listdir("/home/avishkar/deep_learning/CSE_676/CapsNet-Keras/mnist_png/testing/" + digit )
#print ("dirctory is ",dirs)

def rotate_images():
    for item in dirs:
        #print "item is ",item
        if os.path.isfile(path+item) and item != "Thumbs.db":
            angle = random.randint(1,45)
            im = Image.open(path+item)
            arr = numpy.asarray(im, dtype="int32")
            print("angle is ",angle)
            print("shape is ",arr.shape)
            f, e = os.path.splitext(path+item)
            print ("resizing image ",f)
            image_rotate = im.rotate(angle)
            image_rotate.save(target_path + item , 'png', quality=90)

#rotate_images()
#image = Image.open(target_path + "118.png")
#arr = numpy.array(image.getdata())
#print("Reading rotated image shape ",arr.shape)
#arr = numpy.asarray(image,dtype="int32")
#print("Reading rotated image shape ",arr.shape)
#image.show()

def plot_mnist_digit(image,label):
    """ Plot a single MNIST image."""
    x_dim = 28
    y_dim = 28
    pixels = image.reshape(x_dim,y_dim)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(pixels, cmap = 'gray')
    plt.title('Digit is {label}'.format(label=label))
    plt.xticks(numpy.array([]))
    plt.yticks(numpy.array([]))
    plt.show()


def save_to_npy_file():
    # http://stackoverflow.com/questions/28439701/how-to-save-and-load-numpy-array-data-properly
    print("Inside the Save_to_npy_file function")
    width = 28
    height = 28

    digit = "0"
    path = "/home/avishkar/deep_learning/CSE_676/CapsNet-Keras/mnist_png/rotate_test_data/" + digit + "/"
    dirs = os.listdir(path)
    print("Directory is ",dirs)
    print("Number of items in directory: ", len(dirs))
    arr_store = numpy.zeros((len(dirs), width * height))
    print("array_store shape ",arr_store.shape)
    index = 0;

    for item in dirs:
        # print "item is ",item
        if os.path.isfile(path + item) and item != "Thumbs.db":
            im = Image.open(path + item)
            f, e = os.path.splitext(path + item)
            print("reading image to array ", f)
            im.load()
            arr = numpy.asarray(im, dtype="int32")
            # print arr.shape
            # arr = 255 - arr
            arr_store[index] = arr.ravel()
            index = index + 1
            #img = Image.fromarray(numpy.asarray(numpy.clip(arr, 0, 255), dtype="uint8"), "L")
            #img.save(target_path + item, 'png', quality=90)
            #plot_mnist_digit(arr_store[0],"0")
            #return

    plot_mnist_digit(arr_store[500], digit)
    numpy.save(digit + ".npy", arr_store)

#save_to_npy_file()


def create_mnist_rotated_dataset():
    X = numpy.load("0.npy")
    y = numpy.zeros((X.shape[0], 10))
    print("X shape ",X.shape)
    print("Y shape ",y.shape)

    for i in range(1, 10):
        print("collecting matrix for digit ", i)
        filename = str(i) + ".npy"
        data_x = numpy.load(filename)
        data_y = numpy.zeros((data_x.shape[0], 10))
        data_y[data_y == 0] = i;

        print(data_y.shape)

        X = numpy.vstack((X, data_x))
        y = numpy.vstack((y, data_y))

    print(X.shape)
    print(y.shape)
    numpy.save("mnist_rotated_data.npy", X)
    numpy.save("mnist_rotated_labels.npy", y)
    plot_mnist_digit(X[7992], y[7992])

create_mnist_rotated_dataset()

# http://stackoverflow.com/questions/21517879/python-pil-resize-all-images-in-a-folder
def resize():
    for item in dirs:
        #print "item is ",item
        if os.path.isfile(path+item) and item != "Thumbs.db":
            im = Image.open(path+item)
            #im = im.convert('RGB')
            f, e = os.path.splitext(path+item)
            print ("resizing image ",f)
            imResize = im.resize((28,28), Image.ANTIALIAS)
            #imResize.load()
            #arr = numpy.asarray( imResize, dtype="int32" )
            #print arr.shape
            imResize.save(target_path + item , 'png', quality=90)
            #return;

#resize()

def adjust_color_scale_to_mnist():
    digit = "3" 
    path = "C:\\az\\Semetster 1\\CSE 574 Machine learning\\project_3\\USPSdata\\Numerals\\resized_"+digit+"\\"
    target_path = "C:\\az\\Semetster 1\\CSE 574 Machine learning\\project_3\\USPSdata\\Numerals\\mnist_color_"+digit+"\\"
    dirs = os.listdir( path )
    print ("Number of items in directory: ",len(dirs))
    for item in dirs:
        #print "item is ",item
        if os.path.isfile(path+item) and item != "Thumbs.db":
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            print ("recoloring image ",f)
            im.load()
            arr = numpy.asarray( im, dtype="int32" )
            # print arr.shape
            arr = 255 - arr
            
            img = Image.fromarray( numpy.asarray( numpy.clip(arr,0,255), dtype="uint8"), "L" )
            img.save(target_path + item , 'png', quality=90)
            #imResize.save(target_path + item , 'png', quality=70)
            #return;

#adjust_color_scale_to_mnist()

def save_to_file():
    # http://stackoverflow.com/questions/28439701/how-to-save-and-load-numpy-array-data-properly
    width = 28
    height = 28
    
    digit = "9" 
    path = "C:\\az\\Semetster 1\\CSE 574 Machine learning\\project_3\\USPSdata\\Numerals\\mnist_color_"+digit+"\\"
    dirs = os.listdir( path )
    print ("Number of items in directory: ",len(dirs))
    arr_store = numpy.zeros((len(dirs),width*height))
    print (arr_store.shape)
    index = 0;
    for item in dirs:
        #print "item is ",item
        if os.path.isfile(path+item) and item != "Thumbs.db":
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            print ("reading image to array ",f)
            im.load()
            arr = numpy.asarray( im, dtype="int32" )
            # print arr.shape
            #arr = 255 - arr
            arr_store[index] = arr.ravel()
            index = index + 1
            img = Image.fromarray( numpy.asarray( numpy.clip(arr,0,255), dtype="uint8"), "L" )
            img.save(target_path + item , 'png', quality=90)
            #imResize.save(target_path + item , 'png', quality=70)
            #plot_mnist_digit(arr_store[index],"0")
            #return
    numpy.save(digit+".npy",arr_store)            
            
# save_to_file()
# test = numpy.load("9.npy")
# print "test shape ",test.shape
# plot_mnist_digit(test[1],"0")

def create_usps_dataset():
    X =  numpy.load("0.npy")
    y = numpy.zeros((X.shape[0],1))
    print (X.shape)
    print (y.shape)
    
    for i in range(1,10):
        print ("collecting matrix for digit ",i)
        filename = str(i) + ".npy"
        data_x = numpy.load(filename)
        data_y = numpy.zeros((data_x.shape[0],1))
        data_y [data_y == 0] = i;
        
        print (data_y.shape)
        
        X = numpy.vstack((X,data_x))
        y = numpy.vstack((y,data_y))
    
    print (X.shape)
    print (y.shape)
    numpy.save("usps_data.npy",X)
    numpy.save("usps_labels.npy",y)
    plot_mnist_digit(X[18203],y[18203])
        
# create_usps_dataset()

def adjust_gray_scale():
    file_name = "0001a.png"
    im = Image.open(target_path+file_name)
    im.load()
    data = numpy.asarray( im, dtype="int32" )
    print ("Data shape ",data.shape)
    #print data
    
    data = 255 - data;
    #print data
    plot_mnist_digit(data,"0")
    
    data = data.ravel()
    print ("Data shape after ravel ",data.shape)
    plot_mnist_digit(data,"0")
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mnist = fetch_mldata('MNIST original', data_home=dir_path)
    X = mnist.data
    y = mnist.target
    y [ y == 0 ] = 10
    
    print ("Mnist data shape ",X[0].shape);
    plot_mnist_digit(X[0],y[0])