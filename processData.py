import numpy as np
from numpy import genfromtxt
import glob
from datetime import datetime

found_boxes = genfromtxt('found.csv', delimiter=';')
img_size = genfromtxt('img_size.csv', delimiter=';')
classes = genfromtxt('classes.csv', delimiter=';')

a = datetime(2017, 3, 4, 10, 00)
timestamp = np.array(a.__str__())
timestamp = np.tile(timestamp, (found_boxes.shape[0],1))
print(timestamp)

##for tst purposes :
TEST_IMAGE_PATHS = glob.glob('/Users/barisozcan/Documents/TUBITAK_CAYDEK/testjpg/*.jpg')
foundstr = found_boxes.astype('str')

data = np.concatenate((timestamp,foundstr), axis = 1)

print("data : {}".format(data))
np.savetxt("/Users/barisozcan/Documents/TUBITAK_CAYDEK/tf/found2.csv", data, delimiter=';', fmt = '%s')
