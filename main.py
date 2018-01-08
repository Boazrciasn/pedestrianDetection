import numpy as np
import os
import sys
import tensorflow as tf
import glob
import datetime
import natsort
from collections import defaultdict
from io import StringIO
import csv
from matplotlib import pyplot as plt
from PIL import Image

from utils import label_map_util

from utils import visualization_utils as vis_util


# image to numpy helper.
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# timestamp gen :
def add_time_column(found, current_time):
    timestamp = np.array(current_time.__str__())
    timestamp = np.tile(timestamp, (found.shape[0], 1))
    found_str = found.astype('str')
    data = np.concatenate((timestamp, found_str), axis=1)
    return data


# SET HERE BEFORE START :
objdetect_dir = "/home/vvglab/Tensorflow/models/object_detection"
year = 2017
month = 2
day = 23
dpath = '/media/vvglab/Storage/Images/25_Agustos_2017/'

# number of images to process
n = 360
# min score to consider :
min_score = .2
# seconds per frame :
time_offset = datetime.timedelta(seconds=20)

os.chdir(objdetect_dir)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join('test_ckpt', 'faster_rcnn_inception_resnet_v2_coco/frozen_inference_graph.pb')
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
data = np.empty((0, 3))

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

print("Loading graph...")
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print("Starting detections...")
start = datetime.datetime.now()
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

        TIMES = glob.glob(dpath + '*')

        for time in TIMES:
            # set start date and time
            print(time)
            init_hour = int(time.split('/')[-1][:2])
            init_min = int(time.split('/')[-1][2:4])
            print(init_hour)

            CAMS = glob.glob(time + '/*')
            for cam in CAMS:
                cs = cam.split('/')
                name = cs[-1] + '-' + cs[-3] + '-' + cs[-2] + '.csv'
                print(name)
                savedir = dpath + name
                current_time = datetime.datetime(year, month, day, init_hour, init_min)
                TEST_IMAGE_PATHS = natsort.natsorted(glob.glob(cam + '/*.jpg'))
                print(len(TEST_IMAGE_PATHS))

                for image_path in TEST_IMAGE_PATHS[0:n:2]:
                    a = datetime.datetime.now()
                    print("Processing " + image_path)
                    image = Image.open(image_path)
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = load_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    # Filter out pedestrians only:
                    p_boxes = boxes[:, classes[0, :] == 1, :]
                    p_scores = scores[:, classes[0, :] == 1]
                    p_classes = classes[:, classes[0, :] == 1]

                    # Filter out detections above 0.2 confidence:
                    p_boxes = p_boxes[:, p_scores[0, :] > min_score, :]
                    p_classes = p_classes[:, p_scores[0, :] > min_score]
                    p_scores = p_scores[:, p_scores[0, :] > min_score]

                    print("detected people : " + p_scores.size)
                    if p_scores.size != 0:
                        # find bbox centers :
                        p_boxes = np.reshape(p_boxes, (p_boxes.shape[1], p_boxes.shape[2]))

                        img_size_t = np.tile(image.size[::-1], (p_boxes.shape[0], 2))
                        n_boxes = p_boxes * img_size_t
                        p_center = np.divide([n_boxes[:, 1] + n_boxes[:, 3], n_boxes[:, 0] + n_boxes[:, 2]], 2).T

                        # Size threshold:
                        p_boxesarea = np.multiply(n_boxes[:, 3] - n_boxes[:, 1], n_boxes[:, 2] - n_boxes[:, 0])
                        p_boxesarea = np.reshape(p_boxesarea, (p_boxesarea.size, 1)).T

                        p_boxes = p_boxes[p_boxesarea[0, :] < 35000]
                        #print(p_boxes)
                        # if p_boxes.size != 0:
                        #     vis_util.visualize_boxes_and_labels_on_image_array(
                        #         image_np,
                        #         np.squeeze(p_boxes),
                        #         np.squeeze(p_classes).astype(np.int32),
                        #         np.squeeze(p_scores),
                        #         category_index,
                        #         use_normalized_coordinates=True,
                        #         min_score_thresh=0)
                        #
                        #     plt.imshow(image_np)
                        #     plt.show()
                        current_time_log = current_time - datetime.timedelta(minutes=current_time.minute % 5,
                                                                             seconds=current_time.second)
                        data = np.append(data, add_time_column(p_center, current_time_log), axis=0)

                        b = datetime.datetime.now()
                    current_time = current_time + time_offset
                    print(current_time)

                np.savetxt(savedir, data, delimiter=';', fmt='%s')
                data = np.empty((0, 3))





