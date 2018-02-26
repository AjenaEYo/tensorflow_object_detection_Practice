import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf

from queue import Queue
from threading import Thread
from utils.app_utils import FPS, draw_boxes_and_labels
from object_detection.utils import label_map_util

import sys
import six.moves.urllib as urllib
import tarfile

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2017_11_08'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[5]:

'''opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())'''


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

output = []

# # Detection
def detect_objects(image_np, sess, detection_graph,width,height,one_width,one_height,index,rect_points,class_names,class_colors):
    #print('1'+str(rect_points))
    if index % 3 == 0 :
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

        # Visualization of the results of a detection.
        rect_points, class_names, class_colors = draw_boxes_and_labels(
            boxes=np.squeeze(boxes),
            classes=np.squeeze(classes).astype(np.int32),
            scores=np.squeeze(scores),
            category_index=category_index,
            min_score_thresh=.5
        )
    

    #print('2'+str(rect_points))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for point, name, color in zip(rect_points, class_names, class_colors):
      if point['xmax'] - point['xmin'] < 0.4:
          cv2.rectangle(image_np, (int(point['xmin'] * width), int(point['ymin'] * height)),
                              (int(point['xmax'] * width), int(point['ymax'] * height)), color, 3)
          cv2.rectangle(image_np, (int(point['xmin'] * width), int(point['ymin'] * height)),
                              (int(point['xmin'] * width) + len(name[0]) * 6,
                               int(point['ymin'] * height) - 10), color, -1, cv2.LINE_AA)
          cv2.putText(image_np, name[0], (int(point['xmin'] * width), int(point['ymin'] * height)), font,
                            0.3, (0, 0, 0), 1)
    return cv2.resize(image_np, (one_width,one_height)),rect_points,class_names,class_colors

def worker(videofilename, output_q,one_width,one_height,sess,detection_graph,i,j):

    cap=cv2.VideoCapture(videofilename)
    width = int(cap.get(3))
    height = int(cap.get(4))
    index = 0
    rect_points = []
    class_names = []
    class_colors = []
    frame_counter = 0
    #print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        #if i == 0 and j ==0:
        #    t = time.time()
        ret, frame = cap.read()
        frame_counter = frame_counter + 1
        if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0 #Or whatever as long as it is the same as next line
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        #print('frame : %d' % frame_counter)
        image_np,rect_points,class_names,class_colors=detect_objects(frame, sess, detection_graph,width,height,one_width,one_height,index,rect_points,class_names,class_colors)
        #output_q.put(image_np)
        output[i*one_height:one_height*i+one_height,j*one_width:one_width*j+one_width] = image_np
        index = index + 1
        #if i == 0 and j ==0:
        #    print('[INFO] elapsed time: {:.3f} sec'.format(time.time() - t))

    #fps.stop()
    #sess.close()


if __name__ == '__main__':

    cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Video",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    
    width = 3840#1680#3840#video_capture.get(3)   # float
    height = 2160#1050#2160#video_capture.get(4) # float
    
    #w = 4
    #h = 6
    w = 4
    h = 4
    one_width = int(width / w)
    one_height = int(height / h)


    output = np.zeros((one_height * h,one_width * w , 3), dtype="uint8")

    
    #input_q = Queue(5)  # fps is better if queue is higher but then more lags
    '''with tf.device('/device:GPU:2'):
        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True),graph=detection_graph)
    '''

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = []
    detection_graph = []
    for i in range(4):
        detection_graph.append(tf.Graph())
        with detection_graph[i].as_default():
            od_graph_def = tf.GraphDef()
            deviceidx = i

            if deviceidx == 4:
               deviceidx = 1
            if deviceidx == 5:
               deviceidx = 2
            if deviceidx == 6:
               deviceidx = 3

            print('deviceidx = '+str(deviceidx))
            with tf.device('/device:GPU:%d' % deviceidx):        
                with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

                sess.append(tf.Session(config=config,graph=detection_graph[i]))

    output_qs = []
    t = []
    for i in range(h):
        for j in range(w):
            output_qs.append(Queue())
            t.append(Thread(target=worker, args=('/raid/video/obsample%d.mp4' % (w*i+j), output_qs[w*i+j],one_width,one_height,sess[(w*i+j)%4],detection_graph[(w*i+j)%4],i,j)))
            t[w*i+j].daemon = True
            t[w*i+j].start()

    idx = 0
    while True:
      '''isArryEmpty = False
      for i in range(w*h):
        if output_qs[i].empty():
           isArryEmpty = True

      if isArryEmpty:
         #print('isEmpty'+str(idx))
         idx= idx +1
         continue
      else:
        for i in range(h):
          for j in range(w):
            output[i*one_height:one_height*i+one_height,j*one_width:one_width*j+one_width] = output_qs[w*i+j].get()'''
            
      cv2.imshow('Video', output)
      
      if cv2.waitKey(1) & 0xFF == ord('q'):
         break

    #fps.stop()
    #print('[IN#FO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    #print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    sess[0].close()
    cv2.destroyAllWindows()
