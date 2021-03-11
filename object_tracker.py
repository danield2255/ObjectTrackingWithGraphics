import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import imutils
import math
from scipy import ndimage
from collections import deque
import collections
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def imageOverlay(image, overlay, pos, angle, scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    overlay = ndimage.rotate(overlay, angle)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = image.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
    # loop over all pixels of the overlay and apply the blending equation
    alpha = overlay[:, :, 3]/255.0

    shape_mat = alpha.shape
    #lets say alpha shape is (100, 100)
    indices = np.where(alpha != 0)


    def graphic_blending(overlay_x, overlay_y):
        if x >= rows or y >= cols:
            return
        try:
            image[x + overlay_x - int(w//2)][y + overlay_y - int(h//2)] = (0, 225, 55)
        except:
            pass
            
    v = np.vectorize(graphic_blending)
    return v(indices[0], indices[1])

def bbox2points(bbox):
    """    From bounding box yolo format    to corner points cv2 rectangle    """
    x, y, w, h = bbox    
    xmin = int(round(x - (w / 2)))    
    xmax = int(round(x + (w / 2)))    
    ymin = int(round(y - (h / 2)))    
    ymax = int(round(y + (h / 2)))    
    return xmin, ymin, xmax, ymax

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running



    #SET THE BUFFER OF POINTS
    buffer = 32
    #pts = deque(maxlen=buffer)
    counter = 0
    (dX, dY) = (0, 0)
    direction = ""
    #INITIALIZE TRACKED CENTERS
    tracked_centers = {}
    arrow = cv2.imread("yellow_arrow.png", -1)
    discolor = (0,0,0)
    red = (255, 0, 0)
    yellow = (255,255,0)
    green = (0,255,0) 
    distdict={}

    while True:
        return_value, frame = vid.read()
        try:
          pass
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            pass
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person', 'car']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        

        # update tracks
        cur_view_objs = 0
        cur_distances = {}
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            cur_view_objs += 1
            
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            #Calculate distance from the camera
            left, top, right, bottom = bbox2points(bbox)
            
            focal = 351.0
            height = 1.7
            dis = height*focal/(-top+bottom)


            if dis <=2.5:
                discolor = red
            elif dis >2.5 and dis < 3.0:
                discolor = yellow
            else:
                discolor = green
            
            #Update the history of distances
            label = track.track_id
            if label in distdict.keys():
                distdict[label].extend([dis,left])
            else:
                distdict[label] = [dis,left]

            #Add the dictionary of distances in the current frame
            cur_distances[track.track_id] = (dis, discolor)



            #GET CENTER OF RECTANGLE
            center = ((int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2)

            #check if the tracked object's id exists
            #if it doesn't, create a dictionary with points, dx, dy, direction
            if track.track_id not in tracked_centers.keys():
                tracked_centers[track.track_id] = {'points':deque(maxlen=buffer), 'dX':0, 'dY':0, 'direction':""}
            tracked_centers[track.track_id]['points'].appendleft(center)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            pts = tracked_centers[track.track_id]['points']
            for i in np.arange(1, len(pts)):
                if len(pts) < 10:
                  break
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # check to see if enough points have been accumulated in
                # the buffer
                if counter >= 10 and i == 1 and pts[-10] is not None:
                    # compute the difference between the x and y
                    # coordinates and re-initialize the direction
                    # text variables
                    tracked_centers[track.track_id]['dX'] = pts[-10][0] - pts[i][0]
                    tracked_centers[track.track_id]['dY'] = pts[-10][1] - pts[i][1]
                    (dirX, dirY) = ("", "")
                    # ensure there is significant movement in the
                    # x-direction
                    if np.abs(tracked_centers[track.track_id]['dX']) > 20:
                        dirX = "East" if np.sign(tracked_centers[track.track_id]['dX']) == 1 else "West"
                    # ensure there is significant movement in the
                    # y-direction
                    if np.abs(tracked_centers[track.track_id]['dY']) > 20:
                        dirY = "North" if np.sign(tracked_centers[track.track_id]['dY']) == 1 else "South"
                    # handle when both directions are non-empty
                    if dirX != "" and dirY != "":
                        tracked_centers[track.track_id]['direction'] = "{}-{}".format(dirY, dirX)
                    # otherwise, only one direction is non-empty
                    else:
                        tracked_centers[track.track_id]['direction'] = dirX if dirX != "" else dirY
            try:
                angle = math.degrees(math.atan(tracked_centers[track.track_id]['dY'] / tracked_centers[track.track_id]['dX']))
            except:
                angle = math.degrees(math.atan(tracked_centers[track.track_id]['dY'] / 0.001))
            if np.sign(tracked_centers[track.track_id]['dX']) == 1:
                angle = angle + 180
            imageOverlay(frame, arrow, center, angle)
            #THIS WAS PRINTING THE GENERAL DIRECTION
            #cv2.putText(frame, "id: {}, dir:{}".format(track.track_id, tracked_centers[track.track_id]['direction']), (10, 30* cur_view_objs), cv2.FONT_HERSHEY_SIMPLEX,
            #    0.65, (0, 0, 255), 3)
            
#             cv2.putText(frame, "id: {}, dx: {}, dy: {}".format(track.track_id, tracked_centers[track.track_id]['dX'], tracked_centers[track.track_id]['dY']),
#                 (10, frame.shape[0] - (10 * cur_view_objs)), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.35, (0, 0, 255), 1)
            
            counter += 1
        
        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        try:
            closest_obj_id = min(cur_distances, key = lambda k: cur_distances[k][0])
            cv2.putText(frame,f"Dist.: {round(cur_distances[closest_obj_id][0],2)}m",(30, 70),cv2.FONT_HERSHEY_SIMPLEX ,1,cur_distances[closest_obj_id][1],1,cv2.LINE_AA)
        except:
            print("NO OBJECT IN VIEW YET")
    
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        cv2.putText(frame,f"FPS: {int(fps)}",(30, 30),cv2.FONT_HERSHEY_SIMPLEX ,1,(0, 0, 255),1,cv2.LINE_AA)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
