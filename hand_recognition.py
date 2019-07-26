import anki_vector
import cv2
import numpy as np
from anki_vector.util import degrees
from sklearn.metrics import pairwise

def get_background(img):
    global background
    if background is None:
        background = img.copy().astype('float32')
        return  
    cv2.accumulateWeighted(img.astype(background.dtype), background, 0.5)

def background_sub(img):
    diff = cv2.absdiff(img.astype(background.dtype), background)
    mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) > 0:
        hand_contour = max(contours,key=cv2.contourArea)
        return mask, hand_contour
    return mask, None

def get_finger_contours(hand_contour):
    global img
    cv2.drawContours(img,[hand_contour+(x1,y1)],-1,(0,0,255))
    chull = cv2.convexHull(hand_contour)
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])
    x_bar = int((extreme_left[0] + extreme_right[0]) / 2)
    y_bar = int((extreme_top[1] + extreme_bottom[1]) / 2)
    distance = pairwise.euclidean_distances(X=[(x_bar, y_bar)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    max_distance = distance[distance.argmax()]
    radius = int(0.7 * max_distance)
    circumference = 2 * np.pi * radius
    circular_roi = np.zeros(mask.shape,dtype='uint8')
    cv2.circle(circular_roi,(x_bar,y_bar),radius,(255,255,255))
    circular_roi = cv2.bitwise_and(mask,mask,mask=circular_roi)
    cv2.imshow('circ',circular_roi)
    return cv2.findContours(circular_roi.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], y_bar, circumference

background = None
num_frames = 0
count_accumulator = 0
previous_count = 0
#Initialize camera feed and set vector's head to a appropiate angle
with anki_vector.Robot() as robot:
    robot.camera.init_camera_feed()
    robot.behavior.set_head_angle(degrees(25.0))
    robot.behavior.set_lift_height(0.0)
    while robot.camera.image_streaming_enabled():
        # Perform some pre-processing steps to minimize noise and simplify the processing steps
        img = cv2.cvtColor(np.array(robot.camera.latest_image.raw_image),cv2.COLOR_RGB2BGR)
        img = cv2.flip(img,1)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)
        #Select the ROI  and get the average background of it over the first 40 frames
        if num_frames < 40:
            if num_frames == 0:
                x1,y1,x2,y2 = cv2.selectROI('ROI selector', img, False)
                cv2.destroyWindow('ROI selector')
                print('Calibrating Background...\nDo not move the camera')
            gray = gray[int(y1):int(y1+y2), int(x1):int(x1+x2)]
            get_background(gray)
            num_frames += 1
            continue
        gray = gray[int(y1):int(y1+y2), int(x1):int(x1+x2)]
        cv2.rectangle(img,(x1+x2,y1),(x1,y1+y2),(0,255,0),2)
        #Peform the background subtraction so we can extract the hand
        mask, hand_contour = background_sub(gray)
        #Check to see if the area is a certain size, otherwise it will recognize random noise
        if hand_contour is not None and hand_contour.sum() > 11000:
            #Identify the finger contours, this is performed using a circle technique is this paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.454.3689&rep=rep1&type=pdf
            finger_contours, y_bar, circumference = get_finger_contours(hand_contour)
            #Count the number of finger contours if they're at least 25% of the circuference of the circle and excluding the wrist
            count = 0
            for c in finger_contours:
                (x, y, w, h) = cv2.boundingRect(c)
                if ((y_bar + (y_bar * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
                    count += 1
            if previous_count == count and count < 6:
                count_accumulator += 1
                #Have vector read out the number only if it sees the number for 30 consecutive frames
                if count_accumulator == 30:
                    print(count)
                    robot.behavior.say_text(str(count))
                    count_accumulator = 0
            else:
                previous_count = count
        cv2.imshow('video',img)
        cv2.imshow('mask',mask)
        #Press space bar to exit and close all windows
        if cv2.waitKey(1) == 32:
            break
    robot.camera.close_camera_feed()
    cv2.destroyAllWindows()