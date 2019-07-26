## Anki Vector Hand Gesture Recognition
This repo is a step by step explanation of my procces in making the robot anki vector recognize and read out loud the number of
fingers you're holding up in your hand in real time.



## Installations
Anki Vector SDK - https://developer.anki.com/vector/docs/initial.html  
Open CV 4.1.0 - https://pypi.org/project/opencv-python/  
Numpy and Sci-kit Learn (best way is to install anaconda) - https://docs.anaconda.com/anaconda/install/

## How to use
On top of the dependencies above you need to autheticate your anki vector with your computer here's how to do this on 
[Linux](https://developer.anki.com/vector/docs/install-linux.html#vector-authentication), [Windows](https://developer.anki.com/vector/docs/install-windows.html#vector-authentication)
, [Mac](https://developer.anki.com/vector/docs/install-macos.html#vector-authentication)    
  
Then just clone the repository and run the **hand_recognition.py** with python3  
  
Select the ROI (Region of Interest) and wait a couple seconds for vector to calibrate the background. **During this proccess
the background and vector need to be static**  
  
After that just put your hand on the ROI and he'll say how many fingers you have up.


## How it Works
### Getting the Camera Feed
First we access vector's camera and set his head angle properlly.

```python3
with anki_vector.Robot() as robot:
  robot.camera.init_camera_feed()
  robot.behavior.set_head_angle(degrees(25.0))
  robot.behavior.set_lift_height(0.0)
  while robot.camera.image_streaming_enabled():
    #All code goes in here
```
### Pre-processing
Vector's images are in RGB, but OpenCV only displays BGR images so we have to convert it for display purposes, it's also good
to flip the image that way it isn't a mirror image.
```python3
img = cv2.cvtColor(np.array(robot.camera.latest_image.raw_image),cv2.COLOR_RGB2BGR)
img = cv2.flip(img,1)
```
For processing we are going to convert the image to gray scale, that is because a gray scale image only has one color channel
and color images have 3, with only one color channel processing will be smoother. Also to reduce noise we'll apply a gaussian blur.
```python3
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (7,7), 0)
```
### Background Subtraction / Recognizing Hands
The method I used to recognize the hand was background subtraction + thresholding. In a nutshell we are going to identify the 
background by running the program for 40 frames, then we'll subtract the background from every frame after the first 40.
On top of that we'll apply a binary threshold on every pixel of the result, leaving us with a black and white image of the hand.
Finally we have to extract the contour of the threholded image.
  
[Click here to read more about this method](https://docs.opencv.org/trunk/d1/dc5/tutorial_background_subtraction.html)  
[This video is also excellent](https://www.youtube.com/watch?v=nRt2LPRz704)

We can get the average value of the background in the following way:
```python3
def get_background(img):
    global background
    if background is None:
        background = img.copy().astype('float32')
        return  
    cv2.accumulateWeighted(img.astype(background.dtype), background, 0.5)
```
  
  
Let's also run this function on a specific ROI (Region of Interest) that way we don't waste processing power on the entire image
```python3
if num_frames < 40:
            if num_frames == 0:
                x1,y1,x2,y2 = cv2.selectROI('ROI selector', img, False)
                cv2.destroyWindow('ROI selector')
                print('Calibrating Background...\nDo not move the camera')
            gray = gray[int(y1):int(y1+y2), int(x1):int(x1+x2)]
            get_background(gray)
            num_frames += 1
            continue
```
  
Now we can just peform the subtraction using the function below. The function also find the largest countour
of the threholded image, which we will assume to be the hand.
```python3
def background_sub(img):
    diff = cv2.absdiff(img.astype(background.dtype), background)
    mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) > 0:
        hand_contour = max(contours,key=cv2.contourArea)
        return mask, hand_contour
    return mask, None
```
### Counting the fingers
In order to count how many fingers in a hand I used the algorithm described in [this paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.454.3689&rep=rep1&type=pdf)
and also in [this tutorial](https://gogul09.github.io/software/hand-gesture-recognition-p2). Here's how it works:  
  
1)Get the extreme points of the contour (top, bottom, left, right).  
2)Find the center of the hand using those extreme points.  
3)Draw a circle with the radius 70% of the distance between the center and the top extreme point.  
4)Find the contour of the overlaping parts of the circle and the hand.  
5)Count the remaining parts of the circle excluding the bottom which is should be the wrist.  
  
The following function will do steps 1 - 4 of the algorithm:
```python3
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
```
  
  
Finally this code will count the amout of contours and make vector read it out loud:
```python3
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
```
