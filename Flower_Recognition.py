
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from itertools import combinations
import pafy
#import image_email_car   # Uncomment for alert to email

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

netMain = None
metaMain = None
altNames = None

def YOLO():
    """
    Perform Object detection
    """
    global metaMain, netMain, altNames
    configPath = "./cfg/flowers.cfg"                                 # Path to cfg
    weightPath = "./flowers.weights"                                 # Path to weights
    metaPath = "./cfg/flowers.data"                                         # Path to meta data
    if not os.path.exists(configPath):                                   # Checks whether file exists otherwise return ValueError  
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:                                                  # Checks the metaMain, NetMain and altNames. Loads it in script
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)                  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
            
    #cap = cv2.VideoCapture(0)                                           # Uncomment to use Webcam
    
    cap = cv2.VideoCapture("roses.mp4")                                # Uncomment for Local Stored video detection - Set input video
    
    #url = "https://www.youtube.com/watch?v=isveXCH4NcM"                 # Uncomment these lines for video from youtube
    #video = pafy.new(url)
    #best = video.getbest(preftype="mp4")
    #cap = cv2.VideoCapture()
    #cap.open(best.url)    
    
    #cap = cv2.VideoCapture('http://192.168.0.106:4747/mjpegfeed')       # Uncomment for Video from Mobile Camera (DroidCam Hosted Camera)
    
    frame_width = int(cap.get(3))                                        # Returns the width and height of capture video   
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    #print("Video Reolution: ",(width, height))

    #out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,  # Uncomment to save the output video    # Set the Output path for video writer
            #(new_width, new_height))
    
    # print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)         # Create image according darknet for compatibility of network
    
    while True:                                                          # Load the input frame and write output frame.
        prev_time = time.time()
        ret, frame_read = cap.read()                                   
        # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)          # Convert frame into RGB from BGR and resize accordingly
        frame_resized = cv2.resize(frame_rgb,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())    # Copy that frame bytes to darknet_image

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)    # Detection occurs at this line and return detections, for customize we can change
        image = cvDrawBoxes(detections, frame_resized)                   # Call the function cvDrawBoxes() for colored bounding box per class
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))                             # Prints frames per second
        cv2.imshow('Demo', image)                                    # Display Image window
        cv2.waitKey(3)
        #out.write(image)                                            # Write that frame into output video
        
    cap.release()                                                    # For releasing cap and out. 
    #out.release()                                                   # Uncomment to save the output video 
    print(":::Video Write Completed")

if __name__ == "__main__":
    YOLO()                                                           # Calls the main function YOLO()