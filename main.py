import streamlit as st
import os
import time

import config

import numpy as np
import imutils
import cv2

def run():
    st.title('Object Detection in Video')
    option = st.radio('', ['Choose a test video', 'Upload your own video (.mp4 only)', 'Camera'])
    st.sidebar.title('Parameters') 
    confidence_slider = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, config.DEFALUT_CONFIDENCE, 0.05)
    nms_slider = st.sidebar.slider('Non-Max Suppression Threshold', 0.0, 1.0, config.NMS_THRESHOLD, 0.05)

    if option == 'Choose a test video':
        test_videos = os.listdir(config.INPUT_PATH)
        test_video = st.selectbox('Please choose a test video', test_videos)
    elif option == 'Camera':
        test_video = None
    else:
        test_video = st.file_uploader('Upload a video', type = ['mp4'])

        if test_video is not None:
            pass
        else:
            st.write('** Please upload a test video **')

    if test_video is not None:
        video = config.VIDEO_PATH + test_video
    else:
        video = 0


    if st.button ('Detect Objects'):
        
        time.sleep(3)
        st.write(f"[INFO] Processing Video....")

        FRAME_WINDOW = st.image([])

        # initialize video stream, pointer to output video file and grabbing frame dimension
        vs = cv2.VideoCapture(video)

        (W,H) = (None, None)

        # determine the total number of frames in a video
        try:
            prop = cv2.CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
            total = int(vs.get(prop))
            print(f"[INFO] {total} frames in the video")

        # if error occurs print
        except:
            print(f"[INFO] {total} frames in the video")
            total = -1

        # loop over on entire video frames
        while True:
            # read next frame
            (grabbed, frame) = vs.read()

            # if no frame is grabbed, we reached the end of video, so break the loop
            if not grabbed:
                break
            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H,W) = frame.shape[:2]

            # build blob and feed forward to YOLO to get bounding boxes and probability
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB = True, crop = False)
            start = time.time()
            net.setInput(blob)
            layerOutputs = net.forward(ln)
            end = time.time()

    
            # get metrics from YOLO

            boxes = []
            confidences = []
            classIDs = []

            # loop over each output from layeroutputs
            for output in layerOutputs:
                # loop over each detecton in output
                for detection in output:
                    # extract score, ids and confidence of current object detection
                    score = detection[5:]
                    classID = np.argmax(score)
                    confidence = score[classID]

                    # filter out weak detections with confidence threshold
                    if confidence > confidence_slider:
                        # scale bounding box coordinates back relative to image size
                        # YOLO spits out center (x,y) of bounding boxes followed by 
                        # boxes width and heigth
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype('int')

                        # grab top left coordinate of the box
                        x = int(centerX - (width/2))
                        y = int(centerY - (height/2))

                        boxes.append([x,y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # Apply Non-Max Suppression, draw boxes and write output video 

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_slider, nms_slider)
            # ensure detection exists
            if len(idxs) > 0:
                for i in idxs.flatten():
                    # getting box coordinates
                    (x,y) = (boxes[i][0], boxes[i][1])
                    (w,h) = (boxes[i][2], boxes[i][3])

                    # color and draw boxes
                    color = [int(c) for c in config.COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x,y), (x+w, y+h), color, 1)
                    text = "%s: %0.2f"%(config.LABELS[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            FRAME_WINDOW.image(frame[:,:,::-1])


        vs.release()

if __name__ == '__main__':
    run()