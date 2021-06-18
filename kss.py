import streamlit as st
import os
from detect import *
from PIL import Image
import time

import numpy as np
import imutils
import cv2

def run():
    st.title('Object Detection')
    option = st.radio('', ['Choose a test video', 'Upload your own video (.mp4 only)','upload your own image'])
    st.sidebar.title('Parameters')

    if option == 'Choose a test video':
        test_videos = os.listdir('data/images')
        test_video = st.selectbox('Please choose a test video', test_videos)
    elif option =='upload your own image':
        test_video = None
        style_file = st.file_uploader('upload a image')
        image_slot = st.empty()
        if style_file:
            stringio = style_file.getvalue()
            style_file_path = 'data/images/' + style_file.name
            with open(style_file_path,'wb') as f:
                f.write(stringio)
            image_slot.image(style_file_path)
        if style_file:
            img1 = Image.open(style_file_path)
            img1 =img1.resize((640,640))
            img1.save('runs/detect/exp/' + os.path.basename(style_file_path))
            image_slot.image('runs/detect/exp/' + os.path.basename(style_file_path))
        if st.button('开始预测'):
            my_bar = st.progress(10)
            if __name__ == '__main__':
                parser = argparse.ArgumentParser()
                parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt',
                                    help='model.pt path(s)')
                parser.add_argument('--source', type=str, default='data/images/'+style_file.name,help='source')  # file/folder, 0 for webcam
                parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
                parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
                parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
                parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
                parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
                parser.add_argument('--view-img', action='store_true', help='display results')
                parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
                parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
                parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
                parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
                parser.add_argument('--classes', nargs='+', type=int,
                                    help='filter by class: --class 0, or --class 0 2 3')
                parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
                parser.add_argument('--augment', action='store_true', help='augmented inference')
                parser.add_argument('--update', action='store_true', help='update all models')
                parser.add_argument('--project', default='runs/detect', help='save results to project/name')
                parser.add_argument('--name', default='exp', help='save results to project/name')
                parser.add_argument('--exist-ok', action='store_false',
                                    help='existing project/name ok, do not increment')
                parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
                parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
                parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
                parser.add_argument('--half', type=bool, default=False, help='use FP16 half-precision inference')
                opt = parser.parse_args()
                print(opt)
                check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

                if opt.update:  # update all models (to fix SourceChangeWarning)
                    for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                        detect(opt=opt)
                        strip_optimizer(opt.weights)
                else:
                    detect(opt=opt)
            output_path = 'runs/detect/exp/' + style_file.name
            for i in range(0,100,10):
                my_bar.progress(i+1)
            my_bar.progress(100)
            st.write('预测图片')
            st.image(output_path)
    else:
        test_video_1 = st.file_uploader('Upload a video', type = ['mp4'])
        if test_video_1 is not None:
            test_video = test_video_1.name
        else:
            st.write('** Please upload a test video **')

    if test_video is not None:
        video = test_video
        if st.button('开始预测'):
            my_bar = st.progress(10)
            if __name__ == '__main__':
                parser = argparse.ArgumentParser()
                parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt',
                                    help='model.pt path(s)')
                parser.add_argument('--source', type=str, default=video,
                                    help='source')  # file/folder, 0 for webcam
                parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
                parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
                parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
                parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
                parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
                parser.add_argument('--view-img', action='store_true', help='display results')
                parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
                parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
                parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
                parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
                parser.add_argument('--classes', nargs='+', type=int,
                                    help='filter by class: --class 0, or --class 0 2 3')
                parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
                parser.add_argument('--augment', action='store_true', help='augmented inference')
                parser.add_argument('--update', action='store_true', help='update all models')
                parser.add_argument('--project', default='runs/detect', help='save results to project/name')
                parser.add_argument('--name', default='exp1', help='save results to project/name')
                parser.add_argument('--exist-ok', action='store_false',
                                    help='existing project/name ok, do not increment')
                parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
                parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
                parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
                parser.add_argument('--half', type=bool, default=False, help='use FP16 half-precision inference')
                opt = parser.parse_args()
                print(opt)
                check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

                if opt.update:  # update all models (to fix SourceChangeWarning)
                    for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                        detect(opt=opt)
                        strip_optimizer(opt.weights)
                else:
                    detect(opt=opt)
            output_path = 'runs/detect/exp1/' + video
            for i in range(0, 100, 10):
                my_bar.progress(i + 1)
            my_bar.progress(100)
            st.write('完成')
            st.video(output_path)




if __name__ == '__main__':
    run()