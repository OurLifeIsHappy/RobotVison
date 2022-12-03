#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2, math
import rospy, rospkg, time
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from math import *
import signal
import sys
import os
import random
import time
from model.lanenet.LaneNet import LaneNet
from model_traffic import xycar_A2NN
from deviding import sliding_window
import torch
from torchvision import transforms

from PIL import Image as pim


padding_size = 16
model = xycar_A2NN().to(device='cuda')
# model_seg = LaneNet(arch='DeepLabv3+').to('cuda')

lower_th = (130, 130, 130)
upper_th = (255, 255, 255)


# print(os.getcwd())
model.load_state_dict(torch.load('./logs/tr_detect/nn_state_xycar_2.t7'))
# model_seg.load_state_dict(torch.load('./logs/segs/DeepLabv3/best_model.pth'))
# model.eval()
# model_seg.eval()
devider = sliding_window(padding = padding_size)


resize_height = 256
resize_width = 512

data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])




def signal_handler(sig, frame):
    import time
    time.sleep(3)
    os.system('killall -9 python rosout')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_frame(self):
    frame = self.realimage.copy()
    frame = cv2.resize(frame,(240,320))
    # frame = cv2.blur(frame,(2,2))
    return frame

def extract_bbox(self,frame):
    """input : frame from camera
        output :bbox(preds, x, y) """
    input_data = torch.Tensor(frame).to(device='cuda')
    input_data = input_data.permute(2,0,1)-127.5/127.5
    input_data,size_y,size_x = devider.images(input_data)
    preds = model(input_data)
    return preds, size_x, size_y

def about_traffic_light(self,frame,preds,size_x,size_y):
    """input : frame,preds,size_x,size_y
        output :detection_red, detection_green """
    if len(preds) == size_x*size_y:
        pred_red_list = []
        pred_green_list = []
        for i in range(len(preds)):
            predict_label_ind = torch.argmax(preds[i,:])
            predict_label_val = torch.max(preds[i,:])
            if predict_label_ind == 0 and predict_label_val>13:
                pred_red_list.append((i))
            elif predict_label_ind == 1 and predict_label_val>13:
                pred_green_list.append((i))

        # if len(pred_red_list) != 0 :
        #     print(':: Detect Red ::')

        # if len(pred_green_list) != 0:
        #     print(':: Detect Green ::')
        # print(preds.shape)

    if pred_green_list is not None:
        for index_red in pred_green_list:
            y = index_red//size_x
            x = index_red%size_x
            cv2.rectangle(frame, (x*padding_size,y*padding_size), (x*padding_size+32, y*padding_size+32), (255, 0, 0), 2)


    if pred_red_list is not None:
        for index_red in pred_red_list:
            y = index_red//size_x
            x = index_red%size_x
            cv2.rectangle(frame, (x*padding_size,y*padding_size), (x*padding_size+32, y*padding_size+32), (0, 0, 255), 2)
    # flag
    if len(pred_red_list) != 0:
        print('red_detect!!!!')
        detection_red=1
    else:
        detection_red=0

    if len(pred_green_list) != 0:
        print('green_detect!!!!')
        detection_green=1
    else:
        detection_green=0
    
    return detection_red, detection_green

def line_detect(self,frame):
    """input : frame
    output : transition, mission_1_time """
    img = frame.copy()
    # print('img',img.shape)
    img = img[ 230:300 , 40:]
    # print('img_resize',img.shape)
    img = img.copy()

    # Center
    cv2.circle(img, (100,50), 10, (0,0,255), 3)

    ## Find Line 
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # print(blur.shape)
    detect_image = cv2.inRange(blur, lower_th, upper_th)
    # print(detect_image.shape)
    detect_image_view = detect_image.copy()

    cv2.imshow('detect_img',detect_image_view)

    all_lines = cv2.HoughLinesP(detect_image_view, 1, math.pi/180,30,30,10)
    
    center=100
    if np.array(all_lines).all() != None:
        for line in all_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    

        
        # calculate slope and do filtering
        slopes = []
        new_lines = []
        for line in all_lines:
            x1, y1, x2, y2 = line[0]
            if (x2 - x1) == 0:
                slope = 0
            else:
                slope = float(y2-y1) / float(x2-x1)
            if 0.1 < abs(slope) < 25:
                slopes.append(slope)
                new_lines.append(line[0])
        
        # divide lines left and right
        left_lines = []
        right_lines = []
        for j in range(len(slopes)):
            Line = new_lines[j]
            slope = slopes[j]
            x1, y1, x2, y2 = Line
            if (slope < 0):
                left_lines.append([Line.tolist()])
            elif (slope > 0):
                right_lines.append([Line.tolist()])

        
        ### Left Right Line Test
        for line in left_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for line in right_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            


        ### Find Center 
        x_sum, y_sum, m_sum = 0.0, 0.0, 0.0
        size = len(left_lines)
        for line in left_lines:
            x1, y1, x2, y2 = line[0]
            x_sum += x1 + x2
            y_sum += y1 + y2
            m_sum += float(y2 - y1) / (float(x2 - x1)+0.0001)
            x_avg = x_sum / (size * 2)
            y_avg = y_sum / (size * 2)
            m_left = m_sum / (size)
            b_left = y_avg - m_left * x_avg

            if m_left!=0:
                x1 = int((0.0 - b_left) / m_left)
                x2 = int((155.0 - b_left) / m_left)

        left_center = int((x1+x2)/2)
        cv2.circle(img, (left_center,50), 2, (125,125,125), 3)

    

        x_sum, y_sum, m_sum = 0.0, 0.0, 0.0
        size = len(right_lines)
        for line in right_lines:
            x1, y1, x2, y2 = line[0]
            x_sum += x1 + x2
            y_sum += y1 + y2
            m_sum += float(y2 - y1) / (float(x2 - x1)+0.0001)
            x_avg = x_sum / (size * 2)
            y_avg = y_sum / (size * 2)
            m_right = m_sum / (size)
            b_right = y_avg - m_right * x_avg

            if m_right!=0:
                x1 = int((0.0 - b_right) / m_right)
                x2 = int((155.0 - b_right) / m_right)

        right_center = int((x1+x2)/2)
        cv2.circle(img, (right_center,50), 2, (125,125,125), 3)
            



        center = int((left_center+right_center)/2) 
        cv2.circle(img, (center,50), 4, (0,125,0), 3)


    else: 
        print("No Line")
        self.drive(0,3)

    angle  = int(center-100) # Center: 100
    return angle, img

def go_ahead(self,angle):
    """stop by traffic light
    tracking the lines"""
    self.drive(angle,3)

def stop(self):
    self.drive(0,0)

def turn_left(self,val=30):
    """turn left after traffic light"""
    self.drive(-val,3)

def step_1(self,detection_red,detection_green,angle,current):
    """
    #step 1 : until meeting the first traffic
    input : detection_red,detection_green,angle,current
    output : angle """
    transition = 0
    if detection_red == 0 and detection_green == 0:
        self.drive(angle,3)
        time_check = current 
        time_check_red = current



    # 시작부터 Green인경우 
    elif detection_red == 0 and detection_green == 1 and check_red_point==0:
        if (current-time_check)<2:
            print("type_green_first")
            self.drive(angle,3)
        
        else:
            mission_1_time = current
            transition = 1

    # 빨간불로 멈췄다가 Green인 경우 
    elif detection_red == 0 and detection_green == 1 and check_red_point==1:
        if (current-time_check_red)<1.5:
            print("type_red_first")
            self.drive(0,3)
        else:
            mission_1_time = current
            transition = 1
        
    
    elif detection_red == 1 and detection_green == 0:
        if (current-time_check)<3:
            print("RED")
            self.drive(angle,3)
        else:
            print("RED STOP")
            check_red_point=1
            time_check_red = current
            self.drive(0,0)
    return transition, mission_1_time

def step_2(self,angle_custom,mission_1_time,current):
    """
    #step 2 : turn left and enter the other side of course
    input : angle_custom,mission_1_time,current
    output : transition,mission_time_1
    """
    mission_time_1 = current-mission_1_time
    if mission_time_1<3.5:
        print('TURN')
        self.drive(-angle_custom,3)
        
    else:
        mission_time_1  = current
        transition=2
    return transition,mission_time_1

def step_3(self,angle,current,mission_time_1):
    """
    #step 3 : until meeting the second traffic
    힘으로 돌려
    input : angle_custom,mission_time_1,current
    output : transition,mission_time_2
    """
    mission_time_2 = current - mission_time_1
    if mission_time_2<3:
        print('mission2 drive',mission_time_2)
        self.drive(angle,3)
    
    elif mission_time_2<9:
        print('mission2 drive!!',mission_time_2)
        self.drive(40,3)
    
    elif mission_time_2<10:
        print('mission2 back!!',mission_time_2)
        self.drive(-20,0)

    elif mission_time_2<12:
        print('mission2 drive!!',mission_time_2)
        self.drive(-20,-3)

    elif mission_time_2<20:
        print('mission2 drive!!',mission_time_2)
        self.drive(40,3)
    elif mission_time_2<23:
        print('mission2 drive',mission_time_2)
        self.drive(0,0)
    else:
        mission_time_2 = current
        transition=3
    return transition, mission_time_2

def step_4(self,angle,mission_time_2,current):
    """
    #step 4 : go straigh and enter the other side of course
    input : mission_time_2,current
    output : transition,mission_time_1
    """
    mission_time_3 = current-mission_time_2
    if mission_time_3<5:
        print('mission3 drive',mission_time_3)
        self.drive(20,3)
    
    elif mission_time_3<7:
        print('mission2 drive!!',mission_time_3)
        self.drive(angle,3)

    elif mission_time_3<15:
        print('mission2 drive!!',mission_time_3)
        self.drive(40,3)

    return transition

def step_5(self):
    """
    #step 5 : until meeting the no traffic
    input : mission_time_2,current
    output : transition,mission_time_1
    """
    pass

def step_6(self):
    """
    #step 6 : turn left and enter the other side of course
    input : mission_time_2,current
    output : transition,mission_time_1
    """
    pass

def step_7(self):
    """
    #step 7 : until meeting the success line
    input : mission_time_2,current
    output : transition,mission_time_1
    """
    pass


class Robotvisionsystem:
    def __init__(self,test):
        self.start_time = time.time()
        self.realimage = np.empty(shape=[480, 640, 3])
        self.bridge = CvBridge() 
        self.motor = None 
        self.angle = 0
        self.speed = 0.1
        self.stop = 0
        self.test = test
        self.CAM_FPS = 30
        self.WIDTH, self.HEIGHT = 640, 480

        rospy.init_node('driving')
        
        self.motor = rospy.Publisher('/xycar_motor', xycar_motor, queue_size=1)
        self.real_image = rospy.Subscriber('/usb_cam/image_raw/compressed',CompressedImage, self.realimg_callback)

        print("----- Xycar self driving -----")
        print('!!!')
        if test == 0:
            self.start()
        if test != 0:
            self.start_test(test)

    def realimg_callback(self, data):
        # print data
        try:
            self.realimage = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8") # mono8, mono16, bgr8, rgb8, bgra8, rgba8, passthrough
            # print(self.realimage.shape)
        except CvBridgeError as e:
            print("___Error___")
            print(e)
        
        # np_arr = np.formstring(data, np.unit8)
        # self.realimage = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
    

    def drive(self, angle, speed):
        motor_msg = xycar_motor()
        motor_msg.angle = angle
        motor_msg.speed = speed
        # rospy.loginfo(motor_msg.angle)
        self.motor.publish(motor_msg)


    def start(self):
        prev=0
        detection_red=0
        detection_green=0
        center=100
        transition=0
        check_red_point = 0
        time_check=0

        rotate_1 = 0
        rotate_2 = 0

        while not rospy.is_shutdown(): # Main Loop
            current = time.time()-self.start_time

            # print(self.realimage.shape)
            
            #get frame from camera
            frame = get_frame(self)

            #extracting bbox
            preds,size_x,size_y = extract_bbox(self,frame)

            # about traffic light
            detection_red, detection_green = about_traffic_light(self,frame,preds,size_x,size_y)

            #line detecting
            angle,img = line_detect(self,frame)
            # print('exponential_moving_avg : ',exponential_moving_avg)

            # initialization
            if current<5:
                print('READY')
                self.drive(0,0)
            
            # DRIVING
            elif transition == 0:
                #step 1 : until meeting the first traffic
                transition, mission_1_time = step_1(self,detection_red,detection_green,angle,current)
            elif transition == 1:
                #step 2 : turn left and enter the other side of course
                transition, mission_time_1 = step_2(self,40,mission_time_1,current)
            elif transition == 2:
                #step 3 : until meeting the second traffic
                transition = step_3(self,40,mission_1_time,current)
            elif transition == 3:
                #step4 : go straight and enter the other side of course
                transition, mission_time_1 = step_4(self,40,mission_time_1,current)
            elif transition == 4:
                #step5 : until meeting the no traffic
                transition, mission_time_1 = step_5(self,40,mission_time_1,current)
            elif transition == 5:
                #step6 : turn left and enter the other side of course
                transition, mission_time_1 = step_6(self,40,mission_time_1,current)
            elif transition == 6:
                #step5 : until meeting the success line
                transition, mission_time_1 = step_7(self,40,mission_time_1,current)
                ############



            print("Time : ",current)
            cv2.imshow('img',img)
            cv2.imshow('trafficlight',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cv2.destroyAllWindows()
    
    def start_test(self,test):
        prev=0
        detection_red=0
        detection_green=0
        center=100
        transition=0
        check_red_point = 0
        time_check=0

        rotate_1 = 0
        rotate_2 = 0

        while not rospy.is_shutdown(): # Main Loop
            current = time.time()-self.start_time

            # print(self.realimage.shape)
            
            #get frame from camera
            frame = get_frame(self)

            #extracting bbox
            preds,size_x,size_y = extract_bbox(self,frame)

            # about traffic light
            detection_red, detection_green = about_traffic_light(self,frame,preds,size_x,size_y)

            #line detecting
            angle,img = line_detect(self,frame)
            # print('exponential_moving_avg : ',exponential_moving_avg)

            # initialization
            if current<5:
                print('READY')
                self.drive(0,0)
            
            # DRIVING
            elif test == 1:
                #step 1 : until meeting the first traffic
                transition, mission_1_time = step_1(self,detection_red,detection_green,angle,current)
            elif test == 2:
                #step 2 : turn left and enter the other side of course
                transition, mission_time_1 = step_2(self,40,mission_time_1,current)
            elif test == 3:
                #step 3 : until meeting the second traffic
                transition = step_3(self,40,mission_1_time,current)
            elif test == 4:
                #step4 : go straight and enter the other side of course
                transition, mission_time_1 = step_4(self,40,mission_time_1,current)
            elif test == 5:
                #step5 : until meeting the no traffic
                transition, mission_time_1 = step_5(self,40,mission_time_1,current)
            elif test == 6:
                #step6 : turn left and enter the other side of course
                transition, mission_time_1 = step_6(self,40,mission_time_1,current)
            elif test == 7:
                #step5 : until meeting the success line
                transition, mission_time_1 = step_7(self,40,mission_time_1,current)
                ############



            print("Time : ",current)
            cv2.imshow('img',img)
            cv2.imshow('trafficlight',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cv2.destroyAllWindows()


            
            
if __name__ == '__main__':
    RVS = Robotvisionsystem()
    #step_1 test
    #RVS = Robotcisionsystem(test=1)
    #step_2 test
    #RVS = Robotcisionsystem(test=2)
    #step_3 test
    #RVS = Robotcisionsystem(test=3)
    #step_4 test
    #RVS = Robotcisionsystem(test=4)
    #step_5 test
    #RVS = Robotcisionsystem(test=5)
    #step_6 test
    #RVS = Robotcisionsystem(test=6)
    #step_7 test
    #RVS = Robotcisionsystem(test=7)


