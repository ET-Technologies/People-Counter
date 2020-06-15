"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import time

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference_project import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-a", "--alarm", type=float, default=7,
                        help="There are just 6 people in the video. Your output is wrong!"
                        "(7 by default)")
    parser.add_argument("-f", "--file", type=str, default="test.csv",
                        help="Here you con specify your output file"
                        "(Default = test.csv)")
    return parser

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

#def alarm(self):
    # Not finished yet
    # if more than e.g. 5 people went into the room a alarm is send.

#def writeincsv(self):
    # Not finished yet
    # Here the output text is writen in a txt file

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    num_requests = 0
    
    # Flag for the single image
    single_image_mode = False

    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    ### TODO: Load the model through `infer_network` ###
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1, num_requests, args.cpu_extension)[1]

    ### TODO: Handle the input stream ###
    
    # Check for Webcam
    if args.input =="CAM":
        input_stream = 0
        
    # Check for Image (jpg, bmp, png)
    elif args.input.endswith(".jpg") or args.input.endswith(".bmp") or args.input.endswith(".png") :
        single_image_mode = True
        input_stream = args.input
        
    # Check for video    
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "There is no video file"

    ### TODO: Loop until stream is over ###
    cap = cv2.VideoCapture(input_stream)
    
    if input_stream:
        cap.open(args.input)
    if not cap.isOpened():
        log.error("Error: No video source")
    
    ### Variables
    total_count = 0
    duration = 0
    person_on_screen = False
    person_count = 0
    no_person_count = 0
    people_count = 0
    duration_time = 0
    width = cap.get(3)
    height = cap.get(4)
    durration_flag = 0
    new_person_flag = 0
    person_leaves_flag = 0
    request_id = 0
    i_start = 0
    person_detected = 0

        ### TODO: Read from the video capture ###
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (w, h))
        # HWC to CHW
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(image,request_id)
        inf_start_time = time.time()

        ### TODO: Wait for the result ###
        if infer_network.wait(request_id) == 0:
            det_time = time.time() - inf_start_time
            current_count = 0
            
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(request_id)
            
            ##Put the inference time in the frame ###
            time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            mes_on_frame_01 = "Model: " + args.model
            mes_on_frame_02 = "Threshold: " + str(prob_threshold)
            cv2.putText(frame,time_message, (15, 15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame,mes_on_frame_01, (15, 30),cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame,mes_on_frame_02, (15, 45),cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)    
            
            for obj in result[0][0]:
                if obj[2] > prob_threshold:
                    current_count = current_count + 1
                    xmin = int(obj[3] * width)
                    xmin = int(obj[3] * width)
                    ymin = int(obj[4] * height)
                    xmax = int(obj[5] * width)
                    ymax = int(obj[6] * height)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255),1)
                    person_detected = 1
            
            if person_detected:
                person_count = person_count + 1
                no_person_count = 0
            
            else:
                no_person_count = no_person_count +1
                person_count = 0
                
            person_detected = 0
            
            if person_count == 5 and person_on_screen == False:
                person_on_screen = True
                current_count = 1
                i_start = time.time()
                person_count = 0
                no_person_count = 0
                durration_flag = 0
            
            elif no_person_count == 5 and person_on_screen == True:
                person_on_screen = False
                current_count = 0
                i_start = time.time() - i_start
                person_count = 0
                no_person_count = 0
                total_count = total_count +1
                duration_time = duration_time + i_start
                i_start = 0
                duration = round(duration_time / total_count)
                durration_flag = 1
        
            client.publish("person", json.dumps({"count":current_count}))
            if durration_flag:
                client.publish("person/duration", json.dumps({"duration": duration}))
                durration_flag = 0

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.write("out.jpg", frame)
            
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()
    


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
