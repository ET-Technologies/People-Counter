#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """
    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        
        self.net= None #network
        iec = None #IECore
        self.input_blob = None #input_blob
        self.out_blob = None #output_blob
        self.net_plugin = None #exec_network
        self.infer_request_handel = None #infer_request
        self.input_shape = None
        
    def load_model(self, model, device, input_size, output_size, num_requests, cpu_extension=None):
        
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0]  + ".bin"
        
         ### TODO: Add any necessary extensions ###
        iec = IECore()
        if cpu_extension and 'CPU' in device:
            iec.add_extension(cpu_extension, device)
            
        # Read the IR as a IENetwork
        log.info("Reading IR ...")
        self.net = IENetwork(model=model_xml, weights=model_bin)
        
        ### TODO: Check for supported layers ###
        if "CPU" in device:
            supported_layers = iec.query_network(self.net, "CPU")
            not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                sys.exit(1)
                      
        if num_requests == 0:
            self.net_plugin = iec.load_network (network = self.net, device_name = device)
        else:
            self.net_plugin = iec.load_network (network = self.net, device_name = device, num_requests = num_requests)
            
        #self.net_plugin = iec.load_network(network = self.net, device_name = device)
        log.info("Loading IR to the plugin ...")
        
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        
        assert len(self.net.inputs.keys()) == input_size, \
            "Supports only {} input topologies".format(len(self.net.inputs))
        assert len(self.net.outputs) == output_size, \
            "Supports only {} output topologies".format(len(self.net.outputs))
        
        return iec, self.get_input_shape()
    
    def get_input_shape(self):
        
        return self.net.inputs[self.input_blob].shape
    
    #def exec_net(self, request_id, frame):
    def exec_net(self, frame, request_id=0):
        
        self.infer_request_handle = self.net_plugin.start_async(request_id=request_id, inputs={self.input_blob: frame})
        
        return self.net_plugin
        
    def wait(self, request_id=0):
        wait_process = self.net_plugin.requests[request_id].wait(-1)
        
        return wait_process
    
    def get_output(self, request_id=0, output=None):
        if output:
            res = self.infer_request_handle.output[output]
        else:
            res = self.net_plugin.requests[request_id].outputs[self.out_blob]
        
        return res
            
    def clean():
        del self.net_plugin
        del self.net
