## Model Research

### Model 1: SSD MobileNet V2 COCO (ssd_mobilenet_v2_coco_2018_03_29)

- I converted the model to an IR with following arguments:

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
--input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb \
--tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config \
--reverse_input_channels \
--tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

# Project Write-Up

## Overview
This project was a very good experience in many ways. First of all, it was a repetition of the previous course. With all the challenges from the model optimizer to the inference engine. Second, I had to deal with many different model types, frameworks and data sets. Questions are as follows: Which model is suitable, on which data set was it trained and how fast is the inference time? And finally, what problems can arise if the model doesn't work as well as we would like, or how I can debug my code if the user interface doesn't work properly.
Still, I couldn't solve all the problems and the openvino toolkit revealed some problems, but still the project was very interesting and I'm looking forward to the next one.


## Explaining Custom Layers

The Openvino Toolkit supports several frameworks and their neural network model layers. Although custom layers are all layers that are not in the list of known layers.

The **Model Optimizer** workflow contains following steps:

- The model optimizer searches the list of known layers for each layer in the input model before buidling the model internal representation.
- Optimize the model
- Produce the IF files

The **Inference Engine** workflow contains following steps:

- Loads the IR files into the specified device plugin
- Searches a list of known layer implementations for the device
- If that list contains layers not on that list the IE reports an error

### Custom Layer implementation
If you need to implement a custom layer for your model, you will need to add extensions to the Model Optimizer and the Inference Engine.
#### Custom Layer for the Model Optimizer
The MO extracts Information from the input model which includes 
- the topology of the model layers 
- parameters
- input and output format ...

If the MO detects an custom layer the libary must be extended. The custom layer extensions are:
- Custom Layer Extractor
- Custom Layer Operation

#### Custom Layer Extensions for the Inference Engine

The custom layer extensin in the IE is implemented according to the target device:
- Custom Layer CPU Extension
- Custom Layer GPU Extension


## Comparing Model Performance

Text will follow!


## Assess Model Use Cases

Text will follow!

## Assess Effects on End User Needs
