# Project Write-Up

## Overview
This project was a very good experience in many ways. First of all, it was a repetition of the previous course. With all the challenges from the model optimizer to the inference engine. Second, I had to deal with many different model types, frameworks and data sets. Questions accrued like: Which model is suitable, on which data set was it trained and how fast is the inference time? And finally, what problems can arise if the model doesn't work as well as we would like, or how I can debug my code if the user interface doesn't work properly.
Though, I could finish the project, but I couldn't solve all problems which would accure in the real world. But still the project was very interesting and I'm looking forward to the next one.

## Explaining Custom Layers
The Openvino Toolkit supports several frameworks and their neural network model layers. Although each of the different frameworks has different requirements and specifications. Custom layers are therefore all layers that are not included in the list of known layers, but are still required by the framework. Custom layers help the model optimizer to understand the network and remove the restriction if someone wants their own function that would otherwise not be recognized by the model optimizer.

The **Model Optimizer** workflow contains following steps:

- The model optimizer searches the list of known layers for each layer in the input model before buidling the model internal representation.
- Optimize the model
- Produce the IF files

The **Inference Engine** workflow contains following steps:

- Loads the IR files into the specified device plugin
- Searches a list of known layer implementations for the device
- If that list contains layers not on that list the IE reports an error

#### Custom Layer implementation
If you need to implement a custom layer for your model, you will need to add extensions to the Model Optimizer and the Inference Engine.

## Comparing Model Performance

Text will follow!


## Assess Model Use Cases

The People Counter app can be used in many ways. For example, you could put a camera in front of a shop, bank, or train station.
The app can count how many people are in the facility. This amount can be compared to the number of people allowed in the facility. If there are too many people, an alarm is sent. A pipeline of other models could be implemented. Example: Corona mask recognition, range finder or face recognition. It can be used as an access control system or restricted area control app. The number of use cases is innumerable.

## Assess Effects on End User Needs

At the start of this project, I wasn't aware of how many parameters could affect a good output. At first I had problems with the right model. It turns out that out-of-the-box models didn't perform as well as pretrained models. You have to consider frameworks, data sets and model size. These parameters are particularly important for AI on the edge applications. Large models have a higher inference time and also higher computing power. Small models are often less accurate, so this is a compromise. The accuracy also comes with the quality of the video footage. If the light is poor or the focal length is not adjusted, this can affect the output of the model. Therefore, before building an application, you need to consider how these compromises affect accuracy and how to fix or limit the bad impact.

## Model Research

### Explainatian:
I had the most problems with this part. I converted some models using the model optimizer, but the output was not satisfactory.
First, I was unable to convert some of these models. Part of it was probably my fault, but then the model optimizer also had some problems with it. Second, I was faced with different datasets types. As a mentor told me, an SSD trained on Coco Dataset was unable to perform with this project. He was right, but I was still confused because many of my fellow students used the same model. Third, I was confused with different input- and output_blobs of the model. And to be honest, this was one of my biggest problems. I couldn't find the right settings. I'm not going to dissuade myself from blaming bad documentation, it's just a lesson I still have to learn.

### Models used:
In investigating potential people counter models, I tried each of the following three models:

#### Model 1: SSD MobileNet V2 COCO (ssd_mobilenet_v2_coco_2018_03_29)

##### [Model Source]:
- http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

##### Results:
- Inference time: 61-73 ms
- Total counted people: 8
- Duration:00:30
- Threshold: 0,2
- Comments:
The model had problems with Person 2, so I set the threshold to 0.2.
Still, the issue was not solved and the second person was not counted correctly.
One of my mentors later told me that a coco-trained model was not good for this job.

#### Model 2: ssd_inception_v2_coco_2018_01_28

##### [Model Source]:
- http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz

##### Results:
- Inference time: 61-73 ms
- Total counted people: 8
- Duration:00:30
- Threshold: 0,2
- Comments:
The model had problems with Person 2, so I set the threshold to 0.2.
Still, the issue was not solved and the second person was not counted correctly.
One of my mentors later told me that a coco-trained model was not good for this job.

#### Model 3: ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03

##### [Model Source]:
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz

##### Results:
- Inference time: ~ 1840ms
- Total counted people:
- Duration:
- Threshold:
- Comments

##### I converted the model to an IR with following arguments:
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
--input_model ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb \
--tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config \
--reverse_input_channels \
--tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

