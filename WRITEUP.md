## Model Research

### Model 1: SSD MobileNet V2 COCO (ssd_mobilenet_v2_coco_2018_03_29)

- I converted the model to an IR with following arguments:

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
--input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb \
--tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config \
--reverse_input_channels \
--tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

# Project Write-Up

Text will follow!

## Explaining Custom Layers

The Openvino Toolkit supports several frameworks and their neural network model layers. Although custom layers are all layers that are not in the list of known layers.

Since the model optimizer 


## Comparing Model Performance

Text will follow!


## Assess Model Use Cases

Text will follow!

## Assess Effects on End User Needs
