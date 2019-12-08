# Deployment of FaceNet Tensorflow model with OpenCV Dnn Tools 

This project will show you how to deploy a pretrain [tf faceNet model](https://github.com/davidsandberg/facenet) using the OpenCV-Dnn tools. 

* OpenCV Dnn tools will give you a 10x inference speed up on CPU. 
* Tips for convert a standard tensorflow model to a opencv readable ones will be clarify.

## Requirments 
* *python 3*
* *OpenCV 4.1.2*
* *Test on Linux / MacOS*

## Offical pre-trained faceNet models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

## Key Steps for OpenCV-Dnn deployment

> Key Steps for OpenCV-Dnn deployment

>> [1] Tensorflow `.pb` model should be in the deployment phase before being used.
>>> * Unused nodes should be remove.
>>> * BatchNorm layers should be folded and turned to testing phase completely, which means nodes like 'switch' should be remove from graph
![BatchNorm layers in faceNet .pb](https://user-images.githubusercontent.com/25801568/54905709-ba876b80-4ef2-11e9-809c-46da9aeb159f.png)

>> [2] Trim `phase_train` node in faceNet models. 
>>> when run in inference node, the input node `phase_train` can be convert to a constant value: `False`.

## Usage

### Convert
***Convert official faceNet .pb model to cv-dnn readable ones, before feeding it to OpenCV dnn***

#### [Step 1] Trim 'phase_train' node
```bash
python convert_variable_to_constant.py
```
* your can find more detail of this code in [here](https://stackoverflow.com/questions/43332342/is-it-possible-to-replace-placeholder-with-a-constant-in-an-existing-graph)

#### [Step 2] Transform for Inference mode.
```bash
python convert_tf_pb_to_cv_pb.py
```

a. Load graph from .pb
```text 
graph_def = tf.GraphDef()
with tf.gfile.FastGFile(INPUT_GRAPH_DEF_FILE, 'rb') as f:
    graph_def.ParseFromString(f.read())
```
b. Remove 'Switch' and 'Merge' node in graph  
[reference](https://github.com/opencv/opencv/issues/14224)
```text
for i in reversed(range(len(graph_def.node))):
    op = graph_def.node[i].op
    name = graph_def.node[i].name

    if op == 'Switch' or op == 'Merge':
        # get input of 'Switch'/'Merge' node
        inp = graph_def.node[i].input[0]

        # find nodes connected to 'Switch'/'Merge' node in graph,
        # cut their connection, and redirect to the input of 'Switch'/'Merge';
        for node in graph_def.node:
            for j in range(len(node.input)):
                if name == node.input[j]:
                    node.input[j] = inp
        del graph_def.node[i]
```

c. Turned model to testing phase  
```text
graph_def = TransformGraph(graph_def,
                               ['input'],
                               ['embeddings'],
                               [
                                   'strip_unused_nodes(type=float, shape="1,160,160,3")',
                                   'remove_nodes(op=Identity, op=CheckNumerics, op=PlaceholderWithDefault)',
                                   'fold_constants(ignore_errors=true)',
                                   'sort_by_execution_order',
                                   'fold_batch_norms',
                                   'fold_old_batch_norms',
                                   'remove_device'
                               ]
                               )
```
* You can get more details on how to use the 'TransformGraph' tool from [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms)

> Tips :
>> [1] Do not use 'optimize_for_inference_lib' tools  
>>> 'optimize_for_inference_lib' tool can't handle the diff versions of 'batchNorm' layers in Tensorflow, which may cause 'can't find BatchNorm Error'  

>> [2] 'shape' value in 'strip_unused_nodes()' should be consistent with your deploying 'batch-size'  
>>> Which means the 'batch-size' is not on demand. I try to feed the dnn model with diff batch size of images, but failed. Maybe you can help me out?  


d. (option) get .pbtxt file
```text
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].op == 'Const':
            del graph_def.node[i]
        for attr in ['T', 'data_format', 'Tshape', 'N', 'Tidx', 'Tdim',
                     'use_cudnn_on_gpu', 'Index', 'Tperm', 'is_training',
                     'Tpaddings']:
            if attr in graph_def.node[i].attr:
                del graph_def.node[i].attr[attr]
```
#### [Step 3] Extract face feature.
```bash
python extract_face_feature.py
```

face image preprocess
```text
im = cv2.imread(face_image_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
resized = cv2.resize(im, (160, 160), interpolation=cv2.INTER_LINEAR)
prewhitened = prewhiten(resized)
# HWC -> CHW
input_face_img = prewhitened.transpose([2, 0, 1])
# CHW -> NCHW
input_face_img = np.expand_dims(input_face_img, axis=0)
```

forward
```text
""" Load .pb model """
cvNet = load_model(model_path)

""" Forward """
cvNet.setInput(input_face_img)
stime = time.time()
cvOut = cvNet.forward()
etime = time.time()
```
> Tips
>> Do not use 'blobfromimage()' or 'blobfromimages()'. Cus after 'prewhitened', dtype of face image convert from 'uint8' to 'float'.  
>> 'blobfromimage' will run 'cv2.resize' by default, but 'cv2.resize' require a 'uint8' image  

## What i have(not) try

* I have try cv.__version__ < 3.4.5 to deploy the converted model, all i got is 'readNet error'. So, I personally recommend  cv.__version__ > 4.0.0   
* I have try to use 'blobfromimage()' feeding the face image, 'cvConvert' error appears. It's probably a dtype error.  

## Performance
test on macbook-pro 2018, Speed up more than 10X 

|Tools       | Tensorflow CPU API | OpenCV_Dnn |
|------------|--------------------|------------|
| Inference Time(for one image) | ~1s     | ~0.09s   |

## Topics on this issue that may help 

> OpenCV forum
```text
[1]https://answers.opencv.org/question/183507/opencv-dnn-import-error-for-keras-pretrained-vgg16-model/
[2]https://answers.opencv.org/question/173293/cv-dnn-caffe-model-with-two-inputs-of-different-size/
```
> Github OpenCV 
```text
[1]https://github.com/opencv/opencv/issues/11452  
[2]https://github.com/opencv/opencv/issues/14073
[3]https://github.com/opencv/opencv/issues/14224
```



