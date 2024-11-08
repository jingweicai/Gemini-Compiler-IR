# Frontend IR Format Description

## Document Notation
### Filename
The filename indicates the model name, the batch size, and the data type of the model. For example, `"int8_resnet50.onnx.sim_b4.json"` represents the Fontend IR for a ResNet50 with a batch size of 4 and a data type of INT8, converted from ONNX. The model may be quantized.

### Symbols
- **N**: Batch size
- **K**: Output channel number
- **C**: Input channel number
- **H**: Height of the ifmap/ofmap tensor
- **W**: Width of the ifmap/ofmap tensor
- **R**: Kernel height
- **S**: Kernel width

In the tensors appearing in this IR, unless otherwise specified, the data layout for ofmap is **`[N, K, H, W]`**, for ifmap it is **`[N, C, H, W]`**, and for weights it is **`[K, C, R, S]`**.

## Overall Structure
```json
{
    "Conv_0": {
        ...
    },
    "MaxPool_2_1": {
        ...
    },
    ...
}
```

Frontend IR contains many layers, and most keys are the same for each layer, but some layers have additional parameters for easier parsing. Here, we'll look at two examples:

### Example 1: Conv_3

```json
"Conv_0": {
    "layer_index": 0,
    "name": "Conv_0",
    "operation": "conv2d",
    "device": "npu",
    "input_dtype": [
        "float16",
        "float16"
    ],
    "output_dtype": [
        "float16"
    ],
    "input_shape": [
        [ 1, 3, 224, 224 ],
        [ 64, 3, 7, 7 ]
    ],
    "output_shape": [
        [ 1, 64, 112, 112 ]
    ],
    "previous_layer": [
        "input1",
        "constant_497"
    ],
    "next_layer": [
        "MaxPool_2_1"
    ],
    "file_list": {
        "input_activation1": "Conv_0_input_activation1.npy",
        "input_activation2": "Conv_0_input_activation2.npy",
        "output_activation1": "Conv_0_output_activation1.npy",
        "k": "Conv_0_k.npy",
        "b": "Conv_0_b.npy"
    },
    "input_batchdim": [ 0 ],
    "output_batchdim": [ 0 ],
    "activation_type": "relu",
    "activation_attr": null,
    "kernel_size": [ 7, 7 ],
    "padding": [ 3, 3, 3, 3 ],
    "stride": [ 2, 2 ],
    "ori_name": "Relu_1"
},
```
Here’s a breakdown of each key in the Frontend IR of the `Conv_0` layer:

- **`"layer_index"`**: `0`  
  Indicates the position of the layer in the model. This is the first layer, hence `0`.

- **`"name"`**: `"Conv_0"`  
  The name of the layer. In this case, it's named `Conv_0`, representing a convolutional layer.

- **`"operation"`**: `"conv2d"`  
  Specifies the type of operation performed by this layer. In this case, it's a 2D convolution (`conv2d`).

- **`"device"`**: `"npu"`  
  Indicates the device where this operation will be executed. In this case, it's an NPU (Neural Processing Unit).

- **`"input_dtype"`**: `["float16", "float16"]`  
  Specifies the data types of the inputs. This layer takes two inputs, both of type `float16`.

- **`"output_dtype"`**: `["float16"]`  
  Specifies the data type of the output. The output of this layer is `float16`.

- **`"input_shape"`**: `[[1, 3, 224, 224], [64, 3, 7, 7]]`  
  Represents the shapes of the inputs. 
  - The first input has a shape of `[1, 3, 224, 224]` (batch size 1, 3 channels, 224x224 spatial dimensions).
  - The second input has a shape of `[64, 3, 7, 7]` (representing 64 filters, 3 input channels, and 7x7 kernel size).

- **`"output_shape"`**: `[[1, 64, 112, 112]]`  
  Specifies the shape of the output tensor, which is `[1, 64, 112, 112]` (batch size 1, 64 channels, 112x112 spatial dimensions).

- **`"previous_layer"`**: `["input1", "constant_497"]`  
  Lists the layers or inputs that provide data to this layer. This layer takes input from `"input1"` and a constant tensor `"constant_497"`.

- **`"next_layer"`**: `["MaxPool_2_1"]`  
  Specifies the next layer(s) in the model. After the `Conv_0` layer, the output is passed to the `MaxPool_2_1` layer.

- **`"file_list"`**:  
  A list of files associated with this layer, typically storing activations, weights, or biases:
  - `"input_activation1"`: `"Conv_0_input_activation1.npy"` (file containing the first input activation)
  - `"input_activation2"`: `"Conv_0_input_activation2.npy"` (file containing the second input activation)
  - `"output_activation1"`: `"Conv_0_output_activation1.npy"` (file containing the output activation)
  - `"k"`: `"Conv_0_k.npy"` (file containing the convolutional kernel)
  - `"b"`: `"Conv_0_b.npy"` (file containing the bias)

- **`"input_batchdim"`**: `[0]`  
  Indicates the batch dimension for the inputs. Here, the batch dimension is the 0th axis.

- **`"output_batchdim"`**: `[0]`  
  Indicates the batch dimension for the output. The batch dimension is also the 0th axis for the output.

- **`"activation_type"`**: `"relu"`  
  Specifies the type of activation function applied to the output of this layer. In this case, it's the ReLU (Rectified Linear Unit) activation function.

- **`"activation_attr"`**: `null`  
  Attributes related to the activation function. It's set to `null`, meaning there are no additional attributes for the ReLU activation.

- **`"kernel_size"`**: `[7, 7]`  
  Indicates the size of the convolutional kernel, which is 7x7.

- **`"padding"`**: `[3, 3, 3, 3]`  
  Specifies the padding applied to the input. This layer applies a padding of 3 on all sides (top, bottom, left, right).

- **`"stride"`**: `[2, 2]`  
  Indicates the stride used for the convolution operation. A stride of 2 is applied in both the height and width dimensions.

- **`"ori_name"`**: `"Relu_1"`  
  The original name of this layer in the model, possibly indicating that it was named `Relu_1` before being renamed `Conv_0` during optimization or conversion.



### Example 2: MaxPool_2_1

```json
"MaxPool_2_1": {
    "layer_index": 1,
    "name": "MaxPool_2_1",
    "operation": "max_pool2d",
    "device": "npu",
    "pool_size": [ 3, 3 ],
    "padding": [ 1, 1, 1, 1 ],
    "stride": [ 2, 2 ],
    "ceil_mode": 0,
    "file_list": {
        "input_activation1": "MaxPool_2_1_input_activation1.npy",
        "output_activation1": "MaxPool_2_1_output_activation1.npy"
    },
    "previous_layer": [ "Conv_0" ],
    "next_layer": [ "Conv_3", "Conv_8" ],
    "input_shape": [
        [ 1, 64, 112, 112 ]
    ],
    "output_shape": [
        [ 1, 64, 56, 56 ]
    ],
    "input_dtype": [ "float16" ],
    "output_dtype": [ "float16" ],
    "tensor0_lut": [ 1, 1, 1 ],
    "module_list": {
        "pre_cal": {
            "input_shape": [
                [ 1, 64, 112, 112 ]
            ],
            "output_shape": [
                [ 1, 64, 112, 112 ]
            ],
            "requan_scale_a": 1,
            "requan_zp_a": 0
        },
        "compare_group": {
            "cmp_mode": "cumulative",
            "cmp_max_en": "max",
            "cmp_dim": "kernel",
            "input_shape": [
                [ 1, 64, 112, 112 ]
            ],
            "output_shape": [
                [ 1, 64, 56, 56 ]
            ]
        },
        "post_cal": {
            "input_shape": [
                [ 1, 64, 56, 56 ]
            ],
            "output_shape": [
                [ 1, 64, 56, 56 ]
            ],
            "mul_k": 1,
            "sum_b": 0,
            "sumsub_mode": "add"
        }
    },
    "ori_name": "MaxPool_2",
    "input_batchdim": [ 0 ],
    "output_batchdim": [ 0 ]
},
```

Some operators are computed on the Vector Processor inside the NPU, such as MaxPool, scalar, etc. Most of the content was explained in [Example 1](#example-1-conv_3), so we will only discuss the parts that were not covered previously.

- **`"pool_size"`**: `[3, 3]`  
  Specifies the size of the pooling window, which in this case is 3x3 for max pooling.

- **`"ceil_mode"`**: `0`  
  Indicates whether the pooling operation uses ceiling or floor for computing output dimensions. A value of `0` means floor is used.

- **`"tensor0_lut"`**: `[1, 1, 1]`  
  Some dimensions of the Vector Processor (VP) operator may not be used, such as the W dimension for a 1D vector. However, for consistency, all operators are represented as 3D tensors (C, H, W). A value of `1` indicates that the corresponding dimension is enabled. In this case, all dimensions (C, H, W) are enabled, as indicated by `[1, 1, 1]`.

- **`"module_list"`**:  
  Contains modules and corresponding configs for the VP calculation, including pre-processing, VP ops, and post-processing. Note that although there are many modules in the VP, but typically only some of them are enabled at the same time.
  
  - **`"pre_cal"`**:  
    Pre-calculation module.
    - **`"input_shape"`**: `[[1, 64, 112, 112]]`  
      The input shape for this stage of the computation.
    - **`"output_shape"`**: `[[1, 64, 112, 112]]`  
      The output shape for the pre-calculation stage, which remains the same in this case.
    - **`"requan_scale_a"`**: `1`  
      Scaling factor for quantization.
    - **`"requan_zp_a"`**: `0`  
      Zero-point for quantization.
  
  - **`"compare_group"`**:  
    Handles the comparison for max pooling on the compare module.
    - **`"cmp_mode"`**: `"cumulative"`  
      The mode for comparison, which is cumulative across the pooling window.
    - **`"cmp_max_en"`**: `"max"`  
      Enables maximum comparison, which is typical for max pooling.
    - **`"cmp_dim"`**: `"kernel"`  
      Specifies that the comparison is done over the kernel (pooling window).
    - **`"input_shape"`**: `[[1, 64, 112, 112]]`  
      Input shape for the comparison stage.
    - **`"output_shape"`**: `[[1, 64, 56, 56]]`  
      Output shape after the comparison step, which reduces the spatial dimensions.

  - **`"post_cal"`**:  
    Post-calculation module, often used for scaling, bias addition, or other adjustments after the pooling operation.
    - **`"input_shape"`**: `[[1, 64, 56, 56]]`  
      Input shape for the post-calculation stage.
    - **`"output_shape"`**: `[[1, 64, 56, 56]]`  
      Output shape remains unchanged after post-calculation.
    - **`"mul_k"`**: `1`  
      Multiplication factor applied during post-calculation (no scaling applied here, as it is `1`).
    - **`"sum_b"`**: `0`  
      Bias added to the result, which is `0` in this case.
    - **`"sumsub_mode"`**: `"add"`  
      Specifies that the operation performed is addition, which adds the bias to the result (though the bias is `0` here).