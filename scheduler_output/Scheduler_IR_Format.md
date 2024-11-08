# Scheduler IR Format Description

This is the output of our Gemini scheduler after Address Allocation and IR gen. 

## Document Notation
### Filename
The prefix in the filename comes from the front-end. For example, `"int8_resnet50.onnx.sim_b4_c1_bw16_stschedule.json"` represents the scheduling result for a batch size of 4, with 1 core, and a DRAM bandwidth of 16 GB/s.

### Symbols
- **N**: Batch size
- **K**: Output channel number
- **C**: Input channel number
- **H**: Height of the ifmap/ofmap tensor
- **W**: Width of the ifmap/ofmap tensor
- **R**: Kernel height
- **S**: Kernel width

In the tensors appearing in this IR, unless otherwise specified, the data layout for ofmap is **`[N, K, H, W]`**, for ifmap it is **`[N, C, H, W]`**, and for weights it is **`[K, C, R, S]`**.

## TOP Level Keys

```json
{
   "-1" : {
      "in" : [
         ...
      ],
      "out" : [
         ...
      ]
   },
   "0" : [
      ...
   ],
   "buffersize" : 8388608,
   "top_batch_cut" : 1,
   "xlen" : 1,
   "ylen" : 1
}

```

The multi-core IR is provided in JSON format, with the following **top-level keys**: 

- **`"core_id"`**: `"-1"` contains all the data related to DRAM, while `"0"` and larger keys represent the `core_id`. In `"-1"`, DRAM tensors are grouped into `"in"` and `"out"`, whereas in `"0"`, each element represents a workload (a COMP tile and its configuration).
- **`"buffersize"`**: the size of the L2 buffer (in Bytes).
- **`"top_batch_cut"`**: the size of the num_bgrp at the root node.
- **`"xlen"` and `"ylen"`**: the mesh arrangement of the cores.

## DRAM Data

The key `"in"` represents the data input to DRAM, generally ofmap data, while `"out"` represents the data output from DRAM, including weights, ifmap, constants, etc. 

Both `"in"` and `"out"` are in list format, with each element in the list corresponding to a block of data. Each block is also represented in JSON format, where the keys provide specific information related to the corresponding data.

Here is an example:

```json
"-1" : {
    "in" : [
        {
            "core_id" : 0,
            "layer_name" : "Conv_1",
            "lower" : [ 2, 0, 0, 0 ],
            "related_ofmap" : [10, 13],
            "transfer_id" : 11,
            "upper" : [ 3, 127, 13, 27 ],
            "workload_id" : 2
        },
        ...
    ],
    "out" : [
        {
            "destination" : [
                {
                    "core_id" : 0,
                    "layer_name" : "Conv_2",
                    "type" : "core",
                    "workload_id" : 4
                }
            ],
            "lower" : [ 0, 0, 0, 0 ],
            "related_ifmap" : [],
            "size" : 27136,
            "transfer_id" : 0,
            "type" : "weight",
            "upper" : [ 0, 63, 2, 48 ]
        },
        {
            "destination" : [
                {
                    "core_id" : 0,
                    "layer_name" : "Conv_2",
                    "type" : "core",
                    "workload_id" : 4
                }
            ],
            "lower" : [ 0, 0, 0, 0 ],
            "related_ifmap" : [ 7, 8 ],
            "size" : 107520,
            "transfer_id" : 6,
            "type" : "fmap",
            "upper" : [ 1, 127, 14, 27 ]
        },
        ...
    ]
},
```

Let's focus on the DRAM tensors shown in `IR["-1"]["in"][0]` above. 

- This DRAM tensor is the ofmap for the `Conv_1` layer, generated in the workload with `"workload_id"` `2` on core `0`.
- **`"lower"` and `"upper"`** are the absolute coordinates of this ofmap DRAM tensor within the entire layer's ofmap.
- **`"transfer_id"`** uniquely identifies the transfer of a DRAM tensor and appears in both the source and destination of the data. For this example, the `"transfer_id"` is `11`, so there must be another DRAM tensor with the same `"transfer_id"` (`11`) in the ofmap of some other workload.
- **`"related_ofmap"`** refers to the `"transfer_id"` of a DRAM tensor in the DRAM out that is related to this DRAM tensor. It could refer to the entire block or only a portion of the current DRAM tensor.

Now we explain `IR["-1"]["out"][0]` and `IR["-1"]["out"][1]`:

- **`"destination"`** indicates which workloads use this DRAM tensor, specifying the `"core_id"` and `"workload_id"` of the destination. If the DRAM tensor has multiple destinations, they are added to the list sequentially. 
- **`"type"`** represents the input type (weight/fmap).
- **`"lower"` and `"upper"`** represent the absolute coordinates of this DRAM tensor in the big ifmap of its corresponding layer. 
- **`"related_ifmap"`** refers to the `"transfer_id"` of the related DRAM tensor in `IR["-1"]["in"]`. If the current DRAM tensor overlaps with multiple DRAM tensors in `IR["-1"]["out"]`, the number of elements in the list will be greater than one.
- **`"size"`** represents the size of the current DRAM tensor, with the unit in Bytes. Note that some tensors are aligned, and weight tensors contain BatchNorm parameters, so simply multiplying every dimension of `"upper" - "lower"` may not equal `"size"`.

## Core Data

```json
"0" : [
    {
        "buffer" : [
            ...
        ],
        "ifmap" : [
            ...
        ],
        "layer_name" : "Conv_0",
        "layer_type" : "pe",
        "ofmap" : [
            ...
        ],
        "ofmap_size" : 1605632,
        "ring_buffer_info" : [
            [ 0, 8388608 ]
        ],
        "tile_info" : {
            ...
        },
        "tile_padding_tblr" : [ 3, 2, 3, 2 ],
        "time" : 154590,
        "weight" : {
            "lower" : [ 0, 0, 0, 0 ],
            "size" : 27136,
            "transfer_id" : [ 0 ],
            "upper" : [ 0, 63, 2, 48 ]
        },
        "wl0_buffer" : [
            ...
        ],
        "workload" : [
            [ 0, 0, 0, 0 ],
            [ 1, 63, 111, 111 ]
        ],
        "workload_id" : 0
    },
    ...
]
```

A core workload (that is, a COMP tile & related config) contains the following information:

- **`"workload_id"`**: Workloads are processed sequentially in ascending order of `"workload_id"` on the same core. The `"workload_id"` and computation sequence are independent across different cores.
- **`"layer_name"`**: The name of the layer corresponding to the current workload.
- **`"layer_type"`**: The module to which the layer belongs, categorized by hardware definition as `"pe"` (processing element), `"vp"` (vector processor), or `"dt"` (data transfer).
- **`"tile_info"`**: Information about the tiling returned from Intra-Core Compiler, including `"loop_order"`, `"cut_mode"`, and other details. This will be explained in more detail later.
- **`"time"`**: Estimated running time of this workload. This is caculated based on `"tile_info"`.

- **`"ifmap"`**: The ifmap of the current workload, including information such as `"lower"`, `"upper"`, and `"transfer_id"`. More details will be explained later.
- **`"tile_padding_tblr"`**: the padding info of *this workload's* ofmap

- **`"ofmap"`**: The ofmap of the current workload and its destination. The output of the current layer may be sent to DRAM or forwarded to another core as input for the next layer. More details will be explained later.
- **`"workload"`**: The `"lower"` and `"upper"` coordinates of *this workload's* ofmap.
- **`"ofmap_size"`**: The size of *this workload's* ofmap.
- **`"ring_buffer_info"`**: We use a configurable ring buffer. In each region, once the address exceeds the size of that region, it will wrap around and start from 0 again. This info shows the ring buffer regions.

- **`"buffer"`**: A snapshot of the L2 buffer, recording the tensors and address allocations on the L2 buffer at the current time *before this COMP tile starts*. Typically, it includes the ifmap/ofmap/weight of the current layer or prefetched ifmap/weight. This will be explained in more detail later.
- **`"wl0_buffer"`**: A snapshot of the WL0 buffer, recording the tensors and address allocations on the WL0 buffer at the current time *before this COMP tile starts*. **It only contains weights.** Our WL0 can directly read weight data from DRAM or from the L2 Buffer.

### `"tile_info"`

Our Intra-Core Compiler searches for the optimal way to execute a workload within a single core and records the intra-core scheduling results in **`"tile_info"`**. There may be multiple intra-core tiles in this workload.

```json
"tile_info" : {
    "tile_num_0" : {
        "act_mini_tile" : [ 1, 1, 9, 9 ],
        "ci_cut_num" : 1,
        "cut_mode" : "KH",
        "ifmap_lower" : [ 0, 0, 0, 0 ],
        "ifmap_upper" : [ 0, 2, 223, 223 ],
        "layer_name" : "Conv_0",
        "loop_order" : "KNHW",
        "mtc0_mtk0" : [ 1, 1 ],
        "ofmap_lower" : [ 0, 0, 0, 0 ],
        "ofmap_upper" : [ 0, 63, 111, 111 ],
        "out_mini_tile" : [ 1, 8, 2, 2 ],
        "single_tile_time_pred" : 77373.94499999999,
        "tile_padding_tblr" : [ 3, 2, 3, 2 ]
    },
    ...
},
```

### `"ifmap"`

This workload's ifmap tensors on the L2 buffer. Note that a workload could have multiple ifmaps, and for every ifmap:
- **`"align"`**: The number of channels to which this tensor needs to be aligned, default using 64-bit alignment.
- **`"bitwidth"`**: The bit width of this tensor, where int8 corresponds to 8, and fp16/int16 corresponds to 16.
- **`"lower"`** and **`"upper"`**: The absolute coordinates of this tensor in the ifmap of this workload, represented by four dimensions: N, C, H, and W.
- **`"size"`**: The size of this tensor, in Bytes.
- **`"transfer_id"`**: The transfer ID associated with this tensor. If this tensor comes from DRAM, there will be a corresponding tensor with the same **`"transfer_id"`** in `IR["-1"]["out"]`. If this tensor comes from another core, the corresponding **`"transfer_id"`** tensor can be found in the destination of the ofmap of that core.

```json
"ifmap" : [
    {
        "align" : 8,
        "bitwidth" : 8,
        "lower" : [ 0, 0, 0, 0 ],
        "size" : 401408,
        "transfer_id" : [ 55 ],
        "upper" : [ 0, 2, 223, 223 ]
    }
],
```

### `"ofmap"`

`"ofmap"` is similar to `"ifmap"`, but without `"align"` and has `"destination"`. `"type"` can be either core or DRAM. For example, in the following JSON, the tensor from the current workload’s ofmap in the range `[0,0,0,0]~[0,63,111,111]` will be sent to `workload_id  2` and DRAM. In the buffer snapshot of `workload_id 2`, there will be an ifmap tensor with `transfer_id 56`.

```json
"ofmap" : [
    {
        "destination" : [
            {
                "core_id" : 0,
                "type" : "core",
                "workload_id" : 2
            },
            {
                "core_id" : -1,
                "type" : "DRAM"
            }
        ],
        "lower" : [ 0, 0, 0, 0 ],
        "size" : 802816,
        "transfer_id" : 56,
        "upper" : [ 0, 63, 111, 111 ]
    }
],
```

### `"buffer"`

The buffer snapshot is a list, where each element represents a tensor on the L2 buffer when this workload starts. Most keys in entries of `"buffer"` are similar to those in `"ifmap"` and `"ofmap"`, so we will only explain the keys that are not present in those.

- **`"tensor_id"`**: The ID of this tensor.
- **`"tensor_order"`**: The absolute order of the tensor in the DRAM access queue, as determined by Gemini.
- **`"layer_name"`**: The name of the layer to which the workload containing this tensor belongs.
- **`"address"`**: The starting address of this tensor in L2, in Bytes.
- **`"newly_added"`**: Indicates whether this tensor was newly added at the beginning of this workload.
- **`"cur_wl_ifmap"`**: Specifies whether this is the ifmap for the current workload. This mainly helps the assembler quickly locate the dependencies of the current workload, as there may be many ifmaps in the buffer without individually recording their destinations.
- **`"type"`**: The type of data, which can be ifmap, ofmap, or weight.
- **`"source"`**: Represents the source of this tensor, formatted as a list. 
    - **If the data comes from DRAM**, the `"source"` list will contain only one element with the type `"DRAM"`, and the `core_id` will be `"-1"`. The `"lower"` and `"upper"` in this element represent the range of the current tensor, consistent with the overall `"lower"` and `"upper"` outside `"source"`.
    - **If the data comes from a core**, the intersection between this tensor and the ofmap of the `"source_layer"` on each core will be calculated and stored in the `"source"` list. In such case, each element in the `"source"` list represents a tensor with a unique `"transfer_id"`. The union of the `"transfer_id"` values in `"source"` forms the `"transfer_id"` list outside `"source"`. The union of the tensors (represented by `"lower"` and `"upper"`) in `"source"` forms the overall `"lower"` and `"upper"`  outside `"source"`.
```json
"buffer" : [
    {
        "address" : 0,
        "align" : 8,
        "bitwidth" : 8,
        "layer_name" : "Conv_0",
        "lower" : [ 0, 0, 0, 0 ],
        "newly_added" : false,
        "size" : 27136,
        "source" : [
            {
                "core_id" : -1,
                "lower" : [ 0, 0, 0, 0 ],
                "size" : 27136,
                "transfer_id" : 0,
                "type" : "DRAM",
                "upper" : [ 0, 63, 2, 48 ]
            }
        ],
        "tensor_id" : 2,
        "tensor_order" : 1,
        "transfer_id" : [ 0 ],
        "type" : "weight",
        "upper" : [ 0, 63, 2, 48 ]
    },
    {
        "address" : 428544,
        "align" : 8,
        "bitwidth" : 8,
        "cur_wl_ifmap" : true,
        "lower" : [ 0, 0, 0, 0 ],
        "newly_added" : true,
        "size" : 802816,
        "source" : [
            {
                "core_id" : 0,
                "layer_name" : "Conv_0",
                "lower" : [ 0, 0, 0, 0 ],
                "size" : 802816,
                "transfer_id" : 56,
                "type" : "core",
                "upper" : [ 0, 63, 111, 111 ]
            }
        ],
        "tensor_id" : 3,
        "tensor_order" : 80,
        "transfer_id" : [ 56 ],
        "type" : "ifmap",
        "upper" : [ 0, 63, 111, 111 ]
    },
    ...
],
```

### `"wl0_buffer"`

`"wl0_buffer"` is much simpler than `"buffer"` and are more similar to `"ifmap"` and `"ofmap"`. Note that our WL0 can directly read weight data from DRAM or from the L2 Buffer, so we need to mark its `"source"`.

```json
"wl0_buffer" : [
    {
        "address" : 0,
        "align" : 8,
        "bitwidth" : 8,
        "layer_name" : "Conv_0",
        "size" : 896,
        "source" : "CORE",
        "tensor_order" : null,
        "transfer_id" : [ 0 ]
    }
],
```