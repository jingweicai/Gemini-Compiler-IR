# ISA Format Description

In the assembler, we represent Machine Code with Assembly Instructions in the form of human readable json lines. Assembly Instruction types can be classified into 9 categories: **CT**, **LD**, **LW**, **PE**, **VP**, **DT**, **ST**, **SI**, **SO**. Here we explain instructions with examples.

## Overall Format

```json
[pe_conv, {'slyr': 19, 'clyr': [], 's_flyr': 0, 'c_flyr': [], 'wkl': 19, 'tile': 0}, {..., 'Prev_Insts': ['LW'], 'Next_Insts': ['VP', 'LW']}]
```
All instructions have 3 parts: 

1. The first part is the **instruction name**. Here it is `pe_conv`
2. The second part represents the **mapping relationship to the upstream IR**. This part doesn't directly map to some bit feild within the actural Machine Codes, but we utilize this information to generate them.
   1. **slyr** (source layer): Represents the layer where the instruction is scheduled, taken from the original frontend IR’s `layer_idx`. For delayed store instructions (STA), `slyr` indicates the actual layer where the instruction is scheduled, but it may not be the data of that layer.
   2. **clyr** (consume source layers, List[idx]): Represents the actual layer(s) the instruction belongs to, taken from the original frontend IR’s `layer_idx`. For prefetch instructions (like `lda` or `lw`), `slyr` is not in `clyr`. The relationship between `slyr` and `clyr` sizes is uncertain, as prefetching may occur within a fused layer. For delayed store instructions (STA), `clyr` represents the layer whose data is being stored. `clyr` is meaningful only for data transfer instructions, and if it's an empty list `[]`, it means it is not relevant.
   3. **s_flyr** (source fusion layer): Represents the fusion layer index of the workload where the instruction is scheduled, taken from the inter-core IR's fusion layer name.
   4. **c_flyr** (consume source layers, List[idx]): Represents the actual fusion layer index that the instruction belongs to, taken from the inter-core IR’s fusion layer name.
   5. **wkl** (workload): Represents the workload where the instruction is scheduled, taken from the inter-core IR’s `workload_idx`.
   6. **tile**: Represents the tile within the workload where the instruction is scheduled, taken from the inter-core IR’s `tile_idx`. Some data read/write instructions do not have a tile field.
   **Note**:
   1. `-1` indicates that the corresponding field is not relevant.
   2. For synchronization-only instructions, all fields in this part are irrelevant.
3. The third part is the **parameter dictionary** for the corresponding instruction. 
    - We use a multi-task out-of-order instruction dispatch queue. To resolve instruction dependency issues, we include dependency bits (`'Prev_Insts'` and `'Next_Insts'`) in the instructions.
    - We will discuss other parameters in this part in more details further.

## Control Instructions:

### CT (Control)
Overall control, such as starting and ending tasks, and managing the allocation of loop space in the L2 buffer.

**Example**:
```json
[ct_st, {...}, {'Prev_Insts': [], 'Next_Insts': []}]
[ct_bc, {...}, {'BCS': 0, 'BufCut': [0, 8191, 0, 0, 0, 0, 0, 0], 'Prev_Insts': [], 'Next_Insts': []}]
[ct_bc, {...}, {'BCS': 3, 'BufCut': [0, 0, 0, 0, 0, 0, 0, 0], 'Prev_Insts': [], 'Next_Insts': ['LD']}]
...
[ct_ed, {...}, {'Prev_Insts': ['ST'], 'Next_Insts': []}]
```
These instructions marks the start or the end of a DNN task, and indicates how to allocate buffer. 

## Data Loading Instructions:

### LD (Load Data): 

Reading data from DRAM into the L2 buffer. The data can be activations, weights, or other parameters.

**Example**:
```json
[lda_cfg, {...}, {'SME': 1, 'RS': 0, 'BAS': 0, 'DF': <DataFmt.FP16: 3>, 'STile': [3, 224, 224], 'DTile': [3, 224, 224]}]
[lda_mov, {...}, {'RS': 0, 'DBSG': 0, 'STaddr': 0, 'DTaddr': 428544, 'Tile': [1, 3, 224, 224], 'Prev_Insts': ['CT'], 'Next_Insts': ['VP']}]
```

1. `lda_cfg`
    - **SME**: `1`, enables source data dense packing (set to `1` to indicate packed data).
    - **RS**: `0`, specifies that this configuration is written to the `cfg0` parameter register.
    - **BAS**: `0`, base address selection, choosing base address register 0 for data transfer.
    - **DF**: `FP16`, data format is `FP16` (floating-point 16-bit).
    - **STile**: `[3, 224, 224]`, source data block dimensions are `[3, 224, 224]`, which correspond to the number of channels (C), height (H), and width (W) of the source feature map.
    - **DTile**: `[3, 224, 224]`, destination data block dimensions are also `[3, 224, 224]`, matching the source dimensions.

    - **Explanation**:
    The `lda_cfg` instruction sets up the configuration for a data load operation, including dense data packing (`SME = 1`), writing to the `cfg0` parameter register (`RS = 0`), specifying the base address (`BAS = 0`), and setting the data format to FP16. The source and destination tiles are both `[3, 224, 224]`, which indicates the size of the data being loaded.

2. `lda_mov`
    - **RS**: `0`, specifies that the `lda_mov` instruction uses the parameters configured in `cfg0` (the same configuration register as `lda_cfg`).
    - **DBSG**: `0`, base address register selection for the destination buffer (chooses base address register 0).
    - **STaddr**: `0`, source offset address, which is the starting address of the data in DDR memory (relative to the base address).
    - **DTaddr**: `428544`, destination address in the L2 buffer, where the data will be stored. This address is represented in 64-bit units, so it translates to `(428544 >> 3) & 0x1ffff = 53568` in L2.
    - **Tile**: `[1, 3, 224, 224]`, specifies the size of the data block being transferred, with 1 batch, 3 channels, and spatial dimensions of 224x224 (N, C, H, W).
    - **Prev_Insts**: `['CT']`, indicates that this instruction depends on the last `CT` instruction.
    - **Next_Insts**: `['VP']`, indicates that the next `VP` instruction relies on this instruction.

    - **Explanation**:
    The `lda_mov` instruction is responsible for moving the data from the source (DDR memory) to the destination (L2 buffer). The parameters indicate that the configuration register `cfg0` is used (`RS = 0`), and the data is moved from address `STaddr = 0` in DDR to address `DTaddr = 428544` in the L2 buffer. The size of the data block being transferred is `[1, 3, 224, 224]` (batch size 1, 3 channels, and spatial dimensions 224x224). This instruction depends on the last control instruction (`CT`) and its result is required by a vector processing instruction (`VP`).

### LW (Load Weight): 
Reading weight data and storing it in the **WL0 buffer**. 

**Example**:
```json
[lw_cfg, {...}, {'BAS': 1, 'DF': <DataFmt.INT8: 1>, 'MKC': 0, 'MKEN': 0, 'SWsize': [64, 1, 1], 'DWsize': [64, 1, 1]}]
[lw_mov, {...}, {'SS': 1, 'XYmode': [<XCutMode.CUT_K: 3>, <YCutMode.CUT_H: 1>], 'SBSG': 0, 'STaddr': 27136, 'DTaddr': 0, 'Wsize': [64, 64, 1, 1], 'Prev_Insts': ['PE'], 'Next_Insts': ['PE']}]
```

1. `lw_cfg`
    - **BAS**: `1`  
    Base address selection. The instruction selects base address register 1 for accessing the memory during data load operations.
    - **DF**: `INT8`  
    Data format is set to `INT8` (integer 8-bit). This means the data being loaded consists of 8-bit integers.
    - **MKC**: `0`  
    Mask channel. The instruction does not apply any mask to the channels, as the mask channel is set to `0`.
    - **MKEN**: `0`  
    Mask enable. The mask is disabled (`0`), meaning no masking is applied during the weight loading.
    - **SWsize**: `[64, 1, 1]`  
    Source weight block size. This indicates that the source data block for the weight has dimensions `[64, 1, 1]` (channels, spatial height, spatial width). The values correspond to the size of the source weight data.
    - **DWsize**: `[64, 1, 1]`  
    Destination weight block size. This indicates that the destination data block (where the weight will be stored) has the same dimensions as the source, `[64, 1, 1]`.

2. `lw_mov`
    - **SS**: `1`  
    Source selection. This indicates that the source data comes from an external source (DDR or another core) rather than the local L2 buffer.
    - **XYmode**: `[CUT_K, CUT_H]`  
    Defines the split mode for data handling. 
    - **XCutMode.CUT_K**: Indicates that the data is being split along the `K` dimension (the number of output channels).
    - **YCutMode.CUT_H**: Indicates that the data is being split along the `H` dimension (the height of the feature map).
    - **SBSG**: `0`  
    Source base address selection group, which chooses base address register 0.
    - **STaddr**: `27136`  
    Source offset address in DDR memory. This specifies the starting address (relative to the base address) from which the weight data will be loaded.
    - **DTaddr**: `0`  
    Destination address, which is the starting address in the weight L1 buffer (WL0). The address is given in units of 256 bits.
    - **Wsize**: `[64, 64, 1, 1]`  
    Weight data block size. The dimensions of the weight data block are `[64, 64, 1, 1]` (output channels, input channels, height, width), which indicates the size of the data being transferred.

The **`lw_cfg`** instruction configures the load weight parameters (such as data format, base address, and size), while the **`lw_mov`** instruction performs the actual movement of the weight data from DDR to the L1 buffer with specific slicing modes and addresses. 

### SI (Sync Load In): 
Loads activation or weight data from the remote core’s L2 buffer into the local L2 buffer.

## Storage/Output Instructions:

### ST (Store from L2): 
Stores data from the L2 buffer to DDR memory. This instruction is very similar to LD. 

### SO (Sync Out from L2): 
Stores data from the local L2 buffer into the remote L2 buffer.

## Computation Instructions:

### DT (Data Transform): 
Loads data from L2, performs dimension transformations, and writes the result back to the L2 buffer.

**Example**:
```json
[dt_cfg, {...}, {'DF': <DataFmt.INT8: 1>, 'STile': [512, 28, 28], 'DTile': [512, 28, 28]}]
[dt_cal, {...}, {'Order': 0, 'SBSG': 0, 'DBSG': 0, 'STaddr': 1204224, 'DTaddr': 1605632, 'Tile': [1, 512, 28, 28], 'Prev_Insts': ['VP'], 'Next_Insts': ['PE']}]
```
1. `dt_cfg`
    - **DF**: `INT8`  
    The data format is set to `INT8` (8-bit integer), indicating that the transformation will operate on data with an 8-bit integer format.
    - **STile**: `[512, 28, 28]`  
    The source tile size is `[512, 28, 28]`, which represents the dimensions of the source feature map: 512 channels (C), 28 height (H), and 28 width (W).
    - **DTile**: `[512, 28, 28]`  
    The destination tile size is `[512, 28, 28]`, matching the source dimensions, indicating that the transformation does not change the size of the data but may change its layout or format.
2. `dt_cal`
    - **Order**: `0`  
    Specifies the vector reordering type (dimension transformation format). The value `0` corresponds to the `NHWC` format (batch size N, height H, width W, and channels C).
    - **SBSG**: `0`  
    Source base address selection group, which selects base address register 0 for the source data.
    - **DBSG**: `0`  
    Destination base address selection group, which selects base address register 0 for the destination data.
    - **STaddr**: `1204224`  
    The source address in the L2 buffer from where the data is loaded. The address is in 64-bit units, so the actual address is `(1204224 >> 3) = 150528`.
    - **DTaddr**: `1605632`  
    The destination address in the L2 buffer where the data is written. The address is also in 64-bit units, so the actual address is `(1605632 >> 3) = 200704`.
    - **Tile**: `[1, 512, 28, 28]`  
    The tile size for this calculation is `[1, 512, 28, 28]`, indicating a batch size of 1, 512 channels, and spatial dimensions of 28x28.

In short, `dt_cfg` sets up the configuration for data transformation with specific tile sizes and data format (INT8), while `dt_cal` performs the actual transformation, including address mappings, tile size, and reordering the data layout to `NHWC` format.

### PE (Tensor Process Element): 
Loads data from L2 into the AL1 cache, retrieves activation and weight data from AL1 and WL0, and performs tensor convolution or fully connected (FC) tasks. The resulting data is stored in OL1 and then written back to L2. there are many types of conv such as deconv and group conv, but in the example below we will only discuss the most common 2D conv.

**Example**:
```json
[pe_cfg_fm, {...}, {'IF': <DataSign.SIGNED: 1>, 'OF': <DataSign.SIGNED: 1>, 'SF': <DataFmt.INT8: 1>, 'WF': <DataFmt.INT8: 1>, 'DF': <DataFmt.INT8: 1>, 'PADV': 6.0, 'STile': [3, 224, 224], 'DTile': [64, 112, 112]}]
[pe_cfg_pm, {...}, {'ACTMD': <ActMode.RELU: 1>, 'ACTPM': 0.0, 'QC': <Float2Fixed.ROUND: 2>, 'QB': -128.0, 'QK': 43.62298146446928, 'ADDEN': 0, 'STRIDE': [2, 2], 'XYmode': [<XCutMode.CUT_K: 3>, <YCutMode.CUT_H: 1>]}]
[pe_cfg_dq, {...}, {'DQC': 0, 'DQB': 0, 'DQK': 0}]
[pe_cfg_tp, {...}, {'KERNEL': [7, 7], 'Tile': [1, 64, 3, 112, 112], 'MTK0': 1, 'MTC0': 1, 'MTI': [1, 1]}]
[pe_conv, {...}, {'PS': 0, 'SPHI': -3, 'SPWI': -3, 'SBSG': 0, 'DBSG': 0, 'STaddr': 21736, 'SWaddr': 0, 'DTaddr': 428544, 'Prev_Insts': ['VP', 'LW'], 'Next_Insts': ['VP', 'LW']}]
```

1.  `pe_cfg_fm`
    - **IF**: `SIGNED`  
    The input data type is signed integers, meaning the input data values can be negative.
    - **OF**: `SIGNED`  
    The output data type is also signed integers.
    - **SF**: `INT8`  
    The source data format is 8-bit integers.
    - **WF**: `INT8`  
    The weight format is 8-bit integers.
    - **DF**: `INT8`  
    The destination data format is also 8-bit integers.
    - **PADV**: `6.0`  
    The padding value is `6.0`, meaning that any padding pixels in the input feature map will be set to this value.
    - **STile**: `[3, 224, 224]`  
    The source tile size, representing 3 channels (C), 224 height (H), and 224 width (W).
    - **DTile**: `[64, 112, 112]`  
    The destination tile size, representing 64 output channels (C), 112 height (H), and 112 width (W).

2. `pe_cfg_pm`
    - **ACTMD**: `RELU`  
    The activation mode is ReLU (Rectified Linear Unit), meaning any negative values will be set to zero.
    - **ACTPM**: `0.0`  
    Activation parameter is `0.0`, which is not used for ReLU but could be relevant for other activation modes like leaky ReLU or ReLU_x.
    - **QC**: `ROUND`  
    The quantization rounding mode is set to "round," meaning the quantization process will round the values.
    - **QB**: `-128.0`  
    The quantization zero point is `-128.0`, which shifts the input data for quantization.
    - **QK**: `43.62298146446928`  
    The quantization scale is `43.62298146446928`, which is used to scale the values during quantization.
    - **ADDEN**: `0`  
    Addition enable is disabled, meaning no additional accumulation is applied to the results.
    - **STRIDE**: `[2, 2]`  
    The stride is set to `[2, 2]`, meaning the operation will downsample the input by a factor of 2 in both height and width.
    - **XYmode**: `[CUT_K, CUT_H]`  
    The mode for slicing data:
    - **CUT_K**: The data is being sliced along the `K` (output channels) dimension.
    - **CUT_H**: The data is being sliced along the `H` (height) dimension.

3. `pe_cfg_dq`
    - **DQC**: `0`  
    The dequantization clipping mode is disabled or set to no clipping.
    - **DQB**: `0`  
    Dequantization zero point is `0`, meaning no shift is applied to the values during dequantization.
    - **DQK**: `0`  
    Dequantization scale is `0`, meaning no scaling is applied during dequantization.

4. `pe_cfg_tp`
    - **KERNEL**: `[7, 7]`  
    The kernel size is 7x7, representing the dimensions of the convolution filter.
    - **Tile**: `[1, 64, 3, 112, 112]`  
    The tile size is `[1, 64, 3, 112, 112]`, representing 1 batch (N), 64 output channels (C), 3 input channels (C), and 112x112 height and width (H, W).
    - **MTK0**: `1`  
    Mini tile size for `K` dimension is set to `1`.
    - **MTC0**: `1`  
    Mini tile size for `C` (input channels) dimension is also set to `1`.
    - **MTI**: `[1, 1]`  
    Mini tile input size for both height and width is set to `1`.

5. `pe_conv`
    - **PS**: `0`  
    Phase shift for the channels, used to handle conflicts in RAM access during the convolution operation.
    - **SPHI**: `-3`  
    Start point for height in the input feature map, indicating a padding offset of `-3`.
    - **SPWI**: `-3`  
    Start point for width in the input feature map, indicating a padding offset of `-3`.
    - **SBSG**: `0`  
    Source base address selection group is `0`, indicating base address register 0 is used for the source.
    - **DBSG**: `0`  
    Destination base address selection group is `0`, indicating base address register 0 is used for the destination.
    - **STaddr**: `21736`  
    The source address in the L2 buffer from where the input feature map data is loaded, in 64-bit units.
    - **SWaddr**: `0`  
    The source address for weights in the weight L1 buffer (WL0).
    - **DTaddr**: `428544`  
    The destination address in the L2 buffer where the output feature map is stored, in 64-bit units.


### VP (Vector Processor): 
Retrieves activation data from AL2 and performs tasks like pooling and other certain computation tasks. Here we show an example of 2D pooling.

**Example**:
```json
[vp_cfg_gcs, {...}, {'CMPS': <CMPS.MAX: 1>, 'SG0M': <SG0M.DEFAULT: 0>, 'CMPVS': 0, 'SG0VS': 0, 'SG0V': 0, 'CMPV': 0}]
[vp_cfg_qp, {...}, {'QCS': <Float2Fixed.ROUND: 2>, 'QSMD': <QSMD.DEFAULT: 0>, 'QB': -128.0, 'QK': 43.62298146446928}]
[vp_cfg_iq, {...}, {'IQCS': 0, 'IQB0': 2.9342331886291504, 'IQK0': 0.022923696786165237}]
[vp_cfg_fm, {...}, {'IF': <DataSign.SIGNED: 1>, 'OF': <DataSign.SIGNED: 1>, 'IDF': <DataFmt.INT8: 1>, 'ODF': <DataFmt.INT8: 1>, 'VF32': 0, 'STile': [64, 112, 112], 'DTile': [64, 56, 56]}]
[vp_cfg_tp, {...}, {'STRIDE': [2, 2], 'KERNEL': [3, 3], 'Tile': [64, 56, 56], 'T1F': 0, 'T0F': 7, 'PADV_AREA': -32768}]
[vp_pool, {...}, {'IQEN': 1, 'QEN': 3, 'SHMD': 0, 'GPMD': 4, 'CMPD': <CMPD.DEFAULT: 0>, 'TREED': <TREED.DEFAULT: 0>, 'ACCD': <ACCD.DEFAULT: 0>, 'SWRS': <SWRS.DEFAULT: 0>, 'LUT': 17, 'IPH': -1, 'IPW': -1, 'SBSG': 0, 'DBSG': 0, 'STaddr': 421312, 'DTaddr': 0, 'Prev_Insts': ['PE'], 'Next_Insts': ['PE', 'LW']}]
```
Here's the explanation for the **Instruction Parameters** part of each instruction:

1. `vp_cfg_gcs`
    - **CMPS**: `MAX`  
    Compare mode is set to `MAX`, meaning that the comparison will choose the maximum value.
    - **SG0M**: `DEFAULT`  
    Group 0 calculation mode is set to the default, which is addition.
    - **CMPVS**: `0`  
    The scalar value for the compare module is taken from the instruction (`CMPV`), not from an internal register.
    - **SG0VS**: `0`  
    The scalar value for group 0 addition is taken from the instruction (`SG0V`), not from an internal register.
    - **SG0V**: `0`  
    Scalar immediate value for group 0 addition.
    - **CMPV**: `0`  
    Scalar immediate value for the compare module.

2. `vp_cfg_qp`
    - **QCS**: `ROUND`  
    The floating-point to fixed-point conversion mode is set to round the values.
    - **QSMD**: `DEFAULT`  
    The quantization mode is set to addition (default).
    - **QB**: `-128.0`  
    The quantization bias (zero point) is set to `-128.0`.
    - **QK**: `43.62298146446928`  
    The quantization scale factor is `43.62298146446928`.

3. `vp_cfg_iq`**
    - **IQCS**: `0`  
    The conversion mode for floating-point to fixed-point is set to default  rounding mode).
    - **IQB0**: `2.9342331886291504`  
    The inverse quantization bias (zero point) is set to `2.9342331886291504`.
    - **IQK0**: `0.022923696786165237`  
    The inverse quantization scale factor is `0.022923696786165237`.

4. `vp_cfg_fm`**
    - **IF**: `SIGNED`  
    The input data type is signed integers.
    - **OF**: `SIGNED`  
    The output data type is also signed integers.
    - **IDF**: `INT8`  
    The input data format is 8-bit integers.
    - **ODF**: `INT8`  
    The output data format is 8-bit integers.
    - **VF32**: `0`  
    Vector FP32 format enable is set to `0`, meaning FP32 is not enabled.
    - **STile**: `[64, 112, 112]`  
    The source tile size is 64 channels, 112 height, and 112 width.
    - **DTile**: `[64, 56, 56]`  
    The destination tile size is 64 channels, 56 height, and 56 width.

5. `vp_cfg_tp`**
    - **STRIDE**: `[2, 2]`  
    The stride is `[2, 2]`, meaning the pooling or transformation operation will reduce the size by a factor of 2 in both the height and width dimensions.
    - **KERNEL**: `[3, 3]`  
    The kernel size is 3x3, used for convolution or pooling.
    - **Tile**: `[64, 56, 56]`  
    The size of the tile is 64 channels, 56 height, and 56 width.
    - **T1F**: `0`  
    The tensor1 dimension enable flag is `0`, meaning tensor1 is not being used.
    - **T0F**: `7`  
    The tensor0 dimension enable flag is `7`, indicating that all dimensions (height, width, and channels) of tensor0 are enabled.
    - **PADV_AREA**: `-32768`  
    Padding value for upsample area correction. A value of `-32768` means no correction is applied.

6. `vp_pool`**
    - **IQEN**: `1`  
    Inverse quantization is enabled.
    - **QEN**: `3`  
    Quantization is enabled.
    - **SHMD**: `0`  
    The shift mode is disabled, meaning no shifting is applied to the data.
    - **GPMD**: `4`  
    The mode selection for the pooling operation is set to a specific configuration  controlling aspects like accumulation or comparison).
    - **CMPD**: `DEFAULT`  
    The compare mode is set to default, which may be comparison based on certain dimensions or thresholds.
    - **TREED**: `DEFAULT`  
    The tree mode is set to default, determining how accumulation or reduction is performed.
    - **ACCD**: `DEFAULT`  
    Accumulation mode is set to default, specifying how results are accumulated across dimensions.
    - **SWRS**: `DEFAULT`  
    Scalar result storage register selection is set to default, indicating where scalar results will be written.
    - **LUT**: `17`  
    Lookup table value, which could refer to an index in a predefined table for specific operations.
    - **IPH**: `-1`  
    Integer part of the starting position for pooling in the height dimension, set to `-1`, indicating no specific position.
    - **IPW**: `-1`  
    Integer part of the starting position for pooling in the width dimension, set to `-1`, indicating no specific position.
    - **SBSG**: `0`  
    Source base address selection group is `0`, indicating the source address is taken from base register 0.
    - **DBSG**: `0`  
    Destination base address selection group is `0`, indicating the destination address is taken from base register 0.
    - **STaddr**: `421312`  
    The source address in L2 memory where the data is read from, given in 64-bit units.
    - **DTaddr**: `0`  
    The destination address in L2 memory where the pooled data is written, given in 64-bit units.
