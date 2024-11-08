# Future Plan

In the future, we plan to establish a cloud platform that allows users to access our series of chiplet-based chips, enabling deployment from mapping schemes to instructions with direct insights into performance and even energy efficiency (detailed examples and descriptions are provided below). Our first-generation chip, comprising one IO Die and two compute Dies with a total compute capacity of 64 TOPs, returned from fabrication on November 5th (shown below). Since testing is still in progress, we are currently presenting preliminary results based on the ZEBU verification platform.

![Package Shot](./Package_Shot.jpg)

# Gemini-based Compiler Workflow Overview

This document outlines the workflow of our compiler. The initial input supports networks in TFLite, PyTorch, and ONNX formats. After frontend processing—including quantization and basic fusion (e.g., CONV+BN+ReLU)—it outputs the Frontend IR. Next, the Gemini Scheduler performs global optimization, generating the Scheduler IR, which includes explicit address optimizations as noted. Finally, the Assembler converts the IR into instructions for deployment on the chip. Since our chip just recently completed tape-out and is still in the testing phase, we will release some actual measured data later. Therefore, we are showing the results from the ZEBU platform.

The workload example we use is an **INT8-quantized ResNet50** with a **batch size ranging from `1 to 64`**. The platform is a **single-core** scaled-down accelerator (due to ZEBU resource limitations, simulating one core with a single board already exceeds 50% LUT resource utilization). For a more detailed hardware synthesis log of the ZEBU, please refer to [backend_log](./ZeBu_files/backend_default_globalLog.log).


![ZeBu](./ZeBu_files/ZeBu.jpeg)

## Workflow

### 1. **Frontend Processing**
   - **Input**: ONNX, Pytorch, TFLite model (A ResNet-50 ONNX example is located in [`model_input/`](./model_input/))
   - **Output**: Frontend IR (located in [`frontend_output/`](./frontend_output/))
   - **Process**: 
     - The first step in the compiler workflow is to convert the ONNX model into a Frontend IR, which includes quantization and basic fusion (e.g., CONV+BN+ReLU). This IR contains all layers, operations, data types, and dependencies required for further processing by the scheduling and assembly stages.
   
### 2. **Gemini Scheduler**
   - **Input**: Frontend IR (from [`frontend_output/`](./frontend_output/))
   - **Output**: Scheduler IR (located in [`scheduler_output/`](./scheduler_output/))
   - **Process**:
     - Gemini reads the Frontend IR and performs the optimization introduced in the paper. Once scheduling is complete, the system uses the IRgen tool to generate the Scheduler IR, which is a custom high-level intermediate representation. This decouples it from low-level instruction generation, facilitating continued optimization or even replacement of the scheduler.


### 3. **Assembler**
   - **Input**: Scheduler IR (from [`scheduler_output/`](./scheduler_output/))
   - **Output**: Assembly Instructions (located in [`assembler_output/`](./assembler_output/))
   - **Process**:
     - The assembly generation phase involves parsing the Scheduler IR and translating it into a series of assembly instructions. These instructions represent the final executable form that will run on the target hardware. Each instruction contains detailed information on data movement, computation, and control flow.

### 4. **ZEBU**
   - **Input**: Assembly Instructions (from [`assembler_output/`](./assembler_output/))
   - **Output**: Execution times for each instruction (located in [`zebu_trace_output/`](./zebu_trace_output/))
   - **Process**:
     - The assembly instructions can be executed on the ZEBU platform, where we integrated a specialized trace capture hardware mechanism in our accelerator to record the start and end times of each instruction and read them out. The validation we conducted in Sec. V-E is also based on this trace data.


## Understanding the IR Formats

Each stage of the process produces an IR that follows a specific format. For detailed information on the content and structure of these IRs, refer to the `*_Format.md` file located in each of the output directories:

- [`frontend_output/Frontend_IR_Format.md`](./frontend_output/Frontend_IR_Format.md)
- [`scheduler_output/Scheduler_IR_Format.md`](./scheduler_output/Scheduler_IR_Format.md)
- [`assembler_output/ISA_Format.md`](./assembler_output/ISA_Format.md)
- [`zebu_trace_output/ZeBu_Trace_Format.md`](./zebu_trace_output/ZeBu_Trace_Format.md)

These documents provide comprehensive explanations of the fields and structures used in the respective IRs.

## Folder Structure

- **[`frontend_output/`](./frontend_output/)**: Contains the quantized Frontend IR.
- **[`scheduler_output/`](./scheduler_output/)**: Contains the Scheduler IR generated by Gemini and IRgen.
- **[`assembler_output/`](./assembler_output/)**: Contains the final assembly instructions.
- **[`zebu_trace_output/`](./ZeBu_trace_output/)**: Contains the execution time logs from the Synopsys ZeBu Emulation System.

By following this structured pipeline, our compiler efficiently transforms models into hardware-executable instructions.
