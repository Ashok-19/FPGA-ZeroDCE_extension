



# Vitis-AI workflow


## Prerequisites:

![Requirements](https://github.com/Ashok-19/zerodce_extension_FPGA/blob/774385801b62efd721d7b1eb4cbe51f2512aedd0/screenshots/Vitis_prerequisites.png)

It is recommended to use ubuntu 22.04 as CentOS versions already reached its EndOfLife.

* Clear Understanding of [Vitis_ai_guide](https://docs.amd.com/r/3.0-English/ug1414-vitis-ai)


* DPU accelerated board such as Kria KV260, ZCU104 or similar ones. 


## Tutorials:

 [Vitis-AI-Tutorials](https://github.com/Xilinx/Vitis-AI-Tutorials.git)



## Installation:

Refer  [Vitis 3.0 docs](https://xilinx.github.io/Vitis-AI/3.0/html/docs/install/install.html) to install docker and docker image of Vitis-AI.

#### Note : 

If you are facing any difficulties in GPU docker build, pull this 3rd party docker image in your system.

        docker pull samcts2309/vitis-ai-pytorch-gpu

This works only for pytorch.


## Setup

Activate the docker and conda environment in Vitis AI. Then follow the steps below,

* Install JupyterLab for better experience

        conda install jupyterlab

        pip install chardet

* Push and Commit the docker to save the changes permanently.


* Create a directory for zerodce under Vitis-AI/

        cd Vitis-AI
        mkdir zdce_extension
    

In this directory, move the entire [compile_for_dpu](https://github.com/Ashok-19/zerodce_extension_FPGA/tree/22482e414fae7766ecaf7d2c2031815155d0839c/zdce_extension_fpga/compile_for_dpu) folder.

* To access the resnet18 example, go to

        cd src/vai_quantizer/vai_q_pytorch/example


## Important Note:

If you want to make changes in the model's structure, modify [model.py](https://github.com/Ashok-19/zerodce_extension_FPGA/blob/774385801b62efd721d7b1eb4cbe51f2512aedd0/zdce_extension_fpga/compile_for_dpu/model.py)


Refer [supported operations](https://docs.amd.com/r/3.0-English/ug1414-vitis-ai/Currently-Supported-Operators) for DPU compatible operations.

Note that if you're making some drastic changes, you need to retrain the entire model for updated weights.

* The current model ported in ZCU104 was not optimized/pruned. To optimize it refer [Vitis-AI-optimizer](https://docs.amd.com/r/3.0-English/ug1333-ai-optimizer).

* The pretrained model provided by default was used in porting to the board. So, results might  not have a drastic difference.






## Inspection

Inspect the model to confirm all the operators are mapped to DPU. If any of the operators were not mapped, either wrap them as a custom operator or approximate them using supported DPU operations.




Note that custom operators will never be executed on DPU (PL) using Vitis-AI


    Target -> DPUCZDX8G_ISA1_B4096 (ZCU104)

Target is specified in the code itself, no need to mention it explicitly.

    python zdce.py --mode float --inspect

By running the above command, you should get a result that says "All operations are mapped to DPU".

A dot image will be created in the [/inspect](https://github.com/Ashok-19/zerodce_extension_FPGA/tree/774385801b62efd721d7b1eb4cbe51f2512aedd0/zdce_extension_fpga/compile_for_dpu/quantize_result/inspect) folder.


## Calibration

Calibrate the model by using calib data provided in data folder with fastfinetune

    python zdce.py --mode calib --fast_finetune




This calibrates the model and creates [quant_info.json](https://github.com/Ashok-19/zerodce_extension_FPGA/blob/774385801b62efd721d7b1eb4cbe51f2512aedd0/zdce_extension_fpga/compile_for_dpu/quantize_result/quant_info.json)



## Testing and Deployment

    python zdce.py --mode test --deploy --fast_finetune

This step quantizes the model and creates .pt ,onnx and xmodel files of ZeroDCE++.

Inspect the xmodel file in [netron](https://netron.app/) to see the structure and workflow of the model.


## Onnx to Custom IP (Optional)

If you want to create verilog code and custom IP then the generated onnx model file can be used to create a custom IP for the target board using [NNgen](https://github.com/NNgen/nngen.git)

## Compiling the xmodel

Compile the xmodel for the board specific architecture using the following command,


    vai_c_xir -x <xmodel_file_path> -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json -o <output_directory> -n <netname>


The xmodel file will be compiled to board specific architecture with the name provided as <netname> in <output_directory>


To create the graph of this compiled model, use the following command,

    xdputil xmodel <compiled_xmodel> -s <name_for_svg_file>


All the output and compiled files are already provided in the repository.


# Board Setup and working

MicroSD - 16 GB or Higher (32 GB recommended)

Refer [board setup](https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html#setup-the-target)


Connect the board to the host through putty or any other terminal.



* Copy the [output](https://github.com/Ashok-19/zerodce_extension_FPGA/tree/774385801b62efd721d7b1eb4cbe51f2512aedd0/zdce_extension_fpga/compile_for_dpu/output) folder to the board.



## OS limitations

* Prebuilt image available on Vitis AI documentation lacks customization. The petalinux image provided [here](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx-zcu104-v2022.2-10141622.bsp) can be used for Vitis AI 3.0 and also be configured (I haven't tested it).

* If you did use the prebuilt image provided in Vitis AI documentation, live video capture will be limited to only MJPG and raw formats.


## Booting the image

After flashing the image on sd card and making necessary connections. Follow below steps,


### Accessing the board's terminal

        [Host] $ sudo screen /dev/ttyUSB<number> 115200

* Use scp command to transfer files.


## Benchmarking


Using xdputil, we can easily benchmark  the DPU subgraph of the xmodel file.


        [Target] $ xdputil benchmark <xmodel_file> -i <subgraph_number> <num_threads>


#### EXAMPLE OUTPUT:

        root@xilinx-zcu104-20222:~/zdce/models# xdputil benchmark zdce++_og_fft.xmodel -i 1 2


        OUTPUT:

        WARNING: Logging before InitGoogleLogging() is written to STDERR
        I0713 18:37:48.229240   849 test_dpu_runner_mt.cpp:474] shuffle results for batch...
        I0713 18:37:48.241431   849 performance_test.hpp:73] 0% ...
        I0713 18:37:54.241619   849 performance_test.hpp:76] 10% ...
        I0713 18:38:00.241824   849 performance_test.hpp:76] 20% ...
        I0713 18:38:06.242061   849 performance_test.hpp:76] 30% ...
        I0713 18:38:12.242331   849 performance_test.hpp:76] 40% ...
        I0713 18:38:18.242538   849 performance_test.hpp:76] 50% ...
        I0713 18:38:24.242769   849 performance_test.hpp:76] 60% ...
        I0713 18:38:30.242974   849 performance_test.hpp:76] 70% ...
        I0713 18:38:36.243189   849 performance_test.hpp:76] 80% ...
        I0713 18:38:42.243458   849 performance_test.hpp:76] 90% ...
        I0713 18:38:48.243785   849 performance_test.hpp:76] 100% ...
        I0713 18:38:48.243886   849 performance_test.hpp:79] stop and waiting for all threads terminated....
        I0713 18:38:48.279520   849 performance_test.hpp:85] thread-0 processes 904 frames
        I0713 18:38:48.279599   849 performance_test.hpp:85] thread-1 processes 810 frames
        I0713 18:38:48.279624   849 performance_test.hpp:93] it takes 35712 us for shutdown
        I0713 18:38:48.279647   849 performance_test.hpp:94] FPS= 28.5485 number_of_frames= 1714 time= 60.0382 seconds.
        I0713 18:38:48.279701   849 performance_test.hpp:96] BYEBYE
        Test PASS.


* The output tells us that the model's DPU subgraph can process 28.5485 frames per second.

* During Live feed testing, the FPS will obviously drop because of pre and postprocessing of image frames that's not done in DPU.


## Using Inference scripts

* Write an inference script either in python or C++ for processing the videos.

* For C++ scripts, you have to build the source code.
        Refer [Inference scripts](https://github.com/Xilinx/Vitis-AI/tree/091f75041ed941b74361c1068bbcd6528e61cbc6/examples/vai_runtime)


The [output]() folder has a sample python script. It does the following,

 * Prints load time ,Preprocess time, Inference time, Postprocess time, Write time, Total time taken for each image from a prerecorded video.

* For capturing video using see3cam, use

        #This command captures video in MJPG format wrapped inside .avi format

        gst-launch-1.0 -e   v4l2src device=/dev/video0 io-mode=dmabuf num-buffers=300   ! 'image/jpeg,width=2304,height=1296,framerate=60/1'   ! jpegparse   ! avimux   ! filesink location=/home/root/video.avi


* Adjust num-buffers for video duration.
* Use supported resolution and fps.

### Usage:

        python zdce.py <xmodel_file_path> <input_video> <output_video> <num_threads>


Output :
        

        Frames: 194/300 | Load: 0.2ms | Preprocess: 36.8ms | Inference: 67.5ms | Postprocess: 43.2ms | Write: 4.7ms | Total: 152.5ms | Est. FPS: 6.6 | 60FPS Live Ratio: 0.11x
                                .
                                .
                                .

        ================================================================================
        VIDEO PROCESSING SUMMARY
        ================================================================================
        Total frames processed: 300
        Total processing time: 64.55 seconds
        Overall throughput: 4.65 FPS
        Output resolution: 2304x1296 (original)

        Per-frame timing breakdown (average):
        Load time:        0.22 ms
        Preprocess time:  36.78 ms
        Inference time:   67.47 ms
        Postprocess time: 44.52 ms
        Write time:       4.77 ms
        Total per frame:  153.77 ms

        Estimated live performance:
        Max sustainable FPS: 6.5
        60 FPS live ratio: 0.11x
        ✗ Cannot process 60 FPS video in real-time

## Strategies to improve performance and FPS:

* Writing full C++ inference script.

* Optimizing the model by using effective pruning Strategies.


### Recommended pruning method:

As ZeroDCE++ contains Depthwise conv2D layers, it is recommended to use OnceForAll pruning methods. 

Check out [Vitis AI 3.0 optimizer](https://docs.amd.com/r/3.0-English/ug1333-ai-optimizer/PyTorch-Version-vai_p_pytorch)


## Results

* [Low_light_video](https://drive.google.com/file/d/12DEGCwvVJlLndzmaSsMPucO3VgIAnO34/view?usp=sharing) --------- [Enhanced Low_light_video](https://drive.google.com/file/d/1mXACRFsJJa-bMrnR8nXSUUumrAAeFone/view?usp=sharing)

* [Medium_light_video](https://drive.google.com/file/d/16Sy67MSjuE6ua0Q4f47BaaIPNszQY7Ku/view?usp=sharing) ---------- [Enhanced Medium_light_video](https://drive.google.com/file/d/16sTpci3rpoJbdL_YMGQNmZSR86ZUK6Hh/view?usp=sharing)



# Future works

* Pruning ZeroDCE++ to achieve higher FPS with lower latency

* Training the base model with more Data to make it much more robust

* Porting [Zero-TCE](https://www.mdpi.com/2076-3417/15/2/701)  to DPU.

* Combining both ZeroDCE++ and a Yolox model as a single xmodel file and deploying it with an optimized C++ inference script achieving low latency in live feed.



## Contact

 If you have any suggestions/questions, kindly mail ashokraja1910@gmail.com