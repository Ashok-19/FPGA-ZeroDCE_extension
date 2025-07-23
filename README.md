
# FPGA-ZeroDCE++

Porting image enhancement model ZeroDCE++ on DPU accelerated FPGA devices such as Kria KV260, Zynq Ultrascale boards, Versal boards with help of Vitis-AI.

# Important Note

Currently this project is undergoing some improvements. Stay tuned...

## Introduction:

This project is inspired from [FPGA-OwlsEye](https://github.com/Gaurav-Shah05/FPGA-OwlsEye.git).

In FPGA-Owlseye, the ZeroDCE model was originally mapped to ARM processor of Zynq Ultrascale zcu104 board. This project focuses on mapping the ZeroDCE++ model to DPU (Deep Learning Processing Units) of the zcu104 board.The reason why ZeroDCE++ is chosen instead of ZeroDCE is because the difference between number of params while providing same results.
    
        ZeroDCE   - ~80K params
        ZeroDCE++ - ~10K params

Some key changes are made in the original design of the model to map it to DPU.






## Modified ZeroDCE++

Original [ZeroDCE++](https://github.com/Li-Chongyi/Zero-DCE_extension.git) model used tanh activation in final convolution layer and power operations in LE curve estimation formula. These two operations are not supported by DPU. So, zeroDCE++ was modified with approximating the unsupported operations with supported operations as follows,

        tanh = 2 * hardSigmoid(x) - 1

        torch.pow(x, 2) = torch.mul(x, x)


![Tanh_approximation](https://github.com/Ashok-19/ZeroDCE_extension_DPU/blob/bdfb515a0c60ec21acb895896cf4c9fd1dbdad7f/screenshots/tanh_approx.png)

### Power Approximation

In Vitis-AI, you cannot leverage **torch.mul(x,2)** to perform the same as **torch.pow(x,2)**. So two options were there to calculate **x**2**.
	* Registering torch.pow() as a custom operator but it'll execute on CPU which defeats the whole purpose of porting the model.
 	* Approximating x**2. 

  I went with second option. A very small noise is added to same tensor to make it appear as a different tensor.

  		x1 = x
		x2 = x + 0.00000001

  		x_squared = torch.mul(x1,x2)

 This method was used in LE curve approximation formula.

 It is to be noted that these changes are only for calibrating ,and dumping xmodel from pretrained model to map all the operations on DPU and avoid unsupported operations getting assigned to CPU. It's not preferred to use this approximations for training the model.
  
## Prerequisites:

* Ubuntu 20.04 or 22.04 / CentOS 7.8,7.9,8.1 . Do note that CentOS versions mentioned here already reached their EndOFLife.

* Clone [Vitis AI - 3.0](https://xilinx.github.io/Vitis-AI/3.0/html/index.html) with tag -v3.0. 
* [ZeroDCE++](https://github.com/Li-Chongyi/Zero-DCE_extension.git) , if you want to train the model from scratch.
* Zynq® UltraScale+™ MPSoC ZCU104 / Kria KV260 or any other boards that support Vitis-AI 

    

## Training the model

Refer [readme_train_test](https://github.com/Ashok-19/ZeroDCE_extension_DPU/blob/b996d588f07fdd8e43bed78a78efef3a95faeb03/zdce_extension_fpga/train_test_host/README.md)


## Vitis AI Workflow

Refer [readme_compile_for_dpu](https://github.com/Ashok-19/ZeroDCE_extension_DPU/blob/b996d588f07fdd8e43bed78a78efef3a95faeb03/zdce_extension_fpga/compile_for_dpu/README.md)
    









