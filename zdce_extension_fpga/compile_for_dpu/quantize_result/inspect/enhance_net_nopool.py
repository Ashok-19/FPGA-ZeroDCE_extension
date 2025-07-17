# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class enhance_net_nopool(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.module_0 = py_nndct.nn.Module('nndct_const') #enhance_net_nopool::6168
        self.module_1 = py_nndct.nn.Module('nndct_const') #enhance_net_nopool::6172
        self.module_2 = py_nndct.nn.Module('nndct_const') #enhance_net_nopool::6174
        self.module_3 = py_nndct.nn.Module('nndct_const') #enhance_net_nopool::6170
        self.module_4 = py_nndct.nn.Module('nndct_const') #enhance_net_nopool::6176
        self.module_5 = py_nndct.nn.Input() #enhance_net_nopool::input_0
        self.module_6 = py_nndct.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=3, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv1]/Conv2d[depth_conv]/input.3
        self.module_7 = py_nndct.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv1]/Conv2d[point_conv]/input.5
        self.module_8 = py_nndct.nn.ReLU(inplace=True) #enhance_net_nopool::enhance_net_nopool/ReLU[relu]/input.7
        self.module_9 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=32, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv2]/Conv2d[depth_conv]/input.9
        self.module_10 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv2]/Conv2d[point_conv]/input.11
        self.module_11 = py_nndct.nn.ReLU(inplace=True) #enhance_net_nopool::enhance_net_nopool/ReLU[relu]/input.13
        self.module_12 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=32, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv3]/Conv2d[depth_conv]/input.15
        self.module_13 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv3]/Conv2d[point_conv]/input.17
        self.module_14 = py_nndct.nn.ReLU(inplace=True) #enhance_net_nopool::enhance_net_nopool/ReLU[relu]/input.19
        self.module_15 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=32, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv4]/Conv2d[depth_conv]/input.21
        self.module_16 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv4]/Conv2d[point_conv]/input.23
        self.module_17 = py_nndct.nn.ReLU(inplace=True) #enhance_net_nopool::enhance_net_nopool/ReLU[relu]/3540
        self.module_18 = py_nndct.nn.Cat() #enhance_net_nopool::enhance_net_nopool/input.25
        self.module_19 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv5]/Conv2d[depth_conv]/input.27
        self.module_20 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv5]/Conv2d[point_conv]/input.29
        self.module_21 = py_nndct.nn.ReLU(inplace=True) #enhance_net_nopool::enhance_net_nopool/ReLU[relu]/3582
        self.module_22 = py_nndct.nn.Cat() #enhance_net_nopool::enhance_net_nopool/input.31
        self.module_23 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv6]/Conv2d[depth_conv]/input.33
        self.module_24 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv6]/Conv2d[point_conv]/input.35
        self.module_25 = py_nndct.nn.ReLU(inplace=True) #enhance_net_nopool::enhance_net_nopool/ReLU[relu]/3624
        self.module_26 = py_nndct.nn.Cat() #enhance_net_nopool::enhance_net_nopool/input.37
        self.module_27 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv7]/Conv2d[depth_conv]/input
        self.module_28 = py_nndct.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #enhance_net_nopool::enhance_net_nopool/CSDN_Tem[e_conv7]/Conv2d[point_conv]/3665
        self.module_29 = py_nndct.nn.Hardsigmoid(inplace=False) #enhance_net_nopool::enhance_net_nopool/3666
        self.module_30 = py_nndct.nn.Module('nndct_elemwise_mul') #enhance_net_nopool::enhance_net_nopool/6169
        self.module_31 = py_nndct.nn.Add() #enhance_net_nopool::enhance_net_nopool/inp
        self.module_32 = py_nndct.nn.Module('nndct_elemwise_mul') #enhance_net_nopool::enhance_net_nopool/6173
        self.module_33 = py_nndct.nn.Module('nndct_elemwise_mul') #enhance_net_nopool::enhance_net_nopool/6175
        self.module_34 = py_nndct.nn.Add() #enhance_net_nopool::enhance_net_nopool/6105
        self.module_35 = py_nndct.nn.Add() #enhance_net_nopool::enhance_net_nopool/6177
        self.module_36 = py_nndct.nn.Add() #enhance_net_nopool::enhance_net_nopool/6110

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(data=2.0, dtype=torch.float, device='cpu')
        output_module_1 = self.module_1(data=2.0, dtype=torch.float, device='cpu')
        output_module_2 = self.module_2(data=-1.0, dtype=torch.float, device='cpu')
        output_module_3 = self.module_3(data=-1.0, dtype=torch.float, device='cpu')
        output_module_4 = self.module_4(data=-0.699999988079071, dtype=torch.float, device='cpu')
        output_module_5 = self.module_5(input=args[0])
        output_module_6 = self.module_6(output_module_5)
        output_module_6 = self.module_7(output_module_6)
        output_module_6 = self.module_8(output_module_6)
        output_module_9 = self.module_9(output_module_6)
        output_module_9 = self.module_10(output_module_9)
        output_module_9 = self.module_11(output_module_9)
        output_module_12 = self.module_12(output_module_9)
        output_module_12 = self.module_13(output_module_12)
        output_module_12 = self.module_14(output_module_12)
        output_module_15 = self.module_15(output_module_12)
        output_module_15 = self.module_16(output_module_15)
        output_module_15 = self.module_17(output_module_15)
        output_module_18 = self.module_18(dim=1, tensors=[output_module_12,output_module_15])
        output_module_18 = self.module_19(output_module_18)
        output_module_18 = self.module_20(output_module_18)
        output_module_18 = self.module_21(output_module_18)
        output_module_22 = self.module_22(dim=1, tensors=[output_module_9,output_module_18])
        output_module_22 = self.module_23(output_module_22)
        output_module_22 = self.module_24(output_module_22)
        output_module_22 = self.module_25(output_module_22)
        output_module_26 = self.module_26(dim=1, tensors=[output_module_6,output_module_22])
        output_module_26 = self.module_27(output_module_26)
        output_module_26 = self.module_28(output_module_26)
        output_module_26 = self.module_29(output_module_26)
        output_module_26 = self.module_30(input=output_module_26, other=output_module_0)
        output_module_26 = self.module_31(input=output_module_26, other=output_module_3, alpha=1)
        output_module_32 = self.module_32(input=output_module_5, other=output_module_1)
        output_module_33 = self.module_33(input=output_module_5, other=output_module_2)
        output_module_32 = self.module_34(input=output_module_32, other=output_module_33, alpha=1)
        output_module_32 = self.module_35(input=output_module_32, other=output_module_4, alpha=1)
        output_module_36 = self.module_36(input=output_module_5, other=output_module_32, alpha=1)
        return (output_module_36,output_module_26)
