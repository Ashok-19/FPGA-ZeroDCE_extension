# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class ZeroDCEQ(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(ZeroDCEQ, self).__init__()
        self.module_0 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1499
        self.module_1 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1505
        self.module_2 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1509
        self.module_3 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1513
        self.module_4 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1517
        self.module_5 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1521
        self.module_6 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1525
        self.module_7 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1529
        self.module_8 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1533
        self.module_9 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1501
        self.module_10 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1507
        self.module_11 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1511
        self.module_12 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1515
        self.module_13 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1519
        self.module_14 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1523
        self.module_15 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1527
        self.module_16 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1531
        self.module_17 = py_nndct.nn.Module('nndct_const') #ZeroDCEQ::1503
        self.module_18 = py_nndct.nn.Input() #ZeroDCEQ::input_0
        self.module_19 = py_nndct.nn.Interpolate() #ZeroDCEQ::ZeroDCEQ/Upsample[upsample_down]/input.3
        self.module_20 = py_nndct.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=3, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv1]/Conv2d[depth_conv]/input.5
        self.module_21 = py_nndct.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv1]/Conv2d[point_conv]/input.7
        self.module_22 = py_nndct.nn.ReLU(inplace=True) #ZeroDCEQ::ZeroDCEQ/ReLU[relu]/input.9
        self.module_23 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=32, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv2]/Conv2d[depth_conv]/input.11
        self.module_24 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv2]/Conv2d[point_conv]/input.13
        self.module_25 = py_nndct.nn.ReLU(inplace=True) #ZeroDCEQ::ZeroDCEQ/ReLU[relu]/input.15
        self.module_26 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=32, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv3]/Conv2d[depth_conv]/input.17
        self.module_27 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv3]/Conv2d[point_conv]/input.19
        self.module_28 = py_nndct.nn.ReLU(inplace=True) #ZeroDCEQ::ZeroDCEQ/ReLU[relu]/input.21
        self.module_29 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=32, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv4]/Conv2d[depth_conv]/input.23
        self.module_30 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv4]/Conv2d[point_conv]/input.25
        self.module_31 = py_nndct.nn.ReLU(inplace=True) #ZeroDCEQ::ZeroDCEQ/ReLU[relu]/1215
        self.module_32 = py_nndct.nn.Cat() #ZeroDCEQ::ZeroDCEQ/input.27
        self.module_33 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv5]/Conv2d[depth_conv]/input.29
        self.module_34 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv5]/Conv2d[point_conv]/input.31
        self.module_35 = py_nndct.nn.ReLU(inplace=True) #ZeroDCEQ::ZeroDCEQ/ReLU[relu]/1257
        self.module_36 = py_nndct.nn.Cat() #ZeroDCEQ::ZeroDCEQ/input.33
        self.module_37 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv6]/Conv2d[depth_conv]/input.35
        self.module_38 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv6]/Conv2d[point_conv]/input.37
        self.module_39 = py_nndct.nn.ReLU(inplace=True) #ZeroDCEQ::ZeroDCEQ/ReLU[relu]/1299
        self.module_40 = py_nndct.nn.Cat() #ZeroDCEQ::ZeroDCEQ/input.39
        self.module_41 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv7]/Conv2d[depth_conv]/input.41
        self.module_42 = py_nndct.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ZeroDCEQ::ZeroDCEQ/CSDN_Tem[e_conv7]/Conv2d[point_conv]/1340
        self.module_43 = py_nndct.nn.Hardsigmoid(inplace=False) #ZeroDCEQ::ZeroDCEQ/1341
        self.module_44 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1500
        self.module_45 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/input
        self.module_46 = py_nndct.nn.Interpolate() #ZeroDCEQ::ZeroDCEQ/Upsample[upsample]/x_r
        self.module_47 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1504
        self.module_48 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1356
        self.module_49 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1506
        self.module_50 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1360
        self.module_51 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1361
        self.module_52 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1363
        self.module_53 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1508
        self.module_54 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1367
        self.module_55 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1510
        self.module_56 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1371
        self.module_57 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1372
        self.module_58 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1374
        self.module_59 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1512
        self.module_60 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1378
        self.module_61 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1514
        self.module_62 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1382
        self.module_63 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1383
        self.module_64 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1385
        self.module_65 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1516
        self.module_66 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1389
        self.module_67 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1518
        self.module_68 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1393
        self.module_69 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1394
        self.module_70 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1396
        self.module_71 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1520
        self.module_72 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1400
        self.module_73 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1522
        self.module_74 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1404
        self.module_75 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1405
        self.module_76 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1407
        self.module_77 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1524
        self.module_78 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1411
        self.module_79 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1526
        self.module_80 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1415
        self.module_81 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1416
        self.module_82 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1418
        self.module_83 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1528
        self.module_84 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1422
        self.module_85 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1530
        self.module_86 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1426
        self.module_87 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1427
        self.module_88 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1429
        self.module_89 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1532
        self.module_90 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1433
        self.module_91 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1534
        self.module_92 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1437
        self.module_93 = py_nndct.nn.Module('nndct_elemwise_mul') #ZeroDCEQ::ZeroDCEQ/1438
        self.module_94 = py_nndct.nn.Add() #ZeroDCEQ::ZeroDCEQ/1440

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(data=2.0, dtype=torch.float, device='cuda')
        output_module_1 = self.module_1(data=-1.0, dtype=torch.float, device='cuda')
        output_module_2 = self.module_2(data=-1.0, dtype=torch.float, device='cuda')
        output_module_3 = self.module_3(data=-1.0, dtype=torch.float, device='cuda')
        output_module_4 = self.module_4(data=-1.0, dtype=torch.float, device='cuda')
        output_module_5 = self.module_5(data=-1.0, dtype=torch.float, device='cuda')
        output_module_6 = self.module_6(data=-1.0, dtype=torch.float, device='cuda')
        output_module_7 = self.module_7(data=-1.0, dtype=torch.float, device='cuda')
        output_module_8 = self.module_8(data=-1.0, dtype=torch.float, device='cuda')
        output_module_9 = self.module_9(data=-1.0, dtype=torch.float, device='cuda')
        output_module_10 = self.module_10(data=9.99999993922529e-09, dtype=torch.float, device='cuda')
        output_module_11 = self.module_11(data=9.99999993922529e-09, dtype=torch.float, device='cuda')
        output_module_12 = self.module_12(data=9.99999993922529e-09, dtype=torch.float, device='cuda')
        output_module_13 = self.module_13(data=9.99999993922529e-09, dtype=torch.float, device='cuda')
        output_module_14 = self.module_14(data=9.99999993922529e-09, dtype=torch.float, device='cuda')
        output_module_15 = self.module_15(data=9.99999993922529e-09, dtype=torch.float, device='cuda')
        output_module_16 = self.module_16(data=9.99999993922529e-09, dtype=torch.float, device='cuda')
        output_module_17 = self.module_17(data=9.99999993922529e-09, dtype=torch.float, device='cuda')
        output_module_18 = self.module_18(input=args[0])
        output_module_19 = self.module_19(input=output_module_18, size=None, scale_factor=[0.5,0.5], mode='bilinear', align_corners=False)
        output_module_19 = self.module_20(output_module_19)
        output_module_19 = self.module_21(output_module_19)
        output_module_19 = self.module_22(output_module_19)
        output_module_23 = self.module_23(output_module_19)
        output_module_23 = self.module_24(output_module_23)
        output_module_23 = self.module_25(output_module_23)
        output_module_26 = self.module_26(output_module_23)
        output_module_26 = self.module_27(output_module_26)
        output_module_26 = self.module_28(output_module_26)
        output_module_29 = self.module_29(output_module_26)
        output_module_29 = self.module_30(output_module_29)
        output_module_29 = self.module_31(output_module_29)
        output_module_32 = self.module_32(dim=1, tensors=[output_module_26,output_module_29])
        output_module_32 = self.module_33(output_module_32)
        output_module_32 = self.module_34(output_module_32)
        output_module_32 = self.module_35(output_module_32)
        output_module_36 = self.module_36(dim=1, tensors=[output_module_23,output_module_32])
        output_module_36 = self.module_37(output_module_36)
        output_module_36 = self.module_38(output_module_36)
        output_module_36 = self.module_39(output_module_36)
        output_module_40 = self.module_40(dim=1, tensors=[output_module_19,output_module_36])
        output_module_40 = self.module_41(output_module_40)
        output_module_40 = self.module_42(output_module_40)
        output_module_40 = self.module_43(output_module_40)
        output_module_40 = self.module_44(input=output_module_40, other=output_module_0)
        output_module_40 = self.module_45(input=output_module_40, other=output_module_9, alpha=1)
        output_module_40 = self.module_46(input=output_module_40, size=None, scale_factor=[2.0,2.0], mode='bilinear', align_corners=False)
        output_module_47 = self.module_47(input=output_module_18, other=output_module_17, alpha=1)
        output_module_48 = self.module_48(input=output_module_18, other=output_module_47)
        output_module_49 = self.module_49(input=output_module_18, other=output_module_1)
        output_module_48 = self.module_50(input=output_module_48, other=output_module_49, alpha=1)
        output_module_51 = self.module_51(input=output_module_40, other=output_module_48)
        output_module_52 = self.module_52(input=output_module_18, other=output_module_51, alpha=1)
        output_module_53 = self.module_53(input=output_module_52, other=output_module_10, alpha=1)
        output_module_54 = self.module_54(input=output_module_52, other=output_module_53)
        output_module_55 = self.module_55(input=output_module_18, other=output_module_2)
        output_module_54 = self.module_56(input=output_module_54, other=output_module_55, alpha=1)
        output_module_57 = self.module_57(input=output_module_40, other=output_module_54)
        output_module_58 = self.module_58(input=output_module_18, other=output_module_57, alpha=1)
        output_module_59 = self.module_59(input=output_module_58, other=output_module_11, alpha=1)
        output_module_60 = self.module_60(input=output_module_58, other=output_module_59)
        output_module_61 = self.module_61(input=output_module_18, other=output_module_3)
        output_module_60 = self.module_62(input=output_module_60, other=output_module_61, alpha=1)
        output_module_63 = self.module_63(input=output_module_40, other=output_module_60)
        output_module_64 = self.module_64(input=output_module_18, other=output_module_63, alpha=1)
        output_module_65 = self.module_65(input=output_module_64, other=output_module_12, alpha=1)
        output_module_66 = self.module_66(input=output_module_64, other=output_module_65)
        output_module_67 = self.module_67(input=output_module_18, other=output_module_4)
        output_module_66 = self.module_68(input=output_module_66, other=output_module_67, alpha=1)
        output_module_69 = self.module_69(input=output_module_40, other=output_module_66)
        output_module_70 = self.module_70(input=output_module_18, other=output_module_69, alpha=1)
        output_module_71 = self.module_71(input=output_module_70, other=output_module_13, alpha=1)
        output_module_72 = self.module_72(input=output_module_70, other=output_module_71)
        output_module_73 = self.module_73(input=output_module_18, other=output_module_5)
        output_module_72 = self.module_74(input=output_module_72, other=output_module_73, alpha=1)
        output_module_75 = self.module_75(input=output_module_40, other=output_module_72)
        output_module_76 = self.module_76(input=output_module_18, other=output_module_75, alpha=1)
        output_module_77 = self.module_77(input=output_module_76, other=output_module_14, alpha=1)
        output_module_78 = self.module_78(input=output_module_76, other=output_module_77)
        output_module_79 = self.module_79(input=output_module_18, other=output_module_6)
        output_module_78 = self.module_80(input=output_module_78, other=output_module_79, alpha=1)
        output_module_81 = self.module_81(input=output_module_40, other=output_module_78)
        output_module_82 = self.module_82(input=output_module_18, other=output_module_81, alpha=1)
        output_module_83 = self.module_83(input=output_module_82, other=output_module_15, alpha=1)
        output_module_84 = self.module_84(input=output_module_82, other=output_module_83)
        output_module_85 = self.module_85(input=output_module_18, other=output_module_7)
        output_module_84 = self.module_86(input=output_module_84, other=output_module_85, alpha=1)
        output_module_87 = self.module_87(input=output_module_40, other=output_module_84)
        output_module_88 = self.module_88(input=output_module_18, other=output_module_87, alpha=1)
        output_module_89 = self.module_89(input=output_module_88, other=output_module_16, alpha=1)
        output_module_90 = self.module_90(input=output_module_88, other=output_module_89)
        output_module_91 = self.module_91(input=output_module_18, other=output_module_8)
        output_module_90 = self.module_92(input=output_module_90, other=output_module_91, alpha=1)
        output_module_93 = self.module_93(input=output_module_40, other=output_module_90)
        output_module_94 = self.module_94(input=output_module_18, other=output_module_93, alpha=1)
        return output_module_94
