import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy 
#from pytorch_nndct.utils import register_custom_op

#from torch.jit import script



'''
@script
def _ts_mul(x_r):
    return tensor_split(x_r)'''

class enhance_net_nopool(nn.Module):

	def __init__(self,scale_factor=None):
		super(enhance_net_nopool, self).__init__()
		self.scale_factor = scale_factor

		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 

		#self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		#self.upsample = nn.Upsample(scale_factor=4,mode='nearest')
		#self.upsample_down = nn.Upsample(scale_factor=0.25,mode='nearest')
	
	'''
	@register_custom_op(op_type="tensor_split",mapping_to_xir = True)
    def tensor_split(ctx, x_r):'''
        
        
        
	def forward(self, x):

		x_down = x

		x1 = self.relu(self.e_conv1(x_down))
		# p1 = self.maxpool(x1)
		x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4(x3))

		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

		#x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        
		x_r = F.hardsigmoid(self.e_conv7(torch.cat([x1,x6], 1))) 
		x_r = 2*x_r + (-1)
		'''
		r1 = x_r[:, 0:3, :, :]
		r2 = x_r[:, 3:6, :, :]
		r3 = x_r[:, 6:9, :, :]
		r4 = x_r[:, 9:12, :, :]
		r5 = x_r[:, 12:15, :, :]
		r6 = x_r[:, 15:18, :, :]
		r7 = x_r[:, 18:21, :, :]
		r8 = x_r[:, 21:24, :, :]'''
		
		x = x.unsqueeze(2)


		#x_r = self.upsample(x_r)
		
		

        
		x1 = x + 0.00000001
		x = x + x_r*(torch.mul(x,x1)+ (-1)*x)
        
		x1 = x + 0.00000001
		x = x + x_r*(torch.mul(x,x1)+ (-1)*x)
        
		x1 = x + 0.00000001
		x = x + x_r*(torch.mul(x,x1)+ (-1)*x)
        
		x1 = x + 0.00000001
		x = x + x_r*(torch.mul(x,x1)+ (-1)*x)
        
		x1 = x + 0.00000001
		x = x + x_r*(torch.mul(x,x1)+ (-1)*x)
        
		x1 = x + 0.00000001
		x = x + x_r*(torch.mul(x,x1)+ (-1)*x)
        
		x1 = x + 0.00000001
		x = x + x_r*(torch.mul(x,x1)+ (-1)*x)
        
		x1 = x + 0.00000001
		x = x + x_r*(torch.mul(x,x1)+ (-1)*x)
		
		
        
         

		#r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
		return x
