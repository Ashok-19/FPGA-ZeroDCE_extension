#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

# 1) Custom op registration

import model as zmod
import numpy

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
'''
@register_custom_op(op_type="tensor_mul",mapping_to_xir = True) #,attrs_list=["iterations"])
def custom_mul(ctx, x, x_r): #, iterations):
    # element-wise multiplication
    
    #result = mytensor.tensor_multiply(x,x_r)
    result = torch.mul(x,x_r)
   
    
    #print(result.shape)
    #result = torch.Tensor(result)
    #print(result.shape)
    return result'''

class ZeroDCEQ(zmod.enhance_net_nopool):
    def __init__(self, scale_factor=1, iterations=8):
        super().__init__(scale_factor)
        self.iterations = iterations

    def forward(self, x):
        # x_r: the per-pixel curve from the original network
        x, x_r = super().forward(x)  # shape [N,3,H,W]
        # now apply LE​-curve for `iterations` steps using custom op
        return x


# 3) Data loaders (batch_size=1, normalize [0,1])
def get_loader(data_dir, batch_size=1, shuffle=False):
    """
    Loads images from ImageFolder(data_dir), applies YOLO-style letterbox
    resizing to 512x512 with gray padding (114), and returns normalized tensors.
    """
    to_tensor = ToTensor()  # converts PIL Image → tensor [0,1]

    def letterbox_tensor(img_tensor):
        c, h0, w0 = img_tensor.shape
        size = 512
        pad_val = 114.0 / 255.0  # normalized
        scale = size / max(h0, w0)
        new_h = int(h0 * scale)
        new_w = int(w0 * scale)
        # resize
        img_resized = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        # compute symmetric padding
        pad_h = size - new_h
        pad_w = size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded = F.pad(
            img_resized,
            (left, right, top, bottom),
            mode='constant',
            value=pad_val
        )
        return padded

    def transform(img):
        t = to_tensor(img)
        return letterbox_tensor(t)

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Evaluate for fast finetuning
def evaluate_fn(model, loader, *_):
    model.eval()
    with torch.no_grad():
        for img, _ in loader:
            img = img.to(device)
            _ = model(img)

# 4) Main entrypoint
def main():
    parser = argparse.ArgumentParser(
        description="Zero-DCE Quant/Test/Float + export xmodel"
    )
    parser.add_argument("--calib_dir", default='/workspace/src/vai_quantizer/vai_q_pytorch/zdce_extension/data/calib', help="Folder of calibration images")
    parser.add_argument("--test_dir", default='/workspace/src/vai_quantizer/vai_q_pytorch/zdce_extension/data/test/', help="Folder of test images")
    parser.add_argument("--model_path", default='original.pth', help="Path to trained `enhance_net_nopool` weights (.pth)")
    parser.add_argument("--xmodel_out", default="./quantize_result", help="Directory to save exported .xmodel")
    parser.add_argument("--mode", default="calib", choices=["float", "calib", "test"], help="float / calib / test")
    parser.add_argument("--iterations", default=8, type=int, help="Number of LE​-curve iterations")
    parser.add_argument("--deploy", action="store_true", help="Also export .xmodel in test mode")
    parser.add_argument('--fast_finetune', action='store_true', help="Fast fine-tune model during calibration")
    parser.add_argument("--inspect", action="store_true", help="Run Vitis​-AI Inspector on the model before quantization")
    parser.add_argument("--target", default="DPUCZDX8G_ISA1_B4096", help="DPU target IP for Inspector (e.g. DPUCZDX8G_ISA1_B4096)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = ZeroDCEQ(scale_factor=2, iterations=args.iterations).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()

    dummy_input = torch.rand(1, 3, 512, 512)

    if args.inspect:
        from pytorch_nndct import Inspector
        inspector = Inspector(args.target)
        print(f"=> Inspector on {args.target}")
        inspector.inspect(net, (dummy_input,), device=device, output_dir="inspect/", image_format="png")
        print("=> Inspector done")

    from pytorch_nndct.apis import torch_quantizer
    quantizer = torch_quantizer(args.mode, net, dummy_input, device=device)
    qnet = quantizer.quant_model

    if args.mode == "calib":
        loader = get_loader(args.calib_dir)
        with torch.no_grad():
            for i, (img, _) in enumerate(loader):
                img = img.to(device)
                _ = qnet(img)
                print('count', i+1)
        if args.fast_finetune:
            ft_loader = get_loader(args.calib_dir, shuffle=True)
            quantizer.fast_finetune(evaluate_fn, (qnet, ft_loader, None))
        quantizer.export_quant_config()
        print("Calibration complete. Quant config exported.")

    if args.mode in ("float", "test"):
        if args.fast_finetune:
            quantizer.load_ft_param()
        loader = get_loader(args.test_dir)
        print(f"Running {args.mode} inference on {len(loader)} images...")
        with torch.no_grad():
            for img, _ in loader:
                img = img.to(device)
                enhanced = qnet(img)

    if args.mode == "test" and args.deploy:
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel(deploy_check=False)

    print("Done.")

if __name__ == "__main__":
    main()
