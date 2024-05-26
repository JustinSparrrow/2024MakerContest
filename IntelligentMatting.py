from skimage import io
import torch
import os
import numpy as np
from PIL import Image
from RMBG.briarmbg import BriaRMBG
from RMBG.utilities import preprocess_image, postprocess_image

im_path = "/Users/moqi/Desktop/竞赛/2024创客赛/me.jpg"

net = BriaRMBG()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
net.to(device)

# prepare input
model_input_size = [1024, 1024]
orig_im = io.imread(im_path)

# 如果图像有 alpha 通道，将其转换为 RGB
if orig_im.shape[2] == 4:
    orig_im = orig_im[:, :, :3]

orig_im_size = orig_im.shape[0:2]
image = preprocess_image(orig_im, model_input_size).to(device)

# inference 
result = net(image)

# post process
result_image = postprocess_image(result[0][0], orig_im_size)

# 计算扣出部分的边界
non_zero_coords = np.argwhere(result_image)
top_left = non_zero_coords.min(axis=0)
bottom_right = non_zero_coords.max(axis=0)

# 裁剪图像
cropped_result_image = result_image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

# save result
pil_im = Image.fromarray(cropped_result_image)
no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
orig_image = Image.open(im_path).convert("RGB")  # 确保原始图像是 RGB 格式
orig_cropped_image = orig_image.crop((top_left[1], top_left[0], bottom_right[1]+1, bottom_right[0]+1))
no_bg_image.paste(orig_cropped_image, mask=pil_im)
no_bg_image.save("example_image_no_bg.png")