# 2024创客赛 —— 爱芽印记
---

## 作品海报
![作品海报](6001716566639_.pic_hd.jpg)

## 作品背景

## 适用群体&使用场景

## 作品流程
### 软件
软件主要负责模型的生成和AR效果，分为客户端和服务端两部分：
- 客户端进行拍照并上传，并接收来自服务端的新图片和模型下载链接，并且在下载完模型之后产生AR效果；
- 服务端接收图片，先智能抠图，再将图片进行3D建模，并生成模型下载链接。同时将图片传给paligemma获取图片鉴赏，再将新图片、下载链接和图片鉴赏传给客户端。

### 客户端
服务端代码我们选择了Flask作为框架；
……（后续更新）

#### 智能抠图
我们使用了RMBG-1.4开源库，该库可以识别图片主体部分并进行分离。它不仅仅是扣除背景，而是将其能够识别到的第一图层抠出。  
先下载必要的包（[网址链接](https://huggingface.co/briaai/RMBG-1.4)）：
```
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# When prompted for a password, use an access token with write permissions.
# Generate one from your settings: https://huggingface.co/settings/tokens
git clone https://huggingface.co/briaai/RMBG-1.4

# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/briaai/RMBG-1.4
```
使用RMBG处理图片，先导入其库
```python
from RMBG.briarmbg import BriaRMBG
from RMBG.utilities import preprocess_image, postprocess_image
```
进行模型初始化  
```python
net = BriaRMBG()                                                            # 实例化背景去除模型。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       # 检测是否有可用的GPU，如果有则使用GPU，否则使用CPU。
net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")                           # 加载预训练的背景去除模型。
net.to(device)                                                              # 将模型移动到指定的设备（GPU或CPU）上。
```
在RMBG中，其使用的卷积神经网络（CNN）是基于RGB图像训练的，所以它只期望图片有三个通道：R、G、B，如果通道数大于3为4（A）的话便会产生错误，所以我们需要先判断图片有多少通道，并且将其固定为三通道图片。
```python
# 如果图像有 alpha 通道，将其转换为 RGB
if orig_im.shape[2] == 4:
    orig_im = orig_im[:, :, :3]
```
对图像进行处理：
```python
orig_im_size = orig_im.shape[0:2]
image = preprocess_image(orig_im, model_input_size).to(device)

# inference 
result = net(image)

# post process
result_image = postprocess_image(result[0][0], orig_im_size)
```
最后就是利用numpy对图像进行裁剪与抠出
```python
np.argwhere(result_image)                        # 找到结果图像中所有非零像素的坐标。
top_left = non_zero_coords.min(axis=0)           # 计算非零像素区域的左上角坐标。
bottom_right = non_zero_coords.max(axis=0)       # 计算非零像素区域的右下角坐标。
```
保存结果
```python
Image.fromarray(cropped_result_image)                   # 将裁剪后的图像转换为PIL图像。
Image.new("RGBA", pil_im.size, (0, 0, 0, 0))            # 创建一个新的透明背景图像。
Image.open(im_path).convert("RGB")                      # 打开原始图像并确保其为RGB格式。
orig_image.crop(...)                                    # 裁剪原始图像中对应区域。
no_bg_image.paste(orig_cropped_image, mask=pil_im)      # 将裁剪后的原始图像粘贴到透明背景图像上，使用去除背景后的图像作为掩码。
no_bg_image.save("example_image_no_bg.png")             # 保存最终的去除背景后的图像。
```

#### 3D建模
目前的3D建模还没有找到很好的人工智能api，所以先简单处理了一下，通过trimesh库将2d图片转换为3d模型，其实就是将2d图片增加了些许厚度。最后通过collada库导出为dae文件。
```python
import trimesh
import numpy as np
from PIL import Image
import collada
```
转化为3D时需要图片有四通道，即有透明度。透明度的意义在于在图像分割或背景去除任务中，透明度信息可以帮助确定哪些像素属于前景，哪些像素属于背景并且通常只需要处理非透明部分（即前景）。通过检查Alpha通道的值，可以轻松地忽略透明部分，从而只处理实际需要的像素。例如，if image_data[y, x, 3] > 0 这一条件只处理Alpha值大于0的像素，即非透明部分。
```python
def image_to_3d_model(image_path, thickness, output_path):
    # 打开图片并转换为RGBA图
    image = Image.open(image_path).convert('RGBA')
    image_data = np.array(image)
```