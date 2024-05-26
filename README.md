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
- net = BriaRMBG()：实例化背景去除模型。
- device：检测是否有可用的GPU，如果有则使用GPU，否则使用CPU。
- BriaRMBG.from_pretrained("briaai/RMBG-1.4")：加载预训练的背景去除模型。
- net.to(device)：将模型移动到指定的设备（GPU或CPU）上。
```python
net = BriaRMBG()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
net.to(device)
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
- np.argwhere(result_image)：找到结果图像中所有非零像素的坐标。
- top_left = non_zero_coords.min(axis=0)：计算非零像素区域的左上角坐标。
- bottom_right = non_zero_coords.max(axis=0)：计算非零像素区域的右下角坐标。
```python
np.argwhere(result_image)：找到结果图像中所有非零像素的坐标。
top_left = non_zero_coords.min(axis=0)：计算非零像素区域的左上角坐标。
bottom_right = non_zero_coords.max(axis=0)：计算非零像素区域的右下角坐标。
```