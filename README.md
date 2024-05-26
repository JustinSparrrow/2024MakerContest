# 2024创客赛 —— 爱芽印记
---

## 作品海报
![作品海报](6001716566639_.pic_hd.jpg)

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