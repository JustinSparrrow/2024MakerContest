# 2024 Maker Contest - Love Bud Mark

---

## Works poster
![Poster](6001716566639_.pic_hd.jpg)

## Background of the work

## Suitable for groups & usage scenarios

## Work process
## Software
The software is mainly responsible for the generation and AR effect of the model, which is divided into two parts: client and server:
- The client takes photos and uploads, and receives new pictures and model download links from the server, and produces AR effects after downloading the model;
- The server receives the picture, first matting the picture intelligently, then 3D modeling the picture, and generating the model download link. At the same time, send the picture to paligemma for picture appreciation, and then send the new picture, download link and picture appreciation to the client.

## Client
Flask was chosen as the framework for the server code;...(Later updated)

### Smart matting
We used the RMBG-1.4 open source library, which can identify the main part of the picture and separate it. It doesn't just subtract the background, it takes the first layer that it can recognize.
To download the necessary packages ([url] (https://huggingface.co/briaai/RMBG-1.4)) :
```
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# When prompted for a password, use an access token with write permissions.
# Generate one from your settings: https://huggingface.co/settings/tokens
Git clone https://huggingface.co/briaai/RMBG-1.4

# If you want to clone without large files - just their pointers
Git clone https://huggingface.co/briaai/RMBG-1.4 GIT_LFS_SKIP_SMUDGE = 1
```
Use RMBG to process pictures and import them first
```python
from RMBG.briarmbg import BriaRMBG
from RMBG.utilities import preprocess_image, postprocess_image
` ` `
Initialize the model
```python
net = BriaRMBG()                                                             # Instantiate the background removal model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        # Detects if there is a GPU available, if so use the GPU, otherwise use the CPU.
net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")                            # Load pre-trained background removal model.
net.to(device)                                                               # Moves the model to the specified device (GPU or CPU).
```
In RMBG, the convolutional neural network (CNN) used is trained based on RGB images, so it only expects the picture to have three channels: R, G and B. If the number of channels is greater than 3 and 4 (A), it will produce errors, so we need to first determine the number of channels in the picture and fix it as a three-channel picture.
```python
# If the image has an alpha channel, convert it to RGB
if orig_im.shape[2] == 4:
orig_im = orig_im[:, :, :3]
```
Image processing:
```python
orig_im_size = orig_im.shape[0:2]
image = preprocess_image(orig_im, model_input_size).to(device)

# inference
result = net(image)

# post process
result_image = postprocess_image(result[0][0], orig_im_size)
```
Finally, numpy is used to crop and cut out the image
```python
np.argwhere(result_image)                       # Finds the coordinates of all non-zero pixels in the resulting image.
top_left = non_zero_coords.min(axis=0)          # Calculates the top-left coordinates of the non-zero pixel region.
bottom_right = non_zero_coords.max(axis=0)      # Calculates the bottom-right coordinates of the non-zero pixel region.
```
Save the result
```python
Image.fromarray(cropped_result_image)                     # Converts the cropped Image to a PIL image.
Image.new("RGBA", pil_im.size, (0, 0, 0))                 # Creates a new transparent background Image.
Image.open(im_path).convert("RGB")                        # open the original Image and make sure it is in RGB format.
orig_image.crop(...)                                      # Crop the corresponding area in the original image.
no_bg_image.paste(orig_cropped_image, mask=pil_im)        # Paste the cropped original image onto the transparent background image, using the removed background image as the mask.
no_bg_image.save("example_image_no_bg.png")               # Save the final image with the background removed.
```

### 3D modeling
The current 3D modeling has not found a good artificial intelligence api, so the first simple processing, through the trimesh library to convert 2d images to 3d models, in fact, the 2d image to increase a little thickness. Finally, export the file to dae using the collada library.
```python
import trimesh
import numpy as np
from PIL import Image
import collada
```
When converted to 3D, the image needs to have four channels, that is, transparency. The significance of transparency is that in image segmentation or background removal tasks, transparency information can help determine which pixels belong to the foreground and which pixels belong to the background and usually only need to deal with the non-transparent part (i.e. the foreground). By checking the value of the Alpha channel, you can easily ignore the transparent part and work with only the pixels that are actually needed. For example, the condition if image_data[y, x, 3] > 0 deals only with pixels whose Alpha value is greater than 0, i.e. the non-transparent part.
```python
def image_to_3d_model(image_path, thickness, output_path):
# Open image and convert to RGBA image
image = Image.open(image_path).convert('RGBA')
image_data = np.array(image)
```