import base64
import urllib
import requests
import json
import time
import torch
import os
import numpy as np
from PIL import Image
from RMBG.briarmbg import BriaRMBG
from RMBG.utilities import preprocess_image, postprocess_image
import trimesh
import collada
from skimage import io

API_KEY = "y8SxSzo6f4rw2mD2E4G9loga"
SECRET_KEY = "3y9if5vKzzCPpQvgq0RfBpMmQHkuxLrK"

def remove_background(im_path):
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
    no_bg_image_path = "example_image_no_bg.png"
    no_bg_image.save(no_bg_image_path)
    return no_bg_image_path

def image_to_3d_model(image_path, thickness, output_path):
    # 打开图片并转换为RGBA图
    image = Image.open(image_path).convert('RGBA')
    image_data = np.array(image)

    # 获取图片的宽度和高度
    height, width, _ = image_data.shape

    # 创建顶点数组和面数组
    vertices = []
    front_vertex_indices = {}
    back_vertex_indices = {}
    faces = []

    # 创建前表面顶点
    index = 0
    for y in range(height):
        for x in range(width):
            if image_data[y, x, 3] > 0:  # 只处理非透明部分
                z = image_data[y, x, 0] / 255.0 * thickness  # 使用红色通道作为高度
                vertices.append([x, y, z])
                front_vertex_indices[(x, y)] = index
                index += 1

    # 创建后表面顶点
    for y in range(height):
        for x in range(width):
            if image_data[y, x, 3] > 0:  # 只处理非透明部分
                z = 0
                vertices.append([x, y, z])
                back_vertex_indices[(x, y)] = index
                index += 1

    # 创建前表面面
    for y in range(height - 1):
        for x in range(width - 1):
            if (x, y) in front_vertex_indices and (x + 1, y) in front_vertex_indices and (x, y + 1) in front_vertex_indices and (x + 1, y + 1) in front_vertex_indices:
                v1 = front_vertex_indices[(x, y)]
                v2 = front_vertex_indices[(x + 1, y)]
                v3 = front_vertex_indices[(x, y + 1)]
                v4 = front_vertex_indices[(x + 1, y + 1)]
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])

    # 创建后表面面
    for y in range(height - 1):
        for x in range(width - 1):
            if (x, y) in back_vertex_indices and (x + 1, y) in back_vertex_indices and (x, y + 1) in back_vertex_indices and (x + 1, y + 1) in back_vertex_indices:
                v1 = back_vertex_indices[(x, y)]
                v2 = back_vertex_indices[(x + 1, y)]
                v3 = back_vertex_indices[(x, y + 1)]
                v4 = back_vertex_indices[(x + 1, y + 1)]
                faces.append([v1, v3, v2])
                faces.append([v2, v3, v4])

    # 创建侧面面
    for y in range(height - 1):
        for x in range(width):
            if (x, y) in front_vertex_indices and (x, y + 1) in front_vertex_indices and (x, y) in back_vertex_indices and (x, y + 1) in back_vertex_indices:
                v1 = front_vertex_indices[(x, y)]
                v2 = front_vertex_indices[(x, y + 1)]
                v3 = back_vertex_indices[(x, y)]
                v4 = back_vertex_indices[(x, y + 1)]
                faces.append([v1, v3, v2])
                faces.append([v2, v3, v4])

    for y in range(height):
        for x in range(width - 1):
            if (x, y) in front_vertex_indices and (x + 1, y) in front_vertex_indices and (x, y) in back_vertex_indices and (x + 1, y) in back_vertex_indices:
                v1 = front_vertex_indices[(x, y)]
                v2 = front_vertex_indices[(x + 1, y)]
                v3 = back_vertex_indices[(x, y)]
                v4 = back_vertex_indices[(x + 1, y)]
                faces.append([v1, v3, v2])
                faces.append([v2, v3, v4])

    # 将顶点和面转换为numpy数组
    vertices = np.array(vertices)
    faces = np.array(faces)

    # 创建3D网格
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # 计算Bounding Box
    bounding_box = mesh.bounding_box.bounds
    print("Original Bounding Box:", bounding_box)

    # 计算缩放因子和偏移量
    min_bounds, max_bounds = bounding_box
    scale = 1.0 / np.max(max_bounds - min_bounds)
    offset = -min_bounds * scale

    # 缩放和平移顶点
    vertices = (vertices * scale) + offset

    # 更新网格
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # 计算新的Bounding Box
    new_bounding_box = mesh.bounding_box.bounds
    print("Scaled Bounding Box:", new_bounding_box)

    # 使用pycollada导出为.dae文件
    mesh_to_dae(mesh, output_path)

def mesh_to_dae(mesh, output_path):
    # 创建Collada文档
    mesh_data = collada.Collada()

    # 创建顶点数据
    vertex_data = mesh.vertices.flatten()
    vert_src = collada.source.FloatSource("vertices-array", vertex_data, ('X', 'Y', 'Z'))

    # 创建面数据
    indices = mesh.faces.flatten()
    geom = collada.geometry.Geometry(mesh_data, "geometry0", "my_mesh", [vert_src])

    input_list = collada.source.InputList()
    input_list.addInput(0, 'VERTEX', "#vertices-array")

    triset = geom.createTriangleSet(indices, input_list, "materialref")
    geom.primitives.append(triset)
    mesh_data.geometries.append(geom)

    # 创建场景
    mat = collada.material.Material("material0", "mymaterial", collada.material.Effect("effect0", [], "lambert", (1, 1, 1)))
    mesh_data.materials.append(mat)
    matnode = collada.scene.MaterialNode("materialref", mat, inputs=[])
    geomnode = collada.scene.GeometryNode(geom, [matnode])
    node = collada.scene.Node("node0", children=[geomnode])
    myscene = collada.scene.Scene("myscene", [node])
    mesh_data.scenes.append(myscene)
    mesh_data.scene = myscene

    # 保存为.dae文件
    mesh_data.write(output_path)

def submit_request(path, urlencoded=False):
    url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/image-understanding/request?access_token=" + get_access_token()
    
    # image 可以通过 get_file_content_as_base64("C:\fakepath\example.jpg",False) 方法获取
    payload = json.dumps({
        "image": get_file_content_as_base64(path, urlencoded),
        "question": "请用优雅的语言表达鉴赏一下图片内容",
        "output_CHN": True
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    response_data = response.text
    data = json.loads(response_data)
    task_id = data['result']['task_id']
    return task_id

def get_request(path, urlencoded=False):
    task_id = submit_request(path, urlencoded)
    
    while True:
        url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/image-understanding/get-result?access_token=" + get_access_token()
        
        payload = json.dumps({
            "task_id": task_id
        })
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        
        response_data = response.text
        data = json.loads(response_data)
        
        # 检查 ret_code
        ret_code = data['result']['ret_code']
        if ret_code == 0:
            response_data = response.text
            data = json.loads(response_data)
            text = data['result']['description']
            print("Final Result:", text)
            break
        else:
            print("Processing... Retrying in 5 seconds.")
            time.sleep(5)  # 等待5秒后再检查

def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded 
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

if __name__ == '__main__':
    im_path = '/Users/moqi/Desktop/竞赛/2024创客赛/me.jpg'
    
    # 1. 去除背景
    no_bg_image_path = remove_background(im_path)
    
    # 2. 生成3D模型
    thickness = 10.0  # 设置厚度
    output_path = 'output_model.dae'  # 设置输出文件路径
    image_to_3d_model(no_bg_image_path, thickness, output_path)
    
    # 3. 提交图像处理请求并获取结果
    get_request(no_bg_image_path, False)