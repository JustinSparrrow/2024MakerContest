import os
import threading
import uuid
from flask import Flask, request, jsonify, send_file
import qrcode.constants
import torch
from skimage import io
from PIL import Image
from RMBG.briarmbg import BriaRMBG
from RMBG.utilities import preprocess_image, postprocess_image
from huggingface_hub import hf_hub_download
import qrcode
import trimesh
import numpy as np
import collada

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
DOWNLOAD_FOLDER = 'downloads'

# 创建文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# 加载背景移除模型
model_path = hf_hub_download("briaai/RMBG-1.4", 'model.pth')
net = BriaRMBG()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)
net.eval()

def remove_background(image_path):
    orig_im = io.imread(image_path)
    
    # 如果图像有 alpha 通道，将其转换为 RGB
    if orig_im.shape[2] == 4:
        orig_im = orig_im[:, :, :3]
    
    orig_im_size = orig_im.shape[0:2]
    model_input_size = [1024, 1024]
    image = preprocess_image(orig_im, model_input_size).to(device)
    result = net(image)
    result_image = postprocess_image(result[0][0], orig_im_size)
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.open(image_path).convert("RGB")  # 确保原始图像是 RGB 格式
    no_bg_image.paste(orig_image, mask=pil_im)
    no_bg_image_path = os.path.join(UPLOAD_FOLDER, 'example_image_no_bg.png')
    no_bg_image.save(no_bg_image_path)
    return no_bg_image_path

def image_to_3d_model(image_path, thickness, output_path):
    # 打开图片并转换为灰度图
    image = Image.open(image_path).convert('L')
    image_data = np.array(image)
    # 获取图片的宽度和高度
    height, width = image_data.shape
    # 创建顶点数组和面数组
    vertices = []
    faces = []
    # 创建前表面顶点
    for y in range(height):
        for x in range(width):
            z = image_data[y, x] / 255.0 * thickness
            vertices.append([x, y, z])
    # 创建后表面顶点
    for y in range(height):
        for x in range(width):
            z = 0
            vertices.append([x, y, z])
    # 创建前表面面
    for y in range(height - 1):
        for x in range(width - 1):
            v1 = y * width + x
            v2 = v1 + 1
            v3 = v1 + width
            v4 = v3 + 1
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    # 创建后表面面
    offset = height * width
    for y in range(height - 1):
        for x in range(width - 1):
            v1 = offset + y * width + x
            v2 = v1 + 1
            v3 = v1 + width
            v4 = v3 + 1
            faces.append([v1, v3, v2])
            faces.append([v2, v3, v4])
    # 创建侧面面
    for y in range(height - 1):
        for x in range(width):
            v1 = y * width + x
            v2 = v1 + width
            v3 = offset + v1
            v4 = offset + v2
            faces.append([v1, v3, v2])
            faces.append([v2, v3, v4])
    for y in range(height):
        for x in range(width - 1):
            v1 = y * width + x
            v2 = v1 + 1
            v3 = offset + v1
            v4 = offset + v2
            faces.append([v1, v3, v2])
            faces.append([v2, v3, v4])
    # 将顶点和面转换为numpy数组
    vertices = np.array(vertices)
    faces = np.array(faces)
    # 创建3D网格
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = str(uuid.uuid4()) + '.png'
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    no_bg_image_path = remove_background(image_path)
    model_filename = str(uuid.uuid4()) + '.dae'
    model_path = os.path.join(MODEL_FOLDER, model_filename)

    thread = threading.Thread(target=image_to_3d_model, args=(no_bg_image_path, 10.0, model_path))
    thread.start()
    thread.join()

    download_link = request.host_url + 'download/' + model_filename
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(download_link)
    qr.make(fit=True)
    qr_img = qr.make_image(fill='black', back_color='white')

    qr_filename = str(uuid.uuid4()) + '.png'
    qr_path = os.path.join(DOWNLOAD_FOLDER, qr_filename)
    qr_img.save(qr_path)

    print(f"Thread {threading.current_thread().name} processed the request.")

    return jsonify({'download_link': download_link, 'qr_code': qr_path})

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(MODEL_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    # 使用新的端口，如8080
    app.run(debug=True, host='0.0.0.0', port=8080)