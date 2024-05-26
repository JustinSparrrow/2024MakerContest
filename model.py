import trimesh
import numpy as np
from PIL import Image
import collada

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

# 使用示例
image_path = 'example_image_no_bg.png'  # 替换为你的图片路径
thickness = 10.0  # 设置厚度
output_path = 'output_model.dae'  # 设置输出文件路径

image_to_3d_model(image_path, thickness, output_path)