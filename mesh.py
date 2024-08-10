import copy
import math

import kaolin as kal
import trimesh
import torch
import utils
from utils import device
import kaolin
import numpy as np
import os
from render import Renderer
from torchvision import transforms
from Normalization import MeshNormalizer, Normalizer
from pathlib import Path


def export_results(frontview_center, output_path, mesh, color):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    base_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5)
    final_color = torch.clamp(color + base_color, 0, 1)

    save_rendered_results(frontview_center, output_path, final_color, mesh)


def save_rendered_results(frontview_center, output_path, final_color, mesh):
    default_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5, device=device)
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                   mesh.faces.to(device))
    kal_render = Renderer(
        camera=kal.render.camera.generate_perspective_projection(np.pi / 4, 1280 / 720).to(device),
        dim=(1280, 720))
    MeshNormalizer(mesh)()
    img, mask = kal_render.render_single_view(mesh, frontview_center[1], frontview_center[0],
                                              radius=2.5,
                                              background=torch.tensor([1, 1, 1]).to(device).float(),
                                              return_mask=True)
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(output_path + "/init_cluster.png")

    MeshNormalizer(mesh)()
    # Vertex colorings
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(final_color.unsqueeze(0).to(device),
                                                                   mesh.faces.to(device))
    img, mask = kal_render.render_single_view(mesh, frontview_center[1], frontview_center[0],
                                              radius=2.5,
                                              background=torch.tensor([1, 1, 1]).to(device).float(),
                                              return_mask=True)
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(output_path, f"final_cluster.png"))


class MeshSeg:
    def __init__(self, class_opt, mesh_path, seg_path, color=torch.tensor([0.0, 0.0, 1.0])):
        if ".obj" in mesh_path:
            self.mesh = trimesh.load_mesh(mesh_path)
        else:
            raise ValueError(f"{mesh_path} extension is not the 'obj'.")

        self.vertices = torch.from_numpy(self.mesh.vertices).to(device).float()
        self.faces = torch.from_numpy(self.mesh.faces).to(device)
        self.vertex_normals = torch.nn.functional.normalize(torch.from_numpy(self.mesh.vertex_normals).to(device).float())

        self.face_attributes = None
        self.classes = set()
        self.face_classes = []
        self.vertex_classes = []

        self.set_mesh_color(color)
        if 'sub' in seg_path:
            self.set_vertex_classes_sub(seg_path, class_opt)
        else:
            self.set_face_classes(seg_path)
            self.set_vertex_classes()

    # set the face_attributes (color)
    def set_mesh_color(self, color):
        self.face_attributes = utils.get_face_attributes_from_color(self, color)

    def set_vertex_classes_sub(self, seg_path, class_opt):
        vc_temp_lst = []
        c_temp_set = set()
        with open(seg_path, 'r') as f:
            for line in f:
                vc_temp_lst.append(int(line))
                c_temp_set.add(int(line))
        class_count = len(c_temp_set) - 1
        for v_class in vc_temp_lst:
            if class_opt == 0:
                cla = v_class
            elif class_opt == 1:
                cla = round(v_class / class_count, 2)
            else:
                raise ValueError("class option error (mesh.MeshSeg)")
            self.vertex_classes.append(cla)
            self.classes.add(cla)


    # read face classes from seg file
    def set_face_classes(self, seg_path):
        if ".seg" in seg_path:
            with open(seg_path, 'r') as f:
                for line in f:
                    self.face_classes.append(int(line))
                    self.classes.add(int(line))

    # compute vertex classes from face classes
    def set_vertex_classes(self):
        vert_seg_dict = dict()
        faces = self.faces.detach().cpu().tolist()
        for i in range(len(faces)):
            face = faces[i]
            face_class = self.face_classes[i]
            for v in face:
                if v not in vert_seg_dict:
                    vert_seg_dict[v] = [face_class]
                else:
                    vert_seg_dict[v].append(face_class)
        vert_seg_dict = sorted(vert_seg_dict.items())
        for vs in vert_seg_dict:
            vertex_class = max(set(vs[1]), key=vs[1].count)
            self.vertex_classes.append(vertex_class)

    # visualization
    def visualize(self, output_path, front_view):
        vertex_color_lst = []
        color_dict = dict()
        for i in self.classes:
            color_dict[i] = list((np.random.choice(range(128), size=3) / 256))
        for vc in self.vertex_classes:
            assert vc in color_dict, "There is vertex that have missing class. Use other mesh."
            vertex_color_lst.append(color_dict[vc])
        prior_color = torch.tensor(vertex_color_lst).float()
        export_results(front_view, output_path, self, prior_color)


class Mesh:
    def __init__(self, mesh_path, color=torch.tensor([0.5, 0.5, 0.5])):
        if ".obj" in mesh_path:
            self.mesh = trimesh.load_mesh(mesh_path)
        else:
            raise ValueError(f"{mesh_path} extension is not the 'obj'.")
        self.vertices = torch.from_numpy(self.mesh.vertices).to(device).float()
        self.faces = torch.from_numpy(self.mesh.faces).to(device)
        self.vertex_normals = torch.nn.functional.normalize(torch.from_numpy(self.mesh.vertex_normals).to(device).float())
        self.face_attributes = None
        # self.vertex_colors = torch.from_numpy(self.mesh.visual.vertex_colors[:, :3] / 255.0).to(device).float()
        # self.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(self.vertex_colors.unsqueeze(0).to(device),
        #                                                                self.faces.to(device))
        self.set_mesh_color(color)

    def set_mesh_color(self, color):
        self.face_attributes = utils.get_face_attributes_from_color(self, color)

    def export(self, filename, color=None):
        with open(filename, "w+") as f:
            for vi, v in enumerate(self.vertices):
                if color is None:
                    f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
                else:
                    f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
                if self.vertex_normals is not None:
                    f.write("vn %f %f %f\n" % (
                    self.vertex_normals[vi, 0], self.vertex_normals[vi, 1], self.vertex_normals[vi, 2]))
            for face in self.faces:
                f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))



class MultiMesh:
    def __init__(self, mesh_path_list, device="cpu", color=torch.tensor([0.5, 0.5, 0.5]), offset_scale=1.0):

        self.device = device

        if not isinstance(mesh_path_list, list):
            raise AssertionError("Mesh paths must be wrapped into a list!")

        self.offset_directions = []

        n_meshes = len(mesh_path_list)

        if n_meshes > 1:
            angle_chunk = math.pi * 2 / n_meshes

            for m_idx in range(n_meshes):
                angle = angle_chunk * m_idx
                z_off = round(math.cos(angle), ndigits=3) * offset_scale
                x_off = round(math.sin(angle), ndigits=3) * offset_scale
                self.offset_directions.append([x_off, z_off])
        else:
            self.offset_directions.append([0., 0.])

        self.mesh_v_boundaries = []

        vertices = []
        faces = []
        vertex_normals = []
        offset = 0 # torch.tensor([0]).to(device)
        for m_idx in range(n_meshes):
            mesh_path = mesh_path_list[m_idx]
            if mesh_path[-4:] != ".obj":
                raise ValueError(f"{mesh_path} extension is not '.obj'.")

            mesh = trimesh.load_mesh(mesh_path)

            mesh_vertices = torch.from_numpy(mesh.vertices).to(device).float()
            mesh_faces = torch.from_numpy(mesh.faces).to(device)
            mesh_vertex_normals = torch.nn.functional.normalize(
                torch.from_numpy(mesh.vertex_normals).to(device).float())

            normalizer = Normalizer.get_bounding_sphere_normalizer(mesh_vertices)
            mesh_vertices = normalizer(mesh_vertices)

            n_v = len(mesh.vertices)
            for v_idx in range(n_v):
                mesh_vertices[v_idx][0] += self.offset_directions[m_idx][0]
                mesh_vertices[v_idx][2] += self.offset_directions[m_idx][1] # 왜 z축을 해야하는진 모르겠지만 일단 일케 하면 댐..

            vertices.append(mesh_vertices)

            for f_idx in range(len(mesh_faces)):
                mesh_faces[f_idx][0] += offset
                mesh_faces[f_idx][1] += offset
                mesh_faces[f_idx][2] += offset

            faces.append(mesh_faces)

            vertex_normals.append(mesh_vertex_normals)

            start = offset
            offset += n_v
            self.mesh_v_boundaries.append([start, offset])

        self.vertices = torch.concat(vertices).to(device).float()
        self.faces = torch.concat(faces).to(device)
        self.vertex_normals = torch.nn.functional.normalize(torch.concat(vertex_normals).to(device).float())

        self.face_attributes = None
        self.set_mesh_color(color)

    def set_mesh_color(self, color):
        self.face_attributes = utils.get_face_attributes_from_color(self, color)
        return

    def rotate_mesh(self, mesh_idx, x_deg=0., y_deg=0., z_deg=0.):
        start, end = self.mesh_v_boundaries[mesh_idx]


        # return to origin
        for idx in range(start, end):
            self.vertices[idx][0] -= self.offset_directions[mesh_idx][0]
            self.vertices[idx][2] -= self.offset_directions[mesh_idx][1]  # 왜 z축을 해야하는진 모르겠지만 일단 일케 하면 댐..
        rotation_x = torch.tensor([[1, 0, 0, 0],
                      [0, math.cos(x_deg), -math.sin(x_deg), 0],
                      [0, math.sin(x_deg), math.cos(x_deg), 0],
                      [0, 0, 0, 1]
                      ])
        rotation_y = torch.tensor([[math.cos(y_deg), 0, math.sin(y_deg), 0],
                      [0, 1, 0, 0],
                      [-math.sin(y_deg), 0, math.cos(y_deg), 0],
                      [0, 0, 0, 1]
                      ])
        rotation_z = torch.tensor([[math.cos(z_deg), -math.sin(z_deg), 0, 0],
                      [math.sin(z_deg), math.cos(z_deg), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]
                      ])
        rotation_mat = torch.matmul(rotation_y, rotation_x)
        rotation_mat = torch.matmul(rotation_z, rotation_mat)
        transform_mat = copy.deepcopy(rotation_mat)
        transform_mat[0][3] = self.offset_directions[mesh_idx][0]
        transform_mat[2][3] = self.offset_directions[mesh_idx][1]
        rotation_mat = rotation_mat.to(self.device)
        transform_mat = transform_mat.to(self.device)

        # rotation & return
        # must rotate the vertex normals as well
        for idx in range(start, end):
            vert_homogen_coord = torch.transpose(torch.tensor([self.vertices[idx][0], self.vertices[idx][1], self.vertices[idx][2], 1]).unsqueeze(0), 0, 1).to(self.device)
            vert_norm_homogen_coord = torch.transpose(torch.tensor([self.vertex_normals[idx][0], self.vertex_normals[idx][1], self.vertex_normals[idx][2], 1]).unsqueeze(0), 0, 1).to(self.device)
            self.vertices[idx] = torch.transpose(torch.matmul(transform_mat, vert_homogen_coord), 0, 1)[0][:3]
            self.vertex_normals[idx] = torch.transpose(torch.matmul(rotation_mat, vert_norm_homogen_coord), 0, 1)[0][:3]

        return