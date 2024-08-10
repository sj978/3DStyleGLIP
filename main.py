import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms

from neural_style_field_tango import NeuralStyleField
from render import Renderer
from mesh import Mesh

import clip
import open3d as o3d
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

import matplotlib.pylab as pylab
from pathlib import Path
import numpy as np
import random
import copy
import os
import math
import nltk
import argparse


pylab.rcParams['figure.figsize'] = 20, 12
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# Define report functions
def report_process(dir, i, losses, metrics, rendered_images):
    print(f'\nIter: {i} \nLosses: {losses}')
    torchvision.utils.save_image(rendered_images[:,0:3,:,:], os.path.join(dir, 'iter_{}.jpg'.format(i)))
    f = open(os.path.join(dir, 'losses.txt'), 'a')
    f.write(
        f'Iter: {i}, \nTotal loss: {losses}, \nMetric: {metrics}, \n\n'
    )
    f.close()


# Define loss functions
def get_classification_global_loss(candidate_high_indexes_GT, box_scores_pred, loss_option):
    score_candidate_lst_pred = []
    for lev in range(5):
        high_indexes_TF = candidate_high_indexes_GT[lev][0]
        score_candidate_pred = box_scores_pred[lev][0][high_indexes_TF]
        score_candidate_lst_pred.append(score_candidate_pred)
    score_candidate_flatten = torch.cat(score_candidate_lst_pred, dim=0)
    target = torch.ones(score_candidate_flatten.shape).to(device)
    if loss_option == 'MSE':
        criterion = nn.MSELoss()
        classification_loss = criterion(score_candidate_flatten, target)
    elif loss_option == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
        classification_loss = criterion(score_candidate_flatten, target)
    elif loss_option == 'Negative':
        loss_negative = -score_candidate_flatten
        classification_loss = torch.mean(loss_negative)
    else:
        raise AttributeError(f'GLIP loss option is wrong: {loss_option}')

    return classification_loss


def get_normalize_mesh(pro_path):
    mesh = o3d.io.read_triangle_mesh(pro_path)
    vertices = np.asarray(mesh.vertices)
    shift = np.mean(vertices,axis=0)
    scale = np.max(np.linalg.norm(vertices-shift, ord=2, axis=1))
    vertices = (vertices-shift) / scale
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


def find_noun_phrases(caption: str):
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    return noun_phrases

# Settings
front_view_dict = {'cup': [3.1, -1.0, 2.0], 'lamp': [0.0, 0.5, 2.2], 'bag': [0.3, 0.3, 2.0], 'goblet': [0.0, 0.5, 2.7],
                   'bear': [0.9, 0.0, 2.2], 'headset': [-0.7, 0.0, 2.2], 'candle': [0.0, 0.3, 2.2], 'deer': [-0.8, 0.0, 2.0],
                   'airplane': [0.8, 0.5, 2.0], 'armadillo': [-2.0, 0.0, 2.0], 'bird': [0.8, 0.0, 2.0], 
                   'candle': [1.4, 0.3, 2.0], 'cat': [0.8, 0.0, 2.0], "crab": [-1.1, 0.0, 2.0],
                   "dog": [1.0, 0.0, 2.0], "dragon": [-3 * math.pi / 4, 0., 1.8], "guitar": [1.2, 0.3, 1.8], 
                   'hammer':[-0.5, 0.0, 2.0], 'horse': [-0.5, 0.0, 2.0], 'person': [0.0, 0.0, 2.0],
                   "flower": [0.5, 0.3, 2.5], "mushroom": [0.0, 0.5, 3.0], "rose": [1.3, 0.0, 3.0], "necklace": [1.55, 1.0, 3.0],
                   "necklace1": [1.6, -1.5, 2.5], "necklace2": [1.6, -1.5, 2.5], "ring": [1.7, -0.6, 3.0], "ring1": [1.2, 0.8, 3.0],
                   'hammer1':[-0.5, 0.3, 2.0], 'hammer2':[-0.5, 0.0, 2.0], 'axe':[1.6, 0.5, 2.0],
                   }

tuning_option = [True]
glip_loss_option = ['MSE']
glip_clip_ratio = [1]
clip_loss_option = ['all']

# Main
def train(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    mesh_type = args.mesh_type
    content_prompt = args.content_prompt
    style_prompt = args.style_prompt

    # Hyper-parameter settings
    seed = args.seed
    n_iter = args.n_iter
    img_size = args.img_size

    n_views = args.n_views
    n_normaugs = args.n_normaugs
    frontview_std = args.frontview_std
    lr_decay = args.lr_decay

    background_color = args.background_color
    crop_ratio_min = args.crop_ratio_min
    crop_ratio_max = args.crop_ratio_max

    # SVBRDF
    material_random_pe_numfreq=args.material_random_pe_numfreq
    material_random_pe_sigma=args.material_random_pe_sigma
    material_nerf_pe_numfreq=args.material_nerf_pe_numfreq
    # Light
    num_lgt_sgs=args.num_lgt_sgs
    # NORMAL
    max_delta_theta=args.max_delta_theta
    max_delta_phi=args.max_delta_phi
    normal_random_pe_numfreq=args.normal_random_pe_numfreq
    normal_random_pe_sigma=args.normal_random_pe_sigma
    normal_nerf_pe_numfreq=args.normal_nerf_pe_numfreq
    if_normal_clamp=args.if_normal_clamp
    # Rendered image size
    width=args.width
    # etc
    symmetry=args.symmetry
    init_r_and_s=args.init_r_and_s
    init_roughness=args.init_roughness
    init_specular=args.init_specular
    local_percentage=args.local_percentage


    # Constrain all sources of randomness
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    for is_tune in tuning_option:
        for g_loss_option in glip_loss_option:
            for g_c_ratio in glip_clip_ratio:
                for c_loss_option in clip_loss_option:
                    mesh_path = input_dir + mesh_type + '.obj'
                    output_path = os.path.join(output_dir, os.path.join(mesh_type, content_prompt.replace('. ', '') + '_' + style_prompt.replace('. ', '').replace(', ', '') + '_'+ str(seed)))
                    
                    content_noun = find_noun_phrases(content_prompt)
                    style_noun = find_noun_phrases(style_prompt)
                    if len(content_noun) != len(style_noun):
                        output_path = os.path.join(output_dir, os.path.join(mesh_type, os.path.join('style_failed', content_prompt.replace('. ', '') + '_' + style_prompt.replace('. ', '').replace(', ', '') + '_'+ str(seed))))
                        
                    Path(output_path).mkdir(parents=True, exist_ok=True)
                    
                    print(f'output path = {output_path}')
                    print(f'content: "{content_prompt}", style: "{style_prompt}"')


                    clip_model = None
                    clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)  # embedding shape (1, 512)
                    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))
                    img_resizer = transforms.Resize((224, 224))

                    augment_transform = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(1, 1)),
                        # transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
                        # clip_normalizer
                    ])
                    local_transform = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(crop_ratio_min, crop_ratio_max)),
                        # transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
                        # clip_normalizer
                    ])

                    frontview_center = front_view_dict[mesh_type]
                    radius = frontview_center[-1]
                    center_elev = frontview_center[1]
                    center_azim = frontview_center[0]

                    model = NeuralStyleField(material_random_pe_numfreq=material_random_pe_numfreq,
                                                material_random_pe_sigma=material_random_pe_sigma,
                                                num_lgt_sgs=num_lgt_sgs,
                                                max_delta_theta=max_delta_theta,
                                                max_delta_phi=max_delta_phi,
                                                normal_nerf_pe_numfreq=normal_nerf_pe_numfreq,
                                                normal_random_pe_numfreq=normal_random_pe_numfreq,
                                                symmetry=symmetry,
                                                radius=radius,
                                                background=background_color,
                                                init_r_and_s=init_r_and_s,
                                                width=width,
                                                init_roughness=init_roughness,
                                                init_specular=init_specular,
                                                material_nerf_pe_numfreq=material_nerf_pe_numfreq,
                                                normal_random_pe_sigma=normal_random_pe_sigma,
                                                if_normal_clamp=if_normal_clamp
                                                )
                    if torch.cuda.is_available():
                        model.cuda()

                    model.train()
                    optim = torch.optim.AdamW(model.parameters(), 0.0005)

                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [500, 1000], lr_decay)

                    mesh = get_normalize_mesh(mesh_path)
                    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
                    scene = o3d.t.geometry.RaycastingScene()
                    scene.add_triangles(mesh)

                    original_mesh = get_normalize_mesh(mesh_path)
                    original_mesh = o3d.t.geometry.TriangleMesh.from_legacy(original_mesh)
                    original_scene = o3d.t.geometry.RaycastingScene()
                    original_scene.add_triangles(original_mesh)

                    # For loss (GLIP)
                    config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml"
                    weight_file = "GLIP/MODEL/glip_large_model.pth"
                    if is_tune:
                        config_file = "GLIP/configs/pretrain/Tuned_glip_Swin_L.yaml"
                        weight_file = "GLIP/MODEL/Tuned_" + mesh_type + ".pth"

                    cfg.local_rank = 0
                    cfg.num_gpus = 1
                    cfg.merge_from_file(config_file)
                    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
                    cfg.merge_from_list(["MODEL.DEVICE", device])
                    glip_demo = GLIPDemo(
                        cfg,
                        min_image_size=img_size,
                        confidence_threshold=0.7,
                        show_mask_heatmaps=False
                    )

                    # Training
                    for i in range(n_iter):
                        optim.zero_grad()

                        if n_views > 1:
                            side_view_elev = torch.cat((torch.tensor([frontview_center[1]]), torch.randn(n_views - 1) * np.pi / frontview_std + center_elev))
                            side_view_azim = torch.cat((torch.tensor([frontview_center[0]]),torch.randn(n_views - 1) * 2 * np.pi / frontview_std + center_azim))
                        if n_views == 1:
                            side_view_elev = torch.randn(n_views) * np.pi / frontview_std + center_elev
                            side_view_azim = torch.rand(n_views) * 0.1 + center_azim

                        rendered_mesh_content = model(scene=original_scene,
                                                    num_views=n_views,
                                                    side_azim=side_view_azim,
                                                    side_elev=side_view_elev,
                                                    )
                        rendered_mesh_content = rendered_mesh_content.cuda()

                        rendered_mesh_style = model(scene=scene,
                                                num_views=n_views,
                                                side_azim=side_view_azim,
                                                side_elev=side_view_elev,
                                                )
                        rendered_mesh_style = rendered_mesh_style.cuda()

                        # GLIP loss
                        if i % (g_c_ratio+1) == 1:
                            loss = 0.0
                            glip_loss_lst = []
                            glip_loss_regression_lst = []
                            cropped_stylized_images = dict()
                            for i_v in range(n_views):
                                glip_loss = 0.0
                                glip_loss_regression = 0.0
                                content_mesh = rendered_mesh_content[i_v]  # original mesh
                                stylized_mesh = rendered_mesh_style[i_v]  # stylized mesh

                                c_predictions, c_fused_visual_features, c_dot_product_logits, c_box_regression, c_box_scores, \
                                    c_candidate_indexes, c_candidate_high_indexes_dot, c_candidate_high_indexes_score, c_anchors, c_positive_map \
                                    = glip_demo.compute_prediction(content_mesh, content_prompt)
                                c_top_predictions = glip_demo._post_process(c_predictions, 0.5)

                                s_predictions, s_fused_visual_features, s_dot_product_logits, s_box_regression, s_box_scores, \
                                    s_candidate_indexes, s_candidate_high_indexes_dot, s_candidate_high_indexes_score, s_anchors, s_positive_map \
                                    = glip_demo.compute_prediction_grad_true(stylized_mesh, style_prompt)
                                s_top_predictions = glip_demo._post_process(s_predictions, 0.5)

                                glip_loss = get_classification_global_loss(c_candidate_high_indexes_score, s_box_scores, g_loss_option)
                                glip_loss_lst.append(glip_loss.item())
                                # total loss
                                loss += glip_loss

                            if loss == 0.0:
                                continue

                            loss = loss / n_views
                            loss.backward()
                            optim.step()
                            lr_scheduler.step()

                        # CLIP loss
                        else:
                            if i == 0:
                                loss = 0.0
                                glip_loss_lst = []
                                glip_loss_regression_lst = []
                            cropped_prompt_lst = []
                            s_p_lst = style_prompt.split('. ')
                            for i_c in range(len(s_p_lst)):
                                prompt_crop = s_p_lst[i_c]
                                crop_token = clip.tokenize([prompt_crop]).to(device)
                                cropped_prompt_clip = clip_model.encode_text(crop_token)
                                cropped_prompt_lst.append(cropped_prompt_clip)

                            cropped_stylized_images = dict()
                            for i_v in range(n_views):
                                content_mesh = rendered_mesh_content[i_v]  # original mesh
                                stylized_mesh = rendered_mesh_style[i_v]  # stylized mesh

                                c_predictions, c_fused_visual_features, c_dot_product_logits, c_box_regression, c_box_scores, \
                                    c_candidate_indexes, c_candidate_high_indexes_dot, c_candidate_high_indexes_score, c_anchors, c_positive_map \
                                    = glip_demo.compute_prediction(content_mesh, content_prompt)
                                c_top_predictions = glip_demo._post_process(c_predictions, 0.5)

                                stylized_view = stylized_mesh
                                gt_boxlist = c_top_predictions
                                for b_idx in range(len(gt_boxlist.bbox)):

                                    x1, y1, x2, y2 = map(int, gt_boxlist.bbox[b_idx])

                                    # Color
                                    cropped_colored_segment = transforms.functional.crop(
                                        stylized_view,
                                        y1, x1,
                                        y2 - y1 + 1,
                                        x2 - x1 + 1)
                                    cropped_colored_segment = img_resizer(
                                        cropped_colored_segment)

                                    cls = int(gt_boxlist.extra_fields["labels"][b_idx])
                                    if cls not in cropped_stylized_images:
                                        cropped_stylized_images[cls] = [cropped_colored_segment]
                                    else:
                                        cropped_stylized_images[cls].append(
                                            cropped_colored_segment)

                            for key, value in cropped_stylized_images.items():
                                cropped_stylized_images[key] = torch.stack(value, dim=0)

                            clip_loss = 0.0
                            clip_local_loss = 0.0
                            total_clip_loss = 0.0
                            for key, value in cropped_stylized_images.items():

                                colored_crops = value

                                encoded_text = cropped_prompt_lst[key - 1]

                                if c_loss_option == 'all' or c_loss_option == 'global':
                                    clip_image = augment_transform(colored_crops[:,0:3,:,:])
                                    rendered_clip = clip_model.encode_image(clip_image)
                                    clip_loss -= torch.cosine_similarity(torch.mean(rendered_clip, dim=0, keepdim=True), encoded_text)

                                if c_loss_option == 'all' or c_loss_option == 'local':
                                    for _ in range(n_normaugs):
                                        clip_image_local = local_transform(colored_crops)
                                        shape = clip_image_local.shape[0] * clip_image_local.shape[2] * clip_image_local.shape[3]
                                        object_percent = torch.sum(clip_image_local[:,3,:,:]==1) / shape
                                        while object_percent <= local_percentage: 
                                            clip_image_local = local_transform(colored_crops)
                                            object_percent = torch.sum(clip_image_local[:,3,:,:]==1) / shape

                                        clip_image_local = clip_image_local[:,0:3,:,:]
                                        rendered_clip_local = clip_model.encode_image(clip_image_local)
                                        clip_local_loss -= torch.cosine_similarity(torch.mean(rendered_clip_local, dim=0, keepdim=True), encoded_text)

                                if c_loss_option == 'all':
                                    total_clip_loss += (clip_loss + clip_local_loss)
                                elif c_loss_option == 'global':
                                    total_clip_loss += clip_loss
                                elif c_loss_option == 'local':
                                    total_clip_loss += clip_local_loss
                                else:
                                    raise AttributeError(
                                        f'CLIP loss option is wrong: {c_loss_option}')

                            total_clip_loss.backward()

                            optim.step()
                            lr_scheduler.step()

                        if i % 200 == 0:
                            with torch.no_grad():
                                CLIP_loss_seg = 0.0
                                GLIP_loss_MSE = 0.0
                                GLIP_loss_BCE = 0.0
                                GLIP_loss_Negative = 0.0
                                GLIP_box_score = 0.0

                                GLIP_loss_lst = []
                                GLIP_box_score_lst = []
                                cropped_prompt_lst = []
                                s_p_lst = style_prompt.split('. ')
                                for i_c in range(len(s_p_lst)):
                                    prompt_crop = s_p_lst[i_c]
                                    crop_token = clip.tokenize([prompt_crop]).to(device)
                                    cropped_prompt_clip = clip_model.encode_text(crop_token)
                                    cropped_prompt_lst.append(cropped_prompt_clip)

                                cropped_stylized_images = dict()
                                cropped_gray_images = dict()
                                for i_v in range(n_views):
                                    content_mesh = rendered_mesh_content[i_v]  # original mesh
                                    stylized_mesh = rendered_mesh_style[i_v]  # stylized mesh

                                    c_predictions, c_fused_visual_features, c_dot_product_logits, c_box_regression, c_box_scores, \
                                        c_candidate_indexes, c_candidate_high_indexes_dot, c_candidate_high_indexes_score, c_anchors, c_positive_map \
                                        = glip_demo.compute_prediction(content_mesh, content_prompt)
                                    c_top_predictions = glip_demo._post_process(c_predictions, 0.5)

                                    s_predictions, s_fused_visual_features, s_dot_product_logits, s_box_regression, s_box_scores, \
                                        s_candidate_indexes, s_candidate_high_indexes_dot, s_candidate_high_indexes_score, s_anchors, s_positive_map \
                                        = glip_demo.compute_prediction(stylized_mesh, style_prompt)
                                    s_top_predictions = glip_demo._post_process(s_predictions, 0.5)

                                    # loss & metric
                                    GLIP_box_score_lst.append(torch.mean(s_top_predictions.extra_fields['scores']))

                                    GLIP_loss_MSE -= get_classification_global_loss(c_candidate_high_indexes_score, s_box_scores, 'MSE')
                                    GLIP_loss_BCE -= get_classification_global_loss(c_candidate_high_indexes_score, s_box_scores, 'BCE')
                                    GLIP_loss_Negative -= get_classification_global_loss(c_candidate_high_indexes_score, s_box_scores, 'Negative')

                                    stylized_view = stylized_mesh
                                    gt_boxlist = c_top_predictions
                                    for b_idx in range(len(gt_boxlist.bbox)):

                                        x1, y1, x2, y2 = map(int, gt_boxlist.bbox[b_idx])

                                        # Color
                                        cropped_colored_segment = transforms.functional.crop(stylized_view, y1, x1, y2 - y1 + 1, x2 - x1 + 1)
                                        cropped_colored_segment = img_resizer(cropped_colored_segment)

                                        cls = int(gt_boxlist.extra_fields["labels"][b_idx])
                                        if cls not in cropped_stylized_images:
                                            cropped_stylized_images[cls] = [cropped_colored_segment]
                                        else:
                                            cropped_stylized_images[cls].append(cropped_colored_segment)

                                for key, value in cropped_stylized_images.items():
                                    cropped_stylized_images[key] = torch.stack(value, dim=0)

                                for key, value in cropped_stylized_images.items():
                                    colored_crops = value

                                    encoded_text = cropped_prompt_lst[key - 1]
                                    # global loss
                                    clip_image = augment_transform(colored_crops[:,0:3,:,:])
                                    rendered_clip = clip_model.encode_image(clip_image)
                                    CLIP_loss_seg -= torch.cosine_similarity(torch.mean(rendered_clip, dim=0, keepdim=True), encoded_text)

                                if GLIP_loss_MSE == 0.0 or GLIP_loss_BCE == 0.0 or GLIP_loss_Negative == 0.0:
                                    GLIP_loss_lst.append(GLIP_loss_MSE)
                                    GLIP_loss_lst.append(GLIP_loss_BCE)
                                    GLIP_loss_lst.append(GLIP_loss_Negative)
                                else:
                                    GLIP_loss_lst.append(GLIP_loss_MSE.item())
                                    GLIP_loss_lst.append(GLIP_loss_BCE.item())
                                    GLIP_loss_lst.append(GLIP_loss_Negative.item())
                                GLIP_box_score = sum(GLIP_box_score_lst) / len(GLIP_box_score_lst)

                                if CLIP_loss_seg == 0.0:
                                    metrics = [CLIP_loss_seg, GLIP_loss_lst, GLIP_box_score]
                                else:
                                    metrics = [CLIP_loss_seg.item(), GLIP_loss_lst, GLIP_box_score]
                                if loss == 0.0:
                                    losses = [loss, glip_loss_lst, glip_loss_regression_lst]
                                else:
                                    losses = [loss.item(), glip_loss_lst, glip_loss_regression_lst]
                                report_process(output_path, i, losses, metrics, rendered_mesh_style)

                    single_image, _, _, _, _, _ = model.render_single_image(scene=scene, azim=torch.tensor(frontview_center[0]), elev=torch.tensor(frontview_center[1]))
                    torchvision.utils.save_image(single_image, os.path.join(output_path, 'final.jpg'))

'''Tip of setting hyper-parameters 
if Metalic:
    select value of second comment in each argument
else:
    select default value
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default="input/test/")
    parser.add_argument('--output_dir', type=str, default="output/test/")
    parser.add_argument('--mesh_type', type=str, default="bag")
    parser.add_argument('--content_prompt', type=str, default="handle. body")
    parser.add_argument('--style_prompt', type=str, default="wicker. crocodile-leather")

    parser.add_argument('--seed', type=int, default=6601)
    parser.add_argument('--n_iter', type=int, default=1501)
    # GLIP input image size
    parser.add_argument('--img_size', type=int, default=320)

    parser.add_argument('--n_views', type=int, default=3)  # 3 / 1
    parser.add_argument('--n_normaugs', type=int, default=4)
    parser.add_argument('--frontview_std', type=int, default=4)  # 4 / 16
    parser.add_argument('--lr_decay', type=float, default=1)  # 1 / 0.7

    parser.add_argument('--background_color', type=str, default='white')  # 'black'
    parser.add_argument('--crop_ratio_min', type=float, default=0.1)  # 0.1 / 0.5
    parser.add_argument('--crop_ratio_max', type=float, default=0.1)  # 0.1 / 0.5

    parser.add_argument('--material_random_pe_numfreq', type=int, default=256)  # 256 / 3
    parser.add_argument('--material_random_pe_sigma', type=float, default=12)  # 12 / 0.5
    parser.add_argument('--material_nerf_pe_numfreq', type=int, default=0)

    parser.add_argument('--num_lgt_sgs', type=int, default=32)  # 32 / 64

    parser.add_argument('--max_delta_theta', type=float, default=1.5707)  # 1.5707 / 0, 0.32359
    parser.add_argument('--max_delta_phi', type=float, default=1.5707)  # 1.5707 / 0, 0.32359
    parser.add_argument('--normal_random_pe_numfreq', type=int, default=0)
    parser.add_argument('--normal_random_pe_sigma', type=float, default=0)
    parser.add_argument('--normal_nerf_pe_numfreq', type=int, default=0)
    parser.add_argument('--if_normal_clamp', default=False, action='store_true')

    # Renderer image size
    parser.add_argument('--width', type=int, default=331)  # 224 / 331 / 456 / 512

    parser.add_argument('--symmetry', default=False, action='store_true')  # False
    parser.add_argument('--init_r_and_s', default=False, action='store_true')    # False / True
    parser.add_argument('--init_roughness', type=float, default=0.7)  # 0.7 / 0.1
    parser.add_argument('--init_specular', type=float, default=0.23)  # 0.23 / 0.23
    parser.add_argument('--local_percentage', type=float, default=0.1)  # 0.7 / 0.3

    args = parser.parse_args()
    train(args)