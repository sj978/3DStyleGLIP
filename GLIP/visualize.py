import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import torch
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd

pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_img(img_dir):
    pil_image = Image.open(img_dir)
    pil_image = pil_image.resize((320, 320))
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def img_save(img, caption, output):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig('{}.png'.format(output))


device = "cuda" if torch.cuda.is_available() else "cpu"

# GLPT-L model setup
config_file = "configs/pretrain/glip_Swin_L.yaml"
weight_file = "MODEL/glip_large_model.pth"

# update the config options with the config file
# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", device])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=320,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)


'''
case1: 일반적 image / 올바른 caption
case2: 일반적 image / 틀린 caption
case3: c1 50%, c2 50%
case4: 단일 회색 mesh image / 올바른 caption (part)
case5: 단일 회색 mesh image / 틀린 caption (part)
case6: c4 50%, c5 50%
case7: 부위별 컬러 mesh image / 올바른 caption (part)
case8: 부위별 컬러 mesh image / 틀린 caption (part)
case9: c4 50%, c5 50%

일단 1~6 까지 각각 3~5 case 씩 실험
'''

good_caption_dict = {'normal': 'monitor . keyboard . mouse .', 'animal': 'head . leg . neck . body .', 'chair': 'seat . leg . chairback .',
                     'cup': 'handle . body .'}
bad_caption_dict = {'normal': 'dog . monkey . teapot .', 'animal': 'dog . monkey . teapot . chocolate',
                    'chair': 'dog . monkey . teapot .', 'cup': 'dog . monkey .'}
half_caption_dict = {'normal': 'monitor . keyboard . monkey .', 'animal': 'head . leg . teapot . monkey',
                     'chair': 'seat . leg . monkey .', 'cup': 'handle . monkey .'}
test_caption_dict = {'cup': 'monkey. cup'}


caption_lst = [good_caption_dict, bad_caption_dict, half_caption_dict, test_caption_dict]

condition = 3

for input_type, caption in caption_lst[condition].items():
    input_dir = 'inputs/' + input_type + '_view'
    file_lst = os.listdir(input_dir)
    output_path = 'outputs/{}'.format(condition)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    for file in file_lst:
        predict_dict = {}
        tpredict_dict = {}

        score_lst = []
        label_lst = []
        tscore_lst = []
        tlabel_lst = []

        if file[-3:] not in ['png', 'jpg']:
            continue
        img_path = input_dir + '/' + file
        image = load_img(img_path)
        result, top_predictions, predictions, fused_features, dot_product_logits = glip_demo.run_on_web_image(image, caption, 0.5)
        label_lst.append(predictions.extra_fields['labels'].tolist())
        score_lst.append(predictions.extra_fields['scores'].tolist())

        tlabel_lst.append(top_predictions.extra_fields['labels'].tolist())
        tscore_lst.append(top_predictions.extra_fields['scores'].tolist())

        for i in range(len(label_lst[0])):
            if label_lst[0][i] not in predict_dict:
                predict_dict[label_lst[0][i]] = [score_lst[0][i]]
            else:
                predict_dict[label_lst[0][i]].append(score_lst[0][i])

        for i in range(len(tlabel_lst[0])):
            if tlabel_lst[0][i] not in tpredict_dict:
                tpredict_dict[tlabel_lst[0][i]] = [tscore_lst[0][i]]
            else:
                tpredict_dict[tlabel_lst[0][i]].append(tscore_lst[0][i])

        img_save(result, caption, output_path + '/{}'.format(file))

        for label, score in predict_dict.items():
            predict_dict[label] = sorted(score, reverse=True)
        # print('------------------')
        # print(img_path)
        # print(caption)
        # print(len(label_lst[0]))
        # print(predict_dict)
        # print(len(tlabel_lst[0]))
        # print(tpredict_dict)

