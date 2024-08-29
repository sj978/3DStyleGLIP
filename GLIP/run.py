import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import torch
from PIL import Image
import numpy as np

pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_img(img_dir):
    pil_image = Image.open(img_dir).convert("RGB")
    pil_image = pil_image.resize((800, 800))
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def img_save(img, caption, output):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig('{}.png'.format(output))


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}\n".format(device))


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
    min_image_size=400,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)

glip_demo.color = 0

# image = '102_view.png'
# image_dir = "inputs/chair/{}".format(image)
#
# image = load_img(image_dir)
# caption = 'backrest . legs . seat . chair . cylinder . chairboundingbox .'
# caption = 'chairback . chairlegs . seat .'

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
test_caption_dict = {'normal': 'monitor . keyboard . mouse .'}


os.mkdir('outputs/test')
for typ, caption in test_caption_dict.items():
    img_dir = 'inputs/'+typ
    file_lst = os.listdir(img_dir)
    for file in file_lst:
        if file[-3:] not in ['png', 'jpg']:
            continue
        img_path = img_dir + '/' + file
        image = load_img(img_path)
        result, pred = glip_demo.run_on_web_image(image, caption, 0.5)
        img_save(result, caption, 'outputs/test/{}'.format(file))
