# from roboflow import Roboflow
# rf = Roboflow(api_key="NRy2Ubht1Cbg4s9Zar35")
# project = rf.workspace("ntust-bgvey").project("mri_denoise_1")
# dataset = project.version(2).download("coco-segmentation")

from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tools import get_filenames
from tqdm import tqdm
tasks = ['train', 'valid', 'test']
for task in tasks:
    coco = COCO(f'mri_denoise_1-2/{task}/_annotations.coco.json')
    img_dir = f'mri_denoise_1-2/{task}'
    datas = sorted(get_filenames(f'mri_denoise_1-2/{task}', '*.jpg'))
    print(len(datas))

    for image_id in tqdm(range(len(datas))):
        img = coco.imgs[image_id]
        image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))

        # plt.imshow(image, interpolation='nearest')
        # plt.show()

        # plt.imshow(image)
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        # coco.showAnns(anns)
        if len(anns) > 0:
            mask = coco.annToMask(anns[0])
            for i in range(len(anns)):
                mask += coco.annToMask(anns[i])
        else:
            continue
        mask[mask > 0] = 255
        # plt.imshow(mask, interpolation='nearest')
        # plt.show()
        cv2.imwrite(f'mri_denoise_1-2/{task}/mask/' + img['file_name'], mask)