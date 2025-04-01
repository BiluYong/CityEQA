import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))
from Utils.common_utils import *

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor


def get_args():
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str,
                        default="GroundSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        required=False, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, default="GroundSAM/groundingdino_swint_ogc.pth",
        required=False, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default="GroundSAM/sam_vit_h_4b8939.pth", required=False, help="path to sam checkpoint file"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False,
                        help="bert_base_uncased model path, default=False")
    args = parser.parse_args()

    return args


class GroundSAM:
    def __init__(self, args):
        # args = get_args()
        self.config_file = args.ground_config  # change the path of the model config file
        self.grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
        self.sam_version = args.sam_version
        self.sam_checkpoint = args.sam_checkpoint
        self.box_threshold = args.box_threshold
        self.text_threshold = args.text_threshold
        self.device = args.device
        self.bert_base_uncased_path = args.bert_base_uncased_path
        self.caption = self.landmark_to_caption(args.landmark_list)
        self.load_model()

    def landmark_to_caption(self, landmark_list):
        caption = ""
        for landmark in landmark_list:
            caption += landmark + "."
        return caption

    def load_model(self):
        args = SLConfig.fromfile(self.config_file)
        args.device = self.device
        args.bert_base_uncased_path = self.bert_base_uncased_path
        model = build_model(args)
        checkpoint = torch.load(self.grounded_checkpoint, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        self.Dino_model = model.to(self.device)
        self.SAM_model = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(self.device))

    def get_grounding_output(self, image, caption, with_logits=True):

        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.Dino_model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = self.Dino_model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def get_masks_output(self, image_rgb, boxes_filt):
        # image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        self.SAM_model.set_image(image_rgb)
        transformed_boxes = self.SAM_model.transform.apply_boxes_torch(boxes_filt, image_rgb.shape[:2]).to(self.device)

        masks, _, _ = self.SAM_model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )

        return masks

    def get_groundsam(self, image_cv2):
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)

        boxes_filt, pred_label = self.get_grounding_output(image, self.caption)
        log_info("Get Grounding Output")

        H, W = image_rgb.shape[0], image_rgb.shape[1]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()

        masks = self.get_masks_output(image_rgb, boxes_filt)
        log_info("Get Mask Output")
        return boxes_filt, pred_label, masks

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        ax.text(x0, y0, label)

    def save_result(self, image_cv2, boxes_filt, pred_label, masks, output_dir, step=0):
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.ioff() # 关闭交互模式
        plt.imshow(image_cv2)
        for mask in masks:
            self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_label):
            self.show_box(box.numpy(), plt.gca(), label)

        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, f"grounded_{step}.jpg"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        plt.close() # 关闭图形

        # masks_squeezed = masks.cpu().numpy()[:, 0, :, :]
        # s_name = os.path.join(output_dir, f'{step}_mAl.pkl')
        # with open(s_name, 'wb') as f:
        #     pickle.dump({'masks': masks_squeezed, 'labels': pred_label}, f)

        # value = 0  # 0 for background

        # mask_img = torch.zeros(masks.shape[-2:])
        # for idx, mask in enumerate(masks):
        #     mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        # plt.figure(figsize=(10, 10))
        # plt.imshow(mask_img.numpy())
        # plt.axis('off')
        # plt.savefig(os.path.join(output_dir, f'mask_{step}.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)


if __name__ == "__main__":

    os.chdir('..')
    image_path = "./GroundSAM/0_rgb.png"
    text_prompt = "building. road."
    output_dir = "./GroundSAM/outputs"
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_cv2 = cv2.imread(image_path)

    dinoSAM = GroundSAM()

    boxes_filt, pred_label, masks = dinoSAM.get_groundsam(image_cv2, text_prompt)

    dinoSAM.save_result(image_cv2, boxes_filt, pred_label, masks, output_dir)

    # 将 masks 和 labels 保存到同一个文件中
    masks_squeezed = np.squeeze(masks.cpu().numpy())
    with open('0_mAl.pkl', 'wb') as f:
        pickle.dump({'masks': masks_squeezed, 'labels': pred_label}, f)
