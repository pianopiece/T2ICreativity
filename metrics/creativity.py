import argparse
import torch
import os
import json
import time
from tqdm import tqdm
import base64

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.metrics import davies_bouldin_score
from PIL import Image
import math
import re
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers.util import pytorch_cos_sim
from transformers import AutoImageProcessor, AutoModel

import t2v_metrics

import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import logging
 
logging.basicConfig(level=logging.INFO, filemode='w', filename="./middle.log")

def similarity(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dino = AutoModel.from_pretrained('models/models_visual/dinov2-large').to(device)
    processor_dino = AutoImageProcessor.from_pretrained('models/models_visual/dinov2-large')
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
    processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    value = []
    novelty = []
    surprise = []
    score_llava15_7b = t2v_metrics.VQAScore(model='llava-v1.5-7b', 
                                             cache_dir = 'models/models_lvlm/')

    for idx, line in enumerate(tqdm(args.images_info)): 
        # preparation for similarity computation
        images_name_A = os.listdir(os.path.join(args.images_generated_folder, str(line['id'])))
        images_name_A = [os.path.join(args.images_generated_folder, str(line['id']), image_name) for image_name in images_name_A if image_name.endswith('.png') or image_name.endswith('.jpg')]
        logging.info(str(images_name_A))
        images_name_B = line['reference']
        images_name_B = [os.path.join(args.images_reference_folder, image_name) for image_name in images_name_B if image_name.endswith('.png') or image_name.endswith('.jpg') or image_name.endswith('.jpeg')]
        # images_name_B = os.listdir(os.path.join(args.images_reference_folder, str(line['id'])))
        # images_name_B = [os.path.join(args.images_reference_folder, str(line['id']), image_name) for image_name in images_name_B if image_name.endswith('.png') or image_name.endswith('.jpg')]
        images_A, images_B = [Image.open(image_name) for image_name in images_name_A], [Image.open(image_name) for image_name in images_name_B]
        texts = [line['prompt']] if 'promptref' not in line else [line['promptref']]
        images_A_dino, images_B_dino = [], []
        images_A_clip, images_B_clip = [], []
        
        prompt = args.images_info[idx]['prompt'] if 'promptref' not in line else [line['promptref']]
        inputs_prompt = processor_clip(text=prompt, return_tensors="pt", padding=True).to(device)
        embedding_prompt = model_clip.get_text_features(**inputs_prompt).cpu()
        
        #DINO
        inputs_image = processor_dino(images=images_A, return_tensors="pt").to(device)
        outputs_image = model_dino(**inputs_image)
        last_hidden_states = outputs_image.last_hidden_state
        image_tensor_dino = last_hidden_states[:,0,:].cpu()
        # image_tensor_dino = last_hidden_states.mean(1).cpu()
        images_A_dino = list(image_tensor_dino)
        inputs_image_A = processor_clip(images=images_A, return_tensors="pt", padding=True).to(device)
        image_tensor_clip = model_clip.get_image_features(**inputs_image_A).cpu()
        images_A_clip = list(image_tensor_clip)
        
        inputs_image = processor_dino(images=images_B, return_tensors="pt").to(device)
        outputs_image = model_dino(**inputs_image)
        last_hidden_states = outputs_image.last_hidden_state
        image_tensor_dino = last_hidden_states[:,0,:].cpu()
        # image_tensor_dino = last_hidden_states.mean(1).cpu()
        images_B_dino = list(image_tensor_dino)
        inputs_image_B = processor_clip(images=images_B, return_tensors="pt", padding=True).to(device)
        image_tensor_clip = model_clip.get_image_features(**inputs_image_B).cpu()
        images_B_clip = list(image_tensor_clip)
        
        # value
        logging.info('value')
        # scores_value = score_llava15_7b(images=images_name_A, texts=texts,
        #                                question_template='Does the image contain the contents of this sentence, "{}"? Please answer yes or no.',
        #                                answer_template='Yes')
        scores_value = score_llava15_7b(images=images_name_A, texts=texts,
                                       question_template='Does this figure show "{}"? Please answer yes or no.',
                                       answer_template='Yes')
        logging.info(scores_value)
        value.append(float(scores_value.mean(0).mean(0)))
        
        # novelty
        scores_novelty_sim = []
        scores_novelty_clip = []
        for idx_image_A in range(len(images_A)):
            clip_A = images_A_clip[idx_image_A]
            clip_temp = float(1 - pytorch_cos_sim(clip_A,embedding_prompt))
            scores_novelty_clip.append(clip_temp)
            for idx_image_A_ in range(idx_image_A + 1, len(images_A)):
                dino_A = images_A_dino[idx_image_A]
                dino_A_ = images_A_dino[idx_image_A_]
                score_novelty = float(pytorch_cos_sim(dino_A,dino_A_))
                scores_novelty_sim.append(score_novelty)
            
        clip_avg = sum(scores_novelty_clip)/len(scores_novelty_clip)
        novelty.append(1 - sum(scores_novelty_sim)/len(scores_novelty_sim) * clip_avg)
        
        # surprise
        scores_surprise_sim = []
        scores_surprise_clip = []
        for idx_image_A in range(len(images_A)):
            clip_A = images_A_clip[idx_image_A]
            clip_temp = float(1 - pytorch_cos_sim(clip_A,embedding_prompt))
            scores_surprise_clip.append(clip_temp)
            scores_surprise_temp = []
            for idx_image_B in range(len(images_B)):
                dino_A = images_A_dino[idx_image_A]
                dino_B = images_B_dino[idx_image_B]
                score_surprise = float(pytorch_cos_sim(dino_A,dino_B))
                scores_surprise_temp.append(score_surprise)
            scores_surprise_sim.append(max(scores_surprise_temp))
            
        for idx_image_B in range(len(images_B)):
            clip_B = images_B_clip[idx_image_B]
            clip_temp = float(1 - pytorch_cos_sim(clip_B,embedding_prompt))
            scores_surprise_clip.append(clip_temp)

        clip_avg = sum(scores_surprise_clip)/len(scores_surprise_clip)
        surprise.append(1 - sum(scores_surprise_sim)/len(scores_surprise_sim) * clip_avg)
        
    return value, novelty, surprise

class Config:
    def __init__(self, model_path, text_model, images_generated_folder, images_reference_folder, images_info, model_base,
                      conv_mode, temperature, top_p, num_beams, isapi):
        self.model_path = model_path
        self.text_model = text_model
        self.images_generated_folder = images_generated_folder
        self.images_reference_folder = images_reference_folder
        self.images_info = images_info
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.isapi = isapi
        

def creativity(model_path, text_model, images_generated_folder, images_reference_folder, images_info, model_base=None, 
            conv_mode='mistral_direct', temperature=0, top_p=1, num_beams=1, isapi=False):
    
    args = Config(model_path, text_model, images_generated_folder, images_reference_folder, images_info,
                      model_base, conv_mode, temperature, top_p, num_beams, isapi)

    with torch.no_grad():
        value, novelty, surprise = similarity(args)
    
    return value, novelty, surprise
