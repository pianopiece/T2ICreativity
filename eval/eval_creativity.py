import argparse
import torch
import os
import sys
import json
from tqdm import tqdm

from PIL import Image
import math
from transformers import pipeline, set_seed
sys.path.append('./')
import numpy as np
import pandas as pd

from metrics.creativity import creativity

if __name__ == "__main__":
    set_seed(112)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--text-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--images-generated-folder", type=str, default="")
    parser.add_argument("--images-reference-folder", type=str, default="")
    parser.add_argument("--images-info", type=str, default="tables/question.jsonl")
    parser.add_argument("--save-path", type=str, default="data/results.jsonl")
    parser.add_argument("--conv-mode", type=str, default="mistral_direct")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--isapi", action='store_true')
    args = parser.parse_args()
    
    images_info = [json.loads(q) for q in open(os.path.expanduser(args.images_info), "r")]
    
    value, novelty, surprise = creativity(args.model_path, args.text_model, args.images_generated_folder,
                                args.images_reference_folder,
                                 images_info, model_base= args.model_base, conv_mode = args.conv_mode, 
                                 temperature = args.temperature, top_p = args.top_p, num_beams = args.num_beams,
                                 isapi = args.isapi)
    
    for idx in range(len(images_info)):
        images_info[idx] = {**images_info[idx], 
                            'value': value[idx],
                            'novelty': novelty[idx],  
                            'surprise': surprise[idx], 'lvlm': args.model_path, 
                            'text_model': args.text_model}
    # print(images_info)
    
    with open(args.save_path, 'w') as f:
        for d in images_info:
            f.write(json.dumps(d) + '\n')
            
    print(f"Value: {sum(value)/len(value)}.")
    print(f"Value std: {np.std([i for i in value])}.")
    print(f"Novelty: {sum(novelty)/len(novelty)}.")
    print(f"Novelty std: {np.std([i for i in novelty])}.")
    print(f"Surprise: {sum(surprise)/len(surprise)}.")
    print(f"Surprise std: {np.std([i for i in surprise])}.")
