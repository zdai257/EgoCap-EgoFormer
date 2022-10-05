import os
from os.path import join
import numpy as np
import json
import pickle
from transformers import BertTokenizer
from PIL import Image, ImageOps
from configuration import Config, ConfigEgo
from Eval import predict_qualitative
from pycocoevalcap.bleu.bleu import Bleu, BleuScorer
from pycocoevalcap.meteor.meteor import Meteor, METEOR_JAR
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider, CiderScorer
from pycocoevalcap.spice.spice import Spice, SPICE_JAR


def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        #(Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        #(Spice(), "SPICE")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


# Specify COCO path
coco_dir = '/Users/zhuangzhuangdai/repos/EgoTransformer/images'
coco_filename = 'captions_val2017.json'

# Load annotations
with open(join(coco_dir, 'annotations', coco_filename), 'r') as f:
    annotations = json.load(f)
ana = annotations['annotations']

print("Loaded COCO labels...")
print(len(ana), ana[0])


def _process(image_id):
    val = str(image_id).zfill(12)
    return val + '.jpg'


def Loop_quantitative_eval(config, checkpoint, annotations, split_lst=['val2017'],
                           coco_dir_path='/Users/zhuangzhuangdai/repos/EgoTransformer/images'):
    hypo = {}
    refs = {}
    sample_path_lst = []

    # Loop through validation dir(s)
    for split in split_lst:
        for idx, cap in enumerate(annotations):
            if _process(cap['image_id']) in os.listdir(join(coco_dir_path, split)):
                sample_name = _process(cap['image_id'])
                sample_path = join(coco_dir_path, split, sample_name)

                # Add coco abs sample_path to list()
                sample_path_lst.append(sample_path)
                # Add coco captions (all 5)
                ref = {sample_name: cap['caption']}
                refs.update(ref)
    print("Done sorting refs, total testing samples = ", len(refs))

    # Inference with lists of sample_path
    pred_dict = predict_qualitative(config, sample_path_lst, tags=None, checkpoint_path=checkpoint)
    # Note: pred_dict is of dict: {'<image_name.jpg>': ["cap1.", "cap2." ...], ...}
    hypo.update(pred_dict)

    # Compute Metrics!
    metrics = calc_scores(refs, hypo)
    print(metrics)
    return hypo, refs, metrics


if __name__ == "__main__":
    config_t = Config()
    config_ego = ConfigEgo()

    # Model path
    EgoCO_base = '/Users/zhuangzhuangdai/repos/EgoTransformer/checkpoint_cl.pth'
    EgoCO_blind = '/Users/zhuangzhuangdai/repos/EgoTransformer/EgoCO/EgoCO_blind-best_epoch14_loss21.pth'
    EgoCO_raw = None

    tuples_base = Loop_quantitative_eval(config_t, EgoCO_base, ana)
