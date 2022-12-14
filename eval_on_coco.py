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
coco_dir = '/mnt/datasets/COCO/'  #'/Users/zhuangzhuangdai/repos/EgoTransformer/images'
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


def Loop_quantitative_eval(config, checkpoint, annotation, split_lst=['val2017'],
                           coco_dir_path='/Users/zhuangzhuangdai/repos/EgoTransformer/images'):
    hypo = {}
    refs = {}
    sample_path_lst = []

    # Loop through validation dir(s)
    for split in split_lst:
        for idx, cap in enumerate(annotation):
            if _process(cap['image_id']) in os.listdir(join(coco_dir_path, split)):
                sample_name = _process(cap['image_id'])
                sample_path = join(coco_dir_path, split, sample_name)

                # Add coco abs sample_path to list()
                sample_path_lst.append(sample_path)

                # Add coco captions (all 5)
                ref_lst = []
                for item in annotation:
                    if item['image_id'] == cap['image_id']:
                        ref_lst.append(item['caption'])
                ref = {sample_name: ref_lst}
                refs.update(ref)

    print("Before set() redundant removal, testing samples = ", len(sample_path_lst))
    # Remove redundant images
    sample_path_lst = list(set(sample_path_lst))
    print("Done sorting samples, total testing samples = ", len(sample_path_lst))

    # Inference with lists of sample_path
    pred_dict = predict_qualitative(config, sample_path_lst, tags=None, checkpoint_path=checkpoint)
    # Note: pred_dict is of dict: {'<image_name.jpg>': ["cap1.", "cap2." ...], ...}
    hypo.update(pred_dict)

    # Remove illegal samples based on {hypo}
    refs = {key: refs[key] for key in list(hypo.keys())}

    # Compute Metrics!
    metrics = calc_scores(refs, hypo)
    print(metrics)
    return hypo, refs, metrics


if __name__ == "__main__":
    config_t = Config()
    config_ego = ConfigEgo()

    # Model path
    EgoCO_base = '/home/zdai/repos/EgoCap-EgoFormer/checkpoint_cl.pth'  #'/Users/zhuangzhuangdai/repos/EgoTransformer/checkpoint_cl.pth'
    EgoCO_blind = '/mnt/datasets/COCO/epoch_checks/EgoCO_blind-best_epoch14_loss21.pth'  #'/Users/zhuangzhuangdai/repos/EgoTransformer/EgoCO/EgoCO_blind-best_epoch14_loss21.pth'
    EgoCO_raw = None

    tuples_base = Loop_quantitative_eval(config_t, EgoCO_base, ana, coco_dir_path=coco_dir)
    print(tuples_base[2])

    dict_results = {
        'metrics': tuples_base[2],
        'hypos': tuples_base[0],
        'refs': tuples_base[1]
    }

    with open(join('EgoCO_base' + '-COCOeval.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_results, f, ensure_ascii=False, indent=4)
