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


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
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


def Loop_quantitative_eval(config, checkpoint, annotations, split_lst,
                           egocap_dir_path='/Users/zhuangzhuangdai/repos/EgoCapSurvey'):
    hypo = {}
    refs = {}
    sample_path_lst, tags_lst = [], []

    for split in split_lst:
        for idx, (key, val) in enumerate(annotations.items()):
            if int(split) == int(val['SplitIndex']):
                sample_name = key
                sample_path = join(egocap_dir_path, 'static', 'Split' + split.zfill(2), sample_name)

                if config.modality == 'ego':
                    tags = (val['tag_stats']['where']['majority'], val['tag_stats']['when']['majority'])
                else:
                    tags = None

                sample_path_lst.append(sample_path)
                tags_lst.append(tags)

                ref = {sample_name: val['captions']}
                refs.update(ref)

    # Inference with lists of sample_path & tags
    pred_dict = predict_qualitative(config, sample_path_lst, tags_lst, checkpoint_path=checkpoint)
    hypo.update(pred_dict)

    #metrics = calc_scores(refs, hypo)
    #print(metrics)
    return hypo, refs


def quantitative_eval(config, checkpoint, annotations, split_lst,
                      egocap_dir_path='/Users/zhuangzhuangdai/repos/EgoCapSurvey'):
    hypo = {}
    refs = {}
    for split in split_lst:
        for idx, (key, val) in enumerate(annotations.items()):
            if int(split) == int(val['SplitIndex']):
                sample_name = key
                sample_path = join(egocap_dir_path, 'static', 'Split' + split.zfill(2), sample_name)

                if config.modality == 'ego':
                    tags = (val['tag_stats']['where']['majority'], val['tag_stats']['when']['majority'])
                else:
                    tags = None

                ref = {sample_name: val['captions']}
                refs.update(ref)
                # Inference
                pred_dict = predict_qualitative(config, sample_path, tags=tags, checkpoint_path=checkpoint)
                hypo.update(pred_dict)

    #metrics = calc_scores(refs, hypo)
    #print(metrics)
    return hypo, refs


if __name__ == "__main__":
    egocap_dir = '/Users/zhuangzhuangdai/repos/EgoCapSurvey'
    egocap_filename = 'EgoCap_annatations_ref.json'

    with open(join(egocap_dir, 'doc', egocap_filename), 'r') as f:
        ana = json.load(f)

    eval_split = ['03', '10', '17']

    # Model configs
    config_t = Config()
    config_ego = ConfigEgo()

    baseline = 'EgoFormer/Baseline2-best_epoch32_loss10.pth'  # '13 - finetune-epoch29_loss15.pth'
    egotrans = 'EgoFormer/EgoFormer3-equalloss-best_epoch33_loss10.pth'
    small = 'EgoFormer/EgoFormer-small-best_epoch22_loss13.pth'
    tiny = 'EgoFormer/EgoFormer-tiny-best_epoch25_loss13.pth'

    trans_concat = '../EgoTransformer/EgoFormer/EgoFormerConcat-smallLR-best_epoch19_loss10.pth'
    trans_gatedinfo = 'EgoFormer/GatedViT_smallLR-best_epoch19_loss10.pth'

    egotrans_prefuse = 'EgoFormer/EgoFormer-PreFuse3-best_epoch33_loss10.pth'
    egotrans_contextembed = '16 - contextembed -finetuneEgoTrans-best_epoch50_loss25.pth'
    egotrans_contextfuse = 'finetuneEgoTrans-best_epoch17_loss10.pth'
    egotrans_encodernograd = 'EgoFormer/EgoFormer3-backboneViTnograd-best_epoch32_loss10.pth'
    egotrans_backbonegrad = 'EgoFormer/EgoFormer3-backboneViTwithgrad-best_epoch30_loss10.pth'
    egotrans_blind = 'EgoFormer/EgoFormer-BlindContext-best_epoch21_loss10.pth'

    # testing: Specify MODEL
    tuples0 = Loop_quantitative_eval(config_ego, small, ana, eval_split)

    dict_results = {
        #'metrics': tuples0[2],
        'hypos': tuples0[0],
        'refs': tuples0[1]
    }

    # Specify NAME.json
    with open(join('EgoCap_' + 'small' + '-eval.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_results, f, ensure_ascii=False, indent=4)
