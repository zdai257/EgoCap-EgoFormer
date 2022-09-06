import torch
import os
from os.path import join
import time
from transformers import BertTokenizer
from PIL import Image, ImageOps
import argparse
import matplotlib.pyplot as plt
from models import caption
from datasets import coco, utils
from configuration import *
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu, BleuScorer
from pycocoevalcap.meteor.meteor import Meteor, METEOR_JAR
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider, CiderScorer
from pycocoevalcap.spice.spice import Spice, SPICE_JAR
from transformers import ViTFeatureExtractor


def create_caption_and_mask(start_t, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_t
    mask_template[:, 0] = False

    return caption_template, mask_template


def create_tag_token_and_mask(tokenizer_t, tags=('na', 'na')):
    where_dict = {'indoor': "in indoor inside room", 'outdoor': "out outside outdoor outdoors", 'na': ""}
    when_dict = {'daytime': "day daytime sunny midday", 'night': "night nighttime midnight evening", 'na': ""}

    tags_encoded = tokenizer_t.encode_plus(where_dict[tags[0]] + ' ' + when_dict[tags[1]],
                                           max_length=10, pad_to_max_length=True, return_attention_mask=True,
                                           return_token_type_ids=False, truncation=True)
    tag_token_template = torch.from_numpy(np.array(tags_encoded['input_ids']))
    tag_mask_template = torch.from_numpy((1 - np.array(tags_encoded['attention_mask'])).astype(bool))
    tag_mask_template[0] = True
    tag_mask_template[-1] = True

    return tag_token_template.unsqueeze(0), tag_mask_template.unsqueeze(0)


'''
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
'''


def predict_qualitative(config, sample_path, tags, checkpoint_path=None, map_location='cpu'):

    if checkpoint_path is None:
        model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    elif os.path.exists(checkpoint_path):
        ### Select Model ###
        if config.modality == 'image':
            # Original CATR
            model, criterion = caption.build_model(config)
        elif config.modality == 'ego':
            # Ego Model
            model, criterion = caption.build_model_egovit(config)
        elif config.modality == 'video':
            # Video Model
            model, criterion = caption.build_model_bs(config)
        print("Loading Checkpoint...")
        checkpoint_tmp = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint_tmp['model'])
        print("Current checkpoint epoch = %d" % checkpoint_tmp['epoch'])

    else:
        raise NotImplementedError('Give valid checkpoint path')

    device = torch.device(map_location)
    print(f'Initializing Device: {device}')

    start_t_tokenizer = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True, local_files_only=False)
    print("Loading pretrained Tokenizer takes: %.2fs" % (time.time() - start_t_tokenizer))

    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    print("Total Vocal = ", tokenizer.vocab_size)
    print("Start Token: {}; End Token: {}; Padding: {}".format(tokenizer._cls_token, tokenizer._sep_token,
                                                               tokenizer._pad_token))

    # Load ViT backbone transform
    #root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "datasets"
    if os.path.exists(join(root_dir, "vit_classify-feature_extractor")):
        feature_extractor = ViTFeatureExtractor.from_pretrained(join(root_dir, "vit_classify-feature_extractor"))
    else:
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        feature_extractor.save_pretrained(join(root_dir, "vit_classify-feature_extractor"))

    @torch.no_grad()
    def evaluate(sample_t, cap_t, cap_mask_t, img):
        model.eval()
        decoded_batch_beams = None

        for i in range(config.max_position_embeddings - 1):
            if config.modality == 'ego':
                predictions = model(sample_t, cap_t, cap_mask_t, img)
            else:
                predictions = model(sample_t, cap_t, cap_mask_t)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            if predicted_id[0] == 102:
                break

            cap_t[:, i + 1] = predicted_id[0]
            cap_mask_t[:, i + 1] = False
        out = cap_t
        '''
        ### Greedy ###
        #out, decoded_batch_beams = model.decode(sample, cap, cap_mask, beam_width=None, diverse_m=3)
        ### Beam Search ###
        out, decoded_batch_beams = model.decode(sample, cap, cap_mask, beam_width=5, diverse_m=3)
        '''
        return out, decoded_batch_beams

    if isinstance(sample_path, str):
        # Load Image
        image = Image.open(sample_path)
        # Transpose with respect to EXIF data
        image = ImageOps.exif_transpose(image)
        w, h = image.size
        print("PIL Image width: {}, height: {}".format(w, h))
        sample = coco.val_transform(image)
        sample = sample.unsqueeze(0)

        # Load skeleton caption
        cap, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

        # Context ViT input
        inputs = feature_extractor(image, return_tensors="pt")
        img_tensor = inputs['pixel_values'].squeeze(1).to(device)

        output, outputs = evaluate(sample, cap, cap_mask, img_tensor)

        result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        print('\n' + result.capitalize() + '\n')

        cap_dict = {sample_path.split('/')[-1]: [result]}

    elif isinstance(sample_path, list):
        cap_dict = {}

        for idx, s_path in enumerate(sample_path):
            # Load Image
            image = Image.open(s_path)
            # Transpose with respect to EXIF data
            image = ImageOps.exif_transpose(image)
            w, h = image.size
            print("PIL Image width: {}, height: {}".format(w, h))
            sample = coco.val_transform(image)
            sample = sample.unsqueeze(0)

            # Load skeleton caption
            cap, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

            # Context ViT input
            inputs = feature_extractor(image, return_tensors="pt")
            img_tensor = inputs['pixel_values'].squeeze(1).to(device)

            output, outputs = evaluate(sample, cap, cap_mask, img_tensor)

            result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
            print('\n' + result.capitalize() + '\n')
            sample_dict = {s_path.split('/')[-1]: [result]}
            cap_dict.update(sample_dict)

    else:
        raise TypeError("Sample_path invalid!")

    return cap_dict


