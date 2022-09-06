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


parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image', required=True)
parser.add_argument('--v', type=str, help='version', default='v0')  # default not using pretrained
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
args = parser.parse_args()
image_path = args.path
version = args.v
checkpoint_path = args.checkpoint

config = ConfigEgo()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
    print("Checking for checkpoint.")
    if checkpoint_path is None:
      raise NotImplementedError('No model to chose from!')
    else:
      if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
      print("Found checkpoint! Loading!")

      ### Select Model ###
      if config.modality == 'image':
          # Original CATR
          model, criterion = caption.build_model(config)
      elif config.modality == 'ego':
          # Ego Model
          model, criterion = caption.build_model_ego(config)
      elif config.modality == 'video':
          # Video Model
          model, criterion = caption.build_model_bs(config)

      print("Loading Checkpoint...")
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
      model.load_state_dict(checkpoint['model'])
      print("Current checkpoint epoch = %d" % checkpoint['epoch'])


device = torch.device(config.device)
print(f'Initializing Device: {device}')

load_tokenizer_from_local = False

start_t_tokenizer = time.time()
if load_tokenizer_from_local:
    tokennizer_local_path = '/home/zdai/repos/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(tokennizer_local_path)
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True, local_files_only=False)
print("Loading pretrained Tokenizer takes: %.2fs" % (time.time() - start_t_tokenizer))

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
print("Total Vocal = ", tokenizer.vocab_size)
print("Start Token: {}; End Token: {}; Padding: {}".format(tokenizer._cls_token, tokenizer._sep_token,
                                                           tokenizer._pad_token))

image = Image.open(image_path)
# Transpose with respect to EXIF data
image = ImageOps.exif_transpose(image)
w, h = image.size
print("PIL Image width: {}, height: {}".format(w, h))
sample = coco.val_transform(image)
sample = sample.unsqueeze(0)


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

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


cap, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)

# Get tag_token, tag_mask
def create_tag_token_and_mask():
    tags = ('indoor', 'daytime')
    where_dict = {'indoor': "in indoor inside room", 'outdoor': "out outside outdoor outdoors", 'na': ""}
    when_dict = {'daytime': "day daytime sunny midday", 'night': "night nighttime midnight evening", 'na': ""}

    tags_encoded = tokenizer.encode_plus(where_dict[tags[0]] + ' ' + when_dict[tags[1]],
                                         max_length=10, pad_to_max_length=True, return_attention_mask=True,
                                         return_token_type_ids=False, truncation=True)
    tag_token_template = torch.from_numpy(np.array(tags_encoded['input_ids']))
    tag_mask_template = torch.from_numpy((1 - np.array(tags_encoded['attention_mask'])).astype(bool))
    tag_mask_template[0] = True
    tag_mask_template[-1] = True

    return tag_token_template.unsqueeze(0), tag_mask_template.unsqueeze(0)


tag_token, tag_mask = create_tag_token_and_mask()


@torch.no_grad()
def evaluate():
    model.eval()
    decoded_batch_beams = None


    for i in range(config.max_position_embeddings - 1):
        if config.modality == 'ego':
            predictions = model(sample, cap, cap_mask, tag_token, tag_mask)
        else:
            predictions = model(sample, cap, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            break

        cap[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
    out = cap
    '''
    ### Greedy ###
    #out, decoded_batch_beams = model.decode(sample, cap, cap_mask, beam_width=None, diverse_m=3)
    ### Beam Search ###
    out, decoded_batch_beams = model.decode(sample, cap, cap_mask, beam_width=5, diverse_m=3)
    '''
    return out, decoded_batch_beams

output, outputs = evaluate()

result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
#result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result.capitalize())
if outputs is not None:
    for idx, outs in enumerate(outputs):
        print("\nDiverse-M %d:" % idx)
        result_topk = []
        for i in range(outs.shape[1]):
            result_topk.append(tokenizer.decode(outs[0, i], skip_special_tokens=True))
            print("Top {}: \n{}".format(i, result_topk[-1].capitalize()))

cap_dict = {image_path.split('/')[-1]: [result]}
print_metrics = True

try:
    print_metrics = True
    data_dir = os.path.join('Data', 'amt_data')
    data_file = os.path.join(data_dir, 'amt_list.txt')
    _, anns = coco.read_deepdiary(data_dir, data_file)
    gts = {next(iter(cap_dict)): anns[next(iter(cap_dict))]}

    metrics = calc_scores(gts, cap_dict)
    print('\n', metrics)
except FileNotFoundError:
    print_metrics = False
    print("\nNo ground-truth caption found for this image.")

# Visualization
fig = plt.figure(1, figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)

if print_metrics:

    ax.set_xlabel("GT1: " + anns[next(iter(cap_dict))][0] +
                  '\n' + "CATR: " + result.capitalize() + '\n' + "BLEU-4: {}".format(metrics['Bleu_4'])
                  + '\n' + "METEOR: {}".format(metrics['METEOR'])
                  + '\n' + "ROUGE-L: {}".format(metrics['ROUGE_L'])
                  + '\n' + "CIDEr: {}".format(metrics['CIDEr'])
                  + '\n' + "SPICE: {}".format(metrics['SPICE']),
                  fontsize=18, fontdict=dict(weight='bold'))

    ax.set_xlabel("Ego-Caption: " + anns[next(iter(cap_dict))][1], fontsize=18, fontdict=dict(weight='bold'))
else:
    ax.set_xlabel("CATR: " + result.capitalize(), fontsize=18, fontdict=dict(weight='bold'))

plt.imshow(image)
fig.savefig(join('images', 'deepdiary', image_path.split("/")[-1]), facecolor='w', bbox_inches='tight')
plt.show()
