import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor, ViTModel

from PIL import Image
import numpy as np
import random
import os
from os.path import join
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import pickle
from .utils import nested_tensor_from_tensor_list, read_json

MAX_DIM = 299


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_transform_msvd = tv.transforms.Compose([
    RandomRotation(angles=[0]),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def read_deepdiary(dirname, filename):
    with open(filename, "r") as file:
        pairs_str = file.read()
        pairs = pairs_str.split('\n')[1:]

    anns = []
    for pair in pairs:
        img_name = pair.split(' ')[0]
        sent_ana = pair.split('.jpg ')[-1]
        if img_name in os.listdir(dirname):
            anns.append((img_name, sent_ana))

    img_names = {}

    for pair in pairs:
        img_name = pair.split(' ')[0]
        sent_ana = pair.split('.jpg ')[-1]

        if img_name not in img_names:
            img_names[img_name] = [sent_ana]
        else:
            img_names[img_name].append(sent_ana)

    return anns, img_names


class DeepDiary(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=val_transform, mode='deepdiary'):
        super().__init__()

        self.root = root
        self.transform = transform

        if mode == 'deepdiary':
            # self.annot is a list of tuple of ('000000XXXXXX.jpg', "A man is sitting")
            self.annot = ann
        else:
            raise ValueError("DeepDiary does not support this mode.")

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True, local_files_only=True)
        self.max_length = max_length + 1

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))
        if self.transform:
            image = self.transform(image)

        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)
        # caption_encoded is a dict of {'input_ids': <a list of vocab indexes>, 'attention_mask': [1, 1, 1, 0 ,0 ...]
        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


class EgoCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()
        self.root = root
        self.transform = transform
        # self.annot is a list of tuples [('imgname.jpg', split_index, 'I am doing.', (<where>, <when>))...]
        if mode == 'validation':
            self.annot = ann
        if mode == 'training':
            self.annot = ann

        self.where_dict = {'indoor': "in indoor inside room", 'outdoor': "out outside outdoor outdoors", 'na': ""}
        self.when_dict = {'daytime': "day daytime sunny midday", 'night': "night nighttime midnight evening", 'na': ""}

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True, local_files_only=True)
        self.max_length = max_length + 1

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, split_index, caption, tags = self.annot[idx]
        image = Image.open(os.path.join(self.root, 'static', 'Split' + split_index, image_id))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)
        # caption_encoded is a dict of {'input_ids': <a list of vocab indexes>, 'attention_mask': [1, 1, 1, 0 ,0 ...]
        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        # Tags: popping in Decoder
        tags_encoded = self.tokenizer.encode_plus(self.where_dict[tags[0]] + ' ' + self.when_dict[tags[1]],
                                                  max_length=10, pad_to_max_length=True, return_attention_mask=True,
                                                  return_token_type_ids=False, truncation=True)
        tag_token = np.array(tags_encoded['input_ids'])
        tag_mask = (1 - np.array(tags_encoded['attention_mask'])).astype(bool)
        # Disable attention to [CLS] or [SEP]
        tag_mask[0] = True
        tag_mask[-1] = True

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask, tag_token, tag_mask


class EgoCapViT(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()
        self.root = root
        self.transform = transform
        root_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(join(root_dir, "vit_classify-feature_extractor")):
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(join(root_dir, "vit_classify-feature_extractor"))
        else:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            self.feature_extractor.save_pretrained(join(root_dir, "vit_classify-feature_extractor"))
        # self.annot is a list of tuples [('imgname.jpg', split_index, 'I am doing.', (<where>, <when>))...]
        if mode == 'validation':
            self.annot = ann
        if mode == 'training':
            self.annot = ann
        '''
        self.where_dict = {'indoor': torch.tensor([1, 0, 0]), 'outdoor': torch.tensor([0, 1, 0]), 'na': torch.tensor([0, 0, 1])}
        self.when_dict = {'daytime': torch.tensor([1, 0, 0]), 'night': torch.tensor([0, 1, 0]), 'na':  torch.tensor([0, 0, 1])}
        self.whom_dict = {'human':  torch.tensor([1, 0, 0]), 'object':  torch.tensor([0, 1, 0]), 'na':  torch.tensor([0, 0, 1])}
        '''
        self.where_dict = {'indoor': 0, 'outdoor': 1, 'na': 2}
        self.when_dict = {'daytime': 0, 'night': 1, 'na': 2}
        self.whom_dict = {'human': 0, 'object': 1, 'na': 2}

        self.where_dict_syn = {'indoor': "in indoor inside room", 'outdoor': "out outside outdoor outdoors", 'na': ""}
        self.when_dict_syn = {'daytime': "day daytime sunny midday", 'night': "night nighttime midnight evening", 'na': ""}

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True, local_files_only=False)
        self.max_length = max_length + 1

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, split_index, caption, tags = self.annot[idx]
        image = Image.open(os.path.join(self.root, 'static', 'Split' + split_index, image_id))

        # Context ViT input
        inputs = self.feature_extractor(image, return_tensors="pt")
        # Visual ViT input
        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)
        # caption_encoded is a dict of {'input_ids': <a list of vocab indexes>, 'attention_mask': [1, 1, 1, 0 ,0 ...]
        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        # Tags: popping in Decoder
        tags_encoded = self.tokenizer.encode_plus(self.where_dict_syn[tags[0]] + ' ' + self.when_dict_syn[tags[1]],
                                                  max_length=10, pad_to_max_length=True, return_attention_mask=True,
                                                  return_token_type_ids=False, truncation=True)
        tag_token = np.array(tags_encoded['input_ids'])
        tag_mask = (1 - np.array(tags_encoded['attention_mask'])).astype(bool)
        # Disable attention to [CLS] or [SEP]
        tag_mask[0] = True
        tag_mask[-1] = True

        return (image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask, tag_token, tag_mask,
                inputs, {'where': torch.tensor(self.where_dict[tags[0]]),
                         'when': torch.tensor(self.when_dict[tags[1]]),
                         'whom': torch.tensor(self.whom_dict[tags[2]])})


class EgoCO(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()
        self.root = root
        self.transform = transform
        root_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(join(root_dir, "vit_classify-feature_extractor")):
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(join(root_dir, "vit_classify-feature_extractor"))
        else:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            self.feature_extractor.save_pretrained(join(root_dir, "vit_classify-feature_extractor"))

        self.annot = [(self._process(val['image_id']), val['caption'])
                      for val in ann['annotations']]
        # self.annot is a list of tuples [('imgname.jpg', split_index, 'I am doing.', (<where>, <when>))...]
        if mode == 'validation':
            self.annot = ann
        if mode == 'training':
            self.annot = ann[: limit]

        self.where_dict = {'indoor': "in indoor inside room", 'outdoor': "out outside outdoor outdoors", 'na': ""}
        self.when_dict = {'daytime': "day daytime sunny midday", 'night': "night nighttime midnight evening", 'na': ""}

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True, local_files_only=True)
        self.max_length = max_length + 1

    def _process(self, image_id):
        val = str(image_id).zfill(12)
        return val + '.jpg'

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, split_index, caption, tags = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))
        # Context ViT input
        inputs = self.feature_extractor(image, return_tensors="pt")
        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)
        # caption_encoded is a dict of {'input_ids': <a list of vocab indexes>, 'attention_mask': [1, 1, 1, 0 ,0 ...]
        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        # Tags: popping in Decoder
        tags_encoded = self.tokenizer.encode_plus(self.where_dict[tags[0]] + ' ' + self.when_dict[tags[1]],
                                                  max_length=10, pad_to_max_length=True, return_attention_mask=True,
                                                  return_token_type_ids=False, truncation=True)
        tag_token = np.array(tags_encoded['input_ids'])
        tag_mask = (1 - np.array(tags_encoded['attention_mask'])).astype(bool)
        # Disable attention to [CLS] or [SEP]
        tag_mask[0] = True
        tag_mask[-1] = True

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask, tag_token, tag_mask, inputs


class CocoCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = [(self._process(val['image_id']), val['caption'])
                      for val in ann['annotations']]
        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True, local_files_only=True)
        self.max_length = max_length + 1

    def _process(self, image_id):
        val = str(image_id).zfill(12)
        return val + '.jpg'

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


def build_dataset(config, mode='training'):
    if mode == 'training':
        train_dir = os.path.join(config.dir, 'train2017')
        train_file = os.path.join(
            config.dir, 'annotations', 'captions_train2017.json')
        data = CocoCaption(train_dir, read_json(
            train_file), max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform, mode='training')
        return data

    elif mode == 'validation':
        val_dir = os.path.join(config.dir, 'val2017')
        val_file = os.path.join(
            config.dir, 'annotations', 'captions_val2017.json')
        data = CocoCaption(val_dir, read_json(
            val_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode='validation')
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")


def build_dataset_deepdiary(config, mode='deepdiary'):
    data_dir = join(config.dir, 'amt_data')
    data_file = join(data_dir, 'amt_list.txt')
    anns, _ = read_deepdiary(data_dir, data_file)
    data = DeepDiary(data_dir, anns, max_length=config.max_position_embeddings,
                     limit=config.limit, transform=val_transform, mode=mode)
    return data


def build_dataset_egoco(config, mode='training'):
    if mode == 'training':
        train_dir = os.path.join(config.dir, 'train2017')
        train_file = os.path.join(
            config.dir, 'annotations', 'captions_train2017.json')
        data = EgoCapViT(train_dir, read_json(
            train_file), max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform, mode=mode)
        return data

    elif mode == 'validation':
        val_dir = os.path.join(config.dir, 'val2017')
        val_file = os.path.join(
            config.dir, 'annotations', 'captions_val2017.json')
        data = EgoCapViT(val_dir, read_json(
            val_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode=mode)
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")


def build_dataset_egocap(config, mode='training'):
    egocap_data_dir = config.egocap_data_dir
    egocap_ana_file = join(egocap_data_dir, 'doc', config.egocap_ana_filename)

    with open(egocap_ana_file, 'r') as f:
        egocap_ann = json.load(f)
    egocap_anns, egocap_train, egocap_val, egocap_test = [], [], [], []
    for key, val in egocap_ann.items():
        for cap in val['captions']:
            tags = (val['tag_stats']['where']['majority'], val['tag_stats']['when']['majority'], val['tag_stats']['who']['majority'])
            #egocap_anns.append((key, str(val['SplitIndex']).zfill(2), cap, tags))
            if val['SplitIndex'] in config.train_splits:
                egocap_train.append((key, str(val['SplitIndex']).zfill(2), cap, tags))
            elif val['SplitIndex'] in config.val_splits:
                egocap_val.append((key, str(val['SplitIndex']).zfill(2), cap, tags))
            elif val['SplitIndex'] in config.test_splits:
                pass
            else:
                print(key, val['SplitIndex'])
                raise KeyError("Not in existing Splits!")

    if mode == 'training':
        data = EgoCapViT(egocap_data_dir, egocap_train, max_length=config.max_position_embeddings,
                          limit=config.limit, transform=val_transform, mode=mode)
        return data
    elif mode == 'validation':
        data = EgoCapViT(egocap_data_dir, egocap_val, max_length=config.max_position_embeddings,
                          limit=config.limit, transform=val_transform, mode=mode)
        return data
    else:
        raise NotImplementedError(f"{mode} not supported")
