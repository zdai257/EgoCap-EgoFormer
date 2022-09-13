import os
from os.path import join
import sys
import numpy as np
import onnx
import onnxruntime as rt
import statistics
import time
from transformers import BertTokenizer
from PIL import Image, ImageOps
import torchvision as tv

from datasets import coco
from Eval import create_caption_and_mask


MAX_DIM = 299


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    # force shape 299, 224
    new_shape = (299, 224)
    image = image.resize(new_shape)

    return image


ego_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_image(path=None):
    # Load Image
    image = Image.open(path)
    # Transpose with respect to EXIF data
    image = ImageOps.exif_transpose(image)
    w, h = image.size
    print("PIL Image width: {}, height: {}".format(w, h))

    # transforms
    image = under_max(image)

    img = np.array(image, dtype=np.float32).transpose(2, 0, 1)
    for c in range(3):
        img[c] = (img[c] - np.mean(img[c])) / np.std(img[c])
    print(img)
    return img


def create_caption_and_mask_np(start_t, max_length):
    caption_template = np.zeros((1, max_length), dtype=np.long)
    mask_template = np.ones((1, max_length), dtype=np.bool)

    caption_template[:, 0] = start_t
    mask_template[:, 0] = False

    return caption_template, mask_template


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    ego_model = "./EgoFormer6.onnx"

    sample_path = "./Qualitative_samples/origin/0f4e630b-e834-4ff4-9418-ccfdbdc4ee37_small.jpg"

    # ONNX checker
    onnx_model = onnx.load(ego_model)
    onnx.checker.check_model(onnx_model)

    sess = rt.InferenceSession(ego_model, None)

    # Inputs
    input_name0 = sess.get_inputs()[0].name
    input_name1 = sess.get_inputs()[1].name
    input_name2 = sess.get_inputs()[2].name
    label_name = sess.get_outputs()[0].name
    print("Input names:\n{}\n{}\n{}".format(input_name0, input_name1, input_name2))
    print("Output name: ", label_name)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True, local_files_only=False)
    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)

    # Load skeleton caption
    cap, cap_mask = create_caption_and_mask_np(start_token, 128)
    print(cap.shape, cap_mask.shape)

    # Input formatting
    x = list()
    #x.append(np.expand_dims(np.random.rand(3, 224, 224).astype(np.float32), axis=0))
    # Get Image input
    image = Image.open(sample_path)
    image = ImageOps.exif_transpose(image)
    w, h = image.size
    print("PIL Image width: {}, height: {}".format(w, h))
    sample = coco.val_transform(image)
    sample = sample.unsqueeze(0)
    #sample = ego_transform(image)
    print(sample.shape)

    #np_sample = np.asarray(sample.detach(), dtype=np.float32)
    #print(np_sample.shape)
    #x.append(np.expand_dims(np_sample, axis=0))
    x.append(to_numpy(sample).astype(np.float32))
    x.append(cap)
    x.append(cap_mask)

    for i in range(128 - 1):
        #print(x)
        x = x if isinstance(x, list) else [x]
        inputs = dict([(item.name, x[idx]) for idx, item in enumerate(sess.get_inputs())])

        '''
        for idx, item in enumerate(sess.get_inputs()):
            print("Input %d: %s" % (idx, item.name))
        for idx, item in enumerate(sess.get_outputs()):
            print("Output %d: %s" % (idx, item.name))
        '''
        print(inputs)

        start_t = time.time()
        pred = sess.run([label_name], inputs)
        #pred = sess.run(None, inputs)
        elapsed_t = time.time() - start_t
        print("Word %d onnx sess time = %.04f seconds" % (i, elapsed_t))
        print(pred[0].shape)

        prediction = pred[0][:, i, :]
        predicted_id = np.argmax(prediction, axis=-1)
        print(predicted_id)
        # End of Sentence
        if predicted_id[0] == 102:
            break

        cap[:, i + 1] = predicted_id[0]
        cap_mask[:, i + 1] = False

        # update Inputs
        x[1] = cap
        x[2] = cap_mask

    result = tokenizer.decode(cap[0].tolist(), skip_special_tokens=True)
    print('RESULT: ' + result.capitalize() + '\n')


if __name__ == "__main__":

    main()
