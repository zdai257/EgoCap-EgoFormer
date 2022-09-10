import os
from os.path import join
import sys
import numpy as np
import onnxruntime as rt
import statistics
import time
from transformers import BertTokenizer
from PIL import Image, ImageOps


def get_image(path=None):
    # Load Image
    image = Image.open(path)
    # Transpose with respect to EXIF data
    image = ImageOps.exif_transpose(image)
    w, h = image.size
    print("PIL Image width: {}, height: {}".format(w, h))

    # transforms
    image = under_max(image)

    return np.asarray(image, dtype=np.float32)


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


def create_caption_and_mask(start_t, max_length):
    caption_template = np.zeros((1, max_length), dtype=np.long)
    mask_template = np.ones((1, max_length), dtype=np.bool)

    caption_template[:, 0] = start_t
    mask_template[:, 0] = False

    return caption_template, mask_template


def main():
    ego_model = "./BaseFormer.onnx"

    sample_path = "./Qualitative_samples/origin/fjDvKHkmxs0_119_126.avi00001.jpg"

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
    cap, cap_mask = create_caption_and_mask(start_token, 128)

    # Input formatting
    x = list()
    #x.append(np.expand_dims(np.random.rand(3, 224, 224).astype(np.float32), axis=0))
    x.append(np.expand_dims(get_image(sample_path), axis=0))
    x.append(cap)
    x.append(cap_mask)

    for i in range(128 - 1):

        x = x if isinstance(x, list) else [x]
        inputs = dict([(item.name, x[n]) for n, item in enumerate(sess.get_inputs())])

        start_t = time.time()
        pred = sess.run([label_name], inputs)
        elapsed_t = time.time() - start_t
        print("Word %d onnx sess time = %.04f seconds" % (i, elapsed_t))

        prediction = pred[0][:, i, :]
        predicted_id = np.argmax(prediction, axis=-1)

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
