import argparse
import io
import os
from os.path import join
import time
import torch
from configuration import Config, ConfigEgo
from Eval import create_caption_and_mask
from models import caption
from transformers import BertTokenizer, ViTFeatureExtractor
from datasets import coco
from PIL import Image, ImageOps
import json
from threading import Timer


class RepeatTimer(Timer):  
    def run(self):  
        while not self.finished.wait(self.interval):  
            self.function(*self.args,**self.kwargs)  
            print(' ')


def infer(args, config, model, feature_extractor, log="/home/nvidia/caps.json"):
    #time.sleep(1)
    sorted_frames = sorted(os.listdir(args.path), key=lambda x: int(x.split('.')[0].split('-')[-1]))
    sample_frame = sorted_frames[-1]
    # Static samples
    #sample_path = "./Qualitative_samples/origin/0f4e630b-e834-4ff4-9418-ccfdbdc4ee37_small.jpg"
    #sample_path = [os.path.join("./Qualitative_samples/origin", img) for img in os.listdir("./Qualitative_samples/origin")]
    # Dynamic sample
    sample_path = os.path.join(args.path, sample_frame)
    
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
        return out, decoded_batch_beams
    
    # INFER
    if args.model == "egoformer":
        inputs = feature_extractor(image, return_tensors="pt")
        img_tensor = inputs['pixel_values'].squeeze(1).to(device)
    else:
        img_tensor = None
        
    start_t = time.time()
    output, outputs = evaluate(sample, cap, cap_mask, img_tensor) 
    elapsed_t = time.time() - start_t
    print("Sentence inference time = %.5fs" % elapsed_t)

    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    
    if os.path.isfile(log) and os.access(log, os.R_OK):
        # checks if file exists
        #print ("caps.json exists and is readable")
        pass
        
    else:
        print ("Either caps.json is missing or is not readable, creating file...")
        with io.open(log, 'w') as db_file:
            db_file.write(json.dumps({}))
            
    with open(log, 'r') as f:
        caps_dict = json.load(f)
    caps_dict.update({sample_frame: result})
    with open(log, 'w') as f:
        json.dump(caps_dict, f)


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='Launch inference engine')
    my_parser.add_argument('--path',
                       type=str,
                       required=False,
                       default="/home/nvidia/Pictures/EgoShot/",
                       help='the path to image saving folder')
    my_parser.add_argument('--model',
                       type=str,
                       required=False,
                       default="baseline",
                       help='model selection: baseline or egoformer')
    args = my_parser.parse_args()
    
    # Prepare model
    if args.model == "baseline":
        conf = Config()
    elif args.model == "egoformer":
        conf = ConfigEgo()
    else:
        raise("model not supported!")
        exit()
    
    if args.model == "baseline":
        checkpoint_path = "Baseline2-best_epoch32_loss10.pth"
        model, _ = caption.build_model(conf)
    elif args.model == "egoformer":
        checkpoint_path = "EgoFormer3-equalloss-best_epoch33_loss10.pth"
        model, _ = caption.build_model_egovit(conf)
    print("Loading Checkpoint...")
    checkpoint_tmp = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_tmp['model'])
    print("Current checkpoint epoch = %d" % checkpoint_tmp['epoch'])
    
    start_t_tokenizer = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True, local_files_only=False)
    print("Loading pretrained Tokenizer takes: %.2fs" % (time.time() - start_t_tokenizer))
    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(join(root_dir, 'datasets', "vit_classify-feature_extractor")):
        feature_extractor = ViTFeatureExtractor.from_pretrained(join(root_dir, 'datasets', "vit_classify-feature_extractor"))
    else:
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        feature_extractor.save_pretrained(join(root_dir, 'datasets', "vit_classify-feature_extractor"))
    
    
    ##We are now creating a thread timer and controling it  
    timer = RepeatTimer(30, infer, [args, conf, model, feature_extractor])
        
    timer.start()
    time.sleep(300)
    print('Inference finishing.')
    timer.cancel()
    
