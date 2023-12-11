class Config(object):o
    def __init__(self):
        # Learning Rates
        self.lr_backbone = 3e-5
        self.lr = 1e-4

        # Epochs
        self.epochs = 30
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Backbone
        self.backbone = 'resnet18'
        self.position_embedding = 'sine'
        self.dilation = True

        # Basic
        self.device = 'cuda:1'
        self.seed = 42
        self.batch_size = 32
        self.num_workers = 8
        self.checkpoint = './checkpoint-tiny.pth'
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 256
        self.pad_token_id = 0
        self.max_position_embeddings = 128
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 30522

        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        self.dir = '/users/d/daiz1/COCO'  #TODO: specify COCO dir
        self.limit = -1

        # Dataset type
        self.modality = 'image'
        self.IsFinetune = False
        self.IsBlindEgoco = True
        # TODO: specify pretrained context ViT classifier path
        self.pretrain_ctx_vit = "./ctx_vit_raw0.pth"


class ConfigEgo(object):
    def __init__(self):
        # Learning Rates
        self.lr_backbone = 0e-5
        self.lr = 1e-5
        self.lr_ctx_vit = 0.  #1e-6

        # Epochs
        self.epochs = 80
        self.lr_drop = 20
        self.start_epoch = 0  #12  # Finetune starting from 11 + 1
        self.weight_decay = 1e-4
        # Warm Up: steps / (batch * epochs)
        self.warmup_steps = 24

        # Backbone
        self.backbone = 'resnet101'
        self.position_embedding = 'sine'
        self.dilation = True

        # Basic
        self.device = 'cuda:0'
        self.seed = 42
        self.batch_size = 8
        self.num_workers = 8
        self.checkpoint = './EgoCO_raw.pth'
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 256
        self.pad_token_id = 0
        self.max_position_embeddings = 128
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 30522

        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        self.dir = '/mnt/datasets/COCO'  #TODO: specify COCO dir
        self.limit = -1

        # TYPE of dataset
        self.modality = 'ego'  # 'ego' for EgoFormer; 'image' for baseline Transformer
        self.IsFinetune = True
        self.pretrain_checkpoint = "./checkpoint_cl.pth"  #TODO: specify pretrained baseline Transformer path
        # Ego dataset
        self.egocap_data_dir = "/home/zdai/repos/EgoCapSurvey"  #TODO: specify EgoCap path
        self.egocap_ana_filename = "EgoCap_annatations_ref.json"
        self.train_splits = [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21]
        self.val_splits = [1, 2]
        self.test_splits = [3, 10, 17]

        # Context ViT training
        self.vit_lr = 1e-4
        self.vit_body_lr = 1e-5
        self.vit_weight_decay = 1e-3
        self.vit_weights = (0.9, 0.69, 0.49)
        self.IsBlindEgoco = True
        # TODO: specify pretrained context ViT classifier path
        self.pretrain_ctx_vit = "./ctx_vit_raw.pth"

