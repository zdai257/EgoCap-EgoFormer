class Config(object):
    def __init__(self):
        # Learning Rates
        self.lr_backbone = 1e-5
        self.lr = 1e-4

        # Epochs
        self.epochs = 30
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Backbone
        self.backbone = 'resnet101'
        self.position_embedding = 'sine'
        self.dilation = True

        # Basic
        self.device = 'cuda'
        self.seed = 42
        self.batch_size = 32
        self.num_workers = 8
        self.checkpoint = './checkpoint.pth'
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
        self.dir = '../coco'
        self.limit = -1

        # Video dataset
        self.modality = 'image'


class ConfigEgo(object):
    def __init__(self):
        # Learning Rates
        self.lr_backbone = 0e-5
        self.lr = 1e-5
        self.lr_ctx_vit = 1e-6

        # Epochs
        self.epochs = 50
        self.lr_drop = 20
        self.start_epoch = 12  # Finetune starting from 11 + 1
        self.weight_decay = 1e-4
        # Warm Up: steps / (batch * epochs)
        self.warmup_steps = 24

        # Backbone
        self.backbone = 'resnet101'
        self.position_embedding = 'sine'
        self.dilation = True

        # Basic
        self.device = 'cuda:2'
        self.seed = 42
        self.batch_size = 8
        self.num_workers = 8
        self.checkpoint = './EgoFormer3-equalloss.pth'
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
        self.dir = '/mnt/datasets/COCO'
        self.limit = -1

        # TYPE of dataset
        self.modality = 'ego'
        self.IsFinetune = True
        self.pretrain_checkpoint = "./checkpoint_cl.pth"
        # Ego dataset
        self.egocap_data_dir = "/home/zdai/repos/EgoCapSurvey"
        self.egocap_ana_filename = "EgoCap_annatations_ref.json"
        self.train_splits = [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21]
        self.val_splits = [3, 10, 17]
        self.test_splits = [1, 2]

        # ViT
        self.vit_lr = 1e-4
        self.vit_body_lr = 1e-5
        self.vit_weight_decay = 1e-3
        self.vit_weights = (0.9, 0.69, 0.49)
        self.pretrain_ctx_vit = "./vit_checks/prob_equalloss32-accwhere96_accwhen51_accwhom66.pth"


class Config2(object):
    def __init__(self):
        # Learning Rates
        self.lr_backbone = 3e-4
        self.lr = 3e-3

        # Epochs
        self.epochs = 50
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4
        # Warm Up
        self.warmup_steps = 48

        # Backbone
        self.backbone = 'resnet101'
        self.position_embedding = 'sine'
        self.dilation = True

        # Basic
        self.device = 'cuda:0'
        self.seed = 42
        self.batch_size = 16  # 32
        self.num_workers = 8
        self.checkpoint = './checkpoint.pth'
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 512  # 256
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
        self.dir = "/root/datasets/COCO"
        self.limit = -1

        # Validation sample limit
        self.val_limit = 500
        # Video Dataset
        self.modality = 'image'
        self.msvd_data_dir = "/home/zdai/repos/MSVD"
        self.frame_per_clip = 5
        self.min_frame_per_clip = 7


class Config3(object):
    def __init__(self):
        # Learning Rates
        self.lr_backbone = 1e-5
        self.lr = 1e-4

        # Epochs
        self.epochs = 30
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Backbone
        self.backbone = 'resnet101'
        self.position_embedding = 'sine'  # 'learned' / 'sine'
        self.dilation = True

        # Basic
        self.device = 'cuda'
        self.seed = 42
        self.batch_size = 16  # 32
        self.num_workers = 8
        self.checkpoint = './checkpoint.pth'
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
        self.dir = "/root/datasets/COCO"
        self.limit = -1


class Config4(object):
    def __init__(self):
        # Learning Rates
        self.lr_backbone = 1e-5
        self.lr = 1e-4

        # Epochs
        self.epochs = 30
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4
        # Warm Up
        self.warmup_steps = 10

        # Backbone
        self.backbone = 'resnet101'
        self.position_embedding = 'sine'  # 'learned' / 'sine'
        self.dilation = True

        # Basic
        self.device = 'cuda'
        self.seed = 42
        self.batch_size = 1  # 32
        self.num_workers = 8
        self.checkpoint = './checkpoint_vid.pth'
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
        self.dir = "/home/zdai/repos/EMS-cGAN/coco2017"
        self.limit = -1

        # Validation sample limit
        self.val_limit = 500
        # Video Dataset
        self.modality = 'video'
        self.msvd_data_dir = "/home/zdai/repos/MSVD"
        self.msvd_sub_dir = 'skipped'
        self.frame_per_clip = 2
        self.min_frame_per_clip = 7


class Config5(object):
    def __init__(self):
        # Learning Rates
        self.lr_backbone = 1e-5  # if > 0 allowing training Backbone network
        self.lr = 1e-4

        # Epochs
        self.epochs = 30
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4
        # Warm Up
        self.warmup_steps = 10

        # Backbone
        self.backbone = 'resnet34'
        self.position_embedding = 'sine'  # 'sine' / 'learned'
        self.dilation = False  # True

        # Basic
        self.device = 'cuda'
        self.seed = 42
        self.batch_size = 1  # 32
        self.num_workers = 8
        self.checkpoint = './checkpoint_time_only.pth'
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 512
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
        # self.dir = '../coco'
        self.dir = "/home/zdai/repos/EMS-cGAN/coco2017"
        self.limit = -1
        # Validation sample limit
        self.num_limit = 500
        # Video Dataset
        self.modality = 'video'
        self.msvd_data_dir = "/home/zdai/repos/MSVD"
        self.msvd_sub_dir = 'skip64'
        self.frame_per_clip = 64
        self.min_frame_per_clip = 64
