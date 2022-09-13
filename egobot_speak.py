import time
from configuration import Config, ConfigEgo
from Eval import predict_qualitative


def main(conf):

    checkpoint_path = "Baseline2-best_epoch32_loss10.pth"

    sample_path = "./Qualitative_samples/origin/0f4e630b-e834-4ff4-9418-ccfdbdc4ee37_small.jpg"

    start_t = time.time()
    cap_dict = predict_qualitative(conf, sample_path, tags=None, checkpoint_path=checkpoint_path)
    elapsed_t = time.time() - start_t

    print("Sentence inference time = %.5fs" % elapsed_t)


if __name__ == "__main__":
    config = Config()
    main(config)
