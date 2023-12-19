from datasets.coco import *
from configuration import *

import pickle


def main(config):
    egocap_data_dir = config.egocap_data_dir
    egocap_ana_file = join(egocap_data_dir, 'doc', config.egocap_ana_filename)

    with open(egocap_ana_file, 'r') as f:
        egocap_ann = json.load(f)

    egocap_anns, egocap_train, egocap_val, egocap_test = [], {}, {}, {}

    for key, val in egocap_ann.items():
        for cap in val['captions']:
            tags = (val['tag_stats']['where']['majority'], val['tag_stats']['when']['majority'],
                    val['tag_stats']['who']['majority'])
            egocap_anns.append((key, str(val['SplitIndex']).zfill(2), cap, tags))
            # a dict of 'image_id': ["caption1", "caption2" ...]
            if val['SplitIndex'] in config.train_splits:
                egocap_train['Split' + str(val['SplitIndex']).zfill(2) + '__' + key] = val['captions']
            elif val['SplitIndex'] in config.val_splits:
                egocap_val['Split' + str(val['SplitIndex']).zfill(2) + '__' + key] = val['captions']
            elif val['SplitIndex'] in config.test_splits:
                egocap_test['Split' + str(val['SplitIndex']).zfill(2) + '__' + key] = val['captions']
            else:
                print(key, val['SplitIndex'])
                raise KeyError("Not in existing Splits!")

    # export to .pkl
    with open('./egocap_train.pkl', 'wb') as handle:
        pickle.dump(egocap_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./egocap_val.pkl', 'wb') as handle:
        pickle.dump(egocap_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./egocap_test.pkl', 'wb') as handle:
        pickle.dump(egocap_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    config = ConfigEgo()
    main(config)

