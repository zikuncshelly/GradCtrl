import os
import argparse
import numpy as np
import pickle
from shutil import copyfile



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare dataset for training")
    parser.add_argument(
        "--data_dir", help='Path to latent code-score pairs.', type=str, default='FFHQ_samples'
    )
    parser.add_argument(
        "--attributes", help='Attributes to prepare data for, in the format of attr1,attr2,...', type=str, default='gender,smile,eyeglasses,age'
    )

    parser.add_argument(
        "--out_dir", help='Path to outputs.', type=str, default='FFHQ_data'
    )
    parser.add_argument(
        "--p_perclass", help='top/bottom portion of samples being selected.', type=float, default=0.05
    )
    parser.add_argument(
        "--p_val", help='Portion of samples being used for validation.', type=float, default=0.3
    )
    args = parser.parse_args()
    
    attributes = args.attributes.split(',')
    logits_attributes = []

    os.makedirs(f'{args.out_dir}/train', exist_ok=True)
    os.makedirs(f'{args.out_dir}/val', exist_ok=True)

    for attr in attributes:
        logits_arr = []
        path_to_logits = f'{args.data_dir}/logits/logits_{attr}.pkl'
        
        with open(path_to_logits, 'rb') as handle:
            logits_arr = pickle.load(handle)

        logits_attributes.append(logits_arr)

    n_imgs = len(logits_arr)
    n_perclass = int(n_imgs*args.p_perclass)
    labels_multitask = np.zeros((n_imgs, len(attributes)))

    for f_idx in range(len(attributes)):
        logits_fs = logits_attributes[f_idx].reshape((n_imgs, -1))
        #multiclass
        if logits_fs.shape[-1] !=1:
            labels_multitask[:,f_idx] = np.argmax(logits_fs, axis=1)
        else:
            #binary
            boundary = 0
            positive_indices = logits_fs[:,0]<boundary
            labels_multitask[:,f_idx][positive_indices] = 1

    selected_indices = []
    train_indices = []
    val_indices = []

    selected_indices_train_dict = {}
    selected_indices_val_dict = {}

    for a_i, attr in enumerate(attributes):
        selected_indices = set()
        scores = logits_attributes[a_i].reshape((n_imgs, -1))
        #binary
        if scores.shape[-1] == 1:
            boundary = 0
            sorted_score_indices = np.argsort(scores[:,0])

            top_indices = sorted_score_indices[-n_perclass:]
            bttm_indices = sorted_score_indices[:n_perclass]
            
            selected_indices |= set(list(top_indices))
            selected_indices |= set(list(bttm_indices))

        else:
            #multiclass
            indices = np.asarray(range(len(scores)))
            maxes = np.argmax(scores, axis=1)
            for c in range(scores.shape[-1]):
                temp = scores.copy()[:,c]
                temp[maxes!=c] = 0
                sorted_c_samples = np.argsort(temp)
                top_upper = len(temp[temp>0])
                selected_indices_ = sorted_c_samples[top_upper-n_perclass:top_upper]
                selected_indices |= set(list(selected_indices_))

        selected_indices = list(selected_indices)

        val_indices = set(list(np.random.choice(selected_indices, int(args.p_val*len(selected_indices)), replace=False)))
        train_indices = set(list(selected_indices))-val_indices
        val_indices = sorted(np.array(list(val_indices)))
        train_indices = sorted(np.array(list(train_indices)))
        selected_indices_train_dict[attr] = train_indices
        selected_indices_val_dict[attr] = val_indices

        print(f'{len(val_indices)} validation samples for {attr}')
        print(f'{len(train_indices)} training samples for {attr}')


    copyfile(f'{args.data_dir}/latents.pkl',f'{args.out_dir}/latents.pkl')
    with open(f'{args.out_dir}/labels.pkl', 'wb') as handle:
            pickle.dump(labels_multitask, handle)
    with open(f'{args.out_dir}/train/indices_dict.pkl', 'wb') as handle:
        pickle.dump(selected_indices_train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{args.out_dir}/val/indices_dict.pkl', 'wb') as handle:
        pickle.dump(selected_indices_val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)