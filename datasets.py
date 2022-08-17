import torch
import pickle
import os

#a dataset that for multilabel learning that loads dict of training samples for different classes
#and latents&corresponding labels

class CustomMultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_latents, path_to_labels, path_to_indices, attributes):
        with open(os.path.join(path_to_latents, 'latents.pkl'), 'rb') as f:
            self.latents = pickle.load(f)
        with open(os.path.join(path_to_labels, 'labels.pkl'), 'rb') as f:
            self.labels = pickle.load(f)
        with open(os.path.join(path_to_indices, 'indices_dict.pkl'), 'rb') as f:
            self.indices_dict = pickle.load(f)

        assert len(self.latents) == len(self.labels)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.attributes = attributes

    def __len__(self):
        return len(self.indices_dict[self.attributes[0]])
        
    def __getitem__(self, idx):
        """
        idx: indices to select from training set for each attribute
        out latent is of shape 512 x n_branches, out_label is of shape n_branches x n_branches
        """
        index = [self.indices_dict[f][idx] for f in self.attributes]#get index of selected samples in the entire dataset
        latent = torch.cat([torch.tensor(self.latents[i]).unsqueeze(1) for i in index], dim=1)#latents for different classifier at different columns
        label = torch.cat([torch.tensor(self.labels[i]).unsqueeze(1) for i in index], dim=1)#labels for differnt classifier at different columns

        return latent, label    

