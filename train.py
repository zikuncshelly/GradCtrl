import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import argparse
import os
import shutil


from util_functions import *
from classifiers import *
from datasets import *

def main(args):
    save_dir = args.out_dir
    attributes = args.attributes.split(',')
    shutil.rmtree(save_dir, ignore_errors=True)
    os.mkdir(save_dir)

    #training & model parameters
    hidden_size = args.hidden_size
    n_shared_layers = args.n_shared_layers
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    print_freq = args.print_freq
    running_losses = None
    best_accuracy = 0
    model_params = {'n_attributes': len(attributes), 
                    'n_shared_layers': n_shared_layers,
                    'hidden_size': hidden_size ,
                    'multiclass_classes': args.multiclass_classes}

    #data paths
    latent_path = args.data_dir
    label_path = args.data_dir
    train_indices = os.path.join(args.data_dir, 'train')
    val_indices = os.path.join(args.data_dir, 'val')


    model = MultiLabelClassifier(**model_params).cuda()
    train_loader = DataLoader(CustomMultiLabelDataset(latent_path, label_path, train_indices, attributes), batch_size=batch_size, num_workers=0, shuffle=True)
    test_loader = DataLoader(CustomMultiLabelDataset(latent_path, label_path, val_indices, attributes), batch_size=batch_size, num_workers=0)

    optimizer = optim.Adam(model.parameters())
    bar = tqdm(range(1, n_epochs + 1))
    val_losses = None


    for epoch in bar:
        ce_losses = []
        for batch, data in enumerate(train_loader):
            step_num = calculate_step_num(epoch, batch, train_loader)

            latents, labels = data
            latents, labels = device([latents, labels])
            labels = labels.view(labels.shape[0],-1,labels.shape[-2],labels.shape[-1])
            latents = latents.float()

            predictions, full_pred = model(latents)
            
            loss_ce = 0
            #Multilabel training, only updating each branch with relevant data
            for b in range(labels.shape[1]):
                label_b = labels.type(torch.FloatTensor)[:,:,b,b].cuda()
                loss_ce += F.binary_cross_entropy(predictions[:,:,b].squeeze(), label_b.squeeze())

            loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            p = full_pred
            is_multitask = len(attributes) > 1
            accuracy, _, _ = classification_accuracy(p, labels, is_multitask=is_multitask)

            losses = {'loss_ce': loss_ce.item(), 'accuracy': accuracy}

            running_losses = update_losses(running_losses, losses, print_freq)
            ce_losses.append(loss.item())

            if (step_num + 1) % print_freq == 0:
                running_losses = {l: 0 for l in running_losses}
                if val_losses is None:
                    bar.set_postfix({'train_loss': sum(ce_losses)/len(ce_losses),'train_accuracy':accuracy,
                    'val_accuracy': 'None'})

                else:
                    bar.set_postfix({'train_loss': sum(ce_losses)/len(ce_losses),'train_accuracy':accuracy,
                    'val_accuracy': val_losses['accuracy']})
            
        val_losses, _, _ = evaluate_dataset(model, test_loader, args)
        bar.set_postfix({'train_loss': sum(ce_losses)/len(ce_losses),'train_accuracy':accuracy,
            'val_accuracy': val_losses['accuracy']})
        
        if val_losses['accuracy'] >= best_accuracy:
            save_model(model, os.path.join(save_dir, 'model.pth'))
            best_accuracy = val_losses['accuracy']

    with open(os.path.join(save_dir, 'model_params.pkl'), 'wb') as handle:
        pickle.dump(model_params, handle)
                                                                               

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train model for attribute editing',
    )
    parser.add_argument('--data_dir', help='Path to training data.', default=None)
    parser.add_argument('--out_dir', help='Path to outputs.', required='out_ffhq')
    parser.add_argument('--attributes', help='Attributes to train on, in the format of attr1,attr2,...', default='gender,smile,eyeglasses,age')


    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')

    parser.add_argument('--n_hidden_layers', type=int, default=1)
    parser.add_argument('--n_shared_layers', type=int, default=0,
                        help='number of shared layers')
    parser.add_argument('--hidden_size', type=int, default=16,
                        help='hidden layer dim')
    parser.add_argument('--multiclass_classes', type=str, default=None)

    parser.add_argument('--print_freq', type=int, default=30)
    args = parser.parse_args()

    main(args)
