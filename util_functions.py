import torch
import numpy as np
from torch.nn import functional as F

def device(tensor_list):
    if torch.cuda.is_available():
        return [t.cuda() for t in tensor_list]
    else:
        return tensor_list

def calculate_step_num(epoch, batch, dataloader):
    return (epoch - 1) * len(dataloader) + batch

def update_losses(running_losses, losses, scale):
    if running_losses is None:
        running_losses = {l: 0 for l in losses}
    for l in losses:
        running_losses[l] += losses[l] / scale
    return running_losses

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def save_model(model, save_path):
    model.eval()
    torch.save(model.state_dict(), save_path)
    model.train()


def load_model(model, save_path):
    model.eval()
    model.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage), strict=False)
    model.train()

def classification_accuracy(predictions, targets, n_classes=None, is_multitask=False):
    if is_multitask:
        targets = torch.cat([targets[:,:,i] for i in range(targets.shape[2])], dim=0)
        predictions[predictions<0.5] = 0
        predictions[predictions>0.5] = 1
        correct = 0
        for i in range(targets.shape[0]):
            #target of shape b_size x n_branches x n_branches
            if len(predictions[i][predictions[i]!=targets[i]])==0:
                correct += 1
        accuracy = correct / targets.shape[0]

        accurate_perclass = []
        n_perclass = []
        #correct per class
        for j in range(targets.shape[1]):
            f_pred = predictions[:, j]
            f_label = targets[:,j]
            accurate_f = len(f_pred[f_pred==f_label])
            accurate_perclass.append(accurate_f)
            n_perclass.append(len(predictions))
        return accuracy, np.array(accurate_perclass), np.array(n_perclass)

    else:
        _, top_classes = predictions.max(1)
        targets_onehot = F.one_hot(targets, num_classes=n_classes)
        top_classes_onehot = F.one_hot(top_classes, num_classes=n_classes)
        correct_perclass = (targets_onehot.cpu().numpy()* top_classes_onehot.cpu().numpy()).sum(0)
        n_perclass = targets_onehot.sum(0).cpu().numpy()
        n_correct = (top_classes == targets).sum().item()
        accuracy = n_correct / len(targets)
        return accuracy, correct_perclass, n_perclass

def evaluate_dataset(model, dataloader, args, n_classes=None):
    model.eval()
    n_branches = len(args.attributes.split(','))

    running_losses = None
    correct = np.zeros((n_branches,))
    n = np.zeros((n_branches,))
    with torch.no_grad():
        for data in dataloader:

            images, labels = data
            images, labels = device([images, labels])
            labels = labels.view(labels.shape[0],-1,labels.shape[-2],labels.shape[-1])
            images = images.float()

            predictions, p = model(images)
            if args.multiclass_classes is None:
                accuracy, correct_perclass_, n_perclass_ = classification_accuracy(p, labels, n_classes, is_multitask=True)
                correct += correct_perclass_
                n += n_perclass_
            else:
                p = torch.argmax(predictions.squeeze(1),dim=1)
                label_b = labels.squeeze(2).squeeze(2)
                accuracy = len(p[p==label_b])/len(p)
            running_losses = update_losses(running_losses, {'accuracy': accuracy}, len(dataloader))
    model.train()
    return running_losses, correct, n