import torch
import os
import argparse
import numpy as np
import pickle
from PIL import Image
from stylegan2 import pretrained_networks
from stylegan2 import dnnlib
from dnnlib import tflib
from util_functions import *
from manipulator import *
from tqdm import tqdm
from sample import sample_latentcode
from classifiers import *

EXCLUDE_DICT_FFHQ = {0:[[1,2,3],[250,100,100],[2,10]],1:[[0,2,3],[200,50,100], [2,8]],
        2:[[0,1,3],[200,100,250],[1,4]], 3:[[0,2],[100,300],[4,10]]}

def generate_one(totest, stylegan, Gs_syn_kwargs):
    temp = torch.repeat_interleave(totest.clone().unsqueeze(0), 18, dim=1)
    original = stylegan.components.synthesis.run(temp.numpy(), **Gs_syn_kwargs)[0]
    return Image.fromarray(original, 'RGB')


def main(args):

    #Load StyleGAN2-model
    _, _, stylegan = pretrained_networks.load_networks(f'gdrive:networks/stylegan2-{args.model}-config-f.pkl')
    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False

    #load trained model
    with open(os.path.join(args.model_dir, 'model_params.pkl'), 'rb') as handle:
        model_params = pickle.load(handle)
    model = MultiLabelClassifier(**model_params).cuda()
    load_model(model, os.path.join(args.model_dir, 'model.pth'))

    c_to_optimize = model.attributes.index(args.attribute)
    if args.model == 'ffhq':
        resolution = 1024
        editing_params = EXCLUDE_DICT_FFHQ[c_to_optimize]
        c_to_excludes = editing_params[0] if args.exclude=='default' else [model_params['attributes'].index(a) for a in args.exclude.split(',')]
        top_cs = editing_params[1] if args.top_channels=='default' else [int(c) for c in args.top_channels.split(',')]
        [layer_lower,layer_upper] = editing_params[2] if args.layerwise=='default' else [int(l) for l in args.layerwise.split(':')]
    else:
        if args.model == 'car':
            resolution = 512
        if args.model == 'cat':
            resolution = 256
        layer_lower, layer_upper = 6,14
        c_to_excludes = []
        top_cs = []
    

    os.makedirs(args.out_dir, exist_ok=True)
    if args.input_dir is None:
        #sample latent codes from W space
        latents = sample_latentcode(stylegan,args.n_samples)
    else:
        with open(args.input_dir, 'rb') as handle:
            latents = pickle.load(handle)
    
    for l_i, l in tqdm(enumerate(latents)):
        original_totest = torch.tensor(l).unsqueeze(0)
        original = generate_one(original_totest,stylegan,Gs_syn_kwargs)

        if np.unique(model.multiclass_classes) == [1]:
            directions = [1,-1]
            c_to_optimizes = [c_to_optimize]
        else:#hardcoded for cats, 3 colors
            directions = [1]
            c_to_optimizes = range(3)
        for direction in directions:
            for c_to_optimize in c_to_optimizes:
                new_im = Image.new('RGB', (resolution*10, resolution), "#ddd")
                new_im.paste(original, box=(0,0))
                totest = original_totest.clone().cuda()
                totest.requires_grad = True
                for s in range(9):
                    dims_to_exclude = []
                    predictions_full = model(totest)
                    predictions = predictions_full[0].squeeze()
                    #compute editing direction
                    fz_dz_target = torch.autograd.grad(outputs=predictions[c_to_optimize],
                                                            inputs= totest,
                                                            retain_graph=True,
                                                            create_graph= True,
                                                            allow_unused=True
                                                            )[0]

                    #disentanglement through channel filtering                                        
                    for c_idx, c_to_exclude in enumerate(c_to_excludes):
                        dim_c,_ = grad_cam(model, c_to_exclude, totest)
                        dim_c = dim_c.detach().squeeze().cpu().numpy()
                        excluded = np.argsort(dim_c)[-top_cs[c_idx]:]
                        dims_to_exclude.append(excluded)
                    if len(dims_to_exclude)>0:
                        dims_to_exclude = np.unique(np.concatenate(dims_to_exclude))
                        mask = torch.ones(512).cuda()
                        mask[dims_to_exclude] = 0
                        fz_dz_target *= mask
                    fz_dz_target /= torch.norm(fz_dz_target)

                    with torch.no_grad():
                        totest += fz_dz_target*direction*4/9
                        temp = torch.repeat_interleave(original_totest.unsqueeze(0), 18, dim=1)
                        temp[0,layer_lower:layer_upper,:] = totest

                    out = stylegan.components.synthesis.run(temp.cpu().detach().numpy(), **Gs_syn_kwargs)[0]

                    out = Image.fromarray(out, 'RGB')
                    new_im.paste(out, box=((s+1)*resolution,0))
                if len(directions) == 2:
                    d = 'positive' if direction >0 else 'negative'
                else:
                    d = f'direction_{c_to_optimize}'
                new_im.save(f'{args.out_dir}/{l_i}_{args.attribute}_{d}.jpg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Manipulate latent semantics',
    )
    parser.add_argument('--model', choices=['ffhq', 'cat', 'car'], default='ffhq')

    parser.add_argument('--input_dir', help='Path to latent code inputs.', default=None)
    parser.add_argument('--model_dir', help='Path to the trained model&model params.', default='model_ffhq')
    parser.add_argument('--attribute', help='Attribute to edit.', required=True)
    parser.add_argument('--out_dir', help='Path to output images.', required='out_ffhq')
    parser.add_argument('--exclude', help='Attributes to disentangle in the format of attr1,attr2,...; default: Settings in the paper. ', default='default')
    parser.add_argument('--top_channels', help='Numbers of top channels to exclude in the format of attr1_c,attr2_c...', default='default')
    parser.add_argument('--n_samples', help='Number of samples to edit. Will be ignored if --input_dir is set.', default=10)
    parser.add_argument('--layerwise', help='Ranges of layers in W+ being modified in the format of layer_begin:layer_end, default: Settings in the paper', default='default')

    args = parser.parse_args()

    main(args)
