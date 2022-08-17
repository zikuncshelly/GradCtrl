from stylegan2 import pretrained_networks
from stylegan2 import dnnlib
from dnnlib import tflib
import torch
from PIL import Image
import argparse
import pickle
import numpy as np
import os


def sample_latentcode(stylegan, n_samples=10, truncation_psi=0.4):
    z = np.random.rand(n_samples, 512)
    w = stylegan.components.mapping.run(z, None)[:,0,:]
    w_avg = stylegan.get_var('dlatent_avg')

    return w_avg + (w - w_avg) * truncation_psi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='sample latent codes',
    )
    parser.add_argument('--model', choices=['ffhq', 'cat', 'car'], default='ffhq')
    parser.add_argument('--out_dir', help='Path to output images.', required='FFHQ_samples')
    parser.add_argument('--n_samples', help='Number of samples to edit. Will be ignored if --input_dir is set.', default=10000)

    args = parser.parse_args()

    os.makedirs(f'{args.out_dir}/images', exist_ok=True)

    _, _, stylegan = pretrained_networks.load_networks(f'gdrive:networks/stylegan2-{args.model}-config-f.pkl')
    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False

    codes = sample_latentcode(stylegan, n_samples=args.n_samples)
    for c_i, c in enumerate(codes):
        im = stylegan.components.synthesis.run(c.numpy(), **Gs_syn_kwargs)[0]
        im = Image.fromarray(im, 'RGB')
        im.save(f'{args.out_dir}/images/{c_i}.jpg')
    with open(f'{args.out_dir}/latents.pkl', 'wb') as handle:
        pickle.dump(codes, handle)
