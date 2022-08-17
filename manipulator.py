import torch
from util_functions import *

def grad_cam(C, c_to_optimize, latents):
    '''
    Calculate grad-CAM-based masks of input latents. If stylegan is not none, then z space 
    is used and the gradient is backpropagated to the initial normally-distributed latent code
    '''
    requires_grad(C, True)

    if not latents.requires_grad:
        latents.requires_grad=True
    #mini batch size=1 to avoid oom
    fz_dzs = []
    
    for i in range(latents.shape[0]):
        latent = latents[i].unsqueeze(0)
        predictions, _ = C(latent)
        predictions = predictions.squeeze()[c_to_optimize]
        fz_dz = torch.autograd.grad(outputs=predictions,
                                inputs= latent,
                                retain_graph=True,
                                create_graph= True,
                                allow_unused=True
                                )[0].detach()
        fz_dzs.append(fz_dz)  
    grad = torch.cat(fz_dzs, dim=0)

    grad_cam = torch.abs(grad)
    return grad_cam, grad





