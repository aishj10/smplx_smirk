import numpy as np
import os
import smplx
import torch


reference_file = "/data/aujadhav/champ/celebv/reference_imgs/smpl_results/Video_I2hf-H8Xumk_2_0001.npy"
smirk_out = "/data/aujadhav/smirk/Video_I2hf-H8Xumk_2_0001_smirk_out.npy"
smirk_rend_out = "/data/aujadhav/smirk/Video_I2hf-H8Xumk_2_0001_smirk_render_out.npy"
d4_out = "/data/aujadhav/champ/Video_I2hf-H8Xumk_2_0001_4d_out.npy"


reference_dict = np.load(str(reference_file), allow_pickle=True).item()
smirk_dict = np.load(str(smirk_out), allow_pickle=True).item()
smirk_render_dict = np.load(str(smirk_rend_out), allow_pickle=True).item()
d4_dict = np.load(str(d4_out), allow_pickle=True).item()

model_folder = "/data/aujadhav/smplx/models"
model_type = "smplx"
gender  = "neutral"
#ext = "npy"
ext='npz'

num_betas = len(reference_dict['smpls']['betas'][0])
num_expression_coeffs = len(smirk_dict['expression_params'][0])
betas = torch.Tensor(reference_dict['smpls']['betas'])
expression = torch.Tensor(smirk_dict['expression_params'])

model = smplx.create(model_folder, model_type=model_type,
                         gender=gender,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)

output = model(betas=betas, expression=expression,
                   return_verts=True)

vertices = output.vertices.detach().cpu().numpy().squeeze()
joints = output.joints.detach().cpu().numpy().squeeze()