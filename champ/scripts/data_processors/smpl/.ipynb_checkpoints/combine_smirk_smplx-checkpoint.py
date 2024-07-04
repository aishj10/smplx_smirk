from smplx import build_layer
from omegaconf import OmegaConf
import numpy as np
import torch
import os, sys

from pathlib import Path
import argparse
import torch
import numpy as np
from tqdm import tqdm

import ipdb



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference SMPL with 4D-Humans")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--transfer_folder",
        type=str,
        default="transfer_new",
        help="Folder path to reference imgs",
    )

    parser.add_argument(
        "--smirk_param_path",
        type=str,
        default="driving_videos/Video_ID/smirk/smirk_param",
        help="Folder path to reference imgs",
    )

    parser.add_argument(
        "--model_config",
        type=str,
        default="/data/aujadhav/smplx/config_files/smpl2smplx_celebv_all.yaml",
        help="Folder path to reference imgs",
    )

    args = parser.parse_args()

    exp_cfg = OmegaConf.load(args.model_config)
    model_path = exp_cfg.body_model.folder

    body_model = build_layer(model_path,num_expression_coeffs=50, **exp_cfg.body_model)
    body_model = body_model.to(args.device)
   
    smpl_paths = [
            path for path in os.listdir(os.path.join(args.transfer_folder, "smpl_results") )
        ]
    smpl_paths.sort(key=lambda x: int(x.split(".")[0]))

    os.makedirs(os.path.join(args.transfer_folder, "smplx_smirk_results"), exist_ok=True)
   
    for img_path in tqdm(smpl_paths):
        
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        #print(img_path, img_fn)
        smplx_file = str(os.path.join(args.transfer_folder, "smplx_results", f"{img_fn}_para.npy"))
        
        if not os.path.exists(smplx_file):
            continue
        smplx_dict = np.load(str(smplx_file), allow_pickle=True).item()

        smpl_file = str(os.path.join(args.transfer_folder, "smpl_results", f"{img_fn}.npy"))
        result_dict = np.load(str(smpl_file), allow_pickle=True).item()
        
        smirk_file = str(os.path.join(args.smirk_param_path, f"{img_fn}.npy"))
        smirk_dict = np.load(str(smirk_file), allow_pickle=True).item()

        #print(f"Processing {img_fn}.npy")
        
        expression = torch.Tensor(smirk_dict['expression_params']).to(args.device)
        para_dict_new = {k : torch.Tensor(smplx_dict[k]).to(args.device) for k in ['betas','global_orient', 'body_pose']}        
        para_dict_new['expression'] = expression

        # print([[k,v.shape] for k, v in para_dict_new.items()])
        # print("body_model",body_model)
    
        #ipdb.set_trace()
        
        body_model_output = body_model(
                return_full_pose=True, get_skin=True, **para_dict_new)
    
        vertices = body_model_output.vertices
        
        result_dict["verts"][0] = (
                    vertices.detach().cpu().numpy()
                )
        result_dict['transl'] = smplx_dict['transl'].detach().cpu().numpy()
        np.save(
                    str(
                        os.path.join(
                            args.transfer_folder, "smplx_smirk_results", f"{img_fn}.npy"
                        )
                    ),
                    result_dict,
                )
        