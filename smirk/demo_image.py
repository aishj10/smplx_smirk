import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import os
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='samples/mead_90.png', help='Path to the input image/video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='trained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='output', help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_smirk_generator', action='store_true', help='Use SMIRK neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true', help='Present the result w.r.t. the original image/video size')

    args = parser.parse_args()

    image_size = 224
    
    print("===> args", args)
    # ----------------------- initialize configuration ----------------------- #
    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator

    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    flame = FLAME().to(args.device)
    renderer = Renderer().to(args.device)

    # check if input is an image or a video or webcam or directory

    os.makedirs(args.out_path, exist_ok = True)

    for image_name in sorted(os.listdir(args.input_path)):
    
        print(image_name)
        image = cv2.imread(os.path.join(args.input_path,image_name))
        
        orig_image_height, orig_image_width, _ = image.shape
    
        kpt_mediapipe = run_mediapipe(image)
    
        # crop face if needed
        if args.crop:
            if (kpt_mediapipe is None):
                print('Could not find landmarks for the image using mediapipe and cannot crop the face. Exiting...')
                exit()
            
            kpt_mediapipe = kpt_mediapipe[..., :2]
    
            tform = crop_face(image,kpt_mediapipe,scale=1.4,image_size=image_size)
            
            cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
    
            cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
            cropped_kpt_mediapipe = cropped_kpt_mediapipe[:,:2]
        else:
            cropped_image = image
            cropped_kpt_mediapipe = kpt_mediapipe
    
        
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224,224))
        cropped_image = torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
        cropped_image = cropped_image.to(args.device)
    
        outputs = smirk_encoder(cropped_image)
    
    
        flame_output = flame.forward(outputs)
        renderer_output = renderer.forward(flame_output['vertices'], outputs['cam'],
                                            landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
        
        rendered_img = renderer_output['rendered_img']
    
        #grid = torch.cat([cropped_image, rendered_img], dim=3)
        grid = rendered_img
    
        grid_numpy = grid.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0
        grid_numpy = grid_numpy.astype(np.uint8)
        grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)
  
    
        #image_name = args.input_path.split('/')[-1]
    
        dict_path = os.path.join(args.out_path, "smirk_param")
        os.makedirs(dict_path, exist_ok = True)
        img = image_name.split(".")[0]
        np.save( str(os.path.join(dict_path, f"{img}.npy")),
                        {k: v.detach().cpu().numpy() for k, v in outputs.items()})
    
        render_path = os.path.join(args.out_path, "smirk_render_param")
        os.makedirs(render_path, exist_ok = True)
        
        np.save( str(os.path.join(render_path, f"{img}_render.npy")),
                {k: v.detach().cpu().numpy() for k, v in renderer_output.items()})

        out_image_path = os.path.join(args.out_path, "smirk_images")
        os.makedirs(out_image_path, exist_ok = True)
        
        cv2.imwrite(f"{out_image_path}/{image_name}", grid_numpy)

