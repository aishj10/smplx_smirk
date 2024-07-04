# 💃 SMPL & Rendering

Try Champ with your dance videos! It may take time to setup the environment, follow the instruction step by step🐢, report issue when necessary. 
> Notice that it has been tested only on Linux. Windows user may encounter some environment issues for pyrender.


## Install dependencies

1. Install [CHAMP](https://github.com/fudan-generative-vision/champ)

   Create conda environment:

    ```bash
      conda create -n champ python=3.10
      conda activate champ
    ```
    
    Install packages with `pip`
    
    ```bash
      pip install -r requirements.txt
    ```
    
    Install packages with [poetry](https://python-poetry.org/)
    > If you want to run this project on a Windows device, we strongly recommend to use `poetry`.
    ```shell
    poetry install --no-root
    ```
    
    # Inference
    
    The inference entrypoint script is `${PROJECT_ROOT}/inference.py`. Before testing your cases, there are two preparations need to be completed:
    1. [Download all required pretrained models](#download-pretrained-models).
    2. [Prepare your guidance motions](#preparen-your-guidance-motions).
    2. [Run inference](#run-inference).
    
    ## Download pretrained models
    
    You can easily get all pretrained models required by inference from our [HuggingFace repo](https://huggingface.co/fudan-generative-ai/champ).
    
    Clone the the pretrained models into `${PROJECT_ROOT}/pretrained_models` directory by cmd below:
    ```shell
    git lfs install
    git clone https://huggingface.co/fudan-generative-ai/champ pretrained_models
    ```
    
    Or you can download them separately from their source repo:
       - [Champ ckpts](https://huggingface.co/fudan-generative-ai/champ/tree/main):  Consist of denoising UNet, guidance encoders, Reference UNet, and motion module.
       - [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5): Initialized and fine-tuned from Stable-Diffusion-v1-2. (*Thanks to runwayml*)
       - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse): Weights are intended to be used with the diffusers library. (*Thanks to stablilityai*)
       - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder): Fine-tuned from CompVis/stable-diffusion-v1-4-original to accept CLIP image embedding rather than text embeddings. (*Thanks to lambdalabs*)
    
    Finally, these pretrained models should be organized as follows:
    
    ```text
    ./pretrained_models/
    |-- champ
    |   |-- denoising_unet.pth
    |   |-- guidance_encoder_depth.pth
    |   |-- guidance_encoder_dwpose.pth
    |   |-- guidance_encoder_normal.pth
    |   |-- guidance_encoder_semantic_map.pth
    |   |-- reference_unet.pth
    |   `-- motion_module.pth
    |-- image_encoder
    |   |-- config.json
    |   `-- pytorch_model.bin
    |-- sd-vae-ft-mse
    |   |-- config.json
    |   |-- diffusion_pytorch_model.bin
    |   `-- diffusion_pytorch_model.safetensors
    `-- stable-diffusion-v1-5
        |-- feature_extractor
        |   `-- preprocessor_config.json
        |-- model_index.json
        |-- unet
        |   |-- config.json
        |   `-- diffusion_pytorch_model.bin
        `-- v1-inference.yaml
    ```

2. Install [4D-Humans](https://github.com/shubham-goel/4D-Humans)
    ```shell
    git clone https://github.com/shubham-goel/4D-Humans.git
    conda create --name 4D-humans python=3.10
    conda activate 4D-humans
    pip install -e 4D-Humans
    ```

    or you can install via pip by a simple command
    ```shell
    pip install git+https://github.com/shubham-goel/4D-Humans
    ```

3. Install [detectron2](https://github.com/facebookresearch/detectron2)
    
    gcc and g++ 12 is necessary to build detectron2

    ```shell
    conda install -c conda-forge gcc=12 gxx=12
    ```
    Then
    ```shell
    git clone https://github.com/facebookresearch/detectron2

    pip install -e detectron2
    ```
    or you can install via pip by a simple command
    ```shell
    pip install git+https://github.com/facebookresearch/detectron2
    ```

4. Install [Blender](https://www.blender.org/)

    You can download Blender 3.x version for your operation system from this url [https://download.blender.org/release/Blender3.6](https://download.blender.org/release/Blender3.6/).


    ```
5. Install [SMIRK](https://github.com/georgeretsi/smirk)

      ### Installation
    You need to have a working version of PyTorch and Pytorch3D installed. We provide a `requirements.txt` file that can be used to install the necessary dependencies for a Python 3.9 setup with CUDA 11.7:
    
    ```bash
    conda create -n smirk python=3.9
    pip install -r requirements.txt
    # install pytorch3d now
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html
    ```
    
    Then, in order to download the required models, run:
    
    ```bash
    bash quick_install.sh
    ```
    *The above installation includes downloading the [FLAME](https://flame.is.tue.mpg.de/) model. This requires registration. If you do not have an account you can register at [https://flame.is.tue.mpg.de/](https://flame.is.tue.mpg.de/)*
    
    This command will also download the SMIRK pretrained model which can also be found on [Google Drive](https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE/view?usp=sharing).




6. Install [SMPLX Transfer Model](https://github.com/vchoutas/smplx/tree/main/transfer_model)

    ### Requirements

    1. Install [mesh](https://github.com/MPI-IS/mesh)
    2. Start by cloning the SMPL-X repo:
    ```Shell 
    git clone https://github.com/vchoutas/smplx.git
    ```
    3. Run the following command to install all necessary requirements
    ```Shell
        pip install -r requirements.txt
    ```
    4. Install the Torch Trust Region optimizer by following the instructions [here](https://github.com/vchoutas/torch-trust-ncg)
    5. Install loguru
    6. Install open3d
    7. Install omegaconf
    
    
    ### Data
    
    Register on the [SMPL-X website](http://smpl-x.is.tue.mpg.de/), go to the
    downloads section to get the correspondences and sample data,
    by clicking on the *Model correspondences* button.
    Create a folder
    named `transfer_data` and extract the downloaded zip there. You should have the
    following folder structure now:
    
    ```bash
    transfer_data
    ├── meshes
    │   ├── smpl
    │   ├── smplx
    ├── smpl2smplh_def_transfer.pkl
    ├── smpl2smplx_deftrafo_setup.pkl
    ├── smplh2smpl_def_transfer.pkl
    ├── smplh2smplx_deftrafo_setup.pkl
    ├── smplx2smpl_deftrafo_setup.pkl
    ├── smplx2smplh_deftrafo_setup.pkl
    ├── smplx_mask_ids.npy
    ```


## Download models

1. [DWPose for controlnet](https://github.com/IDEA-Research/DWPose?tab=readme-ov-file#-dwpose-for-controlnet)

    First, you need to download our Pose model dw-ll_ucoco_384.onnx ([baidu](https://pan.baidu.com/s/1nuBjw-KKSxD_BkpmwXUJiw?pwd=28d7), [google](https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing)) and Det model yolox_l.onnx ([baidu](https://pan.baidu.com/s/1fpfIVpv5ypo4c1bUlzkMYQ?pwd=mjdn), [google](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view)), then put them into `${PROJECT_ROOT}/annotator/ckpts/`.


2. HMR2 checkpoints

    ```shell
    python -m scripts.pretrained_models.download --hmr2
    ```
3. Detectron2 model

    ```shell
    python -m scripts.pretrained_models.download --detectron2
    ```
4. SMPL model

    Please download the SMPL model from the official site [https://smpl.is.tue.mpg.de/download.php](https://smpl.is.tue.mpg.de/download.php).
    Then move the `.pkl` model to `4D-Humans/data`:
    ```shell
    mkdir -p 4D-Humans/data/
    mv basicModel_neutral_lbs_10_207_0_v1.0.0.pkl 4D-Humans/data/
   
    
5. To download the *SMPL-X* model go to [this project website](https://smpl-x.is.tue.mpg.de) and register to get access to the downloads section. 


## Produce motion data


1. Prepare video

    Prepare a "dancing" video, and use `ffmpeg` to split it into frame images:
    ```shell
    mkdir -p driving_videos/Video_1/images
    ffmpeg -i your_video_file.mp4 -c:v png driving_videos/Video_1/images/%04d.png
    ```

2. Fit SMPL

    Make sure you have splitted the video into frames and organized the image files as below:
    ```shell
    |-- driving_videos
        |-- your_video_1
            |-- images
                |-- 0000.png
                    ...
                |-- 0020.png
                    ...
        |-- your_video_2
            |-- images
                |-- 0000.png
                    ...
        ...

    |-- reference_imgs
        |-- images
            |-- your_ref_img_A.png
            |-- your_ref_img_B.png
                    ...
    ```

    Then run script below to fit SMPL on reference images and driving videos:

   ```shell
    ```
   ```shell
   conda activate 4D-humans
   cd champ
    ```

    ```shell
    python -m scripts.data_processors.smpl.generate_smpls --reference_imgs_folder reference_imgs --driving_video_path driving_videos/your_video_1 --device YOUR_GPU_ID
    ```

    Once finished, you can check `reference_imgs/visualized_imgs` to see the overlay results. To better fit some extreme figures, you may also append `--figure_scale ` to manually change the figure(or shape) of predicted SMPL, from `-10`(extreme fat) to `10`(extreme slim).



3. Generate SMIRK
    ```shell
    conda activate smirk
    cd smirk
    python demo_image.py  --input_path driving_videos/your_video_1/images/ --out_path driving_videos/your_video_1/images/  --checkpoint pretrained_models/SMIRK_em1.pt --crop
    
    ```
       


4. Smooth SMPL

    ```shell
    blender --background --python scripts/data_processors/smpl/smooth_smpls.py --smpls_group_path driving_videos/your_video_1/smpl_results/smpls_group.npz --smoothed_result_path driving_videos/your_video_1/smpl_results/smpls_group.npz
    ```
    Ignore the warning message like `unknown argument` printed by Blender. There is also a user-friendlty [CEB Blender Add-on](https://www.patreon.com/posts/ceb-4d-humans-0-102810302) to help you visualize it.



5. Transfer SMPL

    ```shell
    python -m scripts.data_processors.smpl.smplx_transfer --reference_path reference_imgs/smpl_results/your_ref_img_A.npy --driving_path driving_videos/your_video_1 --output_folder transferd_result --figure_transfer --view_transfer
    ```

    Append `--figure_transfer` when you want the result matches the reference SMPL's figure, and `--view_transfer` to transform the driving SMPL onto reference image's camera space.

6. Convert SMPL to SMPLX

   ```shell
   conda activate smplx
   cd smplx
    ```

    Modify config_files/smpl2smplx_celebv.yaml for appropriate data_folder and output_folder

    ```shell
    python -m transfer_model --exp-cfg config_files/smpl2smplx_celebv.yaml
    ```

7. Combine SMIRK and SMPLX

   ```shell
   cd champ
    ```

    ```shell
   python -m scripts.data_processors.smpl.combine_smirk_smplx --transfer_folder transferd_result/ --smirk_param_path driving_videos/your_video_1/smirk/smirk_param
   ```
   

8. Render SMPL via Blender

   ```shell
   conda activate 4D-humans
    ```

    ```shell
    blender scripts/data_processors/smpl/blend/smpl_rendering.blend --background --python scripts/data_processors/smpl/render_condition_maps_smplx.py --driving_path transferd_result/smplx_smirk_results --reference_path reference_imgs/images/your_ref_img_A.png
    ```

    This will rendering in CPU on default. Append `--device YOUR_GPU_ID` to select a GPU for rendering. It will skip the exsiting rendered frames under the `transferd_result`. Keep it in mind when you want to overwrite with new rendering results. Ignore the warning message like `unknown argument` printed by Blender.

9. Render DWPose
    Clone [DWPose](https://github.com/IDEA-Research/DWPose)

    DWPose is required by `scripts/data_processors/dwpose/generate_dwpose.py`. You need clone this repo to the specific directory `DWPose` by command below:

    ```shell
    git clone https://github.com/IDEA-Research/DWPose.git DWPose
    conda activate champ
    ```
    Then 
    ```shell
    python -m scripts.data_processors.dwpose.generate_dwpose --input transferd_result/normal --output transferd_result/dwpose
    ```

10. Run CHAMP Inference
    Modify configs/inference/inference.yaml with appropriate ref_image_path and guidance_data_folder

    ```bash
      python inference.py --config configs/inference/inference.yaml
    ```