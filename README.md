# FLAME: Free-form Language-based Motion Synthesis & Editing

[[Project Page]](https://kakaobrain.github.io/flame/) | [[Paper]](https://arxiv.org/abs/2209.00349) | [[Video]](https://youtu.be/LbPNGv0zrto)

Official Implementation of the paper ***FLAME: Free-form Language-based Motion Synthesis & Editing (AAAI'23)*** 

## Generated Samples

<img src="https://user-images.githubusercontent.com/10102721/204811388-748bbe11-bb0f-489b-a532-c668023c22b4.gif" width="640" height="360"/>

## Environment

This project is tested on the following environment. Please install them on your running environment.

* Python 3.8
* [PyTorch](https://pytorch.org/): 1.11
* [PyTorch3D](https://pytorch3d.org/): >=0.7.0

## Prerequisites

### Packages

You may need following packages to run this repo.

```bash
apt install libboost-dev libglfw3-dev libgles2-mesa-dev freeglut3-dev libosmesa6-dev libgl1-mesa-glx 
```

### Dataset

> :exclamation: We cannot directly provide original data files to abide by the license.


<details>
<summary>AMASS Dataset</summary>

Visit https://amass.is.tue.mpg.de/ to download AMASS dataset. We used **SMPL+H G** of following datasets in AMASS:

* ACCAD
* BMLhandball
* BMLmovi
* BMLrub
* CMU
* DanceDB
* DFaust
* EKUT
* EyesJapanDataset
* HDM05
* Human4D
* HumanEva
* KIT
* Mosh
* PosePrior
* SFU
* SSM
* TCDHands
* TotalCapture
* Transitions

Downloaded data are compressed in `bz2` format. All downloaded files need to be located at `data/amass_download_smplhg` .
</details>


<details>
<summary>BABEL</summary>

Visit https://babel.is.tue.mpg.de/ to download BABEL dataset. At the time of experiment, we used `babel_v1.0_release` .
BABEL dataset should be loacated at `data/babel_v1.0_release` . File structures under `data/babel_v1.0_release` looks like:

```
.
├── extra_train.json
├── extra_val.json
├── test.json
├── train.json
└── val.json
```

</details>

<details>
<summary>HumanML3D</summary>

You can access full HumanML3D dataset at [HumanML3D](https://github.com/EricGuo5513/HumanML3D). However, we used original AMASS SMPL data instead of a customized rig. What you will need to prepare to run this repo is:

```
./data/HumanML3D/
├── humanact12
├── HumanML3D.csv
├── test.txt
├── texts.zip
├── train.txt
└── val.txt
```

Note that the files above are located at: `data/HumanML3D/`. Please download [`humanact12`](https://twg.kakaocdn.net/brainrepo/models/FLAME/HumanML3D/e3a6de68c95a042fbc0abba7a0222d58/humanact12_processed.pkl) and [`HumanML3D.csv`](https://twg.kakaocdn.net/brainrepo/models/FLAME/HumanML3D/ef9f9ed8a99e3cd20226e454f9c4e2f7/HumanML3D.csv). You can download other files from the original repo.
</details>

### SMPL & DMPL Models

You may need SMPL and DMPL to preprocess motion data. Please refer to [AMASS](https://github.com/nghorbani/amass) for this. `smpl_model` and `dmpl_model` should be located in the project root directory.

<details>
<summary>SMPL</summary>

```
smpl_model/
├── female
│   └── model.npz
├── info.txt
├── LICENSE.txt
├── male
│   └── model.npz
└── neutral
    └── model.npz
```
</details>

<details>
<summary>DMPL</summary>

```
dmpl_model/
├── female
│   └── model.npz
├── LICENSE.txt
├── male
│   └── model.npz
└── neutral
    └── model.npz
```
</details>


### External Sources

You may need the following packages for visualization.

* [VPoser](https://github.com/nghorbani/human_body_prior)
* [PyOpenGL and PyOpenGL_Accelerate](https://github.com/mcfletch/pyopengl)

## Installation

1. Create a virtual environment and activate it.
    ```bash
    conda create -n flame python=3.8
    conda activate flame
    ```

2. Install required packages. Recommend to install corresponding version of PyTorch and PyTorch3D first.
    ```bash
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"  # PyTorch3D
    ```

3. Install [VPoser](https://github.com/nghorbani/human_body_prior) and [PyOpenGL and PyOpenGL_Accelerate](https://github.com/mcfletch/pyopengl) from their installation guide.

4. Install other required packages.
    ```bash
    pip install -r requirements.txt
    ```

## Preprocessing

1. Preprocess AMASS dataset.
    ```bash
    ./scripts/unzip_dataset.sh
    ```
    This will unzip downloaded AMASS data into data/amass_smplhg. You can also unzip data manually.

2. Prepare HumanML3D dataset.
    ```bash
    python scripts/prepare_humanml3d.py
    ```

3. Prepare BABEL dataset.
    ```bash
    python scripts/prepare_babel_dataset.py
    ```

## Training

You can train your own model by running the following command.
Training configs can be set by config files in `configs/` or by command line arguments (hydra format).

```bash
python train.py
```

## Testing

***Testing takes a long time, since it needs to generate all samples in testset.*** You need to run `test.py` with proper config settings at `configs/test.yaml`. Then, you can run `eval_util.py` to evaluate the results.

## Sampling

### Text-to-Motion Generation

Set your sampling config at `configs/t2m_sample.yaml`. Sampled results will be saved at `outputs/`. You can export `json` output to visualize in Unity Engine. Exported `json` includes the root joint's position and rotation of all other joints in quaternion format.

```bash
python t2m_sample.py
```

### Text-to-Motion Editing

Set your text-to-motion editing config at `configs/edit_motion.yaml`. You can choose a motion to be edited, editing joints, and text prompt. Sampled results will be saved at `outputs/`.

```bash
python edit_motion.py
```

<details>
<summary>Joint Index</summary>

* 00: Pelvis
* 01: L_Hip
* 02: R_Hip
* 03: Spine1
* 04: L_Knee
* 05: R_Knee
* 06: Spine2
* 07: L_Ankle
* 08: R_Ankle
* 09: Spine3
* 10: L_Foot
* 11: R_Foot
* 12: Neck
* 13: L_Collar
* 14: R_Collar
* 15: Head
* 16: L_Shoulder
* 17: R_Shoulder
* 18: L_Elbow
* 19: R_Elbow
* 20: L_Wrist
* 21: R_Wrist
* 22: L_Hand
* 23: R_Hand

</details>


## Pretrained Weights

### HumanML3D
* [Model](https://twg.kakaocdn.net/brainrepo/models/FLAME/weights/eefcd30a4138bf74fbb6d10b7731abe9/flame_hml3d_bc.ckpt) / [mCLIP](https://twg.kakaocdn.net/brainrepo/models/FLAME/weights/5d1aee3a89f046f9b7ec95ecbbd59b04/flame_mclip_hml3d_bc.ckpt)

### BABEL
* [Model](https://twg.kakaocdn.net/brainrepo/models/FLAME/weights/6ee93b403203cb41bd8ee9f4a7c9bdb2/flame_babel_bc.ckpt) / [mCLIP](https://twg.kakaocdn.net/brainrepo/models/FLAME/weights/424f0c9ba8e7641e0d0406134c13ad97/flame_mclip_babel_bc.ckpt)

## Citation

```
@article{kim2022flame,
  title={Flame: Free-form language-based motion synthesis \& editing},
  author={Kim, Jihoon and Kim, Jiseob and Choi, Sungjoon},
  journal={arXiv preprint arXiv:2209.00349},
  year={2022}
}
```

## License

Copyright (c) 2022 Korea University and Kakao Brain Corp. All Rights Reserved. Licensed under the Apache License, Version 2.0. (see [LICENSE](./LICENSE) for details)
