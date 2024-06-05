# EffiVED:Efficient Video Editing via Text-instruction Diffusion Models

Arxiv Link: https://arxiv.org/abs/2403.11568


|Origin Videos & Editing Videos| Instrctuion|
| --------------- | --------------- |
|  ![Input image](assets/1.gif)  |  Turn the rabbit into a fox.|
|  ![Input image](assets/2.gif)  |   make it Van Gogh style|
|  ![Input image](assets/3.gif)  |  make it a white fox in the desert trail|
|  ![Input image](assets/4.gif)  |   make it snowy|
|  ![Input image](assets/5.gif)  |   add a flock of flowers flying.|

## News

**2024.6.5**: Release the inference code

## To Do List
Release the training dataset and code

## Getting Started
This repository is based on [I2VGen-XL](https://github.com/ali-vilab/i2vgen-xl).

### Create Conda Environment (Optional)
It is recommended to install Anaconda.

**Windows Installation:** https://docs.anaconda.com/anaconda/install/windows/

**Linux Installation:** https://docs.anaconda.com/anaconda/install/linux/

```bash
conda create -n animation python=3.10
conda activate animation
```

### Python Requirements
```bash
pip install -r requirements.txt
```

## Running inference
Please download the [pretrained model](https://cloudbook-public-production.oss-cn-shanghai.aliyuncs.com/animation/non_ema_00011000.pth) to checkpoints, then modify the test_model with your download model name. You should add your test videos and edited instruction like provided in data/test_list.txt. Then run the following command:
```bash
python inference.py --cfg configs/effived_infer.yaml
```



## Training

### Obtaining data from image editing datasets.

You can run the following command to generate the video editing pairs:

```bash
python scripts/img2seq_augmenter.py
```
Here we provide a demo to generate the data from MagicBrush. You can download this dataset following this [MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush).

### Obtaining data from narrow videos.

You can automatically caption the videos using the [Video-BLIP2-Preprocessor Script](https://github.com/ExponentialML/Video-BLIP2-Preprocessor) and set the dataset_types and json_path like this:
```
  - dataset_types: 
      - video_blip
    train_data:
      json_path: 'blip_generated.json'
```
Then generate the instruction using the code provided in [InstructPix2pix](https://github.com/timothybrooks/instruct-pix2pix) and generate the editing videos using [CoDeF](https://github.com/qiuyu96/CoDeF).




## Bibtex
Please cite this paper if you find the code is useful for your research:
```
@misc{zhang2024effived,
      title={EffiVED:Efficient Video Editing via Text-instruction Diffusion Models}, 
      author={Zhenghao Zhang and Zuozhuo Dai and Long Qin and Weizhi Wang},
      year={2024},
      eprint={2403.11568},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Shoutouts

- [I2VGen-XL](https://github.com/ali-vilab/i2vgen-xl)
- [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix
- [CoDeF](https://github.com/qiuyu96/CoDeF)

