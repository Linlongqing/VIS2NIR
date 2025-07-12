

# VIS2NIR
### 描述
  图片生成基于[pix2pixHD](https://github.com/NVIDIA/pix2pixHD)实现，本项目重点内容是如何利用大量公开的
可见光人脸数据集和pix2pixHD生成合格的红外图像。之前尝试过CycleGAN合成，最终测试效果不佳，利用pix2pixHD获得了
良好的效果，基于mobilenet+arcface在Casia_NIR数据集上通过率95%@0.0001提升到了99.5%@0.0001，证明了用合成近红
人脸数据的有效性。对于跨模态识别，作者还未进行验证。pix2pixHD训练的重点在于像素对齐，但是一般VIS-NIR图像是由双
目摄像头拍摄，导致无法完全对齐，这里使用81点人脸关键点进行对齐（用68点同样可行，作者手上有很好的81点关键点模型），
并过滤掉对齐误差较大的图像对。如果有能力过滤图像质量较差的图像则是更好的选择。

## Prerequisites
- Linux or macOS
- Python 2 or 3
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install python libraries [dominate](https://github.com/Knio/dominate).
```bash
pip install dominate
```
- Clone this repo:


### Testing
```bash
python ./script/generate_train_data.py --name rgb2ir --no_instance
```

### Training
- Train a model at 256 x 256 resolution:
```bash
python train.py --name rgb2ir --no_instance --continue_train
```

### Multi-GPU training
- Train a model using multiple GPUs:
```bash
python train.py --name rgb2ir --no_instance --continue_train --batchSize 8 --gpu_ids 0,1,2,3,4,5,6,7
```
Note: this is not tested and we trained our model using single GPU only. Please use at your own discretion.

### Dataset
训练数据集VIS_NIR：
可以使用CASIA VIS-NIR 2.0数据等公开数据集用于训练，对齐的数据将后续提供。
生成NIR数据集：
对于亚洲人脸，可以用[Asian-Celeb](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)作为VIS数据，作者的对齐方式和glint-asia一致，无需再次进行对齐。作者暂不提供合成好的数据。

作者使用了部分私有数据集，测试结果上可能会有些差别。

### Tips    
    如果是用于人脸识别，足够的类别和每个类别数据进行均衡对于提点是必要的。

## Citation
If you find this useful for your research, please use the following.

```
@inproceedings{wang2018pix2pixHD,
  title={High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs},
  author={Ting-Chun Wang and Ming-Yu Liu and Jun-Yan Zhu and Andrew Tao and Jan Kautz and Bryan Catanzaro},  
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

## Acknowledgments
This code borrows heavily from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).

## Author
635496116@qq.com