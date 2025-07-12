<img src='imgs/teaser_720.gif' align="right" width=360>

<br><br><br><br>

# pix2pixHD
### [Project](https://tcwang0509.github.io/pix2pixHD/) | [Paper](https://arxiv.org/pdf/1711.11585.pdf) <br>
Pytorch implementation of our method for high-resolution (e.g. 2048x1024) photorealistic image-to-image translation. It can be used for turning semantic label maps into photo-realistic images or synthesizing portraits from face label maps. <br><br>
[High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://tcwang0509.github.io/pix2pixHD/)  
 [Ting-Chun Wang](https://tcwang0509.github.io/)<sup>1</sup>, [Ming-Yu Liu](http://mingyuliu.net/)<sup>1</sup>, [Jun-Yan Zhu](http://people.eecs.berkeley.edu/~junyanz/)<sup>2</sup>, Andrew Tao<sup>1</sup>, [Jan Kautz](http://jankautz.com/)<sup>1</sup>, [Bryan Catanzaro](http://catanzaro.name/)<sup>1</sup>  
 <sup>1</sup>NVIDIA Corporation, <sup>2</sup>UC Berkeley  
 In CVPR 2018.

## Image-to-image translation at 2k/1k resolution


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
The test results will be saved to a html file here: `./results/label2city_1024p/test_latest/index.html`.

More example scripts can be found in the `scripts` directory.


### Dataset


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
