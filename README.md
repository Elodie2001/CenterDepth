# Depth as points: Center Point-based Depth Estimation
Depth estimation using target center points. 

Virtual Datasets: virDepth.

> [**Depth as points**](https://arxiv.org/abs/2504.18773)
> 
> coming soon
> 
> [**Download virDepth**](https://pan.baidu.com/s/1NRKlb4PAvrViTD7QrsaERQ)
> 
> Password:r5by
> 
> Downlod Town_1 section directly into ./data


## Requirements
~~~
torch
torchaudio
torchvision
dcnv2
~~~

## Training

Download virDepth into ./data
~~~
main.py ctdet --exp_id [export dir name] --arch [disdlaconv2d_34, hourglass, resdcn_18, res_18] --gpu 0
~~~

## Testing
~~~
test.py ctdet --exp_id [export dir name] --load_model [model path] --arch [disdlaconv2d_34, hourglass, resdcn_18, res_18] --gpu 0
~~~
