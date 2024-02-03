## Installation
OVTrack builds upon mmdetection framework. 
Please install following packages.
We use python 3.7, pytorch 1.10.0 and cuda 11.3 for our experiments.

### Requirements

Please install those packages following their official installation guide.
- [pytorch >= 1.10](https://pytorch.org/get-started/locally/)
- [mmcv == 1.4.4](https://github.com/open-mmlab/mmcv)
- [mmdetection == 2.23](https://github.com/open-mmlab/mmcv)
- [CLIP](https://github.com/openai/CLIP)
- [diffusers == 0.4.0](https://github.com/huggingface/diffusers)
- [tao](https://github.com/TAO-Dataset/tao/tree/master)

Install other dependencies using following command.
```bash
pip install -r requirements.txt
```

### Install TETA Metric

You can use following command to install TETA metric.
```bash
 python -m pip install git+https://github.com/SysCV/tet.git/#subdirectory=teta 
```
Please refer to [TETA](https://github.com/SysCV/tet/tree/main/teta) for the details of the metric.



