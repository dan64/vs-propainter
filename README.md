# ProPainter
Improving Propagation and Transformer for Video Inpainting using Vapoursynth, based on [ProPainter](https://github.com/sczhou/ProPainter).

The Vapoursynth filter version has the advantage of transforming the images directly in memory, without the need to use the filesystem to store the video frames. Using Vapoursynth the filter is faster and don't have any limitation on the number of frames that can be elaborated. 

In order to improve the inference this implementation includes *scene detection* in the frames provided in the inference batch. In case of a scene detection the first frame of the new scene is included in the next inference batch. The sensitivity of the scene detection can be defined using the parameter *sc_threshold* (see [Miscellaneous Filters](https://amusementclub.github.io/doc3/plugins/misc.html)).

## Dependencies
- [PyTorch](https://pytorch.org/get-started) 2.2.0 or later
- [VapourSynth](http://www.vapoursynth.com/) R68 or later
- [MiscFilters.dll](https://github.com/vapoursynth/vs-miscfilters-obsolete) Vapoursynth's Miscellaneous Filters


## Installation
```
pip install vspropainter-x.x.x-py3-none-any.whl
```
## Models Download
The models are not installed with the package, they must be downloaded from the ProPainter github site. 

The models to download are:

- [ProPainter.pth](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth)
- [raft-things.pth](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth)
- [recurrent_flow_completion.pth](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth)

The _model files_ have to be copied in the **weights** directory usually located in:

.\Lib\site-packages\vspropainter\weights

## Usage
```python
# adjusting color space to RGB24 (full range) for vsProPainter
clip = core.resize.Bicubic(clip=clip, format=vs.RGB24, matrix_in_s="709", range_s="full")
from vspropainter import propainter

# ProPainter using a mask image
clip = propainter(clip, img_mask_path="sample.png")

# ProPainter using a clip mask
clipMask = core.lsmas.LWLibavSource(source="sample_mask.mp4", format="RGB24", cache=0)
clip = propainter(clip, clip_mask=clipMask)

# ProPainter using a mask image region
clip = propainter(clip, img_mask_path="sample.png", mask_region=(460,280,68,28))

# ProPainter using outpainting
w = clip.width + 8
h = clip.high + 32
clip = propainter(clip, model = 1, length=50, mask_dilation=0, outpaint_size=(w, h))

```
See `__init__.py` for the description of the parameters.

## Memory optimization and inference speed-up

Video inpainting typically requires a significant amount of GPU memory. The filter offers various features that facilitate memory-efficient inference, effectively avoiding the _out of memory_ error. You can use the following options to reduce memory usage further:

- Reduce the number of local neighbors through decreasing the parameter *neighbor_length* (default 10). 
- Reduce the number of global references by increasing the parameter *ref_stride* (default 10).
- Set the parameter *enable_fp16* to **True** to use fp16 (half precision) during inference.
- Reduce the sequence's length of frames that the model processes, decreasing the parameter *length* (default 100).
 - Set a smaller mask region via the parameter *mask_region*. The mask region can be specified using a _tuple_ with the following format: (width, height, left, top). The reduction of the mask region will allow to speed up significantly the inference, expecially on HD movies, but the region must be big enough to allow the inference. In the case of bad output it will be necessary to increase its size.
 - Disable scene detection by setting *sc_threshold=0* (default 0.1).

With the only exception of parameter *length* the options to reduce the memory usage will allow also to speed up the inference's speed. 

In the case the mask will not be able to totally remove the masked object it is possible to increase the parameter *mask_dilation* to extend the mask's size (default 8).




