"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-05-26
version:
LastEditors: Dan64
LastEditTime: 2025-11-25
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
main Vapoursynth wrapper for ProPainter.
URL: https://github.com/sczhou/ProPainter
"""

from __future__ import annotations
import math
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs
from functools import partial
from vspropainter.propainter_render import ModelProPainterIn, ModelProPainterOut
from vspropainter.propainter_utils import *

__version__ = "1.2.3"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")


def propainter(
        clip: vs.VideoNode,
        length: int = 80,
        clip_mask: vs.VideoNode = None,
        img_mask_path: str = None,
        mask_dilation: int = 4,
        neighbor_length: int = 10,
        ref_stride: int = 10,
        raft_iter: int = 20,
        mask_region: tuple[int, int, int, int] = None,
        sc_threshold: float = 0.0,
        sc_min_freq: int = 0,
        model: int = 0,
        outpaint_size: tuple[int, int] = None,
        weights_dir: str = model_dir,
        enable_fp16: bool = True,
        device_index: int = 0,
        inference_mode: bool = False
) -> vs.VideoNode:
    """ProPainter: Improving Propagation and Transformer for Video Inpainting

    :param clip:            Clip to process. Only RGB24 "full range" format is supported.
    :param length:          Sequence length that the model processes (min. 12 frames). High values will
                            increase the inference speed but will increase also the memory usage. Default: 80
    :param clip_mask:       Clip mask, must be of the same size and length of input clip. Default: None
    :param img_mask_path:   Path of the mask image, must be of the same size of input clip: Default: None
    :param mask_dilation:   Mask dilation for video and flow masking. Default: 4
    :param neighbor_length: Length of local neighboring frames. Low values decrease the memory usage.
                            High values could help to improve the quality on fast moving sequences.
                            Default: 10
    :param ref_stride:      Stride of global reference frames. High values will allow to
                            reduce the memory usage and increase the inference speed, but could
                            affect the inference quality. Default: 10
    :param raft_iter:       Iterations for RAFT inference. Low values will increase the inference
                            speed but could affect the output quality. Default: 20
    :param mask_region:     Allow to restrict the region of the mask, format: (width, height, left, top).
                            The region must be big enough to allow the inference. Default: None
    :param model:           Model used by ProPainter to render the frames, available values are:
                                0: Inpainting Mask Mode (using img_mask/clip_mask)
                                1: Outpainting Mode (using outpaint_size)
                            Default: 0
    :param outpaint_size:   Size of extrapolated frames, format: (width, height). Default: None
    :param weights_dir:     Path string of location of model weights.
    :param sc_threshold:    If > 0 represent the scene change threshold used to generate the reference frames for
                            ProPainter, range [0,1]. Default = 0.0
    :param sc_min_freq:     Minimum number of frames that must elapse between two accepted scene changes.
                            If > 0, it is clamped to a minimum of 5 frames. Default is 0 (no spacing enforced).
    :param enable_fp16:     If True use fp16 (half precision) during inference. Default: fp16 (for RTX30 or above)
    :param device_index:    Device ordinal of the GPU (if = -1 CPU mode is enabled). Default: 0
    :param inference_mode:  Enable/Disable torch inference mode. Default: False
    """

    if model not in (0, 1):
        raise vs.Error("propainter: model must be 0 or 1")

    if model == 0:
        return propainter_inpaint(clip, length, clip_mask, img_mask_path, mask_dilation, neighbor_length, ref_stride,
                                  raft_iter, mask_region, sc_threshold, sc_min_freq, weights_dir, enable_fp16,
                                  device_index, inference_mode)
    else:
        return propainter_outpaint(clip, length, outpaint_size, mask_dilation, neighbor_length,
                                   ref_stride, raft_iter, weights_dir, enable_fp16, device_index, inference_mode)


# @torch.inference_mode()
def propainter_inpaint(
        clip: vs.VideoNode,
        length: int = 100,
        clip_mask: vs.VideoNode = None,
        img_mask_path: str = None,
        mask_dilation: int = 8,
        neighbor_length: int = 10,
        ref_stride: int = 10,
        raft_iter: int = 20,
        mask_region: tuple[int, int, int, int] = None,
        sc_threshold: float = 0.1,
        sc_min_freq: int = 0,
        weights_dir: str = model_dir,
        enable_fp16: bool = True,
        device_index: int = 0,
        inference_mode: bool = False
) -> vs.VideoNode:
    """ProPainter: Improving Propagation and Transformer for Video Inpainting

    :param clip:            Clip to process. Only RGB24 "full range" format is supported.
    :param length:          Sequence length that the model processes (min. 12 frames). High values will
                            increase the inference speed but will increase also the memory usage. Default: 100
    :param clip_mask:       Clip mask, must be of the same size and length of input clip. Default: None
    :param img_mask_path:   Path of the mask image, must be of the same size of input clip: Default: None
    :param mask_dilation:   Mask dilation for video and flow masking. Default: 8
    :param neighbor_length: Length of local neighboring frames. Low values decrease the memory usage.
                            High values could help to improve the quality on fast moving sequences.
                            Default: 10
    :param ref_stride:      Stride of global reference frames. High values will allow to
                            reduce the memory usage and increase the inference speed, but could
                            affect the inference quality. Default: 10
    :param raft_iter:       Iterations for RAFT inference. Low values will increase the inference
                            speed but could affect the output quality. Default: 20
    :param mask_region:     Allow to restrict the region of the mask, format: (width, height, left, top).
                            The region must be big enough to allow the inference. Default: None
    :param weights_dir:     Path string of location of model weights.
    :param sc_threshold:    If > 0 represent the scene change threshold used to generate the reference frames for
                            ProPainter, range [0,1]. Default = 0.1
    :param sc_min_freq:     Minimum number of frames that must elapse between two accepted scene changes.
                            If > 0, it is clamped to a minimum of 5 frames. Default is 0 (no spacing enforced).
    :param enable_fp16:     If True use fp16 (half precision) during inference. Default: fp16 (for RTX30 or above)
    :param device_index:    Device ordinal of the GPU (if = -1 CPU mode is enabled). Default: 0
    :param inference_mode:  Enable/Disable torch inference mode. Default: False
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("propainter: this is not a clip")

    if clip.format.id != vs.RGB24:
        raise vs.Error("propainter: only RGB24 format is supported")

    if not (clip_mask is None):
        if not isinstance(clip_mask, vs.VideoNode):
            raise vs.Error("propainter: clip_mask is not a clip")
        if clip_mask.num_frames != clip.num_frames:
            raise vs.Error("propainter: clip_mask must have the same length of clip")
        if clip_mask.width != clip.width:
            raise vs.Error(
                f"propainter: clip_mask must have the same width of clip -> {clip_mask.width} <> {clip.width}")
        if clip_mask.height != clip.height:
            raise vs.Error(
                f"propainter: clip_mask must have the same height of clip -> {clip_mask.height} <> {clip.height}")
        if clip_mask.format.id != vs.RGB24:
            raise vs.Error("propainter: only RGB24 clip_mask format is supported")

    if not (img_mask_path is None):
        if not is_img_file(img_mask_path):
            raise vs.Error("propainter: wrong image mask: " + img_mask_path)

    if (clip_mask is None) and (img_mask_path is None):
        raise vs.Error("propainter: a clip/image mask must be provided")

    if device_index != -1 and not torch.cuda.is_available():
        raise vs.Error("propainter: CUDA is not available")

    if length < 12:
        raise vs.Error("propainter: length must be at least 12")

    disable_warnings()

    if device_index == -1:
        device = torch.device("cpu")
        use_half = False
    else:
        device = torch.device("cuda", device_index)
        use_half = enable_fp16

    # enable torch inference mode
    # https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
    if inference_mode:
        torch.backends.cudnn.benchmark = True
        torch.inference_mode()

    # ----------------------------------------- INFERENCE -------------------------------------------------------------

    cache = {}

    def inference_img_mask(n: int, f: list[vs.VideoFrame], v_clip: vs.VideoFrame = None,
                           ppaint: ModelProPainterIn = None,
                           batch_size: int = 25, use_half: bool = False, sc_thresh: bool = False) -> vs.VideoFrame:

        if str(n) not in cache:
            cache.clear()
            # vs.core.log_message(2, "Init Cache at frame_n = " + str(n))

            frames = [frame_to_image(f[0])]

            for i in range(1, batch_size):

                if n + i >= v_clip.num_frames:
                    break

                frame_i = v_clip.get_frame(n + i)

                if sc_thresh:
                    is_scenechange = (frame_i.props['_SceneChangePrev'] == 1 and frame_i.props['_SceneChangeNext'] == 0)
                    is_scenechange = (len(frames) > 4) and is_scenechange
                else:
                    is_scenechange = False

                if is_scenechange:
                    # vs.core.log_message(2, "SceneDetect frame_n = " + str(n + i))
                    break

                frames.append(frame_to_image(frame_i))

            output = ppaint.get_unmasked_frames(video_frames=frames, batch_size=batch_size, use_half=use_half)

            for i in range(len(output)):
                cache[str(n + i)] = output[i]

        return np_array_to_frame(cache[str(n)], f[1].copy())

    def inference_clip_mask(n: int, f: list[vs.VideoFrame], v_clip: vs.VideoFrame = None, m_clip: vs.VideoFrame = None,
                            ppaint: ModelProPainterIn = None, batch_size: int = 25,
                            use_half: bool = False, sc_thresh: bool = False) -> vs.VideoFrame:

        if str(n) not in cache:
            cache.clear()
            # vs.core.log_message(2, "Init Cache at frame_n = " + str(n))

            frames = [frame_to_image(f[0])]
            mask_frames = [frame_to_image(f[2])]

            for i in range(1, batch_size):

                if n + i >= v_clip.num_frames:
                    break

                frame_i = v_clip.get_frame(n + i)

                if sc_thresh:
                    is_scenechange = (frame_i.props['_SceneChangePrev'] == 1 and frame_i.props['_SceneChangeNext'] == 0)
                    is_scenechange = (len(frames) > 4) and is_scenechange
                else:
                    is_scenechange = False

                if is_scenechange:
                    # vs.core.log_message(2, "SceneDetect frame_n = " + str(n + i))
                    break

                frames.append(frame_to_image(frame_i))
                mask_frames.append(frame_to_image(m_clip.get_frame(n + i)))

            output = ppaint.get_unmasked_frames(video_frames=frames, mask_frames=mask_frames, use_half=use_half)

            for i in range(len(output)):
                cache[str(n + i)] = output[i]

        return np_array_to_frame(cache[str(n)], f[1].copy())

    # ----------------------------------------- ModifyFrame -----------------------------------------------------------

    if sc_threshold > 0:
        clip = scene_detect(clip, threshold=sc_threshold, frequency=sc_min_freq)
        sc_thresh = True
    else:
        sc_thresh = False

    ppaint = ModelProPainterIn(device, weights_dir, img_mask_path, mask_dilation, neighbor_length,
                               ref_stride, raft_iter, (clip.width, clip.height))

    base = clip.std.BlankClip(width=clip.width, height=clip.height, keep=True)

    if clip_mask is None:
        if mask_region is None:
            clip_new = base.std.ModifyFrame(clips=[clip, base], selector=partial(inference_img_mask, v_clip=clip,
                                                                                 ppaint=ppaint, batch_size=length,
                                                                                 use_half=use_half,
                                                                                 sc_thresh=sc_thresh))
        else:
            ppaint.img_mask_crop(mask_region)
            base_c = clip_crop(base, mask_region)
            clip_c = clip_crop(clip, mask_region)
            v_cropped = base_c.std.ModifyFrame(clips=[clip_c, base_c],
                                               selector=partial(inference_img_mask, v_clip=clip_c,
                                                                ppaint=ppaint, batch_size=length, use_half=use_half,
                                                                sc_thresh=sc_thresh))
            clip_new = mask_overlay(clip, v_cropped, x=mask_region[2], y=mask_region[3])
    else:
        if mask_region is None:
            clip_new = base.std.ModifyFrame(clips=[clip, base, clip_mask],
                                            selector=partial(inference_clip_mask, v_clip=clip,
                                                             m_clip=clip_mask, ppaint=ppaint, batch_size=length,
                                                             use_half=use_half, sc_thresh=sc_thresh))
        else:
            base_c = clip_crop(base, mask_region)
            clip_mask_c = clip_crop(clip_mask, mask_region)
            clip_c = clip_crop(clip, mask_region)
            v_cropped = base_c.std.ModifyFrame(clips=[clip_c, base_c, clip_mask_c],
                                               selector=partial(inference_clip_mask,
                                                                v_clip=clip_c, m_clip=clip_mask_c, ppaint=ppaint,
                                                                batch_size=length, use_half=use_half,
                                                                sc_thresh=sc_thresh))
            clip_new = mask_overlay(clip, v_cropped, x=mask_region[2], y=mask_region[3])

    return clip_new


def propainter_outpaint(
        clip: vs.VideoNode,
        length: int = 50,
        outpaint_size: tuple[int, int] = None,
        mask_dilation: int = 0,
        neighbor_length: int = 10,
        ref_stride: int = 10,
        raft_iter: int = 20,
        weights_dir: str = model_dir,
        enable_fp16: bool = True,
        device_index: int = 0,
        inference_mode: bool = False
) -> vs.VideoNode:
    """ProPainter: Improving Propagation and Transformer for Video Outpainting

    :param clip:            Clip to process. Only RGB24 "full range" format is supported.
    :param length:          Sequence length that the model processes (min. 10 frames). High values will
                            increase the inference speed but will increase also the memory usage. Default: 50
    :param outpaint_size:   Size of extrapolated frames, format: (width, height). Default: None
    :param mask_dilation:   Mask dilation for video and flow masking. Default: 8
    :param neighbor_length: Length of local neighboring frames. Low values decrease the memory usage.
                            High values could help to improve the quality on fast moving sequences.
                            Default: 10
    :param ref_stride:      Stride of global reference frames. High values will allow to
                            reduce the memory usage and increase the inference speed, but could
                            affect the inference quality. Default: 10
    :param raft_iter:       Iterations for RAFT inference. Low values will increase the inference
                            speed but could affect the output quality. Default: 20
    :param weights_dir:     Path string of location of model weights.
    :param enable_fp16:     If True use fp16 (half precision) during inference. Default: fp16 (for RTX30 or above)
    :param device_index:    Device ordinal of the GPU (if = -1 CPU mode is enabled). Default: 0
    :param inference_mode:  Enable/Disable torch inference mode. Default: False
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("propainter: this is not a clip")

    if clip.format.id != vs.RGB24:
        raise vs.Error("propainter: only RGB24 format is supported")

    if (outpaint_size is None):
        raise vs.Error("propainter: please provide the outpainting size (width, height)")

    if outpaint_size[0] < clip.width and outpaint_size[1] < clip.height:
        raise vs.Error("propainter: outpainting size is lower than clip size")

    if device_index != -1 and not torch.cuda.is_available():
        raise vs.Error("propainter: CUDA is not available")

    if length < 10:
        raise vs.Error("propainter: length must be at least 10")

    disable_warnings()

    if device_index == -1:
        device = torch.device("cpu")
        use_half = False
    else:
        device = torch.device("cuda", device_index)
        use_half = enable_fp16

    # enable torch inference mode
    # https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
    if inference_mode:
        torch.backends.cudnn.benchmark = True
        torch.inference_mode()

    # ----------------------------------------- INFERENCE -------------------------------------------------------------

    cache = {}

    def inference_extrapolated_frames(n: int, f: list[vs.VideoFrame], v_clip: vs.VideoFrame = None,
                                      ppaint: ModelProPainterOut = None,
                                      batch_size: int = 25, use_half: bool = False) -> vs.VideoFrame:

        if str(n) not in cache:
            cache.clear()
            # vs.core.log_message(2, "Init Cache at frame_n = " + str(n))

            frames = [frame_to_image(f[0])]

            for i in range(1, batch_size):

                if n + i >= v_clip.num_frames:
                    break

                frame_i = v_clip.get_frame(n + i)

                frames.append(frame_to_image(frame_i))

            output = ppaint.get_extrapolated_frames(video_frames=frames, batch_size=batch_size, use_half=use_half)

            for i in range(len(output)):
                cache[str(n + i)] = output[i]

        return np_array_to_frame(cache[str(n)], f[1].copy())

    # ----------------------------------------- ModifyFrame -----------------------------------------------------------

    ppaint = ModelProPainterOut(device, weights_dir, mask_dilation, outpaint_size, neighbor_length,
                                ref_stride, raft_iter, (clip.width, clip.height))

    base = clip.std.BlankClip(width=outpaint_size[0], height=outpaint_size[1], keep=True)

    clip_new = base.std.ModifyFrame(clips=[clip, base],
                                    selector=partial(inference_extrapolated_frames,
                                                     v_clip=clip,
                                                     ppaint=ppaint,
                                                     batch_size=length,
                                                     use_half=use_half))
    return clip_new
