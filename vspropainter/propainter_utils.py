"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-05-26
version:
LastEditors: Dan64
LastEditTime: 2024-05-31
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
functions utility for the Vapoursynth wrapper of ProPainter.
"""

from __future__ import annotations
import math
import os
import logging
from PIL import Image
import numpy as np
import vapoursynth as vs

core = vs.core

_IMG_EXTENSIONS = ['.png', '.PNG', 'jpg', 'JPG', '.jpeg', '.JPEG',
    '.ppm', '.PPM', '.bmp', '.BMP']

def frame_to_image(frame: vs.VideoFrame) -> Image:
    npArray = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return Image.fromarray(npArray, 'RGB')

def image_to_frame(img: Image, frame: vs.VideoFrame) -> vs.VideoFrame:
    npArray = np.array(img)
    [np.copyto(np.asarray(frame[plane]), npArray[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame

def frame_to_np_array(frame: vs.VideoFrame) -> np.ndarray:
    npArray = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return npArray

def np_array_to_frame(npArray: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    [np.copyto(np.asarray(frame[plane]), npArray[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame

def disable_warnings():
    logger_blocklist = [
        "matplotlib",
        "PIL",
    ]

    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

def is_img_file(filename: str = "") -> bool:

    if not os.path.isfile(filename):
        return False

    fname_ext = "." + filename.split(".")[1]

    return (fname_ext in _IMG_EXTENSIONS)

def clip_crop(clip: vs.VideoNode, region: tuple[int, int, int, int]) -> vs.VideoNode:
    return clip.std.CropAbs(width=region[0], height=region[1], left=region[2], top=region[3])

def mask_overlay(
    base: vs.VideoNode,
    overlay: vs.VideoNode,
    x: int = 0,
    y: int = 0,
    mask: Optional[vs.VideoNode] = None,
    opacity: float = 1.0,
    mode: str = 'normal',
    planes: Optional[Union[int, Sequence[int]]] = None,
    mask_first_plane: bool = True,
) -> vs.VideoNode:
    '''
    Puts clip overlay on top of clip base using different blend modes, and with optional x,y positioning, masking and opacity.

    Parameters:
    :param base:      This clip will be the base, determining the size and all other video properties of the result.
    :param overlay:   This is the image that will be placed on top of the base clip.
    :param x, y:      Define the placement of the overlay image on the base clip, in pixels. Can be positive or negative.
    :param mask:      Optional transparency mask. Must be the same size as overlay. Where mask is darker, overlay will be more transparent.
    :param opacity:   Set overlay transparency. The value is from 0.0 to 1.0, where 0.0 is transparent and 1.0 is fully opaque.
                      This value is multiplied by mask luminance to form the final opacity.
    :param mode:      Defines how your overlay should be blended with your base image. Available blend modes are:
                      normal, addition, average, difference, divide, exclusion, multiply, overlay, subtract.
    :param planes:    Specifies which planes will be processed. Any unprocessed planes will be simply copied.
    :param mask_first_plane: If true, only the mask's first plane will be used for transparency.
    '''
    if not (isinstance(base, vs.VideoNode) and isinstance(overlay, vs.VideoNode)):
        raise vs.Error('mask_overlay: this is not a clip')

    if mask is not None:
        if not isinstance(mask, vs.VideoNode):
            raise vs.Error('mask_overlay: mask is not a clip')

        if mask.width != overlay.width or mask.height != overlay.height or get_depth(mask) != get_depth(overlay):
            raise vs.Error('mask_overlay: mask must have the same dimensions and bit depth as overlay')

    if base.format.sample_type == vs.INTEGER:
        bits = base.format.bits_per_sample
        neutral = 1 << (bits - 1)
        peak = (1 << bits) - 1
        factor = 1 << bits
    else:
        neutral = 0.5
        peak = factor = 1.0

    plane_range = range(base.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    if base.format.subsampling_w > 0 or base.format.subsampling_h > 0:
        base_orig = base
        base = base.resize.Point(format=base.format.replace(subsampling_w=0, subsampling_h=0))
    else:
        base_orig = None

    if overlay.format.id != base.format.id:
        overlay = overlay.resize.Point(format=base.format)

    if mask is None:
        mask = overlay.std.BlankClip(format=overlay.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0), color=peak)
    elif mask.format.id != overlay.format.id and mask.format.color_family != vs.GRAY:
        mask = mask.resize.Point(format=overlay.format, range_s='full')

    opacity = min(max(opacity, 0.0), 1.0)
    mode = mode.lower()

    # Calculate padding sizes
    l, r = x, base.width - overlay.width - x
    t, b = y, base.height - overlay.height - y

    # Split into crop and padding values
    cl, pl = min(l, 0) * -1, max(l, 0)
    cr, pr = min(r, 0) * -1, max(r, 0)
    ct, pt = min(t, 0) * -1, max(t, 0)
    cb, pb = min(b, 0) * -1, max(b, 0)

    # Crop and padding
    overlay = overlay.std.Crop(left=cl, right=cr, top=ct, bottom=cb)
    overlay = overlay.std.AddBorders(left=pl, right=pr, top=pt, bottom=pb)
    mask = mask.std.Crop(left=cl, right=cr, top=ct, bottom=cb)
    mask = mask.std.AddBorders(left=pl, right=pr, top=pt, bottom=pb, color=[0] * mask.format.num_planes)

    if opacity < 1:
        mask = mask.std.Expr(expr=f'x {opacity} *')

    if mode == 'normal':
        pass
    elif mode == 'addition':
        expr = f'x y +'
    elif mode == 'average':
        expr = f'x y + 2 /'
    elif mode == 'difference':
        expr = f'x y - abs'
    elif mode == 'divide':
        expr = f'y 0 <= {peak} {peak} x * y / ?'
    elif mode == 'exclusion':
        expr = f'x y + 2 x * y * {peak} / -'
    elif mode == 'multiply':
        expr = f'x y * {peak} /'
    elif mode == 'overlay':
        expr = f'x {neutral} < 2 x y * {peak} / * {peak} 2 {peak} x - {peak} y - * {peak} / * - ?'
    elif mode == 'subtract':
        expr = f'x y -'
    else:
        raise vs.Error('mask_overlay: invalid mode specified')

    if mode != 'normal':
        overlay = core.std.Expr([overlay, base], expr=[expr if i in planes else '' for i in plane_range])

    # Return padded clip
    last = core.std.MaskedMerge(base, overlay, mask, planes=planes, first_plane=mask_first_plane)
    if base_orig is not None:
        last = last.resize.Point(format=base_orig.format)
    return last
