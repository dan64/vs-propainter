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
functions utility for the Vapoursynth wrapper of ProPainter.
"""

from __future__ import annotations
import math
import os
import logging
from PIL import Image
import numpy as np
import vapoursynth as vs
from functools import partial
from typing import Sequence

core = vs.core

_IMG_EXTENSIONS = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG',
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

    try:
        with Image.open(filename) as img:
            return img.format is not None
    except (OSError, ValueError):
        return False


def clip_crop(clip: vs.VideoNode, region: tuple[int, int, int, int]) -> vs.VideoNode:
    return clip.std.CropAbs(width=region[0], height=region[1], left=region[2], top=region[3])


def mask_overlay(
        base: vs.VideoNode,
        overlay: vs.VideoNode,
        x: int = 0,
        y: int = 0,
        mask: vs.VideoNode | None = None,
        opacity: float = 1.0,
        mode: str = 'normal',
        planes: (int | Sequence[int]) | None = None,
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

        if (mask.width != overlay.width or mask.height != overlay.height or
                mask.format.bits_per_sample != overlay.format.bits_per_sample):
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
        mask = overlay.std.BlankClip(
            format=overlay.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0), color=peak)
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


def scene_detect(clip: vs.VideoNode, threshold: float = 0.1, frequency: int = 0) -> vs.VideoNode:
    """
    Detect scene changes in a video clip while enforcing a minimum frame distance between detections.

    This function uses MiscFilters' SCDetect to identify potential scene changes, then suppresses
    any detected change that occurs within `frequency` frames of the previously accepted scene change.
    Frame 0 is always treated as a scene change (common convention for keyframe placement).

    note::
        This implementation uses internal mutable state to track the last accepted scene change frame.
        As such, it **assumes sequential, single-threaded frame access** (e.g., during linear encoding).
        It is **not safe for random access, multithreaded evaluation, or caching with reordering**.
        For robust use in complex scripts, consider precomputing scene changes externally.

    Parameters
    ----------
    clip : vs.VideoNode
        Input video clip. Should be a standard RGB or YUV clip (will be converted to GRAY8 internally).
    threshold : float, optional
        Sensitivity threshold for SCDetect (lower = more sensitive). Default is 0.1.
        If set to 0, the function returns the input clip unchanged.
    frequency : int, optional
        Minimum number of frames that must elapse between two accepted scene changes.
        If > 0, it is clamped to a minimum of 5 frames. Default is 0 (no spacing enforced).

    Returns
    -------
    vs.VideoNode
        Clip with `_SceneChangePrev` frame property set to 1 only on frames that are:
        - The first frame (n=0), OR
        - Detected as a scene change by SCDetect AND at least `frequency` frames after the last accepted change.

        All other frames have `_SceneChangePrev` set to 0. The `_SceneChangeNext` is also set to 0.

    Raises
    ------
    vs.Error
        If the 'MiscFilters' plugin (providing `misc.SCDetect`) is not installed or fails to load.

    Examples
    --------
    clip = scene_detect(clip, threshold=0.15, frequency=10) # Ensures at least 10 frames between scene changes
    """

    if threshold == 0:
        return clip

    if frequency > 0:
        frequency = max(5, frequency)  # Enforce minimum spacing of 5 frames

    # Convert to GRAY8 for SCDetect (required format)
    sc_clip = clip.resize.Point(format=vs.GRAY8, matrix_s="709")

    try:
        sc_clip = sc_clip.misc.SCDetect(threshold=threshold)
    except Exception as e:
        raise vs.Error(f"scene_detect: 'MiscFilters' plugin not installed or failed: {e}")

    # Use a mutable container to track last accepted scene change frame
    # Not thread-safe! Assumes sequential frame access.
    state = {"last_sc": -1000}  # Initialize far enough in the past to not block early scenes

    def enforce_min_distance(n: int, f: list[vs.VideoFrame], freq: int) -> vs.VideoFrame:
        src_frame, sc_frame = f[0], f[1]
        out = src_frame.copy()

        is_sc = False

        if n == 0:
            is_sc = True
        elif sc_frame.props.get("_SceneChangePrev", 0) == 1:
            if (n - state["last_sc"]) > freq:
                is_sc = True

        if is_sc:
            state["last_sc"] = n
            out.props["_SceneChangePrev"] = 1
            out.props['_SceneChangeNext'] = 0
        else:
            out.props["_SceneChangePrev"] = 0
            out.props['_SceneChangeNext'] = 0

        return out

    result = clip.std.ModifyFrame(clips=[clip, sc_clip], selector=partial(enforce_min_distance, freq=frequency))
    return result
