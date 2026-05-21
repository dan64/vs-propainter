"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-05-26
version:
LastEditors: Dan64
LastEditTime: 2026-05-20
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
from typing import Sequence, NamedTuple

core = vs.core

_IMG_EXTENSIONS = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG',
                   '.ppm', '.PPM', '.bmp', '.BMP']


# Using NamedTuple for better compatibility with VapourSynth's typical usage patterns
class ClipInfo(NamedTuple):
    clip_orig: vs.VideoNode | None
    format_id: int
    color_family: vs.ColorFamily
    bits_per_sample: int
    matrix: int | None
    color_range: int | None
    chroma_resize: bool
    # luma protect fields
    clip_high_bitdepth: vs.VideoNode | None = None  # YUV444PS reference for luma restoration
    preserve_luma: bool = False
    luma_blend: float = 1.0  # 1.0 = 100% original luma, 0.0 = 100% luma


VIDEO_EXTENSIONS = ['.mpg', '.mp4', '.m4v', '.avi', '.mkv', '.mpeg']

# map integer _Matrix to zimg string
MATRIX_INT_TO_STR = {
    0: "rgb",
    1: "709",
    4: "fcc",
    5: "470bg",
    6: "170m",
    7: "240m",
    8: "ycgco",
    9: "2020ncl",
    10: "2020cl",
}

# bit depth threshold for automatic luma preservation
LUMA_PROTECT_MIN_BITDEPTH = 10

def _matrixIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_Matrix', None)

    # Non specificato o riservato
    if value in (None, 2, 3):
        return True

    # Non un membro valido dell'enum
    if value not in vs.MatrixCoefficients.__members__.values():
        return True

    # Coerenza con il color family
    if clip.format.color_family == vs.RGB and value != 0:
        return True  # RGB deve avere _Matrix=0
    if clip.format.color_family in (vs.YUV, vs.GRAY) and value == 0:
        return True  # YUV/GRAY non può avere _Matrix=RGB

    return False

def _rangeIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    if vs.core.core_version.release_major < 74:
        value = frame.props.get('_ColorRange', None)
        return value is None or value not in vs.ColorRange.__members__.values()
    else:
        value = frame.props.get('_Range', None)
        return value is None or value not in vs.Range.__members__.values()

def _get_matrix_str(clip: vs.VideoNode, default: str = "709") -> str:
    """Read _Matrix from frame props and return the equivalent zimg string."""
    if _matrixIsInvalid(clip):
        return default
    matrix_val = clip.get_frame(0).props.get('_Matrix')
    return MATRIX_INT_TO_STR.get(int(matrix_val), default)


def _matrix_int_to_str(matrix_val, default: str = "709") -> str:
    """Convert an integer/enum _Matrix value to the zimg string."""
    if matrix_val is None:
        return default
    return MATRIX_INT_TO_STR.get(int(matrix_val), default)


def _should_preserve_luma(clip: vs.VideoNode, preserve_luma: bool | None) -> bool:
    """
    Decide whether to enable luma preservation.
    - If preserve_luma is explicitly True/False, honor it.
    - If None (auto), enable only for high bit-depth sources (>= 10-bit) that are YUV/RGB.
    """
    if preserve_luma is not None:
        return preserve_luma
    fmt = clip.format
    if fmt is None:
        return False
    if fmt.bits_per_sample < LUMA_PROTECT_MIN_BITDEPTH:
        return False
    # GRAY: no chroma to colorize-and-merge, luma protect is meaningless
    if fmt.color_family == vs.GRAY:
        return False
    return True


def _build_high_bitdepth_reference(clip: vs.VideoNode) -> vs.VideoNode | None:
    """
    Build a YUV444PS reference clip from the original high-bitdepth input.
    Used later to extract the original Y plane in restore_format().
    Returns None if the input can't be converted.
    """
    fmt = clip.format
    if fmt is None:
        return None

    if fmt.color_family == vs.YUV:
        # YUV → YUV444PS: chroma upsampling + float promotion.
        # No matrix conversion needed (stays YUV).
        return vs.core.resize.Bicubic(clip, format=vs.YUV444PS)

    if fmt.color_family == vs.RGB:
        # RGB → YUV444PS: needs a matrix. Use BT.709 as a conventional choice
        # for HD content; this matrix is only used to derive Y for protection.
        return vs.core.resize.Bicubic(
            clip,
            format=vs.YUV444PS,
            matrix_s="709",
            range_in_s="full",
            range_s="limited",
        )

    return None


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
function to convert video clip to RGB24 format and to restore the original format
"""

def convert_format_RGB24(
    clip: vs.VideoNode,
    chroma_resize: bool = False,
    preserve_luma: bool | None = None,
    luma_blend: float = 1.0,
) -> tuple[vs.VideoNode, ClipInfo]:
    """
    Convert any clip to RGB24 (8-bit full-range RGB).

    :param clip:           input clip (YUV/RGB/GRAY, any bit depth)
    :param chroma_resize:  if True, resize to a chroma-friendly size
    :param preserve_luma:  if True, keep a YUV444PS copy of the source so the
                           original high-bitdepth luma can be restored in
                           restore_format(). If None (default), auto-enable
                           for sources with bit depth >= 10.
    :param luma_blend:     blend factor for luma restoration in [0.0, 1.0].
                           1.0 = use 100% original luma (default),
                           0.0 = use 100% produced luma.
    :return: (rgb24_clip, ClipInfo)
    """

    # Store original clip information before any processing
    original_format = clip.format
    if original_format is None:
        vs.Error("Clip must have a defined format")

    if not isinstance(clip, vs.VideoNode):
        vs.Error("convert_format_RGB24: Input is not a valid clip.")

    # Decide luma protect and build the reference clip if needed
    do_preserve_luma = _should_preserve_luma(clip, preserve_luma)
    high_bd_clip = _build_high_bitdepth_reference(clip) if do_preserve_luma else None
    # If reference build failed, disable luma protect rather than crash later
    if do_preserve_luma and high_bd_clip is None:
        do_preserve_luma = False
    # Clamp blend factor
    luma_blend = max(0.0, min(1.0, float(luma_blend)))

    # Fast path: already RGB24, assume was produced in RGB24 it
    if clip.format.id == vs.RGB24:
        if vs.core.core_version.release_major < 74:
            clip_color_range = vs.ColorRange(vs.RANGE_FULL)
        else:
            clip_color_range = vs.Range(vs.RANGE_FULL)

        clip_info = ClipInfo(
            clip_orig=clip if chroma_resize else None,
            format_id=original_format.id,
            color_family=original_format.color_family,
            bits_per_sample=original_format.bits_per_sample,
            matrix=vs.MatrixCoefficients(vs.MATRIX_RGB),
            color_range=clip_color_range,
            chroma_resize=chroma_resize,
            clip_high_bitdepth=high_bd_clip,
            preserve_luma=do_preserve_luma,
            luma_blend=luma_blend,
        )
        if chroma_resize:
            clip = resize_min_HW(clip)
        return clip, clip_info

    # Not RGB24: normalize missing color metadata to sane defaults
    if _matrixIsInvalid(clip):
        clip = clip.std.SetFrameProps(_Matrix=vs.MATRIX_BT709)
    if _rangeIsInvalid(clip):
        if vs.core.core_version.release_major < 74:
            clip = clip.std.SetFrameProps(_ColorRange=vs.RANGE_LIMITED)
        else:
            clip = clip.std.SetFrameProps(_Range=vs.RANGE_LIMITED)

    # Read props after normalization, so ClipInfo reflects the actual values
    frame = clip.get_frame(0)
    props = frame.props

    if vs.core.core_version.release_major < 74:
        clip_color_range = vs.ColorRange(props.get('_ColorRange', vs.RANGE_LIMITED.value))
    else:
        clip_color_range = vs.Range(props.get('_Range', vs.RANGE_LIMITED.value))

    clip_info = ClipInfo(
        clip_orig=clip if chroma_resize else None,
        format_id=original_format.id,
        color_family=original_format.color_family,
        bits_per_sample=original_format.bits_per_sample,
        matrix=vs.MatrixCoefficients(props.get('_Matrix', vs.MATRIX_BT709.value)),
        color_range=clip_color_range,
        chroma_resize=chroma_resize,
        clip_high_bitdepth=high_bd_clip,
        preserve_luma=do_preserve_luma,
        luma_blend=luma_blend,
    )

    if chroma_resize:
        clip = resize_min_HW(clip)

    # Ensure we're working with 8-bit before the RGB conversion
    if clip.format.bits_per_sample != 8:
        clip = vs.core.resize.Bicubic(clip, format=clip.format.replace(bits_per_sample=8))

    # Convert to RGB24 based on the original color family
    if original_format.color_family == vs.YUV:
        matrix_val = int(clip.get_frame(0).props.get('_Matrix', 1))
        matrix_str = MATRIX_INT_TO_STR.get(matrix_val, "709")
        clip = vs.core.resize.Bicubic(
            clip,
            format=vs.RGB24,
            matrix_in_s=matrix_str,
            range_in_s="limited",
            range_s="full",
            dither_type="error_diffusion",
        )
    elif original_format.color_family == vs.GRAY:
        clip = vs.core.resize.Bicubic(
            clip,
            format=vs.RGB24,
            range_in_s="limited",
            range_s="full",
        )
    else:  # Already RGB but not RGB24 (e.g., RGB48, RGBS)
        clip = vs.core.resize.Bicubic(
            clip,
            format=vs.RGB24,
            range_s="full",
        )

    # After conversion to RGB, _Matrix must be RGB (0)
    clip = clip.std.SetFrameProps(_Matrix=vs.MATRIX_RGB)

    # Mark output as full-range RGB
    if vs.core.core_version.release_major < 74:
        clip = clip.std.SetFrameProps(_ColorRange=vs.RANGE_FULL)
    else:
        clip = clip.std.SetFrameProps(_Range=vs.RANGE_FULL)

    return clip, clip_info


def _restore_with_luma_protect(
    clip: vs.VideoNode,
    clip_info: ClipInfo,
    target_format_id: int | None = None,
) -> vs.VideoNode:
    """
    Convert the RGB24 output to YUV444PS, replace (or blend) the Y plane
    with the original high-bitdepth luma, then convert to the target format.

    :param clip:              processed clip in RGB24 full-range
    :param clip_info:         ClipInfo from convert_format_RGB24
    :param target_format_id:  desired output format; defaults to the original
                              format stored in clip_info
    """
    if clip_info.clip_high_bitdepth is None:
        vs.Error("restore_format: preserve_luma is set but no high-bitdepth reference is available.")

    if target_format_id is None:
        target_format_id = clip_info.format_id

    # 1. Convert RGB24 output to YUV444PS (float, 4:4:4)
    #    The matrix used here must match the matrix the original source carried,
    #    so the merged luma stays semantically coherent with the new chroma.
    matrix_str = _matrix_int_to_str(clip_info.matrix, default="709")

    # Original color range determines the YUV target range
    if clip_info.color_range is not None and clip_info.color_range == vs.RANGE_FULL:
        yuv_range_s = "full"
    else:
        yuv_range_s = "limited"

    havc_yuv = vs.core.resize.Bicubic(
        clip,
        format=vs.YUV444PS,
        matrix_s=matrix_str,
        range_in_s="full",
        range_s=yuv_range_s,
        dither_type="error_diffusion",
    )

    # 2. Extract Y from HAVC output and from the original high-bitdepth reference
    havc_y = vs.core.std.ShufflePlanes(havc_yuv, planes=0, colorfamily=vs.GRAY)
    orig_y = vs.core.std.ShufflePlanes(
        clip_info.clip_high_bitdepth, planes=0, colorfamily=vs.GRAY
    )

    # 3. Build the protected Y plane
    blend = clip_info.luma_blend
    if blend >= 1.0:
        protected_y = orig_y
    elif blend <= 0.0:
        protected_y = havc_y
    else:
        # weighted average: protected = blend * orig + (1 - blend) * havc
        expr = f"x {blend} * y {1.0 - blend} * +"
        protected_y = vs.core.std.Expr([orig_y, havc_y], expr=expr)

    # 4. Recombine planes: protected Y + HAVC U/V
    merged = vs.core.std.ShufflePlanes(
        clips=[protected_y, havc_yuv, havc_yuv],
        planes=[0, 1, 2],
        colorfamily=vs.YUV,
    )

    # 5. Convert to the requested target format
    if merged.format.id == target_format_id:
        return merged

    return vs.core.resize.Bicubic(
        merged,
        format=target_format_id,
        dither_type="error_diffusion",
    )


def restore_format(
    clip: vs.VideoNode,
    clip_info: ClipInfo,
    target_format_id: int | None = None,
) -> vs.VideoNode:
    """
    Restore the colorized RGB24 clip to a format suitable for the original input.
    - If original was GRAY, output YUV420P8 (8-bit color).
    - If original was YUV, restore to original YUV format.
    - If original was RGB, restore to original RGB format.
    Assumes input 'clip' is full-range RGB24.

    :param clip:              clip to process (must be RGB24 full-range).
    :param clip_info:         ClipInfo struct containing original clip information.
    :param target_format_id:  optional override of the output format. If None,
                              uses the original format from clip_info. Useful for
                              keeping the result in YUV444PS for further processing.
    """
    if not isinstance(clip, vs.VideoNode):
        vs.Error("restore_format: Input is not a valid clip.")

    if clip.format.id != vs.RGB24:
        vs.Error("restore_format: Input clip must be RGB24.")

    if clip_info.chroma_resize:
        clip = resize_to_chroma(clip_info.clip_orig, clip)

    # Luma-protect path: preserves original high-bitdepth Y
    if clip_info.preserve_luma:
        return _restore_with_luma_protect(clip, clip_info, target_format_id)

    # Standard path (8-bit round-trip)
    output_format_id = target_format_id if target_format_id is not None else clip_info.format_id

    # If already in target format (unlikely post-colorization), return as-is
    if clip.format.id == output_format_id:
        return clip

    if clip_info.color_family == vs.YUV:
        matrix = clip_info.matrix if clip_info.matrix is not None else vs.MATRIX_BT709
        range_s = "limited"
        if clip_info.color_range is not None:
            range_s = "full" if clip_info.color_range == vs.RANGE_FULL else "limited"

        restored = vs.core.resize.Bicubic(
            clip,
            format=output_format_id,
            matrix_in=vs.MATRIX_RGB,
            matrix=matrix,
            range_in_s="full",
            range_s=range_s,
            dither_type="error_diffusion",
        )
    elif clip_info.color_family == vs.GRAY:
        # Original was grayscale → output 8-bit YUV (colorized result)
        range_s = "limited"
        if clip_info.color_range is not None:
            range_s = "full" if clip_info.color_range == vs.RANGE_FULL else "limited"

        # If target_format_id was overridden to something compatible, honor it;
        # otherwise default to YUV420P8 (consumer-friendly colorized output).
        gray_target = output_format_id if target_format_id is not None else vs.YUV420P8

        restored = vs.core.resize.Bicubic(
            clip,
            format=gray_target,
            matrix=vs.MATRIX_BT709,
            range_in_s="full",
            range_s=range_s,
            dither_type="error_diffusion",
        )
    else:
        # Original was RGB (but not RGB24, e.g., RGB48, RGBS)
        range_s = "full"
        if clip_info.color_range is not None:
            range_s = "full" if clip_info.color_range == vs.RANGE_FULL else "limited"

        restored = vs.core.resize.Bicubic(
            clip,
            format=output_format_id,
            range_in_s="full",
            range_s=range_s,
        )

    return restored


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
        sc_clip=SCDetect(clip=sc_clip, threshold=threshold)
    except Exception as e:
        raise vs.Error(f"SCDetect: {e}")

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

def SCDetect(clip: vs.VideoNode, threshold: float = 0.1, plane: int = 0) -> vs.VideoNode:
    """
    Scene change detection with _SceneChangePrev/_SceneChangeNext frame properties.
    Uses std.PlaneStats-based reimplementation.

    Args:
        clip      : Input clip
        threshold : Scene change threshold (default: 0.1, must be 0.0–1.0)
        plane     : Plane to analyze;

    Returns:
        Clip with _SceneChangePrev and _SceneChangeNext frame properties set.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('SCDetect: this is not a clip')
    if not (0.0 <= threshold <= 1.0):
        raise vs.Error('SCDetect: threshold must be between 0.0 and 1.0')
    if clip.num_frames < 2:
        raise vs.Error('SCDetect: clip must have more than one frame')

    prev_shifted = clip.std.DuplicateFrames(0).std.Trim(last=clip.num_frames - 1)
    prev_stats = core.std.PlaneStats(prev_shifted, clip, plane=plane)
    next_stats = core.std.PlaneStats(clip, clip.std.Trim(first=1), plane=plane)

    def _set_sc_props(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props['_SceneChangePrev'] = int(float(f[1].props.get('PlaneStatsDiff', 0.0)) > threshold)
        fout.props['_SceneChangeNext'] = int(float(f[2].props.get('PlaneStatsDiff', 0.0)) > threshold)
        return fout

    return clip.std.ModifyFrame(
        clips=[clip, prev_stats, next_stats],
        selector=_set_sc_props
    )

def debug_ModifyFrame(f_start: int = 0, f_end: int = 1, clip: vs.VideoNode = None,
                      clips: list[vs.VideoNode] = None, selector: partial = None, silent: bool = True) -> vs.VideoNode:
    f_end = min(f_end, clip.num_frames - 1)
    if len(clips) == 1:
        if f_start > 0:
            frame = clips[0].get_frame(0)
            if not silent:
                print("Debug Frame: ", 0)
            selector(0, frame)
        for n in range(f_start, f_end):
            frame = clips[0].get_frame(n)
            if not silent:
                print("Debug Frame: ", n)
            selector(n, frame)
    else:
        if f_start > 0:
            frame = []
            for j in range(0, len(clips)):
                frame.append(clips[j].get_frame(0))
            if not silent:
                print("Debug Frame: ", 0)
            selector(0, frame)
        for n in range(f_start, f_end):
            frame = []
            for j in range(0, len(clips)):
                frame.append(clips[j].get_frame(n))
            if not silent:
                print("Debug Frame: ", n)
            selector(n, frame)

    return clip

def resize_min_HW(clip: vs.VideoNode, min_size: tuple[int, int] = (512, 480)) -> vs.VideoNode:
    """
    Resize clip so that the max width/height is min_size while maintaining aspect ratio.

    Args:
        clip: Input clip
        min_size: Minimum HxW size, tuple(H,W)  (default: (512, 480))

    Returns:
        Resized clip with minium width/height is min_size (divisible by 2)
    """

    if clip.height < clip.width:
        if clip.height >  min_size[1]:
            return resize_to_height(clip, target_height=min_size[1])
        else:
            return clip # no resize
    else:
        if clip.width >  min_size[0]:
            return resize_to_width(clip, target_width=min_size[0])
        else:
            return clip # no resize


def resize_to_height(clip: vs.VideoNode, target_height: int = 480) -> vs.VideoNode:
    """
    Resize clip to target width while maintaining aspect ratio and ensuring
    height is divisible by 2 (required for many codecs and filters).

    Args:
        clip: Input clip
        target_height: Target height (default: 480)

    Returns:
        Resized clip with target_height and proportional width (divisible by 2)
    """
    # Calculate the proportional height
    target_width = round(clip.width * target_height / clip.height)

    # Ensure height is divisible by 2
    if target_width % 2 != 0:
        target_width -= 1  # or -= 1, but +1 is generally safer to avoid undersizing

    # Resize using spline resampling
    resized_clip = clip.resize.Spline36(width=target_width, height=target_height)

    return resized_clip

def resize_to_width(clip: vs.VideoNode, target_width: int = 512) -> vs.VideoNode:
    """
    Resize clip to target width while maintaining aspect ratio and ensuring
    height is divisible by 2 (required for many codecs and filters).

    Args:
        clip: Input clip
        target_width: Target width (default: 512)

    Returns:
        Resized clip with target_width and proportional height (divisible by 2)
    """
    # Calculate the proportional height
    target_height = round(clip.height * target_width / clip.width)

    # Ensure height is divisible by 2
    if target_height % 2 != 0:
        target_height += 1  # or -= 1, but +1 is generally safer to avoid undersizing

    # Resize using spline resampling
    resized_clip = clip.resize.Spline36(width=target_width, height=target_height)

    return resized_clip


def resize_to_chroma(clip_highres: vs.VideoNode, clip_lowres: vs.VideoNode) -> vs.VideoNode:
    """
        Perform a chroma Resize. The lowres clip will be resized to highres and the Y plane of clip_lowres
        will be replaced by the Y plane of highres clip.

        Args:
            clip_highres: Input highres clip with original plane Y
            clip_lowres: Input lowres clip to apply the chroma resize

        Returns:
            highres clip in RGB24 format with chroma resize
    """
    # perform resize if needed
    if clip_highres.width != clip_lowres.width or clip_highres.height != clip_lowres.height:
        clip_resized = clip_lowres.resize.Spline36(width=clip_highres.width, height=clip_highres.height)
    else:
        clip_resized = clip_lowres
    # convert clips to YUV
    clip_bw = clip_highres.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
    clip_color = clip_resized.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
    # restore orginal Y plane
    clip_yuv = vs.core.std.ShufflePlanes(clips=[clip_bw, clip_color, clip_color], planes=[0, 1, 2], colorfamily=vs.YUV)

    clip_yuv = clip_yuv.std.CopyFrameProps(prop_src=clip_color, props=['_SceneChangePrev', '_SceneChangeNext',
                                                           'sc_threshold', 'sc_frequency', 'sc_luma', 'sc_ratio'])
    # convert result to RGB24
    return clip_yuv.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full", dither_type="error_diffusion")


