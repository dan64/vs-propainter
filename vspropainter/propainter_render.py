"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-05-26
version:
LastEditors: Dan64
LastEditTime: 2026-01-02
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
ProPainter rendering class.
"""
import os
import cv2
import numpy as np
import scipy.ndimage
from PIL import Image

import torch

from vspropainter.model.modules.flow_comp_raft import RAFT_bi
from vspropainter.model.recurrent_flow_completion import RecurrentFlowCompleteNet
from vspropainter.model.propainter import InpaintGenerator
from vspropainter.core.utils import to_tensors

import warnings

warnings.filterwarnings("ignore")

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# configuring torch
torch.backends.cudnn.benchmark = True


class ModelProPainterOut:
    _w = None
    _h = None
    model = None
    fix_flow_complete = None
    fix_raft = None
    flow_masks = None
    masks_dilated = None
    frames = None
    device = None
    video_length = None
    frames_inp = None
    frames_len = None
    frame_size = None
    pred_flows_bi = None
    updated_frames = None
    neighbor_length = None
    neighbor_stride = None
    ref_stride = None
    raft_iter = None
    mask_dilation = None
    clip_size = None
    outpaint_size = None

    def __init__(self, torch_device=None, weights_dir: str = None, mask_dilation: int = 4,
                 outpaint_size: tuple[int, int] = None,
                 neighbor_length: int = 10, ref_stride: int = 4, raft_iter: int = 20,
                 clip_size: tuple[int, int] = None):

        self.model_dir = weights_dir
        self.neighbor_length = neighbor_length
        self.mask_dilation = mask_dilation
        self.neighbor_stride = self.neighbor_length // 2
        self.ref_stride = ref_stride
        self.raft_iter = raft_iter
        self.clip_size = clip_size
        self.outpaint_size = outpaint_size
        self.model_init(torch_device, model_dir)

    def _binary_mask(self, mask, th=0.1):
        mask[mask > th] = 1
        mask[mask <= th] = 0
        return mask

    # read frame-wise masks
    def _extrapolation(self, video_ori: list = None):
        """Prepares the data for video outpainting.
            """
        nFrame = len(video_ori)
        imgW, imgH = video_ori[0].size

        # Defines new FOV.
        imgW_extr = int(self.outpaint_size[0])
        imgH_extr = int(self.outpaint_size[1])
        imgH_extr = imgH_extr - imgH_extr % 8
        imgW_extr = imgW_extr - imgW_extr % 8
        H_start = int((imgH_extr - imgH) / 2)
        W_start = int((imgW_extr - imgW) / 2)

        # Extrapolates the FOV for video.
        frames = []
        for v in video_ori:
            frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
            frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
            frames.append(Image.fromarray(frame))

        # Generates the mask for missing region.
        masks_dilated = []
        flow_masks = []

        dilate_h = 4 if H_start > 10 else 0
        dilate_w = 4 if W_start > 10 else 0
        mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)

        mask[H_start + dilate_h: H_start + imgH - dilate_h,
        W_start + dilate_w: W_start + imgW - dilate_w] = 0
        flow_masks.append(Image.fromarray(mask * 255))

        mask[H_start: H_start + imgH, W_start: W_start + imgW] = 0
        masks_dilated.append(Image.fromarray(mask * 255))

        flow_masks = flow_masks * nFrame
        masks_dilated = masks_dilated * nFrame

        return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)

    def prepare_extrapolation(self, video_frames: list = None):

        self.frames_len = len(video_frames)

        video_frames, self.flow_masks, self.masks_dilated, size = self._extrapolation(video_frames)
        self.frame_size = size
        self._w, self._h = self.frame_size

        self.frames_inp = [np.array(f).astype(np.uint8) for f in video_frames]
        self.frames = to_tensors()(video_frames).unsqueeze(0) * 2 - 1
        self.flow_masks = to_tensors()(self.flow_masks).unsqueeze(0)
        self.masks_dilated = to_tensors()(self.masks_dilated).unsqueeze(0)
        self.frames, self.flow_masks, self.masks_dilated = self.frames.to(self.device), self.flow_masks.to(
            self.device), self.masks_dilated.to(self.device)

    def get_video_length(self) -> int:
        return self.frames_len

    def _get_ref_index(self, mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, ref_stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
            end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
            for i in range(start_idx, end_idx, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_index) > ref_num:
                        break
                    ref_index.append(i)
        return ref_index

    def model_init(self, torch_device, model_dir: str = None):
        self.device = torch_device

        ##############################################
        # set up RAFT and flow competition model
        ##############################################
        ckpt_path = os.path.abspath(os.path.join(model_dir, 'raft-things.pth'))
        self.fix_raft = RAFT_bi(ckpt_path, self.device)

        ckpt_path = os.path.abspath(os.path.join(model_dir, 'recurrent_flow_completion.pth'))
        self.fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(self.device)
        self.fix_flow_complete.eval()

        ##############################################
        # set up ProPainter model
        ##############################################
        ckpt_path = os.path.abspath(os.path.join(model_dir, 'ProPainter.pth'))
        self.model = InpaintGenerator(model_path=ckpt_path).to(self.device)
        self.model.eval()

    def _resize_frames(self, frames):
        out_size = frames[0].size
        process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]

        return frames, process_size, out_size

    def get_extrapolated_frames(self, video_frames: list = None, batch_size: int = 100, use_half: bool = True,
                                convert_to_pil: bool = False):

        resized_frames, frame_size, out_size = self._resize_frames(video_frames)

        self.prepare_extrapolation(resized_frames)
        self.inference(batch_size, use_half)
        extrapolated_frames = self.feature_propagation(convert_to_pil, self.outpaint_size)

        return extrapolated_frames

    def inference(self, batch_size: int = 100, use_half: bool = True):
        ##############################################
        # ProPainter inference
        ##############################################
        self.video_length = self.frames.size(1)
        w = self._w
        h = self._h

        with (torch.no_grad()):
            # ---- compute flow ----
            if self.frames.size(-1) <= 640:
                short_clip_len = 12
            elif self.frames.size(-1) <= 720:
                short_clip_len = 8
            elif self.frames.size(-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2

            # use fp32 for RAFT
            if self.frames.size(1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, self.video_length, short_clip_len):
                    end_f = min(self.video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self.fix_raft(self.frames[:, f:end_f], iters=self.raft_iter)
                    else:
                        flows_f, flows_b = self.fix_raft(self.frames[:, f - 1:end_f], iters=self.raft_iter)

                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()

                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = self.fix_raft(self.frames, iters=self.raft_iter)
                torch.cuda.empty_cache()

            if use_half:
                self.frames, self.flow_masks, self.masks_dilated = self.frames.half(), self.flow_masks.half(), self.masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                self.fix_flow_complete = self.fix_flow_complete.half()
                self.model = self.model.half()

            # ---- complete flow ----
            flow_length = gt_flows_bi[0].size(1)
            if flow_length > batch_size:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f in range(0, flow_length, batch_size):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + batch_size + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + batch_size)
                    pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        self.flow_masks[:, s_f:e_f + 1])
                    pred_flows_bi_sub = self.fix_flow_complete.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        pred_flows_bi_sub,
                        self.flow_masks[:, s_f:e_f + 1])

                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()

                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                self.pred_flows_bi = (pred_flows_f, pred_flows_b)
            else:
                self.pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, self.flow_masks)
                self.pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, self.pred_flows_bi,
                                                                         self.flow_masks)
                torch.cuda.empty_cache()

            # ---- image propagation ----
            self.masked_frames = self.frames * (1 - self.masks_dilated)
            # ensure a minimum of 100 frames for image propagation
            subvideo_length_img_prop = batch_size  # min(100, batch_size)
            if self.video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f in range(0, self.video_length, subvideo_length_img_prop):
                    s_f = max(0, f - pad_len)
                    e_f = min(self.video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(self.video_length, f + subvideo_length_img_prop)

                    b, t, _, _, _ = self.masks_dilated[:, s_f:e_f].size()
                    pred_flows_bi_sub = (self.pred_flows_bi[0][:, s_f:e_f - 1], self.pred_flows_bi[1][:, s_f:e_f - 1])
                    prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(self.masked_frames[:, s_f:e_f],
                                                                                        pred_flows_bi_sub,
                                                                                        self.masks_dilated[:, s_f:e_f],
                                                                                        'nearest')
                    updated_frames_sub = self.frames[:, s_f:e_f] * (1 - self.masks_dilated[:, s_f:e_f]) + \
                                         prop_imgs_sub.view(b, t, 3, h, w) * self.masks_dilated[:, s_f:e_f]
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)

                    updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()

                self.updated_frames = torch.cat(updated_frames, dim=1)
                self.updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = self.masks_dilated.size()
                prop_imgs, updated_local_masks = self.model.img_propagation(self.masked_frames, self.pred_flows_bi,
                                                                            self.masks_dilated, 'nearest')
                self.updated_frames = self.frames * (1 - self.masks_dilated) + prop_imgs.view(b, t, 3, h,
                                                                                              w) * self.masks_dilated
                self.updated_masks = updated_local_masks.view(b, t, 1, h, w)
                torch.cuda.empty_cache()

    def feature_propagation(self, convert_to_pil: bool = False, out_size: list = None, ref_num: int = -1):

        comp_frames = [None] * self.video_length

        for f in range(0, self.video_length, self.neighbor_stride):
            neighbor_ids = [
                i for i in range(max(0, f - self.neighbor_stride),
                                 min(self.video_length, f + self.neighbor_stride + 1))
            ]
            ref_ids = self._get_ref_index(f, neighbor_ids, self.video_length, self.ref_stride, ref_num)
            selected_imgs = self.updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = self.masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = self.updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (
                self.pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                self.pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])

            with torch.no_grad():
                # 1.0 indicates mask
                l_t = len(neighbor_ids)

                # pred_img = selected_imgs # results of image propagation
                pred_img = self.model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)

                pred_img = pred_img.view(-1, 3, self._h, self._w)

                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = self.masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                          + self.frames_inp[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5

                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)

            torch.cuda.empty_cache()

        if convert_to_pil:
            np_comp_frames = [Image.fromarray(cv2.resize(f, out_size), 'RGB') for f in comp_frames]
        else:
            np_comp_frames = [cv2.resize(f, out_size) for f in comp_frames]

        torch.cuda.empty_cache()
        return np_comp_frames


class ModelProPainterIn:
    _w = None
    _h = None
    model = None
    fix_flow_complete = None
    fix_raft = None
    flow_masks = None
    masked_frames = None
    masks_dilated = None
    frames = None
    device = None
    video_length = None
    frames_inp = None
    frames_len = None
    frame_size = None
    pred_flows_bi = None
    updated_frames = None
    updated_masks = None
    img_mask = None
    img_mask_orig = None
    img_mask_is_cropped = False
    neighbor_length = None
    neighbor_stride = None
    ref_stride = None
    raft_iter = None
    mask_dilation = None
    clip_size = None

    def __init__(self, torch_device=None, weights_dir: str = None, mask_path: str = None, mask_dilation: int = 4,
                 neighbor_length: int = 10, ref_stride: int = 4, raft_iter: int = 20,
                 clip_size: tuple[int, int] = None):

        self.model_dir = weights_dir
        self.neighbor_length = neighbor_length
        self.mask_dilation = mask_dilation
        self.neighbor_stride = self.neighbor_length // 2
        self.ref_stride = ref_stride
        self.raft_iter = raft_iter
        self.clip_size = clip_size
        self.model_init(torch_device, model_dir, mask_path)

    def _binary_mask(self, mask, th=0.1):
        mask[mask > th] = 1
        mask[mask <= th] = 0
        return mask

    # read frame-wise masks
    def _read_mask(self, masks_img_list: list = None, flow_mask_dilates=8, mask_dilates=5):

        masks_dilated = []
        flow_masks = []

        size = self.frame_size
        length = self.frames_len

        for mask_img in masks_img_list:
            if size is not None:
                mask_img = mask_img.resize(size, Image.NEAREST)
            mask_img = np.array(mask_img.convert('L'))

            # Dilate 8 pixel so that all known pixel is trustworthy
            if flow_mask_dilates > 0:
                flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
            else:
                flow_mask_img = self._binary_mask(mask_img).astype(np.uint8)
            # Close the small holes inside the foreground objects
            # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
            # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
            flow_masks.append(Image.fromarray(flow_mask_img * 255))

            if mask_dilates > 0:
                mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
            else:
                mask_img = self._binary_mask(mask_img).astype(np.uint8)
            masks_dilated.append(Image.fromarray(mask_img * 255))

        if len(masks_img_list) == 1:
            flow_masks = flow_masks * length
            masks_dilated = masks_dilated * length

        return flow_masks, masks_dilated

    def prepare_mask(self, video_frames: list = None, mask_frames: list = None, frame_size: () = None):

        self.frame_size = frame_size
        self._w, self._h = self.frame_size

        self.frames_len = len(video_frames)

        if mask_frames is None:
            out_size = self.img_mask.size
            if not out_size == frame_size:
                self.img_mask = self.img_mask.resize(frame_size)
            img_mask_list = [self.img_mask]
        else:
            img_mask_list = mask_frames

        self.flow_masks, self.masks_dilated = self._read_mask(masks_img_list=img_mask_list,
                                                              flow_mask_dilates=self.mask_dilation,
                                                              mask_dilates=self.mask_dilation)

        self.frames_inp = [np.array(f).astype(np.uint8) for f in video_frames]
        self.frames = to_tensors()(video_frames).unsqueeze(0) * 2 - 1
        self.flow_masks = to_tensors()(self.flow_masks).unsqueeze(0)
        self.masks_dilated = to_tensors()(self.masks_dilated).unsqueeze(0)
        self.frames, self.flow_masks, self.masks_dilated = self.frames.to(self.device), self.flow_masks.to(
            self.device), self.masks_dilated.to(self.device)

    def get_video_length(self) -> int:
        return self.frames_len

    def _get_ref_index(self, mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, ref_stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
            end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
            for i in range(start_idx, end_idx, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_index) > ref_num:
                        break
                    ref_index.append(i)
        return ref_index

    def img_mask_crop(self, mask_region: tuple[int, int, int, int] = None):

        if self.img_mask is None or self.img_mask_is_cropped:
            return

        left = mask_region[2]
        top = mask_region[3]
        right = left + mask_region[0]
        bottom = top + mask_region[1]

        self.img_mask = self.img_mask.crop((left, top, right, bottom))
        self.img_mask_is_cropped = True

    def model_init(self, torch_device, model_dir: str = None, mpath: str = None):
        self.device = torch_device

        self.img_mask_is_cropped = False

        # load image mask if available
        if mpath is None:
            self.img_mask = None
        else:
            img_m = Image.open(mpath).convert('RGB')
            out_size = img_m.size
            if out_size == self.clip_size:
                self.img_mask = img_m
            else:
                self.img_mask = img_m.resize(self.clip_size)
        self.img_mask_orig = self.img_mask

        ##############################################
        # set up RAFT and flow competition model
        ##############################################
        ckpt_path = os.path.abspath(os.path.join(model_dir, 'raft-things.pth'))
        self.fix_raft = RAFT_bi(ckpt_path, self.device)

        ckpt_path = os.path.abspath(os.path.join(model_dir, 'recurrent_flow_completion.pth'))
        self.fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(self.device)
        self.fix_flow_complete.eval()

        ##############################################
        # set up ProPainter model
        ##############################################
        ckpt_path = os.path.abspath(os.path.join(model_dir, 'ProPainter.pth'))
        self.model = InpaintGenerator(model_path=ckpt_path).to(self.device)
        self.model.eval()

    def _resize_frames(self, frames):
        out_size = frames[0].size
        process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]

        return frames, process_size, out_size

    def get_unmasked_frames(self, video_frames: list = None, mask_frames: list = None, batch_size: int = 100,
                            use_half: bool = True,
                            convert_to_pil: bool = False):

        resized_frames, frame_size, out_size = self._resize_frames(video_frames)

        self.prepare_mask(resized_frames, mask_frames, frame_size)
        self.inference(batch_size, use_half)
        unmasked_frames = self.feature_propagation(convert_to_pil, out_size)

        return unmasked_frames

    def inference(self, batch_size: int = 100, use_half: bool = True):
        ##############################################
        # ProPainter inference
        ##############################################
        self.video_length = self.frames.size(1)
        w = self._w
        h = self._h

        with (torch.no_grad()):
            # ---- compute flow ----
            if self.frames.size(-1) <= 640:
                short_clip_len = 12
            elif self.frames.size(-1) <= 720:
                short_clip_len = 8
            elif self.frames.size(-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2

            # use fp32 for RAFT
            if self.frames.size(1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, self.video_length, short_clip_len):
                    end_f = min(self.video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self.fix_raft(self.frames[:, f:end_f], iters=self.raft_iter)
                    else:
                        flows_f, flows_b = self.fix_raft(self.frames[:, f - 1:end_f], iters=self.raft_iter)

                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()

                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = self.fix_raft(self.frames, iters=self.raft_iter)
                torch.cuda.empty_cache()

            if use_half:
                self.frames, self.flow_masks, self.masks_dilated = self.frames.half(), self.flow_masks.half(), self.masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                self.fix_flow_complete = self.fix_flow_complete.half()
                self.model = self.model.half()

            # ---- complete flow ----
            flow_length = gt_flows_bi[0].size(1)
            if flow_length > batch_size:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f in range(0, flow_length, batch_size):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + batch_size + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + batch_size)
                    pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        self.flow_masks[:, s_f:e_f + 1])
                    pred_flows_bi_sub = self.fix_flow_complete.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        pred_flows_bi_sub,
                        self.flow_masks[:, s_f:e_f + 1])

                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()

                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                self.pred_flows_bi = (pred_flows_f, pred_flows_b)
            else:
                self.pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, self.flow_masks)
                self.pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, self.pred_flows_bi,
                                                                         self.flow_masks)
                torch.cuda.empty_cache()

            # ---- image propagation ----
            self.masked_frames = self.frames * (1 - self.masks_dilated)
            # ensure a minimum of 100 frames for image propagation
            subvideo_length_img_prop = batch_size  # min(100, batch_size)
            if self.video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f in range(0, self.video_length, subvideo_length_img_prop):
                    s_f = max(0, f - pad_len)
                    e_f = min(self.video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(self.video_length, f + subvideo_length_img_prop)

                    b, t, _, _, _ = self.masks_dilated[:, s_f:e_f].size()
                    pred_flows_bi_sub = (self.pred_flows_bi[0][:, s_f:e_f - 1], self.pred_flows_bi[1][:, s_f:e_f - 1])
                    prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(self.masked_frames[:, s_f:e_f],
                                                                                        pred_flows_bi_sub,
                                                                                        self.masks_dilated[:, s_f:e_f],
                                                                                        'nearest')
                    updated_frames_sub = self.frames[:, s_f:e_f] * (1 - self.masks_dilated[:, s_f:e_f]) + \
                                         prop_imgs_sub.view(b, t, 3, h, w) * self.masks_dilated[:, s_f:e_f]
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)

                    updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()

                self.updated_frames = torch.cat(updated_frames, dim=1)
                self.updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = self.masks_dilated.size()
                prop_imgs, updated_local_masks = self.model.img_propagation(self.masked_frames, self.pred_flows_bi,
                                                                            self.masks_dilated, 'nearest')
                self.updated_frames = self.frames * (1 - self.masks_dilated) + prop_imgs.view(b, t, 3, h,
                                                                                              w) * self.masks_dilated
                self.updated_masks = updated_local_masks.view(b, t, 1, h, w)
                torch.cuda.empty_cache()

    def feature_propagation(self, convert_to_pil: bool = False, out_size: list = None, ref_num: int = -1):

        comp_frames = [None] * self.video_length

        for f in range(0, self.video_length, self.neighbor_stride):
            neighbor_ids = [
                i for i in range(max(0, f - self.neighbor_stride),
                                 min(self.video_length, f + self.neighbor_stride + 1))
            ]
            ref_ids = self._get_ref_index(f, neighbor_ids, self.video_length, self.ref_stride, ref_num)
            selected_imgs = self.updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = self.masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = self.updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (
                self.pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                self.pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])

            with torch.no_grad():
                # 1.0 indicates mask
                l_t = len(neighbor_ids)

                # pred_img = selected_imgs # results of image propagation
                pred_img = self.model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)

                pred_img = pred_img.view(-1, 3, self._h, self._w)

                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = self.masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                          + self.frames_inp[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5

                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)

            torch.cuda.empty_cache()

        if convert_to_pil:
            np_comp_frames = [Image.fromarray(cv2.resize(f, out_size), 'RGB') for f in comp_frames]
        else:
            np_comp_frames = [cv2.resize(f, out_size) for f in comp_frames]

        torch.cuda.empty_cache()
        return np_comp_frames
