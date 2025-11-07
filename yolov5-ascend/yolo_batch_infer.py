#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
from acl_net import Net
from PIL import Image
from torchvision.ops import nms
import acl
from concurrent.futures import ThreadPoolExecutor


# ----------------- 工具函数 -----------------
def check_ret(message, ret):
    from constant import ACL_ERROR_NONE

    if ret != ACL_ERROR_NONE:
        raise Exception(f"{message} failed ret={ret}")


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    max_det=300,
):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((xywh2xyxy(x[:, :4]), conf, j.float()), 1)[
            conf.view(-1) > conf_thres
        ]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        if not x.shape[0]:
            continue
        boxes, scores = x[:, :4], x[:, 4]
        i = nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        output[xi] = x[i]
    return output


def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (
        (img1_shape[1] - img0_shape[1] * gain) / 2,
        (img1_shape[0] - img0_shape[0] * gain) / 2,
    )
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords


def resize_img(img, target_size=(640, 640)):
    old_size = img.shape[0:2]
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))
    img_resized = cv2.resize(img, new_size)
    pad_w = target_size[1] - new_size[0]
    pad_h = target_size[0] - new_size[1]
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return img_padded


def preprocess(img_path, input_shape=(640, 640)):
    img = np.array(Image.open(img_path).convert("RGB"))
    org_img = img[:, :, ::-1]  # RGB -> BGR
    img_resized = resize_img(org_img, input_shape)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = img_resized.transpose(2, 0, 1)  # HWC -> CHW
    return org_img, img_resized


def draw_box(image, boxes, names, scores):
    for i, box in enumerate(boxes):
        box = [int(x) for x in box]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        label = f"{names[i]}:{scores[i]:.2f}"
        cv2.putText(
            image,
            label,
            (box[0], box[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return image


def is_file_ready(file_path, interval=0.01, max_wait=5.0):
    elapsed = 0
    while elapsed < max_wait:
        if not os.path.exists(file_path):
            return False
        size1 = os.path.getsize(file_path)
        time.sleep(interval)
        size2 = os.path.getsize(file_path)
        if size1 == size2 and size1 > 0:
            return True
        elapsed += interval
    return False


def preprocess_parallel(image_paths, input_shape):
    def preprocess_single(img_path):
        if not is_file_ready(img_path):
            return None, None
        org_img, img_tensor = preprocess(img_path, input_shape=input_shape)
        return org_img, img_tensor

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(preprocess_single, image_paths))

    org_imgs, img_tensors = zip(*[res for res in results if res[0] is not None])
    return list(org_imgs), list(img_tensors)


# ----------------- 批量推理函数 -----------------
def process_batch(
    net,
    batch,
    input_size,
    conf_thres,
    iou_thres,
    filter_classes,
    agnostic_nms,
    max_det,
    labels,
    output_dir,
    output_format,
):
    import gc

    org_imgs, img_tensors, rel_paths = [], [], []

    # 1️⃣ 并行读取并预处理图片
    img_paths, rel_paths = zip(*batch)
    org_imgs, img_tensors = preprocess_parallel(img_paths, input_size)

    if len(img_tensors) == 0:
        return

    # 2️⃣ 构建批次输入 (batch, C, H, W)
    batch_input = np.stack(img_tensors, axis=0).astype(np.float32)
    batch_input_bytes = np.frombuffer(batch_input.tobytes(), np.float32)

    start = time.time()
    # 3️⃣ 调用 Net.run 执行推理，并自动释放 host/device 内存
    result = net.run([batch_input_bytes])
    end = time.time()
    # 4️⃣ 自动 reshape 输出
    pred_bytes = bytearray(result[0])
    pred_flat = np.frombuffer(pred_bytes, dtype=np.float32)

    batch_size = len(img_tensors)
    num_classes = len(labels)
    pred_size = num_classes + 5
    num_features = pred_flat.size // batch_size
    if num_features % pred_size != 0:
        raise ValueError(f"Output size {num_features} is not divisible by {pred_size}")

    num_boxes = num_features // pred_size
    pred = torch.tensor(
        pred_flat.reshape(batch_size, num_boxes, pred_size), dtype=torch.float32
    )

    # 5️⃣ NMS
    preds = non_max_suppression(
        pred, conf_thres, iou_thres, filter_classes, agnostic_nms, max_det=max_det
    )

    # 6️⃣ 处理每张图片的结果
    for i, det in enumerate(preds):
        org_img = org_imgs[i]
        rel_path = rel_paths[i]
        img_name = os.path.basename(rel_path)
        img_name_noext = os.path.splitext(img_name)[0]
        subdir = os.path.dirname(rel_path)
        detections = []

        if len(det):
            det[:, :4] = scale_coords(input_size, det[:, :4], org_img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                detections.append((cls, xyxy, conf))

        # 保存 label
        if output_format in ["label", "all"]:
            label_dir = os.path.join(output_dir, "label", subdir)
            os.makedirs(label_dir, exist_ok=True)
            txt_path = os.path.join(label_dir, img_name_noext + ".txt")
            with open(txt_path, "w") as f:
                for cls, xyxy, conf in detections:
                    line = (cls, *xyxy, conf)
                    f.write(("%g " * len(line)).rstrip() % line + "\n")

        # 保存 image
        if output_format in ["image", "all"]:
            image_dir = os.path.join(output_dir, "image", subdir)
            os.makedirs(image_dir, exist_ok=True)
            output_path = os.path.join(image_dir, img_name)
            if len(det):
                out_img = draw_box(
                    org_img.copy(),
                    det[:, :4],
                    [labels[int(c)] for c in det[:, -1]],
                    det[:, 4],
                )
            else:
                out_img = org_img
            cv2.imwrite(output_path, out_img)
            if "out_img" in locals():
                del out_img

        del org_img, detections

    # 7️⃣ 释放 Net 内部 host/device memory
    for item in result:
        if isinstance(item, dict) and "buffer" in item:
            acl.rt.free_host(item["buffer"])  # 释放 malloc_host
    del result

    # 8️⃣ 批次内存释放
    del org_imgs, img_tensors, rel_paths, pred, preds
    del batch_input, batch_input_bytes, pred_bytes, pred_flat
    gc.collect()

    print(
        f"Processed batch of {batch_size} images, use {(end - start):.3f}s, avg {((end - start) / batch_size):.3f}s/img"
    )


# ----------------- 主函数 -----------------
def main(opt):
    os.makedirs(opt.output_dir, exist_ok=True)
    input_size = (opt.img_size, opt.img_size)

    # === 初始化 ACL ===
    acl.init()
    device_id = opt.device
    net = Net(device_id, opt.weights)
    print(f"Model loaded: {opt.weights}")

    with open(opt.labels, "r") as f:
        labels = [x.strip() for x in f.readlines()]
    print(f"Loaded {len(labels)} labels")

    # 遍历图片
    img_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_list = []
    for root, _, files in os.walk(opt.input_dir):
        for name in files:
            if name.lower().endswith(img_formats):
                rel_path = os.path.relpath(os.path.join(root, name), opt.input_dir)
                image_list.append((os.path.join(root, name), rel_path))

    print(f"Found {len(image_list)} images")

    # 批量推理
    batch_size = opt.batch_size
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i : i + batch_size]
        process_batch(
            net,
            batch,
            input_size,
            opt.conf_thres,
            opt.iou_thres,
            None,
            False,
            opt.max_det,
            labels,
            opt.output_dir,
            opt.output_format,
        )

    print("All done!")


# ----------------- 命令行参数 -----------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--output-format", type=str, default="all", choices=["image", "label", "all"]
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device id")
    opt = parser.parse_args()
    main(opt)
