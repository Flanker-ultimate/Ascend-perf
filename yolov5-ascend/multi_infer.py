#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import subprocess
import time
from multiprocessing import Process, Queue
from argparse import ArgumentParser


def run_infer(thread_id, device_id, command, q):
    """在独立进程中执行推理命令"""
    print(f"\n[PROC-{thread_id}] Using device {device_id}")
    print(f"[PROC-{thread_id}] Start:\n{command}\n")
    start = time.time()
    result = subprocess.run(command, shell=True)
    end = time.time()
    print(f"[PROC-{thread_id}] Done in {end - start:.2f}s, return={result.returncode}")
    q.put((thread_id, result.returncode))


def split_images(input_dir, num_procs):
    img_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    imgs = [
        p
        for p in glob.glob(os.path.join(input_dir, "**"), recursive=True)
        if p.lower().endswith(img_ext)
    ]
    imgs.sort()
    n = len(imgs)
    if n == 0:
        raise RuntimeError(f"No images found in {input_dir}")
    chunk_size = (n + num_procs - 1) // num_procs
    return [imgs[i : i + chunk_size] for i in range(0, n, chunk_size)]


def main(opt):
    input_dirs = [x.strip() for x in opt.input_dirs.split(",")]
    output_dirs = [x.strip() for x in opt.output_dirs.split(",")]

    # 🧠 自动扩展路径数量
    if len(input_dirs) == 1 and opt.parallel > 1:
        input_dir = input_dirs[0]
        parts = split_images(input_dir, opt.parallel)
        tmp_list_dir = "tmp_input_lists"
        os.makedirs(tmp_list_dir, exist_ok=True)
        input_dirs = []
        output_dirs = []
        for i, part in enumerate(parts):
            list_path = os.path.join(tmp_list_dir, f"proc_{i + 1}.txt")
            with open(list_path, "w") as f:
                f.write("\n".join(part))
            input_dirs.append(list_path)
            output_dirs.append(os.path.join(opt.output_base, f"proc_{i + 1}"))
        print(f"[INFO] 分配 {len(parts)} 个任务列表，每个进程约 {len(parts[0])} 张图")

    num_procs = min(opt.parallel, len(input_dirs))
    print(f"[INFO] Starting {num_procs} processes on {opt.device_count} device(s)")

    q = Queue()
    processes = []

    for i in range(num_procs):
        device_id = i % opt.device_count
        input_arg = (
            f"--input-list {input_dirs[i]}"
            if input_dirs[i].endswith(".txt")
            else f"--input-dir {input_dirs[i]}"
        )
        cmd = (
            f"ASCEND_DEVICE_ID={device_id} python3 {opt.script} "
            f"{input_arg} "
            f"--output-dir {output_dirs[i]} "
            f"--weights {opt.weights} "
            f"--labels {opt.labels} "
            f"--batch-size {opt.batch_size} "
            f"--img-size {opt.img_size} "
            f"--conf-thres {opt.conf_thres} "
            f"--iou-thres {opt.iou_thres} "
            f"--max-det {opt.max_det} "
            f"--output-format {opt.output_format} "
            f"--device {device_id}"
        )

        p = Process(target=run_infer, args=(i + 1, device_id, cmd, q))
        processes.append(p)
        p.start()
        time.sleep(opt.interval)

    for p in processes:
        p.join()

    # 汇总结果
    while not q.empty():
        tid, ret = q.get()
        print(f"[PROC-{tid}] {'✅ Success' if ret == 0 else f'❌ Failed ({ret})'}")

    print("[INFO] All processes finished.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--script", type=str, default="yolo_batch_infer.py")
    parser.add_argument("--input-dirs", type=str, required=True)
    parser.add_argument("--output-dirs", type=str, default="")
    parser.add_argument(
        "--output-base", type=str, default="/home/ubuntu/perf_test/results"
    )
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output-format", type=str, default="all")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--device-count", type=int, default=1)
    opt = parser.parse_args()

    start = time.time()
    main(opt)
    end = time.time()
    print(f"[INFO] Total time: {end - start:.2f} seconds")
