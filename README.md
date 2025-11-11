# test Ascend DRAM bandwidth
## change model input shape
change batch-size if needed

``` sh
python3 export.py --weights yolov5s.pt --opset 12 --simplify --include onnx --batch-size 2
```

``` sh
atc --input_shape="images:2,3,640,640" --input_format=NCHW --output="yolov5s" --soc_version=Ascend310 --framework=5 --model="yolov5s.onnx" --output_type=FP32
```

## launch batch inference


``` sh
cd /home/ubuntu/Ascend-perf/yolov5-ascend

python3 yolo_batch_infer.py --input-dir /home/ubuntu/pics_convert \
  --output-dir /home/ubuntu/Ascend-perf/results \
  --weights /home/ubuntu/Ascend-perf/yolov5-ascend/ascend/yolov5s1b.om \
  --labels /home/ubuntu/Ascend-perf/yolov5-ascend/ascend/yolov5.label \
  --batch-size 1
```


``` sh
python3 multi_infer.py \
  --input-dirs "/home/ubuntu/pics_convert" \
  --output-dirs "/home/ubuntu/Ascend-perf/results" \
  --weights /home/ubuntu/Ascend-perf/yolov5-ascend/ascend/yolov5s1b.om \
  --labels /home/ubuntu/Ascend-perf/yolov5-ascend/ascend/yolov5.label \
  --parallel 16 \
  --device-count 1 \
  --batch-size 1
```
## watch DRAM bandwidth

``` sh
watch -n 0.1 npu-smi info
watch -n 0.1 free -h
```


## Perf results

| batch size | process time(s) | AICore % |
|------------|-----------------|----------|
| 1          | 460             | 5        |
| 2          | 233             | 18       |
| 4          | 120             | 39       |
| 8          | 73              | 77       |
| 16         | 69              | 89       |
