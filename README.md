# test Ascend DRAM bandwidth
## change model input shape
```
python3 export.py --weights yolov5s.pt --opset 12 --simplify --include onnx --batch-size 100
```

```
atc --input_shape="images:100,3,640,640" --input_format=NCHW --output="yolov5s" --soc_version=Ascend310 --framework=5 --model="yolov5s.onnx" --output_type=FP32
```

launch batch inference:
```
python3 yolo_batch_infer.py --input-dir /home/ubuntu/pics_convert \
  --output-dir /home/ubuntu/perf_test/results \
  --weights /home/ubuntu/perf_test/yolov5-ascend/ascend/yolov5s.om \
  --labels /home/ubuntu/perf_test/yolov5-ascend/ascend/yolov5.label \
  --batch-size 20
```
