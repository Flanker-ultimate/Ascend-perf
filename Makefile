test:
	mkdir -p build
	g++ perf_test.cpp -o build/perf_test -I./ -L/usr/local/Ascend/driver/lib64 -ldcmi
	sudo ./build/perf_test

clean:
	rm -f build/perf_test build/memory_stress
	rm -rf build
	rm -rf results
	rm -rf yolov5-ascend/__pycache__
