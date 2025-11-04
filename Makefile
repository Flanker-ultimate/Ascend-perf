test:
	mkdir -p build
	g++ perf_test.cpp -o build/perf_test -I./ -L/usr/local/Ascend/driver/lib64 -ldcmi
	sudo ./build/perf_test

stress:
	mkdir -p build
	g++ memory_bandwidth_stress.cpp -o /build/memory_stress -lpthread
	sudo ./build/memory_stress

clean:
	rm -f build/perf_test build/memory_stress
	rm -rf build
