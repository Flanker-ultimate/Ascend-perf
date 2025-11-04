test:
	g++ perf_test.cpp -o /build/perf_test -I./ -L/usr/local/Ascend/driver/lib64 -ldcmi
	sudo ./perf_test

stress:
	g++ memory_bandwidth_stress.cpp -o /build/memory_stress -lpthread
	sudo ./memory_stress

clean:
	rm -f /build/perf_test /build/memory_stress
