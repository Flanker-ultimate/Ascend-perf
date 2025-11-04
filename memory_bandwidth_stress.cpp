#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>

// 配置参数（可根据需求调整）
#define THREAD_NUM 8          // 并发线程数（越多带宽占用越高）
#define DATA_SIZE_MB 1024     // 每个线程操作的数据量（MB）
#define LOOP_COUNT 100000     // 每个线程的读写循环次数（越大占用时间越长）

// 每个线程的参数
typedef struct {
    char* data;               // 线程操作的内存块
    size_t size;              // 内存块大小（字节）
} ThreadArgs;

// 线程函数：循环读写内存，占用带宽
void* memory_stress(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    char* data = args->data;
    size_t size = args->size;
    char temp = 0;

    // 循环读写：先写后读，最大化带宽占用
    for (int i = 0; i < LOOP_COUNT; i++) {
        // 写操作：填充内存（覆盖写入）
        memset(data, i % 256, size);  // 用循环变量的模值填充，避免编译器优化掉"无用写入"

        // 读操作：遍历内存（强制读取）
        for (size_t j = 0; j < size; j += 64) {  // 64字节对齐，模拟CPU缓存行读取
            temp ^= data[j];  // 异或操作，避免编译器优化掉"无用读取"
        }
    }

    // 防止temp变量被优化（无实际意义，仅为避免编译优化）
    (void)temp;
    return NULL;
}

int main() {
    pthread_t threads[THREAD_NUM];
    ThreadArgs args[THREAD_NUM];
    size_t data_size = DATA_SIZE_MB * 1024 * 1024;  // 转换为字节

    printf("=== 内存带宽压力测试程序 ===\n");
    printf("线程数: %d\n", THREAD_NUM);
    printf("单线程数据量: %d MB\n", DATA_SIZE_MB);
    printf("总数据量: %d MB\n", THREAD_NUM * DATA_SIZE_MB);
    printf("循环次数: %d\n", LOOP_COUNT);
    printf("开始占用内存带宽...（按Ctrl+C停止）\n");

    // 为每个线程分配内存并创建线程
    for (int i = 0; i < THREAD_NUM; i++) {
        // 分配大块内存（使用malloc而非栈，避免栈溢出）
        args[i].data = (char*)malloc(data_size);
        if (args[i].data == NULL) {
            perror("malloc failed");
            exit(EXIT_FAILURE);
        }
        args[i].size = data_size;

        // 创建线程
        if (pthread_create(&threads[i], NULL, memory_stress, &args[i]) != 0) {
            perror("pthread_create failed");
            exit(EXIT_FAILURE);
        }
    }

    // 等待所有线程结束（实际会无限循环，需手动终止）
    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_join(threads[i], NULL);
        free(args[i].data);  // 释放内存（实际不会执行，因线程无限循环）
    }

    return 0;
}
// g++ memory_bandwidth_stress.cpp -o memory_stress -lpthread
