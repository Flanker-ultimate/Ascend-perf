#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "dcmi_interface_api.h"  // 引入DCMI头文件
#include <iostream>

using namespace std;

// 打印错误信息（根据DCMI错误码）
void print_error(int ret) {
    switch (ret) {
        case DCMI_OK:
            return;
        case DCMI_ERR_CODE_INVALID_PARAMETER:
            printf("错误：无效参数\n");
            break;
        case DCMI_ERR_CODE_DEVICE_NOT_EXIST:
            printf("错误：设备不存在\n");
            break;

        default:
            printf("错误：返回码=%d（参考DCMI错误码定义）\n", ret);
    }
}

int main() {
    printf("=== DDR内存带宽利用率监控工具 ===\n");

    // 1. 初始化DCMI
    int ret = dcmi_init();
    if (ret != DCMI_OK) {
        printf("dcmi_init() 失败：");
        print_error(ret);
        return -1;
    }
    printf("DCMI初始化成功\n");

    // 2. 获取设备卡列表（获取系统中存在的昇腾卡ID）
    int card_list[MAX_CARD_NUM] = {0};
    int card_num = 0;
    ret = dcmi_get_card_list(&card_num, card_list, MAX_CARD_NUM);  // V2版本接口
    if (ret != DCMI_OK || card_num <= 0) {
        printf("获取卡列表失败：");
        print_error(ret);
        return -1;
    }
    int card_id = card_list[1];  // 使用第一张卡
    printf("检测到昇腾卡数量：%d，使用卡ID：%d\n", card_num, card_id);

    // 3. 获取卡上的设备数量（单卡可能包含多个设备）
    int device_num = 0;
    ret = dcmi_get_device_num_in_card(card_id, &device_num);
    if (ret != DCMI_OK || device_num <= 0) {
        printf("获取设备数量失败：");
        print_error(ret);
        return -1;
    }
    int device_id = 0;  // 使用第一个设备
    printf("卡上设备数量：%d，使用设备ID：%d\n", device_num, device_id);

    char dcmi_ver[32];
    dcmi_get_dcmi_version(dcmi_ver, 32);
    std::cout << "DCMI版本：" << dcmi_ver << std::endl;

    char driver_ver[64];
    dcmi_get_driver_version(driver_ver, 64);
    std::cout << "驱动版本：" << driver_ver << std::endl;

    // 4. 循环查询DDR带宽利用率（每秒一次）
    printf("\n开始监控DDR带宽利用率（按Ctrl+C停止）...\n");
    printf("时间(s) | DDR带宽利用率(%%)\n");
    printf("-------------------------\n");
    int count = 0;
    while (1) {
        unsigned int ddr_bandwidth_util = 255;  // 存储带宽利用率（0~100）
        int temperature = 0;
        // 核心调用：查询DDR带宽利用率（使用DCMI_UTILIZATION_RATE_DDR_BANDWIDTH标识）
        ret = dcmi_get_device_utilization_rate(
            card_id,
            device_id,
            DCMI_UTILIZATION_RATE_DDR_BANDWIDTH,  // 指定查询DDR带宽利用率
            &ddr_bandwidth_util
        );
        // ret = dcmi_get_device_temperature(card_id, device_id, &temperature);


        // 输出结果
        if (ret == DCMI_OK) {
            printf("%8d | %16d\n", count, ddr_bandwidth_util);
        } else {
            printf("%8d | 查询失败，错误码=%d\n", count, ret);
            print_error(ret);
            // 若连续失败，退出循环
            if (count > 3) break;
        }

        count++;
        sleep(1);  // 每秒查询一次
    }

    return 0;
}
