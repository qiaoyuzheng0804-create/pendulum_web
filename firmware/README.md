# 外设硬件资料（firmware/）

本目录收录「实验获取」模式涉及的全部外设硬件源码、固件与文档，保证仓库在新机器上可完整复原整套实验装置，不依赖任何外部目录。

| 子目录 | 设备 | 用途与内容 |
|---|---|---|
| `stm32标准库/` + `电脑端代码/` | 单摆电磁铁释放（STM32F103C8T6） | Keil 固件工程 + 命令行测试工具，详见下方「STM32 单摆电磁铁释放固件」 |
| `云台夹爪/` | 磁阻尼摆 / 扭摆释放（二维云台＋夹爪，STM32F103C8T6） | Keil 工程 `Keil工程/USER/GIMBAL.uvprojx`、预编译固件 `Keil工程/OBJ/GIMBAL.hex`（免 Keil 直接烧录）、`接线与使用说明.md`、`烧录步骤.md`、PC 测试工具 `pc_keyboard.py` / `test_upper_serial.py` |
| `openmv/` | OpenMV 拍摄 + 舵机云台（OpenMV4 H7 Plus） | 相机端部署脚本 `OpenMV端/main.py`（需复制到相机存储运行）、USBDBG V1 电脑端工具（`电脑端/`）、USB 驱动（`驱动/`），详见 `openmv/README.md` |

三套设备与网页的对应关系：单摆 → 电磁铁（`/api/serial/*`）；磁阻尼摆 / 扭摆 → 云台夹爪（`/api/gripper/*`）；拍摄预览 / 录制 / 舵机云台 → OpenMV（`/api/openmv/*`，双串口架构）。

---

# STM32 单摆电磁铁释放固件（STM32F103C8T6）

本项目是「STM32 电磁铁零抖动释放」的固件源码与电脑端工具，供网页端「实验获取」模式控制电磁铁使用。网页通过串口发送单字节命令，本固件接收后控制继电器通断电磁铁。

## 目录结构

```
firmware/
├── README.md               # 本总览 + 电磁铁固件说明（需求、串口协议、接线、烧录）
├── stm32标准库/             # 电磁铁 Keil 固件工程（与原始工程结构一致）
│   ├── project.uvprojx     # Keil 工程文件，双击用 Keil 打开、编译、烧录
│   ├── User/               # main.c + stm32f10x_it.c（释放逻辑本体）+ stm32f10x_conf.h
│   ├── System/             # 延时模块（Delay.c/h）
│   ├── start/              # 启动文件（startup_stm32f10x_md.s）、寄存器定义（stm32f10x.h）、72MHz 时钟配置
│   └── library/            # ST 标准外设库（GPIO/USART/RCC/NVIC 等封装）
├── 电脑端代码/              # 电磁铁命令行测试工具（controller.py，可选；网页已集成同等功能）
├── openmv/                 # OpenMV 相机端脚本 + 电脑端工具 + USB 驱动（详见 openmv/README.md）
└── 云台夹爪/                # 二维云台＋夹爪固件与文档（详见 云台夹爪/README.md）
```

## 串口协议

波特率 **115200**，8 数据位，1 停止位，无校验，无流控。

| 命令 | 动作 |
|---|---|
| `'1'` (0x31) | PA0 高电平 → 继电器吸合 → 电磁铁通电，吸住小球 |
| `'0'` (0x30) | PA0 低电平 → 继电器断开 → 电磁铁断电，**释放小球** |

## 接线

USB-TTL → STM32（3.3V TTL 电平）：
- `TXD` → `PA10`（USART1_RX）
- `RXD` → `PA9`（USART1_TX）
- `GND` → `GND`

继电器弱电侧：
- `DC+` → 3.3V，`DC-` → GND，`IN1` → `PA0`
- 触发跳线保持 `COM` 与 `H` 短接（高电平触发）

电磁铁 12V 强电回路（与 STM32 电源轨隔离，共地）：
- 12V 正极 → 继电器 `COM1`，`NO1` → 电磁铁线 A，电磁铁线 B → 12V 负极

## 烧录步骤

1. Keil 打开 `stm32标准库/project.uvprojx`，确认目标器件 `STM32F103C8`
2. Build 编译生成 `.hex`（工程文件均为相对路径，移动目录后仍可直接编译）
3. ST-Link 连接，点击 Download 烧录
4. 复位后 PA0 默认为低电平（电磁铁断电）

## 电脑端控制

- 网页「实验获取」模式：连接串口后点「吸合」/「释放」按钮（功能由网站后端 `/api/serial/command` 提供）
- 命令行测试：`python 电脑端代码/controller.py <COM口>`（详见 `电脑端代码/README.md`）
