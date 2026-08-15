# 电脑端串口控制程序

该程序通过 USB-TTL 串口向 STM32 发送单字节命令：

- `1`：PA0 输出高电平，继电器吸合，电磁铁通电
- `0`：PA0 输出低电平，继电器断开，电磁铁断电

程序连接后先发送一次 `0`，确保从已知的断电状态开始。此后每按一次 Enter（或空格），电磁铁状态切换一次。按 `Ctrl+C` 退出。

## 1. 断电检查接线

USB-TTL 与 STM32：

- USB-TTL `TXD` -> STM32 `PA10`（USART1_RX）
- USB-TTL `RXD` -> STM32 `PA9`（USART1_TX）
- USB-TTL `GND` -> STM32 `GND`
- USB-TTL 应使用 3.3 V TTL 电平，不要把 5 V 信号直接接入 STM32

继电器弱电侧：

- 继电器 `DC+` -> STM32 `3.3V`
- 继电器 `DC-` -> STM32 `GND`
- 继电器 `IN1` -> STM32 `PA0`
- 触发选择跳线保持 `COM` 与 `H` 短接，即高电平触发

电磁铁 12 V 回路：

- 12 V 正极 -> 继电器 `COM1`
- 继电器 `NO1` -> 电磁铁线 A
- 电磁铁线 B -> 12 V 负极
- 按现有方案将 12 V 负极与 STM32 GND 共地

接线时必须断开 12 V 电源，严禁把 12 V 接到 STM32 引脚或 3.3 V 电源轨。

## 2. 编译并烧录 STM32

1. 使用 Keil 打开 `stm32标准库/project.uvprojx`。
2. 确认目标器件为 `STM32F103C8`。
3. 点击 Build，工程会编译 `User/main.c` 和 `User/stm32f10x_it.c`。
4. 连接 ST-Link，点击 Download 烧录程序。
5. STM32 复位后 PA0 默认为低电平，电磁铁应保持断电。

固件使用标准外设库，系统时钟 72 MHz，USART1 参数为 115200、8 数据位、1 停止位、无校验、无流控。

## 3. 安装电脑端依赖

在项目根目录打开 PowerShell：

```powershell
python -m pip install -r ".\电脑端代码\requirements.txt"
```

如果 `python` 命令不可用，可尝试：

```powershell
py -m pip install -r ".\电脑端代码\requirements.txt"
```

## 4. 查找串口号

插入 USB-TTL 后执行：

```powershell
python ".\电脑端代码\controller.py" --list
```

也可以在 Windows 设备管理器的“端口（COM 和 LPT）”中查看，例如 `COM3`、`COM5`。

## 5. 运行和操作

将 `COM3` 替换成实际端口：

```powershell
python ".\电脑端代码\controller.py" COM3
```

操作顺序：

1. 程序打开串口并等待 1.5 秒，让可能被串口复位的 STM32 完成启动。
2. 程序发送 `0`，屏幕显示“断电（无磁力）”。
3. 第一次按 Enter，程序发送 `1`，电磁铁变为通电、有磁力。
4. 再按一次 Enter，程序发送 `0`，电磁铁变为断电、无磁力。
5. 继续按 Enter，可以反复切换。
6. 按 `Ctrl+C` 退出。若退出时处于通电状态，程序会尝试先发送 `0`。

屏幕输出的时间戳是电脑提交串口命令的时刻。它不等于继电器触点或单摆实际动作的时刻，因为 USB 串口、继电器和机械机构都会产生延迟。

## 6. 常见问题

- 提示“无法打开 COM 口”：检查串口号，并关闭串口助手、Keil 串口窗口等占用该端口的软件。
- 找不到串口：安装对应 USB-TTL 的 CH340 或 CP210x 驱动，然后重新插拔模块。
- 按键后没有动作：检查 TXD 是否接 PA10、GND 是否共地、波特率是否为 115200。
- 继电器状态相反：检查触发跳线是否确实为 `COM-H` 高电平触发，以及负载是否接在 `COM1-NO1`。
- STM32 重启或工作不稳定：确认继电器模块可由 3.3 V 稳定驱动，强电回路没有把 12 V 串入 STM32 电源轨。
