# OpenMV 实时画面、YOLO 接口与键盘云台

适用硬件：OpenMV4 H7 Plus（现有固件 `4.6.20`）、龙飞 1.8 寸 LCD
扩展板、两个 MG90S。

运行时不需要打开 OpenMV IDE。电脑通过板载数据 USB-C 使用 OpenMV
USBDBG V1 协议读取实时 JPEG 帧；方向键通过同一协议的 `TX_INPUT` 通道发送。
不要同时打开 OpenMV IDE 和电脑端程序，两者不能同时占用 `COM12`。

## 固件限制

本方案按现有 `4.6.20` 固件开发和验证。程序不包含刷固件、进入 bootloader
或自动升级功能。不要使用 IDE 5.0 附带的新版固件覆盖当前固件。

## 已确认接线

- 下层水平舵机信号线：`P1`
- 上层俯仰舵机信号线：`P9`
- 舵机红线：稳定 `+5V`
- 舵机棕线：`GND`
- 舵机电源与 OpenMV 必须共地

两个 MG90S 建议使用稳定 5V、至少 2A 的独立电源。不要从 OpenMV 3.3V
引脚给舵机供电。

## E 盘与 F 盘

本机实测 `E:` 是插在 OpenMV 中的 SD 卡。它以前装过树莓派系统，所以卷内
有 `kernel*.img`、`start*.elf` 等树莓派文件并显示为 boot。`.openmv_disk`
表明 OpenMV 也将它作为存储盘；当前实际启动脚本是 `E:\main.py`。

`F:` 是另一块 OpenMV 存储，目前只有 `3.pgm.pgm`。因为 SD 卡存在，当前
IDE 的“保存到 OpenMV Cam（作为 main.py）”会保存到 E 盘。

不要删除 E 盘中的树莓派文件。部署只替换 `E:\main.py`。

## 文件说明

- `OpenMV端/main.py`：LCD、P1/P9 PWM、同 USB 按键接收。
- `OpenMV端/main_E_backup_20260815.py`：修改前 E 盘脚本的字节级备份。
- `电脑端/openmv_v1.py`：4.6.20 使用的 USBDBG V1 协议客户端。
- `电脑端/openmv_v1_probe.py`：只读版本/取帧探针。
- `电脑端/controller.py`：实时显示、键盘控制与可选 YOLO 推理。

旧架构把 JPEG 直接写入普通串口，会阻塞 OpenMV 并使 LCD 定格。本方案不再
通过 `USB_VCP.write()` 或 UART 发送 JPEG，而是读取固件帧缓冲。

## 已完成实测

在 `COM12`、固件 `4.6.20`、QVGA 条件下：

- USBDBG V1 版本握手成功。
- 连续读取 20/20 帧成功。
- 每帧约 6.5 KiB JPEG。
- 探针平均约 13 FPS。
- 首帧已保存为 `电脑端/probe_frame.jpg`。

## 运行

先关闭 OpenMV IDE，然后在仓库根目录运行：

```powershell
python "firmware\openmv\电脑端\controller.py" --port COM12
```

也可以省略 `--port COM12`，程序会按 OpenMV USB VID 自动识别。

- `↑/↓`：P9 上层俯仰舵机
- `←/→`：P1 下层水平舵机
- `空格`：两个舵机回中

每次物理按键只发送一个命令；程序会抑制按住按键时的系统自动连发。每次
命令改变 `5 us` PWM 脉宽，这是 MG90S 常见死区量级，但实际最小可见角度还会
受舵机个体差异、齿隙和负载影响。

## YOLO

普通显示不需要安装 OpenCV。每一帧在 `controller.py` 中已经解码为
`numpy.ndarray` RGB 数据，变量为 `VideoFrame.rgb`，可直接交给电脑端 YOLO。

安装 YOLO 依赖（仓库根目录 `requirements.txt` 已包含）后，可传入模型：

```powershell
python -m pip install -r requirements.txt
python "firmware\openmv\电脑端\controller.py" --port COM12 --model "models\danbai_best.pt"
```

YOLO 在独立线程中运行。推理跟不上相机时只覆盖旧帧，不阻塞 USB 取帧和按键
命令。QVGA 适合先验证流程；检测小目标时应再实测 VGA 的清晰度和帧率。
