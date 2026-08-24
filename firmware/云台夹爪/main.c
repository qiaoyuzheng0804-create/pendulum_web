#include "stm32f10x.h"
#include "delay.h"

/* STM32F103C8T6 + two EMM42 V5 drivers, common-anode pulse control. */
#define MOVE_PULSES          18U
#define PULSE_PERIOD_US      400U
#define PULSE_LOW_US         5U
#define DIR_SETUP_US         1000U
#define LOWER_REVERSE        0U
#define UPPER_REVERSE        0U

#define LOWER_PORT           GPIOB
#define LOWER_EN_PIN         GPIO_Pin_0
#define LOWER_STP_PIN        GPIO_Pin_1
#define LOWER_DIR_PIN        GPIO_Pin_10
#define UPPER_PORT           GPIOB
#define UPPER_EN_PIN         GPIO_Pin_3
#define UPPER_STP_PIN        GPIO_Pin_5
#define UPPER_DIR_PIN        GPIO_Pin_4

#define KEY_PORT             GPIOB
#define KEY_LEFT_PIN         GPIO_Pin_12
#define KEY_RIGHT_PIN        GPIO_Pin_13
#define KEY_UP_PIN           GPIO_Pin_14
#define KEY_DOWN_PIN         GPIO_Pin_15
#define KEY_GRIP_OPEN_PIN    GPIO_Pin_8
#define KEY_GRIP_CLOSE_PIN   GPIO_Pin_11
#define KEY_PRESSED_LEVEL    Bit_SET
#define GRIPPER_ID           1U
#define GRIPPER_SPEED_RPM10  1800U /* 180.0 RPM; C6 speed unit is 0.1 RPM. */
#define GRIPPER_CURRENT_MA   2000U

/* 0 = unknown, 1 = open, 2 = closed. */
static uint8_t gripper_state = 0U;

static void gpio_init(void)
{
    GPIO_InitTypeDef gpio;

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO |
                           RCC_APB2Periph_GPIOA | RCC_APB2Periph_GPIOB, ENABLE);
    /* PB3/PB4 are JTAG pins; keep SWD for ST-Link and release JTAG pins. */
    GPIO_PinRemapConfig(GPIO_Remap_SWJ_JTAGDisable, ENABLE);
    gpio.GPIO_Pin = LOWER_EN_PIN | LOWER_STP_PIN | LOWER_DIR_PIN;
    gpio.GPIO_Speed = GPIO_Speed_50MHz;
    gpio.GPIO_Mode = GPIO_Mode_Out_PP;
    GPIO_Init(LOWER_PORT, &gpio);

    gpio.GPIO_Pin = UPPER_EN_PIN | UPPER_STP_PIN | UPPER_DIR_PIN;
    GPIO_Init(UPPER_PORT, &gpio);

    GPIO_SetBits(LOWER_PORT, LOWER_EN_PIN | LOWER_STP_PIN | LOWER_DIR_PIN);
    GPIO_SetBits(UPPER_PORT, UPPER_EN_PIN | UPPER_STP_PIN | UPPER_DIR_PIN);

    gpio.GPIO_Pin = KEY_LEFT_PIN | KEY_RIGHT_PIN | KEY_UP_PIN | KEY_DOWN_PIN;
    gpio.GPIO_Speed = GPIO_Speed_2MHz;
    gpio.GPIO_Mode = GPIO_Mode_IPD;
    GPIO_Init(KEY_PORT, &gpio);

    gpio.GPIO_Pin = KEY_GRIP_OPEN_PIN;
    gpio.GPIO_Speed = GPIO_Speed_2MHz;
    gpio.GPIO_Mode = GPIO_Mode_IPD;
    GPIO_Init(GPIOA, &gpio);

    gpio.GPIO_Pin = KEY_GRIP_CLOSE_PIN;
    gpio.GPIO_Speed = GPIO_Speed_2MHz;
    gpio.GPIO_Mode = GPIO_Mode_IPD;
    GPIO_Init(GPIOB, &gpio);

    /* COM is at 3.3 V, so pulling EN low enables the driver. */
    GPIO_ResetBits(LOWER_PORT, LOWER_EN_PIN);
    GPIO_ResetBits(UPPER_PORT, UPPER_EN_PIN);
}

static void uart1_init(void)
{
    GPIO_InitTypeDef gpio;
    USART_InitTypeDef uart;

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA | RCC_APB2Periph_USART1, ENABLE);
    gpio.GPIO_Pin = GPIO_Pin_9;
    gpio.GPIO_Speed = GPIO_Speed_50MHz;
    gpio.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_Init(GPIOA, &gpio);
    gpio.GPIO_Pin = GPIO_Pin_10;
    gpio.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(GPIOA, &gpio);

    uart.USART_BaudRate = 115200U;
    uart.USART_WordLength = USART_WordLength_8b;
    uart.USART_StopBits = USART_StopBits_1;
    uart.USART_Parity = USART_Parity_No;
    uart.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    uart.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
    USART_Init(USART1, &uart);
    USART_Cmd(USART1, ENABLE);
}

static void pulse_move(GPIO_TypeDef *port, uint16_t step_pin,
                       uint16_t dir_pin, uint8_t direction)
{
    uint16_t i;

    if (direction != 0U) GPIO_SetBits(port, dir_pin);
    else GPIO_ResetBits(port, dir_pin);
    /* Give the driver time to latch DIR before the first STP edge. */
    delay_us(DIR_SETUP_US);

    for (i = 0U; i < MOVE_PULSES; ++i) {
        GPIO_ResetBits(port, step_pin);
        delay_us(PULSE_LOW_US);
        GPIO_SetBits(port, step_pin);
        delay_us(PULSE_PERIOD_US - PULSE_LOW_US);
    }
}

static void uart_send_byte(uint8_t byte)
{
    while (USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET) {}
    USART_SendData(USART1, byte);
}

/* Gripper C6 command: addr C6 dir accel speed sync current checksum. */
static void gripper_move(uint8_t direction)
{
    uint8_t frame[11];
    uint8_t i;

    /* Ignore duplicate commands once the requested end state was issued. */
    if ((direction != 0U) && (gripper_state == 1U)) return;
    if ((direction == 0U) && (gripper_state == 2U)) return;

    frame[0] = GRIPPER_ID;
    frame[1] = 0xC6U;
    frame[2] = direction;
    frame[3] = 0xFFU;
    frame[4] = 0xFFU;
    frame[5] = (uint8_t)(GRIPPER_SPEED_RPM10 >> 8);
    frame[6] = (uint8_t)(GRIPPER_SPEED_RPM10 & 0xFFU);
    frame[7] = 0x00U;
    frame[8] = (uint8_t)(GRIPPER_CURRENT_MA >> 8);
    frame[9] = (uint8_t)(GRIPPER_CURRENT_MA & 0xFFU);
    frame[10] = 0x6BU;

    for (i = 0U; i < 11U; ++i) uart_send_byte(frame[i]);
    if (direction != 0U) gripper_state = 1U;
    else gripper_state = 2U;
    delay_ms(150U);
}

static void execute_command(uint8_t command)
{
    switch (command) {
        case 'L': pulse_move(LOWER_PORT, LOWER_STP_PIN, LOWER_DIR_PIN, 0U ^ LOWER_REVERSE); break;
        case 'R': pulse_move(LOWER_PORT, LOWER_STP_PIN, LOWER_DIR_PIN, 1U ^ LOWER_REVERSE); break;
        case 'U': pulse_move(UPPER_PORT, UPPER_STP_PIN, UPPER_DIR_PIN, 0U ^ UPPER_REVERSE); break;
        case 'D': pulse_move(UPPER_PORT, UPPER_STP_PIN, UPPER_DIR_PIN, 1U ^ UPPER_REVERSE); break;
        case 'O': gripper_move(1U); break;
        case 'C': gripper_move(0U); break;
        default: break;
    }
}

static uint8_t read_key(void)
{
    if (GPIO_ReadInputDataBit(KEY_PORT, KEY_LEFT_PIN) == KEY_PRESSED_LEVEL) return 'L';
    if (GPIO_ReadInputDataBit(KEY_PORT, KEY_RIGHT_PIN) == KEY_PRESSED_LEVEL) return 'R';
    if (GPIO_ReadInputDataBit(KEY_PORT, KEY_UP_PIN) == KEY_PRESSED_LEVEL) return 'U';
    if (GPIO_ReadInputDataBit(KEY_PORT, KEY_DOWN_PIN) == KEY_PRESSED_LEVEL) return 'D';
    if (GPIO_ReadInputDataBit(GPIOA, KEY_GRIP_OPEN_PIN) == KEY_PRESSED_LEVEL) return 'O';
    if (GPIO_ReadInputDataBit(GPIOB, KEY_GRIP_CLOSE_PIN) == KEY_PRESSED_LEVEL) return 'C';
    return 0U;
}

int main(void)
{
    uint8_t key;
    uint8_t last_key = 0U;

    delay_init();
    gpio_init();
    uart1_init();

    while (1) {
        if (USART_GetFlagStatus(USART1, USART_FLAG_RXNE) != RESET) {
            execute_command((uint8_t)USART_ReceiveData(USART1));
        }

        key = read_key();
        if ((key != 0U) && (last_key == 0U)) {
            delay_ms(15U);
            if (read_key() == key) execute_command(key);
        }
        last_key = key;
    }
}
