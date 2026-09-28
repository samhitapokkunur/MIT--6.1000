#include <stdio.h>
#include <stdint.h>

/* Header File for 6.190, using the ESP32-C3 */
/* ESP32-C3 Reference: https://www.espressif.com/sites/default/files/documentation/esp32-c3_technical_reference_manual_en.pdf */

///////////////////////////////
// Peripheral Base Addresses //
///////////////////////////////

#define GPIO_BASE_ADDR                      0x60004000  // Base address for memory locations dedicated to GPIO matrix 
#define IO_MUX_BASE_ADDR                    0x60009000  // Base address for memory locations dedicated to IO multiplexer addresses
#define TIMER_BASE_ADDR                     0x6001F000  // Base address for memory locations dedicated to timer 


///////////////////////////////////////////////
// General-Purposed Input/Output Definitions //
///////////////////////////////////////////////

/* GPIO Addresses */
#define GPIO_OUT_ADDR                       (GPIO_BASE_ADDR + 0x0004)
#define GPIO_ENABLE_ADDR                    (GPIO_BASE_ADDR + 0x0020)
#define GPIO_IN_ADDR                        (GPIO_BASE_ADDR + 0x003C)
#define IO_MUX_GPIOn_ADDR                   (IO_MUX_BASE_ADDR + 0x0004) // 1 memory location/pin, need to add 4n for pin n

/* GPIO Pin Modes */
#define GPIO_INPUT 0
#define GPIO_OUTPUT 1

/* GPIO -> Peripheral Mapping */

// Switches
#define SW7 9
#define SW6 8
#define SW5 7
#define SW4 6
#define SW3 5
#define SW2 4
#define SW1 18
#define SW0 19

// Buttons
#define BTNF 9
#define BTNU 8
#define BTNC 7
#define BTND 6
#define BTNR 5
#define BTNL 4

// LEDs
#define LED1 2
#define LED2 3


/* GPIO Functions */
void pinSetup(int pin_num, int mode){
    /* Setup function to configure GPIO pin (pin_num) as either an input
    or an output, as defined by mode.
    
    Arguments: 
        pin_num: GPIO pin number (0 - 31)
        mode: Mode to configure pin as. GPIO_INPUT = 0, GPIO_OUTPUT = 1
    */   

    if (mode == GPIO_INPUT){
        // Configure GPIO pin pin_num as input 
        int* io_mux_gpio_addr = (int*) (IO_MUX_GPIOn_ADDR + (4*pin_num));
        *io_mux_gpio_addr =  // YOUR CODE HERE
    } else if (mode == GPIO_OUTPUT){  
        // Configure GPIO pin pin_num as output.
        int* gpio_enable_addr = (int*) GPIO_ENABLE_ADDR;
        *gpio_enable_addr =  // YOUR CODE HERE
    }
}


void pinWrite(int pin_num, int value){
    /* Function to write either 0 or 1 (value) to an output GPIO pin (pin_num)
    Arguments:
        pin_num: GPIO pin number (0 - 31)
        value: digital value (0 or 1) to write to GPIO pin
    */

    int *output_val = (int*) GPIO_OUT_ADDR;
    if (value != 0) value = 1; // set non-zero inputs to 1 (HIGH)

    // TODO: YOUR CODE HERE...

}


int pinRead(int pin_num){
    /* Function to read the value of an input GPIO pin (pin_num). 
    Arguments: 
        pin_num: GPIO pin number (0 - 31)
    Returns: 
        int: digital value (0 or 1) read from GPIO pin
    */

   int* input_val = (int*) GPIO_IN_ADDR;

   // YOUR CODE HERE...

   return 0; // replace 0 with return value of pinRead
   
}

///////////////////////
// Timer Definitions //
///////////////////////

/* Timer Addresses */
#define TIMER_CONFIG_ADDR               (TIMER_BASE_ADDR + 0x0000)
#define TIMER_COUNT_LO_ADDR             (TIMER_BASE_ADDR + 0x0004)
#define TIMER_COUNT_HI_ADDR             (TIMER_BASE_ADDR + 0x0008)
#define TIMER_COUNT_UPDATE_ADDR         (TIMER_BASE_ADDR + 0x000C)

#define WATCHDOG_CONFIG_ADDR            (TIMER_BASE_ADDR + 0x0048)
#define WATCHDOG_WRITE_PROTECT_ADDR     (TIMER_BASE_ADDR + 0x0064)

/* Timer Functions */
void timerSetup(void){
    /* Setup function to initalize Timer 0 in the ESP32-C3 Timer Group. */

    // Configure timer by choosing clock source (external clock),  prescalar value (0x01) and direction (increment). 
    int* timg_t0_config = (int*) TIMER_CONFIG_ADDR;
    *timg_t0_config = (*timg_t0_config) | (0x01 << 9) | (0x01 << 13) | (0x01 << 30);

    // Start timer by setting most sigificant bit (MSB) of value at timer configuration memory address 
    *timg_t0_config |= (0x01 << 31);

    // Turn off watchdog timer timeout messages 
    int* wdog_addr = (int*) WATCHDOG_WRITE_PROTECT_ADDR;
    *wdog_addr = 0x50d83aa1;

    int* wdog_cfg = (int*) WATCHDOG_CONFIG_ADDR;
    *wdog_cfg &= ~(0x01 << 31); // clear bit 31
}


uint64_t millis(void){
    /* Converts the value of the counter driven by a 40MHz clock
    to a time-stamp in milliseconds.*/
    
    // write to MSB of value at timer counter update address to latch (update/retain) value of counter
    int* timg_to_update = (int*) TIMER_COUNT_UPDATE_ADDR;
    *timg_to_update |= (1 << 31);
   
    // (2) actually read... 
    int* timer_lo = (int*) TIMER_COUNT_LO_ADDR; // read low 32 bits of counter value
    int* timer_hi = (int*) TIMER_COUNT_HI_ADDR; // upper 22 bits of counter value
    uint32_t timer_lo_val = (uint32_t) *timer_lo;
    uint32_t timer_hi_val = (uint32_t) *timer_hi;
    uint64_t bigger_timer_hi = (uint64_t) timer_hi_val;

    return ((bigger_timer_hi << 32) + timer_lo_val) >> 14;
}


uint64_t micros(void){
    /* Converts the value of the counter driven by a 40MHz clock
    to a time-stamp in microseconds.*/
    
    // write to MSB of value at timer counter update address to latch (update/retain) value of counter
    int* timg_to_update = (int*) TIMER_COUNT_UPDATE_ADDR;
    *timg_to_update |= (1 << 31);
   
    int* timer_lo = (int*) TIMER_COUNT_LO_ADDR; // read low 32 bits of counter value
    int* timer_hi = (int*) TIMER_COUNT_HI_ADDR; // upper 22 bits of counter value
    uint32_t timer_lo_val = (uint32_t) *timer_lo;
    uint32_t timer_hi_val = (uint32_t) *timer_hi;
    uint64_t bigger_timer_hi = (uint64_t) timer_hi_val;

    return ((bigger_timer_hi << 32) + timer_lo_val) >> 4;
}
