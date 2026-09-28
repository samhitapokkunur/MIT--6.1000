#include <stdio.h>
#include "6190.h"

void setup(){

    timerSetup();   // Configure timer
    setupDisplay(); // Configure display
    eraseBuffer();  // Erase random data from display
    drawBuffer();   // Display contents of buffer on LED array
}


int messageLength(char* message){

    //////////////////////////////////////////
    // TO-DO: Complete messageLength()      //
    //////////////////////////////////////////

    return 0; // Replace!
}


void fillScreenBuffer(char* message, int total_offset){

    //////////////////////////////////////////
    // TO-DO: Complete fillScreenBuffer()   //
    //////////////////////////////////////////

}   


void app_main(){
    setup();

    char message[] = "     THEY SEE ME SCROLLIN' THEY HATIN'     ";
    int len = messageLength(message);  // TASK 1: CALCULATE MESSAGE LENGTH
    int offset = 0;
    while(1){
        if (offset == len * 8){  
            offset = 0; // Reset if we have reached the end of the message
        } else {
            eraseBuffer();  // Clear screen buffer
            fillScreenBuffer(message, offset); // TASK 2: FILL SCREEN BUFFER WITH CORRECT DATA
            drawBuffer();   // Transmit screen buffer data to screen
            offset++;        // Move forward
        }

        int start = millis();            // Get "start" time stamp
        while(millis() - start < 100);   // Wait until (current time stamp - "start") >= 100ms   
    }
}
