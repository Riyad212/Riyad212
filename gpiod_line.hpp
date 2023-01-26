#pragma once

#include "dense_layer.hpp"
#include "ann.hpp"
#include "unistd.h"
#include "gpiod.h"





enum gpiod_direction { GPIO_DIRECTION_IN, GPIO_DIRECTION_OUT };



/*
* gpiod_line_new: Initierar GPIO-linje till in- eller utport och returnerar en
* GPIO-linjepekare. Passerat namn sparas som GPIO-linjens alias.
* gpiochip0 implementeras via en statisk lokal struktpekare döpt
* chip0, som enbart används för att aktivera GPIO-linjer och hålls
* icke åtkomlig för användare.
**/
struct gpiod_line* gpiod_line_new(const uint8_t pin, const enum gpiod_direction direction, const char* alias) {
    static struct gpiod_chip* chip0 = 0;
    if (!chip0) chip0 = gpiod_chip_open("/dev/gpiochip0");

    struct gpiod_line* self = gpiod_chip_get_line(chip0, pin);

    if (direction == GPIO_DIRECTION_OUT) {
        gpiod_line_request_output(self, alias, 0);
    }
    else {
        gpiod_line_request_input(self, alias);
    }

    return self;
}

/*
* gpiod_line_delay: Genererar fördröjning mätt i millisekunder
**/
void gpiod_line_delay(const size_t delay_time) {
    usleep(delay_time * 1000);
    return;
}



/*
* gpiod_line_toggle: togglar lysdioden
**/
void gpiod_line_toggle(struct gpiod_line* self) {
    if (gpiod_line_get_value(self) == 1) {
        gpiod_line_set_value(self, 0);
    }
    else {
        gpiod_line_set_value(self, 1);
    }
}
