#include "gpiod_line.hpp"

/* Todo: Slutf�r optimeringsfunktionen i klassen ann s� att parametrarna korrigeras vid fel.
 */

static void read_button(gpiod_line* button, vector<double>& data, const size_t index);
static int get_multi_output(ann& multi1, const vector<double>& input);

int main(void)
{
	/* Fel i tr�ningsupps�ttningarna, korrigerade dem: */
	const vector<vector<double>> button_in = {
		{0,0,0,0}, {0,0,0,1}, {0,0,1,0}, {0,0,1,1}, 
        {0,1,0,0}, {0,1,0,1}, {0,1,1,1}, {1,0,0,0},
		{1,0,0,0}, {1,0,0,1}, {1,0,1,0}, {1,0,1,1}, 
        {1,1,0,0}, {1,1,0,1}, {1,1,1,0}, {1,1,1,1} 
    };

	const vector<vector<double>> diod_out = {
		{0}, {1}, {1}, {0}, 
        {1}, {0}, {1}, {1},
        {1}, {0}, {0}, {1}, 
        {0}, {1}, {1}, {0} };
	
	ann multi1 (4, 1, 4, 1);
	multi1.set_training_data(button_in, diod_out);
	multi1.train(80000, 0.03);
	multi1.print();


    /* Array f�r lagring av tryckknapparnas tillst�nd */
	vector<double>input(4, 0);

    struct gpiod_line* led = gpiod_line_new(17, GPIO_DIRECTION_OUT, "led");
    struct gpiod_line* button1 = gpiod_line_new(27, GPIO_DIRECTION_IN, " BUtton1");
    struct gpiod_line* button2 = gpiod_line_new(22, GPIO_DIRECTION_IN, " BUtton2");
    struct gpiod_line* button3 = gpiod_line_new(23, GPIO_DIRECTION_IN, " BUtton3");
    struct gpiod_line* button4 = gpiod_line_new(24, GPIO_DIRECTION_IN, " BUtton4");

    while (1)
    {
        read_button(button1, input, 3);
        read_button(button2, input, 2);
        read_button(button2, input, 1);
        read_button(button2, input, 0);
        gpiod_line_set_value(led, get_multi_output(multi1, input));
    }


    return 0;

}


/***
* Funktionen read_button: L�ser insignal p� inport (button) och l�gger i vector data
* n�r den �r 1 respektive 0. index d� h�ller koll p� vilken index i arrayen ska h�g
* respektive l�g signal sparas.
**/
static void read_button(gpiod_line* button, vector<double>& data, const size_t index) {
    if (gpiod_line_get_value(button))
    {
        data[index] = 1;
    }
    else
    {
        data[index] = 0;
    }

    return;
}

/***
* Funktionen get_multi_output: Via funktionen kopplas buttonstillst�nd till n�tverket,
* som input av n�tverket och sedan preditionen p� den retuneras tillbaka.
**/
static int get_multi_output(ann& multi1, const vector<double>& input) {
    const auto prediction = multi1.predict(input);
    return static_cast<int>(prediction[0] + 0.5);
}
	
	






