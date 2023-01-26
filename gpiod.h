#ifndef GPIOD_H_
#define GPIOD_H_

#ifdef __cplusplus
extern "C"
{
#endif

	/*************************************************************************************
	* Denna headerfil utgör enbart dummyfiler för kompilering i Windowsmiljö.
	* Ta inte med dessa filer vid kompilering i Linuxmiljö, utan inkludera i stället
	* biblioteket gpiod.h genom att lägga till -l gpiod vid kompilering av projektet!
	*
	* Som exempel, för att kompilera samtliga C-filer i en katalog med GCC-kompilatorn
	* och skapa en körbar fil döpt main kan följande kommando användas:
	*
	* gcc *c -o main -Wall -l gpiod
	**************************************************************************************/

	static struct gpiod_chip* gpiod_chip_open(const char* path) { return 0; }
	static struct gpiod_chip* gpiod_chip_open_by_name(const char* name) { return 0; }
	static struct gpiod_chip* gpiod_chip_open_by_number(unsigned int number) { return 0; }
	static struct gpiod_line* gpiod_chip_get_line(struct gpiod_chip* chip, unsigned int offset) { return 0; }
	static void gpiod_chip_close(struct gpiod_chip* chip) { }
	static int gpiod_line_request_input(struct gpiod_line* line, const char* consumer) { return 0; }
	static int gpiod_line_request_output(struct gpiod_line* line, const char* consumer, int default_val) { return 0; }
	static int gpiod_line_set_value(struct gpiod_line* line, int value) { return 0; }
	static int gpiod_line_get_value(struct gpiod_line* line) { return 0; }
	static void gpiod_line_release(struct gpiod_line* line) { }
	static const char* gpiod_line_consumer(struct gpiod_line* line) { return 0; }
	static unsigned int gpiod_line_offset(struct gpiod_line* line) { return 0; }

#ifdef __cplusplus
}
#endif

#endif /* GPIOD_H_ */