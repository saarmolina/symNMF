CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

all: symnmf

symnmf: symnmf.o
	$(CC) symnmf.o -o symnmf -lm

symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c

clean:
	rm -f *.o symnmf

.PHONY: all clean