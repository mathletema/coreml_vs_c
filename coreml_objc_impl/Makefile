CC = clang++
CFLAGS = -O0 -Wall -Werror -Wno-unused-variable
LDFLAGS = -framework Foundation -framework CoreML 
DEPS = main.mm
TARGET = main

main: main.mm
	$(CC) $(CFLAGS) -o $(TARGET) $(DEPS) $(LDFLAGS)

clean:
	rm -f $(TARGET)