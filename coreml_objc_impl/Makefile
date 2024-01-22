CC = clang++
INCLUDES = -I.
DEPS = main.o helper.o
TARGET = main

default: $(TARGET)

$(TARGET): $(DEPS)
	$(CC) -g -o $(TARGET) $(INCLUDES) $(DEPS)

%.o: %.mm
	$(CC) -c -o $@ $(INCLUDES) $<

clean:
	rm -f $(DEPS) $(TARGET)