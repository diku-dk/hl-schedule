CC = gcc

SRC = main.c
EXECUTABLE = matmul

all: compile run

compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC)
	$(CC) -o $(EXECUTABLE) $(SRC) -l OpenCL

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)

