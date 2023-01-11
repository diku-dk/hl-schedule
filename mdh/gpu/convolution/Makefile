CXX = nvcc -O3

SRC = main.cu
EXECUTABLE = mcc

all: compile run

compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC)
	$(CXX) -o $(EXECUTABLE) $(SRC) -l cuda -l nvrtc

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)

