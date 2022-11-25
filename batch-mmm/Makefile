CXX        = nvcc -O3

#-arch=sm_60

SRC = main.cu
HELPERS = kernels.cu.h
EXECUTABLE =bmmm

default: compile run

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC) $(HELPERS)
	$(CXX) -o $(EXECUTABLE) $(SRC)


run: $(EXECUTABLE)
	./$(EXECUTABLE) 131000 8 8 512

clean:
	rm -f $(EXECUTABLE)

