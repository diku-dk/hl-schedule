CXX        = nvcc -O3 

#-arch=sm_60

SRC = mmm-main.cu
HELPERS = mmm-kernels.cu.h
EXECUTABLE =mmm

default: compile run

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC) $(HELPERS)
	$(CXX) -o $(EXECUTABLE) $(SRC)


run: $(EXECUTABLE)
	./$(EXECUTABLE) 2048 4096 2048

clean:
	rm -f $(EXECUTABLE)

