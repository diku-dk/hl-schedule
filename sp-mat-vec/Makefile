CXX        = nvcc -O3

#-arch=sm_60

SRC = main.cu
HELPERS = kernels.cu.h ../helper.h
EXECUTABLE =spmv

default: compile run

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC) $(HELPERS)
	$(CXX) -o $(EXECUTABLE) $(SRC)


run: $(EXECUTABLE)
	./$(EXECUTABLE) 10000 10000

clean:
	rm -f $(EXECUTABLE)

