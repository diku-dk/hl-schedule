CXX = nvcc -O3 -arch=sm_60
LIB = -lcublas -lcurand
#-arch=sm_60

SRC = main-cublas.cu
EXECUTABLE = mmm-cublas

default: compile run

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC) $(HELPERS)
	$(CXX) -o $(EXECUTABLE) $(SRC) $(LIB)


run: $(EXECUTABLE)
	./$(EXECUTABLE) 2048 4096 2048

clean:
	rm -f $(EXECUTABLE)

