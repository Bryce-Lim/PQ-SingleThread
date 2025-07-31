# Makefile for AMX Inner Product with Arrow/Parquet support
CXX = g++
CXXFLAGS = -flax-vector-conversions -fopenmp -std=c++17 -O2 -march=native -fno-strict-aliasing -mavx512bf16

# Find Arrow/Parquet include directories
ARROW_INCLUDE = $(shell pkg-config --cflags arrow)
PARQUET_INCLUDE = $(shell pkg-config --cflags parquet)
ARROW_LIBS = $(shell pkg-config --libs arrow)
PARQUET_LIBS = $(shell pkg-config --libs parquet)

# If pkg-config doesn't work, try these common paths
ifeq ($(ARROW_INCLUDE),)
    ARROW_INCLUDE = -I/usr/include/arrow -I/usr/local/include/arrow
endif
ifeq ($(PARQUET_INCLUDE),)
    PARQUET_INCLUDE = -I/usr/include/parquet -I/usr/local/include/parquet
endif
ifeq ($(ARROW_LIBS),)
    ARROW_LIBS = -larrow
endif
ifeq ($(PARQUET_LIBS),)
    PARQUET_LIBS = -lparquet
endif

INCLUDES = $(ARROW_INCLUDE) $(PARQUET_INCLUDE)
ALL_LIBS = $(ARROW_LIBS) $(PARQUET_LIBS)

# Object files
OBJECTS = large_testing.o AMXInnerProductBF16.o ScalarInnerProduct.o

# Targets
all: large_testing

AMXInnerProductBF16.o: AMXInnerProductBF16.cpp AMXInnerProductBF16.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c AMXInnerProductBF16.cpp -o AMXInnerProductBF16.o

ScalarInnerProduct.o: ScalarInnerProduct.cpp ScalarInnerProduct.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c ScalarInnerProduct.cpp -o ScalarInnerProduct.o

large_testing.o: large_testing.cpp AMXInnerProductBF16.h ScalarInnerProduct.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c large_testing.cpp -o large_testing.o

large_testing: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(ALL_LIBS) -o large_testing

clean:
	rm -f *.o large_testing

.PHONY: all clean
