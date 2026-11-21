NVCC ?= nvcc
NVCC_FLAGS = -O3 -std=c++17 -Xcompiler -Wall
TARGET = batch_edge_detector
SRC = src/main.cu
HEADERS = src/stb_image.h src/stb_image_write.h

all: $(HEADERS) $(TARGET)

$(HEADERS):
	mkdir -p src data/input data/output
	curl -sL -o src/stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
	curl -sL -o src/stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
	@echo "Downloaded stb_image headers successfully."

setup: $(HEADERS)

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f data/output/*.png data/output/*.csv

.PHONY: all setup clean
