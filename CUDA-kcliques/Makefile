CC := /usr/local/cuda/bin/nvcc
CFLAGS := -O3 -arch=sm_70 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70 --compiler-options "-Wall -Wextra -Werror"
all: kcliques

kcliques: kcliques.cu
	$(CC) $(CFLAGS) kcliques.cu -o kcliques 

clean:
	$(RM) kcliques *.o