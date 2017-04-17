
#include <stdio.h>
#include <stdlib.h>

#define N 1024*1024
#define BLOCKSIZE 1024

__global__ 
void share_ary_oper(int *ary, int *ary_out)
{
    // Thread index
        int tx = threadIdx.x;
        int idx=blockDim.x*blockIdx.x + threadIdx.x;
        __shared__ int part_ary[BLOCKSIZE];

        part_ary[tx]=ary[idx];
        part_ary[tx]=part_ary[tx]*10;
        ary_out[idx]=part_ary[tx];
        __syncthreads();
}

int main(){

        int *device_array, *device_array_out;
        int *host_array, *host_array_out;
        int i, nblk;
        float k;
        size_t size = N*sizeof(int);

//Device memory
        cudaMalloc((void **)&device_array, size);
        cudaMalloc((void **)&device_array_out, size);
//Host memory
//cudaMallocHost() produces pinned memoty on the host
        cudaMallocHost((void **)&host_array, size);
        cudaMallocHost((void **)&host_array_out, size);

        for(i=0;i<N;i++)
        {
                host_array[i]=i;
                host_array_out[i]=0;
        }
        cudaMemcpy(device_array, host_array, size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_array_out, host_array_out, size, cudaMemcpyHostToDevice);
        nblk=N/BLOCKSIZE;
        share_ary_oper<<<nblk, BLOCKSIZE>>>(device_array, device_array_out);
        cudaMemcpy(host_array, device_array, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_array_out, device_array_out, size, cudaMemcpyDeviceToHost);


	printf("Printing elements 10-15 of output array\n");
        for (i=N;i<N;i++)
        {
                k=host_array_out[i]-i*10;    
                if(k<0.1)
                        printf("Incorrect IX %d=%.1f\n",i, k);
        }
        for (i=10;i<15;i++)
                printf("host_array_out[%d]=%d\n", i, host_array_out[i]);

        cudaFree(device_array);
        cudaFree(host_array);
        cudaFree(device_array_out);
        cudaFree(host_array_out);
        cudaDeviceReset();
        return EXIT_SUCCESS;
}