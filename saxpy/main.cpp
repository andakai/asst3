#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include "CycleTimer.h"

extern void saxpySerial(int N, float a, float* X, float* Y, float* result);
void saxpyCuda(int N, float alpha, float* x, float* y, float* result);
void printCudaInfo();


void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}

static float
toBW(int bytes, float sec) {
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

static float
toGFLOPS(int ops, float sec) {
    return static_cast<float>(ops) / 1e9 / sec;
}

int main(int argc, char** argv)
{

    // default: arrays of 100M numbers
    int N = 1000 * 1000 * 1000;
    const unsigned int TOTAL_BYTES = 4 * N * sizeof(float);
    const unsigned int TOTAL_FLOPS = 2 * N;
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"arraysize",  1, 0, 'n'},
        {"help",       0, 0, '?'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "?n:", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'n':
            N = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    const float alpha = 2.0f;
    float* xarray = new float[N];
    float* yarray = new float[N];
    float* resultarray = new float[N];

    for (int i=0; i<N; i++) {
        xarray[i] = yarray[i] = i % 10;
        resultarray[i] = 0.f;
   }

    printCudaInfo();
    
    printf("Running 3 timing tests for saxpySerial:\n");
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime =CycleTimer::currentSeconds();
        saxpySerial(N, alpha, xarray, yarray, resultarray);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    
    printf("[saxpy serial]:\t\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
          minSerial * 1000,
          toBW(TOTAL_BYTES, minSerial),
          toGFLOPS(TOTAL_FLOPS, minSerial));

    printf("Running 3 timing tests for saxpyCuda:\n");
    for (int i=0; i<3; i++) {
      saxpyCuda(N, alpha, xarray, yarray, resultarray);
    }


    delete [] xarray;
    delete [] yarray;
    delete [] resultarray;

    return 0;
}
