/* CPU test entry point for the native BF16 FlowMatch Euler update. */
#define main qimg21_runner_main
#include "test_cuda_qimg21_native.c"
#undef main

int main(int argc,char **argv) {
    if(argc==5 && !strcmp(argv[1],"--schedule")) {
        int steps=atoi(argv[2]),tokens=atoi(argv[3]);
        if(steps<1 || steps>100 || tokens<1)return 2;
        float sigmas[101];qimg21_flow_sigmas(steps,tokens,sigmas);
        return npy_write_f32(argv[4],sigmas,(size_t)steps+1,steps+1,1);
    }
    if(argc!=6) {
        fprintf(stderr,"usage: %s sample.npy prediction.npy sigma next_sigma out.npy\n",argv[0]);
        return 2;
    }
    npy_f32 sample={0},prediction={0};
    if(npy_read_f32(argv[1],&sample)||npy_read_f32(argv[2],&prediction))return 1;
    if(sample.n!=prediction.n || sample.ndim!=2 || sample.shape[1]!=64)return 2;
    qimg21_euler_bf16(sample.data,prediction.data,sample.n,strtof(argv[3],NULL),strtof(argv[4],NULL));
    int rc=npy_write_f32(argv[5],sample.data,sample.n,(int)sample.shape[0],64);
    npy_free(&sample);npy_free(&prediction);return rc;
}
