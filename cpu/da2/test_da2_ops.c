/* Test the DA2 GEMM/im2col boundary against independent double accumulation. */
#define main da2_runner_main
#include "../../common/da2_runner.c"
#undef main

static float sample(int i) { return ((i*17+13)%101-50)*.01f; }

int main(int argc,char **argv)
{
    if (argc==2 && !strcmp(argv[1],"--cuda")) {
#ifdef DA2_CUDA
        da2_gpu=1;
        if (da2_cuda_init(0)) return 2;
#else
        return 2;
#endif
    } else if (argc!=1) return 2;
    da2_threads=2;
    float x[11*13],w[7*13],bias[7],y[11*7];
    for (int i=0;i<11*13;i++) x[i]=sample(i);
    for (int i=0;i<7*13;i++) w[i]=sample(i+3);
    for (int i=0;i<7;i++) bias[i]=sample(i+8);
    da2_linear(y,w,bias,x,11,7,13,2);
    double max=0;
    /* torch.interpolate(scale_factor=(3.1/2,2.1/2), bicubic): border overshoot. */
    float pos[]={0,1,2,3}, resized[6];
    const float expected[]={-.1970901042f,.7627105117f,.9104358554f,1.870236874f,2.116253376f,3.076054096f};
    cpu_interp_pos_embed_bicubic(pos,2,resized,3,2,1);
    for (int i=0;i<6;i++) {
        double error=fabs(resized[i]-expected[i]); if (error>max) max=error;
    }
    for (int i=0;i<11;i++) for (int j=0;j<7;j++) {
        double ref=bias[j];
        for (int k=0;k<13;k++) ref+=(double)x[i*13+k]*w[j*13+k];
        double error=fabs(y[i*7+j]-ref); if (error>max) max=error;
    }
    float src[3*5*7],weights[5*3*3*3],b[5],dst[5*3*4];
    for (int i=0;i<3*5*7;i++) src[i]=sample(i);
    for (int i=0;i<5*3*3*3;i++) weights[i]=sample(i+2);
    for (int i=0;i<5;i++) b[i]=sample(i+4);
    da2_conv(dst,src,weights,b,5,7,3,5,3,3,2,1);
    for (int c=0;c<5;c++) for (int iy=0;iy<3;iy++) for (int ix=0;ix<4;ix++) {
        double ref=b[c];
        for (int ci=0;ci<3;ci++) for (int ky=0;ky<3;ky++) for (int kx=0;kx<3;kx++) {
            int sy=iy*2+ky-1,sx=ix*2+kx-1;
            if (sy>=0 && sy<5 && sx>=0 && sx<7)
                ref+=(double)weights[((c*3+ci)*3+ky)*3+kx]*src[(ci*5+sy)*7+sx];
        }
        double error=fabs(dst[(c*3+iy)*4+ix]-ref); if (error>max) max=error;
    }
    printf("DA2 GEMM + strided/padded convolution + bicubic borders: max_abs=%.9g %s\n",max,max<2e-5 ? "PASS" : "FAIL");
#ifdef DA2_CUDA
    da2_cuda_free();
#endif
    return max<2e-5 ? 0 : 1;
}
