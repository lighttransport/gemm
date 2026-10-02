/* Independent scalar/double checks for newly shared image primitives. */
#define VH_MODELS_NO_MAIN
#include "../../common/vhuman_models.c"

int main(void)
{
    omp_set_num_threads(2);
    vh_tensor x=vh_new(2,3,4);
    for(size_t i=0;i<vh_size(x);i++)x.d[i]=sinf(i*.7f);
    float coords[]={-1.1f,-.8f,0,.2f,1.5f,2.9f,3.2f,4.1f};double error=0;
    for(int c=0;c<2;c++)for(int a=0;a<8;a++)for(int b=0;b<8;b++) {
        double ref=0;
        for(int y=0;y<3;y++)for(int col=0;col<4;col++)
            ref+=fmax(0,1-fabs(coords[a]-y))*fmax(0,1-fabs(coords[b]-col))*x.d[(c*3+y)*4+col];
        error=fmax(error,fabs(ref-vh_bilinear_zero(x,c,coords[a],coords[b])));
    }
    printf("deform sampling max_abs=%.9g\n",error);if(error>1e-6)return 1;
    float weights[57];for(int i=0;i<57;i++)weights[i]=cosf(i*.2f)*.1f;
    st_tensor_info tensors[2]={
        {.name="conv.weight",.dtype_str="F32",.shape={3,2,3,3},.n_dims=4,.offset=0,.nbytes=54*4},
        {.name="conv.bias",.dtype_str="F32",.shape={3},.n_dims=1,.offset=54*4,.nbytes=3*4}};
    st_context st={.tensors=tensors,.n_tensors=2,.data=(uint8_t*)weights};
    for(int replicate=0;replicate<2;replicate++)for(int stride=1;stride<=2;stride++) {
        vh_tensor y=vh_conv(&st,"conv",x,stride,1,replicate);error=0;
        for(int c=0;c<3;c++)for(int py=0;py<y.h;py++)for(int px=0;px<y.w;px++) {
            double ref=weights[54+c];
            for(int ic=0;ic<2;ic++)for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++) {
                int iy=py*stride+ky-1,ix=px*stride+kx-1;
                if(replicate){iy=iy<0?0:iy>2?2:iy;ix=ix<0?0:ix>3?3:ix;}
                if(iy>=0 && iy<3 && ix>=0 && ix<4)ref+=(double)x.d[(ic*3+iy)*4+ix]*weights[((c*2+ic)*3+ky)*3+kx];
            }
            error=fmax(error,fabs(ref-y.d[(c*y.h+py)*y.w+px]));
        }
        printf("conv replicate=%d stride=%d max_abs=%.9g\n",replicate,stride,error);
        vh_drop(y);if(error>1e-6)return 1;
    }
    /* Constant preservation includes antialias edge normalization. */
    for(size_t i=0;i<vh_size(x);i++)x.d[i]=.37f;
    for(int h=1;h<=7;h+=2)for(int w=1;w<=9;w+=2) {
        vh_tensor y=vh_resize_aa(x,h,w);
        for(size_t i=0;i<vh_size(y);i++)if(fabsf(y.d[i]-.37f)>2e-7)return 1;
        vh_drop(y);
    }
    vh_drop(x);puts("PASS");return 0;
}
