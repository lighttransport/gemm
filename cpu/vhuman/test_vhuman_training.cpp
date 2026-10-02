/* Independent finite differences exercise the native GRU's full reverse pass. */
#include "training.cpp"
#include <cstdio>
#include <random>

static void close_enough(double a, double b, double absolute, double relative, const char *name)
{
    if (std::abs(a-b) > absolute+relative*std::max(std::abs(a),std::abs(b))) {
        std::fprintf(stderr,"%s: %.10g != %.10g\n",name,a,b);
        std::exit(1);
    }
}
int main()
{
    std::mt19937 rng(19);
    std::uniform_real_distribution<float> random(-.2f,.2f);
    for (int ta = 0; ta < 2; ++ta) for (int tb = 0; tb < 2; ++tb) {
        constexpr int m=5,n=7,k=9;
        float a[m*k],b[k*n],out[m*n];
        for (float &v:a) v=random(rng);
        for (float &v:b) v=random(rng);
        require(!vh_train_gemm(out,a,b,m,n,k,ta,tb),"GEMM failed");
        for (int i=0;i<m;++i) for (int j=0;j<n;++j) {
            double value=0;
            for (int p=0;p<k;++p) value+=double(a[ta?p*m+i:i*k+p])*b[tb?j*k+p:p*n+j];
            close_enough(out[i*n+j],value,1e-6,1e-6,"GEMM transpose");
        }
    }
    float p[]={.7f,-.3f},g[]={3.f,4.f},m[2]={},v[2]={};
    require(!vh_train_adamw(p,g,m,v,2,1,.01,.02,1),"AdamW failed");
    for (int i=0;i<2;++i) {
        double grad=g[i]/(5+1e-6),first=.1*grad,second=.001*grad*grad;
        double initial=i?-.3f:.7f;
        close_enough(p[i],initial*(1-.01*.02)-.01/(1-.9)*first/(std::sqrt(second)/std::sqrt(1-.999)+1e-8),1e-7,1e-6,"AdamW clip");
    }
    float old=p[0];g[0]=std::numeric_limits<float>::quiet_NaN();
    require(vh_train_adamw(p,g,m,v,2,2,.01,.02,1)==-1 && p[0]==old,"invalid optimizer must not mutate weights");

    constexpr int t=4,h=7,c=2,o=c*8;
    motion_trainer model(h,c);
    for (float &value:model.parameters) value=random(rng);
    float input[t*h],target[t*o],initial[256],state[256],prediction[t*o];
    int32_t codes[t*16];
    for (float &value:input) value=random(rng);
    for (float &value:target) value=random(rng)+.4f;
    for (float &value:initial) value=random(rng);
    for (int32_t &value:codes) value=int32_t(rng()%8); // Repeats test embedding gradient accumulation.
    float bounds[]={-.2f,.8f,0.f,1.f},weights[]={1.f,4.f};
    std::memcpy(state,initial,sizeof(state));
    double loss=model.compute(input,codes,target,bounds,weights,state,prediction,t,1,0);
    require(loss>0,"expected nontrivial loss");
    std::vector<float> analytic=model.gradients;
    for (int block=0;block<13;++block) {
        size_t best=model.offsets[block];
        for (size_t i=best;i<model.offsets[block+1];++i) if (std::abs(analytic[i])>std::abs(analytic[best])) best=i;
        float original=model.parameters[best],epsilon=.002f;
        double numeric[2];
        for (int sign=0;sign<2;++sign) {
            model.parameters[best]=original+(sign?epsilon:-epsilon);
            std::memcpy(state,initial,sizeof(state));
            numeric[sign]=model.compute(input,codes,target,bounds,weights,state,prediction,t,0,0);
        }
        model.parameters[best]=original;
        close_enough(analytic[best],(numeric[1]-numeric[0])/(2*epsilon),8e-6,.006,"GRU/embedding/projection/output gradient");
    }
    std::memcpy(state,initial,sizeof(state));
    model.compute(input,codes,target,bounds,weights,state,prediction,t,0,0);
    float chunk_state[256],chunk_prediction[t*o];std::memcpy(chunk_state,initial,sizeof(initial));
    model.compute(input,codes,target,bounds,weights,chunk_state,chunk_prediction,2,0,0);
    model.compute(input+2*h,codes+32,target+2*o,bounds,weights,chunk_state,chunk_prediction+2*o,2,0,0);
    for (int i=0;i<t*o;++i) close_enough(chunk_prediction[i],prediction[i],2e-6,2e-6,"chunk prediction");
    for (int i=0;i<256;++i) close_enough(chunk_state[i],state[i],2e-6,2e-6,"chunk recurrent state");
    std::puts("PASS: repository GEMM transposes, clipped AdamW, all 13 GRU parameter blocks, recurrent chunk state");
    return 0;
}
