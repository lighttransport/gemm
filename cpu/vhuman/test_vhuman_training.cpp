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
    // Independent finite differences across all four corrective MLP blocks.
    constexpr int bn=3,bi=4,bh=5,bo=2;
    constexpr size_t mlp_count=bh*bi+bh+bo*bh+bo;
    float mx[bn*bi],mp[mlp_count],up[bn*bo],mg[mlp_count],my[bn*bo];
    for (float &value:mx) value=random(rng);
    for (float &value:mp) value=random(rng);
    for (float &value:up) value=random(rng);
    require(!vh_train_mlp(my,mg,mx,mp,up,bn,bi,bh,bo),"native MLP backward failed");
    const size_t mlp_offsets[]={0,bh*bi,bh*bi+bh,bh*bi+bh+bo*bh,mlp_count};
    for (int block=0;block<4;++block) {
        size_t best=mlp_offsets[block];
        for (size_t i=best;i<mlp_offsets[block+1];++i) if (std::abs(mg[i])>std::abs(mg[best])) best=i;
        float original=mp[best],epsilon=.001f;double numeric[2]={};
        for (int sign=0;sign<2;++sign) {
            mp[best]=original+(sign?epsilon:-epsilon);
            require(!vh_train_mlp(my,nullptr,mx,mp,nullptr,bn,bi,bh,bo),"native MLP forward failed");
            for (int i=0;i<bn*bo;++i) numeric[sign]+=double(my[i])*up[i];
        }
        mp[best]=original;
        close_enough(mg[best],(numeric[1]-numeric[0])/(2*epsilon),3e-6,.005,"corrective MLP gradient");
    }
    // One-vertex squared sphere penetration has an independent closed form.
    float sx[]={.3f,.4f,0},center[]={0,0,0},threshold[]={.8f},sg[3],depth[1],energy[1];int32_t si[]={0};
    require(!vh_train_spheres(sx,si,center,threshold,1,1,1,1,sg,depth,energy),"sphere gradient failed");
    close_enough(energy[0],.09,1e-7,1e-6,"sphere energy");
    close_enough(sg[0],-.36,1e-7,1e-6,"sphere x gradient");
    close_enough(sg[1],-.48,1e-7,1e-6,"sphere y gradient");
    float px[]={0,0,0,0,1,0},axis[]={0,1,0},floor[]={0},pg[6];int32_t upper[]={0},lower[]={1};
    require(!vh_train_pairs(px,upper,lower,axis,floor,1,2,1,pg,depth,energy),"lip pair failed");
    close_enough(energy[0],1,1e-7,1e-6,"pair energy");
    close_enough(pg[1],-2,1e-7,1e-6,"upper pair gradient");
    close_enough(pg[4],2,1e-7,1e-6,"lower pair gradient");
    float rest[]={0,0,0,1,0,0,0,1,0,0,0,1};
    int32_t mesh_edges[]={0,1,0,2,0,3,1,2,1,3,2,3},mesh_faces[]={0,2,1,0,1,3,0,3,2,1,2,3};
    float edge_rest[18],normal_rest[12]={},normal_scale[4]={},rg[12],rot[36];
    for (int e=0;e<6;++e) {
        int a=mesh_edges[e*2],b=mesh_edges[e*2+1];float norm2=0;
        for (int j=0;j<3;++j) { float d=rest[b*3+j]-rest[a*3+j];edge_rest[e*3+j]=d;norm2+=d*d; }
        normal_scale[a]+=norm2/3;normal_scale[b]+=norm2/3;
    }
    for (int f=0;f<4;++f) {
        float a[3],b[3],n[3];
        for (int j=0;j<3;++j) { a[j]=rest[mesh_faces[f*3+1]*3+j]-rest[mesh_faces[f*3]*3+j];b[j]=rest[mesh_faces[f*3+2]*3+j]-rest[mesh_faces[f*3]*3+j]; }
        cross3(a,b,n);
        for (int c=0;c<3;++c) for (int j=0;j<3;++j) normal_rest[mesh_faces[f*3+c]*3+j]+=n[j];
    }
    for (int vtx=0;vtx<4;++vtx) {
        float *n=normal_rest+vtx*3;float length=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        for (int j=0;j<3;++j) n[j]/=length;
    }
    require(!vh_train_arap(rest,rest,mesh_edges,edge_rest,1,4,6,.4,rg,energy),"ARAP failed");
    close_enough(energy[0],0,1e-7,0,"rest ARAP energy");
    for (float value:rg) close_enough(value,0,1e-7,0,"rest ARAP gradient");
    require(!vh_train_rotations(rest,edge_rest,normal_rest,normal_scale,mesh_edges,mesh_faces,1,4,6,4,rot),"polar rotation failed");
    for (int vtx=0;vtx<4;++vtx) for (int j=0;j<3;++j) for (int k=0;k<3;++k)
        close_enough(rot[vtx*9+j*3+k],j==k?1:0,2e-6,0,"rest polar identity");
    std::puts("PASS: repository GEMM, AdamW, GRU gradients/state, corrective MLP gradients and contact energy");
    return 0;
}
