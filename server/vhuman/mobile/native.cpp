#include "native.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

struct vh_mobile {
    uint32_t vertices=0, expressions=0;
    std::vector<float> bind, basis, corrective, weights, scratch;
    std::array<int32_t,4> parents{};
    std::array<float,12> joints{};
    std::array<float,9> frame{};
    std::array<float,3> origin{};
    float scale=1;
    std::array<float,48> joint_transforms{};
    bool evaluated=false;
};
static bool read_floats(FILE *f,float *p,size_t n) {
    if (fread(p,sizeof(float),n,f)!=n) return false;
    for(size_t i=0;i<n;i++) if(!std::isfinite(p[i])) return false;
    return true;
}
extern "C" vh_mobile *vh_mobile_load(const char *path) {
    if(!path)return nullptr;
    auto close_file=[](FILE *f){fclose(f);};
    std::unique_ptr<FILE,decltype(close_file)> f(fopen(path,"rb"),close_file);
    if(!f) return nullptr;
    char magic[8]; uint32_t dims[4];
    if(fread(magic,1,8,f.get())!=8 || memcmp(magic,"VHGNM001",8) ||
       fread(dims,4,4,f.get())!=4 || dims[0]!=17821 || dims[1]!=383 || dims[2]!=4 || dims[3]!=36) return nullptr;
    try {
        auto m=std::make_unique<vh_mobile>();m->vertices=dims[0];m->expressions=dims[1];
        if(fread(m->parents.data(),4,4,f.get())!=4 || m->parents!=std::array<int32_t,4>{-1,0,1,1}) return nullptr;
        if(!read_floats(f.get(),&m->scale,1) || m->scale<=0 ||
           !read_floats(f.get(),m->frame.data(),9) || !read_floats(f.get(),m->origin.data(),3) ||
           !read_floats(f.get(),m->joints.data(),12)) return nullptr;
        for(int i=0;i<3;i++)for(int j=0;j<3;j++) {
            float dot=0;for(int k=0;k<3;k++)dot+=m->frame[i*3+k]*m->frame[j*3+k];
            if(std::abs(dot-(i==j?1.f:0.f))>1e-4f)return nullptr;
        }
        const size_t n=size_t(m->vertices)*3;
        m->bind.resize(n);m->basis.resize(n*m->expressions);m->corrective.resize(n*36);
        m->weights.resize(size_t(m->vertices)*4);m->scratch.resize(n);
        for(auto *a:{&m->bind,&m->basis,&m->corrective,&m->weights})
            if(!read_floats(f.get(),a->data(),a->size())) return nullptr;
        if(fgetc(f.get())!=EOF) return nullptr;
        for(size_t v=0;v<m->vertices;v++) {
            float sum=0;
            for(size_t j=0;j<4;j++) {float w=m->weights[j*m->vertices+v];if(w<0 || w>1.0001f)return nullptr;sum+=w;}
            if(std::abs(sum-1)>1e-4f)return nullptr;
        }
        return m.release();
    } catch(...) {return nullptr;}
}
extern "C" void vh_mobile_free(vh_mobile *m) {delete m;}
extern "C" size_t vh_mobile_vertices(const vh_mobile *m) {return m?m->vertices:0;}
extern "C" size_t vh_mobile_expressions(const vh_mobile *m) {return m?m->expressions:0;}
static void mul(const float *a,const float *b,float *r) {
    for(int i=0;i<3;i++)for(int j=0;j<3;j++) {
        r[i*3+j]=0;for(int k=0;k<3;k++)r[i*3+j]+=a[i*3+k]*b[k*3+j];
    }
}
static void mv(const float *a,const float *b,float *r) {
    for(int i=0;i<3;i++)r[i]=a[i*3]*b[0]+a[i*3+1]*b[1]+a[i*3+2]*b[2];
}
static void rotation(const float *v,float *r) {
    float theta=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    float a=theta<1e-4f?1-theta*theta/6:std::sin(theta)/theta;
    float b=theta<1e-4f?.5f-theta*theta/24:(1-std::cos(theta))/(theta*theta);
    float k[9]={0,-v[2],v[1],v[2],0,-v[0],-v[1],v[0],0},kk[9];mul(k,k,kk);
    for(int i=0;i<9;i++)r[i]=(i%4==0?1.f:0.f)+a*k[i]+b*kk[i];
}
extern "C" int vh_mobile_eval(vh_mobile *m,const float *expression,const float *rotations,const float *translation,float *out) {
    if(!m || !expression || !rotations || !translation || !out)return -1;
    m->evaluated=false;
    for(size_t i=0;i<m->expressions;i++)if(!std::isfinite(expression[i]) || std::abs(expression[i])>3.0001f)return -2;
    for(int i=0;i<12;i++)if(!std::isfinite(rotations[i]) || std::abs(rotations[i])>3.15f)return -2;
    for(int i=0;i<3;i++)if(!std::isfinite(translation[i]) || std::abs(translation[i])>10)return -2;
    float local[36],world[36],joint[12],offset[12],pose[36];
    for(int j=0;j<4;j++) {
        rotation(rotations+j*3,local+j*9);
        for(int k=0;k<9;k++)pose[j*9+k]=local[j*9+k]-(k%4==0?1.f:0.f);
        int p=m->parents[j];
        if(p<0) {
            std::copy(local+j*9,local+j*9+9,world+j*9);
            for(int k=0;k<3;k++)joint[j*3+k]=m->joints[j*3+k]+translation[k];
        } else {
            mul(world+p*9,local+j*9,world+j*9);
            float delta[3];for(int k=0;k<3;k++)delta[k]=m->joints[j*3+k]-m->joints[p*3+k];
            mv(world+p*9,delta,joint+j*3);
            for(int k=0;k<3;k++)joint[j*3+k]+=joint[p*3+k];
        }
        mv(world+j*9,m->joints.data()+j*3,offset+j*3);
        for(int k=0;k<3;k++)offset[j*3+k]=joint[j*3+k]-offset[j*3+k];
    }
    const size_t n=size_t(m->vertices)*3;
    std::copy(m->bind.begin(),m->bind.end(),m->scratch.begin());
    for(size_t e=0;e<m->expressions;e++)if(expression[e]!=0)
        for(size_t i=0;i<n;i++)m->scratch[i]+=expression[e]*m->basis[e*n+i];
    for(size_t e=0;e<36;e++)if(pose[e]!=0)
        for(size_t i=0;i<n;i++)m->scratch[i]+=pose[e]*m->corrective[e*n+i];
    for(size_t v=0;v<m->vertices;v++) {
        float q[3]={0,0,0};
        for(int j=0;j<4;j++) {
            float t[3];mv(world+j*9,m->scratch.data()+v*3,t);
            float w=m->weights[j*m->vertices+v];
            for(int k=0;k<3;k++)q[k]+=w*(t[k]+offset[j*3+k]);
        }
        mv(m->frame.data(),q,out+v*3);
        for(int k=0;k<3;k++) {
            out[v*3+k]=m->scale*out[v*3+k]+m->origin[k];
            if(!std::isfinite(out[v*3+k]))return -3;
        }
    }
    float transposed[9];for(int i=0;i<3;i++)for(int j=0;j<3;j++)transposed[i*3+j]=m->frame[j*3+i];
    for(int j=0;j<4;j++) {
        float intermediate[9];float *affine=m->joint_transforms.data()+j*12;
        mul(m->frame.data(),world+j*9,intermediate);mul(intermediate,transposed,affine);
        float transformed_origin[3],native_offset[3];mv(affine,m->origin.data(),transformed_origin);
        mv(m->frame.data(),offset+j*3,native_offset);
        for(int k=0;k<3;k++)affine[9+k]=m->origin[k]-transformed_origin[k]+m->scale*native_offset[k];
    }
    m->evaluated=true;
    return 0;
}
extern "C" int vh_mobile_joint_transform(const vh_mobile *m,unsigned joint,float *out) {
    if(!m || !out || joint>=4 || !m->evaluated)return -1;
    std::copy_n(m->joint_transforms.data()+joint*12,12,out);return 0;
}
