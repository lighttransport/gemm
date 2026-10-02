/* Included by training.cpp: original v3 cue CNN and analytic reverse pass. */
struct cue_feature {
    int n,c,h,w;
    std::vector<float> x;
    cue_feature(int bn,int channels,int height,int width):n(bn),c(channels),h(height),w(width),x(size_t(bn)*channels*height*width) {}
    size_t at(int b,int ch,int y,int col) const { return ((size_t(b)*c+ch)*h+y)*w+col; }
};
struct cue_layer {
    int input,output,kernel,stride;
    size_t weight,bias;
};
static std::vector<float> cue_columns(const cue_feature &a,const cue_layer &layer,int h,int w)
{
    int k=a.c*layer.kernel*layer.kernel;
    std::vector<float> columns(size_t(a.n)*h*w*k);
    for (int b=0;b<a.n;++b) for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        size_t row=((size_t(b)*h+y)*w+x)*k;int j=0;
        for (int c=0;c<a.c;++c) for (int ky=0;ky<layer.kernel;++ky) for (int kx=0;kx<layer.kernel;++kx,++j) {
            int iy=y*layer.stride+ky-layer.kernel/2,ix=x*layer.stride+kx-layer.kernel/2;
            columns[row+j]=iy>=0&&iy<a.h&&ix>=0&&ix<a.w ? a.x[a.at(b,c,iy,ix)] : 0;
        }
    }
    return columns;
}
static cue_feature cue_conv(const cue_feature &a,const cue_layer &layer,const float *parameters)
{
    cue_feature out(a.n,layer.output,(a.h+layer.stride-1)/layer.stride,(a.w+layer.stride-1)/layer.stride);
    auto columns=cue_columns(a,layer,out.h,out.w);int pixels=a.n*out.h*out.w,k=a.c*layer.kernel*layer.kernel;
    std::vector<float> rows(size_t(pixels)*layer.output);
    gemm(rows.data(),columns.data(),parameters+layer.weight,pixels,layer.output,k,false,true);
    for (int b=0;b<a.n;++b) for (int y=0;y<out.h;++y) for (int x=0;x<out.w;++x) for (int c=0;c<out.c;++c)
        out.x[out.at(b,c,y,x)]=rows[((size_t(b)*out.h+y)*out.w+x)*out.c+c]+parameters[layer.bias+c];
    return out;
}
static cue_feature cue_conv_reverse(const cue_feature &a,const cue_feature &dy,const cue_layer &layer,
                                     const float *parameters,float *gradient)
{
    int pixels=a.n*dy.h*dy.w,k=a.c*layer.kernel*layer.kernel;
    auto columns=cue_columns(a,layer,dy.h,dy.w);
    std::vector<float> rows(size_t(pixels)*dy.c),dcolumns(columns.size());
    for (int b=0;b<a.n;++b) for (int y=0;y<dy.h;++y) for (int x=0;x<dy.w;++x) for (int c=0;c<dy.c;++c) {
        float value=dy.x[dy.at(b,c,y,x)];rows[((size_t(b)*dy.h+y)*dy.w+x)*dy.c+c]=value;gradient[layer.bias+c]+=value;
    }
    gemm(gradient+layer.weight,rows.data(),columns.data(),dy.c,k,pixels,true,false);
    gemm(dcolumns.data(),rows.data(),parameters+layer.weight,pixels,k,dy.c);
    cue_feature dx(a.n,a.c,a.h,a.w);
    for (int b=0;b<a.n;++b) for (int y=0;y<dy.h;++y) for (int x=0;x<dy.w;++x) {
        size_t row=((size_t(b)*dy.h+y)*dy.w+x)*k;int j=0;
        for (int c=0;c<a.c;++c) for (int ky=0;ky<layer.kernel;++ky) for (int kx=0;kx<layer.kernel;++kx,++j) {
            int iy=y*layer.stride+ky-layer.kernel/2,ix=x*layer.stride+kx-layer.kernel/2;
            if (iy>=0&&iy<a.h&&ix>=0&&ix<a.w) dx.x[dx.at(b,c,iy,ix)]+=dcolumns[row+j];
        }
    }
    return dx;
}
struct cue_sample { int a,b;float weight; };
static cue_sample cue_coordinate(int i,int input,int output)
{
    float position=std::max(0.f,(i+.5f)*input/output-.5f);int a=int(position);
    return {std::min(a,input-1),std::min(a+1,input-1),position-a};
}
static cue_feature cue_resize(const cue_feature &a,int h,int w)
{
    cue_feature out(a.n,a.c,h,w);
    for (int b=0;b<a.n;++b) for (int c=0;c<a.c;++c) for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        auto u=cue_coordinate(y,a.h,h),v=cue_coordinate(x,a.w,w);
        out.x[out.at(b,c,y,x)]=(1-u.weight)*((1-v.weight)*a.x[a.at(b,c,u.a,v.a)]+v.weight*a.x[a.at(b,c,u.a,v.b)])+
            u.weight*((1-v.weight)*a.x[a.at(b,c,u.b,v.a)]+v.weight*a.x[a.at(b,c,u.b,v.b)]);
    }
    return out;
}
static cue_feature cue_resize_reverse(const cue_feature &dy,int h,int w)
{
    cue_feature out(dy.n,dy.c,h,w);
    for (int b=0;b<dy.n;++b) for (int c=0;c<dy.c;++c) for (int y=0;y<dy.h;++y) for (int x=0;x<dy.w;++x) {
        auto u=cue_coordinate(y,h,dy.h),v=cue_coordinate(x,w,dy.w);float d=dy.x[dy.at(b,c,y,x)];
        out.x[out.at(b,c,u.a,v.a)]+=d*(1-u.weight)*(1-v.weight);out.x[out.at(b,c,u.a,v.b)]+=d*(1-u.weight)*v.weight;
        out.x[out.at(b,c,u.b,v.a)]+=d*u.weight*(1-v.weight);out.x[out.at(b,c,u.b,v.b)]+=d*u.weight*v.weight;
    }
    return out;
}
extern "C" int vh_train_cue_resize(const float *input,float *out,int n,int channels,int ih,int iw,int oh,int ow)
{
    try {
        require(out&&n>0&&n<=8&&channels>0&&channels<=6&&ih>0&&ih<=128&&iw>0&&iw<=128&&oh>0&&oh<=128&&ow>0&&ow<=128,"invalid cue resize dimensions");
        finite(input,size_t(n)*channels*ih*iw);cue_feature feature(n,channels,ih,iw);
        std::memcpy(feature.x.data(),input,feature.x.size()*sizeof(float));auto result=cue_resize(feature,oh,ow);
        std::memcpy(out,result.x.data(),result.x.size()*sizeof(float));return 0;
    } catch(const std::exception &e) { train_error=e.what();return -1; }
}
extern "C" int vh_train_cues(const float *parameters,const float *inputs,const float *truth,const float *mask,
                              int n,int h,int w,float *out,float *gradient,double *loss)
{
    try {
        require(n>0&&n<=8&&h>0&&h<=128&&w>0&&w<=128&&out&&loss,"invalid cue training dimensions");
        int ic[]={6,16,24,32,24},oc[]={16,24,32,24,4},kernel[]={5,3,3,3,1},stride[]={1,2,2,1,1};
        cue_layer layers[5];size_t count=0;
        for (int i=0;i<5;++i) { size_t weight=count;count+=size_t(ic[i])*oc[i]*kernel[i]*kernel[i];layers[i]={ic[i],oc[i],kernel[i],stride[i],weight,count};count+=oc[i]; }
        finite(parameters,count);finite(inputs,size_t(n)*6*h*w);
        if (gradient) { finite(truth,size_t(n)*3*h*w);finite(mask,size_t(n)*h*w);std::fill(gradient,gradient+count,0.f); }
        std::vector<cue_feature> activations,preactivation;activations.reserve(6);preactivation.reserve(5);
        activations.emplace_back(n,6,h,w);std::memcpy(activations[0].x.data(),inputs,activations[0].x.size()*sizeof(float));
        for (int i=0;i<5;++i) {
            auto z=cue_conv(activations.back(),layers[i],parameters);preactivation.push_back(z);
            if (i<4) for (float &value:z.x) value*=sigmoid(value);
            activations.push_back(std::move(z));
        }
        auto raw=cue_resize(activations.back(),h,w);cue_feature draw(n,4,h,w);
        double denominator=0,total=0;
        if (gradient) for (size_t i=0;i<size_t(n)*h*w;++i) { require(mask[i]>=0&&mask[i]<=1,"invalid cue mask");denominator+=mask[i]; }
        denominator=std::max(denominator,1.);
        for (int b=0;b<n;++b) for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
            float v[3],tanh_raw[3],normal[3],norm2=0;
            for (int c=0;c<3;++c) { tanh_raw[c]=std::tanh(raw.x[raw.at(b,c,y,x)]);v[c]=inputs[activations[0].at(b,c+3,y,x)]+.15f*tanh_raw[c];norm2+=v[c]*v[c]; }
            float norm=std::sqrt(norm2),divisor=std::max(norm,1e-6f);
            for (int c=0;c<3;++c) { normal[c]=v[c]/divisor;out[raw.at(b,c,y,x)]=normal[c]; }
            float logit=raw.x[raw.at(b,3,y,x)];out[raw.at(b,3,y,x)]=logit;
            if (gradient) {
                size_t pixel=(size_t(b)*h+y)*w+x;float m=mask[pixel],dn[3],dot=0;
                total+=.15*(std::max(logit,0.f)-logit*m+std::log1p(std::exp(-std::abs(logit))))/(n*h*w);
                float cosine=0;
                for (int c=0;c<3;++c) {
                    float target=truth[((size_t(b)*3+c)*h+y)*w+x],prior=inputs[activations[0].at(b,c+3,y,x)],difference=normal[c]-prior;
                    cosine+=normal[c]*target;total+=.02*double(difference)*difference*m/denominator;
                    dn[c]=float((-target+.04*difference)*m/denominator);dot+=dn[c]*normal[c];
                }
                total+=(1-cosine)*m/denominator;
                for (int c=0;c<3;++c) draw.x[draw.at(b,c,y,x)]=(dn[c]-(norm>=1e-6f?normal[c]*dot:0))/divisor*.15f*(1-tanh_raw[c]*tanh_raw[c]);
                draw.x[draw.at(b,3,y,x)]=.15f*(sigmoid(logit)-m)/(n*h*w);
            }
        }
        if (gradient) {
            auto dy=cue_resize_reverse(draw,activations.back().h,activations.back().w);
            for (int i=4;i>=0;--i) {
                if (i<4) for (size_t j=0;j<dy.x.size();++j) { float z=preactivation[i].x[j],s=sigmoid(z);dy.x[j]*=s*(1+z*(1-s)); }
                dy=cue_conv_reverse(activations[i],dy,layers[i],parameters,gradient);
            }
            finite(gradient,count);
        }
        finite(out,size_t(n)*4*h*w);require(std::isfinite(total),"nonfinite cue loss");*loss=total;
        return 0;
    } catch (const std::exception &e) { train_error=e.what();return -1; }
}
