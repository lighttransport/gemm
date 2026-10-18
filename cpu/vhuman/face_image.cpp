/* Included by face.cpp: bounded RGB sampling and detector postprocessing. */
extern "C" int vhf_image_sample(const uint8_t *rgb,int h,int w,const double *center,
    const double *size,double angle,int resolution,int zero_border,float *out) {
    try {
        require(rgb && out && h>=2 && w>=2 && size_t(h)*w<=4000000 && resolution>=1 && resolution<=2048 && std::isfinite(angle),"invalid image sample");
        finite(center,2);finite(size,2);require(size[0]>0 && size[1]>0 && size[0]<1e7 && size[1]<1e7 && std::abs(center[0])<1e7 && std::abs(center[1])<1e7,"invalid crop rectangle");
        double cs=std::cos(angle),sn=std::sin(angle);
        for(int y=0;y<resolution;++y)for(int x=0;x<resolution;++x) {
            double u=(double(x)/resolution-.5)*size[0],v=(double(y)/resolution-.5)*size[1];
            // Match the pinned OpenCV INTER_LINEAR warp's 5-bit interpolation table.
            int qx=int(std::nearbyint((center[0]+cs*u-sn*v)*32)),qy=int(std::nearbyint((center[1]+sn*u+cs*v)*32));
            int ix=int(std::floor(qx/32.)),iy=int(std::floor(qy/32.)),fx=qx-ix*32,fy=qy-iy*32;
            for(int c=0;c<3;++c) {
                int sum=0;
                for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx) {
                    int xx=ix+dx,yy=iy+dy;
                    if(zero_border && (xx<0||xx>=w||yy<0||yy>=h))continue;
                    xx=std::clamp(xx,0,w-1);yy=std::clamp(yy,0,h-1);
                    sum+=(dx?fx:32-fx)*(dy?fy:32-fy)*rgb[(size_t(yy)*w+xx)*3+c];
                }
                out[(size_t(y)*resolution+x)*3+c]=float((sum+512)>>10)/255.f;
            }
        }
        return 0;
    } catch(const std::exception &e) { error=e.what();return -1; }
}
extern "C" int vhf_image_resize(const float *src,int h,int w,int channels,int oh,int ow,int nearest,int quantize,float *out) {
    try {
        require(out && h>0 && w>0 && oh>0 && ow>0 && channels>0 && channels<=32 &&
                size_t(h)*w<=4000000 && size_t(oh)*ow<=4000000 && (nearest==0||nearest==1),"invalid resize dimensions");
        finite(src,size_t(h)*w*channels);
        if(quantize)for(size_t i=0;i<size_t(h)*w*channels;++i)
            require(src[i]>=0 && src[i]<=255 && std::floor(src[i])==src[i],"quantized resize requires byte values");
        for(int y=0;y<oh;++y)for(int x=0;x<ow;++x) {
            double py=nearest?double(y)*h/oh:std::max(0.,(y+.5)*h/oh-.5);
            double px=nearest?double(x)*w/ow:std::max(0.,(x+.5)*w/ow-.5);
            int iy=int(py),ix=int(px);double fy=py-iy,fx=px-ix;
            auto at=[&](int yy,int xx,int c) {return src[(size_t(std::min(yy,h-1))*w+std::min(xx,w-1))*channels+c];};
            for(int c=0;c<channels;++c) {
                double v=nearest?at(iy,ix,c):(1-fy)*((1-fx)*at(iy,ix,c)+fx*at(iy,ix+1,c))+fy*((1-fx)*at(iy+1,ix,c)+fx*at(iy+1,ix+1,c));
                if(quantize && !nearest) {
                    int a0=int(std::nearbyint((1-fx)*2048)),a1=int(std::nearbyint(fx*2048));
                    int b0=int(std::nearbyint((1-fy)*2048)),b1=int(std::nearbyint(fy*2048));
                    int s0=int(at(iy,ix,c))*a0+int(at(iy,ix+1,c))*a1;
                    int s1=int(at(iy+1,ix,c))*a0+int(at(iy+1,ix+1,c))*a1;
                    v=(((b0*(s0>>4))>>16)+((b1*(s1>>4))>>16)+2)>>2;
                }
                out[(size_t(y)*ow+x)*channels+c]=float(quantize?std::floor(v+.5):v);
            }
        }
        return 0;
    } catch(const std::exception &e) { error=e.what();return -1; }
}
extern "C" int vhf_face_detections(const float *raw,float threshold,float *out,int capacity) {
    try {
        require(out && capacity>0 && capacity<=896 && std::isfinite(threshold) && threshold>=0 && threshold<=1,"invalid detector output");finite(raw,896*17);
        std::array<std::array<float,16>,896> box{};std::array<float,896> score{};std::vector<int> ids;
        int id=0;
        for(int level=0;level<2;++level) {
            int grid=level?8:16,repeat=level?6:2;
            for(int y=0;y<grid;++y)for(int x=0;x<grid;++x)for(int k=0;k<repeat;++k,++id) {
                float ax=(x+.5f)/grid,ay=(y+.5f)/grid;
                for(int c=0;c<16;++c)box[id][c]=raw[id*16+c]/128+(c==2||c==3?0:(c%2?ay:ax));
                score[id]=1.f/(1+std::exp(-std::clamp(raw[896*16+id],-80.f,80.f)));
                if(score[id]>=threshold)ids.push_back(id);
            }
        }
        std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return score[a]>score[b];});int count=0;
        while(!ids.empty() && count<capacity) {
            int top=ids[0];double sum=0;std::array<double,16> merged{};std::vector<int> remaining;
            const auto &a=box[top];
            for(int j:ids) {
                const auto &b=box[j];double intersection=1;
                for(int axis=0;axis<2;++axis)intersection*=std::max(0.f,std::min(a[axis]+a[axis+2]/2,b[axis]+b[axis+2]/2)-std::max(a[axis]-a[axis+2]/2,b[axis]-b[axis+2]/2));
                double area=std::max(0.f,a[2])*std::max(0.f,a[3])+std::max(0.f,b[2])*std::max(0.f,b[3]);
                if(j==top || intersection/std::max(area-intersection,1e-12)>.3) { sum+=score[j];for(int c=0;c<16;++c)merged[c]+=b[c]*score[j]; }
                else remaining.push_back(j);
            }
            for(int c=0;c<16;++c)out[count*17+c]=float(merged[c]/sum);
            out[count*17+16]=score[top];++count;ids.swap(remaining);
        }
        return count;
    } catch(const std::exception &e) { error=e.what();return -1; }
}
