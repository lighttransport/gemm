/* Included by face.cpp: fixed-camera objectives and explicit derivatives. */
namespace {
void project_point(const float *p,const double *camera,double *xy,double &depth,double *jac) {
    double q[3]{};for(int a=0;a<3;++a)for(int b=0;b<3;++b)q[a]+=camera[a*3+b]*(p[b]-camera[9+b]);
    depth=-q[2];double z=std::max(depth,.02),fx=camera[12],fy=camera[13],skew=camera[16];
    double nx=fx*q[0]-skew*q[1],ny=-fy*q[1];xy[0]=nx/z+camera[14];xy[1]=ny/z+camera[15];
    if(jac)for(int b=0;b<3;++b) {
        jac[b]=(fx*camera[b]-skew*camera[3+b])/z+(depth>=.02?nx*camera[6+b]/(z*z):0);
        jac[3+b]=-fy*camera[3+b]/z+(depth>=.02?ny*camera[6+b]/(z*z):0);
    }
}
V cross(const V&a,const V&b) {return V{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
double dot3(const V&a,const V&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
}
extern "C" int vhf_project(const float *points,const double *cameras,int frames,int count,float *pixels,float *depths) {
    try {
        require(frames>0 && frames<=1024 && count>0 && count<=65536 && pixels && depths,"invalid projection dimensions");finite(points,size_t(frames)*count*3);finite(cameras,size_t(frames)*17);
        for(int f=0;f<frames;++f)for(int a=0;a<count;++a){size_t k=size_t(f)*count+a;double xy[2],z;project_point(points+k*3,cameras+f*17,xy,z,nullptr);pixels[k*2]=float(xy[0]);pixels[k*2+1]=float(xy[1]);depths[k]=float(z);}
        return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
extern "C" int vhf_reprojection(const float *points,const double *cameras,const float *target,
    const float *weight,const float *ipd,int frames,int count,float *gradient,double *loss) {
    try {
        require(frames>0 && frames<=1024 && count>0 && count<=65536 && gradient && loss,"invalid reprojection dimensions");
        size_t n=size_t(frames)*count;finite(points,n*3);finite(cameras,size_t(frames)*17);finite(target,n*2);finite(weight,n);finite(ipd,frames);
        double denominator=0;for(size_t i=0;i<n;++i){require(weight[i]>=0,"negative observation weight");denominator+=weight[i];}denominator=std::max(1.,denominator);*loss=0;
        for(int f=0;f<frames;++f){require(ipd[f]>0,"invalid inter-pupil normalization");for(int a=0;a<count;++a) {
            size_t k=size_t(f)*count+a;double xy[2],z,jac[6];project_point(points+k*3,cameras+f*17,xy,z,jac);
            double ex=(xy[0]-target[k*2])/ipd[f],ey=(xy[1]-target[k*2+1])/ipd[f],norm=std::sqrt(ex*ex+ey*ey+.005*.005),w=weight[k]/denominator;
            double hidden=std::max(0.,.02-z);*loss+=w*(norm-.005)+10*hidden*hidden/n;
            for(int c=0;c<3;++c)gradient[k*3+c]=float(w*(ex*jac[c]+ey*jac[3+c])/(ipd[f]*norm)+20*hidden*cameras[f*17+6+c]/n);
        }}return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
extern "C" int vhf_quadratic(const float *x,const float *reference,int count,double weight,float *gradient,double *loss) {
    try {
        require(count>0 && count<=67108864 && gradient && loss && std::isfinite(weight) && weight>=0,"invalid prior");finite(x,count);if(reference)finite(reference,count);
        double sum=0;for(int k=0;k<count;++k){double d=x[k]-(reference?reference[k]:0);sum+=d*d;gradient[k]=float(2*weight*d/count);}*loss=weight*sum/count;return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
extern "C" int vhf_area_loss(const float *rest,const float *posed,const int32_t *triangles,
    int vertices,int faces,int frames,float *gradient,double *loss) {
    try {
        require(vertices>0 && vertices<=65536 && faces>0 && faces<=65536 && frames>0 && frames<=1024 && triangles && gradient && loss,"invalid area guard");
        finite(rest,size_t(vertices)*3);finite(posed,size_t(frames)*vertices*3);std::fill(gradient,gradient+size_t(frames)*vertices*3,0);*loss=0;
        std::vector<V> normal(faces);std::vector<double> norm(faces);
        for(int t=0;t<faces;++t) {
            V a,b;for(int j=0;j<3;++j)require(triangles[t*3+j]>=0 && triangles[t*3+j]<vertices,"invalid guard triangle");
            int i0=triangles[t*3],i1=triangles[t*3+1],i2=triangles[t*3+2];
            for(int j=0;j<3;++j){a[j]=rest[i1*3+j]-rest[i0*3+j];b[j]=rest[i2*3+j]-rest[i0*3+j];}
            normal[t]=cross(a,b);norm[t]=std::max(1e-24,dot3(normal[t],normal[t]));
        }
        int active=std::min(64,faces);
        for(int f=0;f<frames;++f) {
            std::vector<std::pair<double,int>> ranked(faces);const float *p=posed+size_t(f)*vertices*3;
            for(int t=0;t<faces;++t) {
                int i0=triangles[t*3],i1=triangles[t*3+1],i2=triangles[t*3+2];V a,b;
                for(int j=0;j<3;++j){a[j]=p[i1*3+j]-p[i0*3+j];b[j]=p[i2*3+j]-p[i0*3+j];}
                double ratio=dot3(cross(a,b),normal[t])/norm[t];ranked[t]={std::max(0.,.15-ratio),t};
            }
            std::partial_sort(ranked.begin(),ranked.begin()+active,ranked.end(),[](const auto&a,const auto&b){return a.first>b.first;});
            for(int k=0;k<active;++k) {
                double h=ranked[k].first;if(h==0)continue;int t=ranked[k].second;
                int i0=triangles[t*3],i1=triangles[t*3+1],i2=triangles[t*3+2];V a,b;
                for(int j=0;j<3;++j){a[j]=p[i1*3+j]-p[i0*3+j];b[j]=p[i2*3+j]-p[i0*3+j];}
                *loss+=10*h*h/(frames*active);double factor=-20*h/(frames*active*norm[t]);
                V da=cross(b,normal[t]),db=cross(normal[t],a);float *g=gradient+size_t(f)*vertices*3;
                for(int j=0;j<3;++j){g[i1*3+j]+=float(factor*da[j]);g[i2*3+j]+=float(factor*db[j]);g[i0*3+j]-=float(factor*(da[j]+db[j]));}
            }
        }return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
extern "C" int vhf_contact(const float *points,const float *joints,const float *rotations,const double *alignment,
    int frames,int anchors,const int32_t *pairs,int pair_count,const int32_t *eyes,const float *radii,int eye_count,
    const float *limit,float *distances,float *gp,float *gj,float *gr,double *loss) {
    try {
        int count=pair_count+eye_count;require(frames>0 && frames<=1024 && anchors>0 && anchors<=65536 && pair_count>=0 && eye_count>=0 && count<=1024 && loss,"invalid contacts");
        finite(points,size_t(frames)*anchors*3);finite(joints,size_t(frames)*12);finite(rotations,size_t(frames)*12);finite(alignment,9);if(count)require(distances,"missing contact output");
        bool backward=limit!=nullptr;*loss=0;
        if(backward) {require(gp && gj && gr,"missing contact gradients");finite(limit,size_t(frames)*count);std::fill(gp,gp+size_t(frames)*anchors*3,0);std::fill(gj,gj+size_t(frames)*12,0);std::fill(gr,gr+size_t(frames)*12,0);}
        for(int k=0;k<pair_count*2;++k)require(pairs && pairs[k]>=0 && pairs[k]<anchors,"invalid lip attachment");
        for(int k=0;k<eye_count;++k){require(eyes && radii && eyes[k*2]>=0 && eyes[k*2]<anchors && eyes[k*2+1]>=0 && eyes[k*2+1]<4 && radii[k]>0 && std::isfinite(radii[k]),"invalid lid attachment");}
        M align;std::copy(alignment,alignment+9,align.begin());
        for(int f=0;f<frames;++f) {
            double r[3];for(int a=0;a<3;++a)r[a]=rotations[f*12+a];M rot;std::array<M,3>dr;rotation(r,rot,backward?&dr:nullptr);
            V up=mv(align,V{rot[1],rot[4],rot[7]});
            for(int k=0;k<count;++k) {
                V delta;double dist,norm=0;
                int a=k<pair_count?pairs[k*2]:eyes[(k-pair_count)*2];
                int b=k<pair_count?pairs[k*2+1]:eyes[(k-pair_count)*2+1];
                for(int c=0;c<3;++c)delta[c]=points[(size_t(f)*anchors+a)*3+c]-(k<pair_count?points[(size_t(f)*anchors+b)*3+c]:joints[f*12+b*3+c]);
                if(k<pair_count)dist=dot3(delta,up);else {norm=std::sqrt(dot3(delta,delta));dist=norm-radii[k-pair_count];}
                distances[f*count+k]=float(dist);
                if(!backward)continue;
                double h=std::max(0.,double(limit[f*count+k])-dist),factor=-1e7*h/(frames*count);*loss+=5e6*h*h/(frames*count);
                for(int c=0;c<3;++c) {
                    double g=factor*(k<pair_count?up[c]:(norm>0?delta[c]/norm:0));gp[(size_t(f)*anchors+a)*3+c]+=float(g);
                    if(k<pair_count)gp[(size_t(f)*anchors+b)*3+c]-=float(g);else gj[f*12+b*3+c]-=float(g);
                }
                if(k<pair_count)for(int c=0;c<3;++c)gr[f*12+c]+=float(factor*dot3(delta,mv(align,V{dr[c][1],dr[c][4],dr[c][7]})));
            }
        }return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
extern "C" int vhf_temporal(const float *expression,const float *rotations,const double *times,int frames,int modes,
    float *ge,float *gr,double *loss) {
    try {
        require(frames>0 && frames<=1024 && modes>0 && modes<=512 && ge && gr && loss,"invalid temporal shape");finite(expression,size_t(frames)*modes);finite(rotations,size_t(frames)*12);finite(times,frames);
        std::fill(ge,ge+size_t(frames)*modes,0);std::fill(gr,gr+size_t(frames)*12,0);*loss=0;
        if(frames==1)return 0;
        std::vector<double> velocity(size_t(frames-1)*modes),dv(velocity.size());
        for(int f=0;f<frames-1;++f) {
            double dt=times[f+1]-times[f];require(dt>0,"temporal timestamps must increase");
            for(int e=0;e<modes;++e) {size_t k=size_t(f)*modes+e;double v=(expression[k+modes]-expression[k])/dt;velocity[k]=v;*loss+=1e-5*v*v/((frames-1)*modes);dv[k]=2e-5*v/((frames-1)*modes);}
            for(int c=0;c<3;++c) {double v=(rotations[(f+1)*12+c]-rotations[f*12+c])/dt;*loss+=2e-5*v*v/((frames-1)*3);double g=4e-5*v/(dt*(frames-1)*3);gr[(f+1)*12+c]+=float(g);gr[f*12+c]-=float(g);}
        }
        for(int f=0;f<frames-2;++f)for(int e=0;e<modes;++e) {
            size_t k=size_t(f)*modes+e;double dt=times[f+2]-times[f],a=2*(velocity[k+modes]-velocity[k])/dt;
            *loss+=1e-8*a*a/((frames-2)*modes);double g=4e-8*a/(dt*(frames-2)*modes);dv[k+modes]+=g;dv[k]-=g;
        }
        for(int f=0;f<frames-1;++f)for(int e=0;e<modes;++e) {size_t k=size_t(f)*modes+e;double g=dv[k]/(times[f+1]-times[f]);ge[k+modes]+=float(g);ge[k]-=float(g);}
        return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
extern "C" int vhf_silhouette(const float *vertices,int count,const int32_t *triangles,const float *bary,int anchors,
    const double *camera,const float *sdf,const float *allowed,int h,int w,double scale,
    const float *target,int targets,float *gradient,double *loss) {
    try {
        require(count>0 && count<=65536 && anchors>0 && anchors<=1024 && targets>0 && targets<=1024 && h>0 && w>0 && size_t(h)*w<=4000000 && gradient && loss && triangles && scale>0 && std::isfinite(scale),"invalid silhouette");
        finite(vertices,size_t(count)*3);finite(bary,size_t(anchors)*3);finite(camera,17);finite(sdf,size_t(h)*w);finite(allowed,size_t(h)*w);finite(target,size_t(targets)*2);
        std::fill(gradient,gradient+size_t(count)*3,0);std::vector<double>xy(size_t(anchors)*2),jac(size_t(anchors)*6),gxy(xy.size()),visible(anchors),distance(anchors),dx(anchors),dy(anchors);
        double denominator=0;*loss=0;
        auto sample=[&](const float *image,double x,double y,bool border,double *gx,double *gy) {
            double xx=border?std::clamp(x,0.,double(w-1)):x,yy=border?std::clamp(y,0.,double(h-1)):y;
            int ix=int(std::floor(xx)),iy=int(std::floor(yy));double fx=xx-ix,fy=yy-iy;
            auto at=[&](int a,int b){return (a>=0 && a<h && b>=0 && b<w)?double(image[a*w+b]):0.;};
            double a=at(iy,ix),b=at(iy,ix+1),c=at(iy+1,ix),d=at(iy+1,ix+1);
            if(gx)*gx=(x>0 && x<w-1)?(1-fy)*(b-a)+fy*(d-c):0;
            if(gy)*gy=(y>0 && y<h-1)?(1-fx)*(c-a)+fx*(d-b):0;
            return (1-fy)*((1-fx)*a+fx*b)+fy*((1-fx)*c+fx*d);
        };
        for(int a=0;a<anchors;++a) {
            float point[3]{};for(int k=0;k<3;++k){int id=triangles[a*3+k];require(id>=0 && id<count,"silhouette attachment outside mesh");for(int c=0;c<3;++c)point[c]+=vertices[id*3+c]*bary[a*3+k];}
            double depth;project_point(point,camera,xy.data()+a*2,depth,jac.data()+a*6);
            double x=xy[a*2]*scale-.5,y=xy[a*2+1]*scale-.5;
            distance[a]=sample(sdf,x,y,true,&dx[a],&dy[a]);visible[a]=sample(allowed,x,y,false,nullptr,nullptr);denominator+=visible[a];
        }
        denominator=std::max(1.,denominator);double factor=.01/std::max(h,w);
        for(int a=0;a<anchors;++a) {
            *loss+=factor*std::abs(distance[a])*visible[a]/denominator;
            double sign=distance[a]>0?1:distance[a]<0?-1:0;
            gxy[a*2]=factor*sign*visible[a]*dx[a]*scale/denominator;gxy[a*2+1]=factor*sign*visible[a]*dy[a]*scale/denominator;
        }
        for(int t=0;t<targets;++t) {
            double best=INFINITY;int chosen=0;
            for(int a=0;a<anchors;++a){double x=xy[a*2]-target[t*2],y=xy[a*2+1]-target[t*2+1],d=x*x+y*y;if(d<best){best=d;chosen=a;}}
            double d=std::sqrt(best);*loss+=factor*d/targets;
            if(d>0)for(int c=0;c<2;++c)gxy[chosen*2+c]+=factor*(xy[chosen*2+c]-target[t*2+c])/(targets*d);
        }
        for(int a=0;a<anchors;++a)for(int c=0;c<3;++c) {
            double g=gxy[a*2]*jac[a*6+c]+gxy[a*2+1]*jac[a*6+3+c];
            for(int k=0;k<3;++k)gradient[triangles[a*3+k]*3+c]+=float(g*bary[a*3+k]);
        }return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
