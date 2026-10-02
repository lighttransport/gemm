/* Included by training.cpp: analytic corrective training, no autodiff runtime. */
static void cross3(const float *a, const float *b, float *out)
{
    out[0]=a[1]*b[2]-a[2]*b[1]; out[1]=a[2]*b[0]-a[0]*b[2]; out[2]=a[0]*b[1]-a[1]*b[0];
}
static void indices(const int32_t *ids, size_t count, int limit)
{
    require(ids != nullptr,"missing indices");
    for (size_t i=0;i<count;++i) require(ids[i]>=0 && ids[i]<limit,"invalid vertex index");
}

extern "C" {
int vh_train_mlp(float *out, float *gradient, const float *x, const float *parameters,
                 const float *upstream, int n, int inputs, int hidden, int outputs)
{
    try {
        require(out && n>0 && n<=65536 && inputs>0 && inputs<=16384 && hidden>0 && hidden<=4096 &&
                outputs>0 && outputs<=4096,"invalid MLP dimensions");
        size_t first=size_t(hidden)*inputs, second=first+hidden, bias=second+size_t(outputs)*hidden, count=bias+outputs;
        finite(x,size_t(n)*inputs); finite(parameters,count);
        if (gradient) finite(upstream,size_t(n)*outputs);
        std::vector<float> activation(size_t(n)*hidden);
        gemm(activation.data(),x,parameters,n,hidden,inputs,false,true);
        for (int i=0;i<n;++i) for (int j=0;j<hidden;++j)
            activation[size_t(i)*hidden+j]=std::max(0.f,activation[size_t(i)*hidden+j]+parameters[first+j]);
        gemm(out,activation.data(),parameters+second,n,outputs,hidden,false,true);
        for (int i=0;i<n;++i) for (int j=0;j<outputs;++j) out[size_t(i)*outputs+j]+=parameters[bias+j];
        finite(out,size_t(n)*outputs);
        if (gradient) {
            std::fill(gradient,gradient+count,0.f);
            gemm(gradient+second,upstream,activation.data(),outputs,hidden,n,true,false);
            std::vector<float> dh(size_t(n)*hidden);
            gemm(dh.data(),upstream,parameters+second,n,hidden,outputs);
            for (int i=0;i<n;++i) {
                for (int j=0;j<outputs;++j) gradient[bias+j]+=upstream[size_t(i)*outputs+j];
                for (int j=0;j<hidden;++j) {
                    size_t k=size_t(i)*hidden+j;
                    if (activation[k]==0) dh[k]=0;
                    gradient[first+j]+=dh[k];
                }
            }
            gemm(gradient,dh.data(),x,hidden,inputs,n,true,false);
            finite(gradient,count);
        }
        return 0;
    } catch (const std::exception &e) { train_error=e.what(); return -1; }
}

int vh_train_spheres(const float *x, const int32_t *ids, const float *centers, const float *thresholds,
                     int samples, int vertices, int points, int spheres, float *gradient, float *depth, float *energy)
{
    try {
        require(samples>0 && vertices>0 && points>=0 && spheres>0 && gradient && depth && energy,"invalid sphere dimensions");
        finite(x,size_t(samples)*vertices*3); indices(ids,points,vertices);
        finite(centers,size_t(samples)*spheres*3); finite(thresholds,size_t(points)*spheres);
        std::fill(gradient,gradient+size_t(samples)*vertices*3,0.f);
        std::fill(depth,depth+size_t(samples)*points,0.f); std::fill(energy,energy+samples,0.f);
        for (int s=0;s<samples;++s) for (int i=0;i<points;++i) {
            size_t q=(size_t(s)*vertices+ids[i])*3;
            for (int j=0;j<spheres;++j) {
                float d[3],norm2=0;
                for (int a=0;a<3;++a) { d[a]=x[q+a]-centers[(size_t(s)*spheres+j)*3+a]; norm2+=d[a]*d[a]; }
                float distance=std::sqrt(norm2), penetration=std::max(0.f,thresholds[size_t(i)*spheres+j]-distance);
                energy[s]+=penetration*penetration;
                depth[size_t(s)*points+i]=std::max(depth[size_t(s)*points+i],penetration);
                if (distance>0) for (int a=0;a<3;++a) gradient[q+a]-=2*penetration*d[a]/distance;
            }
        }
        finite(gradient,size_t(samples)*vertices*3); finite(energy,samples);
        return 0;
    } catch (const std::exception &e) { train_error=e.what(); return -1; }
}

int vh_train_pairs(const float *x, const int32_t *upper, const int32_t *lower, const float *up,
                   const float *floor, int samples, int vertices, int pairs, float *gradient, float *depth, float *energy)
{
    try {
        require(samples>0 && vertices>0 && pairs>=0 && gradient && depth && energy,"invalid pair dimensions");
        finite(x,size_t(samples)*vertices*3); indices(upper,pairs,vertices); indices(lower,pairs,vertices);
        finite(up,size_t(samples)*3); finite(floor,pairs);
        std::fill(gradient,gradient+size_t(samples)*vertices*3,0.f);
        std::fill(energy,energy+samples,0.f);
        for (int s=0;s<samples;++s) for (int i=0;i<pairs;++i) {
            size_t u=(size_t(s)*vertices+upper[i])*3,l=(size_t(s)*vertices+lower[i])*3;
            float separation=0;
            for (int a=0;a<3;++a) separation+=(x[u+a]-x[l+a])*up[s*3+a];
            float penetration=std::max(0.f,floor[i]-separation);
            energy[s]+=penetration*penetration; depth[size_t(s)*pairs+i]=penetration;
            for (int a=0;a<3;++a) { float g=2*penetration*up[s*3+a]; gradient[u+a]-=g; gradient[l+a]+=g; }
        }
        finite(gradient,size_t(samples)*vertices*3); finite(energy,samples);
        return 0;
    } catch (const std::exception &e) { train_error=e.what(); return -1; }
}

int vh_train_arap(const float *x, const float *linear, const int32_t *edges, const float *target,
                  int samples, int vertices, int count, double weight, float *gradient, float *energy)
{
    try {
        require(samples>0 && vertices>0 && count>0 && gradient && energy && std::isfinite(weight) && weight>=0,"invalid ARAP dimensions");
        indices(edges,size_t(count)*2,vertices); finite(x,size_t(samples)*vertices*3);
        finite(linear,size_t(samples)*vertices*3); finite(target,size_t(samples)*count*3);
        for (int s=0;s<samples;++s) {
            double loss=0;
            for (int v=0;v<vertices*3;++v) {
                size_t q=size_t(s)*vertices*3+v; float d=x[q]-linear[q];
                loss+=double(d)*d/vertices; gradient[q]=2*d/vertices;
            }
            for (int i=0;i<count;++i) for (int a=0;a<3;++a) {
                size_t u=(size_t(s)*vertices+edges[2*i])*3+a,v=(size_t(s)*vertices+edges[2*i+1])*3+a;
                float d=x[v]-x[u]-target[(size_t(s)*count+i)*3+a],g=float(2*weight*d/count);
                loss+=weight*double(d)*d/count; gradient[v]+=g; gradient[u]-=g;
            }
            energy[s]=float(loss);
        }
        finite(gradient,size_t(samples)*vertices*3); finite(energy,samples);
        return 0;
    } catch (const std::exception &e) { train_error=e.what(); return -1; }
}

int vh_train_rotations(const float *x, const float *rest_edges, const float *rest_normals, const float *normal_scale,
                       const int32_t *edges, const int32_t *faces, int samples, int vertices, int edge_count,
                       int face_count, float *out)
{
    try {
        require(samples>0 && vertices>0 && edge_count>0 && face_count>0 && out,"invalid rotation dimensions");
        indices(edges,size_t(edge_count)*2,vertices); indices(faces,size_t(face_count)*3,vertices);
        finite(x,size_t(samples)*vertices*3); finite(rest_edges,size_t(edge_count)*3);
        finite(rest_normals,size_t(vertices)*3); finite(normal_scale,vertices);
        std::fill(out,out+size_t(samples)*vertices*9,0.f);
        std::vector<float> normals(size_t(samples)*vertices*3,0.f);
        for (int s=0;s<samples;++s) {
            for (int i=0;i<face_count;++i) {
                const int32_t *f=faces+i*3; float a[3],b[3],n[3];
                for (int j=0;j<3;++j) { a[j]=x[(size_t(s)*vertices+f[1])*3+j]-x[(size_t(s)*vertices+f[0])*3+j]; b[j]=x[(size_t(s)*vertices+f[2])*3+j]-x[(size_t(s)*vertices+f[0])*3+j]; }
                cross3(a,b,n);
                for (int k=0;k<3;++k) for (int j=0;j<3;++j) normals[(size_t(s)*vertices+f[k])*3+j]+=n[j];
            }
            for (int i=0;i<edge_count;++i) {
                int u=edges[i*2],v=edges[i*2+1]; float d[3];
                for (int j=0;j<3;++j) d[j]=x[(size_t(s)*vertices+v)*3+j]-x[(size_t(s)*vertices+u)*3+j];
                for (int j=0;j<3;++j) for (int k=0;k<3;++k) {
                    float q=d[j]*rest_edges[i*3+k];
                    out[(size_t(s)*vertices+u)*9+j*3+k]+=q; out[(size_t(s)*vertices+v)*9+j*3+k]+=q;
                }
            }
            for (int v=0;v<vertices;++v) {
                float *r=out+(size_t(s)*vertices+v)*9,*n=normals.data()+(size_t(s)*vertices+v)*3;
                float length=std::max(std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]),1e-12f),norm2=0;
                for (int j=0;j<3;++j) for (int k=0;k<3;++k) { r[j*3+k]+=normal_scale[v]*n[j]/length*rest_normals[v*3+k]; norm2+=r[j*3+k]*r[j*3+k]; }
                float norm=std::max(std::sqrt(norm2),1e-12f);
                for (int j=0;j<9;++j) r[j]/=norm;
                for (int step=0;step<12;++step) {
                    float cofactor[9]; cross3(r+3,r+6,cofactor); cross3(r+6,r,cofactor+3); cross3(r,r+3,cofactor+6);
                    float det=r[0]*cofactor[0]+r[1]*cofactor[1]+r[2]*cofactor[2];
                    if (std::abs(det)<1e-12f) det=1e-12f;
                    for (int j=0;j<9;++j) r[j]=.5f*(r[j]+cofactor[j]/det);
                }
            }
        }
        finite(out,size_t(samples)*vertices*9);
        return 0;
    } catch (const std::exception &e) { train_error=e.what(); return -1; }
}
}
