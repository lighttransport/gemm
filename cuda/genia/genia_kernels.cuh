/* Compiled together with common/genia_math.h by embed_kernels.py. */
extern "C" {
__global__ void genia_project_kernel(const float *rows, int count, const float *camera,
    const float *k, double *projection, double *jacobian)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    genia_gaussian_project(rows+17*i, camera, k, projection+12*i,
                           jacobian ? jacobian+(size_t)i*260 : nullptr);
}
__global__ void genia_raster_kernel(const double *projection, const int *offsets,
    const int *ids, int width, int height, const float *background,
    const float *rgba_gradient, const float *depth_gradient,
    float *rgba, float *depth, double *projection_gradient)
{
    int tw = (width+15)/16, tile = blockIdx.x;
    int x = (tile%tw)*16+threadIdx.x%16, y = (tile/tw)*16+threadIdx.x/16;
    if (x >= width || y >= height) return;
    int pixel = y*width+x, begin = offsets[tile], end = offsets[tile+1], last = begin-1;
    double transmittance = 1, rgb[3] = {}, depth_sum = 0;
    for (int j = begin; j < end; ++j) {
        const double *q = projection+12*ids[j];
        double dx = q[0]-(x+.5), dy = q[1]-(y+.5);
        double sigma = .5*(q[3]*dx*dx+q[5]*dy*dy)+q[4]*dx*dy;
        double alpha = fmin(.99, q[6]*exp(-sigma));
        if (sigma < 0 || alpha < 1./255) continue;
        double next = transmittance*(1-alpha);
        if (next <= 1e-4) break;
        for (int c = 0; c < 3; ++c) rgb[c] += transmittance*alpha*q[7+c];
        depth_sum += transmittance*alpha*q[2];
        transmittance = next; last = j;
    }
    double alpha_sum = 1-transmittance;
    if (rgba) {
        for (int c = 0; c < 3; ++c) rgba[4*pixel+c] = rgb[c]+transmittance*background[c];
        rgba[4*pixel+3] = alpha_sum;
    }
    if (depth) depth[pixel] = alpha_sum > 1e-10 ? depth_sum/alpha_sum : 0;
    if (!projection_gradient) return;
    double dc[3] = {}, ds = 0, dt = rgba_gradient ? -rgba_gradient[4*pixel+3] : 0;
    if (rgba_gradient) for (int c = 0; c < 3; ++c) {
        dc[c] = rgba_gradient[4*pixel+c]; dt += dc[c]*background[c];
    }
    if (depth_gradient && alpha_sum > 1e-10) {
        ds = depth_gradient[pixel]/alpha_sum;
        dt += depth_gradient[pixel]*depth_sum/(alpha_sum*alpha_sum);
    }
    for (int j = last; j >= begin; --j) {
        int id = ids[j]; const double *q = projection+12*id;
        double dx = q[0]-(x+.5), dy = q[1]-(y+.5);
        double sigma = .5*(q[3]*dx*dx+q[5]*dy*dy)+q[4]*dx*dy;
        double e = exp(-sigma), raw = q[6]*e, alpha = fmin(.99, raw);
        if (sigma < 0 || alpha < 1./255) continue;
        double before = transmittance/(1-alpha), dot = ds*q[2];
        double *g = projection_gradient+10*id;
        for (int c = 0; c < 3; ++c) {
            dot += dc[c]*q[7+c]; atomicAdd(g+7+c, dc[c]*before*alpha);
        }
        atomicAdd(g+2, ds*before*alpha);
        double da = before*(dot-dt);
        dt = alpha*dot+(1-alpha)*dt; transmittance = before;
        if (raw < .99) {
            atomicAdd(g+6, da*e);
            double dp = -da*raw;
            atomicAdd(g, dp*(q[3]*dx+q[4]*dy));
            atomicAdd(g+1, dp*(q[5]*dy+q[4]*dx));
            atomicAdd(g+3, dp*.5*dx*dx); atomicAdd(g+4, dp*dx*dy); atomicAdd(g+5, dp*.5*dy*dy);
        }
    }
}
__global__ void genia_project_backward_kernel(const double *jacobian, const double *projection_gradient,
    int count, float *row_gradient, double *camera_gradient)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    const int row_index[14] = {0,1,2,6,7,8,9,10,11,12,13,14,15,16};
    const double *jac = jacobian+(size_t)i*260, *g = projection_gradient+10*i;
    if (row_gradient) for (int p = 0; p < 17; ++p) row_gradient[17*i+p] = 0;
    for (int p = 0; p < 26; ++p) {
        double value = 0;
        for (int k = 0; k < 10; ++k) value += jac[k*26+p]*g[k];
        if (p < 14) { if (row_gradient) row_gradient[17*i+row_index[p]] = value; }
        else if (camera_gradient) atomicAdd(camera_gradient+p-14, value);
    }
}
__global__ void genia_visible_kernel(const int32_t *xyz, int count, const float *camera,
    const uint8_t *occupancy, int grid, float tolerance, uint8_t *visible)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < count) visible[i] = genia_voxel_visible(xyz+3*i, camera, occupancy, grid, tolerance);
}
__global__ void genia_fuse_kernel(const float *scores, const float *velocities,
    int views, int count, int channels, float temperature, float floor_weight, float *output)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    float s[16], weights[16];
    for (int v = 0; v < views; ++v) s[v] = scores[(size_t)v*count+i];
    genia_fusion_weights(s, views, temperature, floor_weight, weights);
    for (int c = 0; c < channels; ++c) {
        double value = 0;
        for (int v = 0; v < views; ++v) value += (double)weights[v]*velocities[((size_t)v*count+i)*channels+c];
        output[(size_t)i*channels+c] = value;
    }
}
__global__ void genia_tweedie_kernel(const float *latent, const float *velocity,
    int count, float time, float *clean)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < count) clean[i] = latent[i]+(1-time)*velocity[i];
}
__global__ void genia_norm_kernel(const float *velocity, const float *gradient,
    int count, double *norm)
{
    __shared__ double vv[256], gg[256];
    int lane = threadIdx.x;
    double v = 0, g = 0;
    for (int i = lane; i < count; i += blockDim.x) {
        v += (double)velocity[i]*velocity[i];
        g += (double)gradient[i]*gradient[i];
    }
    vv[lane] = v; gg[lane] = g; __syncthreads();
    for (int stride = blockDim.x/2; stride; stride /= 2) {
        if (lane < stride) { vv[lane] += vv[lane+stride]; gg[lane] += gg[lane+stride]; }
        __syncthreads();
    }
    if (!lane) { norm[0] = vv[0]; norm[1] = gg[0]; }
}
__global__ void genia_guidance_kernel(const float *velocity, const float *gradient,
    int count, float weight, int normalize, const double *norm, float *guided)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    double multiplier = normalize ? (norm[1] < 1e-20 ? 0 : weight*sqrt(norm[0]/norm[1])) : weight;
    guided[i] = velocity[i]-multiplier*gradient[i];
}
}
