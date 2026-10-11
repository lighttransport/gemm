/* GenIA alignment primitives. All camera coordinates here are R3 (+X right,
 * +Y down, +Z forward); callers convert SAM3D's P3D poses at the boundary.
 * Independent implementation of the equations in arXiv:2610.12388.
 * Arrays are row-major; coordinates/occupancy use explicit (x,y,z) order.
 */
#ifndef GENIA_MATH_H
#define GENIA_MATH_H
#ifndef __CUDACC_RTC__
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#else
typedef int int32_t;
typedef unsigned char uint8_t;
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GENIA_HD __host__ __device__
#else
#define GENIA_HD
#endif

/* Project a canonical voxel center using column-vector object-to-camera R,t.
 * scale is isotropic; uv is in original observation pixels. */
GENIA_HD static inline int genia_project_voxel(const int32_t xyz[3], int grid,
    const float r[9], const float t[3], float scale, const float k[9],
    int width, int height, float uv[2])
{
    float p[3], q[3];
    if (grid <= 0 || !(scale > 0)) return -1;
    for (int a = 0; a < 3; ++a) p[a] = ((xyz[a] + .5f) / grid - .5f) * scale;
    for (int a = 0; a < 3; ++a)
        q[a] = r[3*a]*p[0] + r[3*a+1]*p[1] + r[3*a+2]*p[2] + t[a];
    if (!(q[2] > 0) || !isfinite(q[2])) return 0;
    uv[0] = k[0]*q[0]/q[2] + k[2];
    uv[1] = k[4]*q[1]/q[2] + k[5];
    return isfinite(uv[0]) && isfinite(uv[1]) && uv[0] >= 0 && uv[0] < width && uv[1] >= 0 && uv[1] < height;
}

/* DDA trace from camera to target. Occupancy index = (x*grid+y)*grid+z.
 * Lowest axis wins a corner/edge tie, as in the pinned upstream DDA.
 * Near-target occupied cells within tolerance are not treated as occluders. */
GENIA_HD static inline int genia_voxel_visible(const int32_t target[3],
    const float camera[3], const uint8_t *occupied, int grid, float tolerance)
{
    double d[3], start[3], length = 0, enter = -1e30, leave = 1e30;
    if (!occupied || grid <= 0 || !(tolerance >= 0)) return -1;
    for (int a = 0; a < 3; ++a) {
        if (target[a] < 0 || target[a] >= grid || !isfinite(camera[a])) return -1;
        d[a] = target[a] + .5 - camera[a];
        length += d[a]*d[a];
    }
    length = sqrt(length);
    if (length < 1e-8) return 1;
    double inverse_length = 1.0 / length;
    for (int a = 0; a < 3; ++a) {
        d[a] *= inverse_length;
        start[a] = camera[a];
        if (fabs(d[a]) < 1e-10) {
            if (start[a] < 0 || start[a] > grid) return 1;
        } else {
            double lo = -start[a]/d[a], hi = (grid-start[a])/d[a];
            if (lo > hi) { double swap = lo; lo = hi; hi = swap; }
            if (lo > enter) enter = lo;
            if (hi < leave) leave = hi;
        }
    }
    if (enter > leave || enter > length) return 1;
    if (enter > 0)
        for (int a = 0; a < 3; ++a) start[a] += d[a]*(enter+.001);
    int cell[3], step[3]; double next[3], delta[3];
    for (int a = 0; a < 3; ++a) {
        cell[a] = (int)floor(start[a]);
        if (cell[a] < 0) cell[a] = 0;
        if (cell[a] >= grid) cell[a] = grid-1;
        step[a] = d[a] >= 0 ? 1 : -1;
        if (fabs(d[a]) < 1e-10) { next[a] = delta[a] = 1e30; }
        else {
            next[a] = (cell[a] + (step[a] > 0 ? 1 : 0) - start[a])/d[a];
            delta[a] = fabs(1/d[a]);
        }
    }
    for (int count = 0; count < grid*3; ++count) {
        if (cell[0] < 0 || cell[1] < 0 || cell[2] < 0 || cell[0] >= grid || cell[1] >= grid || cell[2] >= grid) break;
        double distance2 = 0;
        for (int a = 0; a < 3; ++a) { double v = cell[a]-target[a]; distance2 += v*v; }
        if (occupied[((size_t)cell[0]*grid+cell[1])*grid+cell[2]] && distance2 > (double)tolerance*tolerance) return 0;
        if (cell[0] == target[0] && cell[1] == target[1] && cell[2] == target[2]) break;
        int axis = next[0] <= next[1] ? (next[0] <= next[2] ? 0 : 2) : (next[1] <= next[2] ? 1 : 2);
        cell[axis] += step[axis]; next[axis] += delta[axis];
    }
    return 1;
}

/* Stable softmax, then floor and renormalize, Eq. 6. For shape fusion pass
 * negative attention entropy as scores; for appearance pass binary visibility.
 * out may alias scores. */
GENIA_HD static inline int genia_fusion_weights(const float *scores, int views,
    float temperature, float floor_weight, float *out)
{
    if (!scores || !out || views < 1 || !isfinite(temperature) || !(floor_weight >= 0) || !isfinite(floor_weight)) return -1;
    double max = -1e300, sum = 0;
    for (int v = 0; v < views; ++v) {
        if (!isfinite(scores[v])) return -1;
        double x = (double)temperature*scores[v]; if (x > max) max = x;
    }
    for (int v = 0; v < views; ++v) sum += exp((double)temperature*scores[v]-max);
    double total = 0;
    for (int v = 0; v < views; ++v) {
        out[v] = (float)fmax(exp((double)temperature*scores[v]-max)/sum, (double)floor_weight);
        total += out[v];
    }
    for (int v = 0; v < views; ++v) out[v] = (float)(out[v]/total);
    return 0;
}

/* Approximate passive-stream compensation. fraction is the proportion of
 * biased RGB patches in the active group; non-RGB streams remain unchanged. */
GENIA_HD static inline float genia_attention_compensation(float alpha, float fraction)
{
    return log1pf(fraction*expm1f(alpha));
}

/* The upstream clean-latent estimate treats backbone velocity as detached.
 * Therefore guidance needs decoder/renderer backward, not DiT backward. */
GENIA_HD static inline int genia_tweedie(const float *x, const float *velocity, size_t count,
    float time, float *clean)
{
    if (!x || !velocity || !clean || !isfinite(time) || time < 0 || time > 1) return -1;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(x[i]) || !isfinite(velocity[i])) return -1;
    for (size_t i = 0; i < count; ++i) clean[i] = x[i]+(1-time)*velocity[i];
    return 0;
}

GENIA_HD static inline int genia_guidance(float *velocity, const float *gradient,
    size_t count, float weight, int normalize)
{
    if (!velocity || !gradient || !(weight >= 0) || !isfinite(weight)) return -1;
    double vn = 0, gn = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(velocity[i]) || !isfinite(gradient[i])) return -1;
        vn += (double)velocity[i]*velocity[i]; gn += (double)gradient[i]*gradient[i];
    }
    if (weight == 0 || (normalize && gn < 1e-20)) return 0;
    double multiplier = weight*(normalize ? sqrt(vn/gn) : 1);
    for (size_t i = 0; i < count; ++i) velocity[i] = (float)(velocity[i]-multiplier*gradient[i]);
    return 0;
}
#undef GENIA_HD
#endif
