/* Free anisotropic Gaussian projection with analytic derivatives. Unlike the
 * vhuman trace-v1 projection, no triangle binding or covariance clamp is used.
 * Input is the repository's 17-float PLY layout (log scales, opacity logit,
 * SH DC and wxyz quaternion). Cameras use R3 column-vector conventions.
 */
#ifndef GENIA_GAUSSIAN_H
#define GENIA_GAUSSIAN_H
#ifndef __CUDACC_RTC__
#include <cmath>
#endif
#if defined(__CUDACC__) || defined(__HIPCC__)
#define GG_HD __host__ __device__
#else
#define GG_HD
#endif

struct genia_dual {
    double value, gradient;
    GG_HD genia_dual(double v = 0): value(v), gradient(0) {}
    GG_HD static genia_dual variable(double v, int i, int active) { genia_dual result(v); result.gradient = i == active ? 1 : 0; return result; }
};
GG_HD static inline genia_dual operator+(const genia_dual &a, const genia_dual &b)
{
    genia_dual result(a.value+b.value);
    result.gradient = a.gradient+b.gradient;
    return result;
}
GG_HD static inline genia_dual operator-(const genia_dual &a, const genia_dual &b)
{
    genia_dual result(a.value-b.value);
    result.gradient = a.gradient-b.gradient;
    return result;
}
GG_HD static inline genia_dual operator*(const genia_dual &a, const genia_dual &b)
{
    genia_dual result(a.value*b.value);
    result.gradient = a.gradient*b.value+a.value*b.gradient;
    return result;
}
GG_HD static inline genia_dual operator/(const genia_dual &a, const genia_dual &b)
{
    genia_dual result(a.value/b.value);
    result.gradient = (a.gradient-result.value*b.gradient)/b.value;
    return result;
}
GG_HD static inline genia_dual genia_exp(const genia_dual &a)
{
    genia_dual result(exp(a.value));
    result.gradient = result.value*a.gradient;
    return result;
}
GG_HD static inline genia_dual genia_sqrt(const genia_dual &a)
{
    genia_dual result(sqrt(a.value));
    result.gradient = a.gradient/(2*result.value);
    return result;
}
GG_HD static inline genia_dual genia_sigmoid(const genia_dual &a)
{
    genia_dual result(a.value >= 0 ? 1/(1+exp(-a.value)) : exp(a.value)/(1+exp(a.value)));
    result.gradient = result.value*(1-result.value)*a.gradient;
    return result;
}

/* Outputs u,v,z,conic_xx,conic_xy,conic_yy,opacity,r,g,b,radius_x,radius_y.
 * Jacobian[10,26]: first 14 derivatives are PLY non-normal attributes;
 * last 12 are the row-major world-to-camera affine matrix. */
GG_HD static inline bool genia_gaussian_project_column(const float row[17], const float camera[12],
    const float k[9], int active, double out[12], double *derivatives)
{
    for (int i = 0; i < 12; ++i) out[i] = 0;
    genia_dual xyz[3], dc[3], scales[3], quat[4], transform[12];
    for (int i = 0; i < 3; ++i) {
        xyz[i] = genia_dual::variable(row[i], i, active);
        dc[i] = genia_dual::variable(row[6+i], 3+i, active);
        scales[i] = genia_exp(genia_dual::variable(row[10+i], 7+i, active));
    }
    genia_dual opacity = genia_sigmoid(genia_dual::variable(row[9], 6, active));
    if (opacity.value < 1./255) return false;
    genia_dual length2;
    for (int i = 0; i < 4; ++i) {
        quat[i] = genia_dual::variable(row[13+i], 10+i, active);
        length2 = length2+quat[i]*quat[i];
    }
    if (length2.value < 1e-20) return false;
    genia_dual length = genia_sqrt(length2);
    for (int i = 0; i < 4; ++i) quat[i] = quat[i]/length;
    const genia_dual &w = quat[0], &x = quat[1], &y = quat[2], &z = quat[3];
    genia_dual r[9] = {
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)};
    for (int i = 0; i < 12; ++i) transform[i] = genia_dual::variable(camera[i], 14+i, active);
    genia_dual center[3], covariance[9], rotated[9];
    for (int i = 0; i < 3; ++i) {
        center[i] = transform[i*4+3];
        for (int j = 0; j < 3; ++j) {
            center[i] = center[i]+transform[i*4+j]*xyz[j];
            for (int a = 0; a < 3; ++a) covariance[i*3+j] = covariance[i*3+j]+r[i*3+a]*scales[a]*scales[a]*r[j*3+a];
        }
    }
    if (center[2].value <= .01 || !isfinite(center[2].value)) return false;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b)
            rotated[i*3+j] = rotated[i*3+j]+transform[i*4+a]*covariance[a*3+b]*transform[j*4+b];
    genia_dual invz = 1/center[2];
    genia_dual projection[6] = {k[0]*invz, 0, (0-k[0])*center[0]*invz*invz,
                                0, k[4]*invz, (0-k[4])*center[1]*invz*invz};
    genia_dual c2d[4];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b)
            c2d[i*2+j] = c2d[i*2+j]+projection[i*3+a]*rotated[a*3+b]*projection[j*3+b];
    c2d[0] = c2d[0]+.3; c2d[3] = c2d[3]+.3;
    genia_dual determinant = c2d[0]*c2d[3]-c2d[1]*c2d[2];
    if (!(determinant.value > 0) || !isfinite(determinant.value)) return false;
    genia_dual values[10] = {k[0]*center[0]*invz+k[2], k[4]*center[1]*invz+k[5], center[2],
        c2d[3]/determinant, (0-c2d[1])/determinant, c2d[0]/determinant, opacity,
        .5+.28209479177387814*dc[0], .5+.28209479177387814*dc[1], .5+.28209479177387814*dc[2]};
    for (int i = 7; i < 10; ++i) if (values[i].value < 0) values[i] = genia_dual(0);
    for (int i = 0; i < 10; ++i) {
        if (!isfinite(values[i].value)) return false;
        out[i] = values[i].value;
        if (derivatives) derivatives[i] = values[i].gradient;
    }
    double extent = fmin(3.33, sqrt(2*log(opacity.value*255)));
    out[10] = ceil(extent*sqrt(c2d[0].value));
    out[11] = ceil(extent*sqrt(c2d[3].value));
    return true;
}
/* Scalar duals keep device local-memory usage bounded. A 26-wide dual at
 * every covariance intermediate reserves gigabytes of kernel scratch even
 * for a tiny scene on a many-thread GPU. Re-evaluate one analytic column at
 * a time; this is forward-mode differentiation, not finite differences. */
GG_HD static inline bool genia_gaussian_project(const float row[17], const float camera[12],
    const float k[9], double out[12], double *jacobian)
{
    if (!jacobian) return genia_gaussian_project_column(row, camera, k, -1, out, nullptr);
    for (int p = 0; p < 26; ++p) {
        double derivatives[10];
        if (!genia_gaussian_project_column(row, camera, k, p, out, derivatives)) return false;
        for (int i = 0; i < 10; ++i) jacobian[i*26+p] = derivatives[i];
    }
    return true;
}
#undef GG_HD
#endif
