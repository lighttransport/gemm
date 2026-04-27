/*
 * obj_writer.h — single-header Wavefront .obj writer.
 *
 * Minimal API: vertices + triangle faces (0-based indices on input,
 * emits 1-based per the .obj spec). Optional per-vertex RGB colors
 * as `v x y z r g b` extension (supported by MeshLab/Blender importers).
 *
 * Usage:
 *   #define OBJ_WRITER_IMPLEMENTATION
 *   #include "obj_writer.h"
 *   obj_write("out.obj", verts, nv, faces, nf);
 */

#ifndef OBJ_WRITER_H
#define OBJ_WRITER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int obj_write(const char *path,
              const float *verts, int nv,
              const int32_t *faces, int nf);

int obj_write_i64_faces(const char *path,
                        const float *verts, int nv,
                        const int64_t *faces, int nf);

#ifdef __cplusplus
}
#endif

#endif /* OBJ_WRITER_H */

#ifdef OBJ_WRITER_IMPLEMENTATION

#include <stdio.h>

int obj_write(const char *path,
              const float *verts, int nv,
              const int32_t *faces, int nf)
{
    FILE *fp = fopen(path, "w");
    if (!fp) return 1;
    for (int i = 0; i < nv; i++) {
        fprintf(fp, "v %.6f %.6f %.6f\n",
                verts[i*3+0], verts[i*3+1], verts[i*3+2]);
    }
    for (int i = 0; i < nf; i++) {
        fprintf(fp, "f %d %d %d\n",
                faces[i*3+0]+1, faces[i*3+1]+1, faces[i*3+2]+1);
    }
    fclose(fp);
    return 0;
}

int obj_write_i64_faces(const char *path,
                        const float *verts, int nv,
                        const int64_t *faces, int nf)
{
    FILE *fp = fopen(path, "w");
    if (!fp) return 1;
    for (int i = 0; i < nv; i++) {
        fprintf(fp, "v %.6f %.6f %.6f\n",
                verts[i*3+0], verts[i*3+1], verts[i*3+2]);
    }
    for (int i = 0; i < nf; i++) {
        fprintf(fp, "f %lld %lld %lld\n",
                (long long)(faces[i*3+0]+1),
                (long long)(faces[i*3+1]+1),
                (long long)(faces[i*3+2]+1));
    }
    fclose(fp);
    return 0;
}

#endif /* OBJ_WRITER_IMPLEMENTATION */
