/* Single C++ translation unit that brings in two header-only libraries
 * required by test_paint_back_project_e2e.c (which is C):
 *   - mesh_vertex_inpaint.h  (uses STL containers internally)
 *   - common/image_utils.h   (pulls stb_image_write for PNG output)
 *
 * Both headers expose extern "C" entry points, so the C runner can call
 * `mesh_vertex_inpaint(...)` and `img_write_png(...)` directly after this
 * .o is linked in. */
#define MESH_VERTEX_INPAINT_IMPLEMENTATION
#include "mesh_vertex_inpaint.h"

#define IMAGE_UTILS_IMPLEMENTATION
#include "../../common/image_utils.h"
