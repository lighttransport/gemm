#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <stdint.h>
#include <stdio.h>

static int q21_backend_patch(void *y, const void *x, const void *w, int rows,
                             cudaStream_t stream) {
    cudnnHandle_t handle = nullptr;
    cudnnBackendDescriptor_t xd = nullptr, yd = nullptr, wd = nullptr, cd = nullptr;
    cudnnBackendDescriptor_t op = nullptr, graph = nullptr;
    cudnnBackendDescriptor_t engine = nullptr, config = nullptr, choices[7] = {};
    cudnnBackendDescriptor_t plan = nullptr, pack = nullptr;
    void *workspace = nullptr;
    int64_t workspace_bytes = 0;
    int rc = 1;
    const int64_t xdims[5] = {rows, 3, 2, 16, 16};
    const int64_t xstrides[5] = {1536, 512, 256, 16, 1};
    const int64_t ydims[5] = {rows, 1152, 1, 1, 1};
    const int64_t ystrides[5] = {1152, 1, 1, 1, 1};
    const int64_t wdims[5] = {1152, 3, 2, 16, 16};
    const int64_t wstrides[5] = {1536, 512, 256, 16, 1};
    const int64_t pad[3] = {0, 0, 0}, stride[3] = {2, 16, 16}, dilation[3] = {1, 1, 1};
    const int64_t spatial_dims = 3;
    const int64_t uids[3] = {120, 121, 119};
    void *pointers[3] = {(void *)x, y, (void *)w};
    int64_t alignment = 32;
    cudnnDataType_t bf16 = CUDNN_DATA_BFLOAT16, f32 = CUDNN_DATA_FLOAT;
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    float alpha = 1.0f, beta = 0.0f;
    int64_t engine_index = 23;
    const cudnnBackendKnobType_t knob_types[7] = {
        CUDNN_KNOB_TYPE_TILE_SIZE, CUDNN_KNOB_TYPE_TILEK,
        CUDNN_KNOB_TYPE_STAGES, CUDNN_KNOB_TYPE_REDUCTION_MODE,
        CUDNN_KNOB_TYPE_SPLIT_K_SLC, CUDNN_KNOB_TYPE_IDX_MODE,
        CUDNN_KNOB_TYPE_SPECFILT};
    const int64_t knob_values[7] = {7, 0, 4, 0, 3, 1, 0};

#define BCHK(expr) do { cudnnStatus_t status_ = (expr); if (status_ != CUDNN_STATUS_SUCCESS) { \
    fprintf(stderr, "q21_cudnn_patch: %s failed at line %d: %s\n", #expr, __LINE__, \
            cudnnGetErrorString(status_)); goto done; } } while (0)
#define BSET(desc, attr, type, count, ptr) \
    BCHK(cudnnBackendSetAttribute((desc), (attr), (type), (count), (ptr)))
#define BMAKE(var, type) do { BCHK(cudnnBackendCreateDescriptor((type), &(var))); } while (0)
    BCHK(cudnnCreate(&handle));
    BCHK(cudnnSetStream(handle, stream));
    BMAKE(xd, CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    BSET(xd, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &bf16);
    BSET(xd, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 5, xdims);
    BSET(xd, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 5, xstrides);
    BSET(xd, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uids[0]);
    BSET(xd, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
    BCHK(cudnnBackendFinalize(xd));
    BMAKE(yd, CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    BSET(yd, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &bf16);
    BSET(yd, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 5, ydims);
    BSET(yd, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 5, ystrides);
    BSET(yd, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uids[1]);
    BSET(yd, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
    BCHK(cudnnBackendFinalize(yd));
    BMAKE(wd, CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    BSET(wd, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &bf16);
    BSET(wd, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 5, wdims);
    BSET(wd, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 5, wstrides);
    BSET(wd, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uids[2]);
    BSET(wd, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
    BCHK(cudnnBackendFinalize(wd));
    BMAKE(cd, CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    BSET(cd, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f32);
    BSET(cd, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
    BSET(cd, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial_dims);
    BSET(cd, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 3, pad);
    BSET(cd, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 3, pad);
    BSET(cd, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 3, stride);
    BSET(cd, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 3, dilation);
    BCHK(cudnnBackendFinalize(cd));
    BMAKE(op, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    BSET(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xd);
    BSET(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wd);
    BSET(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yd);
    BSET(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cd);
    BSET(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha);
    BSET(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
    BCHK(cudnnBackendFinalize(op));
    BMAKE(graph, CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    BSET(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
    BSET(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op);
    BCHK(cudnnBackendFinalize(graph));
    BMAKE(engine, CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    BSET(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
    BSET(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &engine_index);
    BCHK(cudnnBackendFinalize(engine));
    for (int i = 0; i < 7; i++) {
        BMAKE(choices[i], CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
        BSET(choices[i], CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
             CUDNN_TYPE_KNOB_TYPE, 1, &knob_types[i]);
        BSET(choices[i], CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
             CUDNN_TYPE_INT64, 1, &knob_values[i]);
        BCHK(cudnnBackendFinalize(choices[i]));
    }
    BMAKE(config, CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    BSET(config, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine);
    BSET(config, CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
         CUDNN_TYPE_BACKEND_DESCRIPTOR, 7, choices);
    BCHK(cudnnBackendFinalize(config));
    BMAKE(plan, CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    BSET(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
    BSET(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &config);
    BCHK(cudnnBackendFinalize(plan));
    BCHK(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                   CUDNN_TYPE_INT64, 1, nullptr, &workspace_bytes));
    if (workspace_bytes && cudaMalloc(&workspace, (size_t)workspace_bytes) != cudaSuccess) goto done;
    BMAKE(pack, CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    BSET(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, pointers);
    BSET(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids);
    BSET(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace);
    BCHK(cudnnBackendFinalize(pack));
    BCHK(cudnnBackendExecute(handle, plan, pack));
    rc = 0;
done:
    if (workspace) cudaFree(workspace);
    if (pack) cudnnBackendDestroyDescriptor(pack);
    if (plan) cudnnBackendDestroyDescriptor(plan);
    if (config) cudnnBackendDestroyDescriptor(config);
    for (int i = 0; i < 7; i++) if (choices[i]) cudnnBackendDestroyDescriptor(choices[i]);
    if (engine) cudnnBackendDestroyDescriptor(engine);
    if (graph) cudnnBackendDestroyDescriptor(graph);
    if (op) cudnnBackendDestroyDescriptor(op);
    if (cd) cudnnBackendDestroyDescriptor(cd);
    if (wd) cudnnBackendDestroyDescriptor(wd);
    if (yd) cudnnBackendDestroyDescriptor(yd);
    if (xd) cudnnBackendDestroyDescriptor(xd);
    if (handle) cudnnDestroy(handle);
    return rc;
#undef BMAKE
#undef BSET
#undef BCHK
}

extern "C" int q21_cudnn_patch_projection(void *output_bf16,
                                             const void *input_bf16,
                                             const void *weight_bf16,
                                             int rows,
                                             cudaStream_t stream) {
    if (!output_bf16 || !input_bf16 || !weight_bf16 || rows <= 0) return 1;

    return q21_backend_patch(output_bf16, input_bf16, weight_bf16,
                             rows, stream);
}
