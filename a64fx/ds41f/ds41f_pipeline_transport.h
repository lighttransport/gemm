#ifndef DS41F_PIPELINE_TRANSPORT_H
#define DS41F_PIPELINE_TRANSPORT_H
#include "ds41f_pipeline_runtime.h"

typedef struct {
    int rank;
    float *send_residual, *recv_residual;
    size_t residual_tile_stride, residual_token_stride;
    uint8_t *send_metadata, *recv_metadata;
    size_t metadata_tile_stride, metadata_token_stride;
    void *send_request[2], *recv_request[2];
    uint8_t *send_wire[2], *recv_wire[2];
    size_t recv_tile[2], recv_position[2], recv_count[2];
} ds41f_pipeline_transport_context;

int ds41f_pipeline_transport_init(ds41f_pipeline_transport_context *, int rank,
                                  float *, size_t, size_t, uint8_t *, size_t, size_t,
                                  float *, size_t, size_t, uint8_t *, size_t, size_t);
void ds41f_pipeline_transport_destroy(ds41f_pipeline_transport_context *);
void ds41f_pipeline_transport_bind(ds41f_pipeline_transport *, ds41f_pipeline_transport_context *);

#endif
