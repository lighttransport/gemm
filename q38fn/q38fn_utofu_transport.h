#ifndef Q38FN_UTOFU_TRANSPORT_H
#define Q38FN_UTOFU_TRANSPORT_H

#include <stdint.h>
#include <utofu.h>

typedef struct q38fn_utofu_transport q38fn_utofu_transport;

q38fn_utofu_transport *q38fn_utofu_transport_create(
    utofu_vcq_hdl_t vcq, const utofu_vcq_id_t *peers,
    int rank, int ranks, int max_count, void (*barrier_fn)(void));
void q38fn_utofu_transport_free(q38fn_utofu_transport *transport);
void q38fn_utofu_transport_sum(q38fn_utofu_transport *transport,
                               float *values, int count);
void q38fn_utofu_transport_argmax(q38fn_utofu_transport *transport,
                                  float *value, int32_t *index);

#endif
