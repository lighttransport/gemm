#include "q38fn_utofu_transport.h"
#include "../a64fx/utofu-tests/tp_allreduce.h"

#include <stdlib.h>

struct q38fn_utofu_transport { tp_comm comm; };

q38fn_utofu_transport *q38fn_utofu_transport_create(
    utofu_vcq_hdl_t vcq, const utofu_vcq_id_t *peers,
    int rank, int ranks, int max_count, void (*barrier_fn)(void))
{
    q38fn_utofu_transport *t = calloc(1, sizeof(*t));
    if (!t || tp_comm_init(&t->comm, vcq, peers, rank, ranks,
                           max_count, barrier_fn)) {
        free(t); return NULL;
    }
    return t;
}

void q38fn_utofu_transport_free(q38fn_utofu_transport *t)
{
    if (t) { tp_comm_free(&t->comm); free(t); }
}

void q38fn_utofu_transport_sum(q38fn_utofu_transport *t,
                               float *values, int count)
{
    tp_allreduce_sum(&t->comm, values, count);
}

void q38fn_utofu_transport_argmax(q38fn_utofu_transport *t,
                                  float *value, int32_t *index)
{
    tp_allreduce_argmax(&t->comm, value, index);
}
