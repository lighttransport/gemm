#define _POSIX_C_SOURCE 200809L
#include "glm53f_pipeline.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t magic;
    int sequence, offset, tokens, flat, source_stage, cuts[2];
} pipeline_header;
typedef struct {
    float *data;
    pipeline_header header;
    MPI_Request receive[2], send[2];
} pipeline_slot;
static int fail(const glm53f_dist *d, const char *phase) {
    fprintf(stderr, "GLM53F_PIPELINE_FAIL world_rank=%d stage=%d phase=%s\n",
        d->map.world_rank, d->map.stage, phase);
    MPI_Abort(d->world, 2);
    return -1;
}
static int post_receive(const glm53f_dist *d, pipeline_slot *s, int slot,
        int tokens, int flat) {
    if (MPI_Irecv(&s->header, sizeof(s->header), MPI_BYTE, d->map.stage - 1,
            4 * slot, d->pipeline, &s->receive[0]) != MPI_SUCCESS ||
        MPI_Irecv(s->data, tokens * flat, MPI_FLOAT, d->map.stage - 1,
            4 * slot + 1, d->pipeline, &s->receive[1]) != MPI_SUCCESS) return -1;
    return 0;
}
static int wait_send(pipeline_slot *s) {
    return MPI_Waitall(2, s->send, MPI_STATUSES_IGNORE) == MPI_SUCCESS ? 0 : -1;
}
int glm53f_pipeline_run(const glm53f_dist *d, int positions, int flat,
        glm53f_pipeline_schedule schedule, glm53f_pipeline_callback producer,
        glm53f_pipeline_callback executor, glm53f_pipeline_callback consumer,
        void *context, glm53f_pipeline_profile *profile) {
    if (!d || !d->initialized) return -1;
    const int batch = d->config.microbatch;
    size_t bytes = 0;
    int valid = positions > 0 && flat > 0 && executor &&
        (d->map.stage != 0 || producer) &&
        (d->map.stage != d->map.stages - 1 || consumer) &&
        (schedule == GLM53F_PIPELINE_SERIAL || schedule == GLM53F_PIPELINE_OVERLAP) &&
        !glm53f_pipeline_buffer_bytes(batch, flat, &bytes) && flat <= INT_MAX / batch;
    int all, settings[3] = {positions, flat, schedule}, low[3], high[3];
    if (MPI_Allreduce(&valid, &all, 1, MPI_INT, MPI_MIN, d->world) != MPI_SUCCESS ||
        MPI_Allreduce(settings, low, 3, MPI_INT, MPI_MIN, d->world) != MPI_SUCCESS ||
        MPI_Allreduce(settings, high, 3, MPI_INT, MPI_MAX, d->world) != MPI_SUCCESS)
        return fail(d, "settings");
    if (!all || memcmp(low, high, sizeof(low))) return -1;
    float *storage = NULL;
    valid = !posix_memalign((void **)&storage, 256, bytes);
    if (MPI_Allreduce(&valid, &all, 1, MPI_INT, MPI_MIN, d->world) != MPI_SUCCESS)
        return fail(d, "allocation");
    if (!all) { free(storage); return -1; }
    pipeline_slot slots[2]; memset(slots, 0, sizeof(slots));
    for (int i = 0; i < 2; ++i) {
        slots[i].data = storage + (size_t)i * batch * flat;
        for (int k = 0; k < 2; ++k) slots[i].send[k] = slots[i].receive[k] = MPI_REQUEST_NULL;
    }
    glm53f_pipeline_profile p = {0};
    const int count = 1 + (positions - 1) / batch;
    if (schedule == GLM53F_PIPELINE_OVERLAP && d->map.stage > 0)
        for (int i = 0; i < 2 && i < count; ++i) {
            int n = positions - i * batch; if (n > batch) n = batch;
            if (post_receive(d, &slots[i], i, n, flat)) return fail(d, "post_receive");
        }
    for (int seq = 0; seq < count; ++seq) {
        const int index = seq % 2, offset = seq * batch;
        int n = positions - offset; if (n > batch) n = batch;
        pipeline_slot *s = &slots[index];
        double t = MPI_Wtime();
        if (wait_send(s)) return fail(d, "send_reuse");
        p.send_wait_seconds += MPI_Wtime() - t;
        if (d->map.stage) {
            if (schedule == GLM53F_PIPELINE_SERIAL && post_receive(d, s, index, n, flat))
                return fail(d, "serial_receive");
            MPI_Status status[2]; t = MPI_Wtime();
            if (MPI_Waitall(2, s->receive, status) != MPI_SUCCESS) return fail(d, "receive");
            p.receive_seconds += MPI_Wtime() - t;
            int received, header_bytes;
            if (MPI_Get_count(&status[0], MPI_BYTE, &header_bytes) != MPI_SUCCESS ||
                MPI_Get_count(&status[1], MPI_FLOAT, &received) != MPI_SUCCESS)
                return fail(d, "receive_count");
            if (header_bytes != (int)sizeof(s->header) || s->header.magic != UINT64_C(0x474c4d5050335434) ||
                s->header.sequence != seq || s->header.offset != offset ||
                s->header.tokens != n || s->header.flat != flat ||
                s->header.source_stage != d->map.stage - 1 || received != n * flat ||
                memcmp(s->header.cuts, d->config.cuts, sizeof(s->header.cuts)))
                return fail(d, "header_or_payload");
        }
        t = MPI_Wtime();
        if ((!d->map.stage && producer(context, d, s->data, offset, n, flat)) ||
            executor(context, d, s->data, offset, n, flat) ||
            (d->map.stage == d->map.stages - 1 && consumer(context, d, s->data, offset, n, flat)))
            return fail(d, "callback");
        p.compute_seconds += MPI_Wtime() - t;
        if (d->map.stage < d->map.stages - 1) {
            memset(&s->header, 0, sizeof(s->header));
            s->header.magic = UINT64_C(0x474c4d5050335434);
            s->header.sequence = seq; s->header.offset = offset;
            s->header.tokens = n; s->header.flat = flat; s->header.source_stage = d->map.stage;
            memcpy(s->header.cuts, d->config.cuts, sizeof(s->header.cuts));
            if (MPI_Isend(&s->header, sizeof(s->header), MPI_BYTE, d->map.stage + 1,
                    4 * index, d->pipeline, &s->send[0]) != MPI_SUCCESS ||
                MPI_Isend(s->data, n * flat, MPI_FLOAT, d->map.stage + 1,
                    4 * index + 1, d->pipeline, &s->send[1]) != MPI_SUCCESS)
                return fail(d, "send");
        }
        if (schedule == GLM53F_PIPELINE_SERIAL || (d->map.stage && seq + 2 < count)) {
            t = MPI_Wtime(); if (wait_send(s)) return fail(d, "send_completion");
            p.send_wait_seconds += MPI_Wtime() - t;
        }
        if (schedule == GLM53F_PIPELINE_OVERLAP && d->map.stage && seq + 2 < count) {
            int next = positions - (seq + 2) * batch; if (next > batch) next = batch;
            if (post_receive(d, s, index, next, flat)) return fail(d, "receive_reuse");
        }
        if (schedule == GLM53F_PIPELINE_SERIAL && MPI_Barrier(d->world) != MPI_SUCCESS)
            return fail(d, "serial_barrier");
        ++p.microbatches; p.positions += n;
    }
    for (int i = 0; i < 2; ++i) if (wait_send(&slots[i])) return fail(d, "drain");
    if (MPI_Barrier(d->world) != MPI_SUCCESS) return fail(d, "drain_barrier");
    free(storage);
    if (profile) *profile = p;
    return 0;
}

/* One-token decode keeps a caller-owned 64KiB stream buffer. A completed
 * step must be followed by the caller's world token broadcast before reuse. */
int glm53f_pipeline_step(const glm53f_dist *d,int sequence,int flat,
        glm53f_pipeline_callback producer,glm53f_pipeline_callback executor,
        glm53f_pipeline_callback consumer,void *context,float *streams,
        glm53f_pipeline_profile *profile) {
    if(!d||!d->initialized)return-1;
    int valid=sequence>=0&&flat>=4&&flat%4==0&&streams&&executor&&(!d->map.stage?producer!=NULL:1)&&
        (d->map.stage==d->map.stages-1?consumer!=NULL:1),all;
    int settings[2]={sequence,flat},low[2],high[2];
    if(MPI_Allreduce(&valid,&all,1,MPI_INT,MPI_MIN,d->world)!=MPI_SUCCESS||
       MPI_Allreduce(settings,low,2,MPI_INT,MPI_MIN,d->world)!=MPI_SUCCESS||
       MPI_Allreduce(settings,high,2,MPI_INT,MPI_MAX,d->world)!=MPI_SUCCESS)return fail(d,"step_settings");
    if(!all||memcmp(low,high,sizeof(low)))return-1;
    pipeline_header header={0};glm53f_pipeline_profile p={0};double t=MPI_Wtime();
    if(d->map.stage){MPI_Status status;int count;
        if(MPI_Recv(&header,sizeof(header),MPI_BYTE,d->map.stage-1,8,d->pipeline,&status)!=MPI_SUCCESS||
           MPI_Get_count(&status,MPI_BYTE,&count)!=MPI_SUCCESS||count!=(int)sizeof(header))return fail(d,"step_header_count");
        if(header.magic!=UINT64_C(0x474c4d5050335434)||header.sequence!=sequence||header.offset!=sequence||header.tokens!=1||header.flat!=flat||header.source_stage!=d->map.stage-1||memcmp(header.cuts,d->config.cuts,sizeof(header.cuts)))return fail(d,"step_header");
        if(MPI_Recv(streams,flat,MPI_FLOAT,d->map.stage-1,9,d->pipeline,&status)!=MPI_SUCCESS||
           MPI_Get_count(&status,MPI_FLOAT,&count)!=MPI_SUCCESS||count!=flat)return fail(d,"step_payload");
        p.receive_seconds=MPI_Wtime()-t;
    }
    t=MPI_Wtime();
    if((!d->map.stage&&producer(context,d,streams,sequence,1,flat))||executor(context,d,streams,sequence,1,flat)||
       (d->map.stage==d->map.stages-1&&consumer(context,d,streams,sequence,1,flat)))return fail(d,"step_callback");
    p.compute_seconds=MPI_Wtime()-t;
    if(d->map.stage<d->map.stages-1){header.magic=UINT64_C(0x474c4d5050335434);header.sequence=sequence;header.offset=sequence;header.tokens=1;header.flat=flat;header.source_stage=d->map.stage;memcpy(header.cuts,d->config.cuts,sizeof(header.cuts));t=MPI_Wtime();
        if(MPI_Send(&header,sizeof(header),MPI_BYTE,d->map.stage+1,8,d->pipeline)!=MPI_SUCCESS||
           MPI_Send(streams,flat,MPI_FLOAT,d->map.stage+1,9,d->pipeline)!=MPI_SUCCESS)return fail(d,"step_send");
        p.send_wait_seconds=MPI_Wtime()-t;
    }
    p.microbatches=p.positions=1;if(profile)*profile=p;return 0;
}
