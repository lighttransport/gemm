#ifndef Q38FN_TP_BLOB_H
#define Q38FN_TP_BLOB_H

#include <stddef.h>
#include <stdint.h>
#include "q38fn_tp_layout.h"
#ifdef Q38FN_TP_BLOB_IMPLEMENTATION
#define Q38FN_LOWBIT_IMPLEMENTATION
#endif
#include "q38fn_lowbit.h"

typedef struct {
    char *name;
    uint64_t offset, bytes, hash;
    q38fn_tp_kind kind;
    int axis, ndims, n_ranges;
    uint64_t shape[4];
    q38fn_tp_range range[Q38FN_TP_MAX_SEGMENTS];
    const uint16_t *data;
    q38fn_q5_block *q5_data;
    size_t q5_bytes;
    const int8_t *q8_data;
    const float *q8_scales;
    size_t q8_bytes;
    size_t q8_block_rows, q8_block_bytes;
    int q8_owned;
} q38fn_tp_blob_entry;

typedef struct {
    void *data;
    void *file_map;
    size_t bytes;
    int mapped;
    q38fn_tp_blob_entry *entries;
    int *name_slots;
    size_t name_capacity;
    int n_entries, rank, ranks, layers, layout;
} q38fn_tp_blob;

int q38fn_tp_blob_open(q38fn_tp_blob *blob, const char *rank_dir, int verify);
void q38fn_tp_blob_close(q38fn_tp_blob *blob);
const q38fn_tp_blob_entry *q38fn_tp_blob_find(const q38fn_tp_blob *blob,
                                               const char *name);

#ifdef Q38FN_TP_BLOB_IMPLEMENTATION

#include <errno.h>
#include <fcntl.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static uint64_t q38fn_tp_blob_hash(const void *data, size_t bytes)
{
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = UINT64_C(1469598103934665603);
    while (bytes--) { h ^= *p++; h *= UINT64_C(1099511628211); }
    return h;
}

static uint64_t q38fn_tp_name_hash(const char *name)
{
    uint64_t h = UINT64_C(1469598103934665603);
    while (*name) { h ^= (unsigned char)*name++; h *= UINT64_C(1099511628211); }
    return h;
}

static int q38fn_tp_blob_parse_entry(char *line, q38fn_tp_blob_entry *entry)
{
    char *save = NULL, *token = strtok_r(line, " \t\r\n", &save), *end;
    unsigned long long value;
#define NEXT_U64(dst) do { token = strtok_r(NULL," \t\r\n",&save); if(!token)return -1; \
    errno=0; value=strtoull(token,&end,0); if(errno||!end||*end)return -1; (dst)=(uint64_t)value; } while(0)
    if (!token || token[0] == '#') return 1;
    errno = 0; value = strtoull(token, &end, 0);
    if (errno || !end || *end) return -1;
    entry->offset = (uint64_t)value;
    NEXT_U64(entry->bytes);
    token = strtok_r(NULL," \t\r\n",&save); if(!token)return -1;
    errno=0; value=strtoull(token,&end,16); if(errno||!end||*end)return -1;
    entry->hash=(uint64_t)value;
    uint64_t temp; NEXT_U64(temp); entry->kind = (q38fn_tp_kind)temp;
    token = strtok_r(NULL," \t\r\n",&save); if(!token)return -1;
    errno=0; long signed_value=strtol(token,&end,10); if(errno||!end||*end)return -1;
    entry->axis = (int)signed_value;
    NEXT_U64(temp); entry->ndims = (int)temp;
    if (entry->ndims < 1 || entry->ndims > 4) return -1;
    for (int d = 0; d < entry->ndims; ++d) NEXT_U64(entry->shape[d]);
    NEXT_U64(temp); entry->n_ranges = (int)temp;
    if (entry->n_ranges < 0 || entry->n_ranges > Q38FN_TP_MAX_SEGMENTS) return -1;
    for (int r = 0; r < entry->n_ranges; ++r) {
        NEXT_U64(entry->range[r].start); NEXT_U64(entry->range[r].count);
    }
    token = strtok_r(NULL, " \t\r\n", &save); if (!token) return -1;
    entry->name = strdup(token); if (!entry->name) return -1;
#undef NEXT_U64
    return 0;
}

static size_t q38fn_tp_entry_q5_bytes(const q38fn_tp_blob_entry *entry)
{
    if (!entry || entry->ndims < 2) return 0;
    uint64_t elements = 1;
    for (int d = 0; d < entry->ndims; ++d) elements *= entry->shape[d];
    size_t columns = (size_t)entry->shape[entry->ndims - 1];
    if (entry->kind != Q38FN_TP_FULL && entry->kind != Q38FN_TP_NGRAM_OWNER) {
        uint64_t selected = 0;
        for (int r = 0; r < entry->n_ranges; ++r) selected += entry->range[r].count;
        elements = elements / entry->shape[entry->axis] * selected;
        if (entry->kind == Q38FN_TP_AXIS1 && entry->kind != Q38FN_TP_EXPERT_GATE_UP)
            columns = (size_t)selected;
    }
    if (!columns || elements % columns) return 0;
    return q38fn_q5_bytes((size_t)(elements / columns), columns);
}

static size_t q38fn_tp_entry_q8_bytes(const q38fn_tp_blob_entry *entry)
{
    if (!entry || entry->ndims < 2) return 0;
    uint64_t elements = 1, selected = 0;
    for (int d = 0; d < entry->ndims; ++d) elements *= entry->shape[d];
    size_t columns = (size_t)entry->shape[entry->ndims - 1];
    if (entry->kind != Q38FN_TP_FULL && entry->kind != Q38FN_TP_NGRAM_OWNER) {
        for (int r = 0; r < entry->n_ranges; ++r) selected += entry->range[r].count;
        elements = elements / entry->shape[entry->axis] * selected;
        if (entry->kind == Q38FN_TP_AXIS1 && entry->kind != Q38FN_TP_EXPERT_GATE_UP)
            columns = (size_t)selected;
    }
    if (!columns || elements % columns) return 0;
    return q38fn_q8_bytes((size_t)(elements / columns), columns);
}

static int q38fn_tp_q8_overlay_name(const char *name)
{
    return q38fn_tp_ends_with(name, ".linear_attn.in_proj_qkv.weight") ||
           q38fn_tp_ends_with(name, ".mlp.gate.weight") ||
           q38fn_tp_ends_with(name, ".mlp.experts.gate_up_proj") ||
           q38fn_tp_ends_with(name, ".mlp.shared_expert.gate_proj.weight") ||
           q38fn_tp_ends_with(name, ".mlp.shared_expert.up_proj.weight");
}

int q38fn_tp_blob_open(q38fn_tp_blob *blob, const char *rank_dir, int verify)
{
    char manifest_path[4096], blob_path[4096], line[8192];
    FILE *manifest = NULL; int fd = -1, rc = -1; struct stat st;
    if (!blob || !rank_dir) return -1;
    memset(blob, 0, sizeof(*blob));
    if (snprintf(manifest_path,sizeof(manifest_path),"%s/tp%d-v%d.manifest",rank_dir,Q38FN_TP_RANKS,Q38FN_TP_LAYOUT_VERSION)>=(int)sizeof(manifest_path)||
        snprintf(blob_path,sizeof(blob_path),"%s/tp%d-v%d.blob",rank_dir,Q38FN_TP_RANKS,Q38FN_TP_LAYOUT_VERSION)>=(int)sizeof(blob_path)) return -1;
    manifest = fopen(manifest_path, "r"); if (!manifest) goto done;
    if (!fgets(line, sizeof(line), manifest)) goto done;
    int q8_overlay_manifest = strstr(line, "q8_overlay=1") != NULL;
    if (
        sscanf(line,"# Q38FNTP layout=%d rank=%d ranks=%d layers=%d",
               &blob->layout,&blob->rank,&blob->ranks,&blob->layers)!=4 ||
        blob->layout != Q38FN_TP_LAYOUT_VERSION || blob->ranks != Q38FN_TP_RANKS) goto done;
    int capacity = 256;
    blob->entries = (q38fn_tp_blob_entry *)calloc((size_t)capacity, sizeof(*blob->entries));
    if (!blob->entries) goto done;
    while (fgets(line, sizeof(line), manifest)) {
        if (!strncmp(line,"# COMPLETE",10)) break;
        if (blob->n_entries == capacity) {
            capacity *= 2;
            q38fn_tp_blob_entry *grown = (q38fn_tp_blob_entry *)realloc(
                blob->entries, (size_t)capacity * sizeof(*grown));
            if (!grown) goto done;
            memset(grown + blob->n_entries, 0,
                   (size_t)(capacity - blob->n_entries) * sizeof(*grown));
            blob->entries = grown;
        }
        int parsed = q38fn_tp_blob_parse_entry(line, &blob->entries[blob->n_entries]);
        if (parsed < 0) goto done;
        if (!parsed) blob->n_entries++;
    }
    blob->name_capacity = 1;
    while (blob->name_capacity < (size_t)blob->n_entries * 2u)
        blob->name_capacity <<= 1;
    blob->name_slots = (int *)calloc(blob->name_capacity, sizeof(*blob->name_slots));
    if (!blob->name_slots) goto done;
    for (int i = 0; i < blob->n_entries; ++i) {
        size_t slot = (size_t)q38fn_tp_name_hash(blob->entries[i].name) &
                      (blob->name_capacity - 1u);
        while (blob->name_slots[slot])
            slot = (slot + 1u) & (blob->name_capacity - 1u);
        blob->name_slots[slot] = i + 1;
    }
    if (stat(blob_path,&st) || st.st_size <= 0 || (uint64_t)st.st_size > SIZE_MAX) goto done;
    blob->bytes = (size_t)st.st_size;
    fd = open(blob_path,O_RDONLY); if (fd < 0) goto done;
    blob->file_map=mmap(NULL,blob->bytes,PROT_READ,MAP_PRIVATE,fd,0);
    if(blob->file_map==MAP_FAILED){blob->file_map=NULL;goto done;}
    int file_backed=getenv("Q38FN_TP_FILE_BACKED")&&*getenv("Q38FN_TP_FILE_BACKED")!='0';
    if(file_backed){
        blob->data=blob->file_map; blob->file_map=NULL; blob->mapped=1;
    }else{
        blob->data=mmap(NULL,blob->bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(blob->data==MAP_FAILED){blob->data=NULL;goto done;}blob->mapped=1;
    }
    /* The thread that faults an anonymous page determines its A64FX CMG.  Use
     * 4 MiB tasks so every large tensor is first-touched across all 48 workers.
     * Replicated DeltaNet QKV tensors are sparse anonymous reservations: load
     * only the Q/K/V rows executed by this rank, recovering ~1.7 GB HBM/node. */
    int read_error=0,load_layers=0;
    int ngram_q5=getenv("Q38FN_TP_NGRAM_Q5")&&*getenv("Q38FN_TP_NGRAM_Q5")!='0';
    int ngram_resident=getenv("Q38FN_TP_NGRAM_RESIDENT")&&*getenv("Q38FN_TP_NGRAM_RESIDENT")!='0';
    {const char*v=getenv("Q38FN_TP_LOAD_LAYERS");if(v&&*v)load_layers=atoi(v);}
    for(int ei=0;ei<blob->n_entries&&!read_error&&!file_backed;ei++){
        q38fn_tp_blob_entry*entry=&blob->entries[ei];
        /* A replacement overlay is opened after the base blob.  Keep the
         * base virtual address reservation, but do not fault duplicate BF16
         * pages into HBM for entries that the Q8 overlay shadows. */
        int skip_shadowed = !q8_overlay_manifest &&
            getenv("Q38FN_TP_SKIP_Q8_OVERLAYED") &&
            *getenv("Q38FN_TP_SKIP_Q8_OVERLAYED") != '0' &&
            q38fn_tp_q8_overlay_name(entry->name);
        if (skip_shadowed) continue;
        if(load_layers>0){int layer=-1;if(sscanf(entry->name,"model.language_model.layers.%d.",&layer)==1&&layer>=load_layers)continue;}
        int ngram=strstr(entry->name,"ngram_embedding.shard_")!=NULL;
        if(!verify&&ngram&&!ngram_resident)continue;
        int sparse=!verify&&entry->kind==Q38FN_TP_FULL&&q38fn_tp_ends_with(entry->name,".linear_attn.in_proj_qkv.weight");
        int nr=sparse?3:1;uint64_t begin[3]={entry->offset,0,0},length[3]={entry->bytes,0,0};
        if(sparse){int vh=Q38FN_LINEAR_VALUE_HEADS/blob->ranks,first_gh=blob->rank*vh,first_kh=first_gh/3,last_kh=(first_gh+vh-1)/3;uint64_t qrows=(uint64_t)(last_kh-first_kh+1)*128,row_bytes=(uint64_t)Q38FN_HIDDEN*2;
            begin[0]=entry->offset+(uint64_t)first_kh*128*row_bytes;length[0]=qrows*row_bytes;
            begin[1]=entry->offset+(uint64_t)(2048+first_kh*128)*row_bytes;length[1]=qrows*row_bytes;
            begin[2]=entry->offset+(uint64_t)(4096+first_gh*128)*row_bytes;length[2]=(uint64_t)vh*128*row_bytes;}
        for(int ri=0;ri<nr&&!read_error;ri++){
            /* Stripe small and routed-expert pages in 64 KiB pieces and
             * permute OpenMP thread order across the four 12-core CMGs:
             * 0,12,24,36,1,13,...  Fine granularity matters for large expert
             * tensors because one selected expert is below 1 MiB. */
            int expert=strstr(entry->name,".mlp.experts.")!=NULL;
            int hot_head=!strcmp(entry->name,"lm_head.weight");
            size_t chunk=(expert||hot_head||length[ri]<(16u<<20))?(64u<<10):(4u<<20);
            size_t blocks=((size_t)length[ri]+chunk-1)/chunk;
#ifdef _OPENMP
#if defined(__aarch64__)
#pragma omp parallel num_threads(48)
#else
#pragma omp parallel
#endif
    {
        int tid=omp_get_thread_num(),nth=omp_get_num_threads();
        int per=nth/4,order=(nth>=4&&!(nth%4))?(tid%per)*4+tid/per:tid;
        for(size_t block=(size_t)order;block<blocks;block+=(size_t)nth){
#else
    { for (size_t block = 0; block < blocks; ++block) {
#endif
        size_t offset=(size_t)begin[ri]+block*chunk,want=(size_t)length[ri]-block*chunk;
        if(want>chunk)want=chunk;
        size_t have = 0;
        while (have < want) {
            ssize_t got = pread(fd,(char*)blob->data+offset+have,want-have,(off_t)(offset+have));
            if (got <= 0) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
                read_error = 1;
                break;
            }
            have += (size_t)got;
        }
        if (have == want)
            posix_fadvise(fd,(off_t)offset,(off_t)want,POSIX_FADV_DONTNEED);
    }
    }
        }
    }
    if (read_error) goto done;
    for (int i = 0; i < blob->n_entries; ++i) {
        q38fn_tp_blob_entry *entry = &blob->entries[i];
        if (entry->offset > blob->bytes || entry->bytes > blob->bytes - entry->offset) goto done;
        size_t stored_q5=q38fn_tp_entry_q5_bytes(entry);
        int is_q5=stored_q5&&entry->bytes==stored_q5;
        size_t stored_q8=q38fn_tp_entry_q8_bytes(entry);
        int is_q8=!is_q5&&stored_q8&&entry->bytes==stored_q8;
        int ngram_file=!verify&&strstr(entry->name,"ngram_embedding.shard_")&&!ngram_resident;
        entry->data=(const uint16_t*)((ngram_file||(!file_backed&&!is_q5&&strstr(entry->name,"ngram_embedding.shard_"))?(const char*)blob->file_map:(const char*)blob->data)+entry->offset);
        if(is_q5){entry->q5_data=(q38fn_q5_block*)((ngram_file?(char*)blob->file_map:(char*)blob->data)+entry->offset);entry->q5_bytes=stored_q5;}
        if(is_q8){uint64_t elements=1,columns=entry->shape[entry->ndims-1],selected=0;for(int d=0;d<entry->ndims;d++)elements*=entry->shape[d];if(entry->kind!=Q38FN_TP_FULL&&entry->kind!=Q38FN_TP_NGRAM_OWNER){for(int r=0;r<entry->n_ranges;r++)selected+=entry->range[r].count;elements=elements/entry->shape[entry->axis]*selected;if(entry->kind==Q38FN_TP_AXIS1)columns=selected;}size_t rows=(size_t)(elements/columns),weights=rows*columns;entry->q8_data=(const int8_t*)blob->data+entry->offset;entry->q8_bytes=stored_q8;if(entry->kind==Q38FN_TP_EXPERT_GATE_UP){/* Each expert block retains the physical last dimension; only axis-1 rows were selected. */entry->q8_block_rows=(size_t)selected;entry->q8_block_bytes=entry->q8_block_rows*((size_t)entry->shape[entry->ndims-1]+sizeof(float));entry->q8_scales=NULL;}else{entry->q8_block_rows=rows;entry->q8_block_bytes=stored_q8;entry->q8_scales=(const float*)((const char*)entry->q8_data+weights);}}
        if(!verify&&!is_q5&&!is_q8&&!strcmp(entry->name,"lm_head.weight")&&
           getenv("Q38FN_TP_HEAD_Q8")){
            size_t columns=(size_t)entry->shape[entry->ndims-1];
            size_t rows=(size_t)entry->range[0].count,elements=rows*columns;
            size_t bytes=q38fn_q8_bytes(rows,columns),allocated=(bytes+255u)&~(size_t)255u;
            int8_t *packed=aligned_alloc(256,allocated);
            if(!packed||q38fn_q8_quantize_bf16(packed,(float*)(packed+elements),
                                                entry->data,rows,columns)){
                free(packed);goto done;
            }
            entry->q8_data=packed;entry->q8_scales=(const float*)(packed+elements);
            entry->q8_bytes=bytes;entry->q8_block_rows=rows;
            entry->q8_block_bytes=bytes;entry->q8_owned=1;
        }
        if (!verify && ngram_resident && ngram_q5 && strstr(entry->name, "ngram_embedding.shard_")) {
            size_t columns = (size_t)entry->shape[entry->ndims - 1];
            size_t rows = 1;
            for (int d = 0; d + 1 < entry->ndims; ++d) rows *= (size_t)entry->shape[d];
            if (is_q5) {
                continue;
            }
            size_t chunk_rows = (8u << 20) / (2 * columns);
            if (!chunk_rows) chunk_rows = 1;
            entry->q5_bytes = q38fn_q5_bytes(rows, columns);
            entry->q5_data = aligned_alloc(256, (entry->q5_bytes + 255u) & ~(size_t)255u);
            uint16_t *temporary = aligned_alloc(256, chunk_rows * columns * 2);
            if (!entry->q5_data || !temporary) { free(temporary); goto done; }
            for (size_t row = 0; row < rows; row += chunk_rows) {
                size_t count = rows - row;
                if (count > chunk_rows) count = chunk_rows;
                size_t input_bytes = count * columns * 2;
                size_t have = 0;
                while (have < input_bytes) {
                    ssize_t got = pread(fd, (char *)temporary + have,
                        input_bytes - have,
                        (off_t)(entry->offset + row * columns * 2 + have));
                    if (got <= 0) { free(temporary); goto done; }
                    have += (size_t)got;
                }
                if (q38fn_q5_quantize_bf16(entry->q5_data + row * (columns / 32),
                                           temporary, count, columns)) {
                    free(temporary); goto done;
                }
                posix_fadvise(fd, (off_t)(entry->offset + row * columns * 2),
                              (off_t)input_bytes, POSIX_FADV_DONTNEED);
            }
            free(temporary);
        }
        if (verify && q38fn_tp_blob_hash(entry->data, (size_t)entry->bytes) != entry->hash) goto done;
    }
    rc = 0;
done:
    if (fd >= 0) close(fd);
    if (manifest) fclose(manifest);
    if (rc) q38fn_tp_blob_close(blob);
    return rc;
}

void q38fn_tp_blob_close(q38fn_tp_blob *blob)
{
    if (!blob) return;
    for (int i = 0; i < blob->n_entries; ++i) {
        free(blob->entries[i].name);
        if (blob->entries[i].q5_data) {
            char *p=(char *)blob->entries[i].q5_data;
            int in_data=p>=(char *)blob->data&&p<(char *)blob->data+blob->bytes;
            int in_file=blob->file_map&&p>=(char *)blob->file_map&&
                        p<(char *)blob->file_map+blob->bytes;
            if(!in_data&&!in_file)free(blob->entries[i].q5_data);
        }
        if(blob->entries[i].q8_owned)free((void*)blob->entries[i].q8_data);
    }
    free(blob->name_slots);free(blob->entries);if(blob->data){if(blob->mapped)munmap(blob->data,blob->bytes);else free(blob->data);}if(blob->file_map)munmap(blob->file_map,blob->bytes);memset(blob,0,sizeof(*blob));
}

const q38fn_tp_blob_entry *q38fn_tp_blob_find(const q38fn_tp_blob *blob,
                                               const char *name)
{
    if (!blob || !name) return NULL;
    if (blob->name_slots && blob->name_capacity) {
        size_t slot = (size_t)q38fn_tp_name_hash(name) & (blob->name_capacity - 1u);
        for (size_t probe = 0; probe < blob->name_capacity; ++probe) {
            int value = blob->name_slots[slot];
            if (!value) return NULL;
            if (!strcmp(blob->entries[value - 1].name, name))
                return &blob->entries[value - 1];
            slot = (slot + 1u) & (blob->name_capacity - 1u);
        }
        return NULL;
    }
    for (int i = 0; i < blob->n_entries; ++i)
        if (!strcmp(blob->entries[i].name, name)) return &blob->entries[i];
    return NULL;
}
#endif
#endif
