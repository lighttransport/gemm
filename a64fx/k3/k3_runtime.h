#ifndef K3_RUNTIME_H
#define K3_RUNTIME_H

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif

#define K3_POOL_ALIGNMENT 256UL

typedef struct k3_pool_block {
    struct k3_pool_block *next;
    size_t capacity;
    size_t requested;
    unsigned long magic;
    int in_use;
} __attribute__((aligned(K3_POOL_ALIGNMENT))) k3_pool_block;

typedef struct {
    k3_pool_block *blocks;
    size_t active_bytes;
    size_t reserved_bytes;
    size_t peak_active_bytes;
    unsigned long numa_mask;
    char name[64];
    char error[512];
} k3_pool;

static inline size_t k3_mem_available_bytes(void) {
    FILE *f=fopen("/proc/meminfo","r");
    if(!f)return 0;char key[64],unit[16];unsigned long long kb=0,value;
    while(fscanf(f,"%63s %llu %15s",key,&value,unit)==3)
        if(!strcmp(key,"MemAvailable:")){kb=value;break;}
    fclose(f);return(size_t)kb*1024;
}

/* Pool accounting excludes libc, OpenMP, and communication-library storage.
 * Procfs RSS/HWM telemetry makes that memory visible in runner postmortems. */
static inline int k3_process_memory_bytes(size_t *rss_out,size_t *hwm_out) {
    if(rss_out)*rss_out=0;if(hwm_out)*hwm_out=0;
    FILE*f=fopen("/proc/self/status","r");if(!f)return errno?errno:EIO;
    char line[256];unsigned long long rss_kb=0,hwm_kb=0;
    while(fgets(line,sizeof line,f)){unsigned long long value;
        if(sscanf(line,"VmRSS: %llu kB",&value)==1)rss_kb=value;
        else if(sscanf(line,"VmHWM: %llu kB",&value)==1)hwm_kb=value;}
    int rc=ferror(f)?EIO:0;if(fclose(f)&&!rc)rc=errno?errno:EIO;
    if(!rc&&(!rss_kb||!hwm_kb))rc=ENODATA;
    if(!rc){if(rss_out)*rss_out=(size_t)rss_kb*1024;
        if(hwm_out)*hwm_out=(size_t)hwm_kb*1024;}
    return rc;
}

static inline void k3_pool_set_error(k3_pool *pool, const char *operation,
        size_t bytes, const char *detail) {
    snprintf(pool->error,sizeof pool->error,
        "k3 pool '%s': %s failed: request=%zu bytes active=%zu reserved=%zu "
        "MemAvailable=%zu bytes%s%s",pool->name,operation,bytes,
        pool->active_bytes,pool->reserved_bytes,k3_mem_available_bytes(),
        detail?" detail=":"",detail?detail:"");
}

/* Interleave future anonymous allocations over the NUMA nodes containing CPUs
 * in this rank's affinity mask.  This excludes A64FX assistant-core nodes. */
static inline unsigned long k3_apply_numa_interleave(void) {
    unsigned long nodemask = 0;
    cpu_set_t affinity;
    if (sched_getaffinity(0, sizeof affinity, &affinity) == 0) {
        for (int node = 0; node < (int)(8 * sizeof nodemask); ++node) {
            char path[128], line[256] = {0};
            snprintf(path, sizeof path,
                     "/sys/devices/system/node/node%d/cpulist", node);
            FILE *f = fopen(path, "r");
            if (!f) continue;
            if (fgets(line, sizeof line, f)) {
                char *p = line;
                while (*p && *p != '\n') {
                    char *end;
                    long first = strtol(p, &end, 10), last = first;
                    if (end == p) break;
                    p = end;
                    if (*p == '-') {
                        last = strtol(p + 1, &end, 10);
                        p = end;
                    }
                    for (long cpu = first; cpu <= last; ++cpu) {
                        if (cpu >= 0 && cpu < CPU_SETSIZE &&
                            CPU_ISSET((int)cpu, &affinity)) {
                            nodemask |= 1UL << node;
                            break;
                        }
                    }
                    if (*p == ',') ++p;
                    else break;
                }
            }
            fclose(f);
        }
    }
    if (!nodemask) nodemask = ~0UL;
    if (syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask,
                (unsigned long)(8 * sizeof nodemask)) != 0) {
        fprintf(stderr, "k3: set_mempolicy(mask=0x%lx): %s\n",
                nodemask, strerror(errno));
        return 0;
    }
    fprintf(stderr, "k3: anonymous allocation NUMA mask=0x%lx\n", nodemask);
    return nodemask;
}

static inline int k3_pool_init(k3_pool *pool, const char *name) {
    if(!pool)return EINVAL;memset(pool,0,sizeof *pool);
    snprintf(pool->name,sizeof pool->name,"%s",name&&*name?name:"k3");
    pool->numa_mask=k3_apply_numa_interleave();
    return 0;
}

static inline void k3_pool_trim(k3_pool *pool) {
    if(!pool)return;k3_pool_block **link=&pool->blocks;
    while(*link){k3_pool_block*b=*link;if(!b->in_use){*link=b->next;
            pool->reserved_bytes-=b->capacity;free(b);}else link=&b->next;}
}

static inline void k3_pool_destroy(k3_pool *pool) {
    if(!pool)return;k3_pool_block*b=pool->blocks;
    while(b){k3_pool_block*next=b->next;free(b);b=next;}
    memset(pool,0,sizeof *pool);
}

static inline void *k3_pool_alloc(k3_pool *pool, size_t bytes) {
    const unsigned long magic=UINT64_C(0x4b33504f4f4c0101);
    if(!pool||!bytes){if(pool)k3_pool_set_error(pool,"allocate",bytes,"invalid size");return NULL;}
    if(bytes>SIZE_MAX-(K3_POOL_ALIGNMENT-1)-sizeof(k3_pool_block)){k3_pool_set_error(pool,"allocate",bytes,"size overflow");return NULL;}
    size_t capacity=(bytes+K3_POOL_ALIGNMENT-1)&~(K3_POOL_ALIGNMENT-1);
    for(k3_pool_block*b=pool->blocks;b;b=b->next)if(!b->in_use&&b->capacity>=capacity){
        b->in_use=1;b->requested=bytes;pool->active_bytes+=b->capacity;
        if(pool->active_bytes>pool->peak_active_bytes)pool->peak_active_bytes=pool->active_bytes;
        return(void*)(b+1);}
    k3_pool_block*b=NULL;int rc=posix_memalign((void**)&b,K3_POOL_ALIGNMENT,sizeof(*b)+capacity);
    if(rc){k3_pool_trim(pool);rc=posix_memalign((void**)&b,K3_POOL_ALIGNMENT,sizeof(*b)+capacity);}
    if(rc){char detail[96];snprintf(detail,sizeof detail,"posix_memalign: %s",strerror(rc));k3_pool_set_error(pool,"allocate",bytes,detail);return NULL;}
    memset(b,0,sizeof *b);b->next=pool->blocks;b->capacity=capacity;b->requested=bytes;b->magic=magic;b->in_use=1;pool->blocks=b;
    pool->active_bytes+=capacity;pool->reserved_bytes+=capacity;
    if(pool->active_bytes>pool->peak_active_bytes)pool->peak_active_bytes=pool->active_bytes;
    return(void*)(b+1);
}

static inline void *k3_pool_calloc(k3_pool *pool, size_t count, size_t size) {
    if(size&&count>SIZE_MAX/size){k3_pool_set_error(pool,"calloc",SIZE_MAX,"size overflow");return NULL;}
    size_t bytes=count*size;void*p=k3_pool_alloc(pool,bytes);if(p)memset(p,0,bytes);return p;
}

static inline int k3_pool_free(k3_pool *pool, void *ptr) {
    const unsigned long magic=UINT64_C(0x4b33504f4f4c0101);
    if(!ptr)return 0;if(!pool)return EINVAL;k3_pool_block*b=pool->blocks;
    while(b&&(void*)(b+1)!=ptr)b=b->next;
    if(!b||b->magic!=magic||!b->in_use){k3_pool_set_error(pool,"free",0,"foreign or duplicate pointer");return EINVAL;}
    b->in_use=0;pool->active_bytes-=b->capacity;b->requested=0;return 0;
}

static inline const char *k3_pool_error(const k3_pool *pool) {
    return pool&&pool->error[0]?pool->error:"no K3 pool error recorded";
}

/* Read a bounded partial-weight blob into anonymous memory.  Dropping source
 * pages as we progress prevents staged files from competing with HBM weights. */
static inline void *k3_pool_load_blob(k3_pool *pool,const char *path,size_t *size_out) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0) {
        char detail[256];snprintf(detail,sizeof detail,"%s: %s",path,strerror(errno));
        k3_pool_set_error(pool,"open/stat blob",0,detail);
        return NULL;
    }
    if (fstat(fd, &st) != 0) {
        char detail[256];snprintf(detail,sizeof detail,"%s: %s",path,strerror(errno));
        k3_pool_set_error(pool,"open/stat blob",0,detail);
        close(fd);
        return NULL;
    }
    if (st.st_size <= 0) {
        char detail[256];snprintf(detail,sizeof detail,"%s: empty blob",path);
        k3_pool_set_error(pool,"open/stat blob",0,detail);
        close(fd);
        return NULL;
    }
    size_t size = (size_t)st.st_size;
    void *base = k3_pool_alloc(pool,size);
    if (!base) {
        close(fd);
        return NULL;
    }
    const size_t chunk = 16UL * 1024 * 1024;
    size_t off = 0;
    while (off < size) {
        size_t want = size - off < chunk ? size - off : chunk;
        size_t done = 0;
        while (done < want) {
            ssize_t n = pread(fd, (char *)base + off + done,
                              want - done, (off_t)(off + done));
            if (n <= 0) {
                char detail[320];snprintf(detail,sizeof detail,"%s offset=%zu: %s",path,off+done,n<0?strerror(errno):"unexpected EOF");
                k3_pool_set_error(pool,"read blob",want-done,detail);
                k3_pool_free(pool,base);
                close(fd);
                return NULL;
            }
            done += (size_t)n;
        }
#ifdef POSIX_FADV_DONTNEED
        (void)posix_fadvise(fd, (off_t)off, (off_t)want,
                            POSIX_FADV_DONTNEED);
#endif
        off += want;
    }
    close(fd);
    if(size_out)*size_out=size;
    return base;
}

#endif
