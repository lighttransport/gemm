#ifndef DS41F_ALLOC_H
#define DS41F_ALLOC_H
#include <errno.h>
#include <stdlib.h>
#if defined(__linux__) && defined(__LP64__)
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

/* Fresh anonymous pages bypass Fugaku's malloc pool: parallel first writes
 * can place rows near their consuming CMG instead of inheriting old pages.
 * These mappings contain resident data and are never backed by weight files.
 * Include from a translation unit defining _GNU_SOURCE before its headers. */
static inline int ds41f_alloc_resident(void **out,size_t bytes,int fresh)
{
    if(!out||!bytes)return EINVAL;
    *out=NULL;
    #if defined(__linux__) && defined(__LP64__)
    if(fresh){
        void *p=(void *)syscall(SYS_mmap,NULL,bytes,PROT_READ|PROT_WRITE,
                               MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(p==MAP_FAILED)return errno;
        *out=p;return 0;
    }
    #else
    (void)fresh;
    #endif
    return posix_memalign(out,256,bytes);
}

static inline void ds41f_free_resident(void *p,size_t bytes,int fresh)
{
    if(!p)return;
    #if defined(__linux__) && defined(__LP64__)
    if(fresh){(void)syscall(SYS_munmap,p,bytes);return;}
    #else
    (void)bytes;(void)fresh;
    #endif
    free(p);
}
#endif
