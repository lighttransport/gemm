#define _GNU_SOURCE
#include <mpi.h>
#define Q38FN_TP_BLOB_IMPLEMENTATION
#include "../common/q38fn_tp_blob.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

enum { CHUNK = 64 << 20 };

static uint64_t hash_update(uint64_t hash, const void *data, size_t bytes)
{
    const unsigned char *p=data;
    while(bytes--){hash^=*p++;hash*=UINT64_C(1099511628211);}
    return hash;
}

static int copy_bytes(int in, int out, uint64_t source, uint64_t destination,
                      uint64_t bytes, void *buffer, uint64_t *hash)
{
    uint64_t done = 0;
    while (done < bytes) {
        size_t count = bytes - done > CHUNK ? CHUNK : (size_t)(bytes - done);
        ssize_t got = pread(in, buffer, count, (off_t)(source + done));
        if (got != (ssize_t)count) return -1;
        if (hash) *hash = hash_update(*hash, buffer, count);
        size_t written = 0;
        while (written < count) {
            ssize_t put = pwrite(out, (char *)buffer + written, count - written,
                                 (off_t)(destination + done + written));
            if (put <= 0) return -1;
            written += (size_t)put;
        }
        posix_fadvise(in, (off_t)(source + done), (off_t)count, POSIX_FADV_DONTNEED);
        posix_fadvise(out, (off_t)(destination + done), (off_t)count, POSIX_FADV_DONTNEED);
        done += count;
    }
    return 0;
}

int main(int argc, char **argv)
{
    int rank, ranks, rc = 1, in = -1, out = -1;
    char dir[4096], old_blob[8192], old_manifest[8192], new_blob[8192], new_manifest[8192];
    char manifest_partial[16384], line[8192];
    FILE *source = NULL, *destination = NULL; void *buffer = NULL; struct stat st;
    MPI_Init(&argc,&argv); MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if (argc != 2 || ranks != Q38FN_TP_RANKS) goto done;
    snprintf(dir,sizeof(dir),"%s/rank-%02d",argv[1],rank);
    snprintf(old_blob,sizeof(old_blob),"%s/tp12-v1.blob",dir);
    snprintf(old_manifest,sizeof(old_manifest),"%s/tp12-v1.manifest",dir);
    snprintf(new_blob,sizeof(new_blob),"%s/tp12-v%d.blob",dir,Q38FN_TP_LAYOUT_VERSION);
    snprintf(new_manifest,sizeof(new_manifest),"%s/tp12-v%d.manifest",dir,Q38FN_TP_LAYOUT_VERSION);
    snprintf(manifest_partial,sizeof(manifest_partial),"%s.partial",new_manifest);
    if (access(new_blob,F_OK) && rename(old_blob,new_blob)) goto done;
    if (stat(new_blob,&st) || st.st_size <= 0 || !(source=fopen(old_manifest,"r")) ||
        !(destination=fopen(manifest_partial,"w")) || !(buffer=malloc(CHUNK)) ||
        (in=open(new_blob,O_RDWR))<0 || (out=dup(in))<0)
        goto done;
    if (!fgets(line,sizeof(line),source)) goto done;
    fprintf(destination,"# Q38FNTP layout=%d rank=%d ranks=%d layers=0\n",
            Q38FN_TP_LAYOUT_VERSION,rank,ranks);
    int tensors = 0;
    while (fgets(line,sizeof(line),source)) {
        if (!strncmp(line,"# COMPLETE",10)) break;
        char parsed[8192]; memcpy(parsed,line,sizeof(parsed));
        q38fn_tp_blob_entry entry={0};
        if (q38fn_tp_blob_parse_entry(parsed,&entry)) goto done;
        if (q38fn_tp_ends_with(entry.name,".linear_attn.in_proj_qkv.weight")) {
            q38fn_tp_plan plan;
            if (q38fn_tp_make_plan(entry.name,entry.shape,entry.ndims,rank,ranks,&plan) ||
                plan.kind != Q38FN_TP_DELTA_QKV) { free(entry.name); goto done; }
            uint64_t offset=entry.offset, hash=UINT64_C(1469598103934665603), cols=entry.shape[1];
            for(int r=0;r<plan.n_ranges;r++){
                uint64_t bytes=plan.range[r].count*cols*2;
                if(copy_bytes(in,out,entry.offset+plan.range[r].start*cols*2,
                              offset,bytes,buffer,&hash)){free(entry.name);goto done;}
                offset+=bytes;
            }
            fprintf(destination,"%llu %llu %016llx %d %d %d",
                    (unsigned long long)entry.offset,(unsigned long long)(offset-entry.offset),
                    (unsigned long long)hash,(int)plan.kind,plan.axis,entry.ndims);
            for(int d=0;d<entry.ndims;d++)fprintf(destination," %llu",(unsigned long long)entry.shape[d]);
            fprintf(destination," %d",plan.n_ranges);
            for(int r=0;r<plan.n_ranges;r++)fprintf(destination," %llu %llu",
                    (unsigned long long)plan.range[r].start,(unsigned long long)plan.range[r].count);
            fprintf(destination," %s\n",entry.name);
        } else fputs(line,destination);
        free(entry.name); tensors++;
    }
    fprintf(destination,"# COMPLETE blob_bytes=%llu tensors=%d\n",
            (unsigned long long)st.st_size,tensors);
    if(fflush(destination)||fsync(fileno(destination))||fdatasync(out))goto done;
    fclose(destination);destination=NULL;close(out);out=-1;
    if(rename(manifest_partial,new_manifest))goto done;
    fprintf(stderr,"Q38FN_TP_UPGRADE rank=%d bytes=%llu tensors=%d\n",rank,
            (unsigned long long)st.st_size,tensors);rc=0;
done:
    if(source)fclose(source);
    if(destination)fclose(destination);
    if(in>=0)close(in);
    if(out>=0)close(out);
    free(buffer);
    if(rc)MPI_Abort(MPI_COMM_WORLD,rc);
    MPI_Finalize();return rc;
}
