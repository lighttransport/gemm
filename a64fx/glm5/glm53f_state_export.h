#ifndef GLM53F_STATE_EXPORT_H
#define GLM53F_STATE_EXPORT_H
/* Canonical comparison metadata keeps TP head intervals explicit. Diagnostic
 * files are exclusive and bounded; no model clone is created. */
static int target_export_blob(const char *path, const void *data, size_t bytes) {
    glm53f_state_io io = {0}; io.file = fopen(path, "wbx");
    if (!io.file) return -1;
    int failed = glm53f_state_io_bytes(&io, data, bytes, "export") ||
        fflush(io.file) || fsync(fileno(io.file));
    failed |= fclose(io.file) != 0;
    return failed ? -1 : 0;
}
static inline int target_export_hidden_begin(glm53f_target_model_12n *m,const char *prefix,int count) {
    int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int root=m->dist?8:0;
    if(rank!=root)return 0;
    if(!prefix||count<1||m->hidden_export)return-1;
    char path[4096];int n=snprintf(path,sizeof(path),"%s.hidden",prefix);
    if(n<0||n>=(int)sizeof(path))return-1;
    glm53f_state_io *io=calloc(1,sizeof(*io));if(!io)return-1;
    io->file=fopen(path,"wbx");if(!io->file){free(io);return-1;}
    m->hidden_export=io;m->hidden_export_count=m->hidden_export_remaining=count;return 0;
}
int glm53f_target_export_fields_12n(glm53f_target_model_12n *m, const char *prefix) {
    if (!m || !prefix || !*prefix || m->trace) return -1;
    int hidden_count=0;
    if(m->hidden_export){
        if(m->hidden_export_remaining)return-1;
        glm53f_state_io *io=m->hidden_export;
        int failed=fflush(io->file)||fsync(fileno(io->file));failed|=fclose(io->file)!=0;
        free(io);m->hidden_export=NULL;if(failed)return-1;hidden_count=m->hidden_export_count;
    }
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char path[4096], filename[4096];
    int n = snprintf(path, sizeof(path), "%s.rank%02d.fields", prefix, rank);
    if (n < 0 || n >= (int)sizeof(path)) return -1;
    FILE *meta = fopen(path, "wx"); if (!meta) return -1;
    int failed = fprintf(meta, "GLM53F_FIELDS_V1 %d %d %d\n", rank, m->first_layer, m->end_layer) < 0;
    if(hidden_count)failed|=fprintf(meta,"HIDDEN %d\n",hidden_count)<0;
    for (int l = m->first_layer; l < m->end_layer && !failed; ++l) {
        n = snprintf(filename, sizeof(filename), "%s.rank%02d.layer%02d.%s", prefix, rank, l, m->kda[l] ? "kda" : "sparse");
        if (n < 0 || n >= (int)sizeof(filename)) { failed = 1; break; }
        if (m->kda[l]) {
            int first, count;
            size_t bytes = glm53f_kda_state_bytes_12n(m->kda[l]);
            void *data = malloc(bytes);
            failed = !data || glm53f_kda_head_range_12n(m->kda[l], &first, &count) ||
                glm53f_kda_save_state_12n(m->kda[l], data, bytes) || target_export_blob(filename, data, bytes);
            free(data);
            if (!failed) failed = fprintf(meta, "KDA %d %d %d\n", l, first, count) < 0;
        } else {
            glm53f_state_io io = {0}; io.file = fopen(filename, "wbx");
            if (!io.file) { failed = 1; break; }
            failed = glm53f_sparse_state_io_12n(m->sparse[l], &io) || fflush(io.file) || fsync(fileno(io.file));
            failed |= fclose(io.file) != 0;
            if (!failed) failed = fprintf(meta, "SPARSE %d\n", l) < 0;
        }
    }
    if (!failed && m->head && m->last_streams) {
        n = snprintf(filename, sizeof(filename), "%s.rank%02d.streams", prefix, rank);
        failed = n < 0 || n >= (int)sizeof(filename) || target_export_blob(filename, m->last_streams, FLAT * sizeof(float));
        if (!failed) failed = fprintf(meta, "STREAMS\n") < 0;
    }
    failed |= fflush(meta) != 0; failed |= fsync(fileno(meta)) != 0; failed |= fclose(meta) != 0;
    return failed ? -1 : 0;
}
static inline int target_export_final(glm53f_target_model_12n *m,const char *prefix) {
    if(!prefix)return 0;
    char path[4096];int n=snprintf(path,sizeof(path),"%s.decode",prefix);
    if(n<0||n>=(int)sizeof(path))return-1;
    return glm53f_target_export_fields_12n(m,path);
}
#endif
