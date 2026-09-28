/* Bounded SSD row source for the Qwen4 PLE tensor in a split GGUF. */
#ifndef QWEN4_PLE_DISK_H
#define QWEN4_PLE_DISK_H

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define QWEN4_PLE_DISK_WORKERS 16
#define QWEN4_PLE_DISK_PAGE 4096
#define QWEN4_PLE_DISK_PAGES_PER_WORKER 1024

typedef struct qwen4_ple_disk qwen4_ple_disk;

typedef struct {
    qwen4_ple_disk *source;
    pthread_t thread;
    uint64_t seen_generation;
    uint64_t page_tags[QWEN4_PLE_DISK_PAGES_PER_WORKER];
    unsigned char *pages;
    uint64_t reads, hits;
    int head;
    int started;
} qwen4_ple_disk_worker;

struct qwen4_ple_disk {
    int buffered_fd, direct_fd;
    off_t tensor_base;
    uint64_t tensor_bytes, row_count;
    size_t row_bytes;
    int head_dim, heads, type;
    pthread_mutex_t mutex;
    pthread_cond_t start, done;
    uint64_t generation;
    int pending, stopped, failed, sync_ready;
    uint64_t rows[QWEN4_PLE_DISK_WORKERS];
    float *output;
    qwen4_ple_disk_worker workers[QWEN4_PLE_DISK_WORKERS];
};

static int qwen4_ple_disk_page(qwen4_ple_disk_worker *worker, uint64_t page,
                               const unsigned char **out) {
    qwen4_ple_disk *source = worker->source;
    uint32_t slot = (uint32_t)(page % QWEN4_PLE_DISK_PAGES_PER_WORKER);
    if (worker->page_tags[slot] == page) {
        ++worker->hits;
        *out = worker->pages + (size_t)slot * QWEN4_PLE_DISK_PAGE;
        return 0;
    }
    off_t offset = (off_t)(page * QWEN4_PLE_DISK_PAGE);
    unsigned char *dst = worker->pages + (size_t)slot * QWEN4_PLE_DISK_PAGE;
    int fd = source->direct_fd >= 0 ? source->direct_fd : source->buffered_fd;
    ssize_t got = pread(fd, dst, QWEN4_PLE_DISK_PAGE, offset);
    if (got < 0 && source->direct_fd >= 0 && errno == EINVAL) {
        got = pread(source->buffered_fd, dst, QWEN4_PLE_DISK_PAGE, offset);
        fd = source->buffered_fd;
    }
    if (got <= 0) return -1;
    if (got < QWEN4_PLE_DISK_PAGE)
        memset(dst + got, 0, QWEN4_PLE_DISK_PAGE - (size_t)got);
    if (fd == source->buffered_fd)
        posix_fadvise(fd, offset, QWEN4_PLE_DISK_PAGE, POSIX_FADV_DONTNEED);
    worker->page_tags[slot] = page;
    ++worker->reads;
    *out = dst;
    return 0;
}

static int qwen4_ple_disk_read(qwen4_ple_disk_worker *worker, uint64_t row,
                               float *dst) {
    qwen4_ple_disk *source = worker->source;
    if (row >= source->row_count || source->row_bytes > 512) return -1;
    uint64_t rel = row * source->row_bytes;
    if (rel > source->tensor_bytes ||
        source->row_bytes > source->tensor_bytes - rel) return -1;
    uint64_t offset = (uint64_t)source->tensor_base + rel;
    unsigned char packed[512];
    size_t copied = 0;
    while (copied < source->row_bytes) {
        const unsigned char *page;
        uint64_t pos = offset + copied;
        size_t in_page = (size_t)(pos % QWEN4_PLE_DISK_PAGE);
        size_t take = QWEN4_PLE_DISK_PAGE - in_page;
        if (take > source->row_bytes - copied) take = source->row_bytes - copied;
        if (qwen4_ple_disk_page(worker, pos / QWEN4_PLE_DISK_PAGE, &page)) return -1;
        memcpy(packed + copied, page + in_page, take);
        copied += take;
    }
    return dequant_row(source->type, packed, dst, source->head_dim);
}

static void *qwen4_ple_disk_worker_main(void *opaque) {
    qwen4_ple_disk_worker *worker = (qwen4_ple_disk_worker *)opaque;
    qwen4_ple_disk *source = worker->source;
    pthread_mutex_lock(&source->mutex);
    for (;;) {
        while (!source->stopped && worker->seen_generation == source->generation)
            pthread_cond_wait(&source->start, &source->mutex);
        if (source->stopped) break;
        worker->seen_generation = source->generation;
        uint64_t row = source->rows[worker->head];
        float *dst = source->output + (size_t)worker->head * source->head_dim;
        pthread_mutex_unlock(&source->mutex);
        int error = qwen4_ple_disk_read(worker, row, dst);
        pthread_mutex_lock(&source->mutex);
        if (error) source->failed = 1;
        if (--source->pending == 0) pthread_cond_signal(&source->done);
    }
    pthread_mutex_unlock(&source->mutex);
    return NULL;
}

static void qwen4_ple_disk_close(qwen4_ple_disk *source) {
    if (!source) return;
    if (source->sync_ready) {
        pthread_mutex_lock(&source->mutex);
        source->stopped = 1;
        pthread_cond_broadcast(&source->start);
        pthread_mutex_unlock(&source->mutex);
        for (int i = 0; i < source->heads; ++i)
            if (source->workers[i].started) pthread_join(source->workers[i].thread, NULL);
        pthread_cond_destroy(&source->start);
        pthread_cond_destroy(&source->done);
        pthread_mutex_destroy(&source->mutex);
    }
    for (int i = 0; i < QWEN4_PLE_DISK_WORKERS; ++i) free(source->workers[i].pages);
    if (source->direct_fd >= 0) close(source->direct_fd);
    if (source->buffered_fd >= 0) close(source->buffered_fd);
    free(source);
}

static qwen4_ple_disk *qwen4_ple_disk_open(const gguf_context *owner,
        const gguf_tensor_info *tensor, int head_dim, int heads) {
    if (!owner || !tensor || owner->fd < 0 || head_dim <= 0 ||
        heads < 1 || heads > QWEN4_PLE_DISK_WORKERS) return NULL;
    size_t row_bytes = dequant_row_size(tensor->type, head_dim);
    if (!row_bytes || row_bytes > 512 || tensor->dims[0] != (uint64_t)head_dim)
        return NULL;
    qwen4_ple_disk *source = (qwen4_ple_disk *)calloc(1, sizeof(*source));
    if (!source) return NULL;
    source->buffered_fd = source->direct_fd = -1;
    source->buffered_fd = dup(owner->fd);
    if (source->buffered_fd < 0) goto fail;
    char fd_path[64];
    snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", owner->fd);
    source->direct_fd = open(fd_path, O_RDONLY | O_DIRECT);
    source->tensor_base = (off_t)(owner->data_offset + tensor->offset);
    source->row_bytes = row_bytes;
    source->row_count = tensor->dims[1];
    source->tensor_bytes = source->row_count * row_bytes;
    source->head_dim = head_dim;
    source->heads = heads;
    source->type = tensor->type;
    struct stat st;
    if (fstat(source->buffered_fd, &st) || source->tensor_base < 0 ||
        (uint64_t)source->tensor_base > (uint64_t)st.st_size ||
        source->tensor_bytes > (uint64_t)st.st_size - (uint64_t)source->tensor_base)
        goto fail;
    if (pthread_mutex_init(&source->mutex, NULL)) goto fail;
    if (pthread_cond_init(&source->start, NULL)) {
        pthread_mutex_destroy(&source->mutex);
        goto fail;
    }
    if (pthread_cond_init(&source->done, NULL)) {
        pthread_cond_destroy(&source->start);
        pthread_mutex_destroy(&source->mutex);
        goto fail;
    }
    source->sync_ready = 1;
    for (int i = 0; i < heads; ++i) {
        qwen4_ple_disk_worker *worker = &source->workers[i];
        worker->source = source;
        worker->head = i;
        for (int j = 0; j < QWEN4_PLE_DISK_PAGES_PER_WORKER; ++j)
            worker->page_tags[j] = UINT64_MAX;
        if (posix_memalign((void **)&worker->pages, QWEN4_PLE_DISK_PAGE,
                (size_t)QWEN4_PLE_DISK_PAGES_PER_WORKER * QWEN4_PLE_DISK_PAGE) ||
            pthread_create(&worker->thread, NULL, qwen4_ple_disk_worker_main, worker))
            goto fail;
        worker->started = 1;
    }
    return source;
fail:
    qwen4_ple_disk_close(source);
    return NULL;
}

static int qwen4_ple_disk_gather(qwen4_ple_disk *source,
        const uint64_t *rows, float *output) {
    if (!source || !rows || !output) return -1;
    pthread_mutex_lock(&source->mutex);
    memcpy(source->rows, rows, (size_t)source->heads * sizeof(*rows));
    source->output = output;
    source->pending = source->heads;
    source->failed = 0;
    ++source->generation;
    pthread_cond_broadcast(&source->start);
    while (source->pending) pthread_cond_wait(&source->done, &source->mutex);
    int result = source->failed ? -1 : 0;
    pthread_mutex_unlock(&source->mutex);
    return result;
}

#endif
