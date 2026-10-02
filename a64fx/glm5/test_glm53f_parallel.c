#include "glm53f_parallel.h"
#include <assert.h>
#include <stdio.h>

static void option(const char *key, const char *value, int expected) {
    glm53f_parallel_config c = glm53f_parallel_default(), before = c;
    char *args[] = {"test", (char *)key, (char *)value};
    int index = 1;
    assert(glm53f_parallel_option(&c, value ? 3 : 2, args, &index) == expected);
    if (expected < 0) assert(!memcmp(&c, &before, sizeof(c)));
}
int main(void) {
    glm53f_parallel_config c = glm53f_parallel_default();
    glm53f_parallel_map m;
    assert(glm53f_parallel_valid(&c));
    for (int rank = 0; rank < 12; ++rank) {
        assert(!glm53f_parallel_map_rank(&c, rank, 12, &m));
        assert(m.stage == 0 && m.stages == 1 && m.tp_rank == rank && m.tp_size == 12);
        assert(m.first_layer == 0 && m.end_layer == 45);
    }
    c.layout = GLM53F_PP3_TP4;
    int layouts = 0;
    for (int a = 1; a < 44; ++a) for (int b = a + 1; b < 45; ++b) {
        c.cuts[0] = a; c.cuts[1] = b;
        int coverage[45] = {0};
        for (int rank = 0; rank < 12; ++rank) {
            assert(!glm53f_parallel_map_rank(&c, rank, 12, &m));
            assert(m.stage == rank / 4 && m.tp_rank == rank % 4 && m.tp_size == 4);
            for (int layer = m.first_layer; layer < m.end_layer; ++layer) ++coverage[layer];
        }
        for (int layer = 0; layer < 45; ++layer) assert(coverage[layer] == 4);
        ++layouts;
    }
    assert(glm53f_parallel_map_rank(&c, -1, 12, &m) < 0);
    assert(glm53f_parallel_map_rank(&c, 12, 12, &m) < 0);
    assert(glm53f_parallel_map_rank(&c, 0, 4, &m) < 0);
    option("--parallel-layout", "pp3-tp4", 1);
    option("--parallel-layout", "tp12", 1);
    option("--parallel-layout", "pp4-tp3", -1);
    option("--pipeline-microbatch", "512", 1);
    option("--pipeline-microbatch", "1024", 1);
    option("--pipeline-microbatch", "2048", 1);
    option("--pipeline-microbatch", "256", -1);
    option("--pipeline-microbatch", "1024garbage", -1);
    option("--pipeline-microbatch", "999999999999999999999", -1);
    option("--pipeline-microbatch", NULL, -1);
    option("--pipeline-cuts", "1,44", 1);
    option("--pipeline-cuts", "0,30", -1);
    option("--pipeline-cuts", "15,45", -1);
    option("--pipeline-cuts", "30,15", -1);
    option("--pipeline-cuts", "15,15", -1);
    option("--pipeline-cuts", "15,30,31", -1);
    option("--pipeline-cuts", "15,", -1);
    option("--unrelated", "anything", 0);
    size_t bytes;
    assert(!glm53f_pipeline_buffer_bytes(2048, 16384, &bytes));
    assert(bytes == (size_t)256 * 1024 * 1024);
    assert(glm53f_pipeline_buffer_bytes(2049, 16384, &bytes) < 0);
    assert(glm53f_pipeline_buffer_bytes(1024, 16383, &bytes) < 0);
    assert(glm53f_pipeline_buffer_bytes(1024, 0, &bytes) < 0);
    assert(glm53f_pipeline_buffer_bytes(1024, 16384, NULL) < 0);
    printf("GLM53F_PARALLEL_PASS cut_layouts=%d ranks=12\n", layouts);
    return 0;
}
