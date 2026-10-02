#ifndef GLM53F_PP_RUNNER_H
#define GLM53F_PP_RUNNER_H
/* Included by the command-line runner after its model implementation. */
typedef struct {
    glm53f_target_model_12n *model;
    const int *prompt;
    int decode, token;
} target_pp_call;
static int target_pp_produce(void *context, const glm53f_dist *d,
        float *streams, int offset, int tokens, int flat) {
    target_pp_call *c = context;
    (void)d;
    if (flat != FLAT || (c->decode && tokens != 1)) return -1;
    return glm53f_target_model_embed_batch_12n(c->model,
        c->decode ? &c->token : c->prompt + offset, tokens, streams);
}
static int target_pp_execute(void *context, const glm53f_dist *d,
        float *streams, int offset, int tokens, int flat) {
    target_pp_call *c = context;
    (void)offset;
    if (flat != FLAT) return -1;
    return glm53f_target_model_layers_batch_12n(c->model, streams, tokens,
        d->map.first_layer, d->map.end_layer);
}
static int target_pp_consume(void *context, const glm53f_dist *d,
        float *streams, int offset, int tokens, int flat) {
    (void)context; (void)d; (void)streams; (void)offset; (void)tokens;
    /* The owned executor retains the final stream outside transfer slots. */
    return flat == FLAT ? 0 : -1;
}
static void target_pp_guard(const glm53f_dist *d, long *minimum) {
    long available = target_available_kb(), global;
    if (MPI_Allreduce(&available, &global, 1, MPI_LONG, MPI_MIN, d->world) != MPI_SUCCESS)
        MPI_Abort(d->world, 2);
    if (global < *minimum) *minimum = global;
    if (global < 6L * 1024 * 1024) {
        if (!d->map.world_rank) fprintf(stderr, "GLM53F_PP_HEADROOM min_GiB=%.6f reject\n", global / 1048576.0);
        MPI_Abort(d->world, 3);
    }
}
static int target_pp_readout(target_pp_call *c, const glm53f_dist *d, float *logit) {
    int root = (d->map.stages - 1) * d->map.tp_size;
    if (d->map.stage == d->map.stages - 1 &&
        glm53f_target_model_readout_12n(c->model, &c->token, logit)) return -1;
    if (MPI_Bcast(&c->token, 1, MPI_INT, root, d->world) != MPI_SUCCESS ||
        MPI_Bcast(logit, 1, MPI_FLOAT, root, d->world) != MPI_SUCCESS) return -1;
    return 0;
}
static int target_pp_eos(int token) {
    return token == 154820 || token == 154827 || token == 154829;
}
static int target_pp_run(const glm53f_parallel_config *parallel,
        glm53f_pipeline_schedule schedule, const char *image_root,
        const char *source, const char *routed, const char *shared, int capacity,
        const glm53f_prefill_config *prefill, const int *prompt, int count,
        int *ids, int steps, const char *output, int ignore_eos,
        int touch_cache, int load_only, const char *state_export) {
    glm53f_dist d;
    if (glm53f_dist_init(&d, MPI_COMM_WORLD, parallel)) MPI_Abort(MPI_COMM_WORLD, 2);
    char paths[6][PATH_MAX];
    const char *names[] = {"core", "dense", "kda", "sparse", "embed", "head"};
    for (int i = 0; i < 6; ++i) {
        int n = snprintf(paths[i], sizeof(paths[i]), "%s/%s", image_root, names[i]);
        if (n < 0 || n >= (int)sizeof(paths[i])) MPI_Abort(d.world, 2);
    }
    glm53f_pp_images images = {source, paths[0], routed, shared,
        paths[1], paths[2], paths[3], paths[4], paths[5]};
    double begin = MPI_Wtime(), elapsed, load;
    glm53f_target_model_12n *model = glm53f_target_model_create_dist(&d, &images, capacity);
    if (!model || glm53f_target_model_configure_prefill_12n(model, prefill) ||
        (touch_cache && glm53f_target_model_touch_cache_12n(model))) MPI_Abort(d.world, 2);
    elapsed = MPI_Wtime() - begin;
    MPI_Reduce(&elapsed, &load, 1, MPI_DOUBLE, MPI_MAX, 0, d.world);
    long minimum = LONG_MAX;
    target_pp_guard(&d, &minimum);
    if (!d.map.world_rank) {
        printf("GLM53F_PP_LOAD seconds=%.6f cuts=%d,%d microbatch=%d schedule=%s capacity=%d min_MemAvailable_GiB=%.6f\n",
            load, parallel->cuts[0], parallel->cuts[1], parallel->microbatch,
            schedule == GLM53F_PIPELINE_SERIAL ? "serialized" : "two-slot", capacity, minimum / 1048576.0);
        fflush(stdout);
    }
    if (load_only) { glm53f_target_model_free_12n(model); glm53f_dist_free(&d); return 0; }
    if (state_export && target_export_hidden_begin(model, state_export, count)) MPI_Abort(d.world, 2);
    if (state_export && model->moe && glm53f_moe_set_route_export_12n(model->moe, state_export)) MPI_Abort(d.world, 2);
    target_pp_call c = {model, prompt, 0, 0};
    glm53f_pipeline_profile profile;
    float logit = 0.0f;
    MPI_Barrier(d.world); begin = MPI_Wtime();
    if (glm53f_pipeline_run(&d, count, FLAT, schedule, target_pp_produce,
            target_pp_execute, target_pp_consume, &c, &profile) ||
        target_pp_readout(&c, &d, &logit)) MPI_Abort(d.world, 2);
    elapsed = MPI_Wtime() - begin;
    double prefill_seconds;
    MPI_Allreduce(&elapsed, &prefill_seconds, 1, MPI_DOUBLE, MPI_MAX, d.world);
    target_pp_guard(&d, &minimum);
    if (state_export && glm53f_target_export_fields_12n(model, state_export)) MPI_Abort(d.world, 2);
    if (!d.map.world_rank) ids[0] = c.token;
    int generated = 1, decode_steps = 0;
    c.decode = 1;
    float *streams = a256(FLAT * sizeof(float));
    if (!streams) MPI_Abort(d.world, 2);
    MPI_Barrier(d.world); begin = MPI_Wtime();
    while (generated < steps && (ignore_eos || !target_pp_eos(c.token))) {
        if (glm53f_pipeline_step(&d, decode_steps, FLAT, target_pp_produce,
                target_pp_execute, target_pp_consume, &c, streams, NULL) ||
            target_pp_readout(&c, &d, &logit)) MPI_Abort(d.world, 2);
        if (!d.map.world_rank) ids[generated] = c.token;
        ++generated; ++decode_steps;
        if (!(decode_steps % 32)) target_pp_guard(&d, &minimum);
    }
    elapsed = MPI_Wtime() - begin;
    double decode_seconds;
    MPI_Allreduce(&elapsed, &decode_seconds, 1, MPI_DOUBLE, MPI_MAX, d.world);
    target_pp_guard(&d, &minimum);
    if (state_export && target_export_final(model, state_export)) MPI_Abort(d.world, 2);
    if (!d.map.world_rank) {
        FILE *f = fopen(output, "w");
        if (!f) MPI_Abort(d.world, 2);
        for (int i = 0; i < generated; ++i) if (fprintf(f, "%d\n", ids[i]) < 0) MPI_Abort(d.world, 2);
        if (fclose(f)) MPI_Abort(d.world, 2);
        printf("GLM53F_PP_RESULT diagnostic=%d prompt_tokens=%d generated=%d decode_steps=%d prefill_seconds=%.9f prefill_tok_s=%.6f decode_seconds=%.9f decode_tok_s=%.6f min_MemAvailable_GiB=%.6f\n",
            state_export != NULL, count, generated, decode_steps, prefill_seconds, count / prefill_seconds,
            decode_seconds, decode_steps ? decode_steps / decode_seconds : 0.0, minimum / 1048576.0);
        fflush(stdout);
    }
    double times[] = {profile.compute_seconds, profile.receive_seconds, profile.send_wait_seconds}, maximum[3];
    MPI_Reduce(times, maximum, 3, MPI_DOUBLE, MPI_MAX, 0, d.world);
    if (!d.map.world_rank) printf("GLM53F_PP_PIPELINE compute_max=%.6f receive_max=%.6f send_wait_max=%.6f microbatches=%d\n",
        maximum[0], maximum[1], maximum[2], profile.microbatches);
    double stage_times[9] = {0}, stage_maximum[9];
    memcpy(stage_times + 3 * d.map.stage, times, sizeof(times));
    MPI_Reduce(stage_times, stage_maximum, 9, MPI_DOUBLE, MPI_MAX, 0, d.world);
    if (!d.map.world_rank) for (int stage = 0; stage < 3; ++stage) {
        int first = stage == 0 ? 0 : parallel->cuts[stage - 1];
        int end = stage == 2 ? LAYERS : parallel->cuts[stage];
        printf("GLM53F_PP_STAGE stage=%d layers=%d:%d compute_max=%.6f receive_max=%.6f send_wait_max=%.6f\n",
            stage, first, end, stage_maximum[3 * stage],
            stage_maximum[3 * stage + 1], stage_maximum[3 * stage + 2]);
    }
    free(streams); glm53f_target_model_free_12n(model); glm53f_dist_free(&d);
    return 0;
}
#endif
