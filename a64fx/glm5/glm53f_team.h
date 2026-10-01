#ifndef GLM53F_TEAM_H
#define GLM53F_TEAM_H
/* Optional persistent OpenMP executor. The controller runs on thread zero;
 * callbacks run once on every thread and may use orphaned omp-for/barrier.
 * Callback arguments remain owned by the controller until dispatch returns.
 * Weak entry points keep standalone kernel builds on their legacy executor. */
typedef void (*glm53f_team_callback)(void *);
extern int glm53f_team_active(void) __attribute__((weak));
extern void glm53f_team_dispatch(glm53f_team_callback fn, void *context) __attribute__((weak));
extern void glm53f_team_run(glm53f_team_callback controller, void *context) __attribute__((weak));
static inline int glm53f_team_available(void) {
    return glm53f_team_active && glm53f_team_active();
}
#endif
