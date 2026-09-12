#ifndef DS41F_TEAM_H
#define DS41F_TEAM_H
#include <stddef.h>
/* The calling thread remains OpenMP thread zero and owns all MPI calls.
 * The body may dispatch bounded work; worker callbacks must not dispatch. */
typedef void (*ds41f_team_body)(void *);
typedef void (*ds41f_team_work)(void *,size_t,size_t);
#if defined(_OPENMP)
int ds41f_team_active(void);
int ds41f_team_thread_id(void);
int ds41f_team_run(ds41f_team_body body,void *context);
int ds41f_team_for(size_t count,ds41f_team_work work,void *context);
#else
static inline int ds41f_team_active(void){return 0;}
static inline int ds41f_team_thread_id(void){return 0;}
static inline int ds41f_team_run(ds41f_team_body body,void *context){body(context);return 0;}
static inline int ds41f_team_for(size_t n,ds41f_team_work work,void *context){if(n)work(context,0,n);return 0;}
#endif
#endif
