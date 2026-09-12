#ifndef DS41F_COMM_H
#define DS41F_COMM_H
#include <stddef.h>
int ds41f_comm_init(int *argc,char ***argv,int *rank,int *ranks);
void ds41f_comm_sum(float *values,size_t count);
void ds41f_comm_ready(void);
void ds41f_comm_broadcast(float *values,size_t count,int owner);
void ds41f_comm_abort(const char *message,int error);
void ds41f_comm_free(void);
#endif
