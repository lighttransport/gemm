#include "ds41f_mtp.h"
#include "ds41f_comm.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv)
{
    int rank,ranks,rc=ds41f_comm_init(&argc,&argv,&rank,&ranks);if(rc)return rc;
    if(argc!=2||ranks!=12)ds41f_comm_abort("usage: test_mtp_weights ROOT",EINVAL);
    ds41f_comm_set_tp(4);char stage[4096];snprintf(stage,sizeof stage,"%s/rank%d",argv[1],rank);
    ds41f_weights backbone={0};ds41f_mtp mtp;
    rc=ds41f_mtp_load(&mtp,&backbone,stage,rank,(size_t)1024*1024*1024,1,1,1);
    if(rc)ds41f_comm_abort("MTP weight load",rc);
    size_t expert_tensors=0;
    for(size_t i=0;i<mtp.weights.count;++i){const ds41f_weight *w=mtp.weights.items+i;
        if(strncmp(w->name,"mtp.",4))ds41f_comm_abort("unexpected backbone duplicate",EINVAL);
        if(strstr(w->name,".ffn.experts."))++expert_tensors;
    }
    if(expert_tensors!=(size_t)(rank<8?11:10)*3*6)ds41f_comm_abort("MTP expert ownership",EINVAL);
    printf("MTP_WEIGHTS PASS rank=%d tensors=%zu expert_tensors=%zu resident=%zu TP4 EP128 INT8 packed_experts\n",rank,mtp.weights.count,expert_tensors,mtp.weights.bytes);
    ds41f_mtp_free(&mtp);ds41f_comm_free();return 0;
}
