/* Build an owned-layer compact core directly from GGUF using the existing
 * TP12 conversion routines. No checkpoint payload or full model copy. */
#define GLM53F_PP_CORE_STAGE 1
#define main pp_core_patch_program_main
#include "glm53f_q2_core_patch.c"
#undef main
#include "glm53f_parallel.h"
#include "glm53f_pp_source.h"
#include <mpi.h>
static uint64_t core_offset;
static FILE *core_manifest;
static int core_fd;
static void core_request(char kind,const char *name,uint64_t a,uint64_t b,uint64_t c,uint64_t bytes) {
    uint64_t begin=(core_offset+255)&~UINT64_C(255);
    if(kind=='R')fprintf(core_manifest,"R %s %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",name,a,b,begin);
    else fprintf(core_manifest,"C %s %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",name,a,b,c,begin);
    core_offset=begin+bytes;
    if(ftruncate(core_fd,(off_t)core_offset))MPI_Abort(MPI_COMM_WORLD,2);
}
static void core_rows(const char *name,uint64_t offset,uint64_t bytes){core_request('R',name,offset,bytes,0,bytes);}
static void core_columns(const char *name,uint64_t row_bytes,uint64_t col0,uint64_t columns,uint64_t rows){core_request('C',name,row_bytes,col0,columns,rows*columns);}
static void layer_plan(int layer,int head0,int heads) {
    char name[512];
#define R(S,O,B) do{snprintf(name,sizeof(name),"model.language_model.layers.%d." S,layer);core_rows(name,O,B);}while(0)
#define C(S,RB,C0,CN,NR) do{snprintf(name,sizeof(name),"model.language_model.layers.%d." S,layer);core_columns(name,RB,C0,CN,NR);}while(0)
    R("hc_attn_fn",0,24u*16384*2);R("hc_attn_base",0,24u*4);R("hc_attn_scale",0,3u*4);
    R("hc_ffn_fn",0,24u*16384*2);R("hc_ffn_base",0,24u*4);R("hc_ffn_scale",0,3u*4);
    R("input_layernorm.weight",0,4096u*2);R("post_attention_layernorm.weight",0,4096u*2);
    if(layer%4!=3){uint64_t qd=heads*128u,h0=head0*128u;
        R("self_attn.A_log",head0*4u,heads*4u);R("self_attn.dt_bias",h0*4,qd*4);
        R("self_attn.q_proj.weight",h0*4096*2,qd*4096*2);
        R("self_attn.k_proj.weight",h0*4096*2,qd*4096*2);
        R("self_attn.v_proj.weight",h0*4096*2,qd*4096*2);
        R("self_attn.q_conv1d.weight",h0*4*2,qd*4*2);
        R("self_attn.k_conv1d.weight",h0*4*2,qd*4*2);
        R("self_attn.v_conv1d.weight",h0*4*2,qd*4*2);
        R("self_attn.f_b_proj.weight",h0*128*2,qd*128*2);
        R("self_attn.b_proj.weight",head0*4096u*2,heads*4096u*2);
        R("self_attn.g_b_proj.weight",h0*128*2,qd*128*2);
        R("self_attn.f_a_proj.weight",0,128u*4096*2);
        R("self_attn.g_a_proj.weight",0,128u*4096*2);
        R("self_attn.o_norm.weight",0,128u*2);
        C("self_attn.o_proj.weight",8192u*2,h0*2,qd*2,4096);

    }else{uint64_t qd=heads*256u,h0=head0*256u;
        R("self_attn.q_a_proj.weight",0,1536u*4096);R("self_attn.q_a_proj.weight_scale_inv",0,12u*32*4);
        R("self_attn.q_a_layernorm.weight",0,1536u*2);
        R("self_attn.q_b_proj.weight",h0*1536,qd*1536);R("self_attn.q_b_proj.weight_scale_inv",(h0/128)*12*4,(qd/128)*12*4);
        R("self_attn.kv_a_proj_with_mqa.weight",0,512u*4096);R("self_attn.kv_a_proj_with_mqa.weight_scale_inv",0,4u*32*4);
        R("self_attn.kv_a_layernorm.weight",0,512u*2);
        R("self_attn.kv_b_proj.weight",head0*512u*512*2,heads*512u*512*2);
        C("self_attn.o_proj.weight",16384,h0,qd,4096);C("self_attn.o_proj.weight_scale_inv",128u*4,(h0/128)*4,(qd/128)*4,32);
        R("self_attn.indexer.wk.weight",0,128u*4096*2);
        R("self_attn.indexer.k_norm.weight",0,128u*2);R("self_attn.indexer.k_norm.bias",0,128u*2);
        R("self_attn.indexer.index_kpool_compress_gate",0,128u*4096*2);
        R("self_attn.indexer.index_kpool_compress_ape",0,4u*128*2);
        R("self_attn.indexer.wq_b.weight",0,32u*128*1536*2);R("self_attn.indexer.weights_proj.weight",0,32u*4096*2);
    }
    if(layer>=3){R("mlp.gate.weight",0,288u*4096*2);R("mlp.gate.e_score_correction_bias",0,288u*4);}
#undef C
#undef R
}
int main(int argc,char **argv) {
    (void)gguf_type_name;(void)ggml_type_name;
    int rank,size;MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    glm53f_parallel_config config=glm53f_parallel_default();config.layout=GLM53F_PP3_TP4;glm53f_parallel_map map;
    for(int i=3;i<argc;i++)if(glm53f_parallel_option(&config,argc,argv,&i)!=1)MPI_Abort(MPI_COMM_WORLD,2);
    if(argc<3||config.layout!=GLM53F_PP3_TP4||strncmp(argv[2],"/local/",7)||glm53f_parallel_map_rank(&config,rank,size,&map))MPI_Abort(MPI_COMM_WORLD,2);
    int low[2],high[2];MPI_Allreduce(config.cuts,low,2,MPI_INT,MPI_MIN,MPI_COMM_WORLD);MPI_Allreduce(config.cuts,high,2,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(memcmp(low,high,sizeof(low)))MPI_Abort(MPI_COMM_WORLD,2);
    gguf_context *g=gguf_open_multi(argv[1],3);uint64_t stamp=0;
    if(!g||g->n_tensors!=1412||glm53f_pp_source_stamp(g,argv[1],&stamp))MPI_Abort(MPI_COMM_WORLD,2);
    if(mkdir(argv[2],0755)&&errno!=EEXIST)MPI_Abort(MPI_COMM_WORLD,2);
    char blob[4096],manifest[4096],bt[4096],mt[4096],final_manifest[4096];
    snprintf(blob,sizeof(blob),"%s/rank%02d.core.blob",argv[2],rank);snprintf(manifest,sizeof(manifest),"%s/rank%02d.core.manifest",argv[2],rank);
    snprintf(bt,sizeof(bt),"%s/.rank%02d.core.blob.%ld",argv[2],rank,(long)getpid());snprintf(mt,sizeof(mt),"%s/.rank%02d.core.manifest.%ld",argv[2],rank,(long)getpid());
    core_fd=open(bt,O_CREAT|O_EXCL|O_RDWR,0644);core_manifest=fopen(mt,"wx");if(core_fd<0||!core_manifest)MPI_Abort(MPI_COMM_WORLD,2);
    fprintf(core_manifest,"# GLM53F_PP_CORE_V1 layout=pp3-tp4 world_rank=%d stage=%d tp_rank=%d tp_size=4 cuts=%d,%d layers=%d:%d source_metadata_fnv1a=%016" PRIx64 "\n",rank,map.stage,map.tp_rank,config.cuts[0],config.cuts[1],map.first_layer,map.end_layer,stamp);
    for(int layer=map.first_layer;layer<map.end_layer;layer++)layer_plan(layer,map.tp_rank*16,16);
    if(map.stage==2)core_rows("model.language_model.norm.weight",0,4096u*2);
    if(fflush(core_manifest)||fclose(core_manifest))MPI_Abort(MPI_COMM_WORLD,2);
    for(int layer=map.first_layer;layer<map.end_layer;layer++){
        if(patch_common(core_fd,mt,g,layer)||(layer%4==3?patch_sparse(core_fd,mt,g,layer):patch_kda(core_fd,mt,g,layer))||(layer>=3&&patch_router_if_present(core_fd,mt,g,layer)))MPI_Abort(MPI_COMM_WORLD,2);
    }
    if(map.stage==2){image_entry e;if(find_entry(mt,"model.language_model.norm.weight",&e))MPI_Abort(MPI_COMM_WORLD,2);
        tensor_ref t=find_tensor(g,"output_norm.weight");if(!t.info||patch_linear(core_fd,&e,&t,2,0))MPI_Abort(MPI_COMM_WORLD,2);}
    snprintf(final_manifest,sizeof(final_manifest),"%s/.rank%02d.core.final.%ld",argv[2],rank,(long)getpid());
    /* Digests are generated only after conversion has finished. */
    FILE *in=fopen(mt,"r"),*out=fopen(final_manifest,"wx");unsigned char buffer[65536];char line[2048];
    if(!in||!out)MPI_Abort(MPI_COMM_WORLD,2);
    while(fgets(line,sizeof(line),in)){fputs(line,out);char kind,name[512];unsigned long long a,b,c,off;uint64_t bytes=0;
        int n=sscanf(line,"%c %511s %llu %llu %llu %llu",&kind,name,&a,&b,&c,&off);
        if(kind=='R'&&n==5){off=c;bytes=b;}
        else if(kind=='C'&&n==6){/* All core column requests have4096 or32 rows. */
            bytes=(strstr(name,"weight_scale_inv")?32u:4096u)*c;}
        else continue;
        uint64_t hash=UINT64_C(1469598103934665603);
        for(uint64_t pos=0;pos<bytes;pos+=sizeof(buffer)){size_t count=bytes-pos;if(count>sizeof(buffer))count=sizeof(buffer);
            if(read_exact(core_fd,off+pos,buffer,count))MPI_Abort(MPI_COMM_WORLD,2);
            for(size_t j=0;j<count;j++){hash^=buffer[j];hash*=UINT64_C(1099511628211);}
            (void)posix_fadvise(core_fd,(off_t)(off+pos),count,POSIX_FADV_DONTNEED);}
        fprintf(out,"# PAYLOAD offset=%llu bytes=%" PRIu64 " fnv1a=%016" PRIx64 "\n",off,bytes,hash);
    }
    fprintf(out,"# COMPLETE bytes=%" PRIu64 "\n",core_offset);
    if(fclose(in)||fflush(out)||fsync(fileno(out))||fclose(out)||fsync(core_fd)||close(core_fd)||rename(bt,blob)||rename(final_manifest,manifest)||unlink(mt))MPI_Abort(MPI_COMM_WORLD,2);
    printf("GLM53F_PP_CORE_STAGE rank=%d layers=%d:%d bytes=%" PRIu64 " PASS\n",rank,map.first_layer,map.end_layer,core_offset);
    gguf_close(g);MPI_Finalize();return 0;
}
