#define _GNU_SOURCE
#include "glm53f_pp_image_source.h"
#include "glm53f_pp_memory.h"
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <inttypes.h>
static void header(const char *dir,const char *suffix,const char *component,
        const glm53f_dist *d,int first,int end,uint64_t stamp) {
    char path[4096];int n=snprintf(path,sizeof(path),"%s/rank%02d.%s",dir,d->map.world_rank,suffix);assert(n>0&&n<(int)sizeof(path));
    FILE *f=fopen(path,"w");assert(f);
    assert(fprintf(f,"# GLM53F_PP_%s_V1 layout=pp3-tp4 world_rank=%d stage=%d tp_rank=%d tp_size=4 cuts=%d,%d layers=%d:%d source_metadata_fnv1a=%016" PRIx64 "\n",component,d->map.world_rank,d->map.stage,d->map.tp_rank,d->config.cuts[0],d->config.cuts[1],first,end,stamp)>0);
    assert(!fclose(f));
}
int main(int argc,char **argv) {
    assert(argc==2);
    char root[4096],dirs[8][4096],path[4096];
    assert(snprintf(root,sizeof(root),"%s/images-XXXXXX",argv[1])<(int)sizeof(root));assert(mkdtemp(root));
    const char *names[]={"core","routed","shared","dense","kda","sparse","embed","head"};
    const char *components[]={"CORE","ROUTED","SHARED","DENSE","KDA","SPARSE","EMBED","HEAD"};
    for(int i=0;i<8;i++){assert(snprintf(dirs[i],sizeof(dirs[i]),"%s/%s",root,names[i])<(int)sizeof(dirs[i]));assert(!mkdir(dirs[i],0700));}
    glm53f_pp_images images={"metadata",dirs[0],dirs[1],dirs[2],dirs[3],dirs[4],dirs[5],dirs[6],dirs[7]};
    const int cuts[][2]={{15,30},{1,2},{17,31},{7,38}};
    int cases=0;
    for(int ci=0;ci<4;ci++)for(int rank=0;rank<12;rank++){
        glm53f_dist d;memset(&d,0,sizeof(d));d.initialized=1;d.config=glm53f_parallel_default();d.config.layout=GLM53F_PP3_TP4;memcpy(d.config.cuts,cuts[ci],sizeof(d.config.cuts));
        assert(!glm53f_parallel_map_rank(&d.config,rank,12,&d.map));
        for(int i=0;i<8;i++){
            int first=i==1&&d.map.first_layer<3?3:d.map.first_layer;
            header(dirs[i],i==0?"core.manifest":"manifest",components[i],&d,first,d.map.end_layer,UINT64_C(0x123456));
            const char *suffix=i==0?"core.blob":i>=6?"f32":"blob";
            assert(snprintf(path,sizeof(path),"%s/rank%02d.%s",dirs[i],rank,suffix)<(int)sizeof(path));
            int fd=open(path,O_CREAT|O_WRONLY,0600);assert(fd>=0&& !ftruncate(fd,1u<<20)&&!close(fd));
        }
        uint64_t stamp=0;assert(!glm53f_pp_images_source(&d,&images,&stamp)&&stamp==UINT64_C(0x123456));
        glm53f_memory_budget budget;uint64_t headroom;
        assert(!glm53f_pp_inventory(&d,&images,8192,47,4096,UINT64_C(100)<<20,&budget));
        assert(!glm53f_memory_budget_fits(&budget,UINT64_C(32)<<30,&headroom)&&headroom>=(UINT64_C(6)<<30));
        assert(glm53f_pp_inventory(&d,&images,32769,47,4096,0,&budget)==-1);
        assert(glm53f_pp_inventory(&d,&images,8192,49,4096,0,&budget)==-1);
        assert(glm53f_pp_inventory(&d,&images,8192,47,2047,0,&budget)==-1);
        assert(glm53f_pp_inventory(&d,&images,8192,47,4096,UINT64_MAX,&budget)==-1);
        header(dirs[0],"core.manifest","CORE",&d,d.map.first_layer,d.map.end_layer,2);
        assert(glm53f_pp_images_source(&d,&images,&stamp)==-1);
        header(dirs[0],"core.manifest","LEGACY",&d,d.map.first_layer,d.map.end_layer,UINT64_C(0x123456));
        assert(glm53f_pp_images_source(&d,&images,&stamp)==-1);
        header(dirs[0],"core.manifest","CORE",&d,d.map.first_layer,d.map.end_layer,UINT64_C(0x123456));
        d.config.cuts[0]++;assert(glm53f_pp_images_source(&d,&images,&stamp)==-1);d.config.cuts[0]--;
        assert(snprintf(path,sizeof(path),"%s/rank%02d.core.blob",dirs[0],rank)<(int)sizeof(path));assert(!unlink(path));
        assert(glm53f_pp_inventory(&d,&images,8192,47,4096,0,&budget)==-1);
        for(int i=0;i<8;i++){
            const char *suffix=i==0?"core.manifest":"manifest";
            assert(snprintf(path,sizeof(path),"%s/rank%02d.%s",dirs[i],rank,suffix)<(int)sizeof(path));assert(!unlink(path));
            if(i){suffix=i>=6?"f32":"blob";assert(snprintf(path,sizeof(path),"%s/rank%02d.%s",dirs[i],rank,suffix)<(int)sizeof(path));assert(!unlink(path));}
        }
        ++cases;
    }
    for(int i=0;i<8;i++)assert(!rmdir(dirs[i]));
    assert(!rmdir(root));
    printf("GLM53F_PP_IMAGES_PASS cases=%d source_and_layout_rejections=144 inventory_shape_overflow_missing_rejections=240\n",cases);
    return 0;
}
