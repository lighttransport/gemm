#include "ds41f_journal.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"JOURNAL FAIL line=%d\n",__LINE__);return 1;}}while(0)
static void mutate(ds41f_attention *a,ds41f_engram *e,size_t pos)
{
    for(int l=0;l<40;++l)for(int k=0;k<512;++k)
        a->window[((size_t)l*128+pos%128)*512+k]=(float)(pos*20480+l*512+k);
    for(int src=0;src<4;++src){
        if(src==3||pos%2)memset(a->compressed[src]+(src==3?pos:pos/2)*356,(int)(pos+src+1)%256,356);
        else for(int k=0;k<512;++k){a->pool_value[src][k]=(float)(pos+k);a->pool_score[src][k]=(float)(pos-k);}}
    memset(a->publication,(int)(pos+1)%256,356);a->selected_count=(pos+1)%513;
    for(int i=0;i<512;++i)a->selected[i]=(int)(pos+i);
    memset(a->candidate_blocks,(int)(pos+1)%256,(a->capacity+7)/8);
    e->history_len=3;for(int i=0;i<4;++i)e->history[i]=(uint32_t)(pos+i);
    for(int i=0;i<2;++i){e->table[i].lookups+=pos+1;e->table[i].local_rows+=pos+2;e->table[i].remote_rows+=pos+3;}
}
static int equal(const ds41f_attention *a,const ds41f_attention *b,const ds41f_engram *x,const ds41f_engram *y)
{
    if(memcmp(a->window,b->window,(size_t)40*128*512*sizeof(float))||
       memcmp(a->pool_value,b->pool_value,sizeof a->pool_value)||memcmp(a->pool_score,b->pool_score,sizeof a->pool_score)||
       memcmp(a->publication,b->publication,356)||memcmp(a->selected,b->selected,sizeof a->selected)||
       a->selected_count!=b->selected_count||memcmp(a->candidate_blocks,b->candidate_blocks,(a->capacity+7)/8)||
       memcmp(x,y,sizeof *x))return 0;
    for(int i=0;i<4;++i)if(memcmp(a->compressed[i],b->compressed[i],(i==3?a->capacity:(a->capacity+1)/2)*356))return 0;
    return 1;
}
int main(void)
{
    const size_t starts[]={0,1,7,8,126,127,128,509,511,512,1022};size_t cases=0;
    ds41f_attention a,b;ds41f_engram *e=calloc(1,sizeof *e),*ref=calloc(1,sizeof *ref);REQUIRE(e&&ref);
    REQUIRE(!ds41f_attention_init(&a,1028)&&!ds41f_attention_init(&b,1028));
    for(size_t k=0;k<sizeof starts/sizeof *starts;++k)for(size_t n=1;n<=6;++n)for(size_t keep=0;keep<=n;++keep){
        size_t start=starts[k];ds41f_journal *j=NULL;
        /* Both states already contain identical unrelated bytes from previous
         * trials, so rollback must preserve more than an initially zero cache. */
        REQUIRE(equal(&a,&b,e,ref));
        REQUIRE(ds41f_journal_create(&j,&a,e,NULL,start,n,ds41f_journal_bytes(1028,n)-1)==ENOMEM&&!j);
        REQUIRE(!ds41f_journal_create(&j,&a,e,NULL,start,n,ds41f_journal_bytes(1028,n)));
        REQUIRE(ds41f_journal_record(j,start+1)==EINVAL);
        for(size_t i=0;i<n;++i){REQUIRE(!ds41f_journal_record(j,start+i));mutate(&a,e,start+i);}
        REQUIRE(ds41f_journal_record(j,start+n)==EINVAL&&ds41f_journal_finish(j,n+1)==EINVAL);
        REQUIRE(!ds41f_journal_finish(j,keep)&&ds41f_journal_finish(j,keep)==EINVAL);
        for(size_t i=0;i<keep;++i)mutate(&b,ref,start+i);
        REQUIRE(equal(&a,&b,e,ref));ds41f_journal_free(j);++cases;
    }
    ds41f_journal *j=NULL;REQUIRE(ds41f_journal_create(&j,&a,e,NULL,1027,2,10000000)==EINVAL);
    REQUIRE(!ds41f_journal_bytes(1048577,6)&&!ds41f_journal_bytes(1048576,7));
    printf("JOURNAL PASS cases=%zu reject_every_prefix wrap compression topk candidates pools Engram budget max_1M_bytes=%zu\n",cases,ds41f_journal_bytes(1048576,6));
    ds41f_attention_free(&a);ds41f_attention_free(&b);free(e);free(ref);return 0;
}
