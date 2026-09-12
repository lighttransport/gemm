#include "ds41f_journal.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
    float window[40][512],pool_value[3][512],pool_score[3][512];
    uint8_t compressed[4][356],publication[356];
    int selected[512];size_t selected_count;
    uint32_t history[4],history_len;
    uint64_t counters[2][3],prefetch_generation;
} checkpoint;
struct ds41f_journal {
    ds41f_attention *attention;ds41f_engram *engram;ds41f_prefetch *prefetch;
    size_t start,inputs,recorded,candidate_bytes;int finished;
    checkpoint *points;uint8_t *candidates;
};
size_t ds41f_journal_bytes(size_t capacity,size_t inputs)
{
    if(!capacity||capacity>1048576||!inputs||inputs>6)return 0;
    return sizeof(ds41f_journal)+inputs*(sizeof(checkpoint)+(capacity+7)/8);
}
int ds41f_journal_create(ds41f_journal **out,ds41f_attention *a,ds41f_engram *e,
                         ds41f_prefetch *p,size_t start,size_t inputs,size_t budget)
{
    if(!out)return EINVAL;
    *out=NULL;
    if(!a||!e||start>=a->capacity||inputs>a->capacity-start||
       !ds41f_journal_bytes(a->capacity,inputs)||!a->window||!a->candidate_blocks)return EINVAL;
    for(int i=0;i<4;++i)if(!a->compressed[i])return EINVAL;
    if(ds41f_journal_bytes(a->capacity,inputs)>budget)return ENOMEM;
    ds41f_journal *j=calloc(1,sizeof *j);if(!j)return ENOMEM;
    j->attention=a;j->engram=e;j->prefetch=p;j->start=start;j->inputs=inputs;
    j->candidate_bytes=(a->capacity+7)/8;
    j->points=calloc(inputs,sizeof(checkpoint));j->candidates=malloc(inputs*j->candidate_bytes);
    if(!j->points||!j->candidates){ds41f_journal_free(j);return ENOMEM;}
    *out=j;return 0;
}
int ds41f_journal_record(ds41f_journal *j,size_t pos)
{
    if(!j||j->finished||j->recorded==j->inputs||pos!=j->start+j->recorded)return EINVAL;
    if(j->prefetch){int rc=ds41f_prefetch_drain(j->prefetch);if(rc)return rc;}
    ds41f_attention *a=j->attention;ds41f_engram *e=j->engram;checkpoint *c=j->points+j->recorded;
    for(int l=0;l<40;++l)memcpy(c->window[l],a->window+((size_t)l*128+pos%128)*512,512*sizeof(float));
    for(int i=0;i<4;++i)memcpy(c->compressed[i],a->compressed[i]+(i==3?pos:pos/2)*356,356);
    memcpy(c->pool_value,a->pool_value,sizeof c->pool_value);memcpy(c->pool_score,a->pool_score,sizeof c->pool_score);
    memcpy(c->publication,a->publication,sizeof c->publication);memcpy(c->selected,a->selected,sizeof c->selected);
    c->selected_count=a->selected_count;
    memcpy(j->candidates+j->recorded*j->candidate_bytes,a->candidate_blocks,j->candidate_bytes);
    memcpy(c->history,e->history,sizeof c->history);c->history_len=e->history_len;
    for(int i=0;i<2;++i){c->counters[i][0]=e->table[i].lookups;c->counters[i][1]=e->table[i].local_rows;c->counters[i][2]=e->table[i].remote_rows;}
    c->prefetch_generation=ds41f_prefetch_generation(j->prefetch);++j->recorded;return 0;
}
int ds41f_journal_finish(ds41f_journal *j,size_t keep)
{
    if(!j||j->finished||keep>j->recorded)return EINVAL;
    if(j->prefetch){int rc=ds41f_prefetch_drain(j->prefetch);if(rc)return rc;}
    ds41f_attention *a=j->attention;ds41f_engram *e=j->engram;
    for(size_t i=j->recorded;i>keep;){--i;size_t pos=j->start+i;checkpoint *c=j->points+i;
        for(int l=0;l<40;++l)memcpy(a->window+((size_t)l*128+pos%128)*512,c->window[l],512*sizeof(float));
        for(int src=0;src<4;++src)memcpy(a->compressed[src]+(src==3?pos:pos/2)*356,c->compressed[src],356);
    }
    if(keep<j->recorded){checkpoint *c=j->points+keep;
        memcpy(a->pool_value,c->pool_value,sizeof c->pool_value);memcpy(a->pool_score,c->pool_score,sizeof c->pool_score);
        memcpy(a->publication,c->publication,sizeof c->publication);memcpy(a->selected,c->selected,sizeof c->selected);
        a->selected_count=c->selected_count;memcpy(a->candidate_blocks,j->candidates+keep*j->candidate_bytes,j->candidate_bytes);
        memcpy(e->history,c->history,sizeof c->history);e->history_len=c->history_len;
        for(int i=0;i<2;++i){e->table[i].lookups=c->counters[i][0];e->table[i].local_rows=c->counters[i][1];e->table[i].remote_rows=c->counters[i][2];}
    }
    /* Prefetch generations remain monotonic. Both saved results were drained
     * and invalidated; the next forward must submit fresh hashes. */
    j->finished=1;return 0;
}
void ds41f_journal_free(ds41f_journal *j)
{if(j){free(j->points);free(j->candidates);free(j);}}
