#include "ds41f_pipeline_runtime.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
typedef struct { int stage; size_t calls, tiles[8], positions[8], counts[8], slots[8]; } trace;
typedef struct { size_t events[64], count; } transport_trace;
static int event(transport_trace *t,size_t code){assert(t->count<64);t->events[t->count++]=code;return 0;}
static int post_receive(void *x,const ds41f_pipeline *p,size_t tile,size_t pos,size_t count,int slot){(void)p;(void)pos;(void)count;return event(x,100+tile*2+slot);}
static int wait_receive(void *x,const ds41f_pipeline *p,size_t tile,int slot){(void)p;return event(x,200+tile*2+slot);}
static int post_send(void *x,const ds41f_pipeline *p,size_t tile,size_t pos,size_t count,int slot){(void)p;(void)pos;(void)count;return event(x,300+tile*2+slot);}
static int wait_send(void *x,const ds41f_pipeline *p,size_t tile,int slot){(void)p;return event(x,400+tile*2+slot);}
static int callback(void *opaque, const ds41f_pipeline *p, size_t tile, size_t position, size_t count, int slot)
{
    trace *t=opaque; assert(t->stage==p->stage); assert(t->calls<8);
    t->tiles[t->calls]=tile; t->positions[t->calls]=position; t->counts[t->calls]=count; t->slots[t->calls++]=(size_t)slot; return 0;
}
int main(void)
{
    ds41f_pipeline invalid={1,0,0,0,12,12,4,0};
    assert(ds41f_pipeline_run(1,DS41F_PIPELINE_TILE_MAX+1,1,&invalid,callback,NULL)==EINVAL);
    assert(ds41f_pipeline_run(SIZE_MAX/36+1,36,1,&invalid,callback,NULL)==EOVERFLOW);
    for(int stage=0;stage<3;++stage){ trace t; memset(&t,0,sizeof t); t.stage=stage;
        ds41f_pipeline p={1,stage*4,stage,stage==0?0:stage==1?13:27,stage==0?12:stage==1?26:39,12,4,0};
        assert(!ds41f_pipeline_run(4,36,110,&p,callback,&t)); assert(t.calls==4);
        for(size_t i=0;i<4;++i){assert(t.tiles[i]==i);assert(t.positions[i]==i*36);assert(t.counts[i]==(i==3?2:36));assert(t.slots[i]==(i&1));}
    }
    trace t;memset(&t,0,sizeof t);t.stage=0;transport_trace tr;memset(&tr,0,sizeof tr);
    ds41f_pipeline p={1,0,0,0,12,12,4,0};
    ds41f_pipeline_transport transport={post_receive,wait_receive,post_send,wait_send,&tr};
    assert(!ds41f_pipeline_run_ex(4,36,110,&p,callback,&t,&transport));
    assert(tr.count==8);assert(tr.events[0]==300&&tr.events[1]==303&&tr.events[2]==400&&tr.events[3]==304&&tr.events[4]==403&&tr.events[5]==307&&tr.events[6]==407&&tr.events[7]==404);
    memset(&t,0,sizeof t); t.stage=1; memset(&tr,0,sizeof tr); p.stage=1; p.rank=4; p.first_layer=13; p.last_layer=26;
    assert(!ds41f_pipeline_run_ex(4,36,110,&p,callback,&t,&transport));
    /* Each receive is posted one tile ahead and reused only after its wait. */
    assert(tr.events[0]==100 && tr.events[1]==200 && tr.events[2]==300 && tr.events[3]==103);
    size_t receives=0; for(size_t i=0;i<tr.count;++i) if(tr.events[i]>=100&&tr.events[i]<200) ++receives; assert(receives==4);
    puts("PIPELINE_RUNTIME PASS wavefront fill drain tile bounds slots"); return 0;
}
