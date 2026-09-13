#include "ds41f_attention.h"
#include "ds41f_cache.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int fail(const char *what)
{fprintf(stderr,"ROW_CACHE FAIL %s\n",what);return 1;}

int main(void)
{
    ds41f_attention state; if(ds41f_attention_init(&state,64))return fail("init");
    if(ds41f_attention_enable_row_cache(&state,2))return fail("enable");
    float source[512],expected[512];uint8_t row[356];memset(row,0,sizeof row);
    for(int i=0;i<512;++i)source[i]=sinf((float)i*.03125f);
    if(ds41f_fp4_pack(row,source,512,16,1))return fail("pack");
    if(ds41f_fp4_unpack(expected,row,512,16,1))return fail("unpack");
    /* Layer 2 publishes every second position; pos=1 is compressed row 0. */
    if(ds41f_attention_receive(&state,2,1,row))return fail("receive");
    uint64_t key=((uint64_t)0<<32)|0;
    if(state.decoded_keys[0]!=key)return fail("key");
    if(memcmp(state.decoded_rows,expected,sizeof expected))return fail("decoded");
    ds41f_attention_clear_row_cache(&state);
    if(state.decoded_keys[0]!=UINT64_MAX||state.decoded_keys[1]!=UINT64_MAX)return fail("clear");
    ds41f_attention_free(&state);
    puts("ROW_CACHE PASS bounded_decode_clear");return 0;
}
