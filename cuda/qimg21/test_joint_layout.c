#include <stdio.h>
#include "joint_layout.h"

int main(void) {
    int slots, text_slots, images;
    if(scanf("%d%d%d",&slots,&text_slots,&images)!=3 || slots<1 || slots>4096 || images<1 || images>32)return 2;
    int mask[4096], heights[32], widths[32];
    for(int i=0;i<slots;i++)if(scanf("%d",&mask[i])!=1)return 2;
    for(int i=0;i<images;i++)if(scanf("%d%d",&heights[i],&widths[i])!=2)return 2;
    q21_joint_layout layout;
    if(q21_layout_build(&layout,mask,slots,text_slots,heights,widths,images))return 1;
    printf("%d %d %d\n",layout.n,layout.prefix,layout.image_tokens);
    for(int i=0;i<layout.n;i++)printf("%d %d %d %d %d %d\n",layout.text_index[i],layout.image_index[i],layout.image_id[i],
        layout.position[i*3],layout.position[i*3+1],layout.position[i*3+2]);
    for(int q=0;q<layout.n;q++) {
        for(int k=0;k<layout.n;k++)printf("%d ",q21_layout_attention_allowed(&layout,q,k));
        putchar('\n');
    }
    q21_layout_free(&layout);
    return 0;
}
