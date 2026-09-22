/* Batch-one Qwen-Image 2.1 text/image slot expansion and RoPE metadata.
 * Image slots expand to four latent tokens. Target image is the final block.
 * No GPU dependencies: shared by native bring-up and CPU reference tests. */
#ifndef QIMG21_JOINT_LAYOUT_H
#define QIMG21_JOINT_LAYOUT_H
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <limits.h>

typedef struct {
    int n, prefix, image_tokens;
    int *text_index, *image_index, *image_id, *position;
} q21_joint_layout;

static void q21_layout_free(q21_joint_layout *layout) {
    free(layout->text_index); free(layout->image_index);
    free(layout->image_id); free(layout->position);
    memset(layout, 0, sizeof(*layout));
}

static int q21_layout_build(q21_joint_layout *out, const int *slot_mask, int slots,
                           int text_slots, const int *heights, const int *widths, int images) {
    memset(out, 0, sizeof(*out));
    if (!slot_mask || !heights || !widths || slots < 1 || slots > 262144 ||
        text_slots < 0 || text_slots >= slots || images < 1 || images > slots) return -1;
    int image_tokens = 0, total = 0;
    for (int b = 0; b < images; b++) {
        if (heights[b] < 1 || widths[b] < 1 || heights[b] > 1024 || widths[b] > 1024 ||
            (heights[b] * widths[b]) % 4) return -1;
        if (image_tokens > 1048576 - heights[b] * widths[b]) return -1;
        image_tokens += heights[b] * widths[b];
    }
    if (slots - text_slots != heights[images-1] * widths[images-1] / 4) return -1;
    int marked = 0;
    for (int s = 0; s < slots; s++) {
        if (slot_mask[s] != 0 && slot_mask[s] != 1) return -1;
        if (s >= text_slots && !slot_mask[s]) return -1;
        total += slot_mask[s] ? 4 : 1;
        marked += slot_mask[s] ? 4 : 0;
    }
    if (marked != image_tokens) return -1;
    out->n = total; out->image_tokens = image_tokens;
    out->prefix = total - heights[images-1] * widths[images-1];
    out->text_index = malloc((size_t)total * sizeof(int));
    out->image_index = malloc((size_t)total * sizeof(int));
    out->image_id = malloc((size_t)total * sizeof(int));
    out->position = malloc((size_t)total * 3 * sizeof(int));
    if (!out->text_index || !out->image_index || !out->image_id || !out->position) goto fail;
    int t=0, b=0, offset=0, image_index=0, position=0;
    for (int s=0;s<slots;s++) {
        if (!slot_mask[s]) {
            if (offset) goto fail; /* An image's latent block must be contiguous. */
            out->text_index[t]=s; out->image_index[t]=-1; out->image_id[t]=-1;
            for(int axis=0;axis<3;axis++)out->position[t*3+axis]=position;
            position++; t++;
        } else {
            for(int j=0;j<4;j++) {
                if(b>=images)goto fail;
                out->text_index[t]=-1; out->image_index[t]=image_index++; out->image_id[t]=b;
                out->position[t*3]=position;
                out->position[t*3+1]=-(heights[b]-heights[b]/2)+offset/widths[b];
                out->position[t*3+2]=-(widths[b]-widths[b]/2)+offset%widths[b];
                offset++; t++;
                if(offset==heights[b]*widths[b]) {
                    position+=heights[b]>widths[b]?heights[b]:widths[b];
                    b++; offset=0;
                }
            }
        }
    }
    if(t!=total || b!=images || offset)goto fail;
    for(int i=out->prefix;i<total;i++)if(out->image_id[i]!=images-1)goto fail;
    return 0;
fail:
    q21_layout_free(out);
    return -1;
}

static inline int q21_layout_attention_allowed(const q21_joint_layout *layout, int query, int key) {
    if(query<0 || key<0 || query>=layout->n || key>=layout->n)return 0;
    return query>=key || (layout->image_id[query]>=0 && layout->image_id[query]==layout->image_id[key]);
}

static int q21_layout_integer(FILE *fp, int *value) {
    char token[64], *end;
    if(fscanf(fp,"%63s",token)!=1)return -1;
    errno=0;
    long number=strtol(token,&end,10);
    if(errno || *end || number<INT_MIN || number>INT_MAX)return -1;
    *value=(int)number;
    return 0;
}

/* Wire format: slots text_slots images; slot mask; image (height,width)
 * pairs. Frame count is implicitly one; final image is the target. */
static int q21_layout_read(const char *path, q21_joint_layout *out,
                          int *text_slots, int *target_height, int *target_width) {
    memset(out,0,sizeof(*out));
    FILE *fp=fopen(path,"r");
    if(!fp)return -1;
    int slots,nt,images,rc=-1;
    int *mask=NULL,*height=NULL,*width=NULL;
    if(q21_layout_integer(fp,&slots) || q21_layout_integer(fp,&nt) || q21_layout_integer(fp,&images) ||
       slots<1 || slots>262144 || nt<0 || nt>=slots || images<1 || images>slots)goto done;
    mask=malloc((size_t)slots*sizeof(int));
    height=malloc((size_t)images*sizeof(int));width=malloc((size_t)images*sizeof(int));
    if(!mask || !height || !width)goto done;
    for(int i=0;i<slots;i++)if(q21_layout_integer(fp,&mask[i]))goto done;
    for(int i=0;i<images;i++)if(q21_layout_integer(fp,&height[i]) || q21_layout_integer(fp,&width[i]))goto done;
    char extra[2];
    if(fscanf(fp,"%1s",extra)!=EOF || ferror(fp))goto done;
    if(q21_layout_build(out,mask,slots,nt,height,width,images))goto done;
    *text_slots=nt;*target_height=height[images-1];*target_width=width[images-1];
    rc=0;
done:
    fclose(fp);free(mask);free(height);free(width);
    return rc;
}
#endif
