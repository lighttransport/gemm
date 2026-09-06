/* Dependency-free GLM5.3 tokenizer.json BPE reader.
 *
 * This is deliberately a small, streaming-oriented implementation for the
 * prompt path: it reads the vocab and merges from tokenizer.json, keeps no
 * model weights, and exposes UTF-8 text -> token IDs. The pre-tokenizer uses
 * the GPT-2 byte-level boundary rules needed by GLM5.3's tokenizer.json.
 */
#ifndef GLM5_BPE_H
#define GLM5_BPE_H

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char *text; int id; } glm5_bpe_vocab_item;
typedef struct { char *left; char *right; int rank; } glm5_bpe_merge;
typedef struct {
    glm5_bpe_vocab_item *vocab; size_t n_vocab;
    glm5_bpe_merge *merges; size_t n_merges;
    int gmask, sop, user, assistant, system, observation;
    int im_start, im_end, think, end_think, endoftext;
} glm5_bpe;

static char *glm5_bpe_strdup(const char *s) { size_t n=strlen(s)+1; char *p=(char*)malloc(n); if(p)memcpy(p,s,n); return p; }
static char *glm5_bpe_strndup(const char *s,size_t n) { char *p=(char*)malloc(n+1); if(p){memcpy(p,s,n);p[n]='\0';} return p; }

static void glm5_bpe_free(glm5_bpe *b) {
    if (!b) return;
    for (size_t i=0;i<b->n_vocab;i++) free(b->vocab[i].text);
    for (size_t i=0;i<b->n_merges;i++) { free(b->merges[i].left); free(b->merges[i].right); }
    free(b->vocab); free(b->merges); memset(b,0,sizeof(*b));
}

static int glm5_bpe_hex(int c) { return c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:-1; }
static size_t glm5_bpe_json_string(const char **pp, char *out, size_t cap) {
    const char *p=*pp; size_t n=0; int h;
    if (*p++!='"') return 0;
    while (*p && *p!='"') {
        unsigned cp=0; unsigned char c=(unsigned char)*p++;
        if (c!='\\') { if(n+1<cap) out[n++]=(char)c; continue; }
        c=(unsigned char)*p++;
        if(c=='n') c='\n'; else if(c=='r') c='\r'; else if(c=='t') c='\t';
        else if(c=='b') c='\b'; else if(c=='f') c='\f';
        else if(c=='u') { for(int i=0;i<4;i++){h=glm5_bpe_hex((unsigned char)*p++);if(h<0)return 0;cp=(cp<<4)|(unsigned)h;}
            if(cp<0x80){if(n+1<cap)out[n++]=(char)cp;} else if(cp<0x800){if(n+2<cap){out[n++]=(char)(0xc0|(cp>>6));out[n++]=(char)(0x80|(cp&63));}}
            else if(n+3<cap){out[n++]=(char)(0xe0|(cp>>12));out[n++]=(char)(0x80|((cp>>6)&63));out[n++]=(char)(0x80|(cp&63));}
            continue;
        }
        else if(c=='"'||c=='\\'||c=='/'){} else return 0;
        if(n+1<cap)out[n++]=(char)c;
    }
    if(*p!='"'||n>=cap)return 0;
    out[n]='\0'; *pp=p+1; return n;
}
static void glm5_bpe_skip(const char **p) { while(isspace((unsigned char)**p))(*p)++; }
static int glm5_bpe_find(const glm5_bpe *b,const char *s) { for(size_t i=0;i<b->n_vocab;i++)if(!strcmp(b->vocab[i].text,s))return b->vocab[i].id; return -1; }
static int glm5_bpe_merge_rank(const glm5_bpe *b,const char *a,const char *z) { for(size_t i=0;i<b->n_merges;i++)if(!strcmp(b->merges[i].left,a)&&!strcmp(b->merges[i].right,z))return b->merges[i].rank; return -1; }
static unsigned glm5_bpe_gpt2_cp(unsigned char c) {
    static int initialized;
    static unsigned map[256];
    if (!initialized) {
        unsigned next = 256;
        for (unsigned i = 0; i < 256; ++i)
            if ((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174)
                map[i] = i;
            else
                map[i] = next++;
        initialized = 1;
    }
    return map[c];
}
static size_t glm5_bpe_put_utf8(char *out, size_t pos, size_t cap, unsigned cp) {
    if (cp < 0x80) {
        if (pos + 1 >= cap) return 0;
        out[pos++] = (char)cp;
    } else if (cp < 0x800) {
        if (pos + 2 >= cap) return 0;
        out[pos++] = (char)(0xc0 | (cp >> 6));
        out[pos++] = (char)(0x80 | (cp & 63));
    } else {
        if (pos + 3 >= cap) return 0;
        out[pos++] = (char)(0xe0 | (cp >> 12));
        out[pos++] = (char)(0x80 | ((cp >> 6) & 63));
        out[pos++] = (char)(0x80 | (cp & 63));
    }
    return pos;
}
static size_t glm5_bpe_utf8_len(unsigned char c) {
    return c < 0x80 ? 1 : (c < 0xe0 ? 2 : (c < 0xf0 ? 3 : 4));
}

static int glm5_bpe_decode_token(const glm5_bpe *b, int id, char *out, size_t cap) {
    const char *text = NULL; size_t pos = 0;
    if (!b || !out || cap == 0) return -1;
    for (size_t i = 0; i < b->n_vocab; ++i) if (b->vocab[i].id == id) { text = b->vocab[i].text; break; }
    if (!text) {
        const char *special = id==b->im_start?"<|im_start|>":id==b->im_end?"<|im_end|>":
                              id==b->think?"<think>":id==b->end_think?"</think>":
                              id==b->endoftext?"<|endoftext|>":NULL;
        if (!special || strlen(special) + 1 > cap) return -1;
        strcpy(out, special); return (int)strlen(out);
    }
    for (size_t i = 0; text[i];) {
        unsigned char c = (unsigned char)text[i]; unsigned cp; size_t n = glm5_bpe_utf8_len(c);
        if (n == 1) cp = c;
        else if (n == 2) cp = ((unsigned)(c&31)<<6) | ((unsigned char)text[i+1]&63);
        else if (n == 3) cp = ((unsigned)(c&15)<<12) | (((unsigned char)text[i+1]&63)<<6) | ((unsigned char)text[i+2]&63);
        else cp = ((unsigned)(c&7)<<18) | (((unsigned char)text[i+1]&63)<<12) | (((unsigned char)text[i+2]&63)<<6) | ((unsigned char)text[i+3]&63);
        int byte = -1; for (int j = 0; j < 256; ++j) if (glm5_bpe_gpt2_cp((unsigned char)j) == cp) { byte = j; break; }
        if (byte < 0 || pos + 1 >= cap) return -1;
        out[pos++] = (char)byte; i += n;
    }
    out[pos] = '\0'; return (int)pos;
}

/* Load only model.vocab and model.merges; tokenizer.json itself is bounded by
 * max_bytes to prevent accidentally feeding a model or unrelated JSON file. */
static int glm5_bpe_load(const char *path, glm5_bpe *b) {
    FILE *f=fopen(path,"rb"); long sz; char *json; const char *p,*q; char key[1024], val[1024];
    if(!f)return -1;
    fseek(f,0,SEEK_END);sz=ftell(f);fseek(f,0,SEEK_SET);
    if(sz<1||sz>128L*1024*1024){fclose(f);return -2;} json=(char*)malloc((size_t)sz+1); if(!json){fclose(f);return -3;}
    if(fread(json,1,(size_t)sz,f)!=(size_t)sz){free(json);fclose(f);return -4;}fclose(f);json[sz]='\0';memset(b,0,sizeof(*b));b->im_start=b->im_end=b->think=b->end_think=b->endoftext=-1;
    p=strstr(json,"\"vocab\""); if(!p){free(json);return -5;} p=strchr(p,'{'); if(!p){free(json);return -5;} p++;
    while(*p&&*p!='}') {glm5_bpe_skip(&p); if(*p=='}')break; if(glm5_bpe_json_string(&p,key,sizeof(key))==0){glm5_bpe_free(b);free(json);return -6;} glm5_bpe_skip(&p);if(*p++!=':'){glm5_bpe_free(b);free(json);return -6;}glm5_bpe_skip(&p);int id=(int)strtol(p,(char**)&q,10);p=q;
        glm5_bpe_vocab_item *nv=(glm5_bpe_vocab_item*)realloc(b->vocab,(b->n_vocab+1)*sizeof(*nv));if(!nv){glm5_bpe_free(b);free(json);return -3;}b->vocab=nv;b->vocab[b->n_vocab].text=glm5_bpe_strdup(key);b->vocab[b->n_vocab++].id=id;glm5_bpe_skip(&p);if(*p==',')p++;}
    p=strstr(json,"\"merges\""); if(!p){glm5_bpe_free(b);free(json);return -7;}p=strchr(p,'[');if(!p){glm5_bpe_free(b);free(json);return -7;}p++;
    while(*p&&*p!=']'){
        glm5_bpe_skip(&p); if(*p==']')break;
        char left[1024],right[1024];
        if(*p=='['){ p++; glm5_bpe_skip(&p); if(!glm5_bpe_json_string(&p,left,sizeof(left))){glm5_bpe_free(b);free(json);return -8;} glm5_bpe_skip(&p);if(*p++!=','){glm5_bpe_free(b);free(json);return -8;}glm5_bpe_skip(&p);if(!glm5_bpe_json_string(&p,right,sizeof(right))){glm5_bpe_free(b);free(json);return -8;}while(*p&&*p!=']')p++;if(*p==']')p++; }
        else { if(!glm5_bpe_json_string(&p,val,sizeof(val))){glm5_bpe_free(b);free(json);return -8;} q=strchr(val,' ');if(!q){glm5_bpe_free(b);free(json);return -8;}size_t nl=(size_t)(q-val);memcpy(left,val,nl);left[nl]='\0';strcpy(right,q+1); }
        glm5_bpe_merge *nm=(glm5_bpe_merge*)realloc(b->merges,(b->n_merges+1)*sizeof(*nm));if(!nm){glm5_bpe_free(b);free(json);return -3;}b->merges=nm;b->merges[b->n_merges].left=glm5_bpe_strdup(left);b->merges[b->n_merges].right=glm5_bpe_strdup(right);b->merges[b->n_merges].rank=(int)b->n_merges;b->n_merges++;glm5_bpe_skip(&p);if(*p==',')p++;
    }
    b->gmask=glm5_bpe_find(b,"[gMASK]");b->sop=glm5_bpe_find(b,"<sop>");b->user=glm5_bpe_find(b,"<|user|>");b->assistant=glm5_bpe_find(b,"<|assistant|>");b->system=glm5_bpe_find(b,"<|system|>");b->observation=glm5_bpe_find(b,"<|observation|>");
    b->im_start = glm5_bpe_find(b, "<|im_start|>");
    b->im_end = glm5_bpe_find(b, "<|im_end|>");
    b->think = glm5_bpe_find(b, "<think>");
    b->end_think = glm5_bpe_find(b, "</think>");
    b->endoftext = glm5_bpe_find(b, "<|endoftext|>");
    /* GLM5 special tokens are tokenizer.json added_tokens, not model.vocab. */
    if(b->gmask<0) b->gmask=154822;
    if(b->sop<0) b->sop=154824;
    if(b->user<0) b->user=154827;
    if(b->assistant<0) b->assistant=154828;
    if(b->system<0) b->system=154826;
    if(b->observation<0) b->observation=154829;
    free(json);return 0;
}

/* Encode the template text. This uses byte-level BPE for ordinary segments
 * and recognizes GLM5 role/special tokens as atomic vocab entries. It is
 * intentionally conservative for Unicode pre-tokenization; unknown pieces
 * are emitted as their individual byte-level vocab symbols. */
static int glm5_bpe_encode(const glm5_bpe *b,const char *text,int *out,int cap) {
    int n=0; const char *p=text; char piece[4096], bu[8192];
    while(*p){
        const char *specials[]={"[gMASK]","<sop>","<|system|>","<|user|>","<|assistant|>","<|observation|>","<|im_start|>","<|im_end|>","<think>","</think>","<|endoftext|>",NULL}; int sid[]={b->gmask,b->sop,b->system,b->user,b->assistant,b->observation,b->im_start,b->im_end,b->think,b->end_think,b->endoftext};int found=-1;
        for(int s=0;specials[s];s++){size_t l=strlen(specials[s]);if(!strncmp(p,specials[s],l)){found=s;break;}}
        if(found>=0){if(n>=cap)return -1;out[n++]=sid[found];p+=strlen(specials[found]);continue;}
        /* GPT-2/Qwen regex pieces carry one leading space with the following
         * word.  Consuming that space at the end of the previous piece emits
         * a standalone Ġ token and prevents common "Ġword" merges. */
        size_t m=0; while(p[m]&&m+1<sizeof(piece)&&!strchr("[]<>",p[m])) {
            if (m > 0 && isspace((unsigned char)p[m])) break;
            m++;
        }
        if(!m){m=1;} if(m>=sizeof(piece))return -1;memcpy(piece,p,m);piece[m]='\0';p+=m;
        size_t u=0;for(size_t i=0;i<m;i++) { /* GPT-2 byte-to-unicode map */
            size_t next = glm5_bpe_put_utf8(bu, u, sizeof(bu),
                                             glm5_bpe_gpt2_cp((unsigned char)piece[i]));
            if (!next) return -1;
            u = next;
        }
        bu[u]='\0';
        char *parts[512];int np=0;for(size_t i=0;i<u&&np<512;) {
            size_t l=glm5_bpe_utf8_len((unsigned char)bu[i]);
            if (i + l > u) return -1;
            parts[np]=glm5_bpe_strndup(bu+i,l);if(!parts[np++])return -1;i+=l;
        }if(np==512&&u>0)return -1;
        /* BPE pair merges; strings are copied so merge lifetime is local. */
        int changed;do{changed=0;int best=-1,br=0;for(int i=0;i+1<np;i++){int r=glm5_bpe_merge_rank(b,parts[i],parts[i+1]);if(r>=0&&(best<0||r<br)){best=i;br=r;}}if(best>=0){size_t l=strlen(parts[best])+strlen(parts[best+1]);char *z=(char*)malloc(l+1);if(!z){for(int i=0;i<np;i++)free(parts[i]);return -1;}strcpy(z,parts[best]);strcat(z,parts[best+1]);free(parts[best]);free(parts[best+1]);parts[best]=z;memmove(parts+best+1,parts+best+2,(size_t)(np-best-2)*sizeof(*parts));np--;changed=1;}}while(changed);
        for(int i=0;i<np;i++){int id=glm5_bpe_find(b,parts[i]);if(id<0){for(size_t k=0;k<strlen(parts[i]);k++){char one[2]={parts[i][k],0};id=glm5_bpe_find(b,one);if(n>=cap){free(parts[i]);return -1;}out[n++]=id;}}else{if(n>=cap){free(parts[i]);return -1;}out[n++]=id;}free(parts[i]);}
    }return n;
}
#endif
