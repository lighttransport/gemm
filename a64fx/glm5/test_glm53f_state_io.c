#define _POSIX_C_SOURCE 200809L
#include "glm53f_state_io.h"

int main(int argc, char **argv) {
    /* Caller supplies /local or repo tmp/; never use libc tmpfile(). */
    if (argc != 2) return 2;
    char path[4096];
    if (snprintf(path,sizeof(path),"%s/glm53f-stateio.XXXXXX",argv[1]) >= (int)sizeof(path)) return 2;
    int fd=mkstemp(path);
    if (fd<0) return 2;
    FILE *f=fdopen(fd,"w+b");
    if (!f) { close(fd); unlink(path); return 2; }
    glm53f_state_io io={0}; io.file=f;
    const float ref[]={1,-1,.5f,0};
    const int meta=37;
    int failed=glm53f_state_io_floats(&io,ref,sizeof(ref),"write") ||
               glm53f_state_io_bytes(&io,&meta,sizeof(meta),"meta") || fflush(f);
    const float near[]={1.0001f,-1,.5f,0};
    const float far[]={2,-1,.5f,0};
    const float nan[]={NAN,-1,.5f,0};
    for (int mode=1; mode<=2; ++mode)
        for (int value=0; value<4; ++value) {
            const float *data=value==0?ref:value==1?near:value==2?far:nan;
            rewind(f); io.offset=io.flushed=0; io.failed=0; io.compare=mode;
            int rc=glm53f_state_io_floats(&io,data,sizeof(ref),"float");
            int expected=value==0 || (mode==2 && value==1);
            failed |= (rc==0) != expected;
            if (!rc) failed |= glm53f_state_io_bytes(&io,&meta,sizeof(meta),"meta");
        }
    rewind(f); io.offset=io.flushed=0; io.failed=0; io.compare=2;
    failed |= glm53f_state_io_floats(&io,near,sizeof(ref),"near");
    int wrong=38;
    failed |= glm53f_state_io_bytes(&io,&wrong,sizeof(wrong),"wrong_meta") == 0;
    failed |= fclose(f) != 0;
    if (unlink(path)) failed=1;
    printf("STATE_IO typed_float_and_exact_metadata %s\n",failed?"FAIL":"PASS");
    return failed ? 1 : 0;
}
