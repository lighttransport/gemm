#ifndef GLM53F_IQ_TILE_H
#define GLM53F_IQ_TILE_H
/* Four output rows, one/two tokens. Integer lane sums and FP accumulation
 * preserve iq_rows_multi's order; activation loads and sums are shared. */
static void iq_rows_tile4(float out[4][4],int type,const uint8_t *row,size_t rb,
        const glm5_iq_q8_block **x,int tokens,int blocks){
    if(tokens>2||(type!=GLM53F_GGML_Q4_K&&type!=GLM53F_GGML_Q5_K)){
        for(int r=0;r<4;r++)iq_rows_multi(out[r],type,row+(size_t)r*rb,x,tokens,blocks);
        return;
    }
    svbool_t p8=svptrue_b8(),p32=svptrue_b32(),pb=svwhilelt_b8(0,32),lo=svwhilelt_b32(0,8);
    svint8_t ones=svdup_s8(1);
    svfloat32_t a00=svdup_f32(0);
    svfloat32_t a01=svdup_f32(0);
    svfloat32_t a10=svdup_f32(0);
    svfloat32_t a11=svdup_f32(0);
    svfloat32_t a20=svdup_f32(0);
    svfloat32_t a21=svdup_f32(0);
    svfloat32_t a30=svdup_f32(0);
    svfloat32_t a31=svdup_f32(0);
    for(int b=0;b<blocks;b++){
        const uint8_t *qs[4],*scales[4],*qh[4];float d[4],dm[4];
        for(int r=0;r<4;r++){
            if(type==GLM53F_GGML_Q4_K){
                const block_q4_K *w=(const block_q4_K*)(row+(size_t)r*rb)+b;
                qs[r]=w->qs;scales[r]=w->scales;qh[r]=NULL;
                d[r]=ggml_fp16_to_fp32(w->d);dm[r]=ggml_fp16_to_fp32(w->dmin);
            }else{
                const block_q5_K *w=(const block_q5_K*)(row+(size_t)r*rb)+b;
                qs[r]=w->qs;scales[r]=w->scales;qh[r]=w->qh;
                d[r]=ggml_fp16_to_fp32(w->d);dm[r]=ggml_fp16_to_fp32(w->dmin);
            }
        }
        svint32_t d00=svdup_s32(0),m00=svdup_s32(0);
        svint32_t d01=svdup_s32(0),m01=svdup_s32(0);
        svint32_t d10=svdup_s32(0),m10=svdup_s32(0);
        svint32_t d11=svdup_s32(0),m11=svdup_s32(0);
        svint32_t d20=svdup_s32(0),m20=svdup_s32(0);
        svint32_t d21=svdup_s32(0),m21=svdup_s32(0);
        svint32_t d30=svdup_s32(0),m30=svdup_s32(0);
        svint32_t d31=svdup_s32(0),m31=svdup_s32(0);
        for(int g=0;g<256;g+=64){
            svint8_t x0=svld1_s8(p8,x[0][b].q+g);
            svint8_t x1=tokens>1?svld1_s8(p8,x[1][b].q+g):svdup_s8(0);
            svint32_t sx0=svdot_s32(svdup_s32(0),ones,x0);
            svint32_t sx1=svdot_s32(svdup_s32(0),ones,x1);
            {
                uint8_t sc0,mn0,sc1,mn1;
                get_scale_min_k4(g/32,scales[0],&sc0,&mn0);
                get_scale_min_k4(g/32+1,scales[0],&sc1,&mn1);
                svuint8_t packed=svld1_u8(pb,qs[0]+g/2);
                svuint8_t q0=svand_n_u8_x(p8,packed,15),q1=svlsr_n_u8_x(p8,packed,4);
                if(qh[0]){
                    svuint8_t hi=svld1_u8(pb,qh[0]);
                    q0=svadd_u8_x(p8,q0,svsel_u8(svcmpne_n_u8(pb,svand_n_u8_x(p8,hi,1u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
                    q1=svadd_u8_x(p8,q1,svsel_u8(svcmpne_n_u8(pb,svand_n_u8_x(p8,hi,2u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
                }
                svint8_t q=svreinterpret_s8_u8(svsplice_u8(pb,q0,q1));
                svint32_t sc=svsel_s32(lo,svdup_s32(sc0),svdup_s32(sc1));
                svint32_t mn=svsel_s32(lo,svdup_s32(mn0),svdup_s32(mn1));
                d00=svmla_s32_x(p32,d00,svdot_s32(svdup_s32(0),q,x0),sc);
                m00=svmla_s32_x(p32,m00,sx0,mn);
                if(tokens>1){
                    d01=svmla_s32_x(p32,d01,svdot_s32(svdup_s32(0),q,x1),sc);
                    m01=svmla_s32_x(p32,m01,sx1,mn);
                }
            }
            {
                uint8_t sc0,mn0,sc1,mn1;
                get_scale_min_k4(g/32,scales[1],&sc0,&mn0);
                get_scale_min_k4(g/32+1,scales[1],&sc1,&mn1);
                svuint8_t packed=svld1_u8(pb,qs[1]+g/2);
                svuint8_t q0=svand_n_u8_x(p8,packed,15),q1=svlsr_n_u8_x(p8,packed,4);
                if(qh[1]){
                    svuint8_t hi=svld1_u8(pb,qh[1]);
                    q0=svadd_u8_x(p8,q0,svsel_u8(svcmpne_n_u8(pb,svand_n_u8_x(p8,hi,1u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
                    q1=svadd_u8_x(p8,q1,svsel_u8(svcmpne_n_u8(pb,svand_n_u8_x(p8,hi,2u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
                }
                svint8_t q=svreinterpret_s8_u8(svsplice_u8(pb,q0,q1));
                svint32_t sc=svsel_s32(lo,svdup_s32(sc0),svdup_s32(sc1));
                svint32_t mn=svsel_s32(lo,svdup_s32(mn0),svdup_s32(mn1));
                d10=svmla_s32_x(p32,d10,svdot_s32(svdup_s32(0),q,x0),sc);
                m10=svmla_s32_x(p32,m10,sx0,mn);
                if(tokens>1){
                    d11=svmla_s32_x(p32,d11,svdot_s32(svdup_s32(0),q,x1),sc);
                    m11=svmla_s32_x(p32,m11,sx1,mn);
                }
            }
            {
                uint8_t sc0,mn0,sc1,mn1;
                get_scale_min_k4(g/32,scales[2],&sc0,&mn0);
                get_scale_min_k4(g/32+1,scales[2],&sc1,&mn1);
                svuint8_t packed=svld1_u8(pb,qs[2]+g/2);
                svuint8_t q0=svand_n_u8_x(p8,packed,15),q1=svlsr_n_u8_x(p8,packed,4);
                if(qh[2]){
                    svuint8_t hi=svld1_u8(pb,qh[2]);
                    q0=svadd_u8_x(p8,q0,svsel_u8(svcmpne_n_u8(pb,svand_n_u8_x(p8,hi,1u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
                    q1=svadd_u8_x(p8,q1,svsel_u8(svcmpne_n_u8(pb,svand_n_u8_x(p8,hi,2u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
                }
                svint8_t q=svreinterpret_s8_u8(svsplice_u8(pb,q0,q1));
                svint32_t sc=svsel_s32(lo,svdup_s32(sc0),svdup_s32(sc1));
                svint32_t mn=svsel_s32(lo,svdup_s32(mn0),svdup_s32(mn1));
                d20=svmla_s32_x(p32,d20,svdot_s32(svdup_s32(0),q,x0),sc);
                m20=svmla_s32_x(p32,m20,sx0,mn);
                if(tokens>1){
                    d21=svmla_s32_x(p32,d21,svdot_s32(svdup_s32(0),q,x1),sc);
                    m21=svmla_s32_x(p32,m21,sx1,mn);
                }
            }
            {
                uint8_t sc0,mn0,sc1,mn1;
                get_scale_min_k4(g/32,scales[3],&sc0,&mn0);
                get_scale_min_k4(g/32+1,scales[3],&sc1,&mn1);
                svuint8_t packed=svld1_u8(pb,qs[3]+g/2);
                svuint8_t q0=svand_n_u8_x(p8,packed,15),q1=svlsr_n_u8_x(p8,packed,4);
                if(qh[3]){
                    svuint8_t hi=svld1_u8(pb,qh[3]);
                    q0=svadd_u8_x(p8,q0,svsel_u8(svcmpne_n_u8(pb,svand_n_u8_x(p8,hi,1u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
                    q1=svadd_u8_x(p8,q1,svsel_u8(svcmpne_n_u8(pb,svand_n_u8_x(p8,hi,2u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
                }
                svint8_t q=svreinterpret_s8_u8(svsplice_u8(pb,q0,q1));
                svint32_t sc=svsel_s32(lo,svdup_s32(sc0),svdup_s32(sc1));
                svint32_t mn=svsel_s32(lo,svdup_s32(mn0),svdup_s32(mn1));
                d30=svmla_s32_x(p32,d30,svdot_s32(svdup_s32(0),q,x0),sc);
                m30=svmla_s32_x(p32,m30,sx0,mn);
                if(tokens>1){
                    d31=svmla_s32_x(p32,d31,svdot_s32(svdup_s32(0),q,x1),sc);
                    m31=svmla_s32_x(p32,m31,sx1,mn);
                }
            }
        }
        {a00=svmla_n_f32_x(p32,a00,svcvt_f32_s32_x(p32,d00),x[0][b].d*d[0]);a00=svmla_n_f32_x(p32,a00,svcvt_f32_s32_x(p32,m00),-x[0][b].d*dm[0]);}
        if(tokens>1) {a01=svmla_n_f32_x(p32,a01,svcvt_f32_s32_x(p32,d01),x[1][b].d*d[0]);a01=svmla_n_f32_x(p32,a01,svcvt_f32_s32_x(p32,m01),-x[1][b].d*dm[0]);}
        {a10=svmla_n_f32_x(p32,a10,svcvt_f32_s32_x(p32,d10),x[0][b].d*d[1]);a10=svmla_n_f32_x(p32,a10,svcvt_f32_s32_x(p32,m10),-x[0][b].d*dm[1]);}
        if(tokens>1) {a11=svmla_n_f32_x(p32,a11,svcvt_f32_s32_x(p32,d11),x[1][b].d*d[1]);a11=svmla_n_f32_x(p32,a11,svcvt_f32_s32_x(p32,m11),-x[1][b].d*dm[1]);}
        {a20=svmla_n_f32_x(p32,a20,svcvt_f32_s32_x(p32,d20),x[0][b].d*d[2]);a20=svmla_n_f32_x(p32,a20,svcvt_f32_s32_x(p32,m20),-x[0][b].d*dm[2]);}
        if(tokens>1) {a21=svmla_n_f32_x(p32,a21,svcvt_f32_s32_x(p32,d21),x[1][b].d*d[2]);a21=svmla_n_f32_x(p32,a21,svcvt_f32_s32_x(p32,m21),-x[1][b].d*dm[2]);}
        {a30=svmla_n_f32_x(p32,a30,svcvt_f32_s32_x(p32,d30),x[0][b].d*d[3]);a30=svmla_n_f32_x(p32,a30,svcvt_f32_s32_x(p32,m30),-x[0][b].d*dm[3]);}
        if(tokens>1) {a31=svmla_n_f32_x(p32,a31,svcvt_f32_s32_x(p32,d31),x[1][b].d*d[3]);a31=svmla_n_f32_x(p32,a31,svcvt_f32_s32_x(p32,m31),-x[1][b].d*dm[3]);}
    }
    out[0][0]=svaddv_f32(p32,a00);
    if(tokens>1) out[0][1]=svaddv_f32(p32,a01);
    out[1][0]=svaddv_f32(p32,a10);
    if(tokens>1) out[1][1]=svaddv_f32(p32,a11);
    out[2][0]=svaddv_f32(p32,a20);
    if(tokens>1) out[2][1]=svaddv_f32(p32,a21);
    out[3][0]=svaddv_f32(p32,a30);
    if(tokens>1) out[3][1]=svaddv_f32(p32,a31);
}
static void iq_rows_tile2(float out[4][4],int type,const uint8_t *row,size_t rb,
        const glm5_iq_q8_block **x,int tokens,int blocks){
    if(tokens>2||(type!=GLM53F_GGML_Q4_K&&type!=GLM53F_GGML_Q5_K)){
        for(int r=0;r<2;r++)iq_rows_multi(out[r],type,row+(size_t)r*rb,x,tokens,blocks);
        return;
    }
    svbool_t p8=svptrue_b8(),p32=svptrue_b32(),pb=svwhilelt_b8(0,32),lo=svwhilelt_b32(0,8);
    svint8_t ones=svdup_s8(1);
    svfloat32_t a00=svdup_f32(0);
    svfloat32_t a01=svdup_f32(0);
    svfloat32_t a10=svdup_f32(0);
    svfloat32_t a11=svdup_f32(0);
    for(int b=0;b<blocks;b++){
        const uint8_t *qs[4],*scales[4],*qh[4];float d[4],dm[4];
        for(int r=0;r<2;r++){
            if(type==GLM53F_GGML_Q4_K){
                const block_q4_K *w=(const block_q4_K*)(row+(size_t)r*rb)+b;
                qs[r]=w->qs;scales[r]=w->scales;qh[r]=NULL;
                d[r]=ggml_fp16_to_fp32(w->d);dm[r]=ggml_fp16_to_fp32(w->dmin);
            }else{
                const block_q5_K *w=(const block_q5_K*)(row+(size_t)r*rb)+b;
                qs[r]=w->qs;scales[r]=w->scales;qh[r]=w->qh;
                d[r]=ggml_fp16_to_fp32(w->d);dm[r]=ggml_fp16_to_fp32(w->dmin);
            }
        }
        svint32_t d00=svdup_s32(0),m00=svdup_s32(0);
        svint32_t d01=svdup_s32(0),m01=svdup_s32(0);
        svint32_t d10=svdup_s32(0),m10=svdup_s32(0);
        svint32_t d11=svdup_s32(0),m11=svdup_s32(0);
        for(int g=0;g<256;g+=64){
            svint8_t x0=svld1_s8(p8,x[0][b].q+g);
            svint8_t x1=tokens>1?svld1_s8(p8,x[1][b].q+g):svdup_s8(0);
            svint32_t sx0=svdot_s32(svdup_s32(0),ones,x0);
            svint32_t sx1=svdot_s32(svdup_s32(0),ones,x1);
            {
                uint8_t sc0,mn0,sc1,mn1;
                get_scale_min_k4(g/32,scales[0],&sc0,&mn0);
                get_scale_min_k4(g/32+1,scales[0],&sc1,&mn1);
                svuint8_t packed=svld1_u8(pb,qs[0]+g/2);
                svuint8_t q0=svand_n_u8_x(p8,packed,15),q1=svlsr_n_u8_x(p8,packed,4);
                if(qh[0]){
                    svuint8_t hi=svld1_u8(pb,qh[0]);
                    q0=svadd_u8_x(p8,q0,svsel_u8(svcmpne_n_u8(pb,svand_n_u8_x(p8,hi,1u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
                    q1=svadd_u8_x(p8,q1,svsel_u8(svcmpne_n_u8(pb,svand_n_u8_x(p8,hi,2u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
                }
                svint8_t q=svreinterpret_s8_u8(svsplice_u8(pb,q0,q1));
                svint32_t sc=svsel_s32(lo,svdup_s32(sc0),svdup_s32(sc1));
                svint32_t mn=svsel_s32(lo,svdup_s32(mn0),svdup_s32(mn1));
                d00=svmla_s32_x(p32,d00,svdot_s32(svdup_s32(0),q,x0),sc);
                m00=svmla_s32_x(p32,m00,sx0,mn);
                if(tokens>1){
                    d01=svmla_s32_x(p32,d01,svdot_s32(svdup_s32(0),q,x1),sc);
                    m01=svmla_s32_x(p32,m01,sx1,mn);
                }
            }
            {
                uint8_t sc0,mn0,sc1,mn1;
                get_scale_min_k4(g/32,scales[1],&sc0,&mn0);
                get_scale_min_k4(g/32+1,scales[1],&sc1,&mn1);
                svuint8_t packed=svld1_u8(pb,qs[1]+g/2);
                svuint8_t q0=svand_n_u8_x(p8,packed,15),q1=svlsr_n_u8_x(p8,packed,4);
                if(qh[1]){
                    svuint8_t hi=svld1_u8(pb,qh[1]);
                    q0=svadd_u8_x(p8,q0,svsel_u8(svcmpne_n_u8(pb,svand_n_u8_x(p8,hi,1u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
                    q1=svadd_u8_x(p8,q1,svsel_u8(svcmpne_n_u8(pb,svand_n_u8_x(p8,hi,2u<<(g/32)),0),svdup_u8(16),svdup_u8(0)));
                }
                svint8_t q=svreinterpret_s8_u8(svsplice_u8(pb,q0,q1));
                svint32_t sc=svsel_s32(lo,svdup_s32(sc0),svdup_s32(sc1));
                svint32_t mn=svsel_s32(lo,svdup_s32(mn0),svdup_s32(mn1));
                d10=svmla_s32_x(p32,d10,svdot_s32(svdup_s32(0),q,x0),sc);
                m10=svmla_s32_x(p32,m10,sx0,mn);
                if(tokens>1){
                    d11=svmla_s32_x(p32,d11,svdot_s32(svdup_s32(0),q,x1),sc);
                    m11=svmla_s32_x(p32,m11,sx1,mn);
                }
            }
        }
        {a00=svmla_n_f32_x(p32,a00,svcvt_f32_s32_x(p32,d00),x[0][b].d*d[0]);a00=svmla_n_f32_x(p32,a00,svcvt_f32_s32_x(p32,m00),-x[0][b].d*dm[0]);}
        if(tokens>1) {a01=svmla_n_f32_x(p32,a01,svcvt_f32_s32_x(p32,d01),x[1][b].d*d[0]);a01=svmla_n_f32_x(p32,a01,svcvt_f32_s32_x(p32,m01),-x[1][b].d*dm[0]);}
        {a10=svmla_n_f32_x(p32,a10,svcvt_f32_s32_x(p32,d10),x[0][b].d*d[1]);a10=svmla_n_f32_x(p32,a10,svcvt_f32_s32_x(p32,m10),-x[0][b].d*dm[1]);}
        if(tokens>1) {a11=svmla_n_f32_x(p32,a11,svcvt_f32_s32_x(p32,d11),x[1][b].d*d[1]);a11=svmla_n_f32_x(p32,a11,svcvt_f32_s32_x(p32,m11),-x[1][b].d*dm[1]);}
    }
    out[0][0]=svaddv_f32(p32,a00);
    if(tokens>1) out[0][1]=svaddv_f32(p32,a01);
    out[1][0]=svaddv_f32(p32,a10);
    if(tokens>1) out[1][1]=svaddv_f32(p32,a11);
}

#endif
