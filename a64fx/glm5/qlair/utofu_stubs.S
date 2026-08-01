    .text
.macro USTUB name
    .global \name
    .type \name, %function
\name:
    ret
    .size \name, .-\name
.endm
    USTUB utofu_get_onesided_tnis
    USTUB utofu_query_onesided_caps
    USTUB utofu_create_vcq
    USTUB utofu_query_vcq_id
    USTUB utofu_reg_mem
    USTUB utofu_put
    USTUB utofu_get
    USTUB utofu_poll_tcq
    USTUB utofu_poll_mrq
    USTUB utofu_query_my_coords
    USTUB utofu_free_vcq

// --- real pthread symbols (qlair intercepts these by name -> ThreadManager) ---
// clair's C frontend remaps pthread_* to fake clair_pthread_* stubs (return
// EAGAIN, no thread). These asm shims tail-call the RAW symbols so qlair's
// interceptor fires instead. Asm call sites are not remapped by clair.
    .global pthread_create
    .type pthread_create, %function
pthread_create:
    ret
    .global pthread_join
    .type pthread_join, %function
pthread_join:
    ret
    .global qlair_spawn
    .type qlair_spawn, %function
qlair_spawn:
    b pthread_create
    .global qlair_join
    .type qlair_join, %function
qlair_join:
    b pthread_join

// myput(x0=hdl, x1=rmt_vcq_id, x2=lcl_stadd, x3=rmt_stadd, x4=len)
//   -> utofu_put(hdl,rmt_id,lcl,rmt,len, edata=0, flags=TCQ_NOTICE(1<<14), cb=1)
// Uses a 5-arg C call (clair passes x0-x4 correctly) and fills x5-x7 in asm,
// avoiding clair's buggy 8-arg call lowering.
    .global myput
    .type myput, %function
myput:
    mov x5, #0
    mov x6, #16384
    mov x7, #1
    b utofu_put

// uint64_t rd_cyc(void) -> CNTVCT_EL0 (qlair: raw simulated cycles @2GHz incl. uTofu latency)
    .global rd_cyc
    .type rd_cyc, %function
rd_cyc:
    .word 0xd53be040
    ret
    .global myputn
    .type myputn, %function
myputn:
    mov x5, #0
    mov x6, #16384
    mov x7, #1
    b utofu_put
