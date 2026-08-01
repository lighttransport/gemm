    .text
.macro USTUB name
    .global \name
    .type \name, %function
\name:
    ret
.endm
    USTUB utofu_get_onesided_tnis
    USTUB utofu_create_vcq
    USTUB utofu_query_vcq_id
    USTUB utofu_reg_mem
    USTUB utofu_put
    USTUB utofu_poll_tcq
    .global rd_cyc
    .type rd_cyc, %function
rd_cyc:
    .inst 0xd53be040      // mrs x0, cntvct_el0
    ret
    .global myputn        // myputn(x0=hdl,x1=rmt_id,x2=lcl_st,x3=rmt_st,x4=len)
    .type myputn, %function
myputn:
    mov x5, #0
    mov x6, #16384        // TCQ_NOTICE
    mov x7, #1
    b utofu_put
