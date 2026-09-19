#ifndef DEQUANT_PIPE_HWB_COMPAT_H
#define DEQUANT_PIPE_HWB_COMPAT_H

/*
 * FJSVxoshwb installs libhwb on Fugaku compute nodes but does not install a
 * public header.  These declarations match the DWARF signatures exported by
 * libhwb.so.1 (FJSVxoshwb 0.0.18).
 */
int vhbm_bar_init(unsigned long core_mask);
int vhbm_bar_fini(int barrier_descriptor);
long vhbm_bar_assign(int barrier_descriptor, long *requested_window);
int vhbm_bar_unassign(int barrier_descriptor);
void vhbm_bar(long window);

#endif
