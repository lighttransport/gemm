/* Stand-in for PyTorch's header when building SageAttention kernels without
 * PyTorch: the upstream headers use torch only inside host-side macros. */
#pragma once
#include <sstream>
#include <stdexcept>
