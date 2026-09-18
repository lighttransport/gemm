/* FCC 4.12 / LLVM 7 workaround for the A64FX reference build.
 *
 * The compiler's SVE instruction selector currently rejects the auto-generated
 * FP16 masked fabs sequence in ggml-cpu/unary-ops.cpp.  Explicit ggml SVE
 * intrinsics remain available; this only disables LLVM's function-level
 * auto-vectorization for the diagnostic build.
 */
#pragma clang optimize off
