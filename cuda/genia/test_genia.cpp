#include "genia_cuda.h"
#include "../../common/genia_math.h"
#include "../../common/sam3d_pose_tokens.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool passed, const char *message)
{
    if (!passed) { std::fprintf(stderr, "FAIL: %s (%s)\n", message, genia_cuda_error()); std::exit(1); }
}
static void close_to(float a, float b, float tolerance = 1e-6f)
{
    require(std::isfinite(a) && std::fabs(a-b) <= tolerance, "numeric comparison");
}

int main(int argc, char **argv)
{
    const bool gpu = argc == 2 && !std::strcmp(argv[1], "--gpu");
    require(argc == 1 || gpu, "usage: test_genia [--gpu]");
    float scores[3] = {1, 0, 1}, weights[3];
    require(!genia_fusion_weights(scores, 3, 30, .001f, weights), "visibility weights");
    close_to(weights[0], .5f/1.001f); close_to(weights[1], .001f/1.001f);
    close_to(weights[2], weights[0]);
    float all_hidden[3] = {0, 0, 0};
    require(!genia_fusion_weights(all_hidden, 3, 30, .001f, weights), "unseen weights");
    for (float value: weights) close_to(value, 1.f/3);
    close_to(genia_attention_compensation(5, 1.f/1369),
             (float)std::log((1368+std::exp(5.))/1369));
    float x[2] = {1, 2}, v[2] = {3, 4}, gradient[2] = {0, 2}, out[2];
    require(!genia_tweedie(x, v, 2, .5f, out), "Tweedie estimate");
    close_to(out[0], 2.5f); close_to(out[1], 4);
    float guided[2] = {3, 4};
    require(!genia_guidance(guided, gradient, 2, .5f, 1), "normalized guidance");
    close_to(guided[0], 3); close_to(guided[1], 1.5f);
    float zero[2] = {0, 0};
    require(!genia_guidance(guided, zero, 2, 1, 1), "zero gradient");
    close_to(guided[1], 1.5f);
    float bad[2] = {0, NAN};
    require(genia_guidance(guided, bad, 2, 1, 1) == -1, "nonfinite gradient rejection");
    close_to(guided[1], 1.5f);

    const int grid = 8;
    std::vector<uint8_t> occupancy(grid*grid*grid, 0);
    int32_t coords[6] = {2, 3, 3, 5, 3, 3};
    occupancy[(2*grid+3)*grid+3] = occupancy[(5*grid+3)*grid+3] = 1;
    float camera[3] = {-2, 3.5f, 3.5f};
    require(genia_voxel_visible(coords, camera, occupancy.data(), grid, 0) == 1, "front voxel visible");
    require(genia_voxel_visible(coords+3, camera, occupancy.data(), grid, 0) == 0, "back voxel occluded");
    require(genia_voxel_visible(coords+3, camera, occupancy.data(), grid, 4) == 1, "neighbor tolerance");
    float behind[3] = {10, 3.5f, 3.5f};
    require(genia_voxel_visible(coords, behind, occupancy.data(), grid, 0) == 0, "reverse occlusion");
    int32_t invalid[3] = {-1, 0, 0};
    require(genia_voxel_visible(invalid, camera, occupancy.data(), grid, 0) == -1, "invalid coordinate");
    float r[9] = {1,0,0,0,1,0,0,0,1}, translation[3] = {0,0,2}, k[9] = {100,0,32,0,100,32,0,0,1}, uv[2];
    int32_t center[3] = {0,0,0};
    require(genia_project_voxel(center, 1, r, translation, 1, k, 64, 64, uv) == 1, "voxel projection");
    close_to(uv[0], 32); close_to(uv[1], 32);

    sam3d_pose_tokens pose = {};
    require(sam3d_pose_tokens_get(&pose, 1, nullptr, 0, nullptr, nullptr) == -4, "unavailable pose tokens");
    float raw[13]; for (int i = 0; i < 13; ++i) raw[i] = float(i);
    float *modalities[5] = {nullptr, raw, raw+6, raw+9, raw+12};
    sam3d_pose_tokens_store(&pose, modalities);
    int n = 0, c = 0; float rotation[6];
    require(!sam3d_pose_tokens_get(&pose, 1, nullptr, 0, &n, &c) && n == 1 && c == 6, "pose shape query");
    require(sam3d_pose_tokens_get(&pose, 1, rotation, 5, &n, &c) == -1, "pose capacity guard");
    require(!sam3d_pose_tokens_get(&pose, 1, rotation, 6, &n, &c), "pose copy");
    for (int i = 0; i < 6; ++i) close_to(rotation[i], float(i));
    require(sam3d_pose_tokens_get(&pose, 0, nullptr, 0, nullptr, nullptr) == -1, "pose modality guard");

    if (gpu) {
        genia_cuda *ctx = genia_cuda_open(0, 1 << 20);
        require(ctx != nullptr, "CUDA context");
        uint8_t visible[2];
        require(!genia_cuda_visibility(ctx, coords, 2, camera, occupancy.data(), grid, 0, visible), "CUDA visibility");
        require(visible[0] == 1 && visible[1] == 0, "CUDA occlusion");
        require(!genia_cuda_tweedie(ctx, x, v, 2, .5f, out), "CUDA Tweedie");
        close_to(out[0], 2.5f); close_to(out[1], 4);
        require(!genia_cuda_guidance(ctx, v, gradient, 2, .5f, 1, out), "CUDA guidance");
        close_to(out[0], 3); close_to(out[1], 1.5f);
        require(!genia_cuda_guidance(ctx, v, zero, 2, 1, 1, out), "CUDA zero guidance");
        close_to(out[0], 3); close_to(out[1], 4);
        require(genia_cuda_guidance(ctx, v, bad, 2, 1, 1, out) == -1, "CUDA nonfinite rejection");
        float velocity[3] = {10, 99, 20}, fused;
        require(!genia_cuda_fuse(ctx, scores, velocity, 3, 1, 1, 30, .001f, &fused), "CUDA fusion");
        close_to(fused, (15.f+.099f)/1.001f, 3e-6f);
        require(genia_cuda_peak_bytes(ctx) < (1 << 20), "allocation accounting");
        float rows[34] = {.015f, -.01f, 2.f, 0, 0, 0, .1f, -.2f, .3f, .5f,
                          -1.8f, -1.7f, -1.6f, 1.f, .15f, -.08f, .12f,
                          -.035f, .025f, 2.45f, 0, 0, 0, -.1f, .15f, .2f, .7f,
                          -1.5f, -1.4f, -1.6f, 1.f, -.13f, .08f, -.04f};
        float camera_matrix[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        float intrinsics[9] = {24,0,4, 0,24,4, 0,0,1}, background[3] = {.1f,.2f,.3f};
        float image[8*8*4], depth[8*8], image_gradient[8*8*4], depth_gradient[8*8], row_gradient[34], camera_gradient[12];
        for (int i = 0; i < 8*8*4; ++i) image_gradient[i] = .001f*(1+i%7);
        for (int i = 0; i < 8*8; ++i) depth_gradient[i] = .003f;
        require(!genia_cuda_render(ctx, rows, 1, camera_matrix, intrinsics, 8, 8, background,
            image_gradient, depth_gradient, image, depth, row_gradient, camera_gradient), "CUDA Gaussian render/backward");
        for (float value: depth) close_to(value, 2.f, 1e-6f);
        require(!genia_cuda_render(ctx, rows, 2, camera_matrix, intrinsics, 8, 8, background,
            image_gradient, depth_gradient, image, depth, row_gradient, camera_gradient), "CUDA multi-Gaussian compositing/backward");
        for (float value: depth) require(value >= 2 && value <= 2.45f, "expected-depth mixture");
        for (int i = 3; i < 6; ++i) close_to(row_gradient[i], 0);
        auto evaluate = [&]() {
            require(!genia_cuda_render(ctx, rows, 2, camera_matrix, intrinsics, 8, 8, background,
                nullptr, nullptr, image, depth, nullptr, nullptr), "CUDA Gaussian finite-difference render");
            double loss = 0;
            for (int i = 0; i < 8*8*4; ++i) loss += double(image[i])*image_gradient[i];
            for (int i = 0; i < 8*8; ++i) loss += double(depth[i])*depth_gradient[i];
            return loss;
        };
        for (int i = 0; i < 46; ++i) {
            float *parameter = i < 34 ? rows+i : camera_matrix+i-34;
            float saved = *parameter, epsilon = .001f;
            *parameter = saved+epsilon; double plus = evaluate();
            *parameter = saved-epsilon; double minus = evaluate();
            *parameter = saved;
            float numerical = (float)((plus-minus)/(2*epsilon));
            float analytic = i < 34 ? row_gradient[i] : camera_gradient[i-34];
            if (std::fabs(numerical-analytic) >= 1e-4f+1e-3f*std::fabs(numerical))
                std::fprintf(stderr, "gradient parameter %d: numerical=%.9g analytic=%.9g\n", i, numerical, analytic);
            require(std::fabs(numerical-analytic) < 1e-4f+1e-3f*std::fabs(numerical), "Gaussian/pose finite-difference gradient");
        }
        genia_cuda_close(ctx);
        ctx = genia_cuda_open(0, 1);
        require(ctx != nullptr, "small budget context");
        require(genia_cuda_tweedie(ctx, x, v, 2, .5f, out) == -1, "device budget rejection");
        genia_cuda_close(ctx);
    }
    std::printf("PASS: GenIA alignment, visibility, guidance, pose retention%s\n", gpu ? ", Gaussian/pose gradients (CUDA + host)" : " (host)");
    return 0;
}
