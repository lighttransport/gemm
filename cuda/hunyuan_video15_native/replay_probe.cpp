#include "models.hpp"
#include <iostream>
using namespace hv15n;
int main(int argc, char **argv) {
    try {
        require(argc == 4, "replay_probe CASE_DIR OUTPUT_DIR legacy|candidate");
        require(std::string(argv[3]) == "candidate" || std::string(argv[3]) == "legacy", "replay mode");
        fs::path dir = argv[1], out = argv[2];
        require(!fs::exists(out) || fs::is_empty(out), "replay output must be empty/new");
        fs::create_directories(out);
        Json description(dir / "case.json");
        auto root = description.value.get();
        auto start = std::chrono::steady_clock::now();
        Gpu g(0, 14336, false, false, std::string(argv[3]) == "candidate");
        auto seconds = [](auto a) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count(); };
        auto input = [&](const char *name) {
            Json meta(dir / (std::string(name) + ".json"));
            auto list = field(meta.value.get(), "shape");
            require(list && list->type == JSON_ARRAY, "fixture shape");
            std::vector<int> shape;
            for (int i = 0; i < list->arr.count; i++)
                shape.push_back(int(list->arr.items[i].num));
            return g.upload(read_f32(dir / (std::string(name) + ".f32"), product(shape)), shape);
        };
        auto number = [&](const char *name) {
            auto value = field(root, name);
            require(value && value->type == JSON_NUMBER, std::string("fixture number: ") + name);
            return int(value->num);
        };
        std::string kind = string(root, "kind");
        Tensor x, w, q, k, v, img, txt, vec, negative_img, negative_txt, negative_vec, latent;
        std::unique_ptr<Weights> weights;
        if (kind == "gemm") { x = input("x"); w = input("w"); }
        else if (kind == "attention") { q = input("q"); k = input("k"); v = input("v"); }
        else {
            weights = std::make_unique<Weights>(string(root, "checkpoint"));
            if (kind == "dit_block" || kind == "dit_pair" || kind == "dit_final" || kind == "dit_pair_final") {
                img = input("img"); txt = input("txt"); vec = input("vec");
                if (kind == "dit_pair" || kind == "dit_pair_final") {
                    negative_img = input("negative_img"); negative_txt = input("negative_txt"); negative_vec = input("negative_vec");
                }
                if (kind == "dit_final" || kind == "dit_pair_final") latent = input("latent");
            }
            else x = input("x");
        }
        double setup = seconds(start);
        std::vector<double> device, wall;
        Tensor result, text_result, negative_result, negative_text_result, guided, updated;
        int repeats = number("repeats");
        require(repeats > 0 && repeats <= 10, "fixture repetitions");
        for (int i = 0; i < repeats; ++i) {
            CUevent begin = nullptr, end = nullptr;
            g.check(cuEventCreate(&begin, CU_EVENT_DEFAULT), "timer begin");
            g.check(cuEventCreate(&end, CU_EVENT_DEFAULT), "timer end");
            auto tick = std::chrono::steady_clock::now();
            g.check(cuEventRecord(begin, g.stream), "record begin");
            if (kind == "gemm") result = g.matmul(x, w);
            else if (kind == "attention") result = g.attention(q, k, v, number("heads"), number("heads"));
            else if (kind == "dit_block" || kind == "dit_pair") {
                int first=number("index"),blocks=field(root,"blocks")?number("blocks"):1;
                require(blocks>=1 && blocks<=16 && first+blocks<=54,"bounded block range");
                std::vector<DitState> states{{img,txt,vec,{}}};
                if (kind == "dit_pair") states.push_back({negative_img,negative_txt,negative_vec,{}});
                dit_blocks(g,*weights,states,first,blocks,number("height"),number("width"));
                result=std::move(states[0].img);text_result=std::move(states[0].txt);
                if (kind == "dit_pair") {
                    negative_result=std::move(states[1].img);negative_text_result=std::move(states[1].txt);
                }
            } else if (kind == "dit_final" || kind == "dit_pair_final") {
                result = dit_finish(g,*weights,{img,txt,vec,latent.shape});
                if (kind == "dit_pair_final") {
                    negative_result = dit_finish(g,*weights,{negative_img,negative_txt,negative_vec,latent.shape});
                    auto difference=g.op(result,1,&negative_result,nullptr,-1.f);
                    guided=g.op(negative_result,1,&difference,nullptr,6.f);
                } else guided=result;
                updated=g.op(latent,1,&guided,nullptr,float(field(root,"delta")->num));
            } else if (kind == "vae_decode") result = vae(g, *weights, x, false);
            else if (kind == "qwen" || kind == "siglip" || kind == "byt5") result = encoder_block(g, *weights, kind, number("index"), x);
            else if (kind == "conv") result = g.conv(*weights, string(root, "prefix"), x, number("causal") != 0);
            else throw std::runtime_error("unknown replay kind");
            g.check(cuEventRecord(end, g.stream), "record end");
            g.check(cuEventSynchronize(end), "finish measurement");
            float ms = 0;
            g.check(cuEventElapsedTime(&ms, begin, end), "elapsed");
            wall.push_back(seconds(tick)); device.push_back(ms / 1000.);
            cuEventDestroy(begin); cuEventDestroy(end);
        }
        g.dump(result, out, "output");
        if (text_result.pointer) g.dump(text_result, out, "text_output");
        if (negative_result.pointer) g.dump(negative_result, out, "negative_output");
        if (negative_text_result.pointer) g.dump(negative_text_result, out, "negative_text_output");
        if (guided.pointer) g.dump(guided, out, "guided_output");
        if (updated.pointer) g.dump(updated, out, "updated_latent");
        std::ofstream report(out / "timing.json");
        report << "{\"setup_seconds\":" << setup << ",\"device_seconds\":[";
        for (size_t i = 0; i < device.size(); i++) report << (i ? "," : "") << device[i];
        report << "],\"wall_seconds\":[";
        for (size_t i = 0; i < wall.size(); i++) report << (i ? "," : "") << wall[i];
        report << "],\"metrics\":" << g.metrics() << "}\n";
        std::cout << "PASS " << kind << " " << g.metrics() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
