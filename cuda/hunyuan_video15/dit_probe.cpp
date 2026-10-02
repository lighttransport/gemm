/* Fast12 DiT diagnostic: first step or complete Euler chain from saved inputs. */
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include "stable-diffusion.h"
#include "model/diffusion/hunyuan.hpp"
#include "model_manager.h"
#include "runtime/denoiser.hpp"
#include "core/hv15_dump.hpp"

static bool read_tensor(const std::string& path, sd::Tensor<float>& value) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(value.numel()*sizeof(float))) return false;
    file.seekg(0);
    file.read(reinterpret_cast<char *>(value.data()), value.numel()*sizeof(float));
    if (!file) return false;
    for (float x : value.values()) if (!std::isfinite(x)) return false;
    return true;
}

int main(int argc, char **argv) {
    if (argc != 4 && !(argc == 5 && (!std::strcmp(argv[4], "--f32-acc") || !std::strcmp(argv[4], "--f32-weights") || !std::strcmp(argv[4], "--first-block") || !std::strcmp(argv[4], "--full-denoise")))) {
        std::fprintf(stderr,"usage: %s DIT.safetensors INPUT_DUMPS EXISTING_OUT_DIR [--f32-acc|--f32-weights|--first-block|--full-denoise]\n",argv[0]);
        return 2;
    }
    const bool full_denoise = argc == 5 && !std::strcmp(argv[4], "--full-denoise");
    const std::string input = argv[2];
    std::ifstream metadata(input+"/noise_input.shape.json",std::ios::binary|std::ios::ate);
    if (!metadata || metadata.tellg() <= 0 || metadata.tellg() > 256) return 2;
    metadata.seekg(0);
    std::string shape((std::istreambuf_iterator<char>(metadata)),std::istreambuf_iterator<char>());
    long long batch=0, channels=0, t=0, h=0, w=0;
    int consumed=0;
    if (std::sscanf(shape.c_str()," [ %lld , %lld , %lld , %lld , %lld ] %n",
            &batch,&channels,&t,&h,&w,&consumed) != 5 || consumed != static_cast<int>(shape.size()) ||
        batch != 1 || channels != 32 || t<1 || t>31 || h<1 || h>80 || w<1 || w>80 || w*h*t>33390) return 2;
    sd::Tensor<float> noise({w,h,t,32,1}), encoded({w,h,1,32,1}), vision({1152,729,1});
    std::ifstream qfile(input+"/qwen_hidden.f32",std::ios::binary|std::ios::ate);
    if (!qfile) return 2;
    auto qbytes = qfile.tellg();
    if (qbytes <= 0 || qbytes % (3584*sizeof(float)) || qbytes/(3584*sizeof(float)) > 1000) return 2;
    sd::Tensor<float> context({3584,static_cast<int64_t>(qbytes/(3584*sizeof(float))),1});
    if (!read_tensor(input+"/noise_input.f32",noise) || !read_tensor(input+"/vae_encoded.f32",encoded) ||
        !read_tensor(input+"/siglip_hidden.f32",vision) || !read_tensor(input+"/qwen_hidden.f32",context)) return 2;
    sd::Tensor<float> condition({w,h,t,33,1});
    std::fill(condition.values().begin(),condition.values().end(),0.f);
    const size_t area = w*h, plane = area*t;
    for (int c=0;c<32;++c) std::copy(encoded.data()+c*area,encoded.data()+(c+1)*area,condition.data()+c*plane);
    std::fill(condition.data()+32*plane,condition.data()+32*plane+area,1.f);
    auto timestep = sd::Tensor<float>::from_vector({1000.f});
    float sigma = 1.f-1.f/12.f;
    auto timestep_r = sd::Tensor<float>::from_vector({7.f*sigma/(1.f+6.f*sigma)});
    sd_set_log_callback([](sd_log_level_t,const char *text,void *) { std::fprintf(stderr,"%s",text); },nullptr);
    if (setenv("HV15_DUMP_DIR",argv[3],1)) return 2;
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("CUDA0",nullptr);
    auto cpu = ggml_backend_init_by_name("CPU",nullptr);
    if (!backend || !cpu) return 1;
    int result = 1;
    {
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(argv[1],"model.diffusion_model.",VERSION_HUNYUAN_VIDEO)) return 1;
        if (argc == 5 && !std::strcmp(argv[4], "--f32-weights")) loader.set_wtype_override(GGML_TYPE_F32);
        auto storage = loader.get_tensor_storage_map();
        if (argc == 5 && !std::strcmp(argv[4], "--first-block")) {
            const std::string prefix = "model.diffusion_model.double_blocks.";
            for (auto it = storage.begin(); it != storage.end();) {
                if (it->first.compare(0,prefix.size(),prefix)==0 &&
                    std::strtol(it->first.c_str()+prefix.size(),nullptr,10)>0) it=storage.erase(it);
                else ++it;
            }
        }
        auto manager = std::make_shared<ModelManager>();
        if (!manager->set_loader(std::move(loader))) return 1;
        manager->set_enable_mmap(true);
        manager->prepare_file_io();
        Hunyuan::HunyuanVideoRunner runner(backend,storage,"model.diffusion_model",VERSION_HUNYUAN_VIDEO,manager);
        runner.set_max_graph_vram_bytes(4ULL*1024*1024*1024);
        runner.set_flash_attention_enabled(true);
        std::map<std::string,ggml_tensor *> tensors;
        runner.get_param_tensors(tensors,"model.diffusion_model");
        if (manager->register_param_tensors(ModelComponent::Diffusion,std::move(tensors),
                ModelManager::ResidencyMode::ParamBackend,backend,cpu)) {
            auto evaluate = [&](const sd::Tensor<float>& current) {
              auto graph = [&]() {
                auto gf = runner.build_graph(current,timestep,context,condition,{},{},{},vision,timestep_r);
                hv15_dump("rope_pe", sd::Tensor<float>({2,2,64,static_cast<int64_t>(runner.pe_vec.size()/256)}, runner.pe_vec));
                if (argc == 5 && !full_denoise && std::strcmp(argv[4], "--first-block")) for (int i=0;i<ggml_graph_n_nodes(gf);++i) {
                    auto node = ggml_graph_node(gf,i);
                    if (node->op == GGML_OP_MUL_MAT) ggml_prec_set_acc(node,GGML_PREC_F32);
                }
                return gf;
            };
              return runner.GGMLRunner::compute(graph,8,false);
            };
            auto output = full_denoise ? std::optional<sd::Tensor<float>>{} : evaluate(noise);
            if (full_denoise) {
                std::vector<float> sigmas(13);
                std::ofstream schedule(std::string(argv[3])+"/sigmas.json");
                schedule.precision(9); schedule << '[';
                for (int i=0;i<=12;++i) {
                    const float t=1.f-static_cast<float>(i)/12.f;
                    sigmas[i]=7.f*t/(1.f+6.f*t);
                    if (i) schedule << ',';
                    schedule << sigmas[i];
                }
                schedule << "]\n";
                if (!schedule) return 1;
                auto denoise = [&](const sd::Tensor<float>& current, float sigma, int step) {
                    sd::guidance::GuiderOutput result;
                    char name[32];
                    if (step>1) {
                        std::snprintf(name,sizeof(name),"latent_step_%02d",step-1);
                        hv15_dump(name,current);
                    }
                    timestep.data()[0]=sigma*1000.f;
                    timestep_r.data()[0]=sigmas.at(step);
                    auto value=evaluate(current);
                    if (!value || value->empty()) return result;
                    while (value->dim()<5) value->unsqueeze_(value->dim());
                    if (value->shape()!=noise.shape()) return result;
                    for (float x : value->values()) if (!std::isfinite(x)) return result;
                    if (step==1) hv15_dump("dit_first",*value);
                    std::snprintf(name,sizeof(name),"dit_step_%02d",step);
                    hv15_dump(name,*value);
                    result.pred=current-(*value)*sigma;
                    std::fprintf(stderr,"DENOISE_STEP %d 12\n",step);
                    return result;
                };
                auto final=sample_euler(denoise,noise,sigmas);
                if (!final.empty()) {
                    for (float x : final.values()) if (!std::isfinite(x)) return 1;
                    hv15_dump("latent_step_12",final);
                    hv15_dump("latent_final",final);
                    result=0;
                }
            }
            if (output && !output->empty()) {
                auto prediction = std::move(*output);
                while (prediction.dim()<5) prediction.unsqueeze_(prediction.dim());
                bool finite = true;
                for (float x : prediction.values()) finite = finite && std::isfinite(x);
                if (finite && prediction.shape()==noise.shape()) {
                    hv15_dump("dit_first",prediction);
                    result = 0;
                }
            }
            runner.runner_end();
        }
        manager->unregister_param_tensors(ModelComponent::Diffusion);
    }
    std::fprintf(stderr,"HV15 precise attention calls: %llu\n",
                 static_cast<unsigned long long>(hv15_cuda_attention_calls()));
    ggml_backend_free(cpu); ggml_backend_free(backend);
    return result;
}
