#include "player.h"
#include "native.h"
#include <filament/Engine.h>
#include <filament/Camera.h>
#include <filament/Scene.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <backend/PixelBufferDescriptor.h>
#include <filament/Renderer.h>
#include <filament/SwapChain.h>
#include <filament/LightManager.h>
#include <filament/IndirectLight.h>
#include <filament/RenderableManager.h>
#include <filament/VertexBuffer.h>
#include <filament/IndexBuffer.h>
#include <gltfio/AssetLoader.h>
#include <gltfio/FilamentAsset.h>
#include <gltfio/MaterialProvider.h>
#include <gltfio/ResourceLoader.h>
#include <gltfio/TextureProvider.h>
#include <gltfio/materials/uberarchive.h>
#include <utils/EntityManager.h>
#include <utils/NameComponentManager.h>
#include <math/mat3.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <chrono>

using namespace filament;
using namespace filament::math;
namespace vhuman {
namespace {
template<class T>void read(std::ifstream& f,T* out,size_t n) {
    if(!f.read(reinterpret_cast<char*>(out),sizeof(T)*n))throw std::runtime_error("truncated mobile asset");
}
float3 unit(float3 v) {float n=length(v);return n>1e-10f?v/n:float3{0,0,1};}
backend::BufferDescriptor upload(const void* source,size_t size) {
    auto* copy=new uint8_t[size];memcpy(copy,source,size);
    return {copy,size,[](void* data,size_t,void*){delete[] static_cast<uint8_t*>(data);}};
}
struct Vertex {float3 position;quatf tangent;float2 uv;float4 color{1};};
struct Part {
    std::string name;int joint=-1;bool native=false;
    float3 center{0};
    std::vector<float3> rest,position,normal,offset;
    std::vector<float2> uv;
    std::vector<uint32_t> triangles,ids;
    std::vector<float> weights;
    std::vector<uint8_t> direct;  // native parts: per-vertex native copy vs appended surface binding
    std::vector<Vertex> vertices;
    bool shared(size_t i) const {return native&&direct[i];}
    VertexBuffer* vb=nullptr;IndexBuffer* ib=nullptr;
};
}
struct Player::Impl {
    Engine* engine=nullptr;Renderer* renderer=nullptr;SwapChain* swap=nullptr;
    Scene* scene=nullptr;View* view=nullptr;Camera* camera=nullptr;
    IndirectLight* ambient=nullptr;
    gltfio::MaterialProvider* materials=nullptr;gltfio::AssetLoader* loader=nullptr;
    gltfio::FilamentAsset* asset=nullptr;vh_mobile* model=nullptr;
    utils::NameComponentManager names{utils::EntityManager::get()};
    utils::Entity camera_entity{},key{},fill{};
    uint32_t width=0,height=0;std::vector<Part> parts;std::vector<float3> native;
    ~Impl() {
        if(!engine)return;
        engine->flushAndWait();
        if(asset)loader->destroyAsset(asset);
        for(auto& part:parts){engine->destroy(part.vb);engine->destroy(part.ib);}
        if(loader)gltfio::AssetLoader::destroy(&loader);
        if(materials){materials->destroyMaterials();delete materials;}
        for(auto entity:{key,fill})if(entity){engine->destroy(entity);utils::EntityManager::get().destroy(entity);}
        engine->destroy(view);engine->destroy(scene);engine->destroy(ambient);engine->destroy(renderer);engine->destroy(swap);
        if(camera_entity){engine->destroyCameraComponent(camera_entity);utils::EntityManager::get().destroy(camera_entity);}
        vh_mobile_free(model);Engine::destroy(&engine);
    }
    void load_parts(const std::string& dir) {
        std::ifstream f(dir+"/streams.bin",std::ios::binary),bind(dir+"/bindings.bin",std::ios::binary);
        char magic[8];uint32_t count,count2;read(f,magic,8);
        if(memcmp(magic,"VHMES001",8))throw std::runtime_error("bad mesh format");
        read(f,&count,1);read(bind,magic,8);read(bind,&count2,1);
        bool fitted=!memcmp(magic,"VHBND002",8);
        if((!fitted&&memcmp(magic,"VHBND001",8))||count!=count2||count>128)throw std::runtime_error("bad binding format");
        parts.resize(count);size_t total=0,total_vertices=0;
        for(auto& part:parts) {
            uint32_t dims[3];read(f,dims,3);
            if(!dims[0]||dims[0]>128||!dims[1]||dims[1]>500000||!dims[2]||dims[2]>80000)throw std::runtime_error("invalid mesh dimensions");
            total+=dims[2];if(total>80000)throw std::runtime_error("triangle budget exceeded");
            total_vertices+=dims[1];if(total_vertices>500000)throw std::runtime_error("vertex budget exceeded");
            part.name.resize(dims[0]);read(f,part.name.data(),dims[0]);
            size_t n=dims[1];part.rest.resize(n);part.position.resize(n);part.normal.resize(n);part.uv.resize(n);
            part.triangles.resize(size_t(dims[2])*3);part.vertices.resize(n);
            read(f,part.rest.data(),n);read(f,part.normal.data(),n);read(f,part.uv.data(),n);read(f,part.triangles.data(),part.triangles.size());
            for(size_t i=0;i<n;i++) {
                for(int k=0;k<3;k++)if(!std::isfinite(part.rest[i][k]))throw std::runtime_error("nonfinite mesh position");
                for(int k=0;k<2;k++)if(!std::isfinite(part.uv[i][k]))throw std::runtime_error("nonfinite UV");
            }
            for(auto i:part.triangles)if(i>=n)throw std::runtime_error("mesh index out of bounds");
            int32_t info[3];read(bind,info,3);part.joint=info[1];part.native=info[2]!=0;
            if(info[0]!=int(n)||part.joint < -1||part.joint>3||(info[2]!=0&&info[2]!=1))throw std::runtime_error("invalid bindings");
            if(fitted)read(bind,&part.center,1);
            for(int k=0;k<3;k++)if(!std::isfinite(part.center[k]))throw std::runtime_error("invalid fitted center");
            part.ids.resize(n*6);part.weights.resize(n*6);part.offset.resize(n);
            read(bind,part.ids.data(),n*6);read(bind,part.weights.data(),n*6);read(bind,part.offset.data(),n);
            for(auto i:part.ids)if(i>=native.size())throw std::runtime_error("binding index out of bounds");
            for(size_t i=0;i<n;i++) {
                float sum=0;
                for(int k=0;k<6;k++) {
                    float weight=part.weights[i*6+k];
                    if(!std::isfinite(weight)||weight<0||weight>1.0001f)throw std::runtime_error("invalid binding weight");
                    sum+=weight;
                }
                if(part.joint<0&&std::abs(sum-1)>1e-4f)throw std::runtime_error("binding weights do not sum to one");
                for(int k=0;k<3;k++)if(!std::isfinite(part.offset[i][k]))throw std::runtime_error("invalid binding offset");
            }
            part.direct.assign(n,0);
            for(size_t i=0;i<n && part.native;i++) {
                const float* w=&part.weights[i*6];
                part.direct[i]=w[0]==1.f&&w[1]==0.f&&w[2]==0.f&&w[3]==0.f&&w[4]==0.f&&w[5]==0.f
                    &&part.offset[i].x==0.f&&part.offset[i].y==0.f&&part.offset[i].z==0.f;
            }
            part.position=part.rest;
            utils::Entity entity{};
            for(size_t i=0;i<asset->getRenderableEntityCount();i++) {
                auto e=asset->getRenderableEntities()[i];const char* name=asset->getName(e);
                if(name && part.name==name){entity=e;break;}
            }
            if(!entity)throw std::runtime_error("GLB and mesh names differ");
            part.vb=VertexBuffer::Builder().vertexCount(uint32_t(n)).bufferCount(1)
                .attribute(VertexAttribute::POSITION,0,VertexBuffer::AttributeType::FLOAT3,offsetof(Vertex,position),sizeof(Vertex))
                .attribute(VertexAttribute::TANGENTS,0,VertexBuffer::AttributeType::FLOAT4,offsetof(Vertex,tangent),sizeof(Vertex))
                .attribute(VertexAttribute::UV0,0,VertexBuffer::AttributeType::FLOAT2,offsetof(Vertex,uv),sizeof(Vertex))
                .attribute(VertexAttribute::UV1,0,VertexBuffer::AttributeType::FLOAT2,offsetof(Vertex,uv),sizeof(Vertex))
                .attribute(VertexAttribute::COLOR,0,VertexBuffer::AttributeType::FLOAT4,offsetof(Vertex,color),sizeof(Vertex)).build(*engine);
            part.ib=IndexBuffer::Builder().indexCount(uint32_t(part.triangles.size())).bufferType(IndexBuffer::IndexType::UINT).build(*engine);
            part.ib->setBuffer(*engine,upload(part.triangles.data(),part.triangles.size()*4));
            auto& rm=engine->getRenderableManager();auto instance=rm.getInstance(entity);
            rm.setGeometryAt(instance,0,RenderableManager::PrimitiveType::TRIANGLES,part.vb,part.ib);
            rm.setCulling(instance,false); // Pose-dependent bounds will replace this reference policy.
        }
        if(f.peek()!=EOF||bind.peek()!=EOF)throw std::runtime_error("unexpected trailing mesh data");
    }
    void upload_geometry() {
        std::vector<float3> shared(native.size(),float3{0});
        for(auto& part:parts) {
            std::fill(part.normal.begin(),part.normal.end(),float3{0});
            for(size_t i=0;i<part.triangles.size();i+=3) {
                auto a=part.triangles[i],b=part.triangles[i+1],c=part.triangles[i+2];
                auto normal=cross(part.position[b]-part.position[a],part.position[c]-part.position[a]);
                for(auto v:{a,b,c}) {part.normal[v]+=normal;if(part.shared(v))shared[part.ids[v*6]]+=normal;}
            }
        }
        for(auto& part:parts) {
            std::vector<float3> tangent(part.position.size(),float3{0});
            for(size_t i=0;i<part.triangles.size();i+=3) {
                auto a=part.triangles[i],b=part.triangles[i+1],c=part.triangles[i+2];
                auto u=part.uv[b]-part.uv[a],v=part.uv[c]-part.uv[a];float det=u.x*v.y-u.y*v.x;
                if(std::abs(det)>1e-10f) {
                    auto t=((part.position[b]-part.position[a])*v.y-(part.position[c]-part.position[a])*u.y)/det;
                    for(auto vertex:{a,b,c})tangent[vertex]+=t;
                }
            }
            for(size_t i=0;i<part.position.size();i++) {
                auto n=unit(part.shared(i)?shared[part.ids[i*6]]:part.normal[i]);
                auto t=tangent[i]-n*dot(n,tangent[i]);
                if(length(t)<1e-8f)t=cross(std::abs(n.y)<.9f?float3{0,1,0}:float3{1,0,0},n);
                t=unit(t);auto b=cross(n,t);
                part.vertices[i]={part.position[i],mat3f::packTangentFrame(mat3f{t,b,n}),part.uv[i]};
            }
            part.vb->setBufferAt(*engine,0,upload(part.vertices.data(),part.vertices.size()*sizeof(Vertex)));
        }
    }
};
Player::Player(const std::string& dir,void* window,uint32_t w,uint32_t h):p(new Impl) {
#ifdef __APPLE__
    p->engine=Engine::create(Engine::Backend::METAL);
#else
    p->engine=Engine::create(Engine::Backend::VULKAN);
#endif
    if(!p->engine)throw std::runtime_error("Filament engine unavailable");
    p->model=vh_mobile_load((dir+"/gnm.bin").c_str());if(!p->model)throw std::runtime_error("invalid native model");
    p->native.resize(vh_mobile_vertices(p->model));
    p->swap=window?p->engine->createSwapChain(window):p->engine->createSwapChain(w,h,SwapChain::CONFIG_READABLE);
    p->renderer=p->engine->createRenderer();p->scene=p->engine->createScene();p->view=p->engine->createView();
    p->camera_entity=utils::EntityManager::get().create();p->camera=p->engine->createCamera(p->camera_entity);
    p->materials=gltfio::createUbershaderProvider(p->engine,UBERARCHIVE_DEFAULT_DATA,UBERARCHIVE_DEFAULT_SIZE);
    p->loader=gltfio::AssetLoader::create({p->engine,p->materials,&p->names});
    std::ifstream f(dir+"/avatar.glb",std::ios::binary|std::ios::ate);
    auto length=f.tellg();if(length<=0||length>256*1024*1024)throw std::runtime_error("invalid GLB size");
    std::vector<uint8_t> bytes(static_cast<size_t>(length));f.seekg(0);read(f,bytes.data(),bytes.size());
    p->asset=p->loader->createAsset(bytes.data(),uint32_t(bytes.size()));if(!p->asset)throw std::runtime_error("invalid GLB");
    {
        std::unique_ptr<gltfio::TextureProvider> decoder(gltfio::createStbProvider(p->engine));
        gltfio::ResourceConfiguration config{};
        config.engine=p->engine;config.normalizeSkinningWeights=true;
        gltfio::ResourceLoader resources(config);
        resources.addTextureProvider("image/png",decoder.get());
        if(!resources.loadResources(p->asset))throw std::runtime_error("GLB resource loading failed");
    }
    p->scene->addEntities(p->asset->getEntities(),p->asset->getEntityCount());
    p->asset->releaseSourceData();p->load_parts(dir);p->upload_geometry();
    p->key=utils::EntityManager::get().create();p->fill=utils::EntityManager::get().create();
    LightManager::Builder(LightManager::Type::DIRECTIONAL).color({1,.96f,.9f}).intensity(65000).direction({-.4f,-.5f,-1}).castShadows(true).build(*p->engine,p->key);
    LightManager::Builder(LightManager::Type::DIRECTIONAL).color({.8f,.9f,1}).intensity(18000).direction({.7f,-.1f,-1}).build(*p->engine,p->fill);
    p->scene->addEntity(p->key);p->scene->addEntity(p->fill);
    const float3 sky{.28f,.3f,.34f};
    p->ambient=IndirectLight::Builder().irradiance(1,&sky).intensity(18000).build(*p->engine);
    p->scene->setIndirectLight(p->ambient);
    auto box=p->asset->getBoundingBox();auto center=box.center();double distance=std::max(double(box.extent().y)*3.3,.35);
    p->camera->lookAt(double3(center)+double3{0,0,distance},double3(center));p->camera->setExposure(16,1.f/125,100);
    p->view->setScene(p->scene);p->view->setCamera(p->camera);
    p->renderer->setClearOptions({{.025f,.025f,.025f,1},true,false});resize(w,h);
}
Player::~Player()=default;
void Player::resize(uint32_t w,uint32_t h) {
    if(!w||!h||w>4096||h>4096)throw std::runtime_error("invalid viewport");
    p->width=w;p->height=h;p->view->setViewport({0,0,w,h});p->camera->setProjection(35.,double(w)/h,.01,10.);
}
void Player::pose(const float* expressions,const float* rotations,const float* translation) {
    if(vh_mobile_eval(p->model,expressions,rotations,translation,reinterpret_cast<float*>(p->native.data())))throw std::runtime_error("invalid pose");
    for(auto& part:p->parts) {
        float affine[12]{},parent[12]{};
        if(part.joint>=0 && vh_mobile_joint_transform(p->model,unsigned(part.joint),affine))throw std::runtime_error("invalid joint");
        if(part.joint>=0 && vh_mobile_joint_transform(p->model,1,parent))throw std::runtime_error("invalid head joint");
        for(size_t i=0;i<part.position.size();i++) {
            if(part.joint>=0) {
                auto d=part.center,x=part.rest[i]-d;
                for(int k=0;k<3;k++)part.position[i][k]=affine[k*3]*x.x+affine[k*3+1]*x.y+affine[k*3+2]*x.z+affine[9+k]
                    +parent[k*3]*d.x+parent[k*3+1]*d.y+parent[k*3+2]*d.z;
            } else if(part.shared(i))part.position[i]=p->native[part.ids[i*6]];
            else {
                float3 root{0};for(int k=0;k<6;k++)root+=p->native[part.ids[i*6+k]]*part.weights[i*6+k];
                auto a=p->native[part.ids[i*6]],b=p->native[part.ids[i*6+1]],c=p->native[part.ids[i*6+2]];
                auto x=unit(b-a),z=unit(cross(b-a,c-a)),y=cross(z,x),d=part.offset[i];
                part.position[i]=root+x*d.x+y*d.y+z*d.z;
            }
        }
    }
    p->upload_geometry();
}
bool Player::render() {
    if(!p->renderer->beginFrame(p->swap))return false;
    p->renderer->render(p->view);p->renderer->endFrame();return true;
}
std::vector<uint8_t> Player::capture() {
    std::vector<uint8_t> pixels(size_t(p->width)*p->height*4);
    p->engine->flushAndWait();
    bool ready=false;
    for(int attempt=0;attempt<100 && !ready;attempt++) {
        ready=p->renderer->beginFrame(p->swap);
        if(!ready)std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if(!ready)throw std::runtime_error("capture frame unavailable");
    p->renderer->render(p->view);
    p->renderer->readPixels(0,0,p->width,p->height,backend::PixelBufferDescriptor(pixels.data(),pixels.size(),backend::PixelDataFormat::RGBA,backend::PixelDataType::UBYTE));
    p->renderer->endFrame();p->engine->flushAndWait();return pixels;
}
}
