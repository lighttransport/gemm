#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace vhuman {
// All calls belong on one render thread. On iOS native_window is CAMetalLayer*.
class Player {
public:
    Player(const std::string& directory,void* native_window,uint32_t width,uint32_t height);
    ~Player();
    void resize(uint32_t width,uint32_t height);
    void pose(const float* expression,const float* rotations,const float* translation);
    bool render();
    std::vector<uint8_t> capture();
private:
    struct Impl;
    std::unique_ptr<Impl> p;
};
}
