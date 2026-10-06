#include "player.h"
#include <fstream>
#include <iostream>
#include <array>
#include <chrono>
#include <algorithm>
int main(int argc,char** argv) {
    if(argc!=3&&argc!=4){std::cerr<<"usage: player_probe PACKAGE OUTPUT.ppm [POSE.f32]\n";return 2;}
    try {
        const unsigned w=540,h=960;vhuman::Player player(argv[1],nullptr,w,h);
        if(argc==4) {
            std::array<float,398> pose;
            std::ifstream source(argv[3],std::ios::binary);
            if(!source.read(reinterpret_cast<char*>(pose.data()),sizeof(pose))||source.peek()!=EOF)
                throw std::runtime_error("pose file must contain 383 expressions, 12 rotations and 3 translations");
            std::array<double,20> times;
            for(auto& time:times) {
                auto start=std::chrono::steady_clock::now();
                player.pose(pose.data(),pose.data()+383,pose.data()+395);
                time=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                player.render();
            }
            std::sort(times.begin(),times.end());
            std::cout<<"CPU pose + vertex upload submission median_ms="<<times[10]<<" (not GPU frame time)\n";
        }
        for(int i=0;i<3;i++)player.render();
        auto image=player.capture();
        std::ofstream f(argv[2],std::ios::binary);f<<"P6\n"<<w<<" "<<h<<"\n255\n";
        for(unsigned y=0;y<h;y++)for(unsigned x=0;x<w;x++)f.write(reinterpret_cast<char*>(image.data()+(y*w+x)*4),3);
        return f?0:1;
    } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
