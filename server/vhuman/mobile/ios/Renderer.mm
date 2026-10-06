#import "Renderer.h"
#include "../player.h"
#include <array>
#include <cstring>
#include <stdexcept>

@implementation VHRenderer {
    std::unique_ptr<vhuman::Player> _player;
}
static BOOL report(NSError **error, const std::exception& e) {
    if(error) *error=[NSError errorWithDomain:@"VHuman.Renderer" code:1
        userInfo:@{NSLocalizedDescriptionKey:@(e.what())}];
    return NO;
}
- (BOOL)openDirectory:(NSString *)directory layer:(CAMetalLayer *)layer
    width:(uint32_t)width height:(uint32_t)height error:(NSError **)error {
    try {
        _player=std::make_unique<vhuman::Player>(directory.fileSystemRepresentation,
            (__bridge void *)layer,width,height);
        return YES;
    } catch(const std::exception& e) {return report(error,e);}
}
- (BOOL)pose:(NSData *)expression rotations:(NSData *)rotations
    translation:(NSData *)translation error:(NSError **)error {
    try {
        if(!_player || expression.length!=383*4 || rotations.length!=12*4 || translation.length!=3*4)
            throw std::runtime_error("Invalid native pose dimensions");
        // NSData does not promise float alignment.
        std::array<float,383> x;std::array<float,12> r;std::array<float,3> t;
        memcpy(x.data(),expression.bytes,expression.length);
        memcpy(r.data(),rotations.bytes,rotations.length);
        memcpy(t.data(),translation.bytes,translation.length);
        _player->pose(x.data(),r.data(),t.data());return YES;
    } catch(const std::exception& e) {return report(error,e);}
}
- (BOOL)resizeWidth:(uint32_t)width height:(uint32_t)height error:(NSError **)error {
    try {
        if(!_player)throw std::runtime_error("Renderer is closed");
        _player->resize(width,height);return YES;
    } catch(const std::exception& e) {return report(error,e);}
}
- (BOOL)draw:(NSError **)error {
    try {
        if(!_player)throw std::runtime_error("Renderer is closed");
        _player->render();return YES;
    } catch(const std::exception& e) {return report(error,e);}
}
- (void)close {_player.reset();}
@end
