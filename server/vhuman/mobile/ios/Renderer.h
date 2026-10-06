#import <Foundation/Foundation.h>
#import <QuartzCore/CAMetalLayer.h>

NS_ASSUME_NONNULL_BEGIN
@interface VHRenderer : NSObject
- (BOOL)openDirectory:(NSString *)directory layer:(CAMetalLayer *)layer
    width:(uint32_t)width height:(uint32_t)height error:(NSError **)error;
- (BOOL)pose:(NSData *)expression rotations:(NSData *)rotations
    translation:(NSData *)translation error:(NSError **)error;
- (BOOL)resizeWidth:(uint32_t)width height:(uint32_t)height error:(NSError **)error;
- (BOOL)draw:(NSError **)error;
- (void)close;
@end
NS_ASSUME_NONNULL_END
