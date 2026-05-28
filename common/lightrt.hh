// Copyright (c) 2026 Light Transport Entertainment, Inc.
// SPDX-License-Identifier: MIT
//
// lightrt.hh - Lightweight ray tracing and BVH kernel
//
// Features:
// - Two-level BVH construction and traversal
// - SIMD optimized (SSE2, AVX, NEON, SVE)
// - Quantization and FP16 support
// - C++17, no RTTI, no exceptions

#ifndef LIGHTRT_HH_
#define LIGHTRT_HH_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <limits>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <queue>

// SIMD detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #if defined(__AVX__)
    #define LIGHTRT_HAS_AVX 1
    #include <immintrin.h>
  #elif defined(__SSE2__)
    #define LIGHTRT_HAS_SSE2 1
    #include <emmintrin.h>
  #endif
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #define LIGHTRT_HAS_NEON 1
  #include <arm_neon.h>
#endif

#if defined(__ARM_FEATURE_SVE)
  #define LIGHTRT_HAS_SVE 1
  #include <arm_sve.h>
#endif

// FP16 support detection
#if defined(__F16C__) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  #define LIGHTRT_HAS_FP16 1
#endif

namespace lightrt {

// ============================================================================
// Constants and Configuration
// ============================================================================

constexpr float kInfinity = std::numeric_limits<float>::infinity();
constexpr float kEpsilon = 1e-6f;
constexpr uint32_t kInvalidIndex = 0xFFFFFFFF;

// ============================================================================
// Task System (Simple Thread Pool)
// ============================================================================

class TaskSystem {
public:
  // Initialize with hardware concurrency
  static void initialize(uint32_t num_threads = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!threads_.empty()) return; // Already initialized

    if (num_threads == 0) num_threads = std::thread::hardware_concurrency();
    if (num_threads < 1) num_threads = 1;

    stop_ = false;
    for (uint32_t i = 0; i < num_threads; ++i) {
      threads_.emplace_back(workerThread);
    }
  }

  // Shutdown threads
  static void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : threads_) {
      if (worker.joinable()) worker.join();
    }
    threads_.clear();
  }

  // Submit a task
  static void submit(std::function<void()> task) {
    // If not initialized, run on main thread (fallback)
    if (threads_.empty()) {
      task();
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.push(std::move(task));
    }
    condition_.notify_one();
  }

  // Try to dequeue and execute one task from the queue.
  // Returns true if a task was executed, false if queue was empty.
  // Used for work-stealing: threads waiting for children help process other tasks.
  static bool tryProcessOne() {
    std::function<void()> task;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (tasks_.empty()) return false;
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
    return true;
  }

  // Parallel For Loop
  // splits range [start, end) into chunks and runs them in parallel
  static void parallelFor(uint32_t start, uint32_t end, std::function<void(uint32_t, uint32_t)> body, uint32_t min_chunk_size = 1024) {
    if (threads_.empty()) {
      body(start, end);
      return;
    }

    uint32_t range = end - start;
    if (range == 0) return;

    if (range <= min_chunk_size) {
      body(start, end);
      return;
    }

    uint32_t num_tasks = (range + min_chunk_size - 1) / min_chunk_size;
    std::atomic<uint32_t> tasks_remaining(num_tasks);

    for (uint32_t i = 0; i < num_tasks; ++i) {
      uint32_t chunk_start = start + i * min_chunk_size;
      uint32_t chunk_end = std::min(end, chunk_start + min_chunk_size);

      submit([chunk_start, chunk_end, body, &tasks_remaining]() {
        body(chunk_start, chunk_end);
        tasks_remaining.fetch_sub(1);
      });
    }

    // Help process tasks while waiting (work-stealing to prevent deadlock)
    while (tasks_remaining.load() > 0) {
      if (!tryProcessOne()) {
        // No tasks available; yield briefly and retry
        std::this_thread::yield();
      }
    }
  }

private:
  static void workerThread() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  static std::vector<std::thread> threads_;
  static std::queue<std::function<void()>> tasks_;
  static std::mutex mutex_;
  static std::condition_variable condition_;
  static bool stop_;
};

// ============================================================================
// Vector Math
// ============================================================================

struct alignas(16) Vec3 {
  float x, y, z;

  Vec3() noexcept : x(0.0f), y(0.0f), z(0.0f) {}
  Vec3(float x_, float y_, float z_) noexcept : x(x_), y(y_), z(z_) {}

  Vec3 operator+(const Vec3& v) const noexcept {
    return Vec3(x + v.x, y + v.y, z + v.z);
  }

  Vec3 operator-(const Vec3& v) const noexcept {
    return Vec3(x - v.x, y - v.y, z - v.z);
  }

  Vec3 operator*(float s) const noexcept {
    return Vec3(x * s, y * s, z * s);
  }

  Vec3 operator/(float s) const noexcept {
    float inv = 1.0f / s;
    return Vec3(x * inv, y * inv, z * inv);
  }

  float dot(const Vec3& v) const noexcept {
    return x * v.x + y * v.y + z * v.z;
  }

  Vec3 cross(const Vec3& v) const noexcept {
    return Vec3(y * v.z - z * v.y,
                z * v.x - x * v.z,
                x * v.y - y * v.x);
  }

  float length() const noexcept {
    return std::sqrt(x * x + y * y + z * z);
  }

  Vec3 normalize() const noexcept {
    float len = length();
    if (len > 0.0f) {
      return *this / len;
    }
    return *this;
  }

  // Array-style access for iteration
  float operator[](int i) const noexcept {
    return (i == 0) ? x : (i == 1) ? y : z;
  }

  float& operator[](int i) noexcept {
    return (i == 0) ? x : (i == 1) ? y : z;
  }
};

// ============================================================================
// Ray
// ============================================================================

struct alignas(16) Ray {
  Vec3 origin;
  Vec3 direction;
  float tmin;
  float tmax;

  Ray() noexcept : tmin(kEpsilon), tmax(kInfinity) {}

  Ray(const Vec3& o, const Vec3& d, float tmin_ = kEpsilon, float tmax_ = kInfinity) noexcept
    : origin(o), direction(d), tmin(tmin_), tmax(tmax_) {}

  Vec3 at(float t) const noexcept {
    return origin + direction * t;
  }
};

// Precomputed ray data for fast traversal (compute once per ray, reuse across all AABB tests)
struct alignas(32) RayContext {
  Vec3 origin;
  Vec3 inv_dir;
  float tmin;
  float tmax;
  int sign[3];  // 0 if dir >= 0, 1 if dir < 0 (for front-to-back ordering)
#if defined(LIGHTRT_HAS_AVX) || defined(LIGHTRT_HAS_SSE2)
  __m128 origin_simd;   // [ox, oy, oz, 0]
  __m128 inv_dir_simd;  // [1/dx, 1/dy, 1/dz, 0]
#endif

  RayContext() noexcept = default;

  explicit RayContext(const Ray& ray) noexcept
    : origin(ray.origin), tmin(ray.tmin), tmax(ray.tmax) {
    sign[0] = (ray.direction.x < 0.0f) ? 1 : 0;
    sign[1] = (ray.direction.y < 0.0f) ? 1 : 0;
    sign[2] = (ray.direction.z < 0.0f) ? 1 : 0;
#if defined(LIGHTRT_HAS_AVX) || defined(LIGHTRT_HAS_SSE2)
    origin_simd = _mm_set_ps(0.0f, ray.origin.z, ray.origin.y, ray.origin.x);
    // Clamp near-zero direction components to avoid NaN from rcp(0)*NR
    // IEEE754 div produces +-inf for 1/0 which slab tests handle, but rcp+NR does not
    constexpr float kMinDir = 1e-20f;
    float dx = (std::abs(ray.direction.x) < kMinDir)
             ? std::copysign(kMinDir, ray.direction.x) : ray.direction.x;
    float dy = (std::abs(ray.direction.y) < kMinDir)
             ? std::copysign(kMinDir, ray.direction.y) : ray.direction.y;
    float dz = (std::abs(ray.direction.z) < kMinDir)
             ? std::copysign(kMinDir, ray.direction.z) : ray.direction.z;
    // Fast reciprocal with Newton-Raphson refinement: ~22-bit precision, ~3x faster than _mm_div_ps
    __m128 dir_simd = _mm_set_ps(1.0f, dz, dy, dx);
    __m128 rcp = _mm_rcp_ps(dir_simd);
    // One NR iteration: rcp = rcp * (2 - dir * rcp)
    __m128 two = _mm_set1_ps(2.0f);
    inv_dir_simd = _mm_mul_ps(rcp, _mm_sub_ps(two, _mm_mul_ps(dir_simd, rcp)));
    // Store back for scalar access
    alignas(16) float inv_arr[4];
    _mm_store_ps(inv_arr, inv_dir_simd);
    inv_dir.x = inv_arr[0];
    inv_dir.y = inv_arr[1];
    inv_dir.z = inv_arr[2];
#else
    inv_dir.x = 1.0f / ray.direction.x;
    inv_dir.y = 1.0f / ray.direction.y;
    inv_dir.z = 1.0f / ray.direction.z;
#endif
  }
};

// ============================================================================
// Ray Packets for SIMD Traversal
// ============================================================================

// Ray packet of 4 rays (for SSE/NEON)
struct alignas(16) Ray4 {
  // SoA (Structure of Arrays) layout for SIMD efficiency
  float origin_x[4];
  float origin_y[4];
  float origin_z[4];
  float dir_x[4];
  float dir_y[4];
  float dir_z[4];
  float tmin[4];
  float tmax[4];
  uint32_t active_mask;  // Bit mask of active rays (0-15)

  Ray4() noexcept : active_mask(0xF) {
    for (int i = 0; i < 4; i++) {
      tmin[i] = kEpsilon;
      tmax[i] = kInfinity;
    }
  }

  // Set ray at index
  void setRay(int idx, const Ray& ray) noexcept {
    origin_x[idx] = ray.origin.x;
    origin_y[idx] = ray.origin.y;
    origin_z[idx] = ray.origin.z;
    dir_x[idx] = ray.direction.x;
    dir_y[idx] = ray.direction.y;
    dir_z[idx] = ray.direction.z;
    tmin[idx] = ray.tmin;
    tmax[idx] = ray.tmax;
  }

  // Get ray at index
  Ray getRay(int idx) const noexcept {
    return Ray(
      Vec3(origin_x[idx], origin_y[idx], origin_z[idx]),
      Vec3(dir_x[idx], dir_y[idx], dir_z[idx]),
      tmin[idx], tmax[idx]
    );
  }

  // Create from 4 rays
  static Ray4 fromRays(const Ray& r0, const Ray& r1, const Ray& r2, const Ray& r3) noexcept {
    Ray4 packet;
    packet.setRay(0, r0);
    packet.setRay(1, r1);
    packet.setRay(2, r2);
    packet.setRay(3, r3);
    return packet;
  }

  // Create from array of rays (up to 4)
  static Ray4 fromRays(const Ray* rays, int count) noexcept {
    Ray4 packet;
    packet.active_mask = (1u << count) - 1;
    for (int i = 0; i < count && i < 4; i++) {
      packet.setRay(i, rays[i]);
    }
    return packet;
  }

  // Check if ray at index is active
  bool isActive(int idx) const noexcept {
    return (active_mask & (1u << idx)) != 0;
  }

  // Set ray as inactive (hit found or terminated)
  void deactivate(int idx) noexcept {
    active_mask &= ~(1u << idx);
  }

  // Count active rays
  int countActive() const noexcept {
    int count = 0;
    uint32_t mask = active_mask;
    while (mask) {
      count += mask & 1;
      mask >>= 1;
    }
    return count;
  }
};

// Ray packet of 8 rays (for AVX)
struct alignas(32) Ray8 {
  float origin_x[8];
  float origin_y[8];
  float origin_z[8];
  float dir_x[8];
  float dir_y[8];
  float dir_z[8];
  float tmin[8];
  float tmax[8];
  uint32_t active_mask;  // Bit mask of active rays (0-255)

  Ray8() noexcept : active_mask(0xFF) {
    for (int i = 0; i < 8; i++) {
      tmin[i] = kEpsilon;
      tmax[i] = kInfinity;
    }
  }

  void setRay(int idx, const Ray& ray) noexcept {
    origin_x[idx] = ray.origin.x;
    origin_y[idx] = ray.origin.y;
    origin_z[idx] = ray.origin.z;
    dir_x[idx] = ray.direction.x;
    dir_y[idx] = ray.direction.y;
    dir_z[idx] = ray.direction.z;
    tmin[idx] = ray.tmin;
    tmax[idx] = ray.tmax;
  }

  Ray getRay(int idx) const noexcept {
    return Ray(
      Vec3(origin_x[idx], origin_y[idx], origin_z[idx]),
      Vec3(dir_x[idx], dir_y[idx], dir_z[idx]),
      tmin[idx], tmax[idx]
    );
  }

  static Ray8 fromRays(const Ray* rays, int count) noexcept {
    Ray8 packet;
    packet.active_mask = (1u << count) - 1;
    for (int i = 0; i < count && i < 8; i++) {
      packet.setRay(i, rays[i]);
    }
    return packet;
  }

  bool isActive(int idx) const noexcept {
    return (active_mask & (1u << idx)) != 0;
  }

  void deactivate(int idx) noexcept {
    active_mask &= ~(1u << idx);
  }

  int countActive() const noexcept {
    int count = 0;
    uint32_t mask = active_mask;
    while (mask) {
      count += mask & 1;
      mask >>= 1;
    }
    return count;
  }
};

// Hit result for packet traversal
struct HitResult4 {
  uint32_t prim_id[4];  // Primitive ID (kInvalidIndex if no hit)
  float t[4];           // Hit distance
  float u[4];           // Barycentric U
  float v[4];           // Barycentric V

  HitResult4() noexcept {
    for (int i = 0; i < 4; i++) {
      prim_id[i] = kInvalidIndex;
      t[i] = kInfinity;
      u[i] = 0.0f;
      v[i] = 0.0f;
    }
  }
};

struct HitResult8 {
  uint32_t prim_id[8];
  float t[8];
  float u[8];
  float v[8];

  HitResult8() noexcept {
    for (int i = 0; i < 8; i++) {
      prim_id[i] = kInvalidIndex;
      t[i] = kInfinity;
      u[i] = 0.0f;
      v[i] = 0.0f;
    }
  }
};

// Single hit record for multi-hit traversal
struct HitRecord {
  uint32_t prim_id;
  float t;
  float u;
  float v;

  HitRecord() noexcept : prim_id(kInvalidIndex), t(kInfinity), u(0), v(0) {}
  HitRecord(uint32_t id, float t_, float u_, float v_) noexcept
    : prim_id(id), t(t_), u(u_), v(v_) {}

  bool operator<(const HitRecord& other) const noexcept { return t < other.t; }
};

// Multi-hit result container (for transparency, volumetrics)
struct MultiHitResult {
  std::vector<HitRecord> hits;
  bool terminated_early;  // True if max_hits reached

  MultiHitResult() noexcept : terminated_early(false) {}

  void clear() noexcept {
    hits.clear();
    terminated_early = false;
  }

  // Sort hits by distance (front to back)
  void sort() noexcept {
    std::sort(hits.begin(), hits.end());
  }

  // Add hit maintaining sorted order
  void addSorted(const HitRecord& hit) noexcept {
    auto it = std::lower_bound(hits.begin(), hits.end(), hit);
    hits.insert(it, hit);
  }

  uint32_t count() const noexcept { return static_cast<uint32_t>(hits.size()); }
  bool empty() const noexcept { return hits.empty(); }
};

// ============================================================================
// AABB (Axis-Aligned Bounding Box)
// ============================================================================

struct alignas(16) AABB {
  Vec3 min;
  Vec3 max;

  AABB() noexcept {
    min = Vec3(kInfinity, kInfinity, kInfinity);
    max = Vec3(-kInfinity, -kInfinity, -kInfinity);
  }

  AABB(const Vec3& min_, const Vec3& max_) noexcept : min(min_), max(max_) {}

  void expand(const Vec3& p) noexcept {
    min.x = std::min(min.x, p.x);
    min.y = std::min(min.y, p.y);
    min.z = std::min(min.z, p.z);
    max.x = std::max(max.x, p.x);
    max.y = std::max(max.y, p.y);
    max.z = std::max(max.z, p.z);
  }

  void expand(const AABB& b) noexcept {
    min.x = std::min(min.x, b.min.x);
    min.y = std::min(min.y, b.min.y);
    min.z = std::min(min.z, b.min.z);
    max.x = std::max(max.x, b.max.x);
    max.y = std::max(max.y, b.max.y);
    max.z = std::max(max.z, b.max.z);
  }

  Vec3 center() const noexcept {
    return (min + max) * 0.5f;
  }

  Vec3 extents() const noexcept {
    return max - min;
  }

  float surfaceArea() const noexcept {
    Vec3 d = extents();
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
  }

  int longestAxis() const noexcept {
    Vec3 d = extents();
    if (d.x > d.y && d.x > d.z) return 0;
    if (d.y > d.z) return 1;
    return 2;
  }

  // Ray-AABB intersection test
  bool intersect(const Ray& ray, float& tmin_out, float& tmax_out) const noexcept;

  // SIMD optimized intersection
  bool intersectSIMD(const Ray& ray, float& tmin_out, float& tmax_out) const noexcept;

  // Fast AABB intersection using precomputed RayContext (no per-call reciprocal)
  bool intersectFast(const RayContext& ctx, float t_limit, float& tmin_out) const noexcept;

  // AABB-AABB intersection test
  bool intersects(const AABB& other) const noexcept {
    return (min.x <= other.max.x && max.x >= other.min.x) &&
           (min.y <= other.max.y && max.y >= other.min.y) &&
           (min.z <= other.max.z && max.z >= other.min.z);
  }

  // Squared distance from point to AABB (0 if inside)
  float distanceSquared(const Vec3& point) const noexcept {
    float dist_sq = 0.0f;
    for (int i = 0; i < 3; i++) {
      float v = point[i];
      float lo = min[i];
      float hi = max[i];
      if (v < lo) dist_sq += (lo - v) * (lo - v);
      else if (v > hi) dist_sq += (v - hi) * (v - hi);
    }
    return dist_sq;
  }

  // Swept AABB intersection (continuous collision detection)
  // Tests if this AABB moving by velocity collides with static other AABB
  // Returns true if collision occurs, with time of first contact in t_first
  // and time of last contact in t_last (for the interval [0, 1])
  bool intersectSwept(const AABB& other, const Vec3& velocity,
                      float& t_first, float& t_last) const noexcept {
    t_first = 0.0f;
    t_last = 1.0f;

    for (int i = 0; i < 3; i++) {
      float v = velocity[i];
      float a_min = min[i], a_max = max[i];
      float b_min = other.min[i], b_max = other.max[i];

      if (std::abs(v) < 1e-10f) {
        // No movement on this axis - check static overlap
        if (a_max < b_min || a_min > b_max) {
          return false;  // No overlap, no collision possible
        }
      } else {
        // Compute time intervals when AABBs overlap on this axis
        float t0 = (b_min - a_max) / v;  // Time when A's max reaches B's min
        float t1 = (b_max - a_min) / v;  // Time when A's min reaches B's max

        if (t0 > t1) std::swap(t0, t1);

        t_first = std::max(t_first, t0);
        t_last = std::min(t_last, t1);

        if (t_first > t_last || t_last < 0.0f || t_first > 1.0f) {
          return false;
        }
      }
    }

    return t_first <= t_last && t_first <= 1.0f && t_last >= 0.0f;
  }

  // Minkowski sum of two AABBs (useful for collision detection)
  AABB minkowskiSum(const AABB& other) const noexcept {
    return AABB(
      Vec3(min.x + other.min.x, min.y + other.min.y, min.z + other.min.z),
      Vec3(max.x + other.max.x, max.y + other.max.y, max.z + other.max.z)
    );
  }

  // Minkowski difference (this - other), useful for GJK-style collision
  AABB minkowskiDifference(const AABB& other) const noexcept {
    return AABB(
      Vec3(min.x - other.max.x, min.y - other.max.y, min.z - other.max.z),
      Vec3(max.x - other.min.x, max.y - other.min.y, max.z - other.min.z)
    );
  }

  // Check if point is inside AABB
  bool contains(const Vec3& point) const noexcept {
    return point.x >= min.x && point.x <= max.x &&
           point.y >= min.y && point.y <= max.y &&
           point.z >= min.z && point.z <= max.z;
  }

  // Compute penetration depth and normal for overlapping AABBs
  // Returns false if AABBs don't overlap
  bool computePenetration(const AABB& other, Vec3& normal, float& depth) const noexcept {
    if (!intersects(other)) {
      return false;
    }

    // Compute overlap on each axis
    float overlap_x = std::min(max.x, other.max.x) - std::max(min.x, other.min.x);
    float overlap_y = std::min(max.y, other.max.y) - std::max(min.y, other.min.y);
    float overlap_z = std::min(max.z, other.max.z) - std::max(min.z, other.min.z);

    // Find minimum penetration axis
    if (overlap_x <= overlap_y && overlap_x <= overlap_z) {
      depth = overlap_x;
      normal = (center().x < other.center().x) ? Vec3(-1, 0, 0) : Vec3(1, 0, 0);
    } else if (overlap_y <= overlap_z) {
      depth = overlap_y;
      normal = (center().y < other.center().y) ? Vec3(0, -1, 0) : Vec3(0, 1, 0);
    } else {
      depth = overlap_z;
      normal = (center().z < other.center().z) ? Vec3(0, 0, -1) : Vec3(0, 0, 1);
    }
    return true;
  }
};

// ============================================================================
// Collision Result Structures
// ============================================================================

// Result of a collision query between two primitives
struct CollisionPair {
  uint32_t prim_a;      // Primitive index from first BVH
  uint32_t prim_b;      // Primitive index from second BVH
  float distance_sq;    // Squared distance (0 if overlapping)

  bool operator<(const CollisionPair& other) const noexcept {
    return distance_sq < other.distance_sq;
  }
};

// Result of swept collision detection
struct SweptCollisionResult {
  uint32_t prim_a;      // Primitive index from moving BVH
  uint32_t prim_b;      // Primitive index from static BVH
  float t_first;        // Time of first contact [0, 1]
  float t_last;         // Time of last contact [0, 1]
  Vec3 normal;          // Collision normal (points from B to A)
};

// ============================================================================
// Frustum (6 planes for view frustum culling)
// ============================================================================

struct alignas(16) Frustum {
  // Plane equation: normal.x * x + normal.y * y + normal.z * z + d = 0
  // Points on positive side are inside the frustum
  struct Plane {
    Vec3 normal;
    float d;

    Plane() noexcept : normal(0, 1, 0), d(0) {}
    Plane(const Vec3& n, float dist) noexcept : normal(n), d(dist) {}
    Plane(float a, float b, float c, float dist) noexcept : normal(a, b, c), d(dist) {}

    // Distance from point to plane (positive = inside)
    float distance(const Vec3& p) const noexcept {
      return normal.dot(p) + d;
    }
  };

  // Standard frustum planes: left, right, bottom, top, near, far
  Plane planes[6];

  Frustum() noexcept = default;

  // Create frustum from view-projection matrix (column-major OpenGL style)
  // Matrix should be: projection * view
  static Frustum fromMatrix(const float m[16]) noexcept {
    Frustum f;
    // Left plane
    f.planes[0] = Plane(m[3] + m[0], m[7] + m[4], m[11] + m[8], m[15] + m[12]);
    // Right plane
    f.planes[1] = Plane(m[3] - m[0], m[7] - m[4], m[11] - m[8], m[15] - m[12]);
    // Bottom plane
    f.planes[2] = Plane(m[3] + m[1], m[7] + m[5], m[11] + m[9], m[15] + m[13]);
    // Top plane
    f.planes[3] = Plane(m[3] - m[1], m[7] - m[5], m[11] - m[9], m[15] - m[13]);
    // Near plane
    f.planes[4] = Plane(m[3] + m[2], m[7] + m[6], m[11] + m[10], m[15] + m[14]);
    // Far plane
    f.planes[5] = Plane(m[3] - m[2], m[7] - m[6], m[11] - m[10], m[15] - m[14]);

    // Normalize planes
    for (int i = 0; i < 6; i++) {
      float len = std::sqrt(f.planes[i].normal.x * f.planes[i].normal.x +
                            f.planes[i].normal.y * f.planes[i].normal.y +
                            f.planes[i].normal.z * f.planes[i].normal.z);
      if (len > 1e-10f) {
        float inv_len = 1.0f / len;
        f.planes[i].normal.x *= inv_len;
        f.planes[i].normal.y *= inv_len;
        f.planes[i].normal.z *= inv_len;
        f.planes[i].d *= inv_len;
      }
    }
    return f;
  }

  // Test if AABB is inside or intersects frustum
  // Returns: -1 = outside, 0 = intersecting, 1 = fully inside
  int testAABB(const AABB& box) const noexcept {
    int result = 1;  // Assume fully inside
    for (int i = 0; i < 6; i++) {
      // Find the corner most positive (p-vertex) and most negative (n-vertex)
      Vec3 p_vertex, n_vertex;
      for (int j = 0; j < 3; j++) {
        if (planes[i].normal[j] >= 0) {
          p_vertex[j] = box.max[j];
          n_vertex[j] = box.min[j];
        } else {
          p_vertex[j] = box.min[j];
          n_vertex[j] = box.max[j];
        }
      }

      // If p-vertex is outside (negative distance), entire box is outside
      if (planes[i].distance(p_vertex) < 0) {
        return -1;  // Outside
      }
      // If n-vertex is outside but p-vertex inside, box intersects the plane
      if (planes[i].distance(n_vertex) < 0) {
        result = 0;  // Intersecting
      }
    }
    return result;
  }
};

// ============================================================================
// KNN Result for K-nearest neighbor queries
// ============================================================================

struct KNNResult {
  uint32_t prim_id;
  float distance_sq;

  bool operator<(const KNNResult& other) const noexcept {
    return distance_sq < other.distance_sq;
  }
};

// ============================================================================
// Triangle
// ============================================================================

struct alignas(16) Triangle {
  Vec3 v0, v1, v2;

  Triangle() noexcept = default;
  Triangle(const Vec3& a, const Vec3& b, const Vec3& c) noexcept : v0(a), v1(b), v2(c) {}

  Vec3 centroid() const noexcept {
    return (v0 + v1 + v2) * (1.0f / 3.0f);
  }

  Vec3 normal() const noexcept {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    return e1.cross(e2).normalize();
  }

  AABB bounds() const noexcept;

  // Moller-Trumbore ray-triangle intersection
  // Returns true if hit, sets t, u, v (barycentric coordinates)
  bool intersect(const Ray& ray, float& t, float& u, float& v) const noexcept;

  // SoA Moller-Trumbore: test one triangle against 4 rays simultaneously
  // Input: SoA ray origins (ox,oy,oz) and directions (dx,dy,dz)
  // hit_mask_in: which rays are active (bit i = ray i)
  // Output: t_out, u_out, v_out for hits; hit_mask_out = rays that hit
  static void intersect4(const Triangle& tri,
                         const float ox[4], const float oy[4], const float oz[4],
                         const float dx[4], const float dy[4], const float dz[4],
                         const float tmin[4], const float tmax[4],
                         float t_out[4], float u_out[4], float v_out[4],
                         uint32_t hit_mask_in, uint32_t& hit_mask_out) noexcept;
};

// ============================================================================
// Quad (Bilinear Patch)
// ============================================================================

struct alignas(16) Quad {
  Vec3 v0, v1, v2, v3;  // Counter-clockwise: v0-v1-v2-v3

  Quad() noexcept = default;
  Quad(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) noexcept
    : v0(a), v1(b), v2(c), v3(d) {}

  Vec3 centroid() const noexcept {
    return (v0 + v1 + v2 + v3) * 0.25f;
  }

  AABB bounds() const noexcept;

  // Ray-quad intersection (bilinear patch)
  // Returns true if hit, sets t, u, v (parametric coordinates)
  bool intersect(const Ray& ray, float& t, float& u, float& v) const noexcept;
};

// ============================================================================
// NGon (Convex Polygon with N vertices)
// ============================================================================

struct NGon {
  std::vector<Vec3> vertices;  // Counter-clockwise vertices
  Vec3 normal;                 // Precomputed normal (for planar ngons)

  NGon() noexcept = default;
  NGon(const std::vector<Vec3>& verts) noexcept;

  void computeNormal() noexcept;
  Vec3 centroid() const noexcept;
  AABB bounds() const noexcept;

  // Ray-ngon intersection (assumes convex, planar polygon)
  bool intersect(const Ray& ray, float& t) const noexcept;
};

// ============================================================================
// Sphere
// ============================================================================

struct alignas(16) Sphere {
  Vec3 center;
  float radius;

  Sphere() noexcept : radius(1.0f) {}
  Sphere(const Vec3& c, float r) noexcept : center(c), radius(r) {}

  Vec3 centroid() const noexcept { return center; }
  AABB bounds() const noexcept;

  // Ray-sphere intersection
  // Returns true if hit, sets t (nearest hit), and optionally normal at hit
  bool intersect(const Ray& ray, float& t) const noexcept;
  bool intersect(const Ray& ray, float& t, Vec3& hit_normal) const noexcept;
};

// ============================================================================
// Disk
// ============================================================================

struct alignas(16) Disk {
  Vec3 center;
  Vec3 normal;
  float radius;

  Disk() noexcept : normal(0, 1, 0), radius(1.0f) {}
  Disk(const Vec3& c, const Vec3& n, float r) noexcept
    : center(c), normal(n.normalize()), radius(r) {}

  Vec3 centroid() const noexcept { return center; }
  AABB bounds() const noexcept;

  // Ray-disk intersection
  bool intersect(const Ray& ray, float& t) const noexcept;
};

// ============================================================================
// OrientedDisk (Screen-oriented / Billboard)
// Always faces toward a reference point (typically camera/ray origin)
// ============================================================================

struct alignas(16) OrientedDisk {
  Vec3 center;
  float radius;

  OrientedDisk() noexcept : radius(1.0f) {}
  OrientedDisk(const Vec3& c, float r) noexcept : center(c), radius(r) {}

  Vec3 centroid() const noexcept { return center; }
  AABB bounds() const noexcept;

  // Ray-oriented disk intersection (disk faces ray origin)
  bool intersect(const Ray& ray, float& t) const noexcept;
};

// ============================================================================
// Curve Types
// ============================================================================

enum class CurveType : uint8_t {
  Linear,       // Linear segments (fast, simple)
  Bezier,       // Cubic Bezier (smooth, uses Phantom algorithm)
  CatmullRom,   // Catmull-Rom spline
  BSpline       // B-spline
};

// ============================================================================
// Curve (Hair/Fiber primitive)
// Implements Phantom Ray-Hair Intersector for robust Bezier curves
// Reference: Reshetov & Luebke, HPG 2018
// ============================================================================

struct Curve {
  std::vector<Vec3> control_points;  // Control points
  std::vector<float> radii;          // Radius at each control point (for varying width)
  CurveType type;

  Curve() noexcept : type(CurveType::Bezier) {}
  Curve(const std::vector<Vec3>& points, float radius, CurveType t = CurveType::Bezier) noexcept;
  Curve(const std::vector<Vec3>& points, const std::vector<float>& r, CurveType t = CurveType::Bezier) noexcept;

  Vec3 centroid() const noexcept;
  AABB bounds() const noexcept;

  // Evaluate curve position at parameter t [0,1]
  Vec3 evaluate(float t) const noexcept;

  // Evaluate curve tangent at parameter t
  Vec3 evaluateTangent(float t) const noexcept;

  // Interpolate radius at parameter t
  float radiusAt(float t) const noexcept;

  // Ray-curve intersection
  // Returns true if hit, sets t_hit (ray parameter), u_hit (curve parameter)
  bool intersect(const Ray& ray, float& t_hit, float& u_hit) const noexcept;

private:
  // Phantom Ray-Hair algorithm for Bezier curves
  bool intersectPhantom(const Ray& ray, float& t_hit, float& u_hit) const noexcept;

  // Simple linear segment intersection (fast approximation)
  bool intersectLinear(const Ray& ray, float& t_hit, float& u_hit) const noexcept;
};

// ============================================================================
// Custom Geometry (AABB + Callback)
// For user-defined intersection routines
// ============================================================================

// Intersection callback function type
// Parameters: ray, user_data, out_t, out_u, out_v
// Returns: true if hit
using IntersectCallback = bool (*)(const Ray&, void*, float&, float&, float&);

// Bounds callback function type (optional, for dynamic bounds)
// Parameters: user_data
// Returns: AABB
using BoundsCallback = AABB (*)(void*);

struct CustomGeometry {
  AABB bounds_cache;           // Cached/static bounds
  void* user_data;             // User-provided data pointer
  IntersectCallback intersect_fn;
  BoundsCallback bounds_fn;    // Optional: for dynamic bounds

  CustomGeometry() noexcept
    : user_data(nullptr), intersect_fn(nullptr), bounds_fn(nullptr) {}

  CustomGeometry(const AABB& b, IntersectCallback fn, void* data = nullptr) noexcept
    : bounds_cache(b), user_data(data), intersect_fn(fn), bounds_fn(nullptr) {}

  Vec3 centroid() const noexcept { return bounds_cache.center(); }

  AABB bounds() const noexcept {
    if (bounds_fn) {
      return bounds_fn(user_data);
    }
    return bounds_cache;
  }

  bool intersect(const Ray& ray, float& t, float& u, float& v) const noexcept {
    if (intersect_fn) {
      return intersect_fn(ray, user_data, t, u, v);
    }
    return false;
  }
};

// ============================================================================
// Quantized Triangle (Low Memory)
// Uses 16-bit quantized coordinates relative to a global bounding box
// Memory: 18 bytes vs 36 bytes for full-precision Triangle
// ============================================================================

struct QuantizedTriangle {
  uint16_t v0[3];  // Quantized vertex 0
  uint16_t v1[3];  // Quantized vertex 1
  uint16_t v2[3];  // Quantized vertex 2

  QuantizedTriangle() noexcept = default;

  // Quantize from full-precision triangle
  void quantize(const Triangle& tri, const Vec3& global_min, const Vec3& global_max) noexcept;

  // Dequantize to full-precision triangle
  Triangle dequantize(const Vec3& global_min, const Vec3& global_max) const noexcept;

  // Direct intersection (dequantizes internally)
  bool intersect(const Ray& ray, const Vec3& global_min, const Vec3& global_max,
                 float& t, float& u, float& v) const noexcept;
};

// ============================================================================
// Gaussian Splat Primitive
// For 3D Gaussian Splatting (3DGS) ray tracing
// Reference: "3D Gaussian Ray Tracing" (SIGGRAPH Asia 2024)
// ============================================================================

// Spherical Harmonics degree for color (0=DC only, 1=4 coeffs, 2=9, 3=16)
enum class SHDegree : uint8_t {
  DC = 0,      // 1 coefficient per channel (3 total)
  Degree1 = 1, // 4 coefficients per channel (12 total)
  Degree2 = 2, // 9 coefficients per channel (27 total)
  Degree3 = 3  // 16 coefficients per channel (48 total)
};

struct GaussianSplat {
  Vec3 position;           // Center position
  Vec3 scale;              // Scale along each axis (before rotation)
  float rotation[4];       // Quaternion (w, x, y, z)
  float opacity;           // Opacity [0, 1]
  float sh_coeffs[48];     // Spherical harmonics (up to degree 3)
  SHDegree sh_degree;      // Active SH degree

  GaussianSplat() noexcept;

  // Compute covariance matrix from scale and rotation
  void getCovariance(float cov[6]) const noexcept;  // Upper triangle: xx, xy, xz, yy, yz, zz

  // Compute world-space 3x3 covariance matrix
  void getCovarianceMatrix(float mat[9]) const noexcept;

  Vec3 centroid() const noexcept { return position; }

  // Conservative AABB bounds (3-sigma ellipsoid)
  AABB bounds() const noexcept;

  // Ray-Gaussian intersection (ellipsoid approximation)
  // Returns true if ray intersects the 3-sigma confidence ellipsoid
  // t_hit: ray parameter at intersection
  // density: Gaussian density at hit point (for alpha compositing)
  bool intersect(const Ray& ray, float& t_hit, float& density) const noexcept;

  // Evaluate Gaussian density at a point
  float evaluate(const Vec3& point) const noexcept;

  // Get color from SH coefficients for a given view direction
  Vec3 getColor(const Vec3& view_dir) const noexcept;
};

// ============================================================================
// Quantized Gaussian Splat (Low Memory)
// Quantized version for large-scale scenes
// Memory: ~32 bytes vs ~220 bytes for full GaussianSplat
// ============================================================================

struct QuantizedGaussianSplat {
  uint16_t position[3];    // Quantized position (relative to scene bounds)
  uint16_t scale[3];       // Quantized log-scale
  int8_t rotation[4];      // Quantized quaternion (normalized, -127 to 127)
  uint8_t opacity;         // Quantized opacity (0-255)
  uint8_t color_dc[3];     // DC color (RGB, 0-255)
  // Optional: additional SH stored separately in SoA layout

  QuantizedGaussianSplat() noexcept = default;

  // Quantize from full-precision Gaussian
  void quantize(const GaussianSplat& gs, const Vec3& pos_min, const Vec3& pos_max,
                float scale_min, float scale_max) noexcept;

  // Dequantize to full-precision Gaussian
  GaussianSplat dequantize(const Vec3& pos_min, const Vec3& pos_max,
                           float scale_min, float scale_max) const noexcept;

  // Direct intersection (dequantizes internally for accuracy)
  bool intersect(const Ray& ray, const Vec3& pos_min, const Vec3& pos_max,
                 float scale_min, float scale_max,
                 float& t_hit, float& density) const noexcept;
};

// ============================================================================
// Primitive Variant (Type-safe union for mixed primitive BVH)
// ============================================================================

enum class PrimitiveType : uint8_t {
  Triangle,
  Quad,
  Sphere,
  Disk,
  OrientedDisk,
  Curve,
  Custom,
  QuantizedTriangle,
  GaussianSplat,
  QuantizedGaussianSplat
};

// ============================================================================
// Quantization Support
// ============================================================================

// Quantize float to 16-bit unsigned integer in range [min, max]
inline uint16_t quantizeFloat16(float value, float min_val, float max_val) noexcept {
  if (max_val - min_val < 1e-10f) return 0;
  float normalized = (value - min_val) / (max_val - min_val);
  normalized = std::max(0.0f, std::min(1.0f, normalized));
  return static_cast<uint16_t>(normalized * 65535.0f);
}

// Quantize float to 8-bit unsigned integer in range [min, max]
inline uint8_t quantizeFloat8(float value, float min_val, float max_val) noexcept {
  if (max_val - min_val < 1e-10f) return 0;
  float normalized = (value - min_val) / (max_val - min_val);
  normalized = std::max(0.0f, std::min(1.0f, normalized));
  return static_cast<uint8_t>(normalized * 255.0f);
}

// Dequantize 16-bit unsigned integer to float in range [min, max]
inline float dequantizeFloat16(uint16_t value, float min_val, float max_val) noexcept {
  float normalized = static_cast<float>(value) / 65535.0f;
  return min_val + normalized * (max_val - min_val);
}

// Dequantize 8-bit unsigned integer to float in range [min, max]
inline float dequantizeFloat8(uint8_t value, float min_val, float max_val) noexcept {
  float normalized = static_cast<float>(value) / 255.0f;
  return min_val + normalized * (max_val - min_val);
}

// Aliases for 16-bit versions (backward compatibility)
inline uint16_t quantizeFloat(float value, float min_val, float max_val) noexcept {
  return quantizeFloat16(value, min_val, max_val);
}

inline float dequantizeFloat(uint16_t value, float min_val, float max_val) noexcept {
  return dequantizeFloat16(value, min_val, max_val);
}

// Quantized AABB using 16-bit integers
struct QuantizedAABB {
  uint16_t min[3];
  uint16_t max[3];

  void quantize(const AABB& aabb, const Vec3& global_min, const Vec3& global_max) noexcept {
    min[0] = quantizeFloat(aabb.min.x, global_min.x, global_max.x);
    min[1] = quantizeFloat(aabb.min.y, global_min.y, global_max.y);
    min[2] = quantizeFloat(aabb.min.z, global_min.z, global_max.z);
    max[0] = quantizeFloat(aabb.max.x, global_min.x, global_max.x);
    max[1] = quantizeFloat(aabb.max.y, global_min.y, global_max.y);
    max[2] = quantizeFloat(aabb.max.z, global_min.z, global_max.z);
  }

  AABB dequantize(const Vec3& global_min, const Vec3& global_max) const noexcept {
    AABB result;
    result.min.x = dequantizeFloat(min[0], global_min.x, global_max.x);
    result.min.y = dequantizeFloat(min[1], global_min.y, global_max.y);
    result.min.z = dequantizeFloat(min[2], global_min.z, global_max.z);
    result.max.x = dequantizeFloat(max[0], global_min.x, global_max.x);
    result.max.y = dequantizeFloat(max[1], global_min.y, global_max.y);
    result.max.z = dequantizeFloat(max[2], global_min.z, global_max.z);
    return result;
  }
};

// ============================================================================
// Compact OBB (Oriented Bounding Box) for Leaf Filtering
// ============================================================================

struct CompactOBB {
  int8_t rotation[4];      // Quaternion (w,x,y,z) in [-127,127]
  uint8_t half_extents[3]; // Quantized to [0,255] relative to AABB half-diagonal
  uint8_t volume_ratio;    // OBB_vol / AABB_vol * 255 (0xff = skip test)
  float half_diag;         // Precomputed AABB half-diagonal length (avoids sqrt per leaf)
  // 12 bytes total

  CompactOBB() noexcept
    : rotation{127, 0, 0, 0}
    , half_extents{255, 255, 255}
    , volume_ratio(255)
    , half_diag(0.0f) {}

  // Test ray against OBB. center = AABB center, aabb_half_diag = half-diagonal length.
  // Returns true if ray potentially hits the OBB (conservative).
  bool intersectRay(const Vec3& center, float aabb_half_diag,
                    const Ray& ray, float t_limit) const noexcept;
};

// ============================================================================
// FP16 Support
// ============================================================================

// Convert float to FP16 (always available with software fallback)
inline uint16_t floatToFP16(float value) noexcept {
#if defined(__F16C__)
  __m128 v = _mm_set_ss(value);
  __m128i h = _mm_cvtps_ph(v, 0);
  return static_cast<uint16_t>(_mm_extract_epi16(h, 0));
#else
  // Software fallback for ARM NEON
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(float));

  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = bits & 0x7FFFFF;

  if (exponent <= 0) {
    // Denormal or zero
    return static_cast<uint16_t>(sign);
  } else if (exponent >= 31) {
    // Infinity or overflow
    return static_cast<uint16_t>(sign | 0x7C00);
  }

  return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
#endif
}

// Convert FP16 to float
inline float fp16ToFloat(uint16_t value) noexcept {
#if defined(__F16C__)
  __m128i h = _mm_cvtsi32_si128(value);
  __m128 v = _mm_cvtph_ps(h);
  return _mm_cvtss_f32(v);
#else
  // Software fallback for ARM NEON
  uint32_t sign = (value & 0x8000) << 16;
  int32_t exponent = (value >> 10) & 0x1F;
  uint32_t mantissa = value & 0x3FF;

  if (exponent == 0) {
    // Denormal or zero
    if (mantissa == 0) {
      uint32_t bits = sign;
      float result;
      std::memcpy(&result, &bits, sizeof(float));
      return result;
    }
    // Denormal
    exponent = 1;
    while ((mantissa & 0x400) == 0) {
      mantissa <<= 1;
      exponent--;
    }
    mantissa &= 0x3FF;
  } else if (exponent == 31) {
    // Infinity or NaN
    uint32_t bits = sign | 0x7F800000 | (mantissa << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
  }

  uint32_t bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  float result;
  std::memcpy(&result, &bits, sizeof(float));
  return result;
#endif
}

// ============================================================================
// BVH Node
// ============================================================================

struct BVHNode {
  AABB bounds;

  union {
    // Interior node
    struct {
      uint32_t left_child;   // Index to left child
      uint32_t right_child;  // Index to right child
    };

    // Leaf node
    struct {
      uint32_t prim_offset;  // Offset into primitive indices
      uint32_t prim_count;   // Number of primitives
    };
  };

  // Flags: bit 0 = is_leaf, bits 1-2 = split axis (0=x, 1=y, 2=z)
  uint32_t flags;
  uint32_t padding; // Ensure alignment

  BVHNode() noexcept : left_child(0), right_child(0), flags(0), padding(0) {}

  bool isLeaf() const noexcept {
    return (flags & 0x1) != 0;
  }

  uint32_t splitAxis() const noexcept {
    return (flags >> 1) & 0x3;
  }

  void setLeaf(uint32_t offset, uint32_t count) noexcept {
    prim_offset = offset;
    prim_count = count;
    flags |= 0x1;
  }

  void setInterior(uint32_t left, uint32_t right, uint32_t axis = 0) noexcept {
    left_child = left;
    right_child = right;
    flags = (axis & 0x3) << 1;  // Clear leaf bit, set axis
  }
};

// ============================================================================
// Build Configuration
// ============================================================================

struct BVHBuildConfig {
  uint32_t max_leaf_size;      // Maximum primitives per leaf
  uint32_t min_leaf_size;      // Minimum primitives to create leaf
  float traversal_cost;        // Cost of traversing interior node
  float intersection_cost;     // Cost of primitive intersection
  bool use_sah;                // Use Surface Area Heuristic
  bool use_binning;            // Use binned SAH for large nodes
  uint32_t num_bins;           // Number of bins for binned SAH
  bool force_max_leaf_size;    // Always enforce max_leaf_size (ignore SAH cost)
  bool use_lbvh;               // Use LBVH (Morton code-based, fast but lower quality)
  bool use_parallel_build;     // Use multi-threading for construction

  BVHBuildConfig() noexcept
    : max_leaf_size(4)
    , min_leaf_size(1)
    , traversal_cost(1.0f)
    , intersection_cost(1.0f)
    , use_sah(true)
    , use_binning(true)
    , num_bins(16)
    , force_max_leaf_size(false)
    , use_lbvh(false)
    , use_parallel_build(true) {}

  // Preset for fast build (LBVH)
  static BVHBuildConfig fast() noexcept {
    BVHBuildConfig cfg;
    cfg.use_lbvh = true;
    cfg.use_parallel_build = true;
    return cfg;
  }

  // Preset for high quality (SAH with binning)
  static BVHBuildConfig quality() noexcept {
    BVHBuildConfig cfg;
    cfg.use_sah = true;
    cfg.use_binning = true;
    cfg.use_parallel_build = true;
    return cfg;
  }
};

// ============================================================================
// SBVH Build Configuration
// ============================================================================

struct SBVHBuildConfig {
  uint32_t max_leaf_size;        // Maximum primitives per leaf
  float traversal_cost;          // Cost of traversing interior node
  float intersection_cost;       // Cost of primitive intersection
  uint32_t num_spatial_bins;     // Number of bins for spatial splits
  uint32_t num_object_bins;      // Number of bins for object splits
  float alpha;                   // Spatial split threshold (overlap ratio)
  float max_split_factor;        // Max reference count increase (e.g., 1.5 = 50% more)

  bool compute_leaf_obbs;        // Compute OBBs for leaf filtering
  float obb_volume_threshold;     // Skip OBB if volume_ratio > threshold

  SBVHBuildConfig() noexcept
    : max_leaf_size(4)
    , traversal_cost(1.0f)
    , intersection_cost(1.0f)
    , num_spatial_bins(64)
    , num_object_bins(32)
    , alpha(1e-5f)
    , max_split_factor(1.5f)
    , compute_leaf_obbs(false)
    , obb_volume_threshold(0.7f) {}
};

// ============================================================================
// Triangle Pre-Splitting Configuration
// ============================================================================

struct PreSplitConfig {
  float max_edge_ratio;           // Split if longest/shortest edge > this
  float max_aabb_looseness;       // Split if AABB SA / triangle SA > this (loose fit)
  float min_aabb_sa_fraction;     // Only split if tri AABB SA > scene AABB SA * this
  uint32_t max_split_depth;       // Max recursion depth (up to 2^depth sub-triangles)
  float max_output_factor;        // Budget cap: output <= input * this
  float sa_reduction_threshold;   // Only split if child SA sum < parent SA * this

  PreSplitConfig() noexcept
    : max_edge_ratio(10.0f)
    , max_aabb_looseness(20.0f)
    , min_aabb_sa_fraction(0.05f)
    , max_split_depth(5)
    , max_output_factor(4.0f)
    , sa_reduction_threshold(1.8f) {}

  static PreSplitConfig aggressive() noexcept {
    PreSplitConfig cfg;
    cfg.max_edge_ratio = 4.0f;
    cfg.max_aabb_looseness = 8.0f;
    cfg.min_aabb_sa_fraction = 0.01f;
    cfg.max_split_depth = 6;
    cfg.max_output_factor = 16.0f;
    cfg.sa_reduction_threshold = 1.8f;
    return cfg;
  }

  static PreSplitConfig conservative() noexcept {
    PreSplitConfig cfg;
    cfg.max_edge_ratio = 20.0f;
    cfg.max_aabb_looseness = 40.0f;
    cfg.min_aabb_sa_fraction = 0.1f;
    cfg.max_split_depth = 3;
    cfg.max_output_factor = 2.0f;
    cfg.sa_reduction_threshold = 1.8f;
    return cfg;
  }
};

struct PreSplitResult {
  std::vector<Triangle> triangles;
  std::vector<uint32_t> original_indices;  // new_id -> original_id mapping
  uint32_t num_split;                      // How many originals were subdivided
  uint32_t num_original;
};

PreSplitResult preSplitTriangles(const std::vector<Triangle>& triangles,
                                  const PreSplitConfig& config = PreSplitConfig()) noexcept;

// ============================================================================
// Traversal Configuration (for limiting primitive tests)
// ============================================================================

struct TraversalConfig {
  uint32_t max_prim_tests;       // Maximum primitive intersection tests (0 = unlimited)
  uint32_t exclude_prim_id;      // Primitive to skip (for self-intersection avoidance)
  bool use_mailboxing;           // Use mailboxing to avoid duplicate tests (for SBVH)
  bool early_termination;        // Stop on first hit (any-hit query)

  TraversalConfig() noexcept
    : max_prim_tests(0)
    , exclude_prim_id(kInvalidIndex)
    , use_mailboxing(false)
    , early_termination(false) {}

  // Preset for fast approximate traversal
  static TraversalConfig fast(uint32_t max_tests = 64) noexcept {
    TraversalConfig cfg;
    cfg.max_prim_tests = max_tests;
    cfg.use_mailboxing = true;
    return cfg;
  }

  // Preset for any-hit (shadow rays)
  static TraversalConfig anyHit() noexcept {
    TraversalConfig cfg;
    cfg.early_termination = true;
    return cfg;
  }

  // Preset for shadow ray from a surface (any-hit + self-intersection avoidance)
  static TraversalConfig shadowRay(uint32_t exclude_prim = kInvalidIndex) noexcept {
    TraversalConfig cfg;
    cfg.early_termination = true;
    cfg.exclude_prim_id = exclude_prim;
    return cfg;
  }

  // Preset for reflection/refraction ray (self-intersection avoidance)
  static TraversalConfig secondaryRay(uint32_t exclude_prim) noexcept {
    TraversalConfig cfg;
    cfg.exclude_prim_id = exclude_prim;
    return cfg;
  }
};

// Traversal statistics
struct TraversalStats {
  uint32_t nodes_visited;
  uint32_t prims_tested;
  uint32_t prims_hit;
  bool terminated_early;         // Hit max_prim_tests limit

  TraversalStats() noexcept
    : nodes_visited(0)
    , prims_tested(0)
    , prims_hit(0)
    , terminated_early(false) {}
};

// Simple mailbox using bitset for small primitive counts, or hash set for large
class Mailbox {
public:
  explicit Mailbox(uint32_t num_prims) noexcept : num_prims_(num_prims) {
    if (num_prims <= kBitsetThreshold) {
      bitset_.resize((num_prims + 63) / 64, 0);
    } else {
      // Use simple hash table for large primitive counts
      hash_table_.resize(std::max(num_prims / 4, 256u), kInvalidIndex);
    }
  }

  // Returns true if primitive was already tested, marks it as tested
  bool testAndMark(uint32_t prim_id) noexcept {
    if (num_prims_ <= kBitsetThreshold) {
      uint32_t word = prim_id / 64;
      uint64_t bit = 1ULL << (prim_id % 64);
      if (bitset_[word] & bit) {
        return true;  // Already tested
      }
      bitset_[word] |= bit;
      return false;
    } else {
      // Linear probing hash table
      uint32_t idx = prim_id % hash_table_.size();
      for (uint32_t i = 0; i < hash_table_.size(); i++) {
        uint32_t probe = (idx + i) % hash_table_.size();
        if (hash_table_[probe] == prim_id) {
          return true;  // Already tested
        }
        if (hash_table_[probe] == kInvalidIndex) {
          hash_table_[probe] = prim_id;
          return false;
        }
      }
      return false;  // Table full, test anyway
    }
  }

  void clear() noexcept {
    if (num_prims_ <= kBitsetThreshold) {
      std::fill(bitset_.begin(), bitset_.end(), 0);
    } else {
      std::fill(hash_table_.begin(), hash_table_.end(), kInvalidIndex);
    }
  }

private:
  static constexpr uint32_t kBitsetThreshold = 65536;
  uint32_t num_prims_;
  std::vector<uint64_t> bitset_;
  std::vector<uint32_t> hash_table_;
};

// ============================================================================
// BVH Builder
// ============================================================================

class BVH {
public:
  BVH() noexcept = default;
  ~BVH() noexcept = default;

  // Move operations
  BVH(BVH&& other) noexcept
    : nodes_(std::move(other.nodes_))
    , prim_indices_(std::move(other.prim_indices_))
    , prim_aabbs_(std::move(other.prim_aabbs_))
    , config_(other.config_)
    , node_allocator_(other.node_allocator_.load()) {}

  BVH& operator=(BVH&& other) noexcept {
    if (this != &other) {
      nodes_ = std::move(other.nodes_);
      prim_indices_ = std::move(other.prim_indices_);
      prim_aabbs_ = std::move(other.prim_aabbs_);
      config_ = other.config_;
      node_allocator_.store(other.node_allocator_.load());
    }
    return *this;
  }

  // Delete copy operations
  BVH(const BVH&) = delete;
  BVH& operator=(const BVH&) = delete;

  // Build BVH from primitives (AABBs)
  // prim_aabbs: Bounding boxes of primitives
  // Returns true on success
  bool build(const std::vector<AABB>& prim_aabbs, const BVHBuildConfig& config = BVHBuildConfig()) noexcept;

  // Traverse BVH and find closest intersection
  // Returns primitive index or kInvalidIndex if no hit
  uint32_t traverse(const Ray& ray, float& hit_t) const noexcept;

  // Traverse BVH using SIMD optimizations
  uint32_t traverseSIMD(const Ray& ray, float& hit_t) const noexcept;

  // Get BVH statistics
  struct Stats {
    uint32_t num_nodes;
    uint32_t num_leaves;
    uint32_t max_depth;
    float avg_leaf_size;
    float sah_cost;
  };

  Stats getStats() const noexcept;

  // Refit BVH bounds from updated primitive AABBs
  // Tree structure remains unchanged, only bounds are updated (bottom-up)
  void refit(const std::vector<AABB>& new_prim_aabbs) noexcept;

  // Frustum culling: collect primitives that intersect an AABB
  // Returns indices of primitives whose AABBs overlap with query_aabb
  void queryAABB(const AABB& query_aabb, std::vector<uint32_t>& results) const noexcept;

  // Sphere query: collect primitives within sphere radius
  void querySphere(const Vec3& center, float radius, std::vector<uint32_t>& results) const noexcept;

  // Frustum culling: collect primitives visible in view frustum
  void queryFrustum(const Frustum& frustum, std::vector<uint32_t>& results) const noexcept;

  // K-nearest neighbor query: find K closest primitives to a point
  // Results are sorted by distance (nearest first)
  void queryKNN(const Vec3& point, uint32_t k, std::vector<KNNResult>& results) const noexcept;

  // Nearest primitive query: find closest primitive to a point
  // Returns kInvalidIndex if BVH is empty
  uint32_t queryNearest(const Vec3& point, float& distance_sq) const noexcept;

  // =========================================================================
  // Collision Detection
  // =========================================================================

  // Find all colliding primitive pairs between this BVH and another
  // Returns pairs where primitive AABBs overlap
  void findCollisions(const BVH& other, std::vector<CollisionPair>& pairs) const noexcept;

  // Find colliding pairs with distance threshold (broad-phase + narrow-phase)
  // max_distance: maximum separation to consider as "near collision"
  void findCollisions(const BVH& other, float max_distance,
                      std::vector<CollisionPair>& pairs) const noexcept;

  // Self-collision detection: find all overlapping primitive pairs within this BVH
  void findSelfCollisions(std::vector<CollisionPair>& pairs) const noexcept;

  // Swept collision: find first collision as this BVH moves by velocity
  // Returns true if collision found, with result containing contact info
  bool findSweptCollision(const BVH& other, const Vec3& velocity,
                          SweptCollisionResult& result) const noexcept;

  // Find all swept collisions (not just the first)
  void findAllSweptCollisions(const BVH& other, const Vec3& velocity,
                              std::vector<SweptCollisionResult>& results) const noexcept;

  // Test if any collision exists (fast early-out)
  bool hasCollision(const BVH& other) const noexcept;

  // Test if any self-collision exists
  bool hasSelfCollision() const noexcept;

  // Access to nodes (for serialization, etc.)
  const std::vector<BVHNode>& getNodes() const noexcept { return nodes_; }
  std::vector<BVHNode>& getMutableNodes() noexcept { return nodes_; }
  const std::vector<uint32_t>& getPrimitiveIndices() const noexcept { return prim_indices_; }

private:
  std::vector<BVHNode> nodes_;
  std::vector<uint32_t> prim_indices_;
  std::vector<AABB> prim_aabbs_;
  BVHBuildConfig config_;
  std::atomic<uint32_t> node_allocator_;

  // Recursive build
  uint32_t buildRecursive(
    uint32_t* indices,
    uint32_t num_prims,
    uint32_t depth,
    const AABB* precomputed_bounds = nullptr) noexcept;

  // Split methods
  struct SplitResult {
    int axis;
    float pos;
    float cost;
    AABB left_bounds;   // Precomputed left child bounds
    AABB right_bounds;  // Precomputed right child bounds
  };

  SplitResult findBestSplit(
    const uint32_t* indices,
    uint32_t num_prims,
    const AABB& centroid_bounds,
    float parent_area) noexcept;

  SplitResult findBestSplitBinned(
    const uint32_t* indices,
    uint32_t num_prims,
    const AABB& centroid_bounds,
    float parent_area) noexcept;

  // LBVH (Linear BVH) construction
  struct MortonPrimitive {
    uint32_t prim_idx;
    uint32_t morton_code;
  };

  uint32_t buildLBVH(MortonPrimitive* morton_prims, uint32_t num_prims,
                     uint32_t bit, const AABB& scene_bounds) noexcept;
  uint32_t computeMortonCode(const Vec3& p, const AABB& bounds) const noexcept;
};

// ============================================================================
// BVH4 - Wide BVH (4 children per node)
// ============================================================================

// Standard FP32 BVH4 Node (SoA layout for SIMD)
struct alignas(16) BVH4Node {
  float min_x[4], min_y[4], min_z[4];
  float max_x[4], max_y[4], max_z[4];
  uint32_t children[4]; // If MSB is 1, it's a leaf (offset in lower bits)
  uint32_t counts[4];   // Primitive count if leaf, 0 if interior

  bool isLeaf(int i) const noexcept { return (children[i] & 0x80000000) != 0; }
  uint32_t getOffset(int i) const noexcept { return children[i] & 0x7FFFFFFF; }
  void setLeaf(int i, uint32_t offset, uint32_t count) noexcept {
    children[i] = offset | 0x80000000;
    counts[i] = count;
  }
  void setInterior(int i, uint32_t child_idx) noexcept {
    children[i] = child_idx;
    counts[i] = 0;
  }
};

// FP16 Quantized BVH4 Node
struct alignas(16) BVH4NodeFP16 {
  uint16_t min_x[4], min_y[4], min_z[4];
  uint16_t max_x[4], max_y[4], max_z[4];
  uint32_t children[4];
  uint32_t counts[4];
};

// Int16 Quantized BVH4 Node
struct alignas(16) BVH4NodeInt16 {
  uint16_t min_x[4], min_y[4], min_z[4];
  uint16_t max_x[4], max_y[4], max_z[4];
  float reference_min[3];
  float reference_max[3];
  uint32_t children[4];
  uint32_t counts[4];
};

// Int8 Quantized BVH4 Node
struct alignas(16) BVH4NodeInt8 {
  uint8_t min_x[4], min_y[4], min_z[4];
  uint8_t max_x[4], max_y[4], max_z[4];
  float reference_min[3];
  float reference_max[3];
  uint32_t children[4];
  uint32_t counts[4];
};

enum class BVH4Precision : uint8_t {
  FP32,
  FP16,
  Int16,
  Int8
};

class BVH4 {
public:
  BVH4() noexcept = default;
  ~BVH4() noexcept = default;

  // Build BVH4 by collapsing a binary BVH
  bool build(const BVH& binary_bvh, const std::vector<AABB>& prim_aabbs, BVH4Precision precision = BVH4Precision::FP32) noexcept;

  // Traverse BVH4 using SIMD
  uint32_t traverse(const Ray& ray, float& hit_t) const noexcept;

  // Access internals
  const std::vector<BVH4Node>& getNodesFP32() const noexcept { return nodes_fp32_; }
  const std::vector<BVH4NodeFP16>& getNodesFP16() const noexcept { return nodes_fp16_; }
  const std::vector<BVH4NodeInt16>& getNodesInt16() const noexcept { return nodes_int16_; }
  const std::vector<BVH4NodeInt8>& getNodesInt8() const noexcept { return nodes_int8_; }
  const std::vector<uint32_t>& getPrimitiveIndices() const noexcept { return prim_indices_; }

private:
  std::vector<BVH4Node> nodes_fp32_;
  std::vector<BVH4NodeFP16> nodes_fp16_;
  std::vector<BVH4NodeInt16> nodes_int16_;
  std::vector<BVH4NodeInt8> nodes_int8_;
  std::vector<uint32_t> prim_indices_;
  std::vector<AABB> prim_aabbs_;
  BVH4Precision precision_ = BVH4Precision::FP32;

  // Collapse helper
  struct CollapseResult {
    uint32_t child_indices[4];
    AABB child_bounds[4];
    int num_children;
  };

  CollapseResult collapseBinaryNode(const BVH& binary_bvh, uint32_t binary_idx) const noexcept;
  uint32_t buildRecursive(const BVH& binary_bvh, uint32_t binary_idx) noexcept;

  void quantizeNodes() noexcept;
};

// ============================================================================
// Two-Level BVH (TLAS + BLAS)
// ============================================================================

// Bottom-Level Acceleration Structure
struct BLAS {
  BVH bvh;
  std::vector<AABB> primitives;  // Primitive AABBs

  bool build(const std::vector<AABB>& prim_aabbs, const BVHBuildConfig& config = BVHBuildConfig()) noexcept {
    primitives = prim_aabbs;
    return bvh.build(prim_aabbs, config);
  }
};

// Instance of a BLAS with transformation
struct BLASInstance {
  uint32_t blas_id;        // Index into BLAS array
  float transform[12];     // 3x4 transformation matrix (row-major)
  float inv_transform[12]; // Inverse transformation
  AABB bounds;             // Transformed world-space bounds

  BLASInstance() noexcept : blas_id(kInvalidIndex) {
    // Identity transform
    std::memset(transform, 0, sizeof(transform));
    std::memset(inv_transform, 0, sizeof(inv_transform));
    transform[0] = transform[5] = transform[10] = 1.0f;
    inv_transform[0] = inv_transform[5] = inv_transform[10] = 1.0f;
  }

  // Transform point from world to local space
  Vec3 worldToLocal(const Vec3& p) const noexcept {
    return Vec3(
      inv_transform[0] * p.x + inv_transform[1] * p.y + inv_transform[2] * p.z + inv_transform[3],
      inv_transform[4] * p.x + inv_transform[5] * p.y + inv_transform[6] * p.z + inv_transform[7],
      inv_transform[8] * p.x + inv_transform[9] * p.y + inv_transform[10] * p.z + inv_transform[11]
    );
  }

  // Transform direction from world to local space
  Vec3 worldToLocalDir(const Vec3& d) const noexcept {
    return Vec3(
      inv_transform[0] * d.x + inv_transform[1] * d.y + inv_transform[2] * d.z,
      inv_transform[4] * d.x + inv_transform[5] * d.y + inv_transform[6] * d.z,
      inv_transform[8] * d.x + inv_transform[9] * d.y + inv_transform[10] * d.z
    );
  }
};

// Top-Level Acceleration Structure
class TLAS {
public:
  TLAS() noexcept = default;
  ~TLAS() noexcept = default;

  // Build TLAS from BLAS instances
  bool build(const std::vector<BLASInstance>& instances, const BVHBuildConfig& config = BVHBuildConfig()) noexcept;

  // Traverse TLAS and find closest intersection
  // Returns instance index and primitive index, or kInvalidIndex if no hit
  struct TraceResult {
    uint32_t instance_id;
    uint32_t primitive_id;
    float t;
  };

  TraceResult trace(const Ray& ray, const std::vector<BLAS>& blas_array) const noexcept;

  const BVH& getBVH() const noexcept { return bvh_; }
  const std::vector<BLASInstance>& getInstances() const noexcept { return instances_; }

private:
  BVH bvh_;
  std::vector<BLASInstance> instances_;
};

// ============================================================================
// Triangle BVH - BVH over triangles with proper ray-triangle intersection
// ============================================================================

class TriangleBVH {
public:
  TriangleBVH() noexcept = default;
  ~TriangleBVH() noexcept = default;

  // Move operations
  TriangleBVH(TriangleBVH&& other) noexcept
    : bvh_(std::move(other.bvh_))
    , triangles_(std::move(other.triangles_)) {}

  TriangleBVH& operator=(TriangleBVH&& other) noexcept {
    if (this != &other) {
      bvh_ = std::move(other.bvh_);
      triangles_ = std::move(other.triangles_);
    }
    return *this;
  }

  // Delete copy
  TriangleBVH(const TriangleBVH&) = delete;
  TriangleBVH& operator=(const TriangleBVH&) = delete;

  // Build BVH from triangles
  bool build(const std::vector<Triangle>& triangles, const BVHBuildConfig& config = BVHBuildConfig()) noexcept;

  // Traverse and find closest triangle intersection
  // Returns triangle index or kInvalidIndex if no hit
  // hit_t: distance to hit, hit_u/hit_v: barycentric coordinates
  uint32_t traverse(const Ray& ray, float& hit_t, float& hit_u, float& hit_v) const noexcept;

  // Traverse with configuration (max prim tests, early termination)
  uint32_t traverseWithConfig(const Ray& ray, float& hit_t, float& hit_u, float& hit_v,
                              const TraversalConfig& config,
                              TraversalStats* stats = nullptr) const noexcept;

  // Get statistics
  BVH::Stats getStats() const noexcept { return bvh_.getStats(); }
  uint32_t getNumPrimitives() const noexcept { return static_cast<uint32_t>(triangles_.size()); }

  // Any-hit traversal (for shadow rays) - returns true if any hit found
  // exclude_prim_id: skip this primitive (for self-intersection avoidance)
  bool traverseAnyHit(const Ray& ray, uint32_t exclude_prim_id = kInvalidIndex) const noexcept;

  // Packet traversal - traverse 4 rays in parallel
  // Uses SIMD for coherent ray processing
  void traverse4(const Ray4& rays, HitResult4& results) const noexcept;

  // Any-hit packet traversal - returns bit mask of rays that hit something
  uint32_t traverse4AnyHit(const Ray4& rays, uint32_t exclude_prim_id = kInvalidIndex) const noexcept;

  // 8-ray packet traversal
  void traverse8(const Ray8& rays, HitResult8& results) const noexcept;

  // Any-hit 8-ray packet - returns bit mask of rays that hit something
  uint32_t traverse8AnyHit(const Ray8& rays, uint32_t exclude_prim_id = kInvalidIndex) const noexcept;

  // Multi-hit traversal (for transparency, volumetrics)
  // Collects all hits along the ray up to max_hits
  // Returns number of hits found
  uint32_t traverseMultiHit(const Ray& ray, MultiHitResult& result,
                            uint32_t max_hits = 16,
                            uint32_t exclude_prim_id = kInvalidIndex) const noexcept;

  // Refit BVH bounds after geometry modification (for animation)
  // Updates all node bounds bottom-up without rebuilding tree structure
  // Call after modifying triangles via getMutableTriangles()
  void refit() noexcept;

  // Spatial queries for culling and collision detection
  void queryAABB(const AABB& query_aabb, std::vector<uint32_t>& triangle_indices) const noexcept;
  void querySphere(const Vec3& center, float radius, std::vector<uint32_t>& triangle_indices) const noexcept;
  void queryFrustum(const Frustum& frustum, std::vector<uint32_t>& triangle_indices) const noexcept;
  void queryKNN(const Vec3& point, uint32_t k, std::vector<KNNResult>& results) const noexcept;
  uint32_t queryNearest(const Vec3& point, float& distance_sq) const noexcept;

  // Collision detection
  void findCollisions(const TriangleBVH& other, std::vector<CollisionPair>& pairs) const noexcept;
  void findCollisions(const TriangleBVH& other, float max_distance,
                      std::vector<CollisionPair>& pairs) const noexcept;
  void findSelfCollisions(std::vector<CollisionPair>& pairs) const noexcept;
  bool findSweptCollision(const TriangleBVH& other, const Vec3& velocity,
                          SweptCollisionResult& result) const noexcept;
  void findAllSweptCollisions(const TriangleBVH& other, const Vec3& velocity,
                              std::vector<SweptCollisionResult>& results) const noexcept;
  bool hasCollision(const TriangleBVH& other) const noexcept;
  bool hasSelfCollision() const noexcept;

  // Serialization - save/load BVH to binary format
  // Format version is embedded for forward compatibility
  bool save(const char* filename) const noexcept;
  bool load(const char* filename) noexcept;

  // Serialization to memory buffer
  bool saveToMemory(std::vector<uint8_t>& buffer) const noexcept;
  bool loadFromMemory(const uint8_t* data, size_t size) noexcept;

  // Batch traversal: trace many rays in parallel across threads
  void traverseBatch(const Ray* rays, uint32_t num_rays,
                     uint32_t* hit_prim_ids, float* hit_ts,
                     float* hit_us, float* hit_vs,
                     uint32_t num_threads = 0) const noexcept;

  void traverseBatchAnyHit(const Ray* rays, uint32_t num_rays,
                           bool* hit_results,
                           uint32_t exclude_prim_id = kInvalidIndex,
                           uint32_t num_threads = 0) const noexcept;

  // Access internals
  const BVH& getBVH() const noexcept { return bvh_; }
  const std::vector<Triangle>& getTriangles() const noexcept { return triangles_; }
  std::vector<Triangle>& getMutableTriangles() noexcept { return triangles_; }

private:
  BVH bvh_;
  std::vector<Triangle> triangles_;
};

// ============================================================================
// SBVH (Split BVH) - BVH with Spatial Splits
// Reference: "Spatial Splits in Bounding Volume Hierarchies" (Stich et al., HPG 2009)
// ============================================================================

class SBVH {
public:
  // Primitive reference with clipped bounds
  struct PrimRef {
    uint32_t prim_id;    // Original primitive index
    AABB bounds;         // Clipped bounding box (may be smaller than original)

    PrimRef() noexcept : prim_id(kInvalidIndex) {}
    PrimRef(uint32_t id, const AABB& b) noexcept : prim_id(id), bounds(b) {}
  };

  SBVH() noexcept = default;
  ~SBVH() noexcept = default;

  // Build SBVH from triangles
  bool build(const std::vector<Triangle>& triangles,
             const SBVHBuildConfig& config = SBVHBuildConfig()) noexcept;

  // Traverse and find closest triangle intersection
  uint32_t traverse(const Ray& ray, float& hit_t, float& hit_u, float& hit_v) const noexcept;

  // Traverse with configuration (max prim tests, mailboxing, early termination)
  // Mailboxing avoids testing same primitive multiple times (important for SBVH)
  uint32_t traverseWithConfig(const Ray& ray, float& hit_t, float& hit_u, float& hit_v,
                              const TraversalConfig& config,
                              TraversalStats* stats = nullptr) const noexcept;

  // Get statistics
  struct Stats {
    uint32_t num_nodes;
    uint32_t num_leaves;
    uint32_t max_depth;
    float avg_leaf_size;
    float sah_cost;
    uint32_t num_references;     // Total references (>= num_primitives due to splits)
    uint32_t num_primitives;     // Original primitive count
    float split_ratio;           // num_references / num_primitives
  };

  Stats getStats() const noexcept;
  uint32_t getNumPrimitives() const noexcept { return static_cast<uint32_t>(triangles_.size()); }

  // Any-hit traversal (for shadow rays) - returns true if any hit found
  bool traverseAnyHit(const Ray& ray, uint32_t exclude_prim_id = kInvalidIndex) const noexcept;

  // Packet traversal
  void traverse4(const Ray4& rays, HitResult4& results) const noexcept;
  uint32_t traverse4AnyHit(const Ray4& rays, uint32_t exclude_prim_id = kInvalidIndex) const noexcept;
  void traverse8(const Ray8& rays, HitResult8& results) const noexcept;
  uint32_t traverse8AnyHit(const Ray8& rays, uint32_t exclude_prim_id = kInvalidIndex) const noexcept;

  // Multi-hit traversal (for transparency, volumetrics)
  uint32_t traverseMultiHit(const Ray& ray, MultiHitResult& result,
                            uint32_t max_hits = 16,
                            uint32_t exclude_prim_id = kInvalidIndex) const noexcept;

  // Batch traversal: trace many rays in parallel across threads
  void traverseBatch(const Ray* rays, uint32_t num_rays,
                     uint32_t* hit_prim_ids, float* hit_ts,
                     float* hit_us, float* hit_vs,
                     uint32_t num_threads = 0) const noexcept;

  void traverseBatchAnyHit(const Ray* rays, uint32_t num_rays,
                           bool* hit_results,
                           uint32_t exclude_prim_id = kInvalidIndex,
                           uint32_t num_threads = 0) const noexcept;

  // Access internals
  const std::vector<BVHNode>& getNodes() const noexcept { return nodes_; }
  const std::vector<PrimRef>& getReferences() const noexcept { return refs_; }
  const std::vector<Triangle>& getTriangles() const noexcept { return triangles_; }

  // OBB leaf filtering
  void computeLeafOBBs(float volume_threshold = 0.7f) noexcept;
  bool hasOBBFiltering() const noexcept { return use_obb_filtering_; }

private:
  std::vector<BVHNode> nodes_;
  std::vector<PrimRef> refs_;          // Leaf references (may have duplicates)
  std::vector<Triangle> triangles_;    // Original triangles
  SBVHBuildConfig config_;
  AABB scene_bounds_;

  // OBB leaf filtering data
  std::vector<CompactOBB> leaf_obbs_;
  bool use_obb_filtering_{false};

  // Timestamp-based mailbox for duplicate avoidance in traversal
  mutable std::vector<uint32_t> prim_timestamps_;  // Per-primitive last-tested ray ID
  mutable uint32_t ray_counter_{0};                 // Monotonic ray counter

  // Split result types
  enum class SplitType { Object, Spatial };

  struct SplitResult {
    SplitType type;
    int axis;
    float pos;
    float cost;
    AABB left_bounds;
    AABB right_bounds;
    uint32_t left_count;
    uint32_t right_count;
  };

  // Spatial bin for binned split finding
  struct SpatialBin {
    AABB bounds;
    uint32_t enter;   // Primitives entering this bin
    uint32_t exit;    // Primitives exiting this bin

    SpatialBin() noexcept : enter(0), exit(0) {}
  };

  // Object bin
  struct ObjectBin {
    AABB bounds;
    uint32_t count;

    ObjectBin() noexcept : count(0) {}
  };

  // Build methods
  uint32_t buildRecursive(std::vector<PrimRef>& refs, uint32_t depth) noexcept;

  // Split finding
  SplitResult findObjectSplit(const std::vector<PrimRef>& refs,
                              const AABB& node_bounds,
                              const AABB& centroid_bounds) noexcept;

  SplitResult findSpatialSplit(const std::vector<PrimRef>& refs,
                               const AABB& node_bounds) noexcept;

  // Perform the split
  void performObjectSplit(std::vector<PrimRef>& refs,
                          const SplitResult& split,
                          std::vector<PrimRef>& left_refs,
                          std::vector<PrimRef>& right_refs) noexcept;

  void performSpatialSplit(std::vector<PrimRef>& refs,
                           const SplitResult& split,
                           std::vector<PrimRef>& left_refs,
                           std::vector<PrimRef>& right_refs) noexcept;

  // Clip triangle AABB to a half-space
  AABB clipTriangleToPlane(const Triangle& tri, int axis, float pos, bool left) const noexcept;

  // Compute overlap between two AABBs
  float computeOverlap(const AABB& a, const AABB& b) const noexcept;
};

// ============================================================================
// SBVH for generic primitives (AABB-based)
// ============================================================================

class SBVHGeneric {
public:
  // Primitive reference with clipped bounds
  struct PrimRef {
    uint32_t prim_id;
    AABB bounds;

    PrimRef() noexcept : prim_id(kInvalidIndex) {}
    PrimRef(uint32_t id, const AABB& b) noexcept : prim_id(id), bounds(b) {}
  };

  SBVHGeneric() noexcept = default;
  ~SBVHGeneric() noexcept = default;

  // Build SBVH from primitive AABBs
  // prim_aabbs: Bounding boxes of primitives (used for both bounds and spatial clipping)
  bool build(const std::vector<AABB>& prim_aabbs,
             const SBVHBuildConfig& config = SBVHBuildConfig()) noexcept;

  // Traverse and find closest intersection (returns primitive index)
  uint32_t traverse(const Ray& ray, float& hit_t) const noexcept;

  // Get statistics
  struct Stats {
    uint32_t num_nodes;
    uint32_t num_leaves;
    uint32_t max_depth;
    float avg_leaf_size;
    float sah_cost;
    uint32_t num_references;
    uint32_t num_primitives;
    float split_ratio;
  };

  Stats getStats() const noexcept;

  const std::vector<BVHNode>& getNodes() const noexcept { return nodes_; }
  const std::vector<PrimRef>& getReferences() const noexcept { return refs_; }

private:
  std::vector<BVHNode> nodes_;
  std::vector<PrimRef> refs_;
  std::vector<AABB> prim_aabbs_;
  SBVHBuildConfig config_;
  AABB scene_bounds_;

  enum class SplitType { Object, Spatial };

  struct SplitResult {
    SplitType type;
    int axis;
    float pos;
    float cost;
    AABB left_bounds;
    AABB right_bounds;
    uint32_t left_count;
    uint32_t right_count;
  };

  struct SpatialBin {
    AABB bounds;
    uint32_t enter;
    uint32_t exit;
    SpatialBin() noexcept : enter(0), exit(0) {}
  };

  struct ObjectBin {
    AABB bounds;
    uint32_t count;
    ObjectBin() noexcept : count(0) {}
  };

  uint32_t buildRecursive(std::vector<PrimRef>& refs, uint32_t depth) noexcept;

  SplitResult findObjectSplit(const std::vector<PrimRef>& refs,
                              const AABB& node_bounds,
                              const AABB& centroid_bounds) noexcept;

  SplitResult findSpatialSplit(const std::vector<PrimRef>& refs,
                               const AABB& node_bounds) noexcept;

  void performObjectSplit(std::vector<PrimRef>& refs,
                          const SplitResult& split,
                          std::vector<PrimRef>& left_refs,
                          std::vector<PrimRef>& right_refs) noexcept;

  void performSpatialSplit(std::vector<PrimRef>& refs,
                           const SplitResult& split,
                           std::vector<PrimRef>& left_refs,
                           std::vector<PrimRef>& right_refs) noexcept;

  AABB clipAABBToPlane(const AABB& aabb, int axis, float pos, bool left) const noexcept;
  float computeOverlap(const AABB& a, const AABB& b) const noexcept;
};

// ============================================================================
// Memory-Mapped BVH (Zero-Copy)
// Supports external primitive data from mmap, with minimal index precision
// ============================================================================

// Primitive type for mmap BVH
enum class MMapPrimType : uint8_t {
  Triangle,
  QuantizedTriangle,
  Sphere,
  Quad,
  Custom
};

// Index precision for primitive indices
enum class IndexPrecision : uint8_t {
  Auto = 0,    // Automatically select based on primitive count
  UInt8 = 1,   // Up to 255 primitives
  UInt16 = 2,  // Up to 65535 primitives
  UInt32 = 4   // Up to 4B primitives
};

// Compact BVH node with quantized bounds for low memory
struct alignas(16) CompactBVHNode {
  // Quantized bounds relative to scene AABB (12 bytes)
  uint16_t bounds_min[3];
  uint16_t bounds_max[3];

  // Node data (8 bytes)
  uint32_t data0;  // left_child or prim_offset
  uint32_t data1;  // right_child or prim_count

  // Flags (4 bytes, but only 1 used)
  uint8_t flags;   // bit 0 = is_leaf
  uint8_t axis;    // Split axis (for traversal order)
  uint16_t pad;

  CompactBVHNode() noexcept : data0(0), data1(0), flags(0), axis(0), pad(0) {
    for (int i = 0; i < 3; i++) {
      bounds_min[i] = 0;
      bounds_max[i] = 0xFFFF;
    }
  }

  bool isLeaf() const noexcept { return (flags & 0x1) != 0; }

  void setLeaf(uint32_t offset, uint32_t count) noexcept {
    data0 = offset;
    data1 = count;
    flags |= 0x1;
  }

  void setInterior(uint32_t left, uint32_t right, uint8_t split_axis) noexcept {
    data0 = left;
    data1 = right;
    axis = split_axis;
    flags &= ~0x1;
  }

  // Dequantize bounds
  AABB dequantizeBounds(const Vec3& scene_min, const Vec3& scene_max) const noexcept {
    AABB result;
    result.min.x = dequantizeFloat(bounds_min[0], scene_min.x, scene_max.x);
    result.min.y = dequantizeFloat(bounds_min[1], scene_min.y, scene_max.y);
    result.min.z = dequantizeFloat(bounds_min[2], scene_min.z, scene_max.z);
    result.max.x = dequantizeFloat(bounds_max[0], scene_min.x, scene_max.x);
    result.max.y = dequantizeFloat(bounds_max[1], scene_min.y, scene_max.y);
    result.max.z = dequantizeFloat(bounds_max[2], scene_min.z, scene_max.z);
    return result;
  }

  // Quantize bounds
  void quantizeBounds(const AABB& bounds, const Vec3& scene_min, const Vec3& scene_max) noexcept {
    bounds_min[0] = quantizeFloat(bounds.min.x, scene_min.x, scene_max.x);
    bounds_min[1] = quantizeFloat(bounds.min.y, scene_min.y, scene_max.y);
    bounds_min[2] = quantizeFloat(bounds.min.z, scene_min.z, scene_max.z);
    bounds_max[0] = quantizeFloat(bounds.max.x, scene_min.x, scene_max.x);
    bounds_max[1] = quantizeFloat(bounds.max.y, scene_min.y, scene_max.y);
    bounds_max[2] = quantizeFloat(bounds.max.z, scene_min.z, scene_max.z);
  }
};

// MMap BVH configuration
struct MMapBVHConfig {
  BVHBuildConfig build;           // Standard build config
  IndexPrecision index_precision; // Index precision (Auto recommended)
  bool use_compact_nodes;         // Use CompactBVHNode with quantized bounds
  bool use_ordered_traversal;     // Order traversal by split axis (slightly faster)

  MMapBVHConfig() noexcept
    : index_precision(IndexPrecision::Auto)
    , use_compact_nodes(true)
    , use_ordered_traversal(true) {}

  // Preset: minimum memory
  static MMapBVHConfig minMemory() noexcept {
    MMapBVHConfig cfg;
    cfg.use_compact_nodes = true;
    cfg.build.max_leaf_size = 8;  // Larger leaves = fewer nodes
    return cfg;
  }

  // Preset: maximum speed
  static MMapBVHConfig maxSpeed() noexcept {
    MMapBVHConfig cfg;
    cfg.use_compact_nodes = false;  // Full precision bounds
    cfg.use_ordered_traversal = true;
    cfg.build.max_leaf_size = 4;
    return cfg;
  }
};

// Memory-mapped Triangle BVH (zero-copy primitive storage)
// Primitives are referenced via pointer, not copied
class MMapTriangleBVH {
public:
  MMapTriangleBVH() noexcept;
  ~MMapTriangleBVH() noexcept = default;

  // Build from external memory-mapped triangles (zero-copy)
  // triangles: Pointer to mmap'd or external memory (must remain valid!)
  // count: Number of triangles
  bool build(const Triangle* triangles, uint32_t count,
             const MMapBVHConfig& config = MMapBVHConfig()) noexcept;

  // Traverse and find closest intersection
  // Returns triangle index or kInvalidIndex if no hit
  uint32_t traverse(const Ray& ray, float& hit_t, float& hit_u, float& hit_v) const noexcept;

  // Traverse with configuration
  uint32_t traverseWithConfig(const Ray& ray, float& hit_t, float& hit_u, float& hit_v,
                              const TraversalConfig& config,
                              TraversalStats* stats = nullptr) const noexcept;

  // Get memory usage (BVH only, not primitives)
  size_t getBVHMemoryUsage() const noexcept;

  // Get total memory including primitive reference
  size_t getTotalMemoryUsage() const noexcept;

  // Get statistics
  struct Stats {
    uint32_t num_nodes;
    uint32_t num_leaves;
    uint32_t max_depth;
    float avg_leaf_size;
    uint32_t num_primitives;
    uint8_t index_bytes;         // Bytes per primitive index
    bool uses_compact_nodes;
    size_t bvh_memory_bytes;
    size_t prim_memory_bytes;    // 0 for mmap (external)
  };

  Stats getStats() const noexcept;

  // Access to primitive data (read-only)
  const Triangle* getTriangles() const noexcept { return triangles_; }
  uint32_t getTriangleCount() const noexcept { return triangle_count_; }

  // Check if primitive data is external (mmap)
  bool isExternalData() const noexcept { return is_external_; }

  // Get scene bounds
  const AABB& getSceneBounds() const noexcept { return scene_bounds_; }

private:
  const Triangle* triangles_;    // Non-owning pointer to external data
  uint32_t triangle_count_;
  bool is_external_;

  // Scene bounds (for compact node dequantization)
  AABB scene_bounds_;

  // Configuration
  MMapBVHConfig config_;

  // BVH storage (compact or full)
  std::vector<CompactBVHNode> compact_nodes_;
  std::vector<BVHNode> full_nodes_;

  // Primitive indices with variable precision
  // Storage format depends on index_bytes_
  std::vector<uint8_t> prim_indices_storage_;
  uint8_t index_bytes_;  // 1, 2, or 4

  // Index access helpers
  uint32_t getPrimIndex(uint32_t offset) const noexcept;
  void reserveIndices(uint32_t count) noexcept;
  void pushIndex(uint32_t index) noexcept;

  // Build helpers
  uint32_t buildRecursive(uint32_t* indices, uint32_t num_prims,
                          uint32_t depth, std::vector<AABB>& prim_bounds) noexcept;

  // Traversal helpers (compact vs full nodes)
  uint32_t traverseCompact(const Ray& ray, float& hit_t, float& hit_u, float& hit_v) const noexcept;
  uint32_t traverseFull(const Ray& ray, float& hit_t, float& hit_u, float& hit_v) const noexcept;
};

// Memory-mapped generic BVH (for custom primitives via callback)
class MMapGenericBVH {
public:
  // Intersection callback: (ray, prim_data, prim_index, out_t, out_u, out_v) -> hit
  using IntersectFn = bool (*)(const Ray&, const void*, uint32_t, float&, float&, float&);

  // Bounds callback: (prim_data, prim_index) -> AABB
  using BoundsFn = AABB (*)(const void*, uint32_t);

  MMapGenericBVH() noexcept;
  ~MMapGenericBVH() noexcept = default;

  // Build from external primitive data
  // prim_data: Pointer to primitive array (any type)
  // count: Number of primitives
  // bounds_fn: Function to compute AABB for each primitive
  bool build(const void* prim_data, uint32_t count,
             BoundsFn bounds_fn,
             const MMapBVHConfig& config = MMapBVHConfig()) noexcept;

  // Traverse with custom intersection function
  uint32_t traverse(const Ray& ray, IntersectFn intersect_fn,
                    float& hit_t, float& hit_u, float& hit_v) const noexcept;

  // Get memory usage
  size_t getBVHMemoryUsage() const noexcept;

  // Get statistics
  MMapTriangleBVH::Stats getStats() const noexcept;

  // Access primitive data
  const void* getPrimitiveData() const noexcept { return prim_data_; }
  uint32_t getPrimitiveCount() const noexcept { return prim_count_; }

private:
  const void* prim_data_;
  uint32_t prim_count_;

  AABB scene_bounds_;
  MMapBVHConfig config_;

  std::vector<CompactBVHNode> compact_nodes_;
  std::vector<BVHNode> full_nodes_;

  std::vector<uint8_t> prim_indices_storage_;
  uint8_t index_bytes_;

  uint32_t getPrimIndex(uint32_t offset) const noexcept;
  void reserveIndices(uint32_t count) noexcept;
  void pushIndex(uint32_t index) noexcept;

  uint32_t buildRecursive(uint32_t* indices, uint32_t num_prims,
                          uint32_t depth, const std::vector<AABB>& prim_bounds) noexcept;
};

// ============================================================================
// TriangleBVH4 — Wide BVH for triangles (4-wide SIMD node testing)
// Wraps BVH4 and adds Moller-Trumbore triangle intersection in leaves.
// ============================================================================

class TriangleBVH4 {
public:
  TriangleBVH4() noexcept = default;
  ~TriangleBVH4() noexcept = default;

  // Build from triangles: construct binary BVH -> collapse to BVH4
  bool build(const std::vector<Triangle>& triangles,
             BVH4Precision precision = BVH4Precision::FP32,
             const BVHBuildConfig& config = BVHBuildConfig()) noexcept;

  // Traverse and find closest triangle intersection
  uint32_t traverse(const Ray& ray, float& hit_t,
                    float& hit_u, float& hit_v) const noexcept;

  uint32_t getNumPrimitives() const noexcept { return static_cast<uint32_t>(triangles_.size()); }

private:
  BVH4 bvh4_;
  std::vector<Triangle> triangles_;
};

// ============================================================================
// CompactTriangleBVH — L1-cache-friendly BVH using CompactBVHNode (24 bytes)
// Uses 16-bit quantized bounds for ~57% memory savings vs standard BVHNode.
// ============================================================================

class CompactTriangleBVH {
public:
  CompactTriangleBVH() noexcept = default;
  ~CompactTriangleBVH() noexcept = default;

  // Build from triangles
  bool build(const std::vector<Triangle>& triangles,
             const BVHBuildConfig& config = BVHBuildConfig()) noexcept;

  // Traverse and find closest triangle intersection
  uint32_t traverse(const Ray& ray, float& hit_t,
                    float& hit_u, float& hit_v) const noexcept;

  uint32_t getNumPrimitives() const noexcept { return static_cast<uint32_t>(triangles_.size()); }
  size_t getBVHMemoryUsage() const noexcept;

private:
  std::vector<CompactBVHNode> nodes_;
  std::vector<Triangle> triangles_;
  std::vector<uint32_t> prim_indices_;
  AABB scene_bounds_;
};

// ============================================================================
// Auto-Tuning for BVH Construction and Traversal
// Samples M primitives from N input, measures costs, selects best algorithms
// ============================================================================

// BVH build method
enum class BVHBuildMethod : uint8_t {
  TriangleBVH,    // Standard BVH with object splits only
  SBVH            // Split BVH with spatial splits
};

// Auto-tune configuration
struct AutoTuneConfig {
  uint32_t sample_prim_count;      // Number of primitives to sample (M), 0 = auto (sqrt(N))
  uint32_t sample_ray_count;       // Number of rays to test for traversal
  uint32_t warmup_iterations;      // Warmup iterations before timing
  uint32_t timing_iterations;      // Iterations for timing measurement
  bool test_sbvh;                  // Test SBVH in addition to TriangleBVH
  bool test_traversal_configs;     // Test different traversal configs
  float build_weight;              // Weight for build time in cost (0-1)
  float traversal_weight;          // Weight for traversal time in cost (0-1)
  float memory_weight;             // Weight for memory usage in cost (0-1)

  AutoTuneConfig() noexcept
    : sample_prim_count(0)         // 0 = auto
    , sample_ray_count(1000)
    , warmup_iterations(3)
    , timing_iterations(10)
    , test_sbvh(true)
    , test_traversal_configs(true)
    , build_weight(0.1f)           // Build time typically matters less
    , traversal_weight(0.8f)       // Traversal performance is most important
    , memory_weight(0.1f) {}       // Memory matters for large scenes

  // Preset: optimize for throughput (many rays)
  static AutoTuneConfig throughput() noexcept {
    AutoTuneConfig cfg;
    cfg.build_weight = 0.05f;
    cfg.traversal_weight = 0.9f;
    cfg.memory_weight = 0.05f;
    return cfg;
  }

  // Preset: optimize for interactive (frequent rebuilds)
  static AutoTuneConfig interactive() noexcept {
    AutoTuneConfig cfg;
    cfg.build_weight = 0.4f;
    cfg.traversal_weight = 0.5f;
    cfg.memory_weight = 0.1f;
    return cfg;
  }

  // Preset: optimize for memory (large scenes)
  static AutoTuneConfig memory() noexcept {
    AutoTuneConfig cfg;
    cfg.build_weight = 0.1f;
    cfg.traversal_weight = 0.5f;
    cfg.memory_weight = 0.4f;
    cfg.test_sbvh = false;  // SBVH uses more memory due to reference duplication
    return cfg;
  }

  // Preset: quick tuning (fewer samples)
  static AutoTuneConfig quick() noexcept {
    AutoTuneConfig cfg;
    cfg.sample_ray_count = 500;
    cfg.warmup_iterations = 1;
    cfg.timing_iterations = 5;
    cfg.test_sbvh = false;
    cfg.test_traversal_configs = false;
    return cfg;
  }
};

// Auto-tune result
struct AutoTuneResult {
  BVHBuildMethod best_method;
  BVHBuildConfig best_bvh_config;        // Used if method == TriangleBVH
  SBVHBuildConfig best_sbvh_config;      // Used if method == SBVH
  TraversalConfig best_traversal_config;

  // Measured metrics (normalized to per-primitive or per-ray)
  float build_time_us_per_prim;          // Build time in microseconds per primitive
  float traversal_time_ns_per_ray;       // Traversal time in nanoseconds per ray
  float memory_bytes_per_prim;           // Memory usage in bytes per primitive

  // Detailed metrics for each tested configuration
  struct ConfigMetrics {
    BVHBuildMethod method;
    float build_time_us_per_prim;
    float traversal_time_ns_per_ray;
    float memory_bytes_per_prim;
    float combined_cost;                 // Weighted cost used for selection
    uint32_t max_leaf_size;
    float sbvh_alpha;                    // Only for SBVH
  };
  std::vector<ConfigMetrics> all_metrics;

  // Scene characteristics detected during tuning
  struct SceneCharacteristics {
    float avg_triangle_area;
    float scene_volume;
    float triangle_density;              // Triangles per unit volume
    float overlap_ratio;                 // Estimated AABB overlap
    bool has_thin_triangles;             // Long thin triangles detected
    bool has_clustered_distribution;     // Spatial clustering detected
    bool has_coplanar_regions;           // Co-planar triangles detected
  };
  SceneCharacteristics scene_info;

  AutoTuneResult() noexcept
    : best_method(BVHBuildMethod::TriangleBVH)
    , build_time_us_per_prim(0)
    , traversal_time_ns_per_ray(0)
    , memory_bytes_per_prim(0) {}
};

// Simple timer for benchmarking (platform-independent)
class SimpleTimer {
public:
  void start() noexcept;
  void stop() noexcept;
  double elapsedMicroseconds() const noexcept;
  double elapsedMilliseconds() const noexcept { return elapsedMicroseconds() / 1000.0; }

private:
  uint64_t start_time_;
  uint64_t end_time_;
};

// Simple RNG for sampling (Xorshift64)
class SimpleRNG {
public:
  explicit SimpleRNG(uint64_t seed = 12345) noexcept : state_(seed ? seed : 1) {}

  uint64_t next() noexcept {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return state_ * 0x2545F4914F6CDD1DULL;
  }

  float nextFloat() noexcept {
    return static_cast<float>(next() & 0xFFFFFF) / static_cast<float>(0x1000000);
  }

  // Generate random float in range [min, max]
  float nextFloat(float min_val, float max_val) noexcept {
    return min_val + nextFloat() * (max_val - min_val);
  }

  // Shuffle array using Fisher-Yates
  template<typename T>
  void shuffle(std::vector<T>& arr) noexcept {
    for (size_t i = arr.size() - 1; i > 0; i--) {
      size_t j = next() % (i + 1);
      std::swap(arr[i], arr[j]);
    }
  }

private:
  uint64_t state_;
};

// AutoTuner class
class AutoTuner {
public:
  // Auto-tune BVH construction for triangles
  // Returns the best configuration based on sampling and measurement
  static AutoTuneResult tune(const std::vector<Triangle>& triangles,
                             const AutoTuneConfig& config = AutoTuneConfig()) noexcept;

  // Tune traversal configuration for an existing TriangleBVH
  // Useful when you already have a BVH and want optimal traversal settings
  static TraversalConfig tuneTraversal(const TriangleBVH& bvh,
                                        const AABB& scene_bounds,
                                        uint32_t sample_ray_count = 1000,
                                        uint32_t timing_iterations = 10) noexcept;

  // Tune traversal configuration for an existing SBVH
  static TraversalConfig tuneTraversal(const SBVH& sbvh,
                                        const AABB& scene_bounds,
                                        uint32_t sample_ray_count = 1000,
                                        uint32_t timing_iterations = 10) noexcept;

  // Analyze scene characteristics (useful for manual tuning decisions)
  static AutoTuneResult::SceneCharacteristics analyzeScene(
      const std::vector<Triangle>& triangles) noexcept;

  // Build BVH using auto-tuned configuration
  // Convenience function that combines tune() and build()
  static bool buildOptimal(const std::vector<Triangle>& triangles,
                           TriangleBVH& out_bvh,
                           const AutoTuneConfig& config = AutoTuneConfig()) noexcept;

  static bool buildOptimal(const std::vector<Triangle>& triangles,
                           SBVH& out_sbvh,
                           const AutoTuneConfig& config = AutoTuneConfig()) noexcept;

private:
  // Sample M primitives from N using stratified sampling
  static std::vector<Triangle> samplePrimitives(
      const std::vector<Triangle>& triangles,
      uint32_t sample_count,
      SimpleRNG& rng) noexcept;

  // Generate test rays covering the scene
  static std::vector<Ray> generateTestRays(
      const AABB& scene_bounds,
      uint32_t ray_count,
      SimpleRNG& rng) noexcept;

  // Measure build time for TriangleBVH
  static double measureBuildTime(
      const std::vector<Triangle>& triangles,
      const BVHBuildConfig& config,
      uint32_t iterations,
      TriangleBVH& out_bvh) noexcept;

  // Measure build time for SBVH
  static double measureBuildTime(
      const std::vector<Triangle>& triangles,
      const SBVHBuildConfig& config,
      uint32_t iterations,
      SBVH& out_sbvh) noexcept;

  // Measure traversal time for TriangleBVH
  static double measureTraversalTime(
      const TriangleBVH& bvh,
      const std::vector<Ray>& rays,
      const TraversalConfig& trav_config,
      uint32_t iterations) noexcept;

  // Measure traversal time for SBVH
  static double measureTraversalTime(
      const SBVH& sbvh,
      const std::vector<Ray>& rays,
      const TraversalConfig& trav_config,
      uint32_t iterations) noexcept;

  // Estimate memory usage
  static size_t estimateMemory(const TriangleBVH& bvh) noexcept;
  static size_t estimateMemory(const SBVH& sbvh) noexcept;
};

// ============================================================================
// Traversal Profiling (Zero-Overhead Template Design)
// Use ProfilePolicy to enable/disable profiling with no runtime cost
// ============================================================================

// Profiling data collected during traversal
struct TraversalProfile {
  uint32_t nodes_visited = 0;    // Total BVH nodes visited
  uint32_t leaf_visits = 0;      // Leaf nodes visited
  uint32_t prims_tested = 0;     // Primitive intersection tests
  uint32_t max_depth = 0;        // Maximum traversal depth reached

  void reset() noexcept {
    nodes_visited = 0;
    leaf_visits = 0;
    prims_tested = 0;
    max_depth = 0;
  }

  // Add another profile (for aggregating stats)
  void add(const TraversalProfile& other) noexcept {
    nodes_visited += other.nodes_visited;
    leaf_visits += other.leaf_visits;
    prims_tested += other.prims_tested;
    max_depth = std::max(max_depth, other.max_depth);
  }
};

// Policy class: No profiling (zero overhead - all calls optimized away)
struct NoProfiler {
  static constexpr bool kEnabled = false;

  void visitNode() noexcept {}
  void visitLeaf() noexcept {}
  void testPrim() noexcept {}
  void testPrims(uint32_t) noexcept {}
  void pushDepth(uint32_t) noexcept {}
  void popDepth() noexcept {}
};

// Policy class: With profiling (collects statistics)
class WithProfiler {
public:
  static constexpr bool kEnabled = true;

  explicit WithProfiler(TraversalProfile* profile) noexcept
    : profile_(profile), current_depth_(0) {}

  void visitNode() noexcept {
    profile_->nodes_visited++;
  }

  void visitLeaf() noexcept {
    profile_->leaf_visits++;
  }

  void testPrim() noexcept {
    profile_->prims_tested++;
  }

  void testPrims(uint32_t count) noexcept {
    profile_->prims_tested += count;
  }

  void pushDepth(uint32_t depth) noexcept {
    current_depth_ = depth;
    if (depth > profile_->max_depth) {
      profile_->max_depth = depth;
    }
  }

  void popDepth() noexcept {
    if (current_depth_ > 0) current_depth_--;
  }

private:
  TraversalProfile* profile_;
  uint32_t current_depth_;
};

// ============================================================================
// Profiled Traversal Methods (Template-based)
// ============================================================================

// Profiled traversal for TriangleBVH
// Usage:
//   TraversalProfile profile;
//   uint32_t hit = traverseProfiled<WithProfiler>(bvh, ray, hit_t, u, v, &profile);
// Or without profiling (zero overhead):
//   uint32_t hit = traverseProfiled<NoProfiler>(bvh, ray, hit_t, u, v, nullptr);
template<typename Profiler = NoProfiler>
uint32_t traverseProfiled(const TriangleBVH& bvh, const Ray& ray,
                          float& hit_t, float& hit_u, float& hit_v,
                          TraversalProfile* profile = nullptr) noexcept;

// Profiled traversal for SBVH
template<typename Profiler = NoProfiler>
uint32_t traverseProfiled(const SBVH& sbvh, const Ray& ray,
                          float& hit_t, float& hit_u, float& hit_v,
                          TraversalProfile* profile = nullptr) noexcept;

// Profiled traversal for MMapTriangleBVH
template<typename Profiler = NoProfiler>
uint32_t traverseProfiled(const MMapTriangleBVH& bvh, const Ray& ray,
                          float& hit_t, float& hit_u, float& hit_v,
                          TraversalProfile* profile = nullptr) noexcept;

// ============================================================================
// Heatmap / Pseudocolor Image Writer
// Supports BMP, TGA, PPM, PNG - no external dependencies
// PNG uses built-in DEFLATE compression (no zlib required)
// ============================================================================

// Colormap types for heatmap visualization
enum class Colormap : uint8_t {
  Grayscale,    // Black to white
  Heat,         // Black -> Red -> Yellow -> White
  Jet,          // Blue -> Cyan -> Green -> Yellow -> Red
  Viridis,      // Perceptually uniform (purple -> blue -> green -> yellow)
  Turbo,        // Improved rainbow (Google's turbo colormap)
  Plasma,       // Perceptually uniform (purple -> pink -> orange -> yellow)
  Inferno,      // Perceptually uniform (black -> purple -> red -> yellow)
  Cool,         // Cyan to Magenta
  Hot           // Black -> Red -> Yellow -> White (classic hot metal)
};

// Image format for output
enum class ImageFormat : uint8_t {
  BMP,          // Windows Bitmap (24-bit RGB, uncompressed)
  TGA,          // Truevision TGA (24-bit RGB, uncompressed)
  PPM,          // Portable Pixmap (binary P6 format)
  PNG           // PNG (24-bit RGB, DEFLATE compressed, no zlib dependency)
};

// RGB color (8-bit per channel)
struct RGB8 {
  uint8_t r, g, b;

  RGB8() noexcept : r(0), g(0), b(0) {}
  RGB8(uint8_t r_, uint8_t g_, uint8_t b_) noexcept : r(r_), g(g_), b(b_) {}

  // Create from float [0,1] values
  static RGB8 fromFloat(float r, float g, float b) noexcept {
    return RGB8(
      static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, r * 255.0f))),
      static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, g * 255.0f))),
      static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, b * 255.0f)))
    );
  }
};

// Heatmap generator and image writer
class HeatmapWriter {
public:
  // Map a value [0, 1] to RGB color using specified colormap
  static RGB8 mapColor(float value, Colormap cmap = Colormap::Viridis) noexcept;

  // Map an integer value to RGB color with automatic normalization
  // value: The value to map
  // max_value: Maximum expected value (values >= max map to 1.0)
  static RGB8 mapColor(uint32_t value, uint32_t max_value,
                       Colormap cmap = Colormap::Viridis) noexcept;

  // Write image to file
  // data: RGB8 array of size width * height (row-major, top-to-bottom)
  // Returns true on success
  static bool writeImage(const char* filename, const RGB8* data,
                         uint32_t width, uint32_t height,
                         ImageFormat format = ImageFormat::BMP) noexcept;

  // Write image from float array [0, 1]
  static bool writeImage(const char* filename, const float* data,
                         uint32_t width, uint32_t height,
                         Colormap cmap = Colormap::Viridis,
                         ImageFormat format = ImageFormat::BMP) noexcept;

  // Write image from uint32 array with automatic normalization
  static bool writeImage(const char* filename, const uint32_t* data,
                         uint32_t width, uint32_t height,
                         uint32_t max_value,
                         Colormap cmap = Colormap::Viridis,
                         ImageFormat format = ImageFormat::BMP) noexcept;

  // Generate heatmap from traversal profiles
  // profiles: Array of TraversalProfile for each pixel (width * height)
  // metric: Which metric to visualize
  enum class Metric { NodesVisited, LeafVisits, PrimsTested, MaxDepth };

  static bool writeHeatmap(const char* filename,
                           const TraversalProfile* profiles,
                           uint32_t width, uint32_t height,
                           Metric metric = Metric::PrimsTested,
                           Colormap cmap = Colormap::Viridis,
                           ImageFormat format = ImageFormat::BMP,
                           uint32_t max_value = 0) noexcept;  // 0 = auto

private:
  // BMP file writing
  static bool writeBMP(const char* filename, const RGB8* data,
                       uint32_t width, uint32_t height) noexcept;

  // TGA file writing
  static bool writeTGA(const char* filename, const RGB8* data,
                       uint32_t width, uint32_t height) noexcept;

  // PPM file writing (binary P6)
  static bool writePPM(const char* filename, const RGB8* data,
                       uint32_t width, uint32_t height) noexcept;

  // PNG file writing (DEFLATE compressed, no external dependencies)
  static bool writePNG(const char* filename, const RGB8* data,
                       uint32_t width, uint32_t height) noexcept;

  // Colormap lookup tables
  static RGB8 grayscale(float t) noexcept;
  static RGB8 heat(float t) noexcept;
  static RGB8 jet(float t) noexcept;
  static RGB8 viridis(float t) noexcept;
  static RGB8 turbo(float t) noexcept;
  static RGB8 plasma(float t) noexcept;
  static RGB8 inferno(float t) noexcept;
  static RGB8 cool(float t) noexcept;
  static RGB8 hot(float t) noexcept;
};

// ============================================================================
// Convenience functions for profiled rendering
// ============================================================================

// Render a single ray and collect profile data
// Returns hit triangle index (or kInvalidIndex)
template<typename BVHType>
uint32_t renderRayProfiled(const BVHType& bvh, const Ray& ray,
                           float& hit_t, float& hit_u, float& hit_v,
                           TraversalProfile& profile) noexcept {
  profile.reset();
  return traverseProfiled<WithProfiler>(bvh, ray, hit_t, hit_u, hit_v, &profile);
}

// Render image and collect per-pixel profile data
// Returns array of profiles (caller must delete[])
// Also fills hit_image with triangle indices (optional, can be nullptr)
template<typename BVHType>
TraversalProfile* renderImageProfiled(
    const BVHType& bvh,
    uint32_t width, uint32_t height,
    const Vec3& camera_pos,
    const Vec3& camera_dir,
    const Vec3& camera_up,
    float fov_y,  // Field of view in radians
    uint32_t* hit_image = nullptr) noexcept;

} // namespace lightrt

// ============================================================================
// Memory Alignment Constants
// ============================================================================


// ============================================================================
// Memory Alignment Constants
// ============================================================================

// 16-byte alignment for SIMD traversal
static constexpr size_t kAlignment = 16;

// 32-byte alignment for AVX/SSE2 RayContext
static constexpr size_t kRayContextAlignment = 32;

// ============================================================================
// Memory Manager Documentation
// ============================================================================

/*
  Memory Alignment Requirements:

  - 16-byte alignment: Used for Vec3, Ray, and base RayContext
    - Required for SIMD packet traversal
    - Aligns 3-vector components to 16-byte boundary
    - Enables efficient SIMD operations (128-bit registers)

  - 32-byte alignment: Used for AVX/SSE2 RayContext
    - Required for AVX/SSE2 packet traversal
    - Aligns RayContext for AVX instructions
    - Enables 128-bit register operations

  Why 16 bytes?
    - Vec3 stores 3 floats (12 bytes) + padding to 16 bytes
    - Enables SIMD operations without padding overhead
    - Minimum for 128-bit SIMD register alignment

  TaskSystem Memory Manager:

    The TaskSystem provides a thread pool with work-stealing:

    - Initialization: TaskSystem::initialize() uses hardware concurrency
    - Thread safety: Uses static members with RAII guard (TaskSystemGuard)
    - Automatic shutdown: RAII guard destructor calls shutdown on exit

    Memory Allocation Patterns:

    1. No explicit allocator - Uses standard std::vector and std::queue
    2. Static state - All TaskSystem state is static (singleton-like)
    3. No external allocation - No custom memory pools or arenas

    Queue Size: Max tasks in queue before blocking is unlimited
    (uses std::queue which grows dynamically)
*/

// ============================================================================
// SIMD Memory Requirements
// ============================================================================

/*
  SIMD Memory Alignment Requirements:

  AVX/SSE2 RayContext (32-byte aligned):
  - Stores SIMD-optimized ray data for packet traversal
  - Uses _mm_set_ps() for AVX/SSE2 intrinsics
  - 128-bit register alignment required

  Memory Manager Notes:

  - All core types are 16-byte aligned (alignas(16))
  - No external dependencies
  - C++17 compatible compiler
  - SIMD-friendly design

  Constants and Configuration:

  - kAlignment: 16 bytes for SIMD traversal
  - kRayContextAlignment: 32 bytes for AVX/SSE2
  - kEpsilon: 1e-6f (near-zero threshold)
  - kInvalidIndex: 0xFFFFFFFF (no-hit result)
  - kInfinity: infinity for traversal limits

  Memory Stats:

  - Track TaskSystem memory usage (queue size, thread count)
  - Monitor alignment overhead (16 vs 32 byte)
*/


#endif // LIGHTRT_HH_
