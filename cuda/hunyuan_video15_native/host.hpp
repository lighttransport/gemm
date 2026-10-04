#ifndef PIXAL3D_HV15N_HOST_HPP
#define PIXAL3D_HV15N_HOST_HPP
#include "../../common/safetensors.h"
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif
namespace hv15n {
namespace fs = std::filesystem;
inline void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
inline size_t product(const std::vector<int> &shape) {
  size_t result = 1;
  for (int dimension : shape) {
    require(dimension > 0 && result <= SIZE_MAX / size_t(dimension),
            "invalid tensor size");
    result *= size_t(dimension);
  }
  require(result <= SIZE_MAX / sizeof(float), "tensor byte size overflow");
  return result;
}
struct Json {
  std::unique_ptr<json_val, decltype(&json_free)> value{nullptr, json_free};
  explicit Json(const fs::path &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file) && file.tellg() >= 0 && file.tellg() <= 64 * 1024 * 1024,
            "cannot read bounded JSON: " + path.string());
    std::string text(size_t(file.tellg()), '\0');
    file.seekg(0);
    file.read(text.data(), text.size());
    require(bool(file), "short JSON read");
    value.reset(json_parse(text.data(), int(text.size())));
    require(bool(value), "invalid JSON: " + path.string());
  }
};
inline json_val *field(json_val *object, const char *key) {
  return json_obj_get(object, key);
}
inline std::string string(json_val *value) {
  require(value && value->type == JSON_STRING, "expected JSON string");
  return std::string(value->str.ptr, size_t(value->str.len));
}
inline std::string string(json_val *object, const char *key) {
  return string(field(object, key));
}
inline std::string escape(const std::string &text) {
  std::string result = "\"";
  for (unsigned char c : text) {
    if (c == '\"' || c == '\\') {
      result += '\\';
      result += char(c);
    } else if (c < 32) {
      const char hex[] = "0123456789abcdef";
      result += "\\u00";
      result += hex[c >> 4];
      result += hex[c & 15];
    } else
      result += char(c);
  }
  return result + "\"";
}
inline fs::path relative_file(const fs::path &root, const std::string &name) {
  fs::path path(name);
  require(!name.empty() && !path.is_absolute(),
          "component path must be relative");
  for (const auto &part : path)
    require(part != "..", "component path escapes model");
  auto resolved = fs::canonical(root / path), base = fs::canonical(root);
  auto relative = resolved.lexically_relative(base);
  require(!relative.empty() && *relative.begin() != "..",
          "component symlink escapes model");
  require(fs::is_regular_file(resolved), "component is not a file");
  return resolved;
}
inline float half_float(uint16_t h) {
  uint32_t bits = uint32_t(h & 0x8000) << 16;
  unsigned exponent = (h >> 10) & 31, mantissa = h & 1023;
  if (exponent == 31)
    bits |= 0x7f800000u | (mantissa << 13);
  else if (exponent)
    bits |= ((exponent + 112) << 23) | (mantissa << 13);
  else if (mantissa) {
    int shift = 0;
    while (!(mantissa & 1024)) {
      mantissa <<= 1;
      ++shift;
    }
    bits |= uint32_t(113 - shift) << 23 | ((mantissa & 1023) << 13);
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx,f16c"))) inline void
decode_f16_fast(const uint16_t *input, float *output, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8)
    _mm256_storeu_ps(output + i,
                     _mm256_cvtph_ps(_mm_loadu_si128(
                         reinterpret_cast<const __m128i *>(input + i))));
  for (; i < count; ++i)
    output[i] = half_float(input[i]);
}
#endif
inline void decode_f16(const uint16_t *input, float *output, size_t count) {
#if defined(__x86_64__) && defined(__GNUC__)
  if (__builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c")) {
    decode_f16_fast(input, output, count);
    return;
  }
#endif
  for (size_t i = 0; i < count; ++i)
    output[i] = half_float(input[i]);
}
struct Weights {
  std::string identity;
  std::unique_ptr<st_context, decltype(&safetensors_close)> context{
      nullptr, safetensors_close};
  std::unordered_map<std::string, int> lookup; // safetensors_find is a linear scan
  explicit Weights(const fs::path &path) {
    identity = fs::absolute(path).string() + ":" + std::to_string(fs::file_size(path)) +
               ":" + std::to_string(fs::last_write_time(path).time_since_epoch().count());
    context.reset(safetensors_open(path.c_str()));
    require(bool(context), "cannot mmap weights: " + path.string());
    require(context->data_offset <= context->map_size,
            "invalid safetensors header extent");
    for (int i = 0; i < context->n_tensors; ++i) {
      const auto &t = context->tensors[i];
      require(t.offset <= context->map_size - context->data_offset &&
                  t.nbytes <=
                      context->map_size - context->data_offset - t.offset,
              "tensor outside safetensors file");
      lookup.emplace(t.name, i);
    }
  }
  int index(const std::string &name) const {
    auto it = lookup.find(name);
    if (it == lookup.end())
      throw std::runtime_error("missing tensor: " + name);
    return it->second;
  }
  bool has(const std::string &name) const {
    return lookup.count(name) != 0;
  }
  std::vector<int> shape(const std::string &name) const {
    int i = index(name);
    std::vector<int> result;
    for (int d = 0; d < safetensors_ndims(context.get(), i); ++d) {
      auto n = safetensors_shape(context.get(), i)[d];
      require(n > 0 && n <= INT32_MAX, "invalid weight shape: " + name);
      result.push_back(int(n));
    }
    require(product(result) * safetensors_dtype_size(
                                  safetensors_dtype(context.get(), i)) ==
                safetensors_nbytes(context.get(), i),
            "weight byte count mismatch");
    return result;
  }
  const void *data(const std::string &name) const {
    return safetensors_data(context.get(), index(name));
  }
  std::string dtype(const std::string &name) const {
    return safetensors_dtype(context.get(), index(name));
  }
  std::vector<float> floats(const std::string &name) const {
    size_t count = product(shape(name));
    std::vector<float> result(count);
    auto type = dtype(name);
    const void *pointer = data(name);
    require(type == "F16" || type == "F32" || type == "BF16",
            "unsupported weight dtype");
    if (type == "F16") {
      decode_f16(static_cast<const uint16_t *>(pointer), result.data(), count);
      return result;
    }
    for (size_t i = 0; i < count; ++i) {
      if (type == "F32")
        result[i] = static_cast<const float *>(pointer)[i];
      else if (type == "F16")
        result[i] = half_float(static_cast<const uint16_t *>(pointer)[i]);
      else {
        uint32_t bits = uint32_t(static_cast<const uint16_t *>(pointer)[i])
                        << 16;
        std::memcpy(&result[i], &bits, sizeof(bits));
      }
    }
    return result;
  }
};
inline std::vector<float> read_f32(const fs::path &path, size_t count) {
  require(fs::file_size(path) == count * sizeof(float),
          "wrong F32 input byte count");
  std::vector<float> result(count);
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char *>(result.data()), count * sizeof(float));
  require(bool(file), "short F32 input");
  for (float v : result)
    require(std::isfinite(v), "nonfinite F32 input");
  return result;
}
inline std::vector<float> schedule(int steps, float shift) {
  require(steps > 0 && shift > 0, "invalid flow schedule");
  std::vector<float> result(size_t(steps) + 1);
  for (int i = 0; i <= steps; ++i) {
    float t = 1.f - float(i) / steps;
    result[i] = shift * t / (1.f + (shift - 1.f) * t);
  }
  return result;
}
inline int relative_bucket(int query, int key, int buckets = 32,
                           int distance = 128) {
  int delta = key - query, result = delta > 0 ? buckets / 2 : 0;
  int n = std::abs(delta), exact = buckets / 4;
  return result +
         (n < exact ? n
                    : std::min(buckets / 2 - 1,
                               exact + int(std::log(float(n) / exact) /
                                           std::log(float(distance) / exact) *
                                           (buckets / 2 - exact))));
}
inline std::vector<std::string> glyph_texts(const std::string &prompt) {
  std::vector<std::string> result;
  size_t cursor = 0;
  while (cursor < prompt.size()) {
    size_t ascii = prompt.find('"', cursor), curved = prompt.find("“", cursor);
    size_t start = std::min(ascii, curved);
    if (start == std::string::npos)
      break;
    bool is_ascii = start == ascii;
    size_t first = start + (is_ascii ? 1 : 3);
    size_t end = is_ascii ? prompt.find('"', first) : prompt.find("”", first);
    if (end == std::string::npos ||
        prompt.substr(first, end - first).find('\n') != std::string::npos) {
      cursor = first;
      continue;
    }
    auto value = prompt.substr(first, end - first);
    if (value.find('\n') == std::string::npos &&
        std::find(result.begin(), result.end(), value) == result.end())
      result.push_back(value);
    cursor = end + (is_ascii ? 1 : 3);
  }
  return result;
}
inline std::vector<int> byt5_tokens(const std::string &prompt) {
  std::string formatted;
  for (const auto &text : glyph_texts(prompt))
    formatted += "Text \"" + text + "\". ";
  std::vector<int> tokens;
  for (unsigned char c : formatted) {
    if (tokens.size() == 255)
      break;
    tokens.push_back(int(c) + 3);
  }
  if (!formatted.empty())
    tokens.push_back(1);
  return tokens;
}
} // namespace hv15n
#endif
