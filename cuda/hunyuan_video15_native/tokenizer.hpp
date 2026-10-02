#ifndef PIXAL3D_HV15N_TOKENIZER_HPP
#define PIXAL3D_HV15N_TOKENIZER_HPP
#include "host.hpp"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
namespace hv15n {
class Tokenizer {
    std::map<std::string, int> vocabulary, merges, special;
    std::vector<std::string> bytes;
    std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)> regex{nullptr, pcre2_code_free};
    static std::string utf8(int c) {
        if (c < 128)
            return std::string(1, char(c));
        return std::string{char(192 | (c >> 6)), char(128 | (c & 63))};
    }
    void piece(const std::string &text, std::vector<int> &output) const {
        std::vector<std::string> parts;
        for (unsigned char c : text)
            parts.push_back(bytes[c]);
        while (parts.size() > 1) {
            int best = INT32_MAX;
            size_t at = 0;
            for (size_t i = 0; i + 1 < parts.size(); i++) {
                auto it = merges.find(parts[i] + " " + parts[i + 1]);
                if (it != merges.end() && it->second < best) {
                    best = it->second;
                    at = i;
                }
            }
            if (best == INT32_MAX)
                break;
            parts[at] += parts[at + 1];
            parts.erase(parts.begin() + at + 1);
        }
        for (const auto &part : parts) {
            auto it = vocabulary.find(part);
            require(it != vocabulary.end(), "BPE token missing from vocabulary");
            output.push_back(it->second);
        }
    }
    void ordinary(const std::string &text, std::vector<int> &output) const {
        std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(
            pcre2_match_data_create_from_pattern(regex.get(), nullptr), pcre2_match_data_free);
        size_t offset = 0;
        while (offset < text.size()) {
            int rc = pcre2_match(regex.get(), reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), offset,
                                 0, match.get(), nullptr);
            require(rc >= 0, "Qwen tokenizer requires valid UTF-8");
            auto span = pcre2_get_ovector_pointer(match.get());
            require(span[0] == offset && span[1] > offset, "tokenizer pre-split gap");
            piece(text.substr(offset, span[1] - offset), output);
            offset = span[1];
        }
    }

  public:
    explicit Tokenizer(const fs::path &path) {
        Json json(path);
        auto model = field(json.value.get(), "model");
        auto vocab = field(model, "vocab");
        require(vocab && vocab->type == JSON_OBJECT, "missing tokenizer vocabulary");
        for (int i = 0; i < vocab->obj.count; i++)
            vocabulary.emplace(vocab->obj.keys[i], int(vocab->obj.vals[i].num));
        auto list = field(model, "merges");
        require(list && list->type == JSON_ARRAY, "missing BPE merges");
        for (int i = 0; i < list->arr.count; i++) {
            auto item = &list->arr.items[i];
            std::string value = item->type == JSON_STRING
                                    ? string(item)
                                    : string(&item->arr.items[0]) + " " + string(&item->arr.items[1]);
            merges.emplace(value, i);
        }
        list = field(json.value.get(), "added_tokens");
        require(list && list->type == JSON_ARRAY, "missing added tokens");
        for (int i = 0; i < list->arr.count; i++) {
            auto item = &list->arr.items[i];
            special.emplace(string(item, "content"), int(field(item, "id")->num));
        }
        int extra = 256;
        for (int c = 0; c < 256; c++)
            bytes.push_back(utf8(
                (c >= 33 && c <= 126) || (c >= 161 && c <= 172) || (c >= 174 && c <= 255) ? c : extra++));
        const char *pattern =
            R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
        int error;
        PCRE2_SIZE at;
        regex.reset(pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern), PCRE2_ZERO_TERMINATED,
                                  PCRE2_UTF | PCRE2_UCP, &error, &at, nullptr));
        require(bool(regex), "cannot compile Qwen pre-tokenizer");
    }
    std::vector<int> encode(const std::string &text) const {
        std::vector<int> result;
        size_t cursor = 0;
        while (cursor < text.size()) {
            size_t first = std::string::npos;
            std::string token;
            int id = 0;
            for (const auto &entry : special) {
                size_t at = text.find(entry.first, cursor);
                if (at < first || (at == first && entry.first.size() > token.size())) {
                    first = at;
                    token = entry.first;
                    id = entry.second;
                }
            }
            if (first == std::string::npos) {
                ordinary(text.substr(cursor), result);
                break;
            }
            ordinary(text.substr(cursor, first - cursor), result);
            result.push_back(id);
            cursor = first + token.size();
        }
        return result;
    }
};
inline std::string qwen_prefix() {
    return "<|im_start|>system\nYou are a helpful assistant. Describe the video by detailing the following "
           "aspects:         "
           "1. The main content and theme of the video.         "
           "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.    "
           "     "
           "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.  "
           "       "
           "4. background environment, light, style and atmosphere.         "
           "5. camera angles, movements, and transitions used in the video.<|im_end|>\n<|im_start|>user\n";
}
} // namespace hv15n
#endif
