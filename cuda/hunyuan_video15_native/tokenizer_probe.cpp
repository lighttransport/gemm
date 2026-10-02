#include "tokenizer.hpp"
#include <iostream>
int main(int argc, char **argv) {
    try {
        hv15n::require(argc == 3, "usage: tokenizer_probe TOKENIZER TEXT");
        hv15n::Tokenizer tokenizer(argv[1]);
        auto tokens = tokenizer.encode(argv[2]);
        std::cout << "[";
        for (size_t i = 0; i < tokens.size(); i++)
            std::cout << (i ? "," : "") << tokens[i];
        std::cout << "]\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
