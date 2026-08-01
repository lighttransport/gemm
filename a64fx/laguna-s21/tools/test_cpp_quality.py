#!/usr/bin/env python3
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import cpp_quality


class CppQualityTest(unittest.TestCase):
    def test_extract_cpp(self):
        self.assertEqual(cpp_quality.extract_cpp("x```cpp\nint x;\n```y"),
                         "int x;\n")

    def test_missing_fence(self):
        with self.assertRaises(ValueError):
            cpp_quality.extract_cpp("int main() {}")

    def test_compile_pass_and_fail(self):
        ok = cpp_quality.check_cpp("int main(){return 0;}", "g++", "c++2a",
                                   True, 5.0)
        self.assertTrue(ok["compile_ok"])
        self.assertTrue(ok["run_ok"])
        bad = cpp_quality.check_cpp("int main( {", "g++", "c++2a",
                                    False, 5.0)
        self.assertFalse(bad["compile_ok"])
        self.assertIn("error:", bad["diagnostic"])


if __name__ == "__main__":
    unittest.main()
