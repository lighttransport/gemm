"""Compile generated lower_index answers and check them against std::lower_bound.

Pass runner/reference logs with '=== Generated text ===' markers. All answers
must be complete C++17 translation units defining lower_index, without main.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

DRIVER = r'''
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>
std::size_t lower_index(const std::vector<int>&, int);
int main() {
    std::mt19937 rng(42);
    std::size_t checks=0;
    for(int trial=0;trial<10000;++trial) {
        std::vector<int> a(trial==0?0:trial==1?1:rng()%257);
        for(auto &x:a) x=int(rng()%2001)-1000;
        if(!a.empty() && trial%17==0) a.front()=INT_MIN;
        if(a.size()>1 && trial%19==0) a.back()=INT_MAX;
        std::sort(a.begin(),a.end());
        for(int x:{INT_MIN,INT_MAX,int(rng()%2201)-1100}) {
            auto expected=std::size_t(std::lower_bound(a.begin(),a.end(),x)-a.begin());
            auto got=lower_index(a,x);
            if(got!=expected) {
                std::fprintf(stderr,"trial=%d size=%zu x=%d got=%zu expected=%zu\n",
                             trial,a.size(),x,got,expected);
                return 1;
            }
            ++checks;
        }
    }
    std::printf("lower_index: %zu boundary/random checks PASS\n",checks);
}
'''


def answers(log):
    texts = re.findall(r'=== Generated text ===\n(.*?)\n=== end ===', log, re.S)
    if not texts:
        raise ValueError('No complete generated answer found')
    for text in texts:
        fences = re.findall(r'```(?:cpp|c\+\+|c)?\s*\n(.*?)```', text, re.S)
        yield fences[0].strip() if fences else text.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', type=Path)
    args = parser.parse_args()
    sources = set()
    count = 0
    for path in args.logs:
        for source in answers(path.read_text()):
            count += 1
            sources.add(source)
    root = Path(__file__).resolve().parent / 'tmp'
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='lower-bound-quality-', dir=root) as directory:
        path = Path(directory)
        (path/'driver.cpp').write_text(DRIVER)
        # LeakSanitizer cannot inspect threads under the agent's ptrace sandbox.
        # Keep bounds/use-after-free and undefined-behavior checks enabled.
        env = os.environ | {
            'TMPDIR': str(path),
            'ASAN_OPTIONS': os.environ.get('ASAN_OPTIONS', '') + ':detect_leaks=0',
            'UBSAN_OPTIONS': os.environ.get('UBSAN_OPTIONS', '') + ':halt_on_error=1',
        }
        for source in sorted(sources):
            (path/'answer.cpp').write_text(source)
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                            str(path/'answer.cpp'), str(path/'driver.cpp'),
                            '-o', str(path/'check')], check=True, timeout=30, env=env)
            subprocess.run([str(path/'check')], check=True, timeout=20, env=env)
    print(f'{count} generated answers, {len(sources)} unique implementations PASS')


if __name__ == '__main__':
    main()
