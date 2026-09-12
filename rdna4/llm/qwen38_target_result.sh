#!/usr/bin/env bash
# Internal footer parser shared by the live benchmark and CPU-only tests.
qwen38_target_result() {
    local log_file="$1" repeats="$2" profile="$3" expected_first="${4:-}" expected_hash="${5:-}"
    local result_lines unique_hashes unique_firsts vram_line pf_stats dec_stats e2e_stats
    local pf_min pf_med dec_min dec_med e2e_min e2e_med
    local -a first_tokens hashes
    result_lines="$(grep -c 'Result: PASS' "${log_file}" || true)"
    mapfile -t first_tokens < <(grep -oE 'First decoded token id=-?[0-9]+' "${log_file}" | sed 's/.*=//')
    mapfile -t hashes < <(grep -oE 'sequence hash=[0-9a-f]+' "${log_file}" | sed 's/sequence hash=//')

    if (( ${#hashes[@]} != repeats )); then
        echo "target gate FAIL: expected ${repeats} sequence-hash footers, saw ${#hashes[@]} (timed out or crashed mid-repeat?)" >&2
        grep -E 'Prefill:|Decode:|End-to-end:|Result:' "${log_file}" | tail -20 >&2 || true
        return 1
    fi
    if (( ${#first_tokens[@]} != repeats )); then
        echo "target gate FAIL: expected ${repeats} first-token footers, saw ${#first_tokens[@]}" >&2
        return 1
    fi
    if (( result_lines != repeats )); then
        echo "target gate FAIL: ${result_lines}/${repeats} repeats returned 'Result: PASS'" >&2
        return 1
    fi

    unique_hashes="$(printf '%s\n' "${hashes[@]}" | sort -u | wc -l)"
    unique_firsts="$(printf '%s\n' "${first_tokens[@]}" | sort -u | wc -l)"
    if (( unique_hashes != 1 )); then
        echo "target gate FAIL: nondeterministic sequence hash across ${repeats} repeats: ${hashes[*]}" >&2
        return 1
    fi
    if (( unique_firsts != 1 )); then
        echo "target gate FAIL: nondeterministic first token across ${repeats} repeats: ${first_tokens[*]}" >&2
        return 1
    fi

    if [[ -n "${expected_hash}" ]]; then
        if [[ "${hashes[0]}" != "${expected_hash}" || "${first_tokens[0]}" != "${expected_first}" ]]; then
            echo "target gate FAIL: scalar reference mismatch: expected first=${expected_first} hash=${expected_hash}; got first=${first_tokens[0]} hash=${hashes[0]}" >&2
            return 1
        fi
        echo "target gate reference parity: PASS"
    fi

    # Throughput: summarize min + median over repeats (clock-variance safe) rather
    # than reporting a single best sample.  tok/s is the value after "->".
    tok_s_values() {
        grep -E "^$1:" "${log_file}" | sed -E 's/.*->[[:space:]]*([0-9.]+)[[:space:]]*tok\/s.*/\1/'
    }
    min_median() {
        awk '
            function med(a, n,   b,i,j,t){ for(i=0;i<n;i++)b[i]=a[i]; for(i=0;i<n-1;i++)for(j=i+1;j<n;j++)if(b[j]<b[i]){t=b[i];b[i]=b[j];b[j]=t} return (n%2)?b[int(n/2)]:(b[n/2-1]+b[n/2])/2 }
            { v[n++]=$1; if(n==1||$1<mn)mn=$1 }
            END { if(n) printf "%.2f %.2f", mn, med(v,n); else printf "0 0" }'
    }
    vram_line="$(grep -E '^VRAM:' "${log_file}" | tail -1 || true)"

    pf_stats="$(tok_s_values Prefill | min_median)"
    dec_stats="$(tok_s_values Decode | min_median)"
    e2e_stats="$(tok_s_values End-to-end | min_median)"
    read -r pf_min pf_med <<<"${pf_stats}"
    read -r dec_min dec_med <<<"${dec_stats}"
    read -r e2e_min e2e_med <<<"${e2e_stats}"

    echo "target gate PASS: profile=${profile} repeats=${repeats} hash=${hashes[0]} first_token=${first_tokens[0]}"
    echo "  prefill tok/s: min=${pf_min} median=${pf_med}"
    echo "  decode  tok/s: min=${dec_min} median=${dec_med}"
    echo "  e2e     tok/s: min=${e2e_min} median=${e2e_med}"
    [[ -n "${vram_line}" ]] && echo "  ${vram_line}"
    echo "  log=${log_file}"
}
