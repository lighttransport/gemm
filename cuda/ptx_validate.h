/*
 * ptx_validate.h - Lightweight PTX string syntax validator
 *
 * Validates PTX assembly strings without a GPU by checking:
 * - Required directives (.version, .target, .address_size)
 * - Entry point declarations (.visible .entry name(...) { ... })
 * - Balanced braces
 * - Register declarations (.reg .type name1, name2, ...)
 * - Shared memory declarations (.shared .align N .type name[size])
 * - Known instruction mnemonics
 * - Parameter declarations (.param .type name)
 *
 * Usage:
 *   ptx_validation_result r;
 *   ptx_validate(ptx_string, &r);
 *   if (r.valid) printf("OK: %d entries\n", r.n_entries);
 *   else printf("Error: %s at line %d\n", r.error, r.error_line);
 */

#ifndef PTX_VALIDATE_H
#define PTX_VALIDATE_H

#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#define PTX_MAX_ENTRIES 32
#define PTX_MAX_REGS    256
#define PTX_MAX_ERROR   256

typedef struct {
    int valid;                          /* 1 if PTX passes all checks */
    char error[PTX_MAX_ERROR];          /* Error description if !valid */
    int error_line;                     /* Line number of first error (1-based) */

    /* Parsed metadata */
    int version_major, version_minor;   /* .version X.Y */
    char target[32];                    /* .target sm_XX */
    int address_size;                   /* .address_size 32|64 */
    int n_entries;                      /* Number of .entry functions */
    char entry_names[PTX_MAX_ENTRIES][64]; /* Entry point names */
    int n_params[PTX_MAX_ENTRIES];      /* Parameter count per entry */
    int n_regs;                         /* Total register declarations */
    int n_shared;                       /* Shared memory declarations */
    int n_instructions;                 /* Instruction count (approximate) */
    int brace_depth_max;                /* Max nesting depth */
} ptx_validation_result;

/* Known PTX instruction prefixes (subset covering common operations) */
static const char *ptx_known_instructions[] = {
    "ld", "st", "mov", "add", "sub", "mul", "mad", "div", "rem",
    "and", "or", "xor", "not", "shl", "shr",
    "setp", "selp", "set",
    "cvt", "cvta",
    "bar", "atom", "red",
    "bra", "ret", "exit", "call",
    "mma", "wmma", "cp",
    "shfl", "vote",
    "min", "max", "abs", "neg",
    "fma", "rcp", "sqrt", "rsqrt", "sin", "cos", "lg2", "ex2",
    "tex", "suld", "sust",
    "membar", "fence",
    "prefetch", "isspacep",
    "tcgen05",
    NULL
};

static int ptx_is_known_instruction(const char *word, int len) {
    for (int i = 0; ptx_known_instructions[i]; i++) {
        int ilen = (int)strlen(ptx_known_instructions[i]);
        if (len >= ilen && memcmp(word, ptx_known_instructions[i], ilen) == 0) {
            /* Match prefix: "ld.param.u64" starts with "ld" */
            if (len == ilen || word[ilen] == '.') return 1;
        }
    }
    return 0;
}

/* (reserved for future use by callers) */

/* Extract next word (alphanumeric + underscore + dot + percent) */
static int ptx_get_word(const char *s, int pos, char *buf, int bufsize) {
    int start = pos;
    while (s[pos] && (isalnum((unsigned char)s[pos]) || s[pos] == '_' ||
                      s[pos] == '.' || s[pos] == '%' || s[pos] == '$')) {
        pos++;
    }
    int len = pos - start;
    if (len >= bufsize) len = bufsize - 1;
    memcpy(buf, s + start, len);
    buf[len] = '\0';
    return pos;
}

static void ptx_validate(const char *ptx, ptx_validation_result *r) {
    memset(r, 0, sizeof(*r));
    r->valid = 1;

    if (!ptx || !ptx[0]) {
        r->valid = 0;
        snprintf(r->error, PTX_MAX_ERROR, "empty PTX string");
        r->error_line = 0;
        return;
    }

    int line = 1;
    int brace_depth = 0;
    int has_version = 0, has_target = 0, has_addr = 0;
    int in_entry = 0; /* Inside an entry function body */
    int in_params = 0; /* Inside parameter list */
    int current_entry = -1;

    const char *p = ptx;
    while (*p) {
        /* Skip whitespace */
        while (*p == ' ' || *p == '\t') p++;

        /* Count newlines */
        if (*p == '\n') {
            line++;
            p++;
            continue;
        }

        /* Skip empty lines and comments */
        if (*p == '\0') break;
        if (*p == '/' && *(p + 1) == '/') {
            while (*p && *p != '\n') p++;
            continue;
        }

        /* Find end of this logical line */
        const char *eol = p;
        while (*eol && *eol != '\n') eol++;

        /* Get first word */
        char word[128];
        int wpos = ptx_get_word(p, 0, word, sizeof(word));
        (void)wpos;

        /* .version directive */
        if (strcmp(word, ".version") == 0) {
            const char *v = p + 8;
            while (*v == ' ') v++;
            if (sscanf(v, "%d.%d", &r->version_major, &r->version_minor) == 2) {
                has_version = 1;
            } else {
                r->valid = 0;
                snprintf(r->error, PTX_MAX_ERROR, "malformed .version directive");
                r->error_line = line;
                return;
            }
        }
        /* .target directive */
        else if (strcmp(word, ".target") == 0) {
            const char *t = p + 7;
            while (*t == ' ') t++;
            int tlen = 0;
            while (t[tlen] && t[tlen] != '\n' && t[tlen] != ',' &&
                   t[tlen] != ' ' && t[tlen] != ';' && tlen < 31) tlen++;
            memcpy(r->target, t, tlen);
            r->target[tlen] = '\0';
            /* Validate target starts with "sm_" */
            if (strncmp(r->target, "sm_", 3) == 0) {
                has_target = 1;
            } else {
                r->valid = 0;
                snprintf(r->error, PTX_MAX_ERROR, "unknown target '%s'", r->target);
                r->error_line = line;
                return;
            }
        }
        /* .address_size directive */
        else if (strcmp(word, ".address_size") == 0) {
            const char *a = p + 13;
            while (*a == ' ') a++;
            r->address_size = atoi(a);
            if (r->address_size == 32 || r->address_size == 64) {
                has_addr = 1;
            } else {
                r->valid = 0;
                snprintf(r->error, PTX_MAX_ERROR, "invalid .address_size %d", r->address_size);
                r->error_line = line;
                return;
            }
        }
        /* .visible .entry */
        else if (strncmp(p, ".visible", 8) == 0) {
            const char *e = p + 8;
            while (*e == ' ') e++;
            if (strncmp(e, ".entry", 6) == 0) {
                e += 6;
                while (*e == ' ') e++;
                /* Extract entry name */
                if (r->n_entries < PTX_MAX_ENTRIES) {
                    current_entry = r->n_entries;
                    int nlen = 0;
                    while (e[nlen] && e[nlen] != '(' && e[nlen] != ' ' &&
                           e[nlen] != '\n' && nlen < 63) nlen++;
                    memcpy(r->entry_names[current_entry], e, nlen);
                    r->entry_names[current_entry][nlen] = '\0';
                    r->n_params[current_entry] = 0;
                    r->n_entries++;
                    in_params = 1;
                }
            }
        }
        /* .param inside entry declaration */
        else if (strncmp(p, ".param", 6) == 0 || (p[0] == ' ' && strstr(p, ".param"))) {
            if (in_params && current_entry >= 0) {
                /* Count a parameter */
                const char *pp = strstr(p, ".param");
                if (pp && pp < eol) {
                    r->n_params[current_entry]++;
                }
            }
        }

        /* Track braces */
        for (const char *bp = p; bp < eol; bp++) {
            if (*bp == '{') {
                brace_depth++;
                if (brace_depth > r->brace_depth_max) {
                    r->brace_depth_max = brace_depth;
                }
                in_entry = 1;
                in_params = 0;
            } else if (*bp == '}') {
                brace_depth--;
                if (brace_depth < 0) {
                    r->valid = 0;
                    snprintf(r->error, PTX_MAX_ERROR, "unmatched closing brace");
                    r->error_line = line;
                    return;
                }
                if (brace_depth == 0) in_entry = 0;
            }
        }

        /* Register declarations */
        if (in_entry && strstr(p, ".reg ") && (const char *)strstr(p, ".reg ") < eol) {
            r->n_regs++;
        }

        /* Shared memory declarations */
        if (strstr(p, ".shared ") && (const char *)strstr(p, ".shared ") < eol) {
            r->n_shared++;
        }

        /* Count instructions (lines that start with a known instruction mnemonic) */
        if (in_entry && word[0] != '.' && word[0] != '/' && word[0] != '\0' &&
            word[0] != '{' && word[0] != '}') {
            /* Check for label (word ending with ':') */
            int wlen = (int)strlen(word);
            if (wlen > 0 && word[wlen - 1] != ':') {
                if (ptx_is_known_instruction(word, wlen)) {
                    r->n_instructions++;
                }
            }
        }

        /* Advance to next line */
        p = eol;
    }

    /* Final checks */
    if (brace_depth != 0) {
        r->valid = 0;
        snprintf(r->error, PTX_MAX_ERROR, "unmatched braces (depth=%d at end)", brace_depth);
        r->error_line = line;
        return;
    }

    if (!has_version) {
        r->valid = 0;
        snprintf(r->error, PTX_MAX_ERROR, "missing .version directive");
        r->error_line = 0;
        return;
    }

    if (!has_target) {
        r->valid = 0;
        snprintf(r->error, PTX_MAX_ERROR, "missing .target directive");
        r->error_line = 0;
        return;
    }

    if (!has_addr) {
        r->valid = 0;
        snprintf(r->error, PTX_MAX_ERROR, "missing .address_size directive");
        r->error_line = 0;
        return;
    }

    if (r->n_entries == 0) {
        r->valid = 0;
        snprintf(r->error, PTX_MAX_ERROR, "no .entry functions found");
        r->error_line = 0;
        return;
    }
}

#endif /* PTX_VALIDATE_H */
