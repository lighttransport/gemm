/* GLM5.3 Flash chat-template subset for C callers.
 *
 * This mirrors the text-only path in models/glm53f/chat_template.jinja:
 * [gMASK]<sop>, a reasoning-effort system line, role markers, and the
 * assistant generation prompt. Tool-call serialization is intentionally left
 * to the later agent/runtime integration; callers can pass rendered tool
 * text as an observation message.
 */
#ifndef GLM5_CHAT_TEMPLATE_H
#define GLM5_CHAT_TEMPLATE_H

#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    const char *role;             /* system, user, assistant, tool */
    const char *content;          /* already normalized UTF-8 text */
    const char *reasoning_content;/* optional assistant reasoning */
} glm5_chat_message;

static int glm5_chat_put(char *out, size_t cap, size_t *pos,
                         const char *text) {
    size_t n = strlen(text);
    if (*pos >= cap || n > cap - *pos - 1) return -1;
    memcpy(out + *pos, text, n);
    *pos += n;
    out[*pos] = '\0';
    return 0;
}

static int glm5_chat_put_content(char *out, size_t cap, size_t *pos,
                                 const char *text) {
    return glm5_chat_put(out, cap, pos, text ? text : "");
}

/* Render the supported GLM5.3 text chat template.
 * Returns output length, or -1 if the output buffer is too small/invalid.
 */
static int glm5_chat_template_render(const glm5_chat_message *messages,
                                     size_t n_messages,
                                     const char *reasoning_effort,
                                     int add_generation_prompt,
                                     char *out, size_t cap) {
    size_t pos = 0;
    const char *effort = reasoning_effort ? reasoning_effort : "max";
    char effort_cap[16];
    if (!out || cap == 0 || (!messages && n_messages != 0)) return -1;
    if (strlen(effort) == 0 || strlen(effort) >= sizeof(effort_cap)) return -1;
    strcpy(effort_cap, effort);
    if (effort_cap[0] >= 'a' && effort_cap[0] <= 'z') effort_cap[0] -= 'a' - 'A';
    out[0] = '\0';
    if (glm5_chat_put(out, cap, &pos, "[gMASK]<sop>") < 0 ||
        glm5_chat_put(out, cap, &pos, "<|system|>Reasoning Effort: ") < 0 ||
        glm5_chat_put(out, cap, &pos, effort_cap) < 0) return -1;

    for (size_t i = 0; i < n_messages; ++i) {
        const glm5_chat_message *m = &messages[i];
        const char *marker = NULL;
        if (!m->role) return -1;
        if (strcmp(m->role, "system") == 0) marker = "<|system|>";
        else if (strcmp(m->role, "user") == 0) marker = "<|user|>";
        else if (strcmp(m->role, "assistant") == 0) marker = "<|assistant|>";
        else if (strcmp(m->role, "tool") == 0) marker = "<|observation|>";
        else return -1;
        if (glm5_chat_put(out, cap, &pos, marker) < 0 ||
            glm5_chat_put_content(out, cap, &pos, m->content) < 0) return -1;
        if (strcmp(m->role, "assistant") == 0 && m->reasoning_content) {
            if (glm5_chat_put(out, cap, &pos, "<think>") < 0 ||
                glm5_chat_put(out, cap, &pos, m->reasoning_content) < 0 ||
                glm5_chat_put(out, cap, &pos, "</think>") < 0) return -1;
        }
    }
    if (add_generation_prompt &&
        (glm5_chat_put(out, cap, &pos, "<|assistant|><think>") < 0)) return -1;
    return (int)pos;
}

#endif
