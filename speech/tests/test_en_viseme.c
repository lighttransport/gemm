/* en_align: IPA token -> viseme classes (affricates vs aspirated/palatalised stops). */
#include "../en_align/en_phoneme.h"

static int check(const char *tok, int want) {
    int got = en_viseme(tok);
    if (got != want) { printf("FAIL %s -> %s (want %s)\n", tok, ja_viseme_names[got], ja_viseme_names[want]); return 1; }
    return 0;
}

int main(void) {
    int bad = 0;
    bad += check("tʃ", JA_VIS_CH) + check("dʒ", JA_VIS_CH) + check("ts", JA_VIS_CH) + check("tɕ", JA_VIS_CH) + check("dʑ", JA_VIS_CH);
    bad += check("t", JA_VIS_DD) + check("d", JA_VIS_DD) + check("tʰ", JA_VIS_DD) + check("tʲ", JA_VIS_DD) + check("dʲ", JA_VIS_DD) + check("dʰ", JA_VIS_DD);
    bad += check("p", JA_VIS_PP) + check("m", JA_VIS_PP) + check("f", JA_VIS_FF) + check("θ", JA_VIS_TH) + check("ð", JA_VIS_TH);
    bad += check("aʊ", JA_VIS_OU) + check("aɪ", JA_VIS_AA) + check("ɹ", JA_VIS_RR) + check("ŋ", JA_VIS_NN) + check("<pad>", JA_VIS_SIL);
    printf(bad ? "%d FAILED\n" : "ALL OK\n", bad);
    return bad != 0;
}
