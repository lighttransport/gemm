"""Basic-emotion names (expression_catalog.EMOTIONS) -> exprdata.EXPRESSIONS control weights.

The expression catalog and reconstruction.create use the 7 Ekman-style names; the rig's authored prompt set
(exprdata.EXPRESSIONS) is control-based. This table is the documented correspondence used by the Cycles A/B
study for wrinkle-library drivers and review labels. Weights are authored priors (0..1), not fitted values.
"""
EMOTION_TO_EXPRESSIONS = {
    'neutral':  {},
    'happy':    {'smile': 1.0, 'squint': 0.4},
    'sad':      {'frown': 1.0, 'brows_up': 0.3},
    'angry':    {'brows_down': 1.0, 'press': 0.6, 'squint': 0.3},
    'fear':     {'brows_up': 1.0, 'stretch': 0.7, 'jaw_open': 0.3},
    'surprise': {'brows_up': 1.0, 'jaw_open': 0.8},
    'disgust':  {'sneer': 1.0, 'brows_down': 0.5, 'squint': 0.3},
}
# Wrinkle library channels expected to activate per emotion (see reconstruction.wrinkle_library.WRINKLES)
EMOTION_WRINKLES = {
    'neutral': [], 'happy': ['crows_feet', 'nasolabial'], 'sad': ['glabella', 'chin'],
    'angry': ['glabella', 'nasal_root'], 'fear': ['forehead'], 'surprise': ['forehead'],
    'disgust': ['nasal_root', 'glabella', 'nasolabial'],
}
# Open-mouth emotions get mouth close-up renders in the study
OPEN_MOUTH = ('happy', 'surprise', 'fear')


def controls(emotion):
    """Merged ARKit-style control dict for an emotion from exprdata.EXPRESSIONS (name -> (prompt, controls, group))."""
    from .exprdata import EXPRESSIONS
    merged = {}
    for name, weight in EMOTION_TO_EXPRESSIONS[emotion].items():
        for key, value in EXPRESSIONS[name][1].items():
            merged[key] = max(merged.get(key, 0.), float(value) * weight)
    return merged
