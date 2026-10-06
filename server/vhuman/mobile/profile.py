"""Explicit quality targets, not claims of measured device performance."""
IPHONE12 = dict(schema='vhuman.mobile_profile.v1', name='iphone12',
    minimum_ios='16.4', target_fps=30, internal_size=[720,1280],
    minimum_size=[540,960], triangle_budgets=[80000,40000,15000],
    face_texture_size=2048, detail_texture_size=1024,
    renderer_memory_mib=350, total_memory_mib=1536,
    lip_sync_p95_ms=80, soak_minutes=20,
    local_llm=False,local_tts=False)
