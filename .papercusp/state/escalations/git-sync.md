---
authority: null
body_embedding_mode: "gemma"
body_tsv: "'1787844785289':60A '32':16A,72A 'admiss':30A,34A 'admissionpersistenceerror':32A 'attent':66A 'block':46A 'canon':23A 'commit':48A,75A 'config':21A 'consecut':13A,73A 'could':36A,49A 'declar':44A,55A 'detail':61A 'dirti':52A 'durabl':33A 'emit':58A 'error':5A,14A,17A 'fail':27A,31A,71A 'failur':12A,69A 'generat':43A 'git':3A,63A 'git-sync':62A 'git-sync-error':2A 'har':6A 'inspect':51A 'kind':1A 'messag':22A,42A 'need':65A 'origin':78A 'overs':57A 'papercusp/libs/generic/rerank':8A 'path':53A 'persist':39A 'push':11A,68A,70A 'push-failur':10A,67A 'reach':77A 'reason':9A 'record':35A 'regener':56A 'repair':45A 'scope':18A,40A 'sidecar':29A 'slug':7A 'spawner':28A 'submodul':20A,24A 'submodule-config':19A 'superproject':41A 'sync':4A,26A,64A 'tick':15A,74A 'url':25A"
escalation: "{\"kind\":\"git-sync-error\",\"harness_slug\":\"papercusp/libs/generic/rerank\",\"reasons\":[\"push-failure\"],\"consecutive_error_ticks\":32,\"errors\":[{\"scope\":\"submodule-config\",\"message\":\"canonical submodule URL sync failed: spawner sidecar admission failed: AdmissionPersistenceError: durable admission record could not be persisted\"},{\"scope\":\"superproject\",\"message\":\"generated declaration repair blocked the commit: could not inspect dirty paths before declaration regeneration\"}],\"oversized\":[],\"emitted_at\":1787844785289,\"detail\":\"git-sync needs attention: push-failure — push failed 32 consecutive ticks (commits NOT reaching origin)\"}"
mtime_ms: 1787844785289
phase: "git-sync"
risk_tier: null
supervisor_notes: null
---


