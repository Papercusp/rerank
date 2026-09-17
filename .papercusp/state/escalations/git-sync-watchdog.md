---
authority: null
body_embedding_mode: "gemma"
body_embedding_profile: null
body_tsv: "'-09':19A '-17':20A '0':26A '04.523':23A '08':22A '1789633614414':31A '2026':18A '3':51A '3070':74A '3h':47A 'activ':48A 'claim':100A 'commit':66A,79A,115A 'commitstal':9A 'common':81A 'cron':49A 'dbos':84A 'deploy':63A 'detail':34A 'edit':70A 'emit':29A 'engin':92A 'error':104A 'errorstal':13A 'everi':50A 'executor':85A 'fals':10A,14A,16A 'fire':45A,91A 'firestal':11A 'fix':98A 'git':3A,36A,77A,89A,109A 'git-sync':35A,76A,88A,108A 'git-sync-watchdog':2A 'har':6A 'head':67A 'headunchangedhr':25A 'inspect':102A 'kind':1A 'lastfiredat':17A 'laststatus':27A 'lock/restart':95A 'matter':61A 'metadata.last':103A 'min':52A 'noth':28A 'papercusp/libs/generic/rerank':8A,107A 'persistentreap':15A 'reach':73A 'reap':86A 'rescu':114A 'rescue-commit':113A 'root':82A 'routin':42A,111A 'run':57A 'scheduler/engine':54A 'ship':64A 'silent':38A 'slug':7A 'stall':39A 'strand':32A,69A,120A 'stuck':87A 'sync':4A,37A,78A,90A,110A 't05':21A 'tree':117A 'true':12A,33A 'urgent':122A 'watchdog':5A,40A 'wedg':93A 'won':96A 'z':24A"
escalation: "{\"kind\":\"git-sync-watchdog\",\"harness_slug\":\"papercusp/libs/generic/rerank\",\"commitStale\":false,\"fireStale\":true,\"errorStall\":false,\"persistentReap\":false,\"lastFiredAt\":\"2026-09-17T05:08:04.523Z\",\"headUnchangedHrs\":0,\"lastStatus\":\"nothing\",\"emitted_at\":1789633614414,\"stranding\":true,\"detail\":\"git-sync silent stall (watchdog): the routine has not FIRED in ~3h (active, cron every 3 min) — the scheduler/engine is not running it. Why it matters: a deploy ships only COMMITTED HEAD, so stranded edits can't reach :3070 until git-sync commits them. Common root: a DBOS executor reaping stuck git-sync fires (engine wedge — a lock/restart won't fix it). Claim it: inspect metadata.last_error on the papercusp/libs/generic/rerank git-sync routine, and rescue-commit the tree if the strand is urgent.\"}"
mtime_ms: 1789633614414
phase: "git-sync-watchdog"
risk_tier: null
supervisor_notes: null
---


