---
authority: null
body_embedding_mode: "gemma"
body_tsv: "'-08':19A '-27':20A '03.740':23A '1':26A '1787859529631':31A '1h':48A '2026':18A '3070':79A '33':22A 'activ':53A 'advanc':46A 'claim':105A 'code':58A 'commit':54A,71A,84A,120A 'commitstal':9A 'common':86A 'dbos':89A 'deploy':63A,68A 'detail':34A 'edit':75A 'emit':29A 'engin':97A 'error':109A 'errorstal':13A 'executor':90A 'fals':12A,14A,16A 'fire':96A 'firestal':11A 'fix':103A 'git':3A,36A,82A,94A,114A 'git-sync':35A,81A,93A,113A 'git-sync-watchdog':2A 'har':6A 'head':43A,72A 'headunchangedhr':25A 'inspect':107A 'kind':1A 'land':57A 'lastfiredat':17A 'laststatus':27A 'lock/restart':100A 'matter':66A 'metadata.last':108A 'papercusp/libs/generic/rerank':8A,112A 'persistentreap':15A 'reach':78A 'reap':91A 'rescu':119A 'rescue-commit':118A 'root':87A 'routin':51A,116A 'ship':69A 'silent':38A 'slug':7A 'stall':39A 'strand':32A,59A,74A,125A 'stuck':92A 'sync':4A,28A,37A,83A,95A,115A 't19':21A 'tree':42A,122A 'true':10A,33A 'un':62A 'un-deploy':61A 'uncommit':60A 'urgent':127A 'watchdog':5A,40A 'wedg':98A 'won':101A 'z':24A"
escalation: "{\"kind\":\"git-sync-watchdog\",\"harness_slug\":\"papercusp/libs/generic/rerank\",\"commitStale\":true,\"fireStale\":false,\"errorStall\":false,\"persistentReap\":false,\"lastFiredAt\":\"2026-08-27T19:33:03.740Z\",\"headUnchangedHrs\":1,\"lastStatus\":\"synced\",\"emitted_at\":1787859529631,\"stranding\":true,\"detail\":\"git-sync silent stall (watchdog): the tree HEAD has not advanced in ~1h while the routine is active — commits are NOT landing (code strands uncommitted + un-deployable). Why it matters: a deploy ships only COMMITTED HEAD, so stranded edits can't reach :3070 until git-sync commits them. Common root: a DBOS executor reaping stuck git-sync fires (engine wedge — a lock/restart won't fix it). Claim it: inspect metadata.last_error on the papercusp/libs/generic/rerank git-sync routine, and rescue-commit the tree if the strand is urgent.\"}"
mtime_ms: 1787859529631
phase: "git-sync-watchdog"
risk_tier: null
supervisor_notes: null
---


