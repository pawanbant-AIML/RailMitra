import json
from collections import Counter, defaultdict
p='scripts/e2e_nlp_results.json'
with open(p,'r',encoding='utf-8') as f:
    data=json.load(f)

single=[i for i in data if isinstance(i,dict) and 'index' in i]
fails=[s for s in single if s.get('clarification') or (s.get('missing'))]
count=len(fails)
by_intent=Counter()
missing_counter=Counter()
examples=defaultdict(list)
for s in fails:
    intent=s.get('intent') or 'UNKNOWN'
    by_intent[intent]+=1
    miss=tuple(sorted(s.get('missing') or []))
    missing_counter.update(miss)
    if len(examples[intent])<5:
        examples[intent].append({'index':s['index'],'query':s['query'],'missing':s.get('missing'),'clarification':s.get('clarification')})

print('total_failing_cases:',count)
print('\nTop intents with failures:')
for k,v in by_intent.most_common(20):
    print(k,v)
print('\nTop missing slot keys:')
for k,v in missing_counter.most_common(20):
    print(k,v)
print('\nSample failures by intent:')
for intent,ex in examples.items():
    print('\n==',intent,'==')
    for e in ex:
        print(e)
