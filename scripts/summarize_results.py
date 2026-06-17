import json
p='scripts/e2e_nlp_results.json'
with open(p,'r',encoding='utf-8') as f:
    data=json.load(f)
single=[]
convs=[]
for item in data:
    if isinstance(item, dict) and 'index' in item:
        single.append(item)
    else:
        convs.append(item)
errors = [s for s in single if 'error' in s]
clarify = [s for s in single if s.get('clarification')]
missing = [s for s in single if s.get('missing')]
print('total_single',len(single))
print('total_conversations',len(convs))
print('errors',len(errors))
print('clarification_needed',len(clarify))
print('have_missing_slots',len(missing))
print('\nExamples needing clarification or missing:')
for s in (clarify+missing)[:10]:
    print(json.dumps({'index':s.get('index'),'query':s.get('query'),'intent':s.get('intent'),'missing':s.get('missing'),'clarification':s.get('clarification')},ensure_ascii=False))
