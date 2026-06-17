import json
p='scripts/e2e_nlp_results.json'
with open(p,'r',encoding='utf-8') as f:
    data=json.load(f)
convs=[item for item in data if isinstance(item, dict) and 'conversation_id' in item]
print('conversations:', len(convs))
for c in convs:
    print('Conversation', c['conversation_id'])
    for step in c['steps']:
        print(' step', step['step'], 'message:', step['message'], 'intent:', step.get('intent'), 'missing:', step.get('missing'), 'clarify:', step.get('clarification'))
    print()