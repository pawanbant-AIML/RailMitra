import sys
sys.path.insert(0, '.')
from enhanced_chat_nlp_service import ChatNLPService, ChatAnalysisRequest
try:
    from app.agent.query_understanding import QueryUnderstanding
except Exception:
    from backend.app.agent.query_understanding import QueryUnderstanding

svc = ChatNLPService()
req = ChatAnalysisRequest(user_message='Vizag to Hyderabad ac ticket', conversation_history=[])
r = svc.analyze(req)
print('intent', r.intent)
print('entities', r.entities)
print('clarification', getattr(r,'clarification_needed',None), 'missing', getattr(r,'missing_required_slots',None))
qu = QueryUnderstanding()
interp = qu.interpret('Vizag to Hyderabad ac ticket')
print('interp slots', interp.slots)
print('normalized', interp.normalized_text)
