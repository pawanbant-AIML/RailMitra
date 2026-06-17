import json
import os
import sys
# Ensure repo root on python path for package-style imports
sys.path.insert(0, os.getcwd())

try:
    from enhanced_chat_nlp_service import ChatNLPService, ChatAnalysisRequest
except Exception:
    # Fallback: load QueryUnderstanding directly from backend path and provide a minimal wrapper
    import importlib.util
    qu_path = os.path.join(os.getcwd(), 'backend', 'app', 'agent', 'query_understanding.py')
    spec = importlib.util.spec_from_file_location('query_understanding', qu_path)
    qumod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qumod)
    QueryUnderstanding = getattr(qumod, 'QueryUnderstanding')

    # Minimal wrapper that provides analyze(request) similar to ChatNLPService
    class ChatAnalysisRequest:
        def __init__(self, user_message, conversation_history):
            self.user_message = user_message
            self.conversation_history = conversation_history

    class SimpleChatNLPService:
        def __init__(self):
            self._qu = QueryUnderstanding()

        def analyze(self, req: ChatAnalysisRequest):
            text = (req.user_message or '').strip()
            memory = {}
            previous_result = {}
            for msg in (req.conversation_history or []):
                if msg.get('role') == 'user' and msg.get('content'):
                    part = msg.get('content')
                    interp_part = self._qu.interpret(part, memory=memory, previous_result=previous_result)
                    try:
                        memory.update({k: v for k, v in (interp_part.slots.__dict__ if interp_part and interp_part.slots else {}).items() if v not in (None, '', [])})
                    except Exception:
                        pass
                    previous_result = interp_part.to_dict() if interp_part else {}
            interp = self._qu.interpret(text, memory=memory, previous_result=previous_result)
            # Build a response-like simple object
            class Resp:
                pass
            r = Resp()
            r.intent = interp.intent
            r.sub_intents = getattr(interp, 'sub_intents', None)
            r.slots = interp.slots
            r.clarification_needed = getattr(interp, 'clarification_needed', False)
            r.missing_slots = getattr(interp, 'missing_slots', None) or getattr(interp, 'missing_slots', None) or getattr(interp, 'missing', None) or getattr(interp, 'missing_slots', None)
            r.missing_required_slots = getattr(interp, 'missing_slots', None)
            r.next_action = None
            r.confidence = getattr(interp, 'confidence', 0.0)
            r.normalized_text = getattr(interp, 'normalized_text', None)
            return r

    ChatAnalysisRequest = ChatAnalysisRequest
    ChatNLPService = SimpleChatNLPService

svc = ChatNLPService()

# Base queries from user categories (condensed)
base_queries = [
    # 1. Basic train search
    "Show trains from Bangalore to Mangalore",
    "Find trains from Bengaluru to Mangaluru",
    "Find trains between Bangalore and Mangalore",
    "Show trains from Pune to Hyderabad",
    "Show trains from Delhi to Chennai",
    "Show trains from Mysore to Bangalore",
    "Show trains from Mangalore to Bangalore",
    "Any trains from Chennai to Hyderabad?",
    "Train from Bangalore to Goa",
    "Train from Hubli to Bangalore",
    # 2. Spelling mistakes
    "Train from Banglore to Manglore",
    "Train from Bengluru to Mangaloor",
    "Train from Hydrabad to Pune",
    "Train from Delhii to Chennai",
    "Train from Mysuru to Bangluru",
    # 3. Station aliases
    "Train from Bengaluru to Mangaluru",
    "Train from Bombay to Delhi",
    "Train from Madras to Bangalore",
    "Train from Calcutta to Delhi",
    "Train from Trivandrum to Kochi",
    "Train from Vizag to Chennai",
    # 4. Train number search
    "Tell me about train 12627",
    "Tell me about train 16585",
    "Route of train 12657",
    "Train details 12627",
    "Show stops for train 12657",
    "How long is train 12657",
    # 5. Route queries
    "Route from Bangalore to Mangalore",
    "Best route from Pune to Hyderabad",
    "Direct route from Bangalore to Goa",
    "Any direct train from Mysore to Chennai",
    "Show stations between Bangalore and Mangalore",
    "Show all stops for train 12627",
    # 6. Fare queries
    "Fare from Bangalore to Mangalore",
    "Sleeper fare from Bangalore to Mangalore",
    "3A fare from Bangalore to Mangalore",
    "General ticket cost Bangalore to Mangalore",
    "Cheapest fare Bangalore to Mangalore",
    "Compare Sleeper and 3A",
    "Compare 2A and 3A",
    # 7. Passenger calculations
    "Fare for 2 passengers",
    "Fare for 3 passengers",
    "Fare for family of 4",
    "Fare for 5 adults",
    "Fare for 2 sleeper tickets",
    "Cost for 3 AC tickets",
    # 8. Cheapest train
    "Which train is cheapest?",
    "Cheapest train Bangalore to Mangalore",
    "Lowest fare Bangalore to Chennai",
    "Most economical train",
    "Budget train Bangalore to Goa",
    # 9. Fastest train
    "Fastest train Bangalore to Mangalore",
    "Quickest route to Chennai",
    "Train with shortest journey",
    "Which train reaches first?",
    "Fastest option Pune to Hyderabad",
    # 10. Time understanding
    "Show trains after 8 PM",
    "Show trains before 6 AM",
    "Show trains between 7 PM and 10 PM",
    "Morning trains",
    "Evening trains",
    "Night trains",
    "Overnight trains",
    "Trains leaving after 9 PM",
    "Trains arriving before 8 AM",
    # 11. Date understanding
    "Train tomorrow",
    "Train today",
    "Train tonight",
    "Train next Monday",
    "Train this weekend",
    "Train on Friday",
    "Train after 2 days",
    "Train on 25th December",
    # 12. Booking flow
    "Book a ticket",
    "Book a sleeper ticket",
    "Book 2 tickets",
    "Book 3 tickets",
    "Book a 3A ticket",
    "Book 2 sleeper tickets",
    "Book ticket Bangalore to Hyderabad",
    "Book ticket Pune to Hyderabad tomorrow",
    # 13. Booking + time
    "Book train after 8 PM",
    "Book sleeper after 8 PM",
    "Book 3A after 9 PM",
    "Book overnight train",
    "Book morning train",
    "Book train tomorrow after 8 PM",
    "Book 2 tickets tomorrow evening",
    # 14. Booking + route + class
    "Book 2 sleeper tickets from Bangalore to Hyderabad",
    "Book 3A ticket from Pune to Hyderabad",
    "Book AC ticket Bangalore to Chennai",
    "Book 2A Bangalore to Goa",
    "Book sleeper Bangalore to Mangalore tomorrow",
    # 15. Booking follow up (as conversation)
]

# Conversation-style test sequences for memory and follow-ups
conversations = [
    ["Book a ticket from Bangalore to Hyderabad", "Tomorrow", "3A", "2 passengers", "Confirm booking"],
    ["Show trains Bangalore to Mangalore", "Which is cheapest?", "Show fare", "Book the first one", "Change to sleeper class", "Confirm booking"],
]

# Ambiguous and failure tests
ambiguous = [
    "I need a train",
    "Book a ticket",
    "Show options",
    "Need cheapest one",
    "Need fastest one",
    "Need something comfortable",
    "Find a train for me",
    "Train from XYZ to ABC",
    "Train from FakeCity to Bangalore",
    "Train from Bangalore to UnknownStation",
]

# Stress tests / compound
stress = [
    "Find cheapest sleeper train from Bangalore to Mangalore tomorrow after 8 PM and compare it with the fastest option and tell me which is better for a family of four.",
    "Book two 3A tickets from Bangalore to Hyderabad tomorrow evening and show total fare.",
    "Recommend an overnight train from Pune to Hyderabad under ₹1000.",
]

# Build expanded set to exceed 150 by generating slight variants
all_queries = list(base_queries)
for q in base_queries[:40]:
    all_queries.append(q + " please")
for q in base_queries[10:50]:
    all_queries.append(q.replace("Bangalore", "Bengaluru").replace("Mangalore", "Mangaluru"))
for i in range(10):
    all_queries.append(f"Train from Bangalore to Mangalore on day {i+1}")

all_queries.extend(sum([[m] for m in ambiguous], []))
all_queries.extend(sum([[s] for s in stress], []))
# Add conversations as separate multi-step tests

# Ensure uniqueness and length
seen = set()
unique_queries = []
for q in all_queries:
    if q not in seen:
        seen.add(q)
        unique_queries.append(q)

# Add standalone conversations as labeled tests
results = []

os.makedirs('scripts', exist_ok=True)

print(f"Running {len(unique_queries)} single-turn NLP tests and {len(conversations)} conversation tests...")

# Run single-turn tests
for idx, q in enumerate(unique_queries, start=1):
    req = ChatAnalysisRequest(user_message=q, conversation_history=[])
    try:
        resp = svc.analyze(req)
        out = {
            'index': idx,
            'query': q,
            'intent': resp.intent,
            'entities': resp.entities.__dict__ if resp.entities else {},
            'missing': resp.missing_slots,
            'clarification': resp.clarification_needed,
            'next_action': resp.next_action,
            'confidence': resp.confidence,
            'normalized_text': resp.normalized_text,
        }
    except Exception as e:
        out = {'index': idx, 'query': q, 'error': str(e)}
    print(json.dumps(out, ensure_ascii=False))
    results.append(out)

# Run conversation tests
for cidx, convo in enumerate(conversations, start=1):
    history = []
    conv_results = []
    for step_idx, msg in enumerate(convo, start=1):
        req = ChatAnalysisRequest(user_message=msg, conversation_history=[{'role': 'user', 'content': m} for m in history])
        try:
            resp = svc.analyze(req)
            out = {
                'conversation': cidx,
                'step': step_idx,
                'message': msg,
                'intent': resp.intent,
                'entities': resp.entities.__dict__ if resp.entities else {},
                'missing': resp.missing_slots,
                'clarification': resp.clarification_needed,
                'next_action': resp.next_action,
                'confidence': resp.confidence,
                'normalized_text': resp.normalized_text,
            }
        except Exception as e:
            out = {'conversation': cidx, 'step': step_idx, 'message': msg, 'error': str(e)}
        print(json.dumps(out, ensure_ascii=False))
        conv_results.append(out)
        history.append(msg)
    results.append({'conversation_id': cidx, 'steps': conv_results})

# Save results
with open('scripts/e2e_nlp_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print('Done. Results written to scripts/e2e_nlp_results.json')
