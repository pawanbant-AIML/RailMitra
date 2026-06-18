import json
import os
import sys

# Make sure the project root is on sys.path
sys.path.insert(0, os.getcwd())

# -------------------------------------------------------------------
#  Import QueryUnderstanding – use standard import first;
#  fall back to importlib with proper module metadata if needed.
# -------------------------------------------------------------------
try:
    from backend.app.agent.query_understanding import QueryUnderstanding
except Exception:
    import importlib.util
    qu_path = os.path.join(os.getcwd(), 'backend', 'app', 'agent', 'query_understanding.py')
    spec = importlib.util.spec_from_file_location('query_understanding', qu_path)
    qumod = importlib.util.module_from_spec(spec)
    # Give the module a proper name so dataclasses work
    qumod.__name__ = 'query_understanding'
    qumod.__package__ = 'backend.app.agent'
    # Fake a sys.modules entry for good measure
    sys.modules['backend.app.agent.query_understanding'] = qumod
    spec.loader.exec_module(qumod)
    QueryUnderstanding = qumod.QueryUnderstanding


class ChatAnalysisRequest:
    def __init__(self, user_message, conversation_history):
        self.user_message = user_message
        self.conversation_history = conversation_history


class SimpleChatNLPService:
    def __init__(self):
        self._qu = QueryUnderstanding()

    def analyze(self, req: ChatAnalysisRequest):
        text = (req.user_message or '').strip()
        memory = {"slots": {}}            # slots under 'slots' key
        previous_result = {}

        for msg in (req.conversation_history or []):
            if msg.get('role') == 'user' and msg.get('content'):
                part = msg.get('content')
                interp_part = self._qu.interpret(part, memory=memory, previous_result=previous_result)
                try:
                    slot_dict = interp_part.slots.__dict__ if interp_part and interp_part.slots else {}
                    for k, v in slot_dict.items():
                        if v not in (None, '', []):
                            memory["slots"][k] = v
                except Exception:
                    pass
                previous_result = interp_part.to_dict() if interp_part else {}

        interp = self._qu.interpret(text, memory=memory, previous_result=previous_result)

        class Resp:
            pass
        r = Resp()
        r.intent = interp.intent
        r.sub_intents = getattr(interp, 'sub_intents', None)
        r.slots = interp.slots                     # QuerySlots object
        r.clarification_needed = getattr(interp, 'clarification_needed', False)
        r.missing_slots = getattr(interp, 'missing_slots', [])
        r.missing_required_slots = getattr(interp, 'missing_slots', [])
        r.next_action = None
        r.confidence = getattr(interp, 'confidence', 0.0)
        r.normalized_text = getattr(interp, 'normalized_text', None)
        return r


svc = SimpleChatNLPService()

# -------------------------------------------------------------------
#  Test queries
# -------------------------------------------------------------------
base_queries = [
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
    "Train from Banglore to Manglore",
    "Train from Bengluru to Mangaloor",
    "Train from Hydrabad to Pune",
    "Train from Delhii to Chennai",
    "Train from Mysuru to Bangluru",
    "Train from Bengaluru to Mangaluru",
    "Train from Bombay to Delhi",
    "Train from Madras to Bangalore",
    "Train from Calcutta to Delhi",
    "Train from Trivandrum to Kochi",
    "Train from Vizag to Chennai",
    "Tell me about train 12627",
    "Tell me about train 16585",
    "Route of train 12657",
    "Train details 12627",
    "Show stops for train 12657",
    "How long is train 12657",
    "Route from Bangalore to Mangalore",
    "Best route from Pune to Hyderabad",
    "Direct route from Bangalore to Goa",
    "Any direct train from Mysore to Chennai",
    "Show stations between Bangalore and Mangalore",
    "Show all stops for train 12627",
    "Fare from Bangalore to Mangalore",
    "Sleeper fare from Bangalore to Mangalore",
    "3A fare from Bangalore to Mangalore",
    "General ticket cost Bangalore to Mangalore",
    "Cheapest fare Bangalore to Mangalore",
    "Compare Sleeper and 3A",
    "Compare 2A and 3A",
    "Fare for 2 passengers",
    "Fare for 3 passengers",
    "Fare for family of 4",
    "Fare for 5 adults",
    "Fare for 2 sleeper tickets",
    "Cost for 3 AC tickets",
    "Which train is cheapest?",
    "Cheapest train Bangalore to Mangalore",
    "Lowest fare Bangalore to Chennai",
    "Most economical train",
    "Budget train Bangalore to Goa",
    "Fastest train Bangalore to Mangalore",
    "Quickest route to Chennai",
    "Train with shortest journey",
    "Which train reaches first?",
    "Fastest option Pune to Hyderabad",
    "Show trains after 8 PM",
    "Show trains before 6 AM",
    "Show trains between 7 PM and 10 PM",
    "Morning trains",
    "Evening trains",
    "Night trains",
    "Overnight trains",
    "Trains leaving after 9 PM",
    "Trains arriving before 8 AM",
    "Train tomorrow",
    "Train today",
    "Train tonight",
    "Train next Monday",
    "Train this weekend",
    "Train on Friday",
    "Train after 2 days",
    "Train on 25th December",
    "Book a ticket",
    "Book a sleeper ticket",
    "Book 2 tickets",
    "Book 3 tickets",
    "Book a 3A ticket",
    "Book 2 sleeper tickets",
    "Book ticket Bangalore to Hyderabad",
    "Book ticket Pune to Hyderabad tomorrow",
    "Book train after 8 PM",
    "Book sleeper after 8 PM",
    "Book 3A after 9 PM",
    "Book overnight train",
    "Book morning train",
    "Book train tomorrow after 8 PM",
    "Book 2 tickets tomorrow evening",
    "Book 2 sleeper tickets from Bangalore to Hyderabad",
    "Book 3A ticket from Pune to Hyderabad",
    "Book AC ticket Bangalore to Chennai",
    "Book 2A Bangalore to Goa",
    "Book sleeper Bangalore to Mangalore tomorrow",
]

conversations = [
    ["Book a ticket from Bangalore to Hyderabad", "Tomorrow", "3A", "2 passengers", "Confirm booking"],
    ["Show trains Bangalore to Mangalore", "Which is cheapest?", "Show fare", "Book the first one", "Change to sleeper class", "Confirm booking"],
]

ambiguous = [
    "I need a train",
    "Show options",
    "Need cheapest one",
    "Need fastest one",
    "Need something comfortable",
    "Find a train for me",
    "Train from XYZ to ABC",
    "Train from FakeCity to Bangalore",
    "Train from Bangalore to UnknownStation",
]

stress = [
    "Find cheapest sleeper train from Bangalore to Mangalore tomorrow after 8 PM and compare it with the fastest option and tell me which is better for a family of four.",
    "Book two 3A tickets from Bangalore to Hyderabad tomorrow evening and show total fare.",
    "Recommend an overnight train from Pune to Hyderabad under ₹1000.",
]

# Build expanded set
all_queries = list(base_queries)
for q in base_queries[:40]:
    all_queries.append(q + " please")
for q in base_queries[10:50]:
    all_queries.append(q.replace("Bangalore", "Bengaluru").replace("Mangalore", "Mangaluru"))
for i in range(10):
    all_queries.append(f"Train from Bangalore to Mangalore on day {i+1}")

all_queries.extend(ambiguous)
all_queries.extend(stress)

seen = set()
unique_queries = []
for q in all_queries:
    if q not in seen:
        seen.add(q)
        unique_queries.append(q)

os.makedirs('scripts', exist_ok=True)

print(f"Running {len(unique_queries)} single-turn NLP tests and {len(conversations)} conversation tests...")
results = []

# -------------------------------------------------------------------
#  Single-turn tests
# -------------------------------------------------------------------
for idx, q in enumerate(unique_queries, start=1):
    req = ChatAnalysisRequest(user_message=q, conversation_history=[])
    try:
        resp = svc.analyze(req)
        from dataclasses import asdict
        out = {
            'index': idx,
            'query': q,
            'intent': resp.intent,
            'entities': asdict(resp.slots),
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

# -------------------------------------------------------------------
#  Conversation tests
# -------------------------------------------------------------------
for cidx, convo in enumerate(conversations, start=1):
    history = []
    conv_results = []
    for step_idx, msg in enumerate(convo, start=1):
        req = ChatAnalysisRequest(
            user_message=msg,
            conversation_history=[{'role': 'user', 'content': m} for m in history]
        )
        try:
            resp = svc.analyze(req)
            from dataclasses import asdict
            out = {
                'conversation': cidx,
                'step': step_idx,
                'message': msg,
                'intent': resp.intent,
                'entities': asdict(resp.slots),
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

with open('scripts/e2e_nlp_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print('Done. Results written to scripts/e2e_nlp_results.json')