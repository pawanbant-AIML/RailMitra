"""End-to-end API tests for the train ticket assistant."""
import urllib.request
import json
import sys

BASE = "http://localhost:8000"

def api(method, path, body=None):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def test(name, passed, details=""):
    icon = "✅" if passed else "❌"
    print(f"  {icon} {name}", f"— {details}" if details else "")
    return passed

print("=" * 60)
print("  AI Train Ticket Assistant - API Test Suite")
print("=" * 60)
results = []

# 1. Health
print("\n1️⃣  Health Check")
code, data = api("GET", "/health")
results.append(test("GET /health", code == 200 and data.get("status") == "ok", f"status={code}"))

# 2. List trains
print("\n2️⃣  Train Listing")
code, data = api("GET", "/api/v1/trains?limit=5")
results.append(test("GET /trains?limit=5", code == 200 and len(data) == 5, f"got {len(data)} trains"))

# 3. Search trains Bangalore → Mumbai
print("\n3️⃣  Search Trains: Bangalore → Mumbai")
code, data = api("GET", "/api/v1/trains/search?from_station=Bangalore&to_station=Mumbai")
results.append(test("GET /trains/search (Bangalore→Mumbai)", code == 200 and len(data) > 0,
                     f"found {len(data)} trains"))
if data:
    print(f"     First train: {data[0]['train_number']} — {data[0]['train_name']}")

# 4. Search trains Delhi → Chennai
print("\n4️⃣  Search Trains: Delhi → Chennai")
code, data = api("GET", "/api/v1/trains/search?from_station=Delhi&to_station=Chennai")
results.append(test("GET /trains/search (Delhi→Chennai)", code == 200,
                     f"found {len(data)} trains"))

# 5. Station search
print("\n5️⃣  Station Search")
code, data = api("GET", "/api/v1/stations?q=bangalore&limit=5")
results.append(test("GET /stations?q=bangalore", code == 200 and len(data) > 0,
                     f"found {len(data)} stations"))

# 6. Chat: Find trains from Bangalore to Mumbai
print("\n6️⃣  Chat: Search trains")
msgs = [{"role": "user", "content": "Find trains from Bangalore to Mumbai"}]
code, data = api("POST", "/api/v1/chat", msgs)
results.append(test("POST /chat (search trains)", code == 200 and len(data) >= 2,
                     f"got {len(data)} messages"))
if len(data) >= 2:
    last = data[-1]["content"]
    results.append(test("  Response mentions trains", "train" in last.lower() or "Found" in last,
                         f"reply length: {len(last)}"))
    print(f"     Reply: {last[:200]}...")

# 7. Chat: Book 2 sleeper tickets from Bangalore to Mumbai tomorrow
print("\n7️⃣  Chat: Book ticket")
msgs = [{"role": "user", "content": "Book 2 sleeper tickets from Bangalore to Mumbai tomorrow"}]
code, data = api("POST", "/api/v1/chat", msgs)
results.append(test("POST /chat (book ticket)", code == 200 and len(data) >= 2,
                     f"got {len(data)} messages"))
if len(data) >= 2:
    last = data[-1]["content"]
    results.append(test("  Booking confirmed", "Booking" in last or "booking" in last or "✅" in last,
                         f"reply: {last[:150]}"))

# 8. Chat: Show my bookings
print("\n8️⃣  Chat: Booking history")
msgs = [{"role": "user", "content": "Show my bookings"}]
code, data = api("POST", "/api/v1/chat", msgs)
results.append(test("POST /chat (booking history)", code == 200 and len(data) >= 2,
                     f"got {len(data)} messages"))
if len(data) >= 2:
    last = data[-1]["content"]
    results.append(test("  Has booking entries", "#" in last or "Booking" in last or "booking" in last,
                         f"reply: {last[:150]}"))

# 9. REST API: List bookings
print("\n9️⃣  REST: List user bookings")
code, data = api("GET", "/api/v1/bookings?user_id=1")
results.append(test("GET /bookings?user_id=1", code == 200 and len(data) > 0,
                     f"got {len(data)} bookings"))
if data:
    booking_id = data[0]["id"]
    print(f"     First booking: #{booking_id} train={data[0]['train_number']} class={data[0]['travel_class']} pax={data[0]['passenger_count']}")

# 10. REST API: Cancel booking
print("\n🔟  REST: Cancel booking")
if data:
    code2, data2 = api("DELETE", f"/api/v1/bookings/{booking_id}")
    results.append(test(f"DELETE /bookings/{booking_id}", code2 == 200,
                         f"status={code2}, data={data2}"))
else:
    print("  ⏭  Skipped (no booking to cancel)")

# 11. Chat: Cancel booking
print("\n1️⃣1️⃣  Chat: Cancel booking")
msgs = [{"role": "user", "content": f"Cancel booking {booking_id}"}]
code, data = api("POST", "/api/v1/chat", msgs)
results.append(test("POST /chat (cancel booking)", code == 200,
                     f"reply: {data[-1]['content'][:100]}" if data else ""))

# 12. Chat: "Is there a train between Pune and Hyderabad?"
print("\n1️⃣2️⃣  Chat: Train between Pune and Hyderabad")
msgs = [{"role": "user", "content": "Is there a train between Pune and Hyderabad?"}]
code, data = api("POST", "/api/v1/chat", msgs)
results.append(test("POST /chat (Pune→Hyderabad)", code == 200 and len(data) >= 2,
                     f"reply: {data[-1]['content'][:150]}" if data else ""))

# 13. Chat: Book 3AC class tickets
print("\n1️⃣3️⃣  Chat: Book 3AC class")
msgs = [{"role": "user", "content": "Book 3 third ac tickets from Mumbai to Delhi tomorrow"}]
code, data = api("POST", "/api/v1/chat", msgs)
results.append(test("POST /chat (book 3AC)", code == 200,
                     f"reply: {data[-1]['content'][:150]}" if data else ""))

# 14. Fares endpoint
print("\n1️⃣4️⃣  Fares for a known train")
code, data = api("GET", "/api/v1/fares/16530")
results.append(test("GET /fares/16530 (Udyan Express)", code == 200 and len(data) > 0,
                     f"got {len(data)} fare entries"))
if data:
    for f in data:
        print(f"     {f['class_type']}: ₹{f['amount']}")

# 15. Route endpoint
print("\n1️⃣5️⃣  Route for a known train")
code, data = api("GET", "/api/v1/routes/16530")
results.append(test("GET /routes/16530 (Udyan Express)", code == 200 and len(data) > 0,
                     f"got {len(data)} stops"))
if data:
    for s in data[:5]:
        print(f"     {s['sequence']:>2}. {s['station_code']:<8} arr:{s.get('arrival_time','--')}  dep:{s.get('departure_time','--')}")

# Summary
print("\n" + "=" * 60)
passed = sum(results)
total  = len(results)
print(f"  RESULTS: {passed}/{total} passed")
if passed == total:
    print("  🎉 ALL TESTS PASSED!")
else:
    print(f"  ⚠️  {total - passed} test(s) failed")
print("=" * 60)

sys.exit(0 if passed == total else 1)
