import random
from app.models import schemas
from typing import List

class ResponseGenerator:
    """
    Generates human-like, friendly responses without relying on an external LLM.
    Uses randomization and dynamic templating to feel natural and conversational.
    """

    GREETINGS = [
        "Hello! I'm RailMitra, your friendly AI train assistant. How can I help you travel today? 🚂",
        "Hi there! 👋 I can help you find trains, book tickets, check fares, or look up your history. What do you need?",
        "Greetings! Ready to plan your journey? Just tell me where you want to go. 🛤️"
    ]

    UNKNOWN = [
        "I'm not quite sure I caught that. I'm best at finding trains, booking tickets, and checking fares! Could you rephrase? 🤔",
        "Hmm, I didn't quite understand. Try asking me something like 'Find trains from Delhi to Mumbai tomorrow'. 🚂",
        "I'm still learning! Could you try asking about train routes, bookings, or cancellations? 🎫"
    ]

    def get_greeting(self) -> str:
        return random.choice(self.GREETINGS)

    def get_unknown(self) -> str:
        return random.choice(self.UNKNOWN)

    def format_search_results(self, src: str, dst: str, results: list, preference: str = None) -> str:
        if not results:
            return random.choice([
                f"Oh no! 😔 I couldn't find any direct trains from **{src}** to **{dst}** for those dates. Try a different date or nearby stations.",
                f"I searched everywhere, but there are no trains running between **{src}** and **{dst}** matching your request. 🚉",
            ])
            
        lines = [random.choice([
            f"Great news! 🎉 I found **{len(results)}** train(s) heading from **{src}** to **{dst}**:\n",
            f"Here are **{len(results)}** options for your journey from **{src}** to **{dst}**:\n",
            f"All aboard! 🚂 I've found **{len(results)}** trains from **{src}** to **{dst}**:\n"
        ])]
        
        for r in results[:10]:
            lines.append(f"  • **{r.train_number}** — {r.train_name} ({r.source_station_code} → {r.destination_station_code})")
            
        if len(results) > 10:
            lines.append(f"\n  _...and {len(results) - 10} more!_")

        if preference:
            lines.append(f"\n💡 Sorted by your preference: **{preference}**")

        lines.append('\n💡 *Want to book? Just say "Book 2 tickets for [Train Number]" or "Book sleeper tickets!"*')
        return "\n".join(lines)

    def format_booking_confirmation(self, booking) -> str:
        date_str = booking.travel_date.strftime("%d %b %Y") if hasattr(booking.travel_date, "strftime") else str(booking.travel_date)
        return random.choice([
            f"✅ **Woohoo! Your booking is confirmed.**\n\n"
            f"🆔 Booking ID: **{booking.id}**\n🚆 Train: **{booking.train_number}**\n🎫 Class: **{booking.travel_class}**\n👥 Passengers: **{booking.passenger_count}**\n📅 Date: **{date_str}**\n\n"
            "Have a fantastic trip! Say *'Show my bookings'* anytime to view this.",
            
            f"✅ **All set! Your tickets are booked.**\n\n"
            f"🆔 Booking ID: **{booking.id}**\n🚆 Train: **{booking.train_number}**\n🎫 Class: **{booking.travel_class}**\n👥 Passengers: **{booking.passenger_count}**\n📅 Date: **{date_str}**\n\n"
            "Safe travels! 🎒 Let me know if you need anything else."
        ])

    def format_booking_failure(self, error: str) -> str:
        return random.choice([
            f"❌ Oops, the booking failed: {error}\nPlease try again! Example: *'Book 2 sleeper tickets from Bangalore to Mumbai tomorrow'*",
            f"❌ Something went wrong while booking: {error}\nLet's try that again. You can say *'Book a ticket from Delhi to Pune'*."
        ])

    def ask_clarification(self, missing_slot: str, custom_question: str = None) -> str:
        if custom_question:
            return f"🤔 {custom_question}"
            
        prompts = {
            "source": ["I'd love to help! Where are you starting your journey?", "Got it. Where are we traveling from? 🚉"],
            "destination": ["Where are we heading to? 🌍", "And what's your destination? 📍"],
            "date": ["When would you like to travel? (e.g. tomorrow, 15 June) 📅", "Got a specific date in mind? 🗓️"],
            "travel_class": ["Which class do you prefer? (e.g., Sleeper, 3AC, CC) 🎫", "Do you have a preferred travel class? (1AC, 2AC, Sleeper...) 🛋️"],
            "passengers": ["How many people are traveling? 👥", "Just you, or are there more passengers? 🧑‍🤝‍🧑"]
        }
        
        choices = prompts.get(missing_slot, [f"Please provide more details about your {missing_slot}."])
        return random.choice(choices)

    def format_fares(self, src: str, dst: str, train_number: str, fares: list) -> str:
        if not fares:
            return f"💰 I couldn't find any fare data for **{src}** → **{dst}** on train {train_number}. Sorry about that!"
            
        lines = [f"💰 **Here are the estimated fares for {src} → {dst} (Train {train_number}):**\n"]
        for f in fares:
            lines.append(f"  • **{f.class_type}**: ₹{f.amount:.0f}")
        return "\n".join(lines)
        
    def format_cancellation(self, bid: str, success: bool) -> str:
        if success:
            return random.choice([
                f"✅ Done. Booking **#{bid}** has been cancelled successfully.",
                f"✅ I've cancelled booking **#{bid}** for you. Let me know if you want to book another trip!"
            ])
        return f"❌ I couldn't find booking **#{bid}**. Are you sure that's the right ID? Say *'Show my bookings'* to check."

    def format_history(self, history: list) -> str:
        if not history:
            return random.choice([
                "📭 It looks like you have no bookings yet! Ready to plan a trip? Just say *'Book a train to Mumbai'*.",
                "📭 Your booking history is empty right now. Let's change that! 🚂"
            ])
            
        lines = [f"📋 **Here are your Bookings** ({len(history)} total):\n"]
        for b in history:
            icon   = "✅" if b.status == "CONFIRMED" else "❌"
            date_s = b.travel_date.strftime("%d %b %Y") if hasattr(b.travel_date, "strftime") else str(b.travel_date)
            lines.append(f"  {icon} **#{b.id}** — Train {b.train_number} | {b.travel_class} × {b.passenger_count} pax | {date_s} | {b.status}")
        return "\n".join(lines)

    def format_route(self, tn: str, stops: list) -> str:
        if not stops:
            return f"🗺️ I couldn't find the route schedule for train **{tn}**. Are you sure the number is correct?"
            
        lines = [random.choice([
            f"🗺️ **Here is the route for Train {tn}** ({len(stops)} stops):\n",
            f"🗺️ **Schedule for Train {tn}**:\n"
        ])]
        
        for s in stops[:20]:
            arr = s.arrival_time or "--:--"
            dep = s.departure_time or "--:--"
            lines.append(f"  {s.sequence:>2}. {s.station_code:<8}  arr:{arr}  dep:{dep}")
            
        if len(stops) > 20:
            lines.append(f"  ... and {len(stops) - 20} more stops!")
        return "\n".join(lines)

    def small_talk_response(self, intent: str) -> str:
        if intent == "GREETING":
            return self.get_greeting()
        elif intent == "THANK_YOU":
            return random.choice([
                "You're very welcome! Have a great day! 😊",
                "Anytime! Let me know if you need anything else. 🚂",
                "Happy to help! Safe travels. 🚆"
            ])
        elif intent == "ABOUT_BOT":
            return "I am RailMitra, your friendly AI assistant. I can help you find trains, check fares, book tickets, and track your history all using natural language!"
        return self.get_unknown()
