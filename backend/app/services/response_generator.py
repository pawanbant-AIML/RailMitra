import os
import random
import requests
from app.models import schemas
from typing import List

class ResponseGenerator:
    """
    Generates human-like, friendly responses.
    Uses Hugging Face Inference API if an API key is provided in the environment.
    Falls back to dynamic templating if the API fails or is not configured.
    """

    def __init__(self):
        self.hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")
        self.hf_api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        
    def _call_llm(self, prompt: str, fallback_text: str) -> str:
        if not self.hf_api_key:
            return fallback_text
            
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        payload = {
            "inputs": f"<|system|>\nYou are RailMitra, a friendly and extremely helpful Indian Railways AI assistant. Keep responses very short, cheerful, and use emojis. Do not output markdown code blocks. Formatting is allowed.<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
            "parameters": {"max_new_tokens": 150, "temperature": 0.7, "return_full_text": False}
        }
        try:
            response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=4)
            if response.status_code == 200:
                text = response.json()[0]['generated_text'].strip()
                if text:
                    return text
        except Exception:
            pass
        return fallback_text

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
        prompt = "Greet the user warmly as a train assistant."
        return self._call_llm(prompt, random.choice(self.GREETINGS))

    def get_unknown(self) -> str:
        prompt = "Politely tell the user you didn't understand and remind them you can find trains, book tickets, and check fares."
        return self._call_llm(prompt, random.choice(self.UNKNOWN))

    def format_search_results(self, src: str, dst: str, results: list, preference: str = None) -> str:
        if not results:
            prompt = f"Tell the user nicely that no trains were found from {src} to {dst}. Suggest they try different dates or stations."
            fallback = random.choice([
                f"Oh no! 😔 I couldn't find any direct trains from **{src}** to **{dst}**. Try a different date or nearby stations.",
                f"I searched everywhere, but there are no trains running between **{src}** and **{dst}**. 🚉",
            ])
            return self._call_llm(prompt, fallback)
            
        lines = [random.choice([
            f"Great news! 🎉 I found **{len(results)}** train(s) heading from **{src}** to **{dst}**:\n",
            f"Here are **{len(results)}** options for your journey from **{src}** to **{dst}**:\n"
        ])]
        
        for r in results[:10]:
            lines.append(f"  • **{r.train_number}** — {r.train_name} ({r.source_station_code} → {r.destination_station_code})")
            
        if len(results) > 10:
            lines.append(f"\n  _...and {len(results) - 10} more!_")

        if preference:
            lines.append(f"\n💡 Sorted by your preference: **{preference}**")

        lines.append('\n💡 *Want to book? Just say "Book 2 tickets for [Train Number]"*')
        return "\n".join(lines)

    def format_booking_confirmation(self, booking) -> str:
        date_str = booking.travel_date.strftime("%d %b %Y") if hasattr(booking.travel_date, "strftime") else str(booking.travel_date)
        prompt = f"Congratulate the user! Their booking is confirmed. Booking ID: {booking.id}, Train: {booking.train_number}, Class: {booking.travel_class}, Passengers: {booking.passenger_count}, Date: {date_str}. Say 'Have a great trip!'"
        
        fallback = random.choice([
            f"✅ **Woohoo! Your booking is confirmed.**\n\n🆔 Booking ID: **{booking.id}**\n🚆 Train: **{booking.train_number}**\n🎫 Class: **{booking.travel_class}**\n👥 Passengers: **{booking.passenger_count}**\n📅 Date: **{date_str}**\n\nHave a fantastic trip!",
        ])
        
        # Since this is highly structured data, we just return the fallback template directly, 
        # as LLMs might mess up the specific exact formatting required by the user interface.
        # But we can let the LLM generate a congratulatory intro.
        intro = self._call_llm("Say 'Woohoo! Your booking is confirmed!' in a highly enthusiastic and friendly way.", "✅ **Woohoo! Your booking is confirmed.**")
        return f"{intro}\n\n🆔 Booking ID: **{booking.id}**\n🚆 Train: **{booking.train_number}**\n🎫 Class: **{booking.travel_class}**\n👥 Passengers: **{booking.passenger_count}**\n📅 Date: **{date_str}**\n\nHave a fantastic trip!"

    def format_booking_failure(self, error: str) -> str:
        prompt = f"Apologize to the user because their train booking failed due to the error: {error}. Tell them to try again."
        fallback = f"❌ Oops, the booking failed: {error}\nPlease try again!"
        return self._call_llm(prompt, fallback)

    def ask_clarification(self, missing_slot: str, custom_question: str = None) -> str:
        if custom_question:
            return f"🤔 {custom_question}"
            
        prompts = {
            "source": ("Ask the user where they are traveling from.", "Got it. Where are we traveling from? 🚉"),
            "destination": ("Ask the user where they are heading to.", "Where are we heading to? 🌍"),
            "date": ("Ask the user what date they want to travel.", "Got a specific date in mind? 🗓️"),
            "travel_class": ("Ask the user which class they prefer (like Sleeper or 3AC).", "Which class do you prefer? (e.g., Sleeper, 3AC, CC) 🎫"),
            "passengers": ("Ask the user how many people are traveling.", "How many people are traveling? 👥"),
            "booking_id": ("Ask the user for their booking ID.", "Please provide your booking ID. 🎫"),
            "train_number": ("Ask the user for the train number.", "What is the train number? 🚂"),
        }
        
        llm_prompt, fallback = prompts.get(missing_slot, ("Ask the user for more details.", f"Please provide more details about your {missing_slot}."))
        return self._call_llm(llm_prompt, random.choice([fallback, f"Please provide your {missing_slot}."]))

    def format_fares(self, src: str, dst: str, train_number: str, fares: list) -> str:
        if not fares:
            return f"💰 I couldn't find any fare data for **{src}** → **{dst}** on train {train_number}. Sorry about that!"
            
        lines = [f"💰 **Here are the estimated fares for {src} → {dst} (Train {train_number}):**\n"]
        for f in fares:
            lines.append(f"  • **{f.class_type}**: ₹{f.amount:.0f}")
        return "\n".join(lines)
        
    def format_cancellation(self, bid: str, success: bool) -> str:
        if success:
            prompt = f"Tell the user their booking #{bid} was successfully cancelled."
            fallback = f"✅ Done. Booking **#{bid}** has been cancelled successfully."
        else:
            prompt = f"Tell the user booking #{bid} was not found, so it could not be cancelled."
            fallback = f"❌ I couldn't find booking **#{bid}**. Are you sure that's the right ID?"
        return self._call_llm(prompt, fallback)

    def format_history(self, history: list) -> str:
        if not history:
            return self._call_llm("Tell the user they have no bookings yet and suggest they book a train.", "📭 It looks like you have no bookings yet!")
            
        lines = [f"📋 **Here are your Bookings** ({len(history)} total):\n"]
        for b in history:
            icon   = "✅" if b.status == "CONFIRMED" else "❌"
            date_s = b.travel_date.strftime("%d %b %Y") if hasattr(b.travel_date, "strftime") else str(b.travel_date)
            lines.append(f"  {icon} **#{b.id}** — Train {b.train_number} | {b.travel_class} × {b.passenger_count} pax | {date_s} | {b.status}")
        return "\n".join(lines)

    def format_route(self, tn: str, stops: list) -> str:
        if not stops:
            return f"🗺️ I couldn't find the route schedule for train **{tn}**. Are you sure the number is correct?"
            
        lines = [f"🗺️ **Schedule for Train {tn}**:\n"]
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
            return self._call_llm("Say 'You are very welcome! Let me know if you need anything else' cheerfully.", "You're very welcome! Have a great day! 😊")
        elif intent == "ABOUT_BOT":
            return self._call_llm("Explain that you are RailMitra, a friendly Indian Railways AI assistant.", "I am RailMitra, your friendly AI assistant. I can help you find trains, check fares, book tickets, and track your history!")
        return self.get_unknown()
