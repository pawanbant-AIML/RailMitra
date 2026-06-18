// frontend/src/pages/ChatAssistant.tsx
import React, { useState, useEffect } from 'react';
import api, { postStructuredChat } from '../api';
import ChatWindow from '../components/ChatWindow';
import BookingDrawer from '../components/BookingDrawer';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

type BookingDrawerValues = {
  source: string;
  destination: string;
  travel_date: string;
  travel_class: string;
  passenger_count: number;
  train_number?: string;
  train_selection?: string;
  time_preference?: string;
  budget?: number;
  direct_only?: boolean;
};

type BookingConfirmationRequest = {
  source: string;
  destination: string;
  travel_date: string;
  travel_class: string;
  passenger_count: number;
  train_selection: string;
  user_id: number;
};

type BookingConfirmationResponse = {
  success: boolean;
  status: string;
  message: string;
  booking?: {
    id: number;
    train_number: string;
    passenger_count: number;
    travel_class: string;
    travel_date: string;
    status: string;
    created_at: string;
  } | null;
  selected_train?: {
    train_number: string;
    train_name: string;
    source_station_code: string;
    destination_station_code: string;
  } | null;
  missing_fields?: string[];
  errors?: string[];
};

const WELCOME: ChatMessage = {
  role: 'assistant',
  content:
    '👋 Hello! I\'m your **AI Train Ticket Assistant** for Indian Railways.\n\n' +
    'I can help you:\n' +
    '  🔍 **Find trains** — *"Find trains from Bangalore to Mumbai"*\n' +
    '  🎫 **Book tickets** — *"Book 2 sleeper tickets from Delhi to Chennai tomorrow"*\n' +
    '  📋 **View bookings** — *"Show my bookings"*\n' +
    '  ❌ **Cancel booking** — *"Cancel booking 5"*\n' +
    '  💰 **Check fares** — *"Fare from Pune to Hyderabad"*\n' +
    '  🗺️ **Check route** — *"Route for train 12657"*\n\n' +
    'How can I assist you today?',
};

const ChatAssistant: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [bookingDraft, setBookingDraft] = useState<any>(null);
  const [bookingSubmitting, setBookingSubmitting] = useState(false);

  useEffect(() => {
    setSessionId(crypto.randomUUID());
  }, []);

  const appendAssistantMessage = (content: string) => {
    setMessages((prev) => [...prev, { role: 'assistant', content }]);
  };

  const sendMessage = async (text: string) => {
    const newHistory: ChatMessage[] = [...messages, { role: 'user', content: text }];
    setMessages(newHistory);
    setLoading(true);

    try {
      const resp = await postStructuredChat({
        message: text,
        session_id: sessionId,
        history: messages,
      });

      if (resp.messages?.length) {
        setMessages(resp.messages);
      }

      if (resp.action === 'open_booking_form' && resp.booking_draft) {
        setBookingDraft(resp.booking_draft);
        setDrawerOpen(true);
      }
    } catch (err: any) {
      const detail =
        err?.response?.data?.detail ||
        'Sorry, I couldn\'t reach the server. Please make sure the backend is running.';
      appendAssistantMessage(`❌ **Error:** ${detail}`);
    } finally {
      setLoading(false);
    }
  };

  const handleBookingSubmit = async (values: BookingDrawerValues) => {
    if (bookingSubmitting) return;

    setBookingSubmitting(true);
    try {
      const trainSelection = (values.train_selection ?? values.train_number ?? '').trim();

      const payload: BookingConfirmationRequest = {
        source: values.source.trim(),
        destination: values.destination.trim(),
        travel_date: values.travel_date.trim(),
        travel_class: values.travel_class.trim(),
        passenger_count: values.passenger_count,
        train_selection: trainSelection,
        user_id: 1,
      };

      const response = await api.post<BookingConfirmationResponse>('/bookings/confirm', payload);
      const data = response.data;

      if (data.success) {
        const bookingId = data.booking?.id;
        const trainNumber = data.booking?.train_number || data.selected_train?.train_number || trainSelection;
        const trainName = data.selected_train?.train_name ? ` (${data.selected_train.train_name})` : '';
        const route = data.selected_train
          ? ` from ${data.selected_train.source_station_code} to ${data.selected_train.destination_station_code}`
          : '';

        appendAssistantMessage(
          `✅ ${data.message || 'Booking confirmed successfully.'} Booking #${bookingId ?? '—'} for train ${trainNumber}${trainName}${route}.`
        );
        setBookingDraft(null);
        setDrawerOpen(false);
        return;
      }

      const failureMessage =
        data.message ||
        data.errors?.[0] ||
        'Sorry, the booking could not be confirmed.';
      appendAssistantMessage(`❌ **Booking failed:** ${failureMessage}`);
    } catch (err: any) {
      const detail =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        'Sorry, the booking could not be confirmed right now.';
      appendAssistantMessage(`❌ **Booking failed:** ${detail}`);
    } finally {
      setBookingSubmitting(false);
    }
  };

  return (
    <div
      className="h-full glass flex flex-col overflow-hidden"
      style={{ maxHeight: 'calc(100vh - 120px)' }}
    >
      <div className="px-5 py-4 border-b border-white/10 flex items-center gap-3 shrink-0">
        <div
          className="w-9 h-9 rounded-xl bg-primary-500/20 border border-primary-500/30
                        flex items-center justify-center text-lg"
        >
          💬
        </div>
        <div>
          <h2 className="text-white font-semibold text-sm">Chat Assistant</h2>
          <p className="text-xs text-slate-400">Powered by Indian Railways data</p>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse-slow" />
          <span className="text-xs text-emerald-400">Online</span>
        </div>
      </div>

      <div className="flex-1 overflow-hidden">
        <ChatWindow messages={messages} onSend={sendMessage} loading={loading} />
      </div>

      <BookingDrawer
        open={drawerOpen}
        onOpenChange={setDrawerOpen}
        bookingDraft={bookingDraft}
        onSubmit={handleBookingSubmit}
      />
    </div>
  );
};

export default ChatAssistant;
