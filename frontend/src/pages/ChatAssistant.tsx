// frontend/src/pages/ChatAssistant.tsx
import React, { useState, useEffect, useRef } from 'react';
import api, { postStructuredChat } from '../api';
import ChatWindow from '../components/ChatWindow';
import BookingDrawer from '../components/BookingDrawer';
import type { schemas } from '../types';

type BookingDrawerValues = {
  source: string;
  destination: string;
  travel_date: string;
  travel_class: string;
  passenger_count: number;
  train_number?: string;
  time_preference?: string;
  budget?: number;
  direct_only?: boolean;
};

const WELCOME: schemas.ChatMessage = {
  role: 'assistant',
  content:
    '👋 Namaste! I\'m **Rail Mitra**, your AI Indian Railways assistant.\n\n' +
    'I can help you:\n' +
    '  🔍 **Search trains** — *"Find trains from Bangalore to Mumbai"*\n' +
    '  🎫 **Book tickets** — *"Book 2 sleeper tickets from Delhi to Chennai tomorrow"*\n' +
    '  📋 **View bookings** — *"Show my bookings"*\n' +
    '  ❌ **Cancel booking** — *"Cancel booking 5"*\n' +
    '  💰 **Check fares** — *"Fare from Pune to Hyderabad"*\n' +
    '  🗺️ **Check route** — *"Route for train 12657"*\n' +
    '  🏠 **Station info** — *"Tell me about Bangalore station"*\n\n' +
    'How can I assist you today?',
};

const ChatAssistant: React.FC = () => {
  const [messages, setMessages] = useState<schemas.ChatMessage[]>([WELCOME]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [bookingDraft, setBookingDraft] = useState<schemas.BookingDraft | null>(null);
  const [bookingSubmitting, setBookingSubmitting] = useState(false);

  // Keep a stable ref to latest messages for the sendMessage closure
  const messagesRef = useRef(messages);
  useEffect(() => { messagesRef.current = messages; }, [messages]);

  useEffect(() => {
    setSessionId(crypto.randomUUID());
  }, []);

  const appendMessage = (msg: schemas.ChatMessage) => {
    setMessages(prev => [...prev, msg]);
  };

  const sendMessage = async (text: string) => {
    const userMsg: schemas.ChatMessage = { role: 'user', content: text };
    const historySnapshot = [...messagesRef.current];
    setMessages([...historySnapshot, userMsg]);
    setLoading(true);

    try {
      const resp = await postStructuredChat({
        message: text,
        session_id: sessionId,
        history: historySnapshot,
      });

      if (resp.messages?.length) {
        setMessages(resp.messages);
      } else {
        // Backend returned no messages — shouldn't happen, but handle gracefully
        setMessages(prev => [
          ...prev,
          { role: 'assistant', content: '⚠️ Received an empty response. Please try again.' },
        ]);
      }

      if (resp.action === 'open_booking_form' && resp.booking_draft) {
        setBookingDraft(resp.booking_draft);
        setDrawerOpen(true);
      }
    } catch (err: unknown) {
      // Error thrown by api.ts — backend unreachable or 5xx
      const axiosErr = err as { response?: { data?: { detail?: string }; status?: number }; message?: string };
      const detail =
        axiosErr?.response?.data?.detail ||
        (axiosErr?.response?.status === 422
          ? 'Your message could not be processed. Please rephrase.'
          : 'Sorry, I couldn\'t reach the server. Please make sure the backend is running.');
      // Append error as assistant bubble — DO NOT wipe existing history
      setMessages(prev => [
        ...prev,
        { role: 'user', content: text },
        { role: 'assistant', content: `❌ **Error:** ${detail}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleBookingSubmit = async (values: BookingDrawerValues) => {
    if (bookingSubmitting) return;
    setBookingSubmitting(true);

    try {
      const trainSelection = (values.train_number ?? '').trim();

      const payload = {
        source: values.source.trim(),
        destination: values.destination.trim(),
        travel_date: values.travel_date.trim(),
        travel_class: values.travel_class.trim(),
        passenger_count: values.passenger_count,
        // Send BOTH field names so either version of the backend accepts it
        train_number: trainSelection,
        train_selection: trainSelection,
        user_id: 1,
      };

      const response = await api.post<{
        success: boolean;
        status: string;
        message: string;
        booking?: { id: number; train_number: string; passenger_count: number; travel_class: string; travel_date: string; status: string } | null;
        selected_train?: { train_number: string; train_name: string; source_station_code: string; destination_station_code: string } | null;
        missing_fields?: string[];
        errors?: string[];
      }>('/bookings/confirm', payload);

      const data = response.data;

      if (data.success) {
        const bookingId = data.booking?.id;
        const trainNumber = data.booking?.train_number || data.selected_train?.train_number || trainSelection;
        const trainName = data.selected_train?.train_name ? ` (${data.selected_train.train_name})` : '';
        const route = data.selected_train
          ? ` · ${data.selected_train.source_station_code} → ${data.selected_train.destination_station_code}`
          : '';

        appendMessage({
          role: 'assistant',
          content: `✅ **Booking Confirmed!** Booking **#${bookingId ?? '—'}** for train **${trainNumber}**${trainName}${route}.\n\n${data.message || ''}`,
        });
        setBookingDraft(null);
        setDrawerOpen(false);
        return;
      }

      const failureMessage = data.message || data.errors?.[0] || 'Booking could not be confirmed.';
      appendMessage({ role: 'assistant', content: `❌ **Booking failed:** ${failureMessage}` });
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string; message?: string } }; message?: string };
      const detail =
        axiosErr?.response?.data?.detail ||
        axiosErr?.response?.data?.message ||
        'Sorry, the booking could not be confirmed right now.';
      appendMessage({ role: 'assistant', content: `❌ **Booking failed:** ${detail}` });
    } finally {
      setBookingSubmitting(false);
    }
  };

  return (
    <div className="h-full glass flex flex-col overflow-hidden" style={{ maxHeight: 'calc(100vh - 120px)' }}>
      {/* Header */}
      <div className="px-5 py-4 border-b border-white/10 flex items-center gap-3 shrink-0">
        <div className="w-9 h-9 rounded-xl bg-primary-500/20 border border-primary-500/30 flex items-center justify-center text-lg">
          🚆
        </div>
        <div>
          <h2 className="text-white font-semibold text-sm">Rail Mitra Assistant</h2>
          <p className="text-xs text-slate-400">Powered by Indian Railways data</p>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs text-emerald-400">Online</span>
        </div>
      </div>

      {/* Chat area */}
      <div className="flex-1 overflow-hidden">
        <ChatWindow messages={messages} onSend={sendMessage} loading={loading} />
      </div>

      {/* Booking drawer */}
      <BookingDrawer
        open={drawerOpen}
        onOpenChange={setDrawerOpen}
        bookingDraft={bookingDraft}
        onSubmit={handleBookingSubmit}
        submitting={bookingSubmitting}
      />
    </div>
  );
};

export default ChatAssistant;
