import React, { useState } from 'react';
import axios from 'axios';
import ChatWindow from '../components/ChatWindow';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

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
  const [loading,  setLoading]  = useState(false);

  const sendMessage = async (text: string) => {
    const newHistory: ChatMessage[] = [...messages, { role: 'user', content: text }];
    setMessages(newHistory);
    setLoading(true);

    try {
      const resp = await axios.post<ChatMessage[]>('/api/v1/chat', newHistory);
      setMessages(resp.data);
    } catch (err: any) {
      const detail =
        err?.response?.data?.detail ||
        'Sorry, I couldn\'t reach the server. Please make sure the backend is running.';
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `❌ **Error:** ${detail}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full glass flex flex-col overflow-hidden"
         style={{ maxHeight: 'calc(100vh - 120px)' }}>
      {/* Header */}
      <div className="px-5 py-4 border-b border-white/10 flex items-center gap-3 shrink-0">
        <div className="w-9 h-9 rounded-xl bg-primary-500/20 border border-primary-500/30
                        flex items-center justify-center text-lg">
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

      {/* Chat window fills remaining space */}
      <div className="flex-1 overflow-hidden">
        <ChatWindow messages={messages} onSend={sendMessage} loading={loading} />
      </div>
    </div>
  );
};

export default ChatAssistant;