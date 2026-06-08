import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import api from '../api';
import ChatWindow from '../components/ChatWindow';
const WELCOME = {
    role: 'assistant',
    content: '👋 Hello! I\'m your **AI Train Ticket Assistant** for Indian Railways.\n\n' +
        'I can help you:\n' +
        '  🔍 **Find trains** — *"Find trains from Bangalore to Mumbai"*\n' +
        '  🎫 **Book tickets** — *"Book 2 sleeper tickets from Delhi to Chennai tomorrow"*\n' +
        '  📋 **View bookings** — *"Show my bookings"*\n' +
        '  ❌ **Cancel booking** — *"Cancel booking 5"*\n' +
        '  💰 **Check fares** — *"Fare from Pune to Hyderabad"*\n' +
        '  🗺️ **Check route** — *"Route for train 12657"*\n\n' +
        'How can I assist you today?',
};
const ChatAssistant = () => {
    const [messages, setMessages] = useState([WELCOME]);
    const [loading, setLoading] = useState(false);
    const sendMessage = async (text) => {
        const newHistory = [...messages, { role: 'user', content: text }];
        setMessages(newHistory);
        setLoading(true);
        try {
            const resp = await api.post('/chat', newHistory);
            setMessages(resp.data);
        }
        catch (err) {
            const detail = err?.response?.data?.detail ||
                'Sorry, I couldn\'t reach the server. Please make sure the backend is running.';
            setMessages(prev => [
                ...prev,
                { role: 'assistant', content: `❌ **Error:** ${detail}` },
            ]);
        }
        finally {
            setLoading(false);
        }
    };
    return (_jsxs("div", { className: "h-full glass flex flex-col overflow-hidden", style: { maxHeight: 'calc(100vh - 120px)' }, children: [_jsxs("div", { className: "px-5 py-4 border-b border-white/10 flex items-center gap-3 shrink-0", children: [_jsx("div", { className: "w-9 h-9 rounded-xl bg-primary-500/20 border border-primary-500/30\r\n                        flex items-center justify-center text-lg", children: "\uD83D\uDCAC" }), _jsxs("div", { children: [_jsx("h2", { className: "text-white font-semibold text-sm", children: "Chat Assistant" }), _jsx("p", { className: "text-xs text-slate-400", children: "Powered by Indian Railways data" })] }), _jsxs("div", { className: "ml-auto flex items-center gap-1.5", children: [_jsx("span", { className: "w-2 h-2 rounded-full bg-emerald-400 animate-pulse-slow" }), _jsx("span", { className: "text-xs text-emerald-400", children: "Online" })] })] }), _jsx("div", { className: "flex-1 overflow-hidden", children: _jsx(ChatWindow, { messages: messages, onSend: sendMessage, loading: loading }) })] }));
};
export default ChatAssistant;
