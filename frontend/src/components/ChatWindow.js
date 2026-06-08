import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useRef, useEffect } from 'react';
/* Render **bold** and line-breaks from assistant replies */
function RenderContent({ text }) {
    const lines = text.split('\n');
    return (_jsx("div", { className: "space-y-0.5", children: lines.map((line, i) => {
            const parts = line.split(/\*\*(.+?)\*\*/g);
            return (_jsx("p", { className: line === '' ? 'h-2' : '', children: parts.map((part, j) => j % 2 === 1
                    ? _jsx("strong", { className: "font-semibold text-white", children: part }, j)
                    : _jsx("span", { children: part }, j)) }, i));
        }) }));
}
const SUGGESTIONS = [
    'Find trains from Bangalore to Mumbai',
    'Book 2 sleeper tickets from Delhi to Chennai tomorrow',
    'Show my bookings',
    'Is there a train between Pune and Hyderabad?',
    'Fare from Kolkata to Varanasi',
];
const ChatWindow = ({ messages, onSend, loading }) => {
    const [input, setInput] = useState('');
    const bottomRef = useRef(null);
    const inputRef = useRef(null);
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, loading]);
    const handleSubmit = (e) => {
        e.preventDefault();
        const text = input.trim();
        if (!text || loading)
            return;
        onSend(text);
        setInput('');
    };
    const handleSuggestion = (s) => {
        onSend(s);
        inputRef.current?.focus();
    };
    const showSuggestions = messages.length <= 1;
    return (_jsxs("div", { className: "flex flex-col h-full", children: [_jsxs("div", { className: "flex-1 overflow-y-auto custom-scroll px-4 py-4 space-y-4", children: [messages.map((msg, i) => (_jsxs("div", { className: `flex items-end gap-2 animate-fade-in ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`, children: [msg.role === 'assistant' && (_jsx("div", { className: "w-8 h-8 rounded-full bg-primary-500/20 border border-primary-500/30 flex items-center justify-center text-sm shrink-0 mb-0.5", children: "\uD83D\uDE86" })), _jsx("div", { className: msg.role === 'user' ? 'bubble-user' : 'bubble-bot', children: msg.role === 'assistant'
                                    ? _jsx(RenderContent, { text: msg.content })
                                    : msg.content }), msg.role === 'user' && (_jsx("div", { className: "w-8 h-8 rounded-full bg-slate-700 border border-slate-600 flex items-center justify-center text-sm shrink-0 mb-0.5", children: "\uD83D\uDC64" }))] }, i))), loading && (_jsxs("div", { className: "flex items-end gap-2 animate-fade-in", children: [_jsx("div", { className: "w-8 h-8 rounded-full bg-primary-500/20 border border-primary-500/30 flex items-center justify-center text-sm shrink-0", children: "\uD83D\uDE86" }), _jsxs("div", { className: "bubble-bot flex items-center gap-1.5 py-3", children: [_jsx("span", { className: "typing-dot", style: { animationDelay: '0ms' } }), _jsx("span", { className: "typing-dot", style: { animationDelay: '150ms' } }), _jsx("span", { className: "typing-dot", style: { animationDelay: '300ms' } })] })] })), showSuggestions && (_jsxs("div", { className: "pt-2 animate-slide-up", children: [_jsx("p", { className: "text-xs text-slate-500 mb-2 px-1", children: "\uD83D\uDCA1 Try asking:" }), _jsx("div", { className: "flex flex-wrap gap-2", children: SUGGESTIONS.map((s, i) => (_jsx("button", { onClick: () => handleSuggestion(s), className: "text-xs px-3 py-1.5 rounded-full border border-primary-500/30 text-primary-400\r\n                             hover:bg-primary-500/10 hover:border-primary-500/60 transition-all duration-150\r\n                             hover:text-primary-300 cursor-pointer", children: s }, i))) })] })), _jsx("div", { ref: bottomRef })] }), _jsx("div", { className: "px-4 pb-4 pt-2 border-t border-white/5", children: _jsxs("form", { onSubmit: handleSubmit, className: "flex gap-3 items-center", children: [_jsx("input", { ref: inputRef, className: "input-field flex-1 text-sm", placeholder: "Ask about trains, bookings, fares\u2026", value: input, onChange: e => setInput(e.target.value), disabled: loading, autoFocus: true }), _jsxs("button", { type: "submit", disabled: !input.trim() || loading, className: "btn-primary shrink-0 flex items-center gap-2 text-sm", children: [loading ? (_jsxs("svg", { className: "animate-spin w-4 h-4", fill: "none", viewBox: "0 0 24 24", children: [_jsx("circle", { className: "opacity-25", cx: "12", cy: "12", r: "10", stroke: "currentColor", strokeWidth: "4" }), _jsx("path", { className: "opacity-75", fill: "currentColor", d: "M4 12a8 8 0 018-8v8z" })] })) : (_jsx("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: _jsx("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 19l9 2-9-18-9 18 9-2zm0 0v-8" }) })), "Send"] })] }) })] }));
};
export default ChatWindow;
