import React, { useState, useRef, useEffect } from 'react';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface Props {
  messages: ChatMessage[];
  onSend: (msg: string) => void;
  loading?: boolean;
}

/* Enhanced Markdown Render */
function RenderContent({ text }: { text: string }) {
  const lines = text.split('\n');
  return (
    <div className="space-y-1">
      {lines.map((line, i) => {
        if (line.trim() === '') return <div key={i} className="h-1" />;
        
        let formattedLine = line;

        // Simple bullet lists
        const isBullet = line.trim().startsWith('•') || line.trim().startsWith('-');
        if (isBullet) {
          formattedLine = line.replace(/^[•-]\s*/, '');
        }

        // Simple numbered lists
        const isNumbered = /^\d+\.\s/.test(line.trim());
        if (isNumbered) {
          formattedLine = line.replace(/^\d+\.\s*/, '');
        }

        // Bold (**text**)
        let parts = formattedLine.split(/\*\*(.+?)\*\*/g);
        const elements = parts.map((part, j) => {
          if (j % 2 === 1) return <strong key={j} className="font-semibold text-white">{part}</strong>;
          
          // Code (`text`)
          const codeParts = part.split(/`(.+?)`/g);
          return (
            <span key={j}>
              {codeParts.map((cPart, k) => {
                if (k % 2 === 1) return <code key={k} className="bg-white/10 px-1 py-0.5 rounded text-accent-300 font-mono text-[0.85em]">{cPart}</code>;
                
                // Italic (_text_ or *text*)
                const italicParts = cPart.split(/(?<!\w)(?:_|\*)([^\s_*]+(?: [^\s_*]+)*)(?:_|\*)(?!\w)/g);
                return (
                  <span key={k}>
                    {italicParts.map((iPart, l) => 
                      l % 2 === 1 ? <em key={l} className="text-slate-300 italic">{iPart}</em> : <span key={l}>{iPart}</span>
                    )}
                  </span>
                );
              })}
            </span>
          );
        });

        if (isBullet) {
          return <p key={i} className="flex gap-2"><span className="text-primary-400">•</span><span>{elements}</span></p>;
        }
        if (isNumbered) {
          const num = line.trim().match(/^(\d+)\./)?.[1] || '';
          return <p key={i} className="flex gap-2"><span className="text-primary-400 font-mono">{num}.</span><span>{elements}</span></p>;
        }

        return <p key={i}>{elements}</p>;
      })}
    </div>
  );
}

const SUGGESTIONS = [
  'Find trains from Bangalore to Mumbai',
  'Book 2 sleeper tickets from Delhi to Chennai tomorrow',
  'Show my bookings',
  'Is there a train between Pune and Hyderabad?',
  'Fare from Kolkata to Varanasi',
];

const ChatWindow: React.FC<Props> = ({ messages, onSend, loading }) => {
  const [input, setInput] = useState('');
  const [hasInteracted, setHasInteracted] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    if (messages.filter(m => m.role === 'user').length > 0) {
      setHasInteracted(true);
    }
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;
    onSend(text);
    setInput('');
    setHasInteracted(true);
  };

  const handleSuggestion = (s: string) => {
    onSend(s);
    setHasInteracted(true);
    inputRef.current?.focus();
  };

  const showSuggestions = !hasInteracted && messages.length <= 1;
  const charsLeft = 200 - input.length;

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto custom-scroll px-4 py-4 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex items-end gap-2 animate-fade-in ${
              msg.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            {msg.role === 'assistant' && (
              <div className="w-8 h-8 rounded-full bg-primary-500/20 border border-primary-500/30 flex items-center justify-center text-sm shrink-0 mb-0.5">
                🚆
              </div>
            )}
            <div className={msg.role === 'user' ? 'bubble-user' : 'bubble-bot'}>
              {msg.role === 'assistant' ? <RenderContent text={msg.content} /> : msg.content}
            </div>
            {msg.role === 'user' && (
              <div className="w-8 h-8 rounded-full bg-slate-700 border border-slate-600 flex items-center justify-center text-sm shrink-0 mb-0.5">
                👤
              </div>
            )}
          </div>
        ))}

        {/* Typing indicator */}
        {loading && (
          <div className="flex items-end gap-2 animate-fade-in">
            <div className="w-8 h-8 rounded-full bg-primary-500/20 border border-primary-500/30 flex items-center justify-center text-sm shrink-0">
              🚆
            </div>
            <div className="bubble-bot flex items-center gap-1.5 py-3">
              <span className="typing-dot" style={{ animationDelay: '0ms' }} />
              <span className="typing-dot" style={{ animationDelay: '150ms' }} />
              <span className="typing-dot" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        )}

        {/* Suggestion chips */}
        {showSuggestions && (
          <div className="pt-2 animate-slide-up">
            <p className="text-xs text-slate-500 mb-2 px-1">💡 Try asking:</p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  onClick={() => handleSuggestion(s)}
                  className="text-xs px-3 py-1.5 rounded-full border border-primary-500/30 text-primary-400
                             hover:bg-primary-500/10 hover:border-primary-500/60 transition-all duration-150
                             hover:text-primary-300 cursor-pointer text-left"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="px-4 pb-4 pt-2 border-t border-white/5 bg-slate-900/50">
        <form onSubmit={handleSubmit} className="flex flex-col gap-2">
          <div className="flex gap-3 items-center">
            <input
              ref={inputRef}
              className="input-field flex-1 text-sm"
              placeholder="Ask about trains, bookings, fares…"
              value={input}
              maxLength={200}
              onChange={e => setInput(e.target.value)}
              disabled={loading}
              autoFocus
            />
            <button
              type="submit"
              disabled={!input.trim() || loading}
              className="btn-primary shrink-0 flex items-center gap-2 text-sm"
            >
              {loading ? (
                <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
                </svg>
              )}
              Send
            </button>
          </div>
          <div className="flex justify-end px-1">
            <span className={`text-[10px] ${charsLeft <= 20 ? 'text-red-400' : 'text-slate-500'}`}>
              {charsLeft} characters remaining
            </span>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatWindow;