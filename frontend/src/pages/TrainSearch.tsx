import React, { useState } from 'react';
import api from '../api';

interface Train {
  train_number: string;
  train_name: string;
  source_station_code: string;
  destination_station_code: string;
}

interface Fare {
  class_type: string;
  amount: number;
}

const CLASS_LABELS: Record<string, string> = {
  SL: 'Sleeper', '3A': '3rd AC', '2A': '2nd AC',
  '1A': '1st AC', CC: 'Chair Car', EC: 'Executive', '2S': '2nd Sitting', GN: 'General',
};

const TrainResultCard: React.FC<{ train: Train }> = ({ train }) => {
  const [fares,        setFares]        = useState<Fare[]>([]);
  const [faresLoading, setFaresLoading] = useState(false);
  const [faresOpen,    setFaresOpen]    = useState(false);

  const toggleFares = async () => {
    if (faresOpen) { setFaresOpen(false); return; }
    setFaresLoading(true);
    try {
      const res = await api.get<Fare[]>(`/api/v1/fares/${train.train_number}`);
      setFares(res.data);
    } catch { setFares([]); }
    finally { setFaresLoading(false); setFaresOpen(true); }
  };

  return (
    <div className="glass p-4 animate-slide-up">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-primary-400 font-bold text-sm font-mono">
              {train.train_number}
            </span>
            <span className="text-white font-semibold text-sm truncate">{train.train_name}</span>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <span className="font-mono bg-white/5 px-2 py-0.5 rounded-md text-xs">
              {train.source_station_code}
            </span>
            <span className="text-primary-400">→</span>
            <span className="font-mono bg-white/5 px-2 py-0.5 rounded-md text-xs">
              {train.destination_station_code}
            </span>
          </div>
        </div>
        <button
          onClick={toggleFares}
          className="shrink-0 text-xs px-3 py-1.5 rounded-lg border border-accent-500/40
                     text-accent-400 hover:bg-accent-500/10 transition-all duration-150"
        >
          {faresLoading ? '…' : faresOpen ? 'Hide Fares' : '💰 Fares'}
        </button>
      </div>

      {faresOpen && (
        <div className="mt-3 pt-3 border-t border-white/10 grid grid-cols-2 sm:grid-cols-4 gap-2 animate-fade-in">
          {fares.length === 0 ? (
            <p className="col-span-4 text-xs text-slate-500 text-center py-1">No fare data available</p>
          ) : (
            fares.map(f => (
              <div key={f.class_type}
                   className="bg-white/5 rounded-lg px-3 py-2 text-center border border-white/5">
                <p className="text-xs text-slate-400">{CLASS_LABELS[f.class_type] ?? f.class_type}</p>
                <p className="text-primary-400 font-bold text-sm mt-0.5">₹{Math.round(f.amount)}</p>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
};

const TrainSearch: React.FC = () => {
  const [src,     setSrc]     = useState('');
  const [dst,     setDst]     = useState('');
  const [date,    setDate]    = useState('');
  const [results, setResults] = useState<Train[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error,   setError]   = useState('');

  const handleSearch = async () => {
    if (!src.trim() || !dst.trim()) {
      setError('Please enter both source and destination.');
      return;
    }
    setError('');
    setLoading(true);
    setSearched(false);
    try {
      const res = await api.get<Train[]>('/api/v1/trains/search', {
      params: { from_station: src.trim(), to_station: dst.trim() },
    });
      setResults(res.data);
      setSearched(true);
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Search failed. Please try again.');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch();
  };

  const POPULAR = [
    ['Bangalore', 'Mumbai'],
    ['Delhi', 'Chennai'],
    ['Kolkata', 'Pune'],
    ['Hyderabad', 'Jaipur'],
  ];

  return (
    <div className="max-w-3xl mx-auto space-y-5 animate-fade-in">
      {/* Header */}
      <div className="glass p-6">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-10 h-10 rounded-xl bg-primary-500/20 border border-primary-500/30
                          flex items-center justify-center text-xl">
            🔍
          </div>
          <div>
            <h2 className="text-white font-bold text-lg">Search Trains</h2>
            <p className="text-slate-400 text-xs">Enter city names or station codes</p>
          </div>
        </div>

        {/* Inputs */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-sm pointer-events-none">🏙️</span>
            <input
              className="input-field pl-9 text-sm"
              placeholder="From (e.g. Bangalore)"
              value={src}
              onChange={e => setSrc(e.target.value)}
              onKeyDown={handleKeyDown}
            />
          </div>
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-sm pointer-events-none">📍</span>
            <input
              className="input-field pl-9 text-sm"
              placeholder="To (e.g. Mumbai)"
              value={dst}
              onChange={e => setDst(e.target.value)}
              onKeyDown={handleKeyDown}
            />
          </div>
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-sm pointer-events-none">📅</span>
            <input
              type="date"
              className="input-field pl-9 text-sm"
              value={date}
              onChange={e => setDate(e.target.value)}
            />
          </div>
        </div>

        {error && (
          <p className="text-red-400 text-sm mb-3 flex items-center gap-1.5">
            <span>⚠️</span>{error}
          </p>
        )}

        <button onClick={handleSearch} disabled={loading} className="btn-primary w-full sm:w-auto">
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
              </svg>
              Searching…
            </span>
          ) : '🔍 Search Trains'}
        </button>

        {/* Popular routes */}
        <div className="mt-4 pt-4 border-t border-white/5">
          <p className="text-xs text-slate-500 mb-2">Popular routes:</p>
          <div className="flex flex-wrap gap-2">
            {POPULAR.map(([s, d]) => (
              <button
                key={`${s}-${d}`}
                onClick={() => { setSrc(s); setDst(d); }}
                className="text-xs px-3 py-1.5 rounded-full border border-white/10 text-slate-400
                           hover:bg-white/5 hover:text-white hover:border-white/20 transition-all duration-150"
              >
                {s} → {d}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Results */}
      {searched && (
        <div>
          <div className="flex items-center justify-between mb-3 px-1">
            <h3 className="text-white font-semibold">
              {results.length > 0
                ? `${results.length} Train${results.length !== 1 ? 's' : ''} Found`
                : 'No Trains Found'}
            </h3>
            {results.length > 0 && (
              <span className="text-xs text-slate-500">{src} → {dst}</span>
            )}
          </div>

          {results.length === 0 ? (
            <div className="glass p-8 text-center">
              <p className="text-4xl mb-3">🚫</p>
              <p className="text-white font-medium">No trains found</p>
              <p className="text-slate-400 text-sm mt-1">
                Try different city names or check spelling.<br />
                Example: <em className="text-slate-300">Bangalore, Mumbai, Delhi, Chennai</em>
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {results.map(train => (
                <TrainResultCard key={train.train_number} train={train} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TrainSearch;
