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
  const [fares, setFares] = useState<Fare[]>([]);
  const [faresLoading, setFaresLoading] = useState(false);
  const [faresOpen, setFaresOpen] = useState(false);
  const [faresError, setFaresError] = useState('');

  const toggleFares = async () => {
    if (faresOpen) { setFaresOpen(false); return; }
    setFaresLoading(true);
    setFaresError('');
    try {
      const res = await api.get<Fare[]>(`/fares/${train.train_number}`);
      setFares(Array.isArray(res.data) ? res.data : []);
    } catch {
      setFaresError('Fare data not available.');
      setFares([]);
    } finally {
      setFaresLoading(false);
      setFaresOpen(true);
    }
  };

  return (
    <div className="glass p-4 animate-slide-up hover:border-primary-500/20 transition-all duration-200">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="text-primary-400 font-bold text-sm font-mono">{train.train_number}</span>
            <span className="text-white font-semibold text-sm truncate">{train.train_name}</span>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <span className="font-mono bg-white/5 px-2 py-0.5 rounded-md text-xs">{train.source_station_code}</span>
            <span className="text-primary-400">→</span>
            <span className="font-mono bg-white/5 px-2 py-0.5 rounded-md text-xs">{train.destination_station_code}</span>
          </div>
        </div>
        <button
          onClick={toggleFares}
          disabled={faresLoading}
          className="shrink-0 text-xs px-3 py-1.5 rounded-lg border border-accent-500/40
                     text-accent-400 hover:bg-accent-500/10 transition-all duration-150
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {faresLoading ? (
            <svg className="animate-spin w-3.5 h-3.5" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
            </svg>
          ) : faresOpen ? 'Hide Fares' : '💰 Fares'}
        </button>
      </div>

      {faresOpen && (
        <div className="mt-3 pt-3 border-t border-white/10 animate-fade-in">
          {faresError ? (
            <p className="text-xs text-slate-500 text-center py-1">{faresError}</p>
          ) : fares.length === 0 ? (
            <p className="text-xs text-slate-500 text-center py-1">No fare data available for this train.</p>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {fares.map(f => (
                <div key={f.class_type} className="bg-white/5 rounded-lg px-3 py-2 text-center border border-white/5">
                  <p className="text-xs text-slate-400">{CLASS_LABELS[f.class_type] ?? f.class_type}</p>
                  <p className="text-primary-400 font-bold text-sm mt-0.5">₹{Math.round(f.amount).toLocaleString('en-IN')}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const POPULAR = [
  ['Bangalore', 'Mumbai'],
  ['Delhi', 'Chennai'],
  ['Kolkata', 'Pune'],
  ['Hyderabad', 'Jaipur'],
  ['Ahmedabad', 'Surat'],
  ['Lucknow', 'Varanasi'],
];

const TrainSearch: React.FC = () => {
  const [src, setSrc] = useState('');
  const [dst, setDst] = useState('');
  const [date, setDate] = useState('');
  const [results, setResults] = useState<Train[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async () => {
    if (!src.trim() || !dst.trim()) {
      setError('Please enter both source and destination.');
      return;
    }
    setError('');
    setLoading(true);
    setSearched(false);
    try {
      const params: Record<string, string> = {
        from_station: src.trim(),
        to_station: dst.trim(),
      };
      // Wire date to the API (backend logs it; date filtering needs availability data)
      if (date) params.date = date;

      const res = await api.get<Train[]>('/trains/search', { params });
      setResults(Array.isArray(res.data) ? res.data : []);
      setSearched(true);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setError(err?.response?.data?.detail || 'Search failed. Please try again.');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch();
  };

  const setRoute = (s: string, d: string) => { setSrc(s); setDst(d); };

  return (
    <div className="max-w-3xl mx-auto space-y-5 animate-fade-in">
      {/* Search card */}
      <div className="glass p-6">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-10 h-10 rounded-xl bg-primary-500/20 border border-primary-500/30 flex items-center justify-center text-xl">🔍</div>
          <div>
            <h1 className="text-white font-bold text-lg">Search Trains</h1>
            <p className="text-slate-400 text-xs">Enter city names or station codes (e.g. SBC, NDLS)</p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-sm pointer-events-none">🏙️</span>
            <input
              className="input-field pl-9 text-sm"
              placeholder="From (e.g. Bangalore)"
              value={src}
              onChange={e => setSrc(e.target.value)}
              onKeyDown={handleKeyDown}
              id="train-search-from"
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
              id="train-search-to"
            />
          </div>
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-sm pointer-events-none">📅</span>
            <input
              type="date"
              className="input-field pl-9 text-sm"
              value={date}
              onChange={e => setDate(e.target.value)}
              min={new Date().toISOString().slice(0, 10)}
              id="train-search-date"
            />
          </div>
        </div>

        {error && (
          <p className="text-red-400 text-sm mb-3 flex items-center gap-1.5">
            <span>⚠️</span>{error}
          </p>
        )}

        <button
          onClick={handleSearch}
          disabled={loading}
          className="btn-primary w-full sm:w-auto"
          id="train-search-btn"
        >
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
          <p className="text-xs text-slate-500 mb-2">⚡ Popular routes:</p>
          <div className="flex flex-wrap gap-2">
            {POPULAR.map(([s, d]) => (
              <button
                key={`${s}-${d}`}
                onClick={() => setRoute(s, d)}
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
            <h2 className="text-white font-semibold">
              {results.length > 0
                ? `${results.length} Train${results.length !== 1 ? 's' : ''} Found`
                : 'No Trains Found'}
            </h2>
            {results.length > 0 && <span className="text-xs text-slate-500">{src} → {dst}{date ? ` · ${date}` : ''}</span>}
          </div>

          {results.length === 0 ? (
            <div className="glass p-8 text-center">
              <p className="text-4xl mb-3">🚫</p>
              <p className="text-white font-medium">No trains found</p>
              <p className="text-slate-400 text-sm mt-1">
                Try different city names, station codes, or check spelling.<br />
                <em className="text-slate-300">Examples: Bangalore, Mumbai, SBC, CSMT</em>
              </p>
              <p className="text-slate-500 text-xs mt-3">
                💡 For complex queries, try the <strong className="text-primary-400">Chat Assistant</strong>
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
