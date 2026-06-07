import React, { useEffect, useState } from 'react';
import api from '../api';
import BookingCard from '../components/BookingCard';

interface BookingData {
  id: number;
  train_number: string;
  passenger_count: number;
  travel_class: string;
  travel_date: string;
  status: string;
  created_at: string;
}

const BookingDashboard: React.FC = () => {
  const [bookings, setBookings] = useState<BookingData[]>([]);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState('');
  const userId = 1; // demo user

  const fetchBookings = async () => {
    setLoading(true);
    setError('');
    try {
      const resp = await api.get<BookingData[]>(`/bookings?user_id=${userId}`);
      setBookings(resp.data);
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Failed to fetch bookings.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchBookings(); }, []);

  const cancel = async (id: number) => {
    try {
      await api.delete(`/bookings/${id}`);
      setBookings(prev =>
        prev.map(b => b.id === id ? { ...b, status: 'CANCELLED' } : b)
      );
    } catch {
      setError('Failed to cancel booking.');
    }
  };

  const confirmed = bookings.filter(b => b.status === 'CONFIRMED');
  const cancelled = bookings.filter(b => b.status === 'CANCELLED');

  return (
    <div className="max-w-3xl mx-auto space-y-5 animate-fade-in">
      {/* Header */}
      <div className="glass p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary-500/20 border border-primary-500/30
                            flex items-center justify-center text-xl">
              🎫
            </div>
            <div>
              <h2 className="text-white font-bold text-lg">My Bookings</h2>
              <p className="text-slate-400 text-xs">View and manage your train bookings</p>
            </div>
          </div>
          <button onClick={fetchBookings} className="btn-primary text-sm !py-2 !px-4">
            🔄 Refresh
          </button>
        </div>

        {/* Stats row */}
        {!loading && bookings.length > 0 && (
          <div className="grid grid-cols-3 gap-3 mt-5 pt-4 border-t border-white/10">
            <div className="bg-white/5 rounded-xl p-3 text-center border border-white/5">
              <p className="text-2xl font-bold text-white">{bookings.length}</p>
              <p className="text-xs text-slate-400 mt-0.5">Total</p>
            </div>
            <div className="bg-emerald-500/10 rounded-xl p-3 text-center border border-emerald-500/20">
              <p className="text-2xl font-bold text-emerald-400">{confirmed.length}</p>
              <p className="text-xs text-emerald-400/70 mt-0.5">Confirmed</p>
            </div>
            <div className="bg-red-500/10 rounded-xl p-3 text-center border border-red-500/20">
              <p className="text-2xl font-bold text-red-400">{cancelled.length}</p>
              <p className="text-xs text-red-400/70 mt-0.5">Cancelled</p>
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="glass p-4 border-red-500/30 text-red-400 text-sm flex items-center gap-2">
          <span>⚠️</span> {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="glass p-12 flex flex-col items-center gap-3">
          <svg className="animate-spin w-8 h-8 text-primary-400" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
          </svg>
          <p className="text-slate-400 text-sm">Loading bookings…</p>
        </div>
      )}

      {/* Empty state */}
      {!loading && bookings.length === 0 && !error && (
        <div className="glass p-12 text-center">
          <p className="text-5xl mb-4">📭</p>
          <p className="text-white font-semibold text-lg">No bookings yet</p>
          <p className="text-slate-400 text-sm mt-2 max-w-sm mx-auto">
            Use the <strong className="text-primary-400">Chat Assistant</strong> to book tickets
            or try: <em className="text-slate-300">"Book 2 sleeper tickets from Bangalore to Mumbai tomorrow"</em>
          </p>
        </div>
      )}

      {/* Booking cards */}
      {!loading && bookings.length > 0 && (
        <div className="space-y-3">
          {bookings.map(b => (
            <BookingCard key={b.id} booking={b} onCancel={() => cancel(b.id)} />
          ))}
        </div>
      )}
    </div>
  );
};

export default BookingDashboard;
