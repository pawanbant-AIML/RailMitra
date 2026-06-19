import React from 'react';

interface Props {
  booking: {
    id: number;
    train_number: string;
    passenger_count: number;
    travel_class: string;
    travel_date: string;
    status: string;
    created_at?: string;
  };
  onCancel: () => void;
  cancelling?: boolean;
}

const CLASS_LABELS: Record<string, string> = {
  SL: 'Sleeper', '3A': '3rd AC', '2A': '2nd AC',
  '1A': '1st AC', CC: 'Chair Car', EC: 'Executive', '2S': '2nd Sitting', GN: 'General',
};

const BookingCard: React.FC<Props> = ({ booking, onCancel, cancelling = false }) => {
  const isCancelled = booking.status === 'CANCELLED';

  let dateStr = '—';
  try {
    dateStr = new Date(booking.travel_date).toLocaleDateString('en-IN', {
      day: 'numeric', month: 'short', year: 'numeric',
    });
  } catch {
    dateStr = booking.travel_date?.slice(0, 10) ?? '—';
  }

  return (
    <div className={`glass p-5 animate-slide-up transition-all duration-200 ${
      isCancelled ? 'opacity-60' : 'hover:border-primary-500/30'
    }`}>
      <div className="flex items-start justify-between gap-4">
        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <span className="text-primary-400 font-bold font-mono text-sm">#{booking.id}</span>
            <span className="text-white font-semibold text-sm">Train {booking.train_number}</span>
            <span className={isCancelled ? 'badge-cancelled' : 'badge-confirmed'}>
              {booking.status}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-sm text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="text-base">👥</span>
              {booking.passenger_count} passenger{booking.passenger_count !== 1 ? 's' : ''}
            </span>
            <span className="badge-class">
              {CLASS_LABELS[booking.travel_class] ?? booking.travel_class}
            </span>
            <span className="flex items-center gap-1.5">
              <span className="text-base">📅</span>
              {dateStr}
            </span>
          </div>
        </div>

        {/* Cancel button — disabled and shows spinner while cancelling */}
        {!isCancelled && (
          <button
            onClick={onCancel}
            disabled={cancelling}
            className="btn-danger shrink-0 flex items-center gap-1.5 min-w-[90px] justify-center"
          >
            {cancelling ? (
              <>
                <svg className="animate-spin w-3.5 h-3.5" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                </svg>
                Cancelling…
              </>
            ) : '✕ Cancel'}
          </button>
        )}
      </div>
    </div>
  );
};

export default BookingCard;