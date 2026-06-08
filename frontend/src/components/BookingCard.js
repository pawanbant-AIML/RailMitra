import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
const CLASS_LABELS = {
    SL: 'Sleeper', '3A': '3rd AC', '2A': '2nd AC',
    '1A': '1st AC', CC: 'Chair Car', EC: 'Executive', '2S': '2nd Sitting', GN: 'General',
};
const BookingCard = ({ booking, onCancel }) => {
    const isCancelled = booking.status === 'CANCELLED';
    const dateStr = new Date(booking.travel_date).toLocaleDateString('en-IN', {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
    });
    return (_jsx("div", { className: `glass p-5 animate-slide-up transition-all duration-200 ${isCancelled ? 'opacity-60' : 'hover:border-primary-500/30'}`, children: _jsxs("div", { className: "flex items-start justify-between gap-4", children: [_jsxs("div", { className: "flex-1 min-w-0", children: [_jsxs("div", { className: "flex items-center gap-2 mb-2 flex-wrap", children: [_jsxs("span", { className: "text-primary-400 font-bold font-mono text-sm", children: ["#", booking.id] }), _jsxs("span", { className: "text-white font-semibold text-sm", children: ["Train ", booking.train_number] }), _jsx("span", { className: isCancelled ? 'badge-cancelled' : 'badge-confirmed', children: booking.status })] }), _jsxs("div", { className: "flex flex-wrap items-center gap-3 text-sm text-slate-400", children: [_jsxs("span", { className: "flex items-center gap-1.5", children: [_jsx("span", { className: "text-base", children: "\uD83D\uDC65" }), booking.passenger_count, " passenger", booking.passenger_count !== 1 ? 's' : ''] }), _jsx("span", { className: "badge-class", children: CLASS_LABELS[booking.travel_class] ?? booking.travel_class }), _jsxs("span", { className: "flex items-center gap-1.5", children: [_jsx("span", { className: "text-base", children: "\uD83D\uDCC5" }), dateStr] })] })] }), !isCancelled && (_jsx("button", { onClick: onCancel, className: "btn-danger shrink-0", children: "\u2715 Cancel" }))] }) }));
};
export default BookingCard;
