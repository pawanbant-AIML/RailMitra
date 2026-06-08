import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from 'react';
import api from '../api';
import BookingCard from '../components/BookingCard';
const BookingDashboard = () => {
    const [bookings, setBookings] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const userId = 1; // demo user
    const fetchBookings = async () => {
        setLoading(true);
        setError('');
        try {
            const resp = await api.get(`/bookings?user_id=${userId}`);
            setBookings(resp.data);
        }
        catch (e) {
            setError(e?.response?.data?.detail || 'Failed to fetch bookings.');
        }
        finally {
            setLoading(false);
        }
    };
    useEffect(() => { fetchBookings(); }, []);
    const cancel = async (id) => {
        try {
            await api.delete(`/bookings/${id}`);
            setBookings(prev => prev.map(b => b.id === id ? { ...b, status: 'CANCELLED' } : b));
        }
        catch {
            setError('Failed to cancel booking.');
        }
    };
    const confirmed = bookings.filter(b => b.status === 'CONFIRMED');
    const cancelled = bookings.filter(b => b.status === 'CANCELLED');
    return (_jsxs("div", { className: "max-w-3xl mx-auto space-y-5 animate-fade-in", children: [_jsxs("div", { className: "glass p-6", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { className: "flex items-center gap-3", children: [_jsx("div", { className: "w-10 h-10 rounded-xl bg-primary-500/20 border border-primary-500/30\r\n                            flex items-center justify-center text-xl", children: "\uD83C\uDFAB" }), _jsxs("div", { children: [_jsx("h2", { className: "text-white font-bold text-lg", children: "My Bookings" }), _jsx("p", { className: "text-slate-400 text-xs", children: "View and manage your train bookings" })] })] }), _jsx("button", { onClick: fetchBookings, className: "btn-primary text-sm !py-2 !px-4", children: "\uD83D\uDD04 Refresh" })] }), !loading && bookings.length > 0 && (_jsxs("div", { className: "grid grid-cols-3 gap-3 mt-5 pt-4 border-t border-white/10", children: [_jsxs("div", { className: "bg-white/5 rounded-xl p-3 text-center border border-white/5", children: [_jsx("p", { className: "text-2xl font-bold text-white", children: bookings.length }), _jsx("p", { className: "text-xs text-slate-400 mt-0.5", children: "Total" })] }), _jsxs("div", { className: "bg-emerald-500/10 rounded-xl p-3 text-center border border-emerald-500/20", children: [_jsx("p", { className: "text-2xl font-bold text-emerald-400", children: confirmed.length }), _jsx("p", { className: "text-xs text-emerald-400/70 mt-0.5", children: "Confirmed" })] }), _jsxs("div", { className: "bg-red-500/10 rounded-xl p-3 text-center border border-red-500/20", children: [_jsx("p", { className: "text-2xl font-bold text-red-400", children: cancelled.length }), _jsx("p", { className: "text-xs text-red-400/70 mt-0.5", children: "Cancelled" })] })] }))] }), error && (_jsxs("div", { className: "glass p-4 border-red-500/30 text-red-400 text-sm flex items-center gap-2", children: [_jsx("span", { children: "\u26A0\uFE0F" }), " ", error] })), loading && (_jsxs("div", { className: "glass p-12 flex flex-col items-center gap-3", children: [_jsxs("svg", { className: "animate-spin w-8 h-8 text-primary-400", fill: "none", viewBox: "0 0 24 24", children: [_jsx("circle", { className: "opacity-25", cx: "12", cy: "12", r: "10", stroke: "currentColor", strokeWidth: "4" }), _jsx("path", { className: "opacity-75", fill: "currentColor", d: "M4 12a8 8 0 018-8v8z" })] }), _jsx("p", { className: "text-slate-400 text-sm", children: "Loading bookings\u2026" })] })), !loading && bookings.length === 0 && !error && (_jsxs("div", { className: "glass p-12 text-center", children: [_jsx("p", { className: "text-5xl mb-4", children: "\uD83D\uDCED" }), _jsx("p", { className: "text-white font-semibold text-lg", children: "No bookings yet" }), _jsxs("p", { className: "text-slate-400 text-sm mt-2 max-w-sm mx-auto", children: ["Use the ", _jsx("strong", { className: "text-primary-400", children: "Chat Assistant" }), " to book tickets or try: ", _jsx("em", { className: "text-slate-300", children: "\"Book 2 sleeper tickets from Bangalore to Mumbai tomorrow\"" })] })] })), !loading && bookings.length > 0 && (_jsx("div", { className: "space-y-3", children: bookings.map(b => (_jsx(BookingCard, { booking: b, onCancel: () => cancel(b.id) }, b.id))) }))] }));
};
export default BookingDashboard;
