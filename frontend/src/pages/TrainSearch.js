import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import api from '../api';
const CLASS_LABELS = {
    SL: 'Sleeper', '3A': '3rd AC', '2A': '2nd AC',
    '1A': '1st AC', CC: 'Chair Car', EC: 'Executive', '2S': '2nd Sitting', GN: 'General',
};
const TrainResultCard = ({ train }) => {
    const [fares, setFares] = useState([]);
    const [faresLoading, setFaresLoading] = useState(false);
    const [faresOpen, setFaresOpen] = useState(false);
    const toggleFares = async () => {
        if (faresOpen) {
            setFaresOpen(false);
            return;
        }
        setFaresLoading(true);
        try {
            const res = await api.get(`/fares/${train.train_number}`);
            setFares(res.data);
        }
        catch {
            setFares([]);
        }
        finally {
            setFaresLoading(false);
            setFaresOpen(true);
        }
    };
    return (_jsxs("div", { className: "glass p-4 animate-slide-up", children: [_jsxs("div", { className: "flex items-start justify-between gap-3", children: [_jsxs("div", { className: "flex-1 min-w-0", children: [_jsxs("div", { className: "flex items-center gap-2 mb-1", children: [_jsx("span", { className: "text-primary-400 font-bold text-sm font-mono", children: train.train_number }), _jsx("span", { className: "text-white font-semibold text-sm truncate", children: train.train_name })] }), _jsxs("div", { className: "flex items-center gap-2 text-sm text-slate-400", children: [_jsx("span", { className: "font-mono bg-white/5 px-2 py-0.5 rounded-md text-xs", children: train.source_station_code }), _jsx("span", { className: "text-primary-400", children: "\u2192" }), _jsx("span", { className: "font-mono bg-white/5 px-2 py-0.5 rounded-md text-xs", children: train.destination_station_code })] })] }), _jsx("button", { onClick: toggleFares, className: "shrink-0 text-xs px-3 py-1.5 rounded-lg border border-accent-500/40\r\n                     text-accent-400 hover:bg-accent-500/10 transition-all duration-150", children: faresLoading ? '…' : faresOpen ? 'Hide Fares' : '💰 Fares' })] }), faresOpen && (_jsx("div", { className: "mt-3 pt-3 border-t border-white/10 grid grid-cols-2 sm:grid-cols-4 gap-2 animate-fade-in", children: fares.length === 0 ? (_jsx("p", { className: "col-span-4 text-xs text-slate-500 text-center py-1", children: "No fare data available" })) : (fares.map(f => (_jsxs("div", { className: "bg-white/5 rounded-lg px-3 py-2 text-center border border-white/5", children: [_jsx("p", { className: "text-xs text-slate-400", children: CLASS_LABELS[f.class_type] ?? f.class_type }), _jsxs("p", { className: "text-primary-400 font-bold text-sm mt-0.5", children: ["\u20B9", Math.round(f.amount)] })] }, f.class_type)))) }))] }));
};
const TrainSearch = () => {
    const [src, setSrc] = useState('');
    const [dst, setDst] = useState('');
    const [date, setDate] = useState('');
    const [results, setResults] = useState([]);
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
            const res = await api.get('/trains/search', {
                params: { from_station: src.trim(), to_station: dst.trim() },
            });
            setResults(res.data);
            setSearched(true);
        }
        catch (e) {
            setError(e?.response?.data?.detail || 'Search failed. Please try again.');
            setResults([]);
        }
        finally {
            setLoading(false);
        }
    };
    const handleKeyDown = (e) => {
        if (e.key === 'Enter')
            handleSearch();
    };
    const POPULAR = [
        ['Bangalore', 'Mumbai'],
        ['Delhi', 'Chennai'],
        ['Kolkata', 'Pune'],
        ['Hyderabad', 'Jaipur'],
    ];
    return (_jsxs("div", { className: "max-w-3xl mx-auto space-y-5 animate-fade-in", children: [_jsxs("div", { className: "glass p-6", children: [_jsxs("div", { className: "flex items-center gap-3 mb-5", children: [_jsx("div", { className: "w-10 h-10 rounded-xl bg-primary-500/20 border border-primary-500/30\r\n                          flex items-center justify-center text-xl", children: "\uD83D\uDD0D" }), _jsxs("div", { children: [_jsx("h2", { className: "text-white font-bold text-lg", children: "Search Trains" }), _jsx("p", { className: "text-slate-400 text-xs", children: "Enter city names or station codes" })] })] }), _jsxs("div", { className: "grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4", children: [_jsxs("div", { className: "relative", children: [_jsx("span", { className: "absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-sm pointer-events-none", children: "\uD83C\uDFD9\uFE0F" }), _jsx("input", { className: "input-field pl-9 text-sm", placeholder: "From (e.g. Bangalore)", value: src, onChange: e => setSrc(e.target.value), onKeyDown: handleKeyDown })] }), _jsxs("div", { className: "relative", children: [_jsx("span", { className: "absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-sm pointer-events-none", children: "\uD83D\uDCCD" }), _jsx("input", { className: "input-field pl-9 text-sm", placeholder: "To (e.g. Mumbai)", value: dst, onChange: e => setDst(e.target.value), onKeyDown: handleKeyDown })] }), _jsxs("div", { className: "relative", children: [_jsx("span", { className: "absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-sm pointer-events-none", children: "\uD83D\uDCC5" }), _jsx("input", { type: "date", className: "input-field pl-9 text-sm", value: date, onChange: e => setDate(e.target.value) })] })] }), error && (_jsxs("p", { className: "text-red-400 text-sm mb-3 flex items-center gap-1.5", children: [_jsx("span", { children: "\u26A0\uFE0F" }), error] })), _jsx("button", { onClick: handleSearch, disabled: loading, className: "btn-primary w-full sm:w-auto", children: loading ? (_jsxs("span", { className: "flex items-center gap-2", children: [_jsxs("svg", { className: "animate-spin w-4 h-4", fill: "none", viewBox: "0 0 24 24", children: [_jsx("circle", { className: "opacity-25", cx: "12", cy: "12", r: "10", stroke: "currentColor", strokeWidth: "4" }), _jsx("path", { className: "opacity-75", fill: "currentColor", d: "M4 12a8 8 0 018-8v8z" })] }), "Searching\u2026"] })) : '🔍 Search Trains' }), _jsxs("div", { className: "mt-4 pt-4 border-t border-white/5", children: [_jsx("p", { className: "text-xs text-slate-500 mb-2", children: "Popular routes:" }), _jsx("div", { className: "flex flex-wrap gap-2", children: POPULAR.map(([s, d]) => (_jsxs("button", { onClick: () => { setSrc(s); setDst(d); }, className: "text-xs px-3 py-1.5 rounded-full border border-white/10 text-slate-400\r\n                           hover:bg-white/5 hover:text-white hover:border-white/20 transition-all duration-150", children: [s, " \u2192 ", d] }, `${s}-${d}`))) })] })] }), searched && (_jsxs("div", { children: [_jsxs("div", { className: "flex items-center justify-between mb-3 px-1", children: [_jsx("h3", { className: "text-white font-semibold", children: results.length > 0
                                    ? `${results.length} Train${results.length !== 1 ? 's' : ''} Found`
                                    : 'No Trains Found' }), results.length > 0 && (_jsxs("span", { className: "text-xs text-slate-500", children: [src, " \u2192 ", dst] }))] }), results.length === 0 ? (_jsxs("div", { className: "glass p-8 text-center", children: [_jsx("p", { className: "text-4xl mb-3", children: "\uD83D\uDEAB" }), _jsx("p", { className: "text-white font-medium", children: "No trains found" }), _jsxs("p", { className: "text-slate-400 text-sm mt-1", children: ["Try different city names or check spelling.", _jsx("br", {}), "Example: ", _jsx("em", { className: "text-slate-300", children: "Bangalore, Mumbai, Delhi, Chennai" })] })] })) : (_jsx("div", { className: "space-y-3", children: results.map(train => (_jsx(TrainResultCard, { train: train }, train.train_number))) }))] }))] }));
};
export default TrainSearch;
