import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
const Register = () => {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const navigate = useNavigate();
    const handleSubmit = (e) => {
        e.preventDefault();
        // Demo – just store a token and redirect
        localStorage.setItem('access_token', btoa(`${email}:${password}`));
        navigate('/chat');
    };
    return (_jsx("div", { className: "flex items-center justify-center min-h-[80vh] px-4", children: _jsxs("div", { className: "glass p-8 w-full max-w-md animate-slide-up", children: [_jsxs("div", { className: "text-center mb-8", children: [_jsx("div", { className: "w-14 h-14 rounded-2xl bg-primary-500/20 border border-primary-500/30\r\n                          flex items-center justify-center text-3xl mx-auto mb-4", children: "\uD83D\uDE86" }), _jsx("h1", { className: "text-white text-xl font-bold", children: "Create Account" }), _jsx("p", { className: "text-slate-400 text-sm mt-1", children: "Join AI Train Assistant" })] }), _jsxs("form", { onSubmit: handleSubmit, className: "space-y-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-xs text-slate-400 mb-1.5 font-medium", children: "Full Name" }), _jsx("input", { type: "text", className: "input-field text-sm", placeholder: "Your full name", value: name, onChange: e => setName(e.target.value), required: true })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-xs text-slate-400 mb-1.5 font-medium", children: "Email" }), _jsx("input", { type: "email", className: "input-field text-sm", placeholder: "you@example.com", value: email, onChange: e => setEmail(e.target.value), required: true })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-xs text-slate-400 mb-1.5 font-medium", children: "Password" }), _jsx("input", { type: "password", className: "input-field text-sm", placeholder: "\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022", value: password, onChange: e => setPassword(e.target.value), required: true })] }), _jsx("button", { type: "submit", className: "btn-primary w-full mt-2", children: "Create Account \u2192" })] }), _jsxs("p", { className: "text-center text-slate-500 text-sm mt-6", children: ["Already have an account?", ' ', _jsx(Link, { to: "/login", className: "text-primary-400 hover:text-primary-300 font-medium", children: "Sign In" })] })] }) }));
};
export default Register;
