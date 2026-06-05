import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';

const Login: React.FC = () => {
  const [email,    setEmail]    = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Demo auth – store a dummy token
    const dummy = btoa(`${email}:${password}`);
    localStorage.setItem('access_token', dummy);
    navigate('/chat');
  };

  return (
    <div className="flex items-center justify-center min-h-[80vh] px-4">
      <div className="glass p-8 w-full max-w-md animate-slide-up">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-primary-500/20 border border-primary-500/30
                          flex items-center justify-center text-3xl mx-auto mb-4">
            🚆
          </div>
          <h1 className="text-white text-xl font-bold">Welcome Back</h1>
          <p className="text-slate-400 text-sm mt-1">Sign in to AI Train Assistant</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1.5 font-medium">Email</label>
            <input
              type="email"
              className="input-field text-sm"
              placeholder="you@example.com"
              value={email}
              onChange={e => setEmail(e.target.value)}
              required
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1.5 font-medium">Password</label>
            <input
              type="password"
              className="input-field text-sm"
              placeholder="••••••••"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
            />
          </div>
          <button type="submit" className="btn-primary w-full mt-2">
            Sign In →
          </button>
        </form>

        <p className="text-center text-slate-500 text-sm mt-6">
          Don't have an account?{' '}
          <Link to="/register" className="text-primary-400 hover:text-primary-300 font-medium">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
};

export default Login;