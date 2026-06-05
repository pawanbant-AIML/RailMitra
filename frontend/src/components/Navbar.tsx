import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';

const Navbar: React.FC = () => {
  const navigate = useNavigate();
  return (
    <nav className="flex items-center justify-between px-6 py-3 border-b border-white/10"
         style={{ background: 'rgba(15,23,42,0.85)', backdropFilter: 'blur(20px)' }}>
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-primary-500 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-primary-500/40">
          🚆
        </div>
        <span className="hidden md:block font-bold text-white text-sm tracking-wide">
          AI Train Assistant
        </span>
      </div>

      {/* Quick nav links (desktop) */}
      <div className="hidden md:flex items-center gap-1">
        {[
          { to: '/chat',     label: 'Chat' },
          { to: '/search',   label: 'Search' },
          { to: '/bookings', label: 'Bookings' },
        ].map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `px-4 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                isActive
                  ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`
            }
          >
            {label}
          </NavLink>
        ))}
      </div>

      {/* Right side */}
      <button
        onClick={() => { localStorage.removeItem('access_token'); navigate('/login'); }}
        className="text-slate-400 hover:text-white text-sm px-3 py-1.5 rounded-lg hover:bg-white/5 transition-all duration-150"
      >
        Sign out
      </button>
    </nav>
  );
};

export default Navbar;