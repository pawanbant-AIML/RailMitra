import React from 'react';
import { NavLink } from 'react-router-dom';

interface NavItem {
  to:    string;
  icon:  string;
  label: string;
  desc:  string;
}

const items: NavItem[] = [
  { to: '/chat',     icon: '💬', label: 'Chat Assistant', desc: 'Ask anything'      },
  { to: '/search',   icon: '🔍', label: 'Search Trains',  desc: 'Find trains'       },
  { to: '/bookings', icon: '🎫', label: 'My Bookings',    desc: 'View & manage'     },
];

const Sidebar: React.FC = () => (
  <aside className="hidden md:flex flex-col w-64 shrink-0 border-r border-white/10 p-4 gap-2"
         style={{ background: 'rgba(15,23,42,0.9)', backdropFilter: 'blur(20px)' }}>
    {/* Branding */}
    <div className="mb-6 px-2">
      <div className="flex items-center gap-3 mb-1">
        <span className="text-2xl">🚆</span>
        <div>
          <p className="font-bold text-white text-sm leading-tight">Rail Mitra</p>
          <p className="text-xs text-primary-400">Indian Railways Assistant</p>
        </div>
      </div>
      <div className="mt-3 h-px bg-gradient-to-r from-primary-500/50 via-purple-500/30 to-transparent" />
    </div>

    {/* Nav links */}
    <nav className="flex flex-col gap-1">
      {items.map(({ to, icon, label, desc }) => (
        <NavLink
          key={to}
          to={to}
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-3 rounded-xl transition-all duration-200 group ${
              isActive
                ? 'bg-primary-500/20 border border-primary-500/30 text-white'
                : 'text-slate-400 hover:bg-white/5 hover:text-white border border-transparent'
            }`
          }
        >
          {({ isActive }) => (
            <>
              <span className={`text-lg transition-transform duration-200 ${isActive ? '' : 'group-hover:scale-110'}`}>
                {icon}
              </span>
              <div className="flex flex-col min-w-0">
                <span className={`text-sm font-medium leading-tight ${isActive ? 'text-primary-300' : ''}`}>
                  {label}
                </span>
                <span className="text-xs text-slate-500 truncate">{desc}</span>
              </div>
              {isActive && (
                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-primary-400 shrink-0" />
              )}
            </>
          )}
        </NavLink>
      ))}
    </nav>

    {/* Footer hint */}
    <div className="mt-auto px-2 pt-4 border-t border-white/5">
      <p className="text-xs text-slate-500 leading-relaxed">
        💡 Try: <em className="text-slate-400">"Find trains from Bangalore to Mumbai"</em>
      </p>
    </div>
  </aside>
);

export default Sidebar;