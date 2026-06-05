import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Login           from './pages/Login';
import Register        from './pages/Register';
import ChatAssistant   from './pages/ChatAssistant';
import TrainSearch     from './pages/TrainSearch';
import BookingDashboard from './pages/BookingDashboard';
import Sidebar         from './components/Sidebar';
import Navbar          from './components/Navbar';

const App: React.FC = () => (
  <div className="flex h-screen overflow-hidden bg-dark">
    {/* Sidebar */}
    <Sidebar />

    {/* Main area */}
    <div className="flex flex-col flex-1 overflow-hidden">
      <Navbar />
      <main className="flex-1 overflow-auto p-4 md:p-6 custom-scroll">
        <Routes>
          <Route path="/login"    element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/chat"     element={<ChatAssistant />} />
          <Route path="/search"   element={<TrainSearch />} />
          <Route path="/bookings" element={<BookingDashboard />} />
          <Route path="/"         element={<Navigate to="/chat" replace />} />
        </Routes>
      </main>
    </div>
  </div>
);

export default App;