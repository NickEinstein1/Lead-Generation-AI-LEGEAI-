"use client";
import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { getSession, logout, Session } from "@/lib/auth";
import Link from "next/link";
import Sidebar from "./Sidebar";
import DigitalClock from "./DigitalClock";
import FuturisticBackground from "./FuturisticBackground";

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const router = useRouter();
  const [session, setSession] = useState<Session | null>(null);
  const [showNotifications, setShowNotifications] = useState(false);

  // Load session on client side only to avoid hydration mismatch
  useEffect(() => {
    setSession(getSession());
  }, []);

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  const handleNotificationClick = () => {
    setShowNotifications(!showNotifications);
  };

  return (
    <div className="flex h-screen relative overflow-hidden">
      {/* Futuristic Background */}
      <FuturisticBackground />

      {/* Sidebar - Now toggleable */}
      <Sidebar />

      {/* Main Content - Full width since sidebar is now overlay */}
      <div className="flex-1 flex flex-col overflow-hidden w-full relative">
        {/* Top Header - Futuristic glassmorphism */}
        <header className="bg-slate-900/40 backdrop-blur-xl border-b border-blue-500/30 shadow-2xl relative">
          {/* Glow effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 via-orange-500/10 to-cyan-500/10"></div>

          <div className="flex items-center justify-between px-3 sm:px-6 py-3 sm:py-4 relative z-10">
            {/* Left: Breadcrumb or Title - with space for hamburger button */}
            <div className="flex items-center gap-2 ml-12 sm:ml-14">
              <span className="text-blue-100 text-xs sm:text-sm font-bold tracking-wider uppercase">Dashboard</span>
            </div>

            {/* Right: User Menu - Responsive */}
            <div className="flex items-center gap-2 sm:gap-4">
              {/* Digital Clock - Hidden on mobile */}
              <div className="hidden md:block">
                <DigitalClock showDate={true} showSeconds={true} use24Hour={false} />
              </div>

              {/* Notifications */}
              <div className="relative">
                <button
                  onClick={handleNotificationClick}
                  className="relative p-1.5 sm:p-2 text-blue-200 hover:text-cyan-300 transition-all duration-300 hover:scale-110 rounded-lg hover:bg-blue-500/20 backdrop-blur-sm border border-blue-500/20 hover:border-cyan-400/50"
                  title="Notifications"
                >
                  <span className="text-lg sm:text-xl">🔔</span>
                  <span className="absolute top-0.5 right-0.5 sm:top-1 sm:right-1 w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-lg shadow-cyan-400/50"></span>
                </button>

                {/* Notifications Dropdown - Futuristic with animation */}
                {showNotifications && (
                  <div className="absolute right-0 mt-2 w-72 sm:w-80 bg-slate-900/95 backdrop-blur-xl rounded-xl shadow-2xl border border-blue-500/50 z-50 max-w-[calc(100vw-2rem)] animate-in fade-in slide-in-from-top-2 duration-300">
                    <div className="p-3 sm:p-4 border-b border-blue-500/30 bg-gradient-to-r from-blue-500/20 to-orange-500/20">
                      <h3 className="font-bold text-blue-100 text-sm sm:text-base flex items-center gap-2">
                        <span>🔔</span>
                        Notifications
                        <span className="ml-auto text-xs bg-cyan-500 text-slate-900 px-2 py-0.5 rounded-full font-bold shadow-lg shadow-cyan-500/50">3</span>
                      </h3>
                    </div>
                    <div className="max-h-72 sm:max-h-96 overflow-y-auto">
                      <div className="p-4 hover:bg-gradient-to-r hover:from-blue-500/20 hover:to-orange-500/20 border-b border-blue-500/20 cursor-pointer transition-all duration-200 group">
                        <p className="text-sm font-semibold text-blue-100 group-hover:text-cyan-300 transition-colors">New lead assigned</p>
                        <p className="text-xs text-slate-400 mt-1">John Doe has been assigned to you</p>
                        <p className="text-xs text-cyan-400 mt-1 font-medium">2 minutes ago</p>
                      </div>
                      <div className="p-4 hover:bg-gradient-to-r hover:from-blue-500/20 hover:to-orange-500/20 border-b border-blue-500/20 cursor-pointer transition-all duration-200 group">
                        <p className="text-sm font-semibold text-blue-100 group-hover:text-cyan-300 transition-colors">Document signed</p>
                        <p className="text-xs text-slate-400 mt-1">Policy #12345 has been signed</p>
                        <p className="text-xs text-cyan-400 mt-1 font-medium">1 hour ago</p>
                      </div>
                      <div className="p-4 hover:bg-gradient-to-r hover:from-blue-500/20 hover:to-orange-500/20 border-b border-blue-500/20 cursor-pointer transition-all duration-200 group">
                        <p className="text-sm font-semibold text-blue-100 group-hover:text-cyan-300 transition-colors">Campaign completed</p>
                        <p className="text-xs text-slate-400 mt-1">Email campaign "Summer Promo" finished</p>
                        <p className="text-xs text-cyan-400 mt-1 font-medium">3 hours ago</p>
                      </div>
                    </div>
                    <div className="p-3 border-t border-blue-500/30 text-center bg-gradient-to-r from-blue-500/20 to-orange-500/20">
                      <button
                        onClick={() => {
                          setShowNotifications(false);
                          router.push("/dashboard/notifications");
                        }}
                        className="text-sm text-cyan-300 hover:text-cyan-200 font-bold hover:underline transition-all"
                      >
                        View all notifications →
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* User Profile - Futuristic */}
              <div className="flex items-center gap-2 sm:gap-3 pl-2 sm:pl-4 border-l border-blue-500/30">
                <div className="text-right hidden sm:block">
                  <p className="text-xs sm:text-sm font-bold text-blue-100">
                    {session?.userId || "User"}
                  </p>
                  <p className="text-xs text-cyan-400 font-medium">Admin</p>
                </div>
                <div className="w-7 h-7 sm:w-9 sm:h-9 bg-gradient-to-br from-cyan-500 via-blue-500 to-orange-600 rounded-full flex items-center justify-center text-white font-bold text-xs sm:text-sm shadow-lg shadow-cyan-500/50 ring-2 ring-cyan-400/50 hover:ring-4 hover:ring-cyan-300/50 transition-all duration-300 hover:scale-110 animate-glow">
                  {(session?.userId || "U").charAt(0).toUpperCase()}
                </div>
              </div>

              {/* Logout Button - Futuristic */}
              <button
                onClick={handleLogout}
                className="ml-1 sm:ml-2 px-3 sm:px-4 py-1.5 sm:py-2 text-xs sm:text-sm text-blue-200 hover:text-white bg-red-500/20 hover:bg-gradient-to-r hover:from-red-500 hover:to-red-600 rounded-lg transition-all duration-300 font-semibold border border-red-500/30 hover:border-red-400 shadow-sm hover:shadow-lg hover:shadow-red-500/50 active:scale-95 backdrop-blur-sm"
              >
                <span className="hidden sm:inline">Logout</span>
                <span className="sm:hidden">Exit</span>
              </button>
            </div>
          </div>
        </header>

        {/* Main Content Area - Responsive padding */}
        <main className="flex-1 overflow-y-auto">
          <div className="p-3 sm:p-4 md:p-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}

