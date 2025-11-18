"use client";
import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { getSession, logout } from "@/lib/auth";
import Link from "next/link";
import Sidebar from "./Sidebar";
import DigitalClock from "./DigitalClock";

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const router = useRouter();
  const session = getSession();
  const [showNotifications, setShowNotifications] = useState(false);

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  const handleNotificationClick = () => {
    setShowNotifications(!showNotifications);
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Header */}
        <header className="bg-white border-b-2 border-blue-200 shadow-sm">
          <div className="flex items-center justify-between px-6 py-4">
            {/* Left: Breadcrumb or Title */}
            <div className="flex items-center gap-2">
              <span className="text-slate-600 text-sm">Dashboard</span>
            </div>

            {/* Right: User Menu */}
            <div className="flex items-center gap-4">
              {/* Digital Clock */}
              <DigitalClock showDate={true} showSeconds={true} use24Hour={false} />

              {/* Notifications */}
              <div className="relative">
                <button
                  onClick={handleNotificationClick}
                  className="relative p-2 text-slate-600 hover:text-blue-700 transition-colors"
                  title="Notifications"
                >
                  <span className="text-xl">🔔</span>
                  <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
                </button>

                {/* Notifications Dropdown */}
                {showNotifications && (
                  <div className="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-xl border-2 border-blue-200 z-50">
                    <div className="p-4 border-b border-gray-200">
                      <h3 className="font-bold text-slate-900">Notifications</h3>
                    </div>
                    <div className="max-h-96 overflow-y-auto">
                      <div className="p-4 hover:bg-blue-50 border-b border-gray-100 cursor-pointer">
                        <p className="text-sm font-semibold text-slate-900">New lead assigned</p>
                        <p className="text-xs text-slate-600 mt-1">John Doe has been assigned to you</p>
                        <p className="text-xs text-blue-600 mt-1">2 minutes ago</p>
                      </div>
                      <div className="p-4 hover:bg-blue-50 border-b border-gray-100 cursor-pointer">
                        <p className="text-sm font-semibold text-slate-900">Document signed</p>
                        <p className="text-xs text-slate-600 mt-1">Policy #12345 has been signed</p>
                        <p className="text-xs text-blue-600 mt-1">1 hour ago</p>
                      </div>
                      <div className="p-4 hover:bg-blue-50 border-b border-gray-100 cursor-pointer">
                        <p className="text-sm font-semibold text-slate-900">Campaign completed</p>
                        <p className="text-xs text-slate-600 mt-1">Email campaign "Summer Promo" finished</p>
                        <p className="text-xs text-blue-600 mt-1">3 hours ago</p>
                      </div>
                    </div>
                    <div className="p-3 border-t border-gray-200 text-center">
                      <button
                        onClick={() => {
                          setShowNotifications(false);
                          router.push("/dashboard/notifications");
                        }}
                        className="text-sm text-blue-700 hover:text-blue-800 font-semibold"
                      >
                        View all notifications
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* User Profile */}
              <div className="flex items-center gap-3 pl-4 border-l border-gray-200">
                <div className="text-right">
                  <p className="text-sm font-medium text-slate-900">
                    {session?.userId || "User"}
                  </p>
                  <p className="text-xs text-slate-500">Admin</p>
                </div>
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-700 rounded-full flex items-center justify-center text-white font-bold text-sm">
                  {(session?.userId || "U").charAt(0).toUpperCase()}
                </div>
              </div>

              {/* Logout Button */}
              <button
                onClick={handleLogout}
                className="ml-2 px-3 py-2 text-sm text-slate-700 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all font-medium"
              >
                Logout
              </button>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 overflow-y-auto">
          <div className="p-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}

