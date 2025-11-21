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
      {/* Sidebar - Now toggleable */}
      <Sidebar />

      {/* Main Content - Full width since sidebar is now overlay */}
      <div className="flex-1 flex flex-col overflow-hidden w-full">
        {/* Top Header - Responsive */}
        <header className="bg-white border-b-2 border-blue-200 shadow-sm">
          <div className="flex items-center justify-between px-3 sm:px-6 py-3 sm:py-4">
            {/* Left: Breadcrumb or Title - with space for hamburger button */}
            <div className="flex items-center gap-2 ml-12 sm:ml-14">
              <span className="text-slate-600 text-xs sm:text-sm font-medium">Dashboard</span>
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
                  className="relative p-1.5 sm:p-2 text-slate-600 hover:text-blue-700 transition-colors"
                  title="Notifications"
                >
                  <span className="text-lg sm:text-xl">🔔</span>
                  <span className="absolute top-0.5 right-0.5 sm:top-1 sm:right-1 w-2 h-2 bg-red-500 rounded-full"></span>
                </button>

                {/* Notifications Dropdown - Responsive */}
                {showNotifications && (
                  <div className="absolute right-0 mt-2 w-72 sm:w-80 bg-white rounded-lg shadow-xl border-2 border-blue-200 z-50 max-w-[calc(100vw-2rem)]">
                    <div className="p-3 sm:p-4 border-b border-gray-200">
                      <h3 className="font-bold text-slate-900 text-sm sm:text-base">Notifications</h3>
                    </div>
                    <div className="max-h-72 sm:max-h-96 overflow-y-auto">
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

              {/* User Profile - Responsive */}
              <div className="flex items-center gap-2 sm:gap-3 pl-2 sm:pl-4 border-l border-gray-200">
                <div className="text-right hidden sm:block">
                  <p className="text-xs sm:text-sm font-medium text-slate-900">
                    {session?.userId || "User"}
                  </p>
                  <p className="text-xs text-slate-500">Admin</p>
                </div>
                <div className="w-7 h-7 sm:w-8 sm:h-8 bg-gradient-to-br from-blue-500 to-blue-700 rounded-full flex items-center justify-center text-white font-bold text-xs sm:text-sm">
                  {(session?.userId || "U").charAt(0).toUpperCase()}
                </div>
              </div>

              {/* Logout Button - Responsive */}
              <button
                onClick={handleLogout}
                className="ml-1 sm:ml-2 px-2 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm text-slate-700 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all font-medium"
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

