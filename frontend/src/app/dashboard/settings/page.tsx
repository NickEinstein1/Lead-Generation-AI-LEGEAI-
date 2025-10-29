"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState("profile");

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Settings</h1>
          <p className="text-slate-600 font-medium mt-1">Manage your account and preferences</p>
        </div>

        {/* Settings Tabs */}
        <div className="flex gap-2 border-b-2 border-blue-200">
          {[
            { id: "profile", label: "👤 Profile", icon: "👤" },
            { id: "team", label: "👥 Team", icon: "👥" },
            { id: "integrations", label: "🔗 Integrations", icon: "🔗" },
            { id: "notifications", label: "🔔 Notifications", icon: "🔔" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-3 font-medium transition-all border-b-2 ${
                activeTab === tab.id
                  ? "border-blue-600 text-blue-700"
                  : "border-transparent text-slate-600 hover:text-slate-900"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Profile Tab */}
        {activeTab === "profile" && (
          <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md space-y-6">
            <h2 className="text-xl font-bold text-slate-900">Profile Settings</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">First Name</label>
                <input type="text" defaultValue="Admin" className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Last Name</label>
                <input type="text" defaultValue="User" className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Email</label>
                <input type="email" defaultValue="admin@leagai.dev" className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Phone</label>
                <input type="tel" defaultValue="+1 (555) 123-4567" className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600" />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-900 mb-2">Company</label>
              <input type="text" defaultValue="LEAGAI Insurance" className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600" />
            </div>

            <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all">
              Save Changes
            </button>
          </div>
        )}

        {/* Team Tab */}
        {activeTab === "team" && (
          <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-slate-900">Team Members</h2>
              <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all text-sm">
                + Add Member
              </button>
            </div>

            <div className="space-y-3">
              {[
                { name: "John Doe", email: "john@leagai.dev", role: "Admin", status: "active" },
                { name: "Sarah Smith", email: "sarah@leagai.dev", role: "Manager", status: "active" },
                { name: "Mike Johnson", email: "mike@leagai.dev", role: "Agent", status: "active" },
              ].map((member, idx) => (
                <div key={idx} className="flex items-center justify-between p-4 border border-blue-200 rounded-lg">
                  <div>
                    <p className="font-medium text-slate-900">{member.name}</p>
                    <p className="text-sm text-slate-600">{member.email}</p>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-sm font-medium text-slate-700">{member.role}</span>
                    <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">✓ {member.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Integrations Tab */}
        {activeTab === "integrations" && (
          <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md space-y-6">
            <h2 className="text-xl font-bold text-slate-900">Integrations</h2>

            <div className="space-y-3">
              {[
                { name: "DocuSeal", status: "connected", icon: "📄" },
                { name: "Stripe", status: "connected", icon: "💳" },
                { name: "Slack", status: "not connected", icon: "💬" },
                { name: "Salesforce", status: "connected", icon: "☁️" },
              ].map((integration, idx) => (
                <div key={idx} className="flex items-center justify-between p-4 border border-blue-200 rounded-lg">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{integration.icon}</span>
                    <p className="font-medium text-slate-900">{integration.name}</p>
                  </div>
                  <button className={`px-4 py-2 rounded-lg font-medium text-sm transition-all ${
                    integration.status === "connected"
                      ? "bg-emerald-100 text-emerald-700 hover:bg-emerald-200"
                      : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                  }`}>
                    {integration.status === "connected" ? "✓ Connected" : "Connect"}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Notifications Tab */}
        {activeTab === "notifications" && (
          <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md space-y-6">
            <h2 className="text-xl font-bold text-slate-900">Notification Preferences</h2>

            <div className="space-y-4">
              {[
                { label: "Email Notifications", description: "Receive updates via email" },
                { label: "SMS Alerts", description: "Get important alerts via SMS" },
                { label: "Push Notifications", description: "Browser push notifications" },
                { label: "Daily Digest", description: "Receive daily summary email" },
              ].map((notif, idx) => (
                <div key={idx} className="flex items-center justify-between p-4 border border-blue-200 rounded-lg">
                  <div>
                    <p className="font-medium text-slate-900">{notif.label}</p>
                    <p className="text-sm text-slate-600">{notif.description}</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" defaultChecked className="sr-only peer" />
                    <div className="w-11 h-6 bg-slate-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
              ))}
            </div>

            <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all">
              Save Preferences
            </button>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}

