"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState({
    emailNotifications: true,
    smsNotifications: true,
    pushNotifications: true,
    leadAlerts: true,
    claimUpdates: true,
    policyReminders: true,
    dailyDigest: false,
    weeklyReport: true,
    monthlyAnalytics: true,
  });

  const toggleNotification = (key: string) => {
    setNotifications(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Notification Settings</h1>
            <p className="text-slate-600 font-medium mt-1">Manage how you receive notifications</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Save Changes
          </button>
        </div>

        {/* Notification Channels */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
          <h2 className="text-xl font-bold text-slate-900 mb-4">Notification Channels</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 border border-blue-200 rounded-lg hover:bg-blue-50 transition">
              <div>
                <p className="font-medium text-slate-900">Email Notifications</p>
                <p className="text-sm text-slate-600">Receive updates via email</p>
              </div>
              <button
                onClick={() => toggleNotification('emailNotifications')}
                className={`w-12 h-6 rounded-full transition-all ${
                  notifications.emailNotifications ? 'bg-emerald-600' : 'bg-slate-300'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-all ${
                  notifications.emailNotifications ? 'ml-6' : 'ml-0.5'
                }`}></div>
              </button>
            </div>
            <div className="flex items-center justify-between p-4 border border-blue-200 rounded-lg hover:bg-blue-50 transition">
              <div>
                <p className="font-medium text-slate-900">SMS Notifications</p>
                <p className="text-sm text-slate-600">Receive updates via SMS</p>
              </div>
              <button
                onClick={() => toggleNotification('smsNotifications')}
                className={`w-12 h-6 rounded-full transition-all ${
                  notifications.smsNotifications ? 'bg-emerald-600' : 'bg-slate-300'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-all ${
                  notifications.smsNotifications ? 'ml-6' : 'ml-0.5'
                }`}></div>
              </button>
            </div>
            <div className="flex items-center justify-between p-4 border border-blue-200 rounded-lg hover:bg-blue-50 transition">
              <div>
                <p className="font-medium text-slate-900">Push Notifications</p>
                <p className="text-sm text-slate-600">Receive in-app notifications</p>
              </div>
              <button
                onClick={() => toggleNotification('pushNotifications')}
                className={`w-12 h-6 rounded-full transition-all ${
                  notifications.pushNotifications ? 'bg-emerald-600' : 'bg-slate-300'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-all ${
                  notifications.pushNotifications ? 'ml-6' : 'ml-0.5'
                }`}></div>
              </button>
            </div>
          </div>
        </div>

        {/* Alert Types */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
          <h2 className="text-xl font-bold text-slate-900 mb-4">Alert Types</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 border border-blue-200 rounded-lg hover:bg-blue-50 transition">
              <div>
                <p className="font-medium text-slate-900">Lead Alerts</p>
                <p className="text-sm text-slate-600">New leads and lead updates</p>
              </div>
              <button
                onClick={() => toggleNotification('leadAlerts')}
                className={`w-12 h-6 rounded-full transition-all ${
                  notifications.leadAlerts ? 'bg-emerald-600' : 'bg-slate-300'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-all ${
                  notifications.leadAlerts ? 'ml-6' : 'ml-0.5'
                }`}></div>
              </button>
            </div>
            <div className="flex items-center justify-between p-4 border border-blue-200 rounded-lg hover:bg-blue-50 transition">
              <div>
                <p className="font-medium text-slate-900">Claim Updates</p>
                <p className="text-sm text-slate-600">Claim status changes</p>
              </div>
              <button
                onClick={() => toggleNotification('claimUpdates')}
                className={`w-12 h-6 rounded-full transition-all ${
                  notifications.claimUpdates ? 'bg-emerald-600' : 'bg-slate-300'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-all ${
                  notifications.claimUpdates ? 'ml-6' : 'ml-0.5'
                }`}></div>
              </button>
            </div>
            <div className="flex items-center justify-between p-4 border border-blue-200 rounded-lg hover:bg-blue-50 transition">
              <div>
                <p className="font-medium text-slate-900">Policy Reminders</p>
                <p className="text-sm text-slate-600">Policy renewals and expirations</p>
              </div>
              <button
                onClick={() => toggleNotification('policyReminders')}
                className={`w-12 h-6 rounded-full transition-all ${
                  notifications.policyReminders ? 'bg-emerald-600' : 'bg-slate-300'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-all ${
                  notifications.policyReminders ? 'ml-6' : 'ml-0.5'
                }`}></div>
              </button>
            </div>
          </div>
        </div>

        {/* Reports & Digests */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
          <h2 className="text-xl font-bold text-slate-900 mb-4">Reports & Digests</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 border border-blue-200 rounded-lg hover:bg-blue-50 transition">
              <div>
                <p className="font-medium text-slate-900">Daily Digest</p>
                <p className="text-sm text-slate-600">Daily summary of activities</p>
              </div>
              <button
                onClick={() => toggleNotification('dailyDigest')}
                className={`w-12 h-6 rounded-full transition-all ${
                  notifications.dailyDigest ? 'bg-emerald-600' : 'bg-slate-300'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-all ${
                  notifications.dailyDigest ? 'ml-6' : 'ml-0.5'
                }`}></div>
              </button>
            </div>
            <div className="flex items-center justify-between p-4 border border-blue-200 rounded-lg hover:bg-blue-50 transition">
              <div>
                <p className="font-medium text-slate-900">Weekly Report</p>
                <p className="text-sm text-slate-600">Weekly performance report</p>
              </div>
              <button
                onClick={() => toggleNotification('weeklyReport')}
                className={`w-12 h-6 rounded-full transition-all ${
                  notifications.weeklyReport ? 'bg-emerald-600' : 'bg-slate-300'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-all ${
                  notifications.weeklyReport ? 'ml-6' : 'ml-0.5'
                }`}></div>
              </button>
            </div>
            <div className="flex items-center justify-between p-4 border border-blue-200 rounded-lg hover:bg-blue-50 transition">
              <div>
                <p className="font-medium text-slate-900">Monthly Analytics</p>
                <p className="text-sm text-slate-600">Monthly analytics and insights</p>
              </div>
              <button
                onClick={() => toggleNotification('monthlyAnalytics')}
                className={`w-12 h-6 rounded-full transition-all ${
                  notifications.monthlyAnalytics ? 'bg-emerald-600' : 'bg-slate-300'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-all ${
                  notifications.monthlyAnalytics ? 'ml-6' : 'ml-0.5'
                }`}></div>
              </button>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

