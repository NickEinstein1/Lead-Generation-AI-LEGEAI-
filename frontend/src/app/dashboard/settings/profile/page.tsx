"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function ProfileSettingsPage() {
  const [profile] = useState({
    name: "John Smith",
    email: "john@leagai.dev",
    phone: "+1 (555) 123-4567",
    role: "Admin",
    department: "Management",
    joinDate: "2023-01-15",
    lastLogin: "2024-10-22 14:32:15",
    status: "Active",
  });

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Profile Settings</h1>
            <p className="text-slate-600 font-medium mt-1">Manage your personal profile information</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Edit Profile
          </button>
        </div>

        {/* Profile Information */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Personal Info Card */}
          <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
            <h2 className="text-xl font-bold text-slate-900 mb-4">Personal Information</h2>
            <div className="space-y-4">
              <div>
                <p className="text-slate-600 text-sm font-medium">Full Name</p>
                <p className="text-slate-900 font-bold mt-1">{profile.name}</p>
              </div>
              <div>
                <p className="text-slate-600 text-sm font-medium">Email Address</p>
                <p className="text-slate-900 font-bold mt-1">{profile.email}</p>
              </div>
              <div>
                <p className="text-slate-600 text-sm font-medium">Phone Number</p>
                <p className="text-slate-900 font-bold mt-1">{profile.phone}</p>
              </div>
              <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all mt-4">
                Edit Personal Info
              </button>
            </div>
          </div>

          {/* Role & Department Card */}
          <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
            <h2 className="text-xl font-bold text-slate-900 mb-4">Role & Department</h2>
            <div className="space-y-4">
              <div>
                <p className="text-slate-600 text-sm font-medium">Role</p>
                <p className="text-slate-900 font-bold mt-1">{profile.role}</p>
              </div>
              <div>
                <p className="text-slate-600 text-sm font-medium">Department</p>
                <p className="text-slate-900 font-bold mt-1">{profile.department}</p>
              </div>
              <div>
                <p className="text-slate-600 text-sm font-medium">Status</p>
                <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700 mt-1 inline-block">
                  {profile.status}
                </span>
              </div>
              <button className="w-full bg-slate-600 hover:bg-slate-700 text-white font-semibold py-2 px-4 rounded-lg transition-all mt-4">
                View Permissions
              </button>
            </div>
          </div>
        </div>

        {/* Account Activity */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
          <h2 className="text-xl font-bold text-slate-900 mb-4">Account Activity</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="border-l-4 border-blue-600 pl-4">
              <p className="text-slate-600 text-sm font-medium">Join Date</p>
              <p className="text-slate-900 font-bold mt-1">{profile.joinDate}</p>
            </div>
            <div className="border-l-4 border-emerald-600 pl-4">
              <p className="text-slate-600 text-sm font-medium">Last Login</p>
              <p className="text-slate-900 font-bold mt-1">{profile.lastLogin}</p>
            </div>
            <div className="border-l-4 border-purple-600 pl-4">
              <p className="text-slate-600 text-sm font-medium">Account Age</p>
              <p className="text-slate-900 font-bold mt-1">1 year, 9 months</p>
            </div>
          </div>
        </div>

        {/* Security Settings */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
          <h2 className="text-xl font-bold text-slate-900 mb-4">Security Settings</h2>
          <div className="space-y-3">
            <button className="w-full text-left p-4 border border-blue-200 rounded-lg hover:bg-blue-50 transition flex items-center justify-between">
              <span className="font-medium text-slate-900">Change Password</span>
              <span className="text-blue-600">→</span>
            </button>
            <button className="w-full text-left p-4 border border-blue-200 rounded-lg hover:bg-blue-50 transition flex items-center justify-between">
              <span className="font-medium text-slate-900">Two-Factor Authentication</span>
              <span className="text-blue-600">→</span>
            </button>
            <button className="w-full text-left p-4 border border-blue-200 rounded-lg hover:bg-blue-50 transition flex items-center justify-between">
              <span className="font-medium text-slate-900">Active Sessions</span>
              <span className="text-blue-600">→</span>
            </button>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

