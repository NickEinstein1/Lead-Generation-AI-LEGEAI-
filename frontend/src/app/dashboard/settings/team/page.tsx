"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function TeamSettingsPage() {
  const [teamMembers] = useState([
    { id: 1, name: "John Smith", email: "john@leagai.dev", role: "Admin", status: "Active", joinDate: "2023-01-15" },
    { id: 2, name: "Sarah Johnson", email: "sarah@leagai.dev", role: "Manager", status: "Active", joinDate: "2023-03-22" },
    { id: 3, name: "Michael Brown", email: "michael@leagai.dev", role: "Agent", status: "Active", joinDate: "2023-05-10" },
    { id: 4, name: "Emily Davis", email: "emily@leagai.dev", role: "Agent", status: "Active", joinDate: "2023-06-18" },
    { id: 5, name: "David Wilson", email: "david@leagai.dev", role: "Viewer", status: "Inactive", joinDate: "2023-02-14" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Team Management</h1>
            <p className="text-slate-600 font-medium mt-1">Manage team members and their roles</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Add Team Member
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Members</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">12</p>
            <p className="text-xs text-slate-600 font-medium mt-2">In organization</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active Members</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">11</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Currently active</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Admins</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">2</p>
            <p className="text-xs text-slate-600 font-medium mt-2">With full access</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Agents</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">7</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Sales team</p>
          </div>
        </div>

        {/* Team Members Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Team Members</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Name</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Email</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Role</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Join Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {teamMembers.map((member) => (
                  <tr key={member.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-medium text-slate-900">{member.name}</td>
                    <td className="p-4 text-slate-700">{member.email}</td>
                    <td className="p-4">
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-blue-100 text-blue-700">
                        {member.role}
                      </span>
                    </td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        member.status === "Active"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-slate-100 text-slate-700"
                      }`}>
                        {member.status}
                      </span>
                    </td>
                    <td className="p-4 text-slate-700">{member.joinDate}</td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Edit</button>
                      <button className="text-red-600 hover:text-red-800 font-medium text-sm">Remove</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

