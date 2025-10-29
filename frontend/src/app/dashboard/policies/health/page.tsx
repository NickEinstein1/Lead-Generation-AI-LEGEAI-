"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function HealthInsurancePage() {
  const [policies] = useState([
    { id: "POL-HEALTH-001", customer: "John Smith", plan: "Premium Plus", premium: "$450/mo", status: "Active", expiry: "2025-01-15" },
    { id: "POL-HEALTH-002", customer: "Sarah Johnson", plan: "Standard", premium: "$320/mo", status: "Active", expiry: "2025-03-22" },
    { id: "POL-HEALTH-003", customer: "Michael Brown", plan: "Premium Plus", premium: "$480/mo", status: "Active", expiry: "2025-02-10" },
    { id: "POL-HEALTH-004", customer: "Emily Davis", plan: "Basic", premium: "$250/mo", status: "Active", expiry: "2025-04-18" },
    { id: "POL-HEALTH-005", customer: "David Wilson", plan: "Standard", premium: "$340/mo", status: "Active", expiry: "2025-05-14" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Health Insurance Policies</h1>
            <p className="text-slate-600 font-medium mt-1">Health and medical insurance coverage</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + New Health Policy
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Health Policies</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">41</p>
            <p className="text-xs text-slate-600 font-medium mt-2">26.3% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Monthly Premium</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">$15.8K</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $385/mo</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active Policies</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">41</p>
            <p className="text-xs text-slate-600 font-medium mt-2">100% active</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Satisfaction</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">4.6/5</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Customer rating</p>
          </div>
        </div>

        {/* Health Policies Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Health Insurance Policies</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Policy ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Plan Type</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Premium</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Expiry Date</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {policies.map((policy) => (
                  <tr key={policy.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">{policy.id}</td>
                    <td className="p-4 font-medium text-slate-900">{policy.customer}</td>
                    <td className="p-4 text-slate-700">{policy.plan}</td>
                    <td className="p-4 font-bold text-slate-900">{policy.premium}</td>
                    <td className="p-4">
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
                        {policy.status}
                      </span>
                    </td>
                    <td className="p-4 text-slate-700">{policy.expiry}</td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">View</button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Upgrade</button>
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

