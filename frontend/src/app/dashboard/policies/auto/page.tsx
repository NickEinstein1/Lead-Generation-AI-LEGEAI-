"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function AutoInsurancePage() {
  const [policies] = useState([
    { id: "POL-AUTO-001", customer: "John Smith", vehicle: "2022 Toyota Camry", premium: "$1,200/yr", status: "Active", expiry: "2025-01-15" },
    { id: "POL-AUTO-002", customer: "Sarah Johnson", vehicle: "2021 Honda Civic", premium: "$950/yr", status: "Active", expiry: "2025-03-22" },
    { id: "POL-AUTO-003", customer: "Michael Brown", vehicle: "2023 Ford F-150", premium: "$1,450/yr", status: "Active", expiry: "2025-02-10" },
    { id: "POL-AUTO-004", customer: "Emily Davis", vehicle: "2020 Mazda 3", premium: "$875/yr", status: "Expiring Soon", expiry: "2024-11-18" },
    { id: "POL-AUTO-005", customer: "David Wilson", vehicle: "2019 Chevrolet Malibu", premium: "$1,100/yr", status: "Active", expiry: "2025-05-14" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Auto Insurance Policies</h1>
            <p className="text-slate-600 font-medium mt-1">Vehicle insurance coverage and management</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + New Auto Policy
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Auto Policies</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">42</p>
            <p className="text-xs text-slate-600 font-medium mt-2">26.9% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Annual Premium</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">$48.6K</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $1,157</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active Policies</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">40</p>
            <p className="text-xs text-slate-600 font-medium mt-2">95.2% active</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Expiring Soon</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">2</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Next 30 days</p>
          </div>
        </div>

        {/* Auto Policies Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Auto Insurance Policies</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Policy ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Vehicle</th>
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
                    <td className="p-4 text-slate-700">{policy.vehicle}</td>
                    <td className="p-4 font-bold text-slate-900">{policy.premium}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        policy.status === "Active"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-amber-100 text-amber-700"
                      }`}>
                        {policy.status}
                      </span>
                    </td>
                    <td className="p-4 text-slate-700">{policy.expiry}</td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">View</button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">Renew</button>
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

