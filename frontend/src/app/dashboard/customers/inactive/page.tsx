"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function InactiveCustomersPage() {
  const [customers] = useState([
    { id: "CUST-045", name: "Robert Taylor", email: "robert@example.com", phone: "+1 (555) 123-4567", policies: 1, value: "$15,000", lastActive: "2024-06-15", reason: "Policy Expired" },
    { id: "CUST-046", name: "Jennifer Lee", email: "jennifer@example.com", phone: "+1 (555) 234-5678", policies: 0, value: "$0", lastActive: "2024-05-22", reason: "Cancelled" },
    { id: "CUST-047", name: "Christopher Martin", email: "chris@example.com", phone: "+1 (555) 345-6789", policies: 1, value: "$8,500", lastActive: "2024-04-10", reason: "No Activity" },
    { id: "CUST-048", name: "Patricia Garcia", email: "patricia@example.com", phone: "+1 (555) 456-7890", policies: 0, value: "$0", lastActive: "2024-03-18", reason: "Cancelled" },
    { id: "CUST-049", name: "Daniel Rodriguez", email: "daniel@example.com", phone: "+1 (555) 567-8901", policies: 1, value: "$12,000", lastActive: "2024-02-28", reason: "No Activity" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Inactive Customers</h1>
            <p className="text-slate-600 font-medium mt-1">Customers with no active policies or expired relationships</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + Re-engage Customer
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Inactive Customers</p>
            <p className="text-3xl font-bold text-slate-600 mt-2">6</p>
            <p className="text-xs text-slate-600 font-medium mt-2">12.5% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Inactivity</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">4.2 mo</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Since last activity</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Potential Recovery</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">$35.5K</p>
            <p className="text-xs text-slate-600 font-medium mt-2">If re-engaged</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Churn Rate</p>
            <p className="text-3xl font-bold text-red-600 mt-2">6%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">↓ 2% vs last year</p>
          </div>
        </div>

        {/* Inactive Customers Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Inactive Customer List</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Name</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Email</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Phone</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Last Active</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Reason</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {customers.map((customer) => (
                  <tr key={customer.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-bold text-blue-700">{customer.id}</td>
                    <td className="p-4 font-medium text-slate-900">{customer.name}</td>
                    <td className="p-4 text-slate-700">{customer.email}</td>
                    <td className="p-4 text-slate-700">{customer.phone}</td>
                    <td className="p-4 text-slate-700">{customer.lastActive}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        customer.reason === "Cancelled"
                          ? "bg-red-100 text-red-700"
                          : "bg-amber-100 text-amber-700"
                      }`}>
                        {customer.reason}
                      </span>
                    </td>
                    <td className="p-4 space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">Re-engage</button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm">View</button>
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

