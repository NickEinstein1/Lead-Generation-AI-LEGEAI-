"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function CustomersPage() {
  const [customers] = useState([
    { id: 1, name: "John Smith", email: "john@example.com", phone: "555-0101", status: "active", policies: 3, totalValue: "$45,000" },
    { id: 2, name: "Sarah Johnson", email: "sarah@example.com", phone: "555-0102", status: "active", policies: 2, totalValue: "$32,500" },
    { id: 3, name: "Michael Brown", email: "michael@example.com", phone: "555-0103", status: "inactive", policies: 1, totalValue: "$15,000" },
    { id: 4, name: "Emily Davis", email: "emily@example.com", phone: "555-0104", status: "active", policies: 4, totalValue: "$68,000" },
    { id: 5, name: "David Wilson", email: "david@example.com", phone: "555-0105", status: "active", policies: 2, totalValue: "$28,500" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Customers</h1>
            <p className="text-slate-600 font-medium mt-1">Manage all your insurance customers</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + New Customer
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Customers</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">48</p>
            <p className="text-xs text-green-600 font-medium mt-2">↑ 5 this month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">42</p>
            <p className="text-xs text-slate-600 font-medium mt-2">87.5% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Policies</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">156</p>
            <p className="text-xs text-slate-600 font-medium mt-2">3.25 avg per customer</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Value</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">$2.4M</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Average: $50K</p>
          </div>
        </div>

        {/* Customers Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">All Customers</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Name</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Email</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Phone</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Policies</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Total Value</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {customers.map((customer) => (
                  <tr key={customer.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                    <td className="p-4 font-medium text-slate-900">{customer.name}</td>
                    <td className="p-4 text-slate-700">{customer.email}</td>
                    <td className="p-4 text-slate-700">{customer.phone}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        customer.status === "active"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-slate-100 text-slate-700"
                      }`}>
                        {customer.status === "active" ? "✓ Active" : "⏸ Inactive"}
                      </span>
                    </td>
                    <td className="p-4 font-medium text-slate-900">{customer.policies}</td>
                    <td className="p-4 font-bold text-blue-700">{customer.totalValue}</td>
                    <td className="p-4">
                      <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">View</button>
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

