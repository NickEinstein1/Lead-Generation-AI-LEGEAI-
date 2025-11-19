"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function ActiveCustomersPage() {
  const [selectedCustomer, setSelectedCustomer] = useState<any>(null);
  const [customers] = useState([
    { id: "CUST-001", name: "John Smith", email: "john@example.com", phone: "+1 (555) 123-4567", policies: 3, value: "$45,000", joinDate: "2023-01-15", status: "Active" },
    { id: "CUST-002", name: "Sarah Johnson", email: "sarah@example.com", phone: "+1 (555) 234-5678", policies: 2, value: "$32,500", joinDate: "2023-03-22", status: "Active" },
    { id: "CUST-003", name: "Michael Brown", email: "michael@example.com", phone: "+1 (555) 345-6789", policies: 4, value: "$58,000", joinDate: "2022-11-10", status: "Active" },
    { id: "CUST-004", name: "Emily Davis", email: "emily@example.com", phone: "+1 (555) 456-7890", policies: 2, value: "$28,500", joinDate: "2023-05-18", status: "Active" },
    { id: "CUST-005", name: "David Wilson", email: "david@example.com", phone: "+1 (555) 567-8901", policies: 3, value: "$42,000", joinDate: "2023-02-14", status: "Active" },
  ]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Active Customers</h1>
            <p className="text-slate-600 font-medium mt-1">Customers with active policies and ongoing relationships</p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg">
            + New Customer
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Active Customers</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">42</p>
            <p className="text-xs text-slate-600 font-medium mt-2">87.5% of total</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Policies</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">148</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: 3.5 per customer</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Value</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">$2.1M</p>
            <p className="text-xs text-slate-600 font-medium mt-2">Avg: $50K per customer</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Retention Rate</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">94%</p>
            <p className="text-xs text-slate-600 font-medium mt-2">↑ 3% vs last year</p>
          </div>
        </div>

        {/* Active Customers Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Active Customer List</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">Customer ID</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Name</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Email</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Phone</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Policies</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Total Value</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Join Date</th>
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
                    <td className="p-4">
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-blue-100 text-blue-700">
                        {customer.policies}
                      </span>
                    </td>
                    <td className="p-4 font-bold text-slate-900">{customer.value}</td>
                    <td className="p-4 text-slate-700">{customer.joinDate}</td>
                    <td className="p-4 space-x-2">
                      <button
                        onClick={() => setSelectedCustomer(customer)}
                        className="text-blue-600 hover:text-blue-800 font-medium text-sm hover:underline"
                      >
                        View
                      </button>
                      <button className="text-slate-600 hover:text-slate-800 font-medium text-sm hover:underline">Edit</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* View Customer Details Modal */}
        {selectedCustomer && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setSelectedCustomer(null)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">👤 Active Customer Details</h2>
                <button onClick={() => setSelectedCustomer(null)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Customer ID</label>
                    <p className="text-lg font-bold text-blue-700">{selectedCustomer.id}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Status</label>
                    <span className="inline-block px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
                      ✓ Active
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Customer Name</label>
                    <p className="text-lg font-semibold text-slate-900">{selectedCustomer.name}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Email Address</label>
                    <p className="text-lg text-slate-700">{selectedCustomer.email}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Phone Number</label>
                    <p className="text-lg text-slate-700">{selectedCustomer.phone}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Join Date</label>
                    <p className="text-lg text-slate-700">{selectedCustomer.joinDate}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Active Policies</label>
                    <p className="text-lg font-bold text-blue-700">{selectedCustomer.policies}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Total Value</label>
                    <p className="text-lg font-bold text-slate-900">{selectedCustomer.value}</p>
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-200">
                  <div className="flex gap-3">
                    <button
                      onClick={() => setSelectedCustomer(null)}
                      className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Close
                    </button>
                    <button
                      onClick={() => alert('Edit customer functionality coming soon!')}
                      className="flex-1 bg-slate-600 hover:bg-slate-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Edit Customer
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}

