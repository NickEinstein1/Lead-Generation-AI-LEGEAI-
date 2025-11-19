"use client";
import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";

export default function CustomersPage() {
  const [showNewCustomerModal, setShowNewCustomerModal] = useState(false);
  const [selectedCustomer, setSelectedCustomer] = useState<any>(null);
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
          <button
            onClick={() => setShowNewCustomerModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
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
                      <button
                        onClick={() => setSelectedCustomer(customer)}
                        className="text-blue-600 hover:text-blue-800 font-medium text-sm hover:underline"
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* New Customer Modal */}
      {showNewCustomerModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowNewCustomerModal(false)}>
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 shadow-xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-2xl font-bold text-slate-900 mb-4">👤 New Customer</h3>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">First Name *</label>
                  <input
                    type="text"
                    placeholder="John"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Last Name *</label>
                  <input
                    type="text"
                    placeholder="Smith"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Email *</label>
                  <input
                    type="email"
                    placeholder="john@example.com"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Phone *</label>
                  <input
                    type="tel"
                    placeholder="+1 (555) 123-4567"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Address</label>
                <input
                  type="text"
                  placeholder="123 Main St, City, State ZIP"
                  className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Date of Birth</label>
                  <input
                    type="date"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Customer Type</label>
                  <select className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600">
                    <option value="">Select type...</option>
                    <option value="individual">Individual</option>
                    <option value="business">Business</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Notes</label>
                <textarea
                  className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  rows={3}
                  placeholder="Additional customer information..."
                />
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowNewCustomerModal(false)}
                className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowNewCustomerModal(false);
                  // In a real app, this would create the customer
                }}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Create Customer
              </button>
            </div>
          </div>
        </div>
      )}

      {/* View Customer Details Modal */}
      {selectedCustomer && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setSelectedCustomer(null)}>
          <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-slate-900">👤 Customer Details</h2>
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
                  <span className={`inline-block px-3 py-1 rounded-full text-xs font-bold ${
                    selectedCustomer.status === "active"
                      ? "bg-emerald-100 text-emerald-700"
                      : "bg-slate-100 text-slate-700"
                  }`}>
                    {selectedCustomer.status === "active" ? "✓ Active" : "⏸ Inactive"}
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
                  <p className="text-lg font-bold text-slate-900">{selectedCustomer.policies}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Total Policy Value</label>
                  <p className="text-lg font-bold text-blue-700">{selectedCustomer.totalValue}</p>
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
                  <button
                    onClick={() => alert('View policies functionality coming soon!')}
                    className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                  >
                    View Policies
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}

