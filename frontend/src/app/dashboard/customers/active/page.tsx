"use client";
import { useState, useEffect } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { customersApi } from "@/lib/api";

export default function ActiveCustomersPage() {
  const [selectedCustomer, setSelectedCustomer] = useState<any>(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editingCustomer, setEditingCustomer] = useState<any>(null);
  const [customers, setCustomers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const [formData, setFormData] = useState({
    name: "",
    email: "",
    phone: "",
    status: "active",
    policies_count: 0,
    total_value: 0,
    join_date: "",
    last_active: ""
  });

  // Fetch active customers on mount
  useEffect(() => {
    fetchActiveCustomers();
  }, []);

  const fetchActiveCustomers = async () => {
    try {
      setLoading(true);
      const data = await customersApi.getAll("active");
      setCustomers(data);
    } catch (error) {
      console.error("Failed to fetch active customers:", error);
      alert("Failed to load active customers. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleEditCustomer = (customer: any) => {
    setEditingCustomer(customer);
    setFormData({
      name: customer.name,
      email: customer.email,
      phone: customer.phone,
      status: customer.status,
      policies_count: customer.policies_count || 0,
      total_value: customer.total_value || 0,
      join_date: customer.join_date || "",
      last_active: customer.last_active || ""
    });
    setShowEditModal(true);
  };

  const handleUpdateCustomer = async () => {
    if (!formData.name || !formData.email || !formData.phone) {
      alert("Please fill in all required fields");
      return;
    }

    try {
      await customersApi.update(editingCustomer.id, formData);
      await fetchActiveCustomers();
      setShowEditModal(false);
      setEditingCustomer(null);
      setFormData({ name: "", email: "", phone: "", status: "active", policies_count: 0, total_value: 0, join_date: "", last_active: "" });
      alert("Customer updated successfully!");
    } catch (error) {
      console.error("Failed to update customer:", error);
      alert("Failed to update customer. Please try again.");
    }
  };

  const handleDeleteCustomer = async (customerId: string) => {
    if (confirm("Are you sure you want to delete this customer? This action cannot be undone.")) {
      try {
        await customersApi.delete(customerId);
        await fetchActiveCustomers();
        alert("Customer deleted successfully!");
      } catch (error) {
        console.error("Failed to delete customer:", error);
        alert("Failed to delete customer. Please try again.");
      }
    }
  };

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
          {loading ? (
            <div className="p-8 text-center text-slate-600">Loading active customers...</div>
          ) : customers.length === 0 ? (
            <div className="p-8 text-center text-slate-600">No active customers found.</div>
          ) : (
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
                          {customer.policies_count || 0}
                        </span>
                      </td>
                      <td className="p-4 font-bold text-slate-900">${customer.total_value?.toLocaleString() || '0'}</td>
                      <td className="p-4 text-slate-700">{customer.join_date || 'N/A'}</td>
                    <td className="p-4 space-x-2">
                      <button
                        onClick={() => setSelectedCustomer(customer)}
                        className="text-blue-600 hover:text-blue-800 font-medium text-sm hover:underline"
                      >
                        View
                      </button>
                      <button
                        onClick={() => handleEditCustomer(customer)}
                        className="text-amber-600 hover:text-amber-800 font-medium text-sm hover:underline"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDeleteCustomer(customer.id)}
                        className="text-red-600 hover:text-red-800 font-medium text-sm hover:underline"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Edit Customer Modal */}
        {showEditModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowEditModal(false)}>
            <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 shadow-xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <h3 className="text-2xl font-bold text-slate-900 mb-4">✏️ Edit Active Customer</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Full Name *</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({...formData, name: e.target.value})}
                    placeholder="John Smith"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-900 mb-2">Email *</label>
                    <input
                      type="email"
                      value={formData.email}
                      onChange={(e) => setFormData({...formData, email: e.target.value})}
                      placeholder="john@example.com"
                      className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-900 mb-2">Phone *</label>
                    <input
                      type="tel"
                      value={formData.phone}
                      onChange={(e) => setFormData({...formData, phone: e.target.value})}
                      placeholder="+1 (555) 123-4567"
                      className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-900 mb-2">Number of Policies</label>
                    <input
                      type="number"
                      value={formData.policies}
                      onChange={(e) => setFormData({...formData, policies: parseInt(e.target.value) || 0})}
                      placeholder="0"
                      className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-900 mb-2">Total Value</label>
                    <input
                      type="text"
                      value={formData.value}
                      onChange={(e) => setFormData({...formData, value: e.target.value})}
                      placeholder="$0"
                      className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Join Date</label>
                  <input
                    type="date"
                    value={formData.joinDate}
                    onChange={(e) => setFormData({...formData, joinDate: e.target.value})}
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => {
                    setShowEditModal(false);
                    setEditingCustomer(null);
                    setFormData({ name: "", email: "", phone: "", policies: 0, value: "$0", joinDate: "" });
                  }}
                  className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all"
                >
                  Cancel
                </button>
                <button
                  onClick={handleUpdateCustomer}
                  className="flex-1 bg-amber-600 hover:bg-amber-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                >
                  Update Customer
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

