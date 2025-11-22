"use client";
import { useState, useEffect } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { reportsApi } from "@/lib/api";

export default function ReportsPage() {
  const [selectedReport, setSelectedReport] = useState<any>(null);
  const [showGenerateModal, setShowGenerateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editingReport, setEditingReport] = useState<any>(null);
  const [reports, setReports] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const [formData, setFormData] = useState({
    name: "",
    report_type: "",
    period: "",
    format: "PDF",
    status: "ready"
  });

  // Fetch reports on mount
  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      setLoading(true);
      const response = await reportsApi.getAll();
      // Backend returns paginated response: { reports: [...], total: X, ... }
      setReports(response.reports || []);
    } catch (error) {
      console.error("Failed to fetch reports:", error);
      alert("Failed to load reports. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!formData.name || !formData.report_type || !formData.period) {
      alert("Please fill in all fields");
      return;
    }

    try {
      await reportsApi.create(formData);
      await fetchReports();
      setShowGenerateModal(false);
      setFormData({ name: "", report_type: "", period: "", format: "PDF", status: "ready" });
      alert("Report generated successfully!");
    } catch (error) {
      console.error("Failed to generate report:", error);
      alert("Failed to generate report. Please try again.");
    }
  };

  const handleEditReport = (report: any) => {
    setEditingReport(report);
    setFormData({
      name: report.name,
      report_type: report.report_type,
      period: report.period,
      format: report.format || "PDF",
      status: report.status
    });
    setShowEditModal(true);
  };

  const handleUpdateReport = async () => {
    if (!formData.name || !formData.report_type || !formData.period) {
      alert("Please fill in all fields");
      return;
    }

    try {
      await reportsApi.update(editingReport.id, formData);
      await fetchReports();
      setShowEditModal(false);
      setEditingReport(null);
      setFormData({ name: "", report_type: "", period: "", format: "PDF", status: "ready" });
      alert("Report updated successfully!");
    } catch (error) {
      console.error("Failed to update report:", error);
      alert("Failed to update report. Please try again.");
    }
  };

  const handleDeleteReport = async (reportId: string) => {
    if (confirm("Are you sure you want to delete this report? This action cannot be undone.")) {
      try {
        await reportsApi.delete(reportId);
        await fetchReports();
        alert("Report deleted successfully!");
      } catch (error) {
        console.error("Failed to delete report:", error);
        alert("Failed to delete report. Please try again.");
      }
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Reports</h1>
            <p className="text-slate-600 font-medium mt-1">Generate and view business reports</p>
          </div>
          <button
            onClick={() => setShowGenerateModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg active:scale-95"
          >
            + Generate Report
          </button>
        </div>

        {/* Quick Report Types */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <button className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg hover:border-blue-400 transition text-left">
            <p className="text-2xl mb-2">💰</p>
            <p className="font-bold text-slate-900">Sales Report</p>
            <p className="text-xs text-slate-600 mt-2">Revenue and deals</p>
          </button>
          <button className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg hover:border-blue-400 transition text-left">
            <p className="text-2xl mb-2">📊</p>
            <p className="font-bold text-slate-900">Pipeline Report</p>
            <p className="text-xs text-slate-600 mt-2">Lead progression</p>
          </button>
          <button className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg hover:border-blue-400 transition text-left">
            <p className="text-2xl mb-2">⚡</p>
            <p className="font-bold text-slate-900">Performance Report</p>
            <p className="text-xs text-slate-600 mt-2">Agent metrics</p>
          </button>
          <button className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md hover:shadow-lg hover:border-blue-400 transition text-left">
            <p className="text-2xl mb-2">📈</p>
            <p className="font-bold text-slate-900">Analytics Report</p>
            <p className="text-xs text-slate-600 mt-2">Trends & insights</p>
          </button>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Revenue</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">$2.4M</p>
            <p className="text-xs text-green-600 font-medium mt-2">↑ 12% vs last month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Deals Closed</p>
            <p className="text-3xl font-bold text-emerald-600 mt-2">156</p>
            <p className="text-xs text-green-600 font-medium mt-2">↑ 8% vs last month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Deal Size</p>
            <p className="text-3xl font-bold text-purple-600 mt-2">$15.4K</p>
            <p className="text-xs text-green-600 font-medium mt-2">↑ 3% vs last month</p>
          </div>
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Win Rate</p>
            <p className="text-3xl font-bold text-amber-600 mt-2">42.5%</p>
            <p className="text-xs text-green-600 font-medium mt-2">↑ 2.1% vs last month</p>
          </div>
        </div>

        {/* Reports Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Recent Reports</h2>
          </div>
          {loading ? (
            <div className="p-8 text-center text-slate-600">Loading reports...</div>
          ) : reports.length === 0 ? (
            <div className="p-8 text-center text-slate-600">No reports found. Generate your first report!</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-blue-50 border-b-2 border-blue-200">
                  <tr>
                    <th className="text-left p-4 text-slate-900 font-bold">Report Name</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Type</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Period</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Created</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Status</th>
                    <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {reports.map((report) => (
                    <tr key={report.id} className="border-t border-blue-100 hover:bg-blue-50 transition">
                      <td className="p-4 font-medium text-slate-900">{report.name}</td>
                      <td className="p-4 text-slate-700">{report.report_type}</td>
                      <td className="p-4 text-slate-700">{report.period}</td>
                      <td className="p-4 text-slate-700">{report.generated_date || 'N/A'}</td>
                      <td className="p-4">
                        <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
                          ✓ {report.status || 'Ready'}
                        </span>
                      </td>
                    <td className="p-4 space-x-2">
                      <button
                        onClick={() => setSelectedReport(report)}
                        className="text-blue-600 hover:text-blue-800 font-medium text-sm hover:underline"
                      >
                        View
                      </button>
                      <button
                        onClick={() => handleEditReport(report)}
                        className="text-amber-600 hover:text-amber-800 font-medium text-sm hover:underline"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDeleteReport(report.id)}
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

        {/* Generate Report Modal */}
        {showGenerateModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowGenerateModal(false)}>
            <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 shadow-xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <h3 className="text-2xl font-bold text-slate-900 mb-4">📊 Generate New Report</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Report Name *</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="e.g., Q4 Sales Performance"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Report Type *</label>
                  <select
                    value={formData.report_type}
                    onChange={(e) => setFormData({ ...formData, report_type: e.target.value })}
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  >
                    <option value="">Select report type...</option>
                    <option value="Sales">💰 Sales Report</option>
                    <option value="Pipeline">📊 Pipeline Report</option>
                    <option value="Performance">⚡ Performance Report</option>
                    <option value="Analytics">📈 Analytics Report</option>
                    <option value="Claims">🔔 Claims Report</option>
                    <option value="Customer">👥 Customer Report</option>
                    <option value="Policy">📄 Policy Report</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Report Period *</label>
                  <select
                    value={formData.period}
                    onChange={(e) => setFormData({ ...formData, period: e.target.value })}
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  >
                    <option value="">Select period...</option>
                    <option value="Today">Today</option>
                    <option value="This Week">This Week</option>
                    <option value="This Month">This Month</option>
                    <option value="Last Month">Last Month</option>
                    <option value="Q1 2024">Q1 2024</option>
                    <option value="Q2 2024">Q2 2024</option>
                    <option value="Q3 2024">Q3 2024</option>
                    <option value="Q4 2024">Q4 2024</option>
                    <option value="YTD 2024">YTD 2024</option>
                    <option value="2024">Full Year 2024</option>
                    <option value="Custom">Custom Range</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Format</label>
                  <select
                    value={formData.format}
                    onChange={(e) => setFormData({ ...formData, format: e.target.value })}
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  >
                    <option value="PDF">PDF Document</option>
                    <option value="Excel">Excel Spreadsheet</option>
                    <option value="CSV">CSV File</option>
                    <option value="JSON">JSON Data</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Include</label>
                  <div className="space-y-2">
                    <label className="flex items-center">
                      <input type="checkbox" defaultChecked className="mr-2" />
                      <span className="text-sm text-slate-700">Charts and Visualizations</span>
                    </label>
                    <label className="flex items-center">
                      <input type="checkbox" defaultChecked className="mr-2" />
                      <span className="text-sm text-slate-700">Summary Statistics</span>
                    </label>
                    <label className="flex items-center">
                      <input type="checkbox" className="mr-2" />
                      <span className="text-sm text-slate-700">Detailed Data Tables</span>
                    </label>
                    <label className="flex items-center">
                      <input type="checkbox" className="mr-2" />
                      <span className="text-sm text-slate-700">Trend Analysis</span>
                    </label>
                  </div>
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => {
                    setShowGenerateModal(false);
                    setFormData({ name: "", report_type: "", period: "", format: "PDF", status: "ready" });
                  }}
                  className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all"
                >
                  Cancel
                </button>
                <button
                  onClick={handleGenerateReport}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                >
                  Generate Report
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Edit Report Modal */}
        {showEditModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowEditModal(false)}>
            <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 shadow-xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <h3 className="text-2xl font-bold text-slate-900 mb-4">✏️ Edit Report</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Report Name *</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="e.g., Q4 Sales Performance"
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Report Type *</label>
                  <select
                    value={formData.report_type}
                    onChange={(e) => setFormData({ ...formData, report_type: e.target.value })}
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  >
                    <option value="">Select report type...</option>
                    <option value="Sales">💰 Sales Report</option>
                    <option value="Pipeline">📊 Pipeline Report</option>
                    <option value="Performance">⚡ Performance Report</option>
                    <option value="Analytics">📈 Analytics Report</option>
                    <option value="Claims">🔔 Claims Report</option>
                    <option value="Customer">👥 Customer Report</option>
                    <option value="Policy">📄 Policy Report</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Report Period *</label>
                  <select
                    value={formData.period}
                    onChange={(e) => setFormData({ ...formData, period: e.target.value })}
                    className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  >
                    <option value="">Select period...</option>
                    <option value="Today">Today</option>
                    <option value="This Week">This Week</option>
                    <option value="This Month">This Month</option>
                    <option value="Last Month">Last Month</option>
                    <option value="Q1 2024">Q1 2024</option>
                    <option value="Q2 2024">Q2 2024</option>
                    <option value="Q3 2024">Q3 2024</option>
                    <option value="Q4 2024">Q4 2024</option>
                    <option value="YTD 2024">YTD 2024</option>
                    <option value="2024">Full Year 2024</option>
                    <option value="Custom">Custom Range</option>
                  </select>
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => {
                    setShowEditModal(false);
                    setEditingReport(null);
                    setFormData({ name: "", report_type: "", period: "", format: "PDF", status: "ready" });
                  }}
                  className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all"
                >
                  Cancel
                </button>
                <button
                  onClick={handleUpdateReport}
                  className="flex-1 bg-amber-600 hover:bg-amber-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                >
                  Update Report
                </button>
              </div>
            </div>
          </div>
        )}

        {/* View Report Details Modal */}
        {selectedReport && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setSelectedReport(null)}>
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-slate-900">📊 Report Details</h2>
                <button onClick={() => setSelectedReport(null)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Report ID</label>
                    <p className="text-lg font-bold text-blue-700">{selectedReport.id}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Status</label>
                    <span className="inline-block px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
                      ✓ Ready
                    </span>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Report Name</label>
                  <p className="text-lg font-semibold text-slate-900">{selectedReport.name}</p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Report Type</label>
                    <p className="text-lg text-slate-700">{selectedReport.type}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-500 mb-1">Period</label>
                    <p className="text-lg text-slate-700">{selectedReport.period}</p>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1">Generated Date</label>
                  <p className="text-lg text-slate-700">{selectedReport.created}</p>
                </div>

                <div className="pt-4 border-t border-slate-200">
                  <div className="flex gap-3">
                    <button
                      onClick={() => setSelectedReport(null)}
                      className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Close
                    </button>
                    <button
                      onClick={() => alert('Download report functionality coming soon!')}
                      className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-4 rounded-lg transition-all active:scale-95"
                    >
                      Download Report
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

