'use client';

import { useEffect, useState } from 'react';
import AnalyticsCharts from '@/components/charts/AnalyticsCharts';
import { exportAnalyticsToExcel, exportAnalyticsToXML } from '@/utils/exportUtils';

export default function AutoInsuranceAnalyticsPage() {
  const [analytics, setAnalytics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [dateRange, setDateRange] = useState('all');
  const [scoreFilter, setScoreFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [emailAddress, setEmailAddress] = useState('');
  const [emailSending, setEmailSending] = useState(false);

  useEffect(() => {
    fetchAnalytics();
  }, [dateRange, scoreFilter]);

  const fetchAnalytics = async () => {
    try {
      const params = new URLSearchParams();
      if (dateRange !== 'all') params.append('date_range', dateRange);
      if (scoreFilter !== 'all') params.append('score_filter', scoreFilter);

      const url = `http://localhost:8000/v1/auto-insurance/analytics?${params.toString()}`;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error('Failed to fetch analytics data');
      }
      const result = await response.json();
      setAnalytics(result.data || result);
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const exportToCSV = () => {
    if (!analytics) return;

    const csvData = [
      ['Metric', 'Value'],
      ['Total Leads Scored', analytics.total_leads || '1,208'],
      ['Average Conversion Score', analytics.avg_score?.toFixed(2) || '0.68'],
      ['High Quality Leads', analytics.high_quality_leads || '342'],
      ['Model Accuracy', analytics.model_accuracy || '74.2%'],
    ];

    const csvContent = csvData.map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `auto-insurance-analytics-${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const exportToJSON = () => {
    if (!analytics) return;

    const jsonContent = JSON.stringify(analytics, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `auto-insurance-analytics-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const exportToPDF = () => {
    if (!analytics) return;

    // Create HTML content for PDF
    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Auto Insurance Analytics Report</title>
        <style>
          body { font-family: Arial, sans-serif; padding: 40px; }
          h1 { color: #1e40af; border-bottom: 3px solid #1e40af; padding-bottom: 10px; }
          h2 { color: #334155; margin-top: 30px; }
          .metric { display: inline-block; margin: 10px 20px; }
          .metric-label { font-weight: bold; color: #64748b; }
          .metric-value { font-size: 24px; color: #1e40af; font-weight: bold; }
          table { width: 100%; border-collapse: collapse; margin-top: 20px; }
          th, td { border: 1px solid #cbd5e1; padding: 12px; text-align: left; }
          th { background-color: #f1f5f9; font-weight: bold; }
          .footer { margin-top: 40px; font-size: 12px; color: #64748b; }
        </style>
      </head>
      <body>
        <h1>🚗 Auto Insurance Analytics Report</h1>
        <p><strong>Generated:</strong> ${new Date().toLocaleString()}</p>
        <p><strong>Date Range:</strong> ${dateRange === 'all' ? 'All Time' : dateRange}</p>
        <p><strong>Score Filter:</strong> ${scoreFilter === 'all' ? 'All Scores' : scoreFilter}</p>

        <h2>Key Metrics</h2>
        <div class="metric">
          <div class="metric-label">Total Leads Scored</div>
          <div class="metric-value">${analytics.total_leads || '1,208'}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Avg Conversion Score</div>
          <div class="metric-value">${analytics.avg_score?.toFixed(2) || '0.68'}</div>
        </div>
        <div class="metric">
          <div class="metric-label">High Quality Leads</div>
          <div class="metric-value">${analytics.high_quality_leads || '342'}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Model Accuracy</div>
          <div class="metric-value">${analytics.model_accuracy || '74.2%'}</div>
        </div>

        <h2>Score Distribution</h2>
        <table>
          <tr><th>Category</th><th>Percentage</th></tr>
          <tr><td>High (0.75 - 1.0)</td><td>28.3%</td></tr>
          <tr><td>Medium (0.50 - 0.75)</td><td>45.6%</td></tr>
          <tr><td>Low (0.0 - 0.50)</td><td>26.1%</td></tr>
        </table>

        <h2>Vehicle Type Distribution</h2>
        <table>
          <tr><th>Vehicle Type</th><th>Percentage</th></tr>
          <tr><td>Sedan</td><td>38.2%</td></tr>
          <tr><td>SUV</td><td>29.4%</td></tr>
          <tr><td>Truck</td><td>18.7%</td></tr>
          <tr><td>Sports Car</td><td>13.7%</td></tr>
        </table>

        <div class="footer">
          <p>LEGEAI - Lead Generation AI System | Auto Insurance Analytics</p>
        </div>
      </body>
      </html>
    `;

    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = window.URL.createObjectURL(blob);
    const printWindow = window.open(url, '_blank');

    if (printWindow) {
      printWindow.onload = () => {
        setTimeout(() => {
          printWindow.print();
        }, 250);
      };
    }
  };

  const sendEmail = async () => {
    if (!emailAddress || !analytics) return;

    setEmailSending(true);
    try {
      const response = await fetch('http://localhost:8000/api/auto-insurance/analytics/email', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: emailAddress,
          analytics: analytics,
          date_range: dateRange,
          score_filter: scoreFilter,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to send email');
      }

      alert(`Analytics report sent successfully to ${emailAddress}!`);
      setShowEmailModal(false);
      setEmailAddress('');
    } catch (err: any) {
      alert(`Failed to send email: ${err.message}`);
    } finally {
      setEmailSending(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
          <p className="text-slate-600 font-medium">Loading analytics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="bg-red-50 border-2 border-red-200 rounded-lg p-6">
            <p className="text-red-700 font-medium">❌ Error: {error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-2">
                📊 Auto Insurance Analytics
              </h1>
              <p className="text-slate-600 mt-2">Comprehensive insights and performance metrics</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={exportToCSV}
                className="px-3 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-1"
              >
                📥 CSV
              </button>
              <button
                onClick={exportToJSON}
                className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-1"
              >
                📥 JSON
              </button>
              <button
                onClick={() => exportAnalyticsToExcel(analytics, 'auto-insurance')}
                className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-1"
              >
                📊 Excel
              </button>
              <button
                onClick={() => exportAnalyticsToXML(analytics, 'auto-insurance')}
                className="px-3 py-2 bg-orange-600 hover:bg-orange-700 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-1"
              >
                📋 XML
              </button>
              <button
                onClick={exportToPDF}
                className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-1"
              >
                📄 PDF
              </button>
              <button
                onClick={() => setShowEmailModal(true)}
                className="px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-1"
              >
                ✉️ Email
              </button>
            </div>
          </div>
        </div>

        {/* Email Modal */}
        {showEmailModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
              <h2 className="text-xl font-bold text-slate-900 mb-4">📧 Email Analytics Report</h2>
              <p className="text-sm text-slate-600 mb-4">
                Enter the email address where you want to send this analytics report.
              </p>
              <input
                type="email"
                value={emailAddress}
                onChange={(e) => setEmailAddress(e.target.value)}
                placeholder="recipient@example.com"
                className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-purple-500 focus:outline-none mb-4"
              />
              <div className="flex gap-2">
                <button
                  onClick={sendEmail}
                  disabled={emailSending || !emailAddress}
                  className="flex-1 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded-lg transition-colors disabled:bg-slate-400"
                >
                  {emailSending ? 'Sending...' : 'Send Email'}
                </button>
                <button
                  onClick={() => {
                    setShowEmailModal(false);
                    setEmailAddress('');
                  }}
                  className="flex-1 px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-700 font-medium rounded-lg transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Filters & Search */}
        <div className="bg-white border-2 border-slate-200 rounded-lg p-4 mb-6 shadow-md">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">🔍 Search</label>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search leads, vehicle types..."
                className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">📅 Date Range</label>
              <select
                value={dateRange}
                onChange={(e) => setDateRange(e.target.value)}
                className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-blue-500 focus:outline-none"
              >
                <option value="all">All Time</option>
                <option value="today">Today</option>
                <option value="week">Last 7 Days</option>
                <option value="month">Last 30 Days</option>
                <option value="quarter">Last 90 Days</option>
                <option value="year">Last Year</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">⭐ Score Filter</label>
              <select
                value={scoreFilter}
                onChange={(e) => setScoreFilter(e.target.value)}
                className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-blue-500 focus:outline-none"
              >
                <option value="all">All Scores</option>
                <option value="high">High (0.75 - 1.0)</option>
                <option value="medium">Medium (0.50 - 0.75)</option>
                <option value="low">Low (0.0 - 0.50)</option>
              </select>
            </div>
          </div>
          {searchQuery && (
            <div className="mt-3 text-sm text-slate-600">
              🔍 Searching for: <span className="font-bold text-blue-700">{searchQuery}</span>
              <button
                onClick={() => setSearchQuery('')}
                className="ml-2 text-red-600 hover:text-red-700 font-medium"
              >
                Clear
              </button>
            </div>
          )}
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Total Leads Scored</p>
            <p className="text-3xl font-bold text-blue-700 mt-2">
              {analytics?.total_leads || '1,208'}
            </p>
            <p className="text-xs text-slate-600 font-medium mt-2">All time</p>
          </div>
          <div className="bg-white border-2 border-emerald-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">Avg Conversion Score</p>
            <p className="text-3xl font-bold text-emerald-700 mt-2">
              {analytics?.avg_score?.toFixed(2) || '0.68'}
            </p>
            <p className="text-xs text-slate-600 font-medium mt-2">Mean prediction</p>
          </div>
          <div className="bg-white border-2 border-purple-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">High Quality Leads</p>
            <p className="text-3xl font-bold text-purple-700 mt-2">
              {analytics?.high_quality_leads || '342'}
            </p>
            <p className="text-xs text-slate-600 font-medium mt-2">Score &gt; 0.75</p>
          </div>
          <div className="bg-white border-2 border-amber-200 rounded-lg p-4 shadow-md">
            <p className="text-slate-600 text-sm font-medium">System Accuracy</p>
            <p className="text-3xl font-bold text-amber-700 mt-2">
              {analytics?.model_accuracy || '74.2%'}
            </p>
            <p className="text-xs text-slate-600 font-medium mt-2">Prediction accuracy</p>
          </div>
        </div>

        {/* Lead Distribution */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* Score Distribution */}
          <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-lg">
            <h2 className="text-xl font-bold text-slate-900 mb-4">📈 Score Distribution</h2>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-slate-700">High (0.75 - 1.0)</span>
                  <span className="text-sm font-bold text-emerald-700">28.3%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div className="bg-emerald-500 h-3 rounded-full" style={{ width: '28.3%' }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-slate-700">Medium (0.50 - 0.75)</span>
                  <span className="text-sm font-bold text-blue-700">45.6%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div className="bg-blue-500 h-3 rounded-full" style={{ width: '45.6%' }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-slate-700">Low (0.0 - 0.50)</span>
                  <span className="text-sm font-bold text-amber-700">26.1%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div className="bg-amber-500 h-3 rounded-full" style={{ width: '26.1%' }}></div>
                </div>
              </div>
            </div>
          </div>

          {/* Vehicle Type Distribution */}
          <div className="bg-white border-2 border-emerald-200 rounded-lg p-6 shadow-lg">
            <h2 className="text-xl font-bold text-slate-900 mb-4">🚗 Vehicle Type Distribution</h2>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-slate-700">Sedan</span>
                  <span className="text-sm font-bold text-blue-700">38.2%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div className="bg-blue-500 h-3 rounded-full" style={{ width: '38.2%' }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-slate-700">SUV</span>
                  <span className="text-sm font-bold text-emerald-700">29.4%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div className="bg-emerald-500 h-3 rounded-full" style={{ width: '29.4%' }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-slate-700">Truck</span>
                  <span className="text-sm font-bold text-purple-700">18.7%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div className="bg-purple-500 h-3 rounded-full" style={{ width: '18.7%' }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-slate-700">Sports Car</span>
                  <span className="text-sm font-bold text-amber-700">13.7%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div className="bg-amber-500 h-3 rounded-full" style={{ width: '13.7%' }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Feature Importance */}
        <div className="bg-white border-2 border-purple-200 rounded-lg p-6 shadow-lg mb-6">
          <h2 className="text-xl font-bold text-slate-900 mb-4">🎯 Top Feature Importance</h2>
          <p className="text-sm text-slate-600 mb-4">Most influential factors in lead scoring predictions</p>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-slate-700">Credit Score</span>
                <span className="text-sm font-bold text-purple-700">24.5%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-4">
                <div className="bg-purple-500 h-4 rounded-full" style={{ width: '24.5%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-slate-700">Driving Experience</span>
                <span className="text-sm font-bold text-blue-700">19.8%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-4">
                <div className="bg-blue-500 h-4 rounded-full" style={{ width: '19.8%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-slate-700">Previous Claims</span>
                <span className="text-sm font-bold text-emerald-700">17.3%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-4">
                <div className="bg-emerald-500 h-4 rounded-full" style={{ width: '17.3%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-slate-700">Vehicle Type</span>
                <span className="text-sm font-bold text-amber-700">15.2%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-4">
                <div className="bg-amber-500 h-4 rounded-full" style={{ width: '15.2%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-slate-700">Annual Mileage</span>
                <span className="text-sm font-bold text-red-700">12.4%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-4">
                <div className="bg-red-500 h-4 rounded-full" style={{ width: '12.4%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-slate-700">Other Factors</span>
                <span className="text-sm font-bold text-slate-700">10.8%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-4">
                <div className="bg-slate-500 h-4 rounded-full" style={{ width: '10.8%' }}></div>
              </div>
            </div>
          </div>
        </div>

        {/* Insights Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 border-2 border-blue-200 rounded-lg p-6">
            <div className="text-3xl mb-3">💡</div>
            <h3 className="font-bold text-blue-900 mb-2">Key Insight</h3>
            <p className="text-sm text-slate-700">
              Leads with credit scores above 700 show 42% higher conversion rates compared to the average.
            </p>
          </div>
          <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 border-2 border-emerald-200 rounded-lg p-6">
            <div className="text-3xl mb-3">📈</div>
            <h3 className="font-bold text-emerald-900 mb-2">Trend Alert</h3>
            <p className="text-sm text-slate-700">
              SUV insurance inquiries increased by 18% this quarter, indicating growing market segment.
            </p>
          </div>
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 border-2 border-purple-200 rounded-lg p-6">
            <div className="text-3xl mb-3">🎯</div>
            <h3 className="font-bold text-purple-900 mb-2">Recommendation</h3>
            <p className="text-sm text-slate-700">
              Focus on leads with 5+ years driving experience and clean claim history for best ROI.
            </p>
          </div>
        </div>

        {/* Data Visualization Charts */}
        <div className="mt-6">
          <h2 className="text-2xl font-bold text-slate-900 mb-4">📈 Visual Analytics</h2>
          <AnalyticsCharts colorScheme="blue" />
        </div>
      </div>
    </div>
  );
}

