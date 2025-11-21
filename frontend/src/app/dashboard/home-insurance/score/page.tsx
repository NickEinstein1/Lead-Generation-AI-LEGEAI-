'use client';

import { useState, useEffect } from 'react';

interface ScoreHistory {
  id: string;
  timestamp: string;
  formData: any;
  prediction: any;
}

export default function HomeInsuranceScorePage() {
  const [formData, setFormData] = useState({
    property_age: '',
    property_type: '',
    square_footage: '',
    num_bedrooms: '',
    num_bathrooms: '',
    construction_type: '',
    roof_age: '',
    location_risk: '',
    security_features: '',
    previous_claims: '',
  });

  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [scoreHistory, setScoreHistory] = useState<ScoreHistory[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showShareModal, setShowShareModal] = useState(false);
  const [shareEmail, setShareEmail] = useState('');
  const [shareSending, setShareSending] = useState(false);

  useEffect(() => {
    const savedHistory = localStorage.getItem('home-insurance-score-history');
    if (savedHistory) {
      setScoreHistory(JSON.parse(savedHistory));
    }
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:8000/api/home-insurance/score', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to score lead');
      }

      const data = await response.json();
      setPrediction(data);

      const newHistoryItem: ScoreHistory = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        formData: { ...formData },
        prediction: data,
      };

      const updatedHistory = [newHistoryItem, ...scoreHistory].slice(0, 10);
      setScoreHistory(updatedHistory);
      localStorage.setItem('home-insurance-score-history', JSON.stringify(updatedHistory));
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setScoreHistory([]);
    localStorage.removeItem('home-insurance-score-history');
  };

  const loadFromHistory = (item: ScoreHistory) => {
    setFormData(item.formData);
    setPrediction(item.prediction);
    setShowHistory(false);
  };

  const shareScore = async () => {
    if (!shareEmail || !prediction) return;

    setShareSending(true);
    try {
      const response = await fetch('http://localhost:8000/api/home-insurance/score/share', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: shareEmail,
          formData: formData,
          prediction: prediction,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to share score');
      }

      alert(`Score shared successfully with ${shareEmail}!`);
      setShowShareModal(false);
      setShareEmail('');
    } catch (err: any) {
      alert(`Failed to share score: ${err.message}`);
    } finally {
      setShareSending(false);
    }
  };

  const exportScoreToPDF = () => {
    if (!prediction) return;

    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Home Insurance Lead Score Report</title>
        <style>
          body { font-family: Arial, sans-serif; padding: 40px; }
          h1 { color: #059669; border-bottom: 3px solid #059669; padding-bottom: 10px; }
          .score-box { display: inline-block; margin: 20px; padding: 20px; border: 2px solid #cbd5e1; border-radius: 8px; }
          .score-label { font-weight: bold; color: #64748b; font-size: 14px; }
          .score-value { font-size: 36px; color: #059669; font-weight: bold; margin-top: 10px; }
          table { width: 100%; border-collapse: collapse; margin-top: 20px; }
          th, td { border: 1px solid #cbd5e1; padding: 12px; text-align: left; }
          th { background-color: #f1f5f9; font-weight: bold; }
        </style>
      </head>
      <body>
        <h1>🏠 Home Insurance Lead Score Report</h1>
        <p><strong>Generated:</strong> ${new Date().toLocaleString()}</p>

        <h2>Lead Conversion Score</h2>
        <div class="score-box" style="text-align: center; padding: 40px;">
          <div class="score-label">Conversion Score</div>
          <div class="score-value" style="font-size: 72px;">${prediction.ensemble_score?.toFixed(1) || prediction.score?.toFixed(1) || 'N/A'}</div>
          <div style="margin-top: 20px; font-size: 18px; color: #64748b;">
            ${(prediction.ensemble_score || prediction.score || 0) >= 75 ? '🟢 High Quality Lead' :
              (prediction.ensemble_score || prediction.score || 0) >= 50 ? '🟡 Medium Quality Lead' :
              '🔴 Low Quality Lead'}
          </div>
        </div>

        <h2>Lead Information</h2>
        <table>
          <tr><th>Field</th><th>Value</th></tr>
          <tr><td>Property Age</td><td>${formData.property_age} years</td></tr>
          <tr><td>Property Type</td><td>${formData.property_type}</td></tr>
          <tr><td>Square Footage</td><td>${formData.square_footage} sq ft</td></tr>
          <tr><td>Construction Type</td><td>${formData.construction_type}</td></tr>
          <tr><td>Roof Age</td><td>${formData.roof_age} years</td></tr>
          <tr><td>Location Risk</td><td>${formData.location_risk}</td></tr>
          <tr><td>Security Features</td><td>${formData.security_features}</td></tr>
          <tr><td>Previous Claims</td><td>${formData.previous_claims}</td></tr>
          <tr><td>Coverage Amount</td><td>$${formData.coverage_amount}</td></tr>
          <tr><td>Deductible</td><td>$${formData.deductible}</td></tr>
        </table>

        <div style="margin-top: 40px; font-size: 12px; color: #64748b;">
          <p>LEGEAI - Lead Generation AI System | Home Insurance Lead Scoring</p>
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

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-emerald-50 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-2">
                🏠 Home Insurance Lead Scoring
              </h1>
              <p className="text-slate-600 mt-2">Score new home insurance leads using our AI-powered scoring system</p>
            </div>
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded-lg transition-colors flex items-center gap-2"
            >
              📜 {showHistory ? 'Hide' : 'Show'} History ({scoreHistory.length})
            </button>
          </div>
        </div>

        {/* Score History */}
        {showHistory && scoreHistory.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-slate-900">📜 Recent Scores</h2>
              <button
                onClick={clearHistory}
                className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded transition-colors"
              >
                Clear History
              </button>
            </div>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {scoreHistory.map((item) => (
                <div
                  key={item.id}
                  className="border-2 border-slate-200 rounded-lg p-4 hover:border-emerald-400 transition-colors cursor-pointer"
                  onClick={() => loadFromHistory(item)}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <p className="text-sm text-slate-600">
                        {new Date(item.timestamp).toLocaleString()}
                      </p>
                      <p className="text-xs text-slate-500 mt-1">
                        {item.formData.property_type} • {item.formData.square_footage} sqft • {item.formData.construction_type}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium text-slate-600">Conversion Score</p>
                      <p className="text-2xl font-bold text-emerald-700">
                        {item.prediction.ensemble_score?.toFixed(1) || item.prediction.score?.toFixed(1) || 'N/A'}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Form */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <form onSubmit={handleSubmit}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Property Age */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Property Age (years)</label>
                <input
                  type="number"
                  name="property_age"
                  value={formData.property_age}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-emerald-500 focus:outline-none"
                  placeholder="Enter property age"
                  required
                />
              </div>

              {/* Property Type */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Property Type</label>
                <select
                  name="property_type"
                  value={formData.property_type}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-emerald-500 focus:outline-none"
                  required
                >
                  <option value="">Select property type</option>
                  <option value="Single Family">Single Family</option>
                  <option value="Condo">Condo</option>
                  <option value="Townhouse">Townhouse</option>
                  <option value="Multi-Family">Multi-Family</option>
                  <option value="Mobile Home">Mobile Home</option>
                </select>
              </div>

              {/* Square Footage */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Square Footage</label>
                <input
                  type="number"
                  name="square_footage"
                  value={formData.square_footage}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-emerald-500 focus:outline-none"
                  placeholder="Total sq ft"
                  required
                />
              </div>

              {/* Number of Bedrooms */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Number of Bedrooms</label>
                <input
                  type="number"
                  name="num_bedrooms"
                  value={formData.num_bedrooms}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-emerald-500 focus:outline-none"
                  placeholder="Bedrooms"
                  min="1"
                  required
                />
              </div>

              {/* Number of Bathrooms */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Number of Bathrooms</label>
                <input
                  type="number"
                  name="num_bathrooms"
                  value={formData.num_bathrooms}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-emerald-500 focus:outline-none"
                  placeholder="Bathrooms"
                  min="1"
                  step="0.5"
                  required
                />
              </div>

              {/* Construction Type */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Construction Type</label>
                <select
                  name="construction_type"
                  value={formData.construction_type}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-emerald-500 focus:outline-none"
                  required
                >
                  <option value="">Select construction</option>
                  <option value="Brick">Brick</option>
                  <option value="Wood Frame">Wood Frame</option>
                  <option value="Concrete">Concrete</option>
                  <option value="Steel">Steel</option>
                  <option value="Mixed">Mixed</option>
                </select>
              </div>

              {/* Roof Age */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Roof Age (years)</label>
                <input
                  type="number"
                  name="roof_age"
                  value={formData.roof_age}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-emerald-500 focus:outline-none"
                  placeholder="Years since roof replacement"
                  min="0"
                  required
                />
              </div>

              {/* Location Risk */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Location Risk</label>
                <select
                  name="location_risk"
                  value={formData.location_risk}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-emerald-500 focus:outline-none"
                  required
                >
                  <option value="">Select risk level</option>
                  <option value="Low">Low Risk</option>
                  <option value="Medium">Medium Risk</option>
                  <option value="High">High Risk</option>
                </select>
              </div>

              {/* Security Features */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Security Features</label>
                <select
                  name="security_features"
                  value={formData.security_features}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-emerald-500 focus:outline-none"
                  required
                >
                  <option value="">Select security level</option>
                  <option value="None">None</option>
                  <option value="Basic">Basic (Locks)</option>
                  <option value="Alarm">Alarm System</option>
                  <option value="Full">Full (Alarm + Cameras)</option>
                </select>
              </div>

              {/* Previous Claims */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Previous Claims</label>
                <input
                  type="number"
                  name="previous_claims"
                  value={formData.previous_claims}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-emerald-500 focus:outline-none"
                  placeholder="Number of claims"
                  min="0"
                  required
                />
              </div>
            </div>

            {/* Submit Button */}
            <div className="mt-6">
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3 px-6 rounded-lg transition-colors disabled:bg-slate-400"
              >
                {loading ? 'Scoring...' : '🎯 Score Lead'}
              </button>
            </div>
          </form>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4 mb-6">
            <p className="text-red-700 font-medium">❌ Error: {error}</p>
          </div>
        )}

        {/* Prediction Results */}
        {prediction && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold text-slate-900">📊 Prediction Results</h2>
              <div className="flex gap-2">
                <button
                  onClick={exportScoreToPDF}
                  className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded transition-colors"
                >
                  📄 Export PDF
                </button>
                <button
                  onClick={() => setShowShareModal(true)}
                  className="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded transition-colors"
                >
                  ✉️ Share
                </button>
              </div>
            </div>

            {/* Conversion Score */}
            <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 border-2 border-emerald-200 rounded-lg p-6 mb-6">
              <div className="text-center">
                <p className="text-lg font-medium text-slate-600 mb-2">Lead Conversion Score</p>
                <p className="text-6xl font-bold text-emerald-700 mb-3">
                  {prediction.ensemble_score?.toFixed(1) || prediction.score?.toFixed(1) || 'N/A'}
                </p>
                <div className="flex items-center justify-center gap-2">
                  <div className="w-full max-w-md bg-slate-200 rounded-full h-3">
                    <div
                      className="bg-emerald-600 h-3 rounded-full transition-all duration-500"
                      style={{ width: `${(prediction.ensemble_score || prediction.score || 0)}%` }}
                    ></div>
                  </div>
                </div>
                <p className="text-sm text-slate-600 mt-3">
                  {(prediction.ensemble_score || prediction.score || 0) >= 75 ? '🟢 High Quality Lead' :
                   (prediction.ensemble_score || prediction.score || 0) >= 50 ? '🟡 Medium Quality Lead' :
                   '🔴 Low Quality Lead'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Share Modal */}
        {showShareModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
              <h2 className="text-xl font-bold text-slate-900 mb-4">📧 Share Score Report</h2>
              <p className="text-sm text-slate-600 mb-4">
                Enter the email address where you want to share this lead score report.
              </p>
              <input
                type="email"
                value={shareEmail}
                onChange={(e) => setShareEmail(e.target.value)}
                placeholder="recipient@example.com"
                className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-purple-500 focus:outline-none mb-4"
              />
              <div className="flex gap-2">
                <button
                  onClick={shareScore}
                  disabled={shareSending || !shareEmail}
                  className="flex-1 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded-lg transition-colors disabled:bg-slate-400"
                >
                  {shareSending ? 'Sending...' : 'Send Email'}
                </button>
                <button
                  onClick={() => {
                    setShowShareModal(false);
                    setShareEmail('');
                  }}
                  className="flex-1 px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-700 font-medium rounded-lg transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

