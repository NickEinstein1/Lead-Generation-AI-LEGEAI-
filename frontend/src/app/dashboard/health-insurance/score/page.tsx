'use client';

import { useState, useEffect } from 'react';

interface ScoreHistory {
  id: string;
  timestamp: string;
  formData: any;
  prediction: any;
}

export default function HealthInsuranceScorePage() {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    bmi: '',
    smoker: '',
    pre_existing_conditions: '',
    family_history: '',
    plan_type: '',
    coverage_level: '',
    deductible: '',
    annual_income: '',
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
    const savedHistory = localStorage.getItem('health-insurance-score-history');
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
      const response = await fetch('http://localhost:8000/api/health-insurance/score', {
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
      localStorage.setItem('health-insurance-score-history', JSON.stringify(updatedHistory));
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setScoreHistory([]);
    localStorage.removeItem('health-insurance-score-history');
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
      const response = await fetch('http://localhost:8000/api/health-insurance/score/share', {
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
        <title>Health Insurance Lead Score Report</title>
        <style>
          body { font-family: Arial, sans-serif; padding: 40px; }
          h1 { color: #7c3aed; border-bottom: 3px solid #7c3aed; padding-bottom: 10px; }
          .score-box { display: inline-block; margin: 20px; padding: 20px; border: 2px solid #cbd5e1; border-radius: 8px; }
          .score-label { font-weight: bold; color: #64748b; font-size: 14px; }
          .score-value { font-size: 36px; color: #7c3aed; font-weight: bold; margin-top: 10px; }
          table { width: 100%; border-collapse: collapse; margin-top: 20px; }
          th, td { border: 1px solid #cbd5e1; padding: 12px; text-align: left; }
          th { background-color: #f1f5f9; font-weight: bold; }
        </style>
      </head>
      <body>
        <h1>⚕️ Health Insurance Lead Score Report</h1>
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
          <tr><td>Age</td><td>${formData.age}</td></tr>
          <tr><td>Gender</td><td>${formData.gender}</td></tr>
          <tr><td>BMI</td><td>${formData.bmi}</td></tr>
          <tr><td>Smoker</td><td>${formData.smoker}</td></tr>
          <tr><td>Pre-existing Conditions</td><td>${formData.pre_existing_conditions}</td></tr>
          <tr><td>Family History</td><td>${formData.family_history}</td></tr>
          <tr><td>Plan Type</td><td>${formData.plan_type}</td></tr>
          <tr><td>Coverage Level</td><td>${formData.coverage_level}</td></tr>
          <tr><td>Deductible</td><td>$${formData.deductible}</td></tr>
          <tr><td>Annual Income</td><td>$${formData.annual_income}</td></tr>
        </table>

        <div style="margin-top: 40px; font-size: 12px; color: #64748b;">
          <p>LEGEAI - Lead Generation AI System | Health Insurance Lead Scoring</p>
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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-purple-50 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-2">
                ⚕️ Health Insurance Lead Scoring
              </h1>
              <p className="text-slate-600 mt-2">Score new health insurance leads using our AI-powered scoring system</p>
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
                  className="border-2 border-slate-200 rounded-lg p-4 hover:border-purple-400 transition-colors cursor-pointer"
                  onClick={() => loadFromHistory(item)}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <p className="text-sm text-slate-600">
                        {new Date(item.timestamp).toLocaleString()}
                      </p>
                      <p className="text-xs text-slate-500 mt-1">
                        Age {item.formData.age} • {item.formData.plan_type} • {item.formData.coverage_level}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium text-slate-600">Conversion Score</p>
                      <p className="text-2xl font-bold text-purple-700">
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
              {/* Age */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Age</label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-purple-500 focus:outline-none"
                  placeholder="Enter age"
                  required
                />
              </div>

              {/* Gender */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Gender</label>
                <select
                  name="gender"
                  value={formData.gender}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-purple-500 focus:outline-none"
                  required
                >
                  <option value="">Select gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>

              {/* BMI */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">BMI (Body Mass Index)</label>
                <input
                  type="number"
                  name="bmi"
                  value={formData.bmi}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-purple-500 focus:outline-none"
                  placeholder="18.5 - 40"
                  step="0.1"
                  required
                />
              </div>

              {/* Smoker */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Smoker Status</label>
                <select
                  name="smoker"
                  value={formData.smoker}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-purple-500 focus:outline-none"
                  required
                >
                  <option value="">Select status</option>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              {/* Pre-existing Conditions */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Pre-existing Conditions</label>
                <select
                  name="pre_existing_conditions"
                  value={formData.pre_existing_conditions}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-purple-500 focus:outline-none"
                  required
                >
                  <option value="">Select</option>
                  <option value="None">None</option>
                  <option value="Minor">Minor (1-2 conditions)</option>
                  <option value="Moderate">Moderate (3-4 conditions)</option>
                  <option value="Severe">Severe (5+ conditions)</option>
                </select>
              </div>

              {/* Family History */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Family History</label>
                <select
                  name="family_history"
                  value={formData.family_history}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-purple-500 focus:outline-none"
                  required
                >
                  <option value="">Select</option>
                  <option value="None">No significant history</option>
                  <option value="Minor">Minor conditions</option>
                  <option value="Moderate">Moderate risk factors</option>
                  <option value="High">High risk factors</option>
                </select>
              </div>

              {/* Plan Type */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Plan Type</label>
                <select
                  name="plan_type"
                  value={formData.plan_type}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-purple-500 focus:outline-none"
                  required
                >
                  <option value="">Select plan type</option>
                  <option value="HMO">HMO</option>
                  <option value="PPO">PPO</option>
                  <option value="EPO">EPO</option>
                  <option value="POS">POS</option>
                  <option value="HDHP">HDHP</option>
                </select>
              </div>

              {/* Coverage Level */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Coverage Level</label>
                <select
                  name="coverage_level"
                  value={formData.coverage_level}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-purple-500 focus:outline-none"
                  required
                >
                  <option value="">Select coverage</option>
                  <option value="Bronze">Bronze</option>
                  <option value="Silver">Silver</option>
                  <option value="Gold">Gold</option>
                  <option value="Platinum">Platinum</option>
                </select>
              </div>

              {/* Deductible */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Deductible ($)</label>
                <select
                  name="deductible"
                  value={formData.deductible}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-purple-500 focus:outline-none"
                  required
                >
                  <option value="">Select deductible</option>
                  <option value="500">$500</option>
                  <option value="1000">$1,000</option>
                  <option value="2500">$2,500</option>
                  <option value="5000">$5,000</option>
                  <option value="7500">$7,500</option>
                </select>
              </div>

              {/* Annual Income */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Annual Income ($)</label>
                <input
                  type="number"
                  name="annual_income"
                  value={formData.annual_income}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-purple-500 focus:outline-none"
                  placeholder="Annual income"
                  required
                />
              </div>
            </div>

            {/* Submit Button */}
            <div className="mt-6">
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-6 rounded-lg transition-colors disabled:bg-slate-400"
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
            <div className="bg-gradient-to-br from-purple-50 to-purple-100 border-2 border-purple-200 rounded-lg p-6 mb-6">
              <div className="text-center">
                <p className="text-lg font-medium text-slate-600 mb-2">Lead Conversion Score</p>
                <p className="text-6xl font-bold text-purple-700 mb-3">
                  {prediction.ensemble_score?.toFixed(1) || prediction.score?.toFixed(1) || 'N/A'}
                </p>
                <div className="flex items-center justify-center gap-2">
                  <div className="w-full max-w-md bg-slate-200 rounded-full h-3">
                    <div
                      className="bg-purple-600 h-3 rounded-full transition-all duration-500"
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

