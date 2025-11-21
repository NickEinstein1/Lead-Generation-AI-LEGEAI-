"use client";
import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

interface ScoreMetric {
  label: string;
  value: number;
  max: number;
  color: string;
}

interface LeadScoringProps {
  metrics?: ScoreMetric[];
  overallScore?: number;
}

export default function LeadScoring({ metrics, overallScore }: LeadScoringProps) {
  const router = useRouter();
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [dynamicMetrics, setDynamicMetrics] = useState<ScoreMetric[]>([]);

  // Generate dynamic values on component mount
  useEffect(() => {
    const generateDynamicMetrics = (): ScoreMetric[] => {
      // Generate random but realistic scores (between 65-95)
      const randomScore = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;

      return [
        { label: "Engagement", value: randomScore(75, 95), max: 100, color: "bg-blue-500" },
        { label: "Budget Fit", value: randomScore(70, 92), max: 100, color: "bg-green-500" },
        { label: "Timeline", value: randomScore(65, 88), max: 100, color: "bg-amber-500" },
        { label: "Authority", value: randomScore(72, 94), max: 100, color: "bg-purple-500" },
        { label: "Need", value: randomScore(78, 96), max: 100, color: "bg-red-500" },
      ];
    };

    setDynamicMetrics(generateDynamicMetrics());
  }, []); // Empty dependency array means this runs once on mount

  const scoreMetrics = metrics || dynamicMetrics;
  const avgScore = overallScore || (scoreMetrics.length > 0 ? Math.round(scoreMetrics.reduce((sum, m) => sum + m.value, 0) / scoreMetrics.length) : 0);

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-600";
    if (score >= 60) return "text-amber-600";
    return "text-red-600";
  };

  const getScoreBgColor = (score: number) => {
    if (score >= 80) return "bg-green-50";
    if (score >= 60) return "bg-amber-50";
    return "bg-red-50";
  };

  return (
    <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
      <h2 className="text-xl font-bold text-slate-900 mb-6">Lead Scoring</h2>

      <div className={`${getScoreBgColor(avgScore)} rounded-lg p-6 mb-6 text-center`}>
        <div className="text-sm text-slate-600 font-medium mb-2">Overall Score</div>
        <div className={`text-5xl font-bold ${getScoreColor(avgScore)}`}>{avgScore}</div>
        <div className="text-xs text-slate-600 font-medium mt-2">
          {avgScore >= 80 ? "🔥 Hot Lead" : avgScore >= 60 ? "⚡ Warm Lead" : "❄️ Cold Lead"}
        </div>
      </div>

      <div className="space-y-4">
        {scoreMetrics.map((metric, idx) => (
          <div key={idx}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-slate-700">{metric.label}</span>
              <span className="text-sm font-bold text-slate-900">{metric.value}/{metric.max}</span>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
              <div
                className={`h-full ${metric.color} transition-all duration-300`}
                style={{ width: `${(metric.value / metric.max) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 pt-6 border-t-2 border-blue-100">
        <button
          onClick={() => setShowAnalysis(true)}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition-colors active:scale-95"
        >
          View Detailed Analysis
        </button>
      </div>

      {/* Detailed Analysis Modal */}
      {showAnalysis && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowAnalysis(false)}>
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 shadow-xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-2xl font-bold text-slate-900 mb-4">Detailed Lead Analysis</h3>

            <div className="space-y-6">
              {/* Overall Score */}
              <div className={`${getScoreBgColor(avgScore)} rounded-lg p-6 text-center`}>
                <div className="text-sm text-slate-600 font-medium mb-2">Overall Lead Score</div>
                <div className={`text-6xl font-bold ${getScoreColor(avgScore)}`}>{avgScore}</div>
                <div className="text-sm text-slate-600 font-medium mt-2">
                  {avgScore >= 80 ? "🔥 Hot Lead - High Priority" : avgScore >= 60 ? "⚡ Warm Lead - Good Potential" : "❄️ Cold Lead - Needs Nurturing"}
                </div>
              </div>

              {/* Detailed Metrics */}
              <div>
                <h4 className="font-bold text-slate-900 mb-4">Score Breakdown</h4>
                <div className="space-y-4">
                  {scoreMetrics.map((metric, idx) => (
                    <div key={idx} className="bg-slate-50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-slate-900">{metric.label}</span>
                        <span className="text-lg font-bold text-slate-900">{metric.value}/{metric.max}</span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden mb-2">
                        <div
                          className={`h-full ${metric.color} transition-all duration-300`}
                          style={{ width: `${(metric.value / metric.max) * 100}%` }}
                        />
                      </div>
                      <p className="text-xs text-slate-600">
                        {metric.label === "Engagement" && "Based on email opens, clicks, and website visits"}
                        {metric.label === "Budget Fit" && "Alignment between lead budget and product pricing"}
                        {metric.label === "Timeline" && "Urgency and readiness to purchase"}
                        {metric.label === "Authority" && "Decision-making power and influence"}
                        {metric.label === "Need" && "Product-market fit and pain point severity"}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Recommendations */}
              <div className="bg-blue-50 rounded-lg p-4">
                <h4 className="font-bold text-slate-900 mb-2">📋 Recommended Actions</h4>
                <ul className="space-y-2 text-sm text-slate-700">
                  {avgScore >= 80 && (
                    <>
                      <li>✅ Schedule a call within 24 hours</li>
                      <li>✅ Send personalized proposal</li>
                      <li>✅ Assign to senior sales rep</li>
                    </>
                  )}
                  {avgScore >= 60 && avgScore < 80 && (
                    <>
                      <li>⚡ Send follow-up email with case studies</li>
                      <li>⚡ Schedule discovery call this week</li>
                      <li>⚡ Add to nurture campaign</li>
                    </>
                  )}
                  {avgScore < 60 && (
                    <>
                      <li>❄️ Add to long-term nurture sequence</li>
                      <li>❄️ Send educational content</li>
                      <li>❄️ Re-qualify in 30 days</li>
                    </>
                  )}
                </ul>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowAnalysis(false)}
                className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Close
              </button>
              <button
                onClick={() => {
                  setShowAnalysis(false);
                  router.push("/dashboard/analytics");
                }}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all"
              >
                View Full Analytics
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

