"use client";
import React from "react";

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
  const defaultMetrics: ScoreMetric[] = [
    { label: "Engagement", value: 85, max: 100, color: "bg-blue-500" },
    { label: "Budget Fit", value: 72, max: 100, color: "bg-green-500" },
    { label: "Timeline", value: 68, max: 100, color: "bg-amber-500" },
    { label: "Authority", value: 90, max: 100, color: "bg-purple-500" },
    { label: "Need", value: 78, max: 100, color: "bg-red-500" },
  ];

  const scoreMetrics = metrics || defaultMetrics;
  const avgScore = overallScore || Math.round(scoreMetrics.reduce((sum, m) => sum + m.value, 0) / scoreMetrics.length);

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
        <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition-colors">
          View Detailed Analysis
        </button>
      </div>
    </div>
  );
}

