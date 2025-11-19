"use client";
import React from "react";

interface Metric {
  label: string;
  value: string | number;
  change: number;
  icon: string;
  color: string;
}

interface KeyMetricsProps {
  metrics?: Metric[];
}

export default function KeyMetrics({ metrics }: KeyMetricsProps) {
  const defaultMetrics: Metric[] = [
    {
      label: "Total Leads",
      value: 1247,
      change: 18.7,
      icon: "👥",
      color: "from-blue-500 to-blue-600",
    },
    {
      label: "This Month",
      value: 342,
      change: 15.4,
      icon: "📈",
      color: "from-green-500 to-green-600",
    },
    {
      label: "Conversion Rate",
      value: "24.8%",
      change: 5.3,
      icon: "🎯",
      color: "from-purple-500 to-purple-600",
    },
    {
      label: "Avg Deal Value",
      value: "$3,850",
      change: 8.9,
      icon: "💰",
      color: "from-amber-500 to-amber-600",
    },
  ];

  const keyMetrics = metrics || defaultMetrics;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {keyMetrics.map((metric, idx) => (
        <div
          key={idx}
          className={`bg-gradient-to-br ${metric.color} rounded-lg p-6 text-white shadow-md hover:shadow-lg transition-shadow`}
        >
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-sm opacity-90 font-medium">{metric.label}</p>
              <p className="text-3xl font-bold mt-2">{metric.value}</p>
            </div>
            <span className="text-3xl">{metric.icon}</span>
          </div>

          <div className="flex items-center gap-1 text-sm">
            <span className={metric.change >= 0 ? "text-green-200" : "text-red-200"}>
              {metric.change >= 0 ? "↑" : "↓"}
            </span>
            <span className="opacity-90">
              {Math.abs(metric.change)}% vs last month
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

