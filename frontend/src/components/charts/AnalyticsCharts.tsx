'use client';

import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';

interface AnalyticsChartsProps {
  scoreDistribution?: any[];
  typeDistribution?: any[];
  trendData?: any[];
  colorScheme?: string;
}

export default function AnalyticsCharts({
  scoreDistribution,
  typeDistribution,
  trendData,
  colorScheme = 'blue',
}: AnalyticsChartsProps) {
  // Color schemes for different insurance types
  const colors = {
    blue: ['#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe'],
    emerald: ['#10b981', '#34d399', '#6ee7b7', '#a7f3d0'],
    purple: ['#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe'],
  };

  const chartColors = colors[colorScheme as keyof typeof colors] || colors.blue;

  // Default data if not provided
  const defaultScoreDistribution = [
    { name: 'High (0.75-1.0)', value: 28.3, count: 342 },
    { name: 'Medium (0.50-0.75)', value: 45.6, count: 551 },
    { name: 'Low (0.0-0.50)', value: 26.1, count: 315 },
  ];

  const defaultTypeDistribution = [
    { name: 'Type A', value: 38.2 },
    { name: 'Type B', value: 29.4 },
    { name: 'Type C', value: 18.7 },
    { name: 'Type D', value: 13.7 },
  ];

  const defaultTrendData = [
    { month: 'Jan', leads: 95, avgScore: 0.65 },
    { month: 'Feb', leads: 108, avgScore: 0.68 },
    { month: 'Mar', leads: 122, avgScore: 0.71 },
    { month: 'Apr', leads: 135, avgScore: 0.69 },
    { month: 'May', leads: 148, avgScore: 0.73 },
    { month: 'Jun', leads: 162, avgScore: 0.75 },
  ];

  const scoreData = scoreDistribution || defaultScoreDistribution;
  const typeData = typeDistribution || defaultTypeDistribution;
  const trendChartData = trendData || defaultTrendData;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Score Distribution - Pie Chart */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-bold text-slate-900 mb-4">📊 Score Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={scoreData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {scoreData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={chartColors[index % chartColors.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Type Distribution - Bar Chart */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-bold text-slate-900 mb-4">📈 Type Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={typeData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="value" fill={chartColors[0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Trend Analysis - Line Chart */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-bold text-slate-900 mb-4">📉 Lead Trend (6 Months)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trendChartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="leads" stroke={chartColors[0]} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Average Score Trend - Area Chart */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-bold text-slate-900 mb-4">📊 Avg Score Trend</h3>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={trendChartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Area type="monotone" dataKey="avgScore" stroke={chartColors[1]} fill={chartColors[1]} fillOpacity={0.6} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

