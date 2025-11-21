// Utility functions for exporting and sharing data
import * as XLSX from 'xlsx';

export const exportToPDF = (htmlContent: string, filename: string) => {
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

export const generateAnalyticsPDFContent = (
  title: string,
  icon: string,
  analytics: any,
  dateRange: string,
  scoreFilter: string,
  distributions: { [key: string]: any }
) => {
  return `
    <!DOCTYPE html>
    <html>
    <head>
      <title>${title} Analytics Report</title>
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
      <h1>${icon} ${title} Analytics Report</h1>
      <p><strong>Generated:</strong> ${new Date().toLocaleString()}</p>
      <p><strong>Date Range:</strong> ${dateRange === 'all' ? 'All Time' : dateRange}</p>
      <p><strong>Score Filter:</strong> ${scoreFilter === 'all' ? 'All Scores' : scoreFilter}</p>
      
      <h2>Key Metrics</h2>
      <div class="metric">
        <div class="metric-label">Total Leads Scored</div>
        <div class="metric-value">${analytics.total_leads || 'N/A'}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Avg Conversion Score</div>
        <div class="metric-value">${analytics.avg_score?.toFixed(2) || 'N/A'}</div>
      </div>
      <div class="metric">
        <div class="metric-label">High Quality Leads</div>
        <div class="metric-value">${analytics.high_quality_leads || 'N/A'}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value">${analytics.model_accuracy || 'N/A'}</div>
      </div>
      
      ${Object.entries(distributions).map(([title, data]) => `
        <h2>${title}</h2>
        <table>
          <tr><th>Category</th><th>Percentage</th></tr>
          ${Object.entries(data as any).map(([key, value]) => `
            <tr><td>${key}</td><td>${value}</td></tr>
          `).join('')}
        </table>
      `).join('')}
      
      <div class="footer">
        <p>LEGEAI - Lead Generation AI System | ${title} Analytics</p>
      </div>
    </body>
    </html>
  `;
};

export const generateScorePDFContent = (
  title: string,
  icon: string,
  formData: any,
  prediction: any,
  fields: { label: string; key: string; format?: (val: any) => string }[]
) => {
  return `
    <!DOCTYPE html>
    <html>
    <head>
      <title>${title} Lead Score Report</title>
      <style>
        body { font-family: Arial, sans-serif; padding: 40px; }
        h1 { color: #1e40af; border-bottom: 3px solid #1e40af; padding-bottom: 10px; }
        .score-box { display: inline-block; margin: 20px; padding: 20px; border: 2px solid #cbd5e1; border-radius: 8px; }
        .score-label { font-weight: bold; color: #64748b; font-size: 14px; }
        .score-value { font-size: 36px; color: #1e40af; font-weight: bold; margin-top: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #cbd5e1; padding: 12px; text-align: left; }
        th { background-color: #f1f5f9; font-weight: bold; }
      </style>
    </head>
    <body>
      <h1>${icon} ${title} Lead Score Report</h1>
      <p><strong>Generated:</strong> ${new Date().toLocaleString()}</p>
      
      <h2>Prediction Scores</h2>
      <div class="score-box">
        <div class="score-label">Ensemble Score</div>
        <div class="score-value">${prediction.ensemble_score?.toFixed(2) || 'N/A'}</div>
      </div>
      <div class="score-box">
        <div class="score-label">XGBoost Score</div>
        <div class="score-value">${prediction.xgboost_score?.toFixed(2) || 'N/A'}</div>
      </div>
      <div class="score-box">
        <div class="score-label">Deep Learning Score</div>
        <div class="score-value">${prediction.deep_learning_score?.toFixed(2) || 'N/A'}</div>
      </div>
      
      <h2>Lead Information</h2>
      <table>
        <tr><th>Field</th><th>Value</th></tr>
        ${fields.map(field => `
          <tr>
            <td>${field.label}</td>
            <td>${field.format ? field.format(formData[field.key]) : formData[field.key]}</td>
          </tr>
        `).join('')}
      </table>
      
      <div style="margin-top: 40px; font-size: 12px; color: #64748b;">
        <p>LEGEAI - Lead Generation AI System | ${title} Lead Scoring</p>
      </div>
    </body>
    </html>
  `;
};

// Export to Excel
export const exportToExcel = (data: any[], filename: string, sheetName: string = 'Sheet1') => {
  const worksheet = XLSX.utils.json_to_sheet(data);
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, sheetName);
  XLSX.writeFile(workbook, `${filename}.xlsx`);
};

// Export Analytics to Excel
export const exportAnalyticsToExcel = (analytics: any, insuranceType: string) => {
  const metricsData = [
    { Metric: 'Total Leads Scored', Value: analytics.total_leads || 'N/A' },
    { Metric: 'Average Conversion Score', Value: analytics.avg_score?.toFixed(2) || 'N/A' },
    { Metric: 'High Quality Leads', Value: analytics.high_quality_leads || 'N/A' },
    { Metric: 'Model Accuracy', Value: analytics.model_accuracy || 'N/A' },
  ];

  const scoreDistribution = [
    { Category: 'High (0.75 - 1.0)', Percentage: '28.3%' },
    { Category: 'Medium (0.50 - 0.75)', Percentage: '45.6%' },
    { Category: 'Low (0.0 - 0.50)', Percentage: '26.1%' },
  ];

  const workbook = XLSX.utils.book_new();

  // Add metrics sheet
  const metricsSheet = XLSX.utils.json_to_sheet(metricsData);
  XLSX.utils.book_append_sheet(workbook, metricsSheet, 'Key Metrics');

  // Add distribution sheet
  const distributionSheet = XLSX.utils.json_to_sheet(scoreDistribution);
  XLSX.utils.book_append_sheet(workbook, distributionSheet, 'Score Distribution');

  const filename = `${insuranceType}-analytics-${new Date().toISOString().split('T')[0]}.xlsx`;
  XLSX.writeFile(workbook, filename);
};

// Export to XML
export const exportToXML = (data: any, rootElement: string, filename: string) => {
  const convertToXML = (obj: any, indent: string = ''): string => {
    let xml = '';
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        const value = obj[key];
        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
          xml += `${indent}<${key}>\n${convertToXML(value, indent + '  ')}${indent}</${key}>\n`;
        } else if (Array.isArray(value)) {
          value.forEach(item => {
            xml += `${indent}<${key}>\n${convertToXML(item, indent + '  ')}${indent}</${key}>\n`;
          });
        } else {
          xml += `${indent}<${key}>${value}</${key}>\n`;
        }
      }
    }
    return xml;
  };

  const xmlContent = `<?xml version="1.0" encoding="UTF-8"?>\n<${rootElement}>\n${convertToXML(data, '  ')}</${rootElement}>`;

  const blob = new Blob([xmlContent], { type: 'application/xml' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${filename}.xml`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
};

// Export Analytics to XML
export const exportAnalyticsToXML = (analytics: any, insuranceType: string) => {
  const xmlData = {
    analytics: {
      metadata: {
        insuranceType: insuranceType,
        generatedAt: new Date().toISOString(),
      },
      keyMetrics: {
        totalLeads: analytics.total_leads || 'N/A',
        avgScore: analytics.avg_score?.toFixed(2) || 'N/A',
        highQualityLeads: analytics.high_quality_leads || 'N/A',
        modelAccuracy: analytics.model_accuracy || 'N/A',
      },
      scoreDistribution: {
        high: '28.3%',
        medium: '45.6%',
        low: '26.1%',
      },
    },
  };

  const filename = `${insuranceType}-analytics-${new Date().toISOString().split('T')[0]}`;
  exportToXML(xmlData, 'insuranceAnalytics', filename);
};

// Export Comparison to Excel
export const exportComparisonToExcel = (insuranceType: string, metrics: any) => {
  const comparisonData = [
    { Model: 'XGBoost', 'R² Score': metrics.xgboost.r2, MAE: metrics.xgboost.mae, RMSE: metrics.xgboost.rmse },
    { Model: 'Deep Learning', 'R² Score': metrics.deepLearning.r2, MAE: metrics.deepLearning.mae, RMSE: metrics.deepLearning.rmse },
    { Model: 'Ensemble', 'R² Score': metrics.ensemble.r2, MAE: metrics.ensemble.mae, RMSE: metrics.ensemble.rmse },
  ];

  const filename = `${insuranceType}-model-comparison-${new Date().toISOString().split('T')[0]}`;
  exportToExcel(comparisonData, filename, 'Model Comparison');
};

