/**
 * API Client for Lead Generation AI
 * Handles all API calls to the backend
 */

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/v1";

interface FetchOptions extends RequestInit {
  headers?: Record<string, string>;
}

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch<T>(endpoint: string, options: FetchOptions = {}): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  
  const defaultHeaders: Record<string, string> = {
    "Content-Type": "application/json",
  };

  // Add auth headers if available
  if (typeof window !== "undefined") {
    const session = localStorage.getItem("session");
    if (session) {
      try {
        const sessionData = JSON.parse(session);
        // Use X-Session-ID header for authentication
        if (sessionData.session_id) {
          defaultHeaders["X-Session-ID"] = sessionData.session_id;
        }
        // Fallback to API key for development
        if (!sessionData.session_id) {
          defaultHeaders["X-API-Key"] = "dev-api-key-12345";
        }
      } catch (e) {
        console.error("Failed to parse session", e);
        // Use dev API key as fallback
        defaultHeaders["X-API-Key"] = "dev-api-key-12345";
      }
    } else {
      // No session - use dev API key for development
      defaultHeaders["X-API-Key"] = "dev-api-key-12345";
    }
  }

  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error: ${response.status} - ${errorText}`);
  }

  return response.json();
}

// Dashboard APIs
export async function getDashboardOverview() {
  return apiFetch("/dashboard/overview");
}

export async function getLeadTimeseries(days: number = 14) {
  return apiFetch(`/dashboard/timeseries/leads?days=${days}`);
}

export async function getScoreTimeseries(days: number = 14) {
  return apiFetch(`/dashboard/timeseries/scores?days=${days}`);
}

// Lead APIs
export async function listLeads() {
  return apiFetch("/leads");
}

export async function getLead(id: string) {
  return apiFetch(`/leads/${id}`);
}

export async function createLead(data: any) {
  return apiFetch("/leads", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function scoreLead(id: string, data: any) {
  return apiFetch(`/leads/${id}/score`, {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function routeLead(id: string, data: any) {
  return apiFetch(`/leads/${id}/route`, {
    method: "POST",
    body: JSON.stringify(data),
  });
}

// Document APIs
export async function listDocumentsForLead(leadId: string) {
  return apiFetch(`/documents?lead_id=${leadId}`);
}

export async function createDocumentForLead(leadId: string, data: any) {
  return apiFetch("/documents", {
    method: "POST",
    body: JSON.stringify({ lead_id: leadId, ...data }),
  });
}

export async function simulateSignDocument(documentId: string) {
  return apiFetch(`/documents/${documentId}/simulate-sign`, {
    method: "POST",
  });
}

// ============ CUSTOMERS API ============
export const customersApi = {
  getAll: (status?: string, page: number = 1, pageSize: number = 10) => {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    params.append('page', page.toString());
    params.append('page_size', pageSize.toString());
    return apiFetch<any>(`/customers?${params.toString()}`);
  },

  getById: (id: string) =>
    apiFetch<any>(`/customers/${id}`),

  create: (data: any) =>
    apiFetch<any>('/customers', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  update: (id: string, data: any) =>
    apiFetch<any>(`/customers/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  delete: (id: string) =>
    apiFetch<any>(`/customers/${id}`, {
      method: 'DELETE',
    }),
};

// ============ POLICIES API ============
export const policiesApi = {
  getAll: (policyType?: string, status?: string, page: number = 1, pageSize: number = 10) => {
    const params = new URLSearchParams();
    if (policyType) params.append('policy_type', policyType);
    if (status) params.append('status', status);
    params.append('page', page.toString());
    params.append('page_size', pageSize.toString());
    return apiFetch<any>(`/policies?${params.toString()}`);
  },

  getById: (id: string) =>
    apiFetch<any>(`/policies/${id}`),

  create: (data: any) =>
    apiFetch<any>('/policies', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  update: (id: string, data: any) =>
    apiFetch<any>(`/policies/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  delete: (id: string) =>
    apiFetch<any>(`/policies/${id}`, {
      method: 'DELETE',
    }),
};

// ============ CLAIMS API ============
export const claimsApi = {
  getAll: (status?: string, claimType?: string, page: number = 1, pageSize: number = 10) => {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    if (claimType) params.append('claim_type', claimType);
    params.append('page', page.toString());
    params.append('page_size', pageSize.toString());
    return apiFetch<any>(`/claims?${params.toString()}`);
  },

  getById: (id: string) =>
    apiFetch<any>(`/claims/${id}`),

  create: (data: any) =>
    apiFetch<any>('/claims', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  update: (id: string, data: any) =>
    apiFetch<any>(`/claims/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  delete: (id: string) =>
    apiFetch<any>(`/claims/${id}`, {
      method: 'DELETE',
    }),
};

// ============ COMMUNICATIONS API ============
export const communicationsApi = {
  getAll: (commType?: string, status?: string, page: number = 1, pageSize: number = 10) => {
    const params = new URLSearchParams();
    if (commType) params.append('comm_type', commType);
    if (status) params.append('status', status);
    params.append('page', page.toString());
    params.append('page_size', pageSize.toString());
    return apiFetch<any>(`/communications?${params.toString()}`);
  },

  getById: (id: string) =>
    apiFetch<any>(`/communications/${id}`),

  create: (data: any) =>
    apiFetch<any>('/communications', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  update: (id: string, data: any) =>
    apiFetch<any>(`/communications/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  delete: (id: string) =>
    apiFetch<any>(`/communications/${id}`, {
      method: 'DELETE',
    }),
};

// ============ REPORTS API ============
export const reportsApi = {
  getAll: (reportType?: string, status?: string, page: number = 1, pageSize: number = 10) => {
    const params = new URLSearchParams();
    if (reportType) params.append('report_type', reportType);
    if (status) params.append('status', status);
    params.append('page', page.toString());
    params.append('page_size', pageSize.toString());
    return apiFetch<any>(`/reports?${params.toString()}`);
  },

  getById: (id: string) =>
    apiFetch<any>(`/reports/${id}`),

  create: (data: any) =>
    apiFetch<any>('/reports', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  update: (id: string, data: any) =>
    apiFetch<any>(`/reports/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  delete: (id: string) =>
    apiFetch<any>(`/reports/${id}`, {
      method: 'DELETE',
    }),
};

// ============ FILE DOCUMENTS API ============
export const fileDocumentsApi = {
  getAll: (params?: { category?: string; status?: string; page?: number; page_size?: number }) => {
    const queryParams = new URLSearchParams();
    if (params?.category) queryParams.append('category', params.category);
    if (params?.status) queryParams.append('status', params.status);
    if (params?.page) queryParams.append('page', params.page.toString());
    if (params?.page_size) queryParams.append('page_size', params.page_size.toString());
    return apiFetch<any>(`/file-management/documents?${queryParams.toString()}`);
  },

  getById: (id: number) =>
    apiFetch<any>(`/file-management/documents/${id}`),

  upload: async (file: File, title: string, category: string, description?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', title);
    formData.append('category', category);
    if (description) formData.append('description', description);

    const response = await fetch(`${API_BASE}/file-management/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    return response.json();
  },

  delete: (id: number, permanent: boolean = false) =>
    apiFetch<any>(`/file-management/documents/${id}?permanent=${permanent}`, {
      method: 'DELETE',
    }),

  download: (id: number) =>
    `${API_BASE}/file-management/documents/${id}/download`,

  getCategories: () =>
    apiFetch<any[]>('/file-management/categories'),

  getStats: () =>
    apiFetch<any>('/file-management/stats'),
};

// Export for backward compatibility
export { API_BASE as default };

