/**
 * Authentication utilities for Lead Generation AI
 * Handles login, registration, session management
 */

import { API_BASE } from "./api";

export interface Session {
  sessionId: string;
  token: string;
  userId: string;
  role: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
}

export interface AuthResponse {
  status: string;
  user_id: string;
  session_id: string;
  token: string;
  role: string;
}

/**
 * Login user
 */
export async function login(credentials: LoginRequest): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(credentials),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Login failed");
  }

  return response.json();
}

/**
 * Register new user
 */
export async function register(data: RegisterRequest): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Registration failed");
  }

  return response.json();
}

/**
 * Get current session from localStorage
 */
export function getSession(): Session | null {
  if (typeof window === "undefined") {
    return null;
  }

  const sessionStr = localStorage.getItem("session");
  if (!sessionStr) {
    return null;
  }

  try {
    return JSON.parse(sessionStr);
  } catch (e) {
    console.error("Failed to parse session", e);
    return null;
  }
}

/**
 * Save session to localStorage
 */
export function setSession(session: Session): void {
  if (typeof window === "undefined") {
    return;
  }

  localStorage.setItem("session", JSON.stringify(session));
}

/**
 * Clear session from localStorage
 */
export function logout(): void {
  if (typeof window === "undefined") {
    return;
  }

  localStorage.removeItem("session");
}

