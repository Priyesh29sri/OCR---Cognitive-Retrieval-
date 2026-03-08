const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

interface AuthResponse {
  access_token: string;
  token_type: string;
  user_id: string;
}

interface UploadResponse {
  document_id: string;
  filename: string;
  message: string;
  summary?: string;
  pages?: number;
  elements_detected?: number;
  text_length?: number;
}

interface QueryResponse {
  answer: string;
  sources?: Array<{
    content: string;
    score: number;
    metadata?: Record<string, any>;
  }>;
  confidence?: number;
}

// Storage keys
const TOKEN_KEY = 'icdi_x_token';
const USER_ID_KEY = 'icdi_x_user_id';

// Token management
export function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(TOKEN_KEY, token);
}

export function getUserId(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(USER_ID_KEY);
}

export function setUserId(userId: string): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(USER_ID_KEY, userId);
}

export function clearAuth(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_ID_KEY);
}

// API Functions

/**
 * Register a new user
 */
export async function register(
  email: string,
  password: string,
  username?: string
): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      email,
      password,
      full_name: username || email.split('@')[0],
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Registration failed');
  }

  const data = await response.json();
  setToken(data.access_token);
  setUserId(data.user_id);
  return data;
}

/**
 * Login existing user
 */
export async function login(email: string, password: string): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      email,
      password,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Login failed');
  }

  const data = await response.json();
  setToken(data.access_token);
  setUserId(data.user_id);
  return data;
}

/**
 * Upload a document
 */
export async function uploadDocument(
  file: File,
  onProgress?: (progress: number) => void
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    // Track upload progress
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        const progress = (e.loaded / e.total) * 100;
        onProgress(progress);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data);
        } catch (error) {
          reject(new Error('Invalid response format'));
        }
      } else {
        try {
          const error = JSON.parse(xhr.responseText);
          reject(new Error(error.detail || 'Upload failed'));
        } catch {
          reject(new Error('Upload failed'));
        }
      }
    });

    xhr.addEventListener('error', () => {
      reject(new Error('Upload failed'));
    });

    xhr.open('POST', `${API_BASE_URL}/upload`);
    xhr.send(formData);
  });
}

/**
 * Query the system
 */
export async function query(
  question: string,
  documentId?: string
): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: question,
      document_id: documentId,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    // FastAPI 422 returns detail as an array; flatten it to a readable string
    if (Array.isArray(error.detail)) {
      const msg = error.detail.map((e: { loc?: string[]; msg: string }) =>
        `${e.loc ? e.loc.join('.') + ': ' : ''}${e.msg}`
      ).join('; ');
      throw new Error(msg || 'Query failed');
    }
    throw new Error(error.detail || 'Query failed');
  }

  return response.json();
}

/**
 * WebSocket chat connection
 */
export function createChatWebSocket(
  onMessage: (message: string) => void,
  onError?: (error: Event) => void
): WebSocket {
  const token = getToken();
  if (!token) {
    throw new Error('Not authenticated');
  }

  const ws = new WebSocket(`ws://127.0.0.1:8000/chat?token=${token}`);

  ws.onmessage = (event) => {
    onMessage(event.data);
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    onError?.(error);
  };

  return ws;
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
  return !!getToken();
}

/**
 * Health check
 */
export async function healthCheck(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/`);
  return response.json();
}
