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
  citations?: Array<{
    chunk_index: number;
    text_preview: string;
    source_type: string;
    document_id?: string;
    relevance_score: number;
  }>;
  confidence?: number;
  metadata?: Record<string, any>;
}

export interface InsightsResponse {
  doc_id: string;
  doc_type: string;
  complexity: 'basic' | 'intermediate' | 'advanced' | 'unknown';
  insights: string[];
  suggested_questions: string[];
  key_entities: string[];
  key_themes: string[];
  ib_coverage: number;
  chunks_analyzed: number;
  chunks_selected_by_ib: number;
  error?: string;
}

export interface StudyGuideResponse {
  doc_id: string;
  title: string;
  summary: string;
  key_concepts: string[];
  vocabulary: Array<{ term: string; definition: string }>;
  blooms_questions: {
    remember: string[];
    understand: string[];
    apply: string[];
    analyze: string[];
    evaluate: string[];
    create: string[];
  };
  concept_map: Array<{ from: string; relation: string; to: string }>;
  estimated_study_time_minutes: number;
  blooms_taxonomy_info: Record<string, string>;
  error?: string;
}

export interface ContradictionResponse {
  contradictions: Array<{
    topic: string;
    doc_a_claim: string;
    doc_b_claim: string;
    contradiction_type: 'direct' | 'implied' | 'scope' | 'methodology';
    severity: 'high' | 'medium' | 'low';
    confidence: number;
    resolution: string;
  }>;
  agreements: string[];
  summary: string;
  overall_agreement_score: number;
  doc_a_name: string;
  doc_b_name: string;
  error?: string;
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

// ── Novel ICDI-X Features ────────────────────────────────────────────────────

/**
 * Get proactive insights for a document (IB-powered)
 */
export async function getInsights(docId: string): Promise<InsightsResponse> {
  const response = await fetch(`${API_BASE_URL}/insights/${docId}`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Insights generation failed');
  }
  return response.json();
}

/**
 * Get Bloom's Taxonomy study guide for a document
 */
export async function getStudyGuide(docId: string): Promise<StudyGuideResponse> {
  const response = await fetch(`${API_BASE_URL}/studyguide/${docId}`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Study guide generation failed');
  }
  return response.json();
}

/**
 * Detect contradictions between two documents
 */
export async function detectContradictions(
  docAId: string,
  docBId: string,
  topic?: string
): Promise<ContradictionResponse> {
  const params = new URLSearchParams({
    doc_a_id: docAId,
    doc_b_id: docBId,
    ...(topic ? { topic } : {}),
  });
  const response = await fetch(`${API_BASE_URL}/contradictions?${params}`, {
    method: 'POST',
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Contradiction detection failed');
  }
  return response.json();
}

/**
 * Query across multiple documents simultaneously
 */
export async function queryMultiDoc(
  question: string,
  documentIds: string[]
): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/query_multi`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: question,
      document_ids: documentIds,
    }),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Multi-document query failed');
  }
  return response.json();
}

/**
 * Streaming query — returns an async generator of tokens (Perplexity-style)
 * Usage:
 *   for await (const event of queryStream("question", "doc_id")) {
 *     if (event.done) { ... } else { appendToken(event.token); }
 *   }
 */
export async function* queryStream(
  question: string,
  documentId?: string
): AsyncGenerator<{ token: string; done: boolean; method?: string; citations?: QueryResponse['citations'] }> {
  const response = await fetch(`${API_BASE_URL}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: question, document_id: documentId }),
  });

  if (!response.ok || !response.body) {
    throw new Error('Streaming query failed');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          yield JSON.parse(line.slice(6));
        } catch {
          // skip malformed event
        }
      }
    }
  }
}

/**
 * Get D3-compatible knowledge graph JSON
 */
export async function getKnowledgeGraphD3(): Promise<{
  nodes: Array<{ id: string; label: string; group: string }>;
  links: Array<{ source: string; target: string; relation: string; value: number }>;
  stats: Record<string, any>;
}> {
  const response = await fetch(`${API_BASE_URL}/knowledge-graph/d3`);
  if (!response.ok) {
    throw new Error('Knowledge graph export failed');
  }
  return response.json();
}
