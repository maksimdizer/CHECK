export type Verdict = 'pass' | 'partial' | 'fail' | 'not_applicable' | 'insufficient_evidence'
export type OutcomeStatus = 'resolved' | 'solution_proposed' | 'escalated' | 'unresolved' | 'unknown'
export type Outcome = {status: OutcomeStatus; explanation: string; evidence: Evidence[]; review_required: boolean}
export type RequestOutcomeState = 'resolved' | 'mostly_resolved' | 'partly_resolved' | 'remedy_proposed' | 'unresolved' | 'unknown'
export type RequestOutcomes = {
  score: number | null; coverage: number | null; scored: number; total: number; reason: string | null; review_required: boolean;
  requests: {request: string; state: RequestOutcomeState; score: number | null; request_evidence: Evidence; evidence: Evidence[]}[];
}
export type Scope = {kind: 'conversation' | 'actor' | 'message'; ref?: string | null}
export type Evidence = {type: 'message' | 'event' | 'knowledge'; ref: string; quote: string}
export type Message = {id: string; actor_id: string; author: string; role: string; at: string; text: string; public: boolean}
export type Event = {id: string; actor_id: string; kind: string; at: string; detail: string}
export type Criterion = {criterion_id: string; name: string; explanation: string; verdict: Verdict; evidence: Evidence[]; severity: string; score: number | null; weight: number; group_id?: string | null; unavailable_reason?: 'policy_unavailable'}
export type Definition = {id: string; group_id?: string; name: string; version: number; description: string; applicability: string; evaluator_type: string; weight: number; severity: string; rubric: Record<string, string>; review_policy: string; required_evidence: string[]; category?: string; channel_threshold_seconds?: Record<string, number>}
export type CardGroup = {id: string; name: string; weight?: number}
export const timingNames = {first_response: 'First response time', time_to_close: 'Time to close'}
export type TimingMetric = keyof typeof timingNames
export type TimingTarget = {default_seconds?: number | null; channel_seconds?: Record<string, number>}
export type TimingResult = {
  id: TimingMetric; name: string; status: 'met' | 'late' | 'no_target' | 'unavailable' | 'not_applicable';
  actual_seconds: number | null; target_seconds: number | null; delay_seconds: number | null;
}
export type Card = {id: string; version: number; name: string; groups?: CardGroup[]; criteria: Definition[]; timing_targets?: Partial<Record<TimingMetric, TimingTarget>>}

export function formatDuration(seconds: number) {
  const whole = Math.floor(seconds)
  return [
    [Math.floor(whole / 86400), 'd'], [Math.floor(whole / 3600) % 24, 'h'],
    [Math.floor(whole / 60) % 60, 'min'], [whole % 60, 's'],
  ].filter(([value]) => value).map(([value, unit]) => `${value} ${unit}`).join(' ') || (seconds > 0 ? '< 1 s' : '0 s')
}

export function cardGroupWeight(group: CardGroup, criteria: Definition[]) {
  return group.weight ?? criteria.filter(item => item.group_id === group.id).reduce((sum, item) => sum + item.weight, 0)
}

export const formatWeight = new Intl.NumberFormat('en', {maximumFractionDigits: 2}).format
export type Summary = {
  id: string; external_id: string; subject: string; source: string; channel: string; category: string;
  status: 'queued' | 'running' | 'complete' | 'failed'; error: string | null; reviewed: boolean;
  review_required: boolean; score: number | null; coverage: number | null; critical: string | null;
  weighted_coverage?: number | null; checked_weight?: number | null; applicable_weight?: number | null; outcome_status?: OutcomeStatus | null;
  created_at: string; scope: Scope;
}
export type Review = {id: string; reviewer: string; action: string; criterion_id: string | null; verdict: Verdict | null; outcome_status?: OutcomeStatus | null; comment: string; created_at: string}
export type Run = Summary & {
  ticket: {messages: Message[]; events: Event[]; messages_complete: boolean; events_complete: boolean; observed_at?: string | null;
    refund?: {decision: string; due_at?: string | null; operation_id?: string | null; owner_id?: string | null; completion_claim_message_id?: string | null} | null};
  result: {criteria: Criterion[]; checked: number; applicable: number; duration_ms: number; outcome?: Outcome; request_outcome?: RequestOutcomes; timing?: TimingResult[]} | null;
  config: {card: Card; model: string; knowledge: {ref: string; title: string; version: number; text: string}[]};
  display_names?: {criteria: Record<string, string>; groups: Record<string, string>};
  reviews: Review[];
}
export type CloudProvider = 'openai' | 'openrouter' | 'anthropic' | 'google'
export type LocalModel = {
  model: string; output_tokens: number; context_tokens: number | null; reasoning_tokens: number | null
  think: boolean | null; agent_mode: boolean; agent_quality: boolean
}
export type EvaluationProfile = {
  mode: 'local' | 'cloud'
  local: LocalModel
  cloud: {provider: CloudProvider; model: string; output_tokens: number}
  configured_cloud_providers: CloudProvider[]
}
export type Settings = {demo: boolean; organization: string; model: string; card: Card; evaluation: EvaluationProfile}
export type KnowledgeVersion = {
  id: string; document: string; version: number; title: string; product: string; category: string; language: string;
  effective_from: string; effective_to: string | null; available_from: string; available_to: string | null;
  audience: string[]; fragment_count: number;
}
export type KnowledgePreview = {
  run_id: string; subject: string; external_id: string; scope: Scope;
  replies: {id: string; author: string; at: string; text: string}[];
  items: {ref: string; title: string; version: number; message_id: string; text: string; search_mode?: 'section' | 'fallback'}[];
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message) }
}

export async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers)
  const token = sessionStorage.getItem('qa-token')
  if (token) headers.set('Authorization', 'Bearer ' + token)
  const response = await fetch('/api' + path, {...options, headers})
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    const detail = body.detail
    const message = Array.isArray(detail)
      ? detail.slice(0, 3).map(item => `${(item.location || item.loc || []).join('.')} — ${item.message || item.msg}`).join('\n')
      : detail || `Request failed (${response.status})`
    throw new ApiError(response.status, message)
  }
  return response.json() as Promise<T>
}

export function post<T>(path: string, value: unknown) {
  return request<T>(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(value)})
}
