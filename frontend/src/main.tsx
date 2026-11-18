import {Fragment, useCallback, useEffect, useRef, useState, type FormEvent, type ReactNode} from 'react'
import {createRoot} from 'react-dom/client'
import {
  Activity, ArrowDownToLine, ArrowLeft, ArrowRight, ArrowUpRight, BookOpen, Bot, Check, CheckCircle2,
  ChevronDown, ChevronLeft, ChevronRight, CircleHelp, ClipboardCheck, Clock3, FileUp, Gauge, History,
  Inbox, LayoutList, Link2, ListChecks, LoaderCircle, MessageCircle, RotateCcw, Search, ShieldAlert,
  Settings as SettingsIcon, ShieldCheck, UserRound, X,
} from 'lucide-react'
import {ApiError, cardGroupWeight, formatDuration, formatWeight, post, request, timingNames, type Card, type Criterion, type Evidence, type OutcomeStatus, type RequestOutcomeState, type Run, type Scope, type Settings, type Summary, type TimingResult, type Verdict} from './api'
import './style.css'
import {CardEditor} from './CardEditor'
import {EvaluationSettings} from './ModelSettings'
import {Knowledge} from './Knowledge'

const verdicts: Record<Verdict, string> = {
  pass: 'Pass', fail: 'Fail', partial: 'Partial', not_applicable: 'Not applicable', insufficient_evidence: 'Insufficient evidence',
}
const categories: Record<string, string> = {refund: 'Refund', access: 'Account access', delivery: 'Delivery', general: 'General'}
const channels: Record<string, string> = {chat: 'Chat', email: 'Email'}
const actions: Record<string, string> = {confirm: 'Confirmed', correct: 'Criterion corrected', correct_outcome: 'Outcome corrected', rubric_issue: 'Rubric issue', missing_data: 'Insufficient evidence'}
const outcomes: Record<OutcomeStatus, string> = {resolved: 'Resolution confirmed', solution_proposed: 'Solution proposed', escalated: 'Escalated', unresolved: 'Unresolved', unknown: 'Outcome unknown'}
const outcomeTone: Record<OutcomeStatus, string> = {resolved: 'green', solution_proposed: 'violet', escalated: 'violet', unresolved: 'amber', unknown: 'neutral'}
const requestStates: Record<RequestOutcomeState, string> = {
  resolved: 'Resolved', mostly_resolved: 'Mostly resolved', partly_resolved: 'Partly resolved',
  remedy_proposed: 'Remedy proposed', unresolved: 'Unresolved', unknown: 'Outcome unknown',
}
const refundDecisions: Record<string, string> = {approved: 'Approved', declined: 'Declined', pending: 'Pending', not_required: 'Not required', unknown: 'Unknown'}
const time = new Intl.DateTimeFormat('en', {hour: '2-digit', minute: '2-digit'})
const date = new Intl.DateTimeFormat('en', {day: 'numeric', month: 'long', year: 'numeric'})
const dateTime = new Intl.DateTimeFormat('en', {dateStyle: 'short', timeStyle: 'short'})
const errorText = (error: unknown) => error instanceof Error ? error.message : 'The request failed'

const criterionName = (run: Run, item: Criterion) => run.display_names?.criteria[item.criterion_id] ?? item.name

function Badge({children, tone = 'neutral'}: {children: ReactNode; tone?: string}) {
  return <span className={'badge ' + tone}>{children}</span>
}

function status(run: Summary) {
  if (run.status === 'queued') return <Badge><Clock3 size={12}/> Queued</Badge>
  if (run.status === 'running') return <Badge tone="violet"><LoaderCircle size={12} className="spin"/> Evaluating</Badge>
  if (run.status === 'failed') return <Badge tone="red"><ShieldAlert size={12}/> Error</Badge>
  if (run.reviewed) return <Badge tone="green"><Check size={12}/> Reviewed</Badge>
  if (run.review_required) return <Badge tone="amber"><span className="status-dot"/> Needs review</Badge>
  return <Badge tone="green"><Check size={12}/> Evaluated</Badge>
}

function VerdictBadge({verdict}: {verdict: Verdict}) {
  const tone = verdict === 'pass' ? 'green' : verdict === 'fail' ? 'red' : verdict === 'not_applicable' ? 'neutral' : 'amber'
  return <Badge tone={tone}>{verdict === 'pass' ? <Check size={12}/> : <span className="status-dot"/>}{verdicts[verdict]}</Badge>
}

function Modal({title, close, children}: {title: string; close: () => void; children: ReactNode}) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => { dialog.current?.showModal() }, [])
  return <dialog ref={dialog} aria-label={title} onCancel={event => {event.preventDefault(); close()}} onClick={event => {if (event.target === event.currentTarget) close()}}>
    <div className="modal-inner">
      <div className="modal-head"><h2>{title}</h2><button className="icon-button" aria-label="Close" onClick={close}><X size={20}/></button></div>
      {children}
    </div>
  </dialog>
}

function ImportDialog({close, imported}: {close: () => void; imported: (id?: string) => void}) {
  const [file, setFile] = useState<File>()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!file || busy) return
    setBusy(true)
    setError('')
    try {
      const format = file.name.toLowerCase().endsWith('.csv') ? 'csv' : 'json'
      const result = await request<{items: {run_id: string | null}[]}>(`/import?format=${format}`, {method: 'POST', body: file})
      imported(result.items.find(item => item.run_id)?.run_id ?? undefined)
    } catch (err) { setError(errorText(err)) }
    finally { setBusy(false) }
  }
  return <Modal title="Import tickets" close={busy ? () => {} : close}>
    <form onSubmit={submit} className="form">
      <p>Upload a JSON or CSV file up to 5 MB. Closed tickets are automatically queued for evaluation.</p>
      <label className="file-field"><FileUp size={24}/><span>{file?.name || 'Choose a JSON or CSV file'}</span><input type="file" accept=".json,.csv" onChange={event => setFile(event.target.files?.[0])}/></label>
      <a className="inline-link" href="/example.json" download><ArrowDownToLine size={15}/> Download sample JSON</a>
      <a className="inline-link" href="/example-refund.json" download><ArrowDownToLine size={15}/> Download a confirmed refund example</a>
      {error && <div className="form-error" role="alert">{error}</div>}
      <button className="button primary" disabled={!file || busy}>{busy ? 'Importing…' : 'Upload tickets'}</button>
    </form>
  </Modal>
}

function ReviewDialog({run, close, saved}: {run: Run; close: () => void; saved: () => void}) {
  const [action, setAction] = useState('confirm')
  const [criterion, setCriterion] = useState(run.result?.criteria[0]?.criterion_id || '')
  const [verdict, setVerdict] = useState<Verdict>('pass')
  const [outcomeStatus, setOutcomeStatus] = useState<OutcomeStatus>(run.result?.outcome?.status || 'unknown')
  const [comment, setComment] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      await post(`/runs/${run.id}/reviews`, {action, criterion_id: action === 'correct' ? criterion : null, verdict: action === 'correct' ? verdict : null, outcome_status: action === 'correct_outcome' ? outcomeStatus : null, comment})
      saved()
    } catch (err) {setError(errorText(err))}
    finally {setBusy(false)}
  }
  return <Modal title="Reviewer decision" close={busy ? () => {} : close}>
    <form className="form" onSubmit={submit}>
      <label>Action<select value={action} onChange={event => setAction(event.target.value)}>{Object.entries(actions).filter(([value]) => value !== 'correct_outcome' || run.result?.outcome).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      {action === 'correct' && <>
        <label>Criterion<select value={criterion} onChange={event => setCriterion(event.target.value)}>{run.result?.criteria.map(item => <option key={item.criterion_id} value={item.criterion_id}>{criterionName(run, item)}</option>)}</select></label>
        <label>Verdict<select value={verdict} onChange={event => setVerdict(event.target.value as Verdict)}>{Object.entries(verdicts).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      </>}
      {action === 'correct_outcome' && <label>Ticket outcome<select value={outcomeStatus} onChange={event => setOutcomeStatus(event.target.value as OutcomeStatus)}>{Object.entries(outcomes).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>}
      <label>Comment<textarea value={comment} onChange={event => setComment(event.target.value)} rows={3} maxLength={2000} required={action !== 'confirm'}/></label>
      {error && <div className="form-error" role="alert">{error}</div>}
      <button className="button primary" disabled={busy}>{busy ? 'Saving…' : 'Save decision'}</button>
    </form>
  </Modal>
}

function RerunDialog({run, close, queued}: {run: Run; close: () => void; queued: (id: string) => void}) {
  const [kind, setKind] = useState<Scope['kind']>(run.scope.kind)
  const [ref, setRef] = useState(run.scope.ref || '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const support = run.ticket.messages.filter(item => item.public && ['agent', 'bot'].includes(item.role))
  const choices = kind === 'actor' ? Array.from(new Map(support.map(item => [item.actor_id, item.author]))) : support.map(item => [item.id, item.author + ' · ' + time.format(new Date(item.at))])
  const selected = choices.some(item => item[0] === ref) ? ref : choices[0]?.[0]
  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const value = await post<{id: string}>(`/runs/${run.id}/rerun`, {scope: {kind, ref: kind === 'conversation' ? null : selected}})
      queued(value.id)
    } catch (err) {setError(errorText(err))}
    finally {setBusy(false)}
  }
  return <Modal title="New evaluation" close={busy ? () => {} : close}>
    <form className="form" onSubmit={submit}>
      <label>Evaluation scope<select value={kind} onChange={event => {setKind(event.target.value as Scope['kind']); setRef('')}}>
        <option value="conversation">Entire ticket</option><option value="actor">Participant</option><option value="message">Single reply</option>
      </select></label>
      {kind !== 'conversation' && <label>{kind === 'actor' ? 'Participant' : 'Reply'}<select value={selected || ''} onChange={event => setRef(event.target.value)}>
        {!choices.length && <option value="">No public replies</option>}
        {choices.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
      </select></label>}
      {error && <div className="form-error" role="alert">{error}</div>}
      <button className="button primary" disabled={busy || (kind !== 'conversation' && !selected)}>{busy ? 'Starting…' : 'Start evaluation'}</button>
    </form>
  </Modal>
}

function CriteriaCard({card}: {card: Card}) {
  const groups = card.groups || []
  const totalWeight = groups.reduce((sum, group) => sum + cardGroupWeight(group, card.criteria), 0)
  return <section className="panel card-page">
    <header className="panel-heading"><div className="panel-title"><BookOpen size={19}/><h2>{card.name}</h2></div><Badge tone="violet">Version {card.version}</Badge></header>
    <p className="card-intro">Not applicable criteria are excluded from scoring. Insufficient evidence lowers coverage.</p>
    {(card.groups || [{id: 'all', name: 'Criteria'}]).map(group => {
      const weight = cardGroupWeight(group, card.criteria)
      return <section key={group.id}>
        <h3 className="criteria-group-title">{group.name}{card.groups && <span>Group weight {formatWeight(weight)} · {formatWeight(totalWeight ? 100 * weight / totalWeight : 0)}% of scorecard</span>}</h3>
        {card.criteria.filter(item => !card.groups || item.group_id === group.id).map(item => <details className="definition" key={item.id}>
          <summary><div><strong>{item.name}</strong><p>{item.description}</p></div><Badge>{item.weight && (!card.groups || weight) ? `Weight ${formatWeight(item.weight)}` : 'Unscored check'}</Badge><ChevronDown size={15}/></summary>
          <dl><div><dt>When it applies</dt><dd>{item.applicability}</dd></div>{Object.entries(item.rubric).map(([key, value]) => <div key={key}><dt>{verdicts[key as Verdict]}</dt><dd>{value}</dd></div>)}</dl>
        </details>)}
      </section>
    })}
    {card.timing_targets && <section className="timing-summary" aria-label="Timing targets">
      <h3>Timing targets</h3>
      {Object.entries(card.timing_targets).map(([id, target]) => <div className="timing-row" key={id}>
        <strong>{timingNames[id as keyof typeof timingNames]}</strong>
        <p>{[
          ...(target.default_seconds == null ? [] : [`Default ${formatDuration(target.default_seconds)}`]),
          ...Object.entries(target.channel_seconds ?? {}).map(([channel, seconds]) => `${channel} ${formatDuration(seconds)}`),
        ].join(' · ') || 'No target configured'}</p>
      </div>)}
    </section>}
  </section>
}

function TimingSummary({items}: {items: TimingResult[]}) {
  return <section className="timing-summary" aria-label="Timing">
    <h3><Clock3 size={16}/> Timing</h3>
    {items.map(item => <div className="timing-row" key={item.id}>
      <div className="timing-heading">
        <strong>{item.name}</strong>
        <Badge tone={item.status === 'late' ? 'red' : item.status === 'met' ? 'green' : 'neutral'}>
          {item.status === 'late' && item.delay_seconds != null ? `Late by ${formatDuration(item.delay_seconds)}`
            : {met: 'On time', late: 'Late', no_target: 'No target', unavailable: 'Unavailable', not_applicable: 'Not applicable'}[item.status]}
        </Badge>
      </div>
      <p>{item.actual_seconds != null ? `Actual ${formatDuration(item.actual_seconds)}` : 'Actual time unavailable'}
        {item.target_seconds != null && ` · Target ${formatDuration(item.target_seconds)}`}</p>
    </div>)}
  </section>
}

function EvidenceLinks({items, show}: {items: Evidence[]; show: (item: Evidence) => void}) {
  return <div className="evidence-list">{items.map((entry, index) => <button key={index} onClick={() => show(entry)} title={entry.quote}>
    {entry.type === 'knowledge' ? <BookOpen size={13}/> : <Link2 size={13}/>}
    {entry.type === 'knowledge' ? 'Policy' : entry.type === 'event' ? 'Event ' + entry.ref : 'Message ' + entry.ref}
    <ArrowUpRight size={12}/>
  </button>)}</div>
}

function Criteria({run, view, setView, evidence, review}: {run: Run; view: 'criteria' | 'reviews'; setView: (value: 'criteria' | 'reviews') => void; evidence: (item: Evidence) => void; review: () => void}) {
  const [expanded, setExpanded] = useState<string[]>([])
  useEffect(() => {
    const items = run.result?.criteria || []
    setExpanded(items.find(item => item.verdict === 'fail') ? [items.find(item => item.verdict === 'fail')!.criterion_id] : [])
  }, [run.id, run.status])
  const toggle = (id: string) => setExpanded(old => old.includes(id) ? old.filter(item => item !== id) : [...old, id])
  const result = run.result
  const groups = run.config.card.groups
  const groupedWeights = groups?.length && groups.every(group => group.weight != null)
  const missing = result?.criteria.filter(item => item.verdict === 'insufficient_evidence') || []
  const outcomeReview = run.reviews.findLast(item => item.action === 'correct_outcome' && item.outcome_status)
  const requests = result?.request_outcome
  const orderedCriteria = result ? (run.config.card.groups
    ? [...run.config.card.groups.flatMap(group => result.criteria.filter(item => item.group_id === group.id)), ...result.criteria.filter(item => !run.config.card.groups?.some(group => group.id === item.group_id))]
    : result.criteria) : []
  return <section className="panel assessment">
    <header className="panel-heading"><div className="panel-title"><ClipboardCheck size={19}/><h2>Quality evaluation</h2></div><Badge>Version {run.config.card.version}</Badge></header>
    {result ? <>
      <div className="metrics">
        <div className="metric"><span><Gauge size={15}/> Quality score</span><strong>{run.score ?? '—'}<small>{run.score != null && ' / 100'}</small></strong><p>Based on scored criteria</p></div>
        <div className="metric"><span><Activity size={15}/> Weighted coverage</span><strong>{run.weighted_coverage == null ? '—' : run.weighted_coverage + '%'}</strong><p>{run.applicable_weight == null ? 'Unavailable' : `Weight ${formatWeight(run.checked_weight ?? 0)} of ${formatWeight(run.applicable_weight)}`}</p></div>
      </div>
      {result.timing?.length ? <TimingSummary items={result.timing}/> : null}
      <div className={'critical-summary ' + run.critical}><ShieldAlert size={16}/><span>{run.critical === 'violation' ? 'Critical violation confirmed' : run.critical === 'uncertain' ? 'Critical checks need attention' : run.critical === 'not_applicable' ? 'Critical checks are not applicable' : 'No critical violations found in evaluated checks'}</span></div>
      {!!missing.length && <details className="coverage-note"><summary>Could not evaluate: {missing.length}</summary><p>Coverage shows how much of the scorecard contributes to the score.</p><ul>{missing.map(item => <li key={item.criterion_id}>{criterionName(run, item)}{item.weight ? ` · weight ${item.weight}` : ' · unscored check'}</li>)}</ul></details>}
      <section className="outcome-card" aria-label="Ticket outcome">
        <div className="outcome-heading"><strong>Ticket outcome</strong><span>Entire conversation · separate from the quality score</span></div>
        {result.outcome ? <>
          {outcomeReview && <p className="automatic-label">Automated evaluation</p>}
          <Badge tone={outcomeTone[result.outcome.status]}>{outcomes[result.outcome.status]}</Badge>
          <p>{result.outcome.explanation}</p>
          <EvidenceLinks items={result.outcome.evidence} show={evidence}/>
          {outcomeReview?.outcome_status && <p className="human-verdict">Reviewer decision: {outcomes[outcomeReview.outcome_status]}<br/>{outcomeReview.comment}</p>}
        </> : <p>This evaluation has no separate outcome. Run a new evaluation to assess it.</p>}
        {requests && <section className="request-outcomes" aria-label="Outcome by request">
          <div className="request-outcome-metrics">
            <div><span>Outcome score</span><strong>{requests.score == null ? '—' : `${requests.score} / 100`}</strong></div>
            <div><span>Outcome coverage</span><strong>{requests.coverage == null ? '—' : `${requests.coverage}%`}</strong>
              {requests.total > 0 && <small>{requests.scored} of {requests.total} requests scored</small>}</div>
          </div>
          {requests.reason && <p>{requests.reason}</p>}
          {requests.requests.length > 0 && <ul className="request-outcome-list">{requests.requests.map(row =>
            <li key={row.request_evidence.ref + ':' + row.request}>
              <div><strong>{row.request}</strong><span>{requestStates[row.state]} · {row.score == null ? 'Unscored' : `${row.score} / 100`}</span></div>
              <EvidenceLinks items={[row.request_evidence, ...row.evidence.filter(item => item.ref !== row.request_evidence.ref || item.type !== 'message')]} show={evidence}/>
            </li>
          )}</ul>}
        </section>}
      </section>
      <div className="tabs">
        <button className={view === 'criteria' ? 'active' : ''} onClick={() => setView('criteria')}>Criteria <span>{result.criteria.length}</span></button>
        <button className={view === 'reviews' ? 'active' : ''} onClick={() => setView('reviews')}>History <span>{run.reviews.length}</span></button>
      </div>
      {view === 'criteria' ? <div className="criteria">
        {orderedCriteria.map((item: Criterion, index) => {
          const open = expanded.includes(item.criterion_id)
          const latest = run.reviews.findLast(entry => entry.criterion_id === item.criterion_id && entry.verdict)
          const group = groups?.find(group => group.id === item.group_id)
          const isWeighted = item.weight > 0 && (!groupedWeights || group?.weight !== 0)
          return <Fragment key={item.criterion_id}>
          {item.group_id && item.group_id !== orderedCriteria[index - 1]?.group_id && <h3 className="criteria-group-title compact">{group ? run.display_names?.groups[group.id] ?? group.name : 'Criteria'}{groupedWeights && group?.weight != null && <span>Group weight {formatWeight(group.weight)}</span>}</h3>}
          <article className={'criterion ' + (open ? 'expanded' : '')}>
            <button className="criterion-toggle" onClick={() => toggle(item.criterion_id)} aria-expanded={open}>
              <span>{criterionName(run, item)}<span className="criterion-meta">{isWeighted ? `${groupedWeights ? 'Weight within group' : 'Weight'} ${formatWeight(item.weight)}` : 'Unscored check'}{item.severity === 'critical' && ' · critical'}</span></span>
              {item.unavailable_reason === 'policy_unavailable' ? <Badge>Policy unavailable</Badge> : <VerdictBadge verdict={item.verdict}/>}<ChevronDown size={15} className="chevron"/>
            </button>
            {open && <div className="criterion-body">
              <p>{item.explanation}</p>
              {latest?.verdict && <p className="human-verdict">Reviewer decision · {verdicts[latest.verdict]}</p>}
              <EvidenceLinks items={item.evidence} show={evidence}/>
            </div>}
          </article></Fragment>
        })}
      </div> : <div className="reviews">
        {run.reviews.length ? run.reviews.map(entry => <article key={entry.id}>
          <strong>{actions[entry.action]}</strong><span>{entry.reviewer} · {date.format(new Date(entry.created_at))}</span>
          {entry.criterion_id && <p>{run.display_names?.criteria[entry.criterion_id] ?? result.criteria.find(item => item.criterion_id === entry.criterion_id)?.name} · {entry.verdict && verdicts[entry.verdict]}</p>}
          {entry.outcome_status && <p>{outcomes[entry.outcome_status]}</p>}
          {entry.comment && <p>{entry.comment}</p>}
        </article>) : <div className="empty compact"><History size={23}/><p>No review decisions yet</p></div>}
      </div>}
      <footer className="panel-footer"><span><CheckCircle2 size={15}/> Criteria: {result.checked} of {result.applicable}{run.coverage != null && ` · ${run.coverage}%`}</span><button className="button primary small" onClick={review}>{run.reviewed ? 'Add decision' : 'Review' }<ArrowRight size={15}/></button></footer>
    </> : <div className="empty compact">
      {run.status === 'failed' ? <ShieldAlert size={26}/> : <LoaderCircle size={26} className="spin"/>}
      <p>{run.error || 'The evaluation runs in the background'}</p>
    </div>}
  </section>
}

function exportRun(run: Run) {
  const {display_names, ...snapshot} = run
  const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot, null, 2)], {type: 'application/json'}))
  const link = document.createElement('a')
  link.href = url
  link.download = 'qa-' + run.id + '.json'
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function App() {
  const [settings, setSettings] = useState<Settings>()
  const [runs, setRuns] = useState<Summary[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)
  const [id, setId] = useState('')
  const [run, setRun] = useState<Run>()
  const [section, setSection] = useState<'queue' | 'card' | 'knowledge'>('queue')
  const [editingCard, setEditingCard] = useState(false)
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')
  const [mobileDetail, setMobileDetail] = useState(false)
  const [tab, setTab] = useState<'messages' | 'events'>('messages')
  const [reviewTab, setReviewTab] = useState<'criteria' | 'reviews'>('criteria')
  const [modal, setModal] = useState<'import' | 'review' | 'rerun' | 'settings'>()
  const [knowledge, setKnowledge] = useState<Evidence>()
  const [highlight, setHighlight] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [locked, setLocked] = useState(false)
  const [token, setToken] = useState('')
  const [authEpoch, setAuthEpoch] = useState(0)
  const selected = run?.id === id ? run : undefined

  const failed = useCallback((err: unknown) => {
    if (err instanceof ApiError && err.status === 401) setLocked(true)
    else setError(errorText(err))
  }, [])
  const refresh = useCallback(async (isCurrent: () => boolean = () => true) => {
    try {
      const value = await request<{items: Summary[]; total: number}>(`/runs?offset=${page * 100}&status=${filter}&search=${encodeURIComponent(search.trim())}`)
      if (!isCurrent()) return
      setRuns(value.items)
      setTotal(value.total)
      setId(old => old || value.items.find(item => item.source === 'demo' && item.external_id === '1048')?.id || value.items[0]?.id || '')
    } catch (err) {if (isCurrent()) failed(err)}
  }, [page, filter, search, failed])
  const detail = useCallback(async (runId: string, isCurrent: () => boolean = () => true) => {
    try {
      const value = await request<Run>(`/runs/${runId}`)
      if (isCurrent()) setRun(value)
    } catch (err) {if (isCurrent()) failed(err)}
  }, [failed])

  useEffect(() => {
    if (locked) return
    let active = true
    request<Settings>('/settings').then(value => {if (active) setSettings(value)}).catch(err => {if (active) failed(err)})
    return () => {active = false}
  }, [authEpoch, failed, locked])
  useEffect(() => {
    if (locked) return
    let active = true
    const update = () => {void refresh(() => active); if (id) void detail(id, () => active)}
    update()
    const timer = setInterval(update, 2500)
    return () => {active = false; clearInterval(timer)}
  }, [refresh, detail, id, authEpoch, locked])
  useEffect(() => {
    if (!highlight) return
    document.getElementById(highlight)?.scrollIntoView({behavior: 'smooth', block: 'nearest'})
  }, [tab, highlight])

  function select(id: string) {
    setId(id); setTab('messages'); setReviewTab('criteria'); setHighlight(''); setMobileDetail(true)
  }
  const showEvidence = (item: Evidence) => {
    if (item.type === 'knowledge') setKnowledge(item)
    else {
      setTab(item.type === 'message' ? 'messages' : 'events')
      setHighlight(item.type + '-' + item.ref)
    }
  }
  const added = (newId?: string) => {
    setModal(undefined)
    setPage(0)
    setFilter('all')
    setSearch('')
    if (newId) select(newId)
    setNotice(newId ? 'Evaluation queued' : 'Tickets imported')
    if (page === 0) void refresh()
  }
  const submitToken = (event: FormEvent) => {
    event.preventDefault(); sessionStorage.setItem('qa-token', token.trim()); setLocked(false); setAuthEpoch(old => old + 1)
  }
  const scopeLabel = selected?.scope.kind === 'actor'
    ? selected.ticket.messages.find(item => item.actor_id === selected.scope.ref)?.author
    : selected?.scope.kind === 'message' ? 'Reply ' + selected.scope.ref : 'Entire ticket'
  const knowledgeDocument = selected?.config.knowledge.find(item => item.ref === knowledge?.ref)
  const participant = selected?.ticket.messages.find(item => ['agent', 'bot'].includes(item.role))
  return <div className="app">
    <header className="topbar">
      <button className="brand" onClick={() => setSection('queue')}><span className="brand-icon"><ShieldCheck size={21}/></span>CHECK</button>
      <nav aria-label="Sections">
        <button className={section === 'queue' ? 'active' : ''} onClick={() => setSection('queue')}><LayoutList size={17}/> Tickets</button>
        <button className={section === 'card' ? 'active' : ''} onClick={() => setSection('card')}><ListChecks size={17}/> Scorecard</button>
        <button className={section === 'knowledge' ? 'active' : ''} onClick={() => setSection('knowledge')} disabled={locked}><BookOpen size={17}/> Knowledge</button>
      </nav>
      <div className="header-meta">
        {settings && !locked && <span className="model-status" data-mode={settings.evaluation.mode} title={settings.model || undefined}>
          <span className="status-dot"/><strong>{settings.evaluation.mode === 'cloud' ? 'Cloud' : 'Local'}</strong>
          <span className="model-name">{settings.model || 'No model configured'}</span>
        </span>}
        <button className="settings-button" title="Model settings" aria-label="Model settings" aria-haspopup="dialog"
          disabled={!settings || locked} onClick={() => setModal('settings')}><SettingsIcon size={17}/></button>
      </div>
    </header>
    <main>
      <div className="page-heading">
        <div><h1>{section === 'queue' ? 'Support tickets' : section === 'card' ? 'Scorecard' : 'Knowledge'}</h1></div>
        {section === 'queue' && <button className="button primary import-button" onClick={() => setModal('import')} disabled={locked}><FileUp size={17}/> Import</button>}
        {section === 'card' && !editingCard && settings && <button className="button primary" onClick={() => setEditingCard(true)}>Edit scorecard</button>}
      </div>
      {error && <div className="banner error" role="alert">{error}<button className="icon-button" onClick={() => setError('')} aria-label="Dismiss error"><X size={16}/></button></div>}
      {notice && <div className="banner notice" role="status"><Check size={17}/>{notice}<button className="icon-button" onClick={() => setNotice('')} aria-label="Dismiss notification"><X size={16}/></button></div>}
      {settings && <div hidden={section !== 'card'}>{editingCard
        ? <CardEditor key={settings.card.version} card={settings.card} close={() => setEditingCard(false)} saved={card => {setSettings(old => old && {...old, card}); setEditingCard(false); setNotice(`Scorecard version ${card.version} saved. New evaluations will use this version.`)}}/>
        : <CriteriaCard card={settings.card}/>}</div>}
      {section === 'knowledge' && !locked && <Knowledge selected={selected}/>}
      {section === 'queue' && <div className={'workspace ' + (mobileDetail ? 'mobile-detail' : '')}>
        <aside className="panel queue">
          <header className="panel-heading"><div className="panel-title"><Inbox size={18}/><h2>Queue</h2></div><Badge>{total}</Badge></header>
          <div className="queue-controls">
            <label className="search"><Search size={16}/><input placeholder="Ticket ID or subject" maxLength={120} aria-label="Search queue" value={search} onChange={event => {setSearch(event.target.value); setPage(0)}}/></label>
            <select aria-label="Evaluation status" value={filter} onChange={event => {setFilter(event.target.value); setPage(0)}}><option value="all">All evaluations</option><option value="review">Needs review</option><option value="reviewed">Reviewed</option><option value="failed">Failed evaluations</option></select>
          </div>
          <div className="queue-items">{runs.map(item => <button key={item.id} onClick={() => select(item.id)} className={'queue-item ' + (item.id === id ? 'selected' : '')} aria-current={item.id === id ? 'true' : undefined}>
            <span className="queue-row"><span>#{item.external_id}</span>{item.id === id && <ChevronRight size={15}/>}</span>
            <strong>{item.subject}</strong>
            {item.outcome_status && <span className="queue-outcome">{outcomes[item.outcome_status]}</span>}
            <span className="queue-row bottom"><span>{categories[item.category] || item.category}</span>{status(item)}</span>
          </button>)}</div>
          {!runs.length && <div className="empty compact"><Inbox size={22}/><p>{search || filter !== 'all' ? 'No matching tickets' : 'The queue is empty'}</p></div>}
          {total > 100 && <footer className="panel-footer pagination">
            <button className="icon-button" aria-label="Previous page" disabled={page === 0} onClick={() => setPage(page - 1)}><ChevronLeft size={17}/></button>
            <span>{page + 1} / {Math.ceil(total / 100)}</span>
            <button className="icon-button" aria-label="Next page" disabled={(page + 1) * 100 >= total} onClick={() => setPage(page + 1)}><ChevronRight size={17}/></button>
          </footer>}
        </aside>
        {selected ? <>
          <section className="panel conversation">
            <header className="ticket-head">
              <div className="ticket-top"><div><button className="icon-button mobile-back" aria-label="Back to queue" onClick={() => setMobileDetail(false)}><ArrowLeft size={17}/></button><span>#{selected.external_id}</span>{status(selected)}</div>
                <div><button className="icon-button rerun" aria-label="Download evaluation" title="Download evaluation" onClick={() => exportRun(selected)}><ArrowDownToLine size={16}/></button><button className="icon-button rerun" title="New evaluation" aria-label="New evaluation" onClick={() => setModal('rerun')}><RotateCcw size={16}/></button></div>
              </div>
              <h2>{selected.subject}</h2>
              <div className="ticket-meta"><span><MessageCircle size={14}/>{channels[selected.channel] || selected.channel}</span><span>{categories[selected.category] || selected.category}</span><span>{scopeLabel}</span></div>
            </header>
            <div className="tabs">
              <button className={tab === 'messages' ? 'active' : ''} onClick={() => setTab('messages')}><MessageCircle size={15}/> Conversation <span>{selected.ticket.messages.length}</span></button>
              <button className={tab === 'events' ? 'active' : ''} onClick={() => setTab('events')}><Activity size={15}/> Events <span>{selected.ticket.events.length}</span></button>
            </div>
            {tab === 'messages' ? <div className="timeline">
              <div className="timeline-date"><span/>{date.format(new Date(selected.ticket.messages[0].at))}<span/></div>
              {selected.ticket.messages.map(message => <article id={'message-' + message.id} key={message.id} className={`message ${message.role} ${highlight === 'message-' + message.id ? 'highlight' : ''}`}>
                <span className="avatar">{message.role === 'bot' ? <Bot size={17}/> : message.role === 'agent' ? message.author.split(' ').slice(0, 2).map(word => word[0]).join('') : <UserRound size={17}/>}</span>
                <div className="message-content"><header><strong>{message.author}</strong><time>{time.format(new Date(message.at))}</time>{!message.public && <Badge>Internal note</Badge>}</header><p>{message.text}</p></div>
              </article>)}
            </div> : <div className="timeline">
              {selected.ticket.refund && <div className="refund-context">
                <strong>Refund verification details</strong>
                <dl>
                  <div><dt>Decision</dt><dd>{refundDecisions[selected.ticket.refund.decision] || selected.ticket.refund.decision}</dd></div>
                  <div><dt>Due date</dt><dd>{selected.ticket.refund.due_at ? dateTime.format(new Date(selected.ticket.refund.due_at)) : 'Not specified'}</dd></div>
                  <div><dt>Transaction</dt><dd>{selected.ticket.refund.operation_id || 'Not specified'}</dd></div>
                  <div><dt>Owner</dt><dd>{selected.ticket.messages.find(m => m.actor_id === selected.ticket.refund?.owner_id)?.author || selected.ticket.refund.owner_id || 'Unassigned'}</dd></div>
                  {selected.ticket.observed_at && <div><dt>Snapshot time</dt><dd>{dateTime.format(new Date(selected.ticket.observed_at))}</dd></div>}
                </dl>
                {selected.ticket.refund.completion_claim_message_id && <button className="inline-link" onClick={() => {setTab('messages'); setHighlight('message-' + selected.ticket.refund!.completion_claim_message_id)}}><Link2 size={13}/> Refund completion claim</button>}
              </div>}
              {selected.ticket.events.map(event => <article id={'event-' + event.id} key={event.id} className={'message system ' + (highlight === 'event-' + event.id ? 'highlight' : '')}><span className="avatar"><Activity size={17}/></span><div className="message-content"><header><strong>{event.kind}</strong><time>{time.format(new Date(event.at))}</time></header><p>{event.detail}</p></div></article>)}
              {!selected.ticket.events.length && <div className="empty compact"><Activity size={24}/><p>No events recorded</p></div>}
            </div>}
            <footer className="panel-footer ticket-footer"><span><UserRound size={15}/> {participant?.author || 'Support participant'}</span><span>{tab === 'messages' ? (selected.ticket.messages_complete ? 'Complete conversation history' : 'Conversation history may be incomplete') : (selected.ticket.events_complete ? 'Complete event log' : 'Event log may be incomplete')}</span></footer>
          </section>
          <Criteria run={selected} view={reviewTab} setView={setReviewTab} evidence={showEvidence} review={() => setModal('review')}/>
        </> : <section className="panel blank-detail empty"><CircleHelp size={30}/><h2>{id ? 'Loading ticket' : 'Select a ticket'}</h2><p>{id ? 'Loading the conversation and evaluation results.' : 'Import a file or open a ticket from the queue.'}</p></section>}
      </div>}
    </main>
    {locked && <Modal title="Sign in to CHECK" close={() => {}}>
      <form className="form" onSubmit={submitToken}><label>Access token<input type="password" required value={token} onChange={event => setToken(event.target.value)}/></label><button className="button primary">Sign in</button></form>
    </Modal>}
    {modal === 'import' && <ImportDialog close={() => setModal(undefined)} imported={added}/>}
    {modal === 'review' && selected && <ReviewDialog run={selected} close={() => setModal(undefined)} saved={() => {
      setModal(undefined); setReviewTab('reviews'); setNotice('Decision saved'); void refresh(); void detail(selected.id)
    }}/>}
    {modal === 'rerun' && selected && <RerunDialog run={selected} close={() => setModal(undefined)} queued={added}/>}
    {modal === 'settings' && settings && !locked && <Modal title="Model settings" close={() => setModal(undefined)}>
      <EvaluationSettings profile={settings.evaluation} close={() => setModal(undefined)} save={evaluation => {
        setSettings(old => old && {...old, evaluation, model: evaluation[evaluation.mode].model})
        setModal(undefined)
        setNotice('Model settings saved for new evaluations.')
      }}/>
    </Modal>}
    {knowledge && <Modal title={knowledgeDocument?.title || 'Evidence source'} close={() => setKnowledge(undefined)}>
      <div className="form"><Badge>Version {knowledgeDocument?.version ?? '—'}</Badge><blockquote>{knowledge.quote}</blockquote>{knowledgeDocument && <p>{knowledgeDocument.text}</p>}</div>
    </Modal>}
  </div>
}

createRoot(document.getElementById('root')!).render(<App/>)
