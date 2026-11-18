import {useEffect, useState, type FormEvent} from 'react'
import {BookOpen, Search} from 'lucide-react'
import {request, type KnowledgePreview, type KnowledgeVersion, type Run, type Summary} from './api'
import {KnowledgeBundleImport} from './KnowledgeBundleImport'

const dateTime = new Intl.DateTimeFormat('en', {dateStyle: 'medium', timeStyle: 'short'})
const date = (value: string) => dateTime.format(new Date(value))
const errorText = (error: unknown) => error instanceof Error ? error.message : 'Request failed.'
const pageSize = 50

export function Knowledge({selected}: {selected?: Run | null}) {
  const [page, setPage] = useState(0)
  const [versions, setVersions] = useState<{items: KnowledgeVersion[]; total: number}>()
  const [revision, setRevision] = useState(0)
  const [libraryError, setLibraryError] = useState('')
  const [search, setSearch] = useState(selected?.external_id ?? '')
  const [matches, setMatches] = useState<Summary[]>([])
  const [runId, setRunId] = useState(selected?.id ?? '')
  const [preview, setPreview] = useState<KnowledgePreview>()
  const [replyId, setReplyId] = useState('')
  const [previewError, setPreviewError] = useState('')
  const [searching, setSearching] = useState(false)
  const [previewing, setPreviewing] = useState(false)

  useEffect(() => {
    let active = true
    request<{items: KnowledgeVersion[]; total: number}>(`/knowledge?limit=${pageSize}&offset=${page * pageSize}`)
      .then(data => {if (active) {setVersions(data); setLibraryError('')}})
      .catch(error => {if (active) setLibraryError(errorText(error))})
    return () => {active = false}
  }, [page, revision])

  function versionPublished() {
    setPage(0); setRevision(old => old + 1); setPreview(undefined); setReplyId('')
  }

  async function findTicket(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setSearching(true); setPreviewError(''); setPreview(undefined); setReplyId('')
    try {
      const {items} = await request<{items: Summary[]}>(`/runs?limit=50&search=${encodeURIComponent(search.trim())}`)
      setMatches(items); setRunId(items[0]?.id ?? '')
      if (!items.length) setPreviewError('No tickets match this search.')
    } catch (error) {setMatches([]); setRunId(''); setPreviewError(errorText(error))}
    finally {setSearching(false)}
  }

  async function loadRules() {
    setPreviewing(true); setPreview(undefined); setReplyId(''); setPreviewError('')
    try {setPreview(await request<KnowledgePreview>(`/knowledge/preview?run_id=${encodeURIComponent(runId)}`))}
    catch (error) {setPreviewError(errorText(error))}
    finally {setPreviewing(false)}
  }

  const visibleRules = preview?.items.filter(item => !replyId || item.message_id === replyId) ?? []
  const replies = new Map(preview?.replies.map(item => [item.id, item]))
  const selectedReply = replies.get(replyId)
  return <div className="knowledge-layout">
    <section className="panel knowledge-library" aria-label="Instruction library">
      <header className="panel-heading"><div className="panel-title"><BookOpen size={18}/><h2>Instruction library</h2></div>{versions && <span>{versions.total} versions</span>}</header>
      <div className="knowledge-content">
        <KnowledgeBundleImport onPublished={versionPublished} disabled={previewing}/>
        {libraryError && <p className="form-error" role="alert">{libraryError}</p>}
        {!libraryError && !versions && <p role="status">Loading instructions…</p>}
        {versions?.items.length === 0 && <p className="knowledge-empty">No instruction versions published.</p>}
        <ul className="knowledge-versions">{versions?.items.map(version => <li key={version.id}>
          <strong>{version.title}</strong><span>{version.document} · v{version.version}</span>
          <span>{version.product} · {version.category} · {version.language} · {version.fragment_count} fragments</span>
          <span>Effective {date(version.effective_from)}{version.effective_to && ` – ${date(version.effective_to)}`}</span>
          <span>Available {date(version.available_from)}{version.available_to && ` – ${date(version.available_to)}`}</span>
          {version.audience.length > 0 && <span>Audience {version.audience.join(', ')}</span>}
        </li>)}</ul>
        {versions && versions.total > pageSize && <div className="knowledge-pagination">
          <button className="button secondary small" disabled={!page} onClick={() => setPage(old => old - 1)}>Previous</button>
          <button className="button secondary small" disabled={(page + 1) * pageSize >= versions.total} onClick={() => setPage(old => old + 1)}>Next</button>
        </div>}
      </div>
    </section>
    <section className="panel" aria-label="Rule preview">
      <header className="panel-heading"><div className="panel-title"><Search size={18}/><h2>Rule preview</h2></div></header>
      <div className="knowledge-content">
        <form className="form" onSubmit={findTicket}>
          <label>Find a ticket<input maxLength={120} value={search} disabled={searching || previewing}
            placeholder="Ticket ID or subject" onChange={event => setSearch(event.target.value)}/></label>
          <button className="button secondary" disabled={searching || previewing}>{searching ? 'Searching…' : 'Search'}</button>
        </form>
        <div className="form">
          <label>Evaluation<select value={runId} disabled={searching || previewing}
            onChange={event => {setRunId(event.target.value); setPreview(undefined); setReplyId(''); setPreviewError('')}}>
            <option value="">Select an evaluation</option>
            {selected && !matches.some(item => item.id === selected.id) && <option value={selected.id}>{selected.external_id} · {selected.subject}</option>}
            {matches.map(item => <option key={item.id} value={item.id}>{item.external_id} · {item.subject} · {date(item.created_at)} · {item.scope.kind}</option>)}
          </select></label>
          <button className="button primary" disabled={!runId || previewing || searching} onClick={loadRules}>{previewing ? 'Finding rules…' : 'Preview matched rules'}</button>
          <p>Uses the current library and the evaluation’s original scope and reply dates.</p>
          {previewError && <div className="form-error" role="alert">{previewError}</div>}
          {preview && <label>Support reply<select value={replyId} onChange={event => setReplyId(event.target.value)}>
            <option value="">All support replies</option>
            {preview.replies.map(reply => <option key={reply.id} value={reply.id}>{reply.author} · {date(reply.at)} · #{reply.id}</option>)}
          </select></label>}
          {selectedReply && <blockquote>{selectedReply.text}</blockquote>}
          {preview && !visibleRules.length && <p role="status">{preview.replies.length ? 'No applicable policy fragments matched.' : 'No public support replies in this scope.'}</p>}
        </div>
        {visibleRules.length > 0 && <ul className="knowledge-versions">{visibleRules.map(item => <li key={item.ref}>
          <strong>{item.title} · v{item.version}</strong><span>Reply #{item.message_id}{replies.get(item.message_id) && ` · ${date(replies.get(item.message_id)!.at)}`}</span>
          {item.search_mode && <span>{item.search_mode === 'section' ? 'Index section' : 'Whole library fallback'}</span>}
          <p>{item.text}</p>
        </li>)}</ul>}
      </div>
    </section>
  </div>
}
