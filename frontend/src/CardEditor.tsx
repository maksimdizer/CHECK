import {useState} from 'react'
import {ArrowDown, ArrowUp, Plus, Save, Trash2, X} from 'lucide-react'
import {cardGroupWeight, formatDuration, formatWeight, request, timingNames, type Card, type CardGroup, type Definition, type TimingMetric, type TimingTarget} from './api'

const rubricLabels: Record<string, string> = {pass: 'Pass', partial: 'Partial', fail: 'Fail', not_applicable: 'Not applicable', insufficient_evidence: 'Insufficient evidence'}
const newId = (prefix: string) => prefix + '_' + crypto.randomUUID().replaceAll('-', '')

export function CardEditor({card, saved, close}: {card: Card; saved: (card: Card) => void; close: () => void}) {
  const [draft, setDraft] = useState<Card>(() => structuredClone(card))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dirty, setDirty] = useState(false)
  const [expanded, setExpanded] = useState<string[]>([])
  const [groupToDelete, setGroupToDelete] = useState<string>()
  const [channelNames, setChannelNames] = useState<Partial<Record<TimingMetric, string>>>({})
  const groups = draft.groups || []
  const total = groups.reduce((sum, group) => sum + cardGroupWeight(group, draft.criteria), 0)
  const update = (next: Card) => {setDraft(next); setDirty(true); setError('')}
  const changeGroup = (id: string, patch: Partial<CardGroup>) => update({
    ...draft, groups: groups.map(group => group.id === id ? {...group, ...patch} : group),
  })
  const changeCriterion = (id: string, patch: Partial<Definition>) => update({...draft, criteria: draft.criteria.map(item => item.id === id ? {...item, ...patch} : item)})
  const deleteCriterion = (id: string) => {
    setExpanded(old => old.filter(item => item !== id))
    update({...draft, criteria: draft.criteria.filter(item => item.id !== id)})
  }
  const deleteGroup = (id: string) => {
    const criteria = draft.criteria.filter(item => item.group_id !== id)
    setExpanded(old => old.filter(item => criteria.some(criterion => criterion.id === item)))
    setGroupToDelete(undefined)
    update({...draft, groups: groups.filter(group => group.id !== id), criteria})
  }
  const changeTiming = (id: TimingMetric, patch: Partial<TimingTarget>) => update({
    ...draft, timing_targets: {...draft.timing_targets, [id]: {...draft.timing_targets?.[id], ...patch}},
  })
  const addCriterion = (groupId: string) => {
    const id = newId('custom')
    setExpanded(old => [...old, id])
    update({...draft, criteria: [...draft.criteria, {
      id, group_id: groupId, version: 1, name: 'New criterion',
      description: '', applicability: 'Public support reply', evaluator_type: 'llm',
      weight: 5, severity: 'standard', review_policy: 'on_fail', required_evidence: ['message'],
      rubric: {pass: '', partial: '', fail: '', not_applicable: '', insufficient_evidence: 'There is insufficient evidence to make an assessment.'},
    }]})
  }
  const moveGroup = (index: number, direction: number) => {
    const next = [...groups]
    ;[next[index], next[index + direction]] = [next[index + direction], next[index]]
    update({...draft, groups: next})
  }
  const save = async () => {
    if (busy) return
    if (!draft.name.trim() || groups.some(group => !group.name.trim())) {setError('Enter a name for the scorecard and each group.'); return}
    const invalid = draft.criteria.find(item => !item.name.trim() || !item.description.trim() || !item.applicability.trim() || !item.rubric.pass?.trim() || !item.rubric.fail?.trim() || !Number.isFinite(item.weight) || item.weight < 0 || item.weight > 1000)
    if (invalid) {setError(`Enter a name, description, applicability, and Pass and Fail rules for “${invalid.name}”. Its weight must be between 0 and 1,000.`); return}
    if (groups.some(({weight = 0}) => !Number.isFinite(weight) || weight < 0 || weight > 40000)) {
      setError('Each group weight must be between 0 and 40,000.'); return
    }
    if (total <= 0) {setError('Give at least one group a positive weight.'); return}
    const emptyWeightedGroup = groups.find(group => group.weight && !draft.criteria.some(item => item.group_id === group.id && item.weight > 0))
    if (emptyWeightedGroup) {
      setError(`Add a criterion with a positive weight to “${emptyWeightedGroup.name}” or set the group weight to 0.`); return
    }
    const targets = Object.values(draft.timing_targets ?? {}).flatMap(target => [
      target.default_seconds, ...Object.values(target.channel_seconds ?? {}),
    ]).filter(value => value != null)
    if (targets.some(value => !Number.isInteger(value) || value < 1 || value > 31536000)) {
      setError('Time targets must be whole seconds between 1 and 31,536,000.'); return
    }
    setBusy(true); setError('')
    try {
      const value = await request<Card>('/card', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
        base_version: card.version, name: draft.name, groups, timing_targets: draft.timing_targets,
        criteria: draft.criteria.map(item => ({...item, rubric: Object.fromEntries(Object.entries(item.rubric).filter(([, value]) => value.trim()))})),
      })})
      saved(value)
    } catch (err) {setError(err instanceof Error ? err.message : 'Could not save the scorecard')}
    finally {setBusy(false)}
  }
  return <section className="panel card-editor" aria-label="Scorecard editor">
    <header className="editor-header"><div><h2>Edit scorecard</h2><p>Version {card.version}{dirty && ' · Unsaved changes'}</p></div><div className="editor-actions">
      <button className="button secondary" onClick={close} disabled={busy}><X size={15}/> Cancel</button>
      <button className="button primary" onClick={save} disabled={busy || !dirty}><Save size={15}/> {busy ? 'Saving…' : 'Save version'}</button>
    </div></header>
    <fieldset disabled={busy} className="editor-content">
      <label className="editor-field">Scorecard name<input value={draft.name} maxLength={120} onChange={event => update({...draft, name: event.target.value})}/></label>
      <p className="editor-help">Group weights determine each group’s share of the scorecard. Criterion weights distribute that share within the group. Groups and criteria with a weight of 0 are evaluated separately from the score.</p>
      <div className="editor-totals"><strong>{groups.length} groups · {draft.criteria.length} criteria</strong><span>Total group weight: {formatWeight(total)}</span></div>
      {error && <div className="form-error" role="alert">{error}</div>}
      <section className="editor-group timing-settings">
        <h3>Timing targets</h3>
        <p className="editor-help">Targets use calendar time. A channel-specific target takes priority. Timing is tracked separately from the quality score.</p>
        {(Object.keys(timingNames) as TimingMetric[]).map(id => {
          const target = draft.timing_targets?.[id] ?? {}
          return <div key={id} className="editor-row">
            <label className="editor-field">{timingNames[id]} default target (seconds)
              <input type="number" min={1} max={31536000} step={1} placeholder="No default" value={target.default_seconds ?? ''}
                onChange={event => changeTiming(id, {default_seconds: event.target.value === '' ? null : Number(event.target.value)})}/>
              {target.default_seconds ? <small>{formatDuration(target.default_seconds)}</small> : null}
            </label>
            {Object.entries(target.channel_seconds ?? {}).map(([channel, seconds]) => <div key={channel} className="editor-field">
              <label className="editor-field">{timingNames[id]} · {channel} target (seconds)
                <input type="number" min={1} max={31536000} step={1} value={seconds || ''} onChange={event =>
                  changeTiming(id, {channel_seconds: {...target.channel_seconds, [channel]: Number(event.target.value)}})}/>
              </label>
              <button type="button" className="button secondary small" onClick={() => {
                const channels = {...target.channel_seconds}
                delete channels[channel]
                changeTiming(id, {channel_seconds: channels})
              }}>Remove {channel} override</button>
            </div>)}
            <div className="editor-field">
              <label className="editor-field">{timingNames[id]} override channel
                <input maxLength={120} value={channelNames[id] ?? ''} placeholder="e.g. email"
                  onChange={event => setChannelNames({...channelNames, [id]: event.target.value})}/>
              </label>
              <button type="button" className="button secondary small" disabled={!channelNames[id]?.trim() || Object.hasOwn(target.channel_seconds ?? {}, channelNames[id]!.trim())}
                onClick={() => {
                  const channel = channelNames[id]?.trim()
                  if (!channel) return
                  changeTiming(id, {channel_seconds: {...target.channel_seconds, [channel]: target.default_seconds ?? 0}})
                  setChannelNames({...channelNames, [id]: ''})
                }}>Add channel override</button>
            </div>
          </div>
        })}
      </section>
      {groups.map((group, index) => {
        const items = draft.criteria.filter(item => item.group_id === group.id)
        const weight = cardGroupWeight(group, draft.criteria)
        return <section className="editor-group" key={group.id}>
          <div className="editor-group-head">
            <label className="editor-field">Group name<input aria-label={`Group name ${index + 1}`} value={group.name} maxLength={120} onChange={event => changeGroup(group.id, {name: event.target.value})}/></label>
            <label className="editor-field editor-group-weight">Group weight<input aria-label={`Group weight ${index + 1}`} type="number" min={0} max={40000} step="any" value={weight} onChange={event => changeGroup(group.id, {weight: Number(event.target.value)})}/></label>
            <span className="group-weight">{formatWeight(total > 0 ? 100 * weight / total : 0)}% of scorecard</span>
            <div className="editor-actions">
              <button className="icon-button" title="Move up" aria-label={`Move group ${index + 1} up`} disabled={index === 0} onClick={() => moveGroup(index, -1)}><ArrowUp size={16}/></button>
              <button className="icon-button" title="Move down" aria-label={`Move group ${index + 1} down`} disabled={index === groups.length - 1} onClick={() => moveGroup(index, 1)}><ArrowDown size={16}/></button>
              <button className="icon-button" title="Delete group" aria-label={`Delete group ${group.name || index + 1}`} disabled={groups.length === 1} onClick={() => items.length ? setGroupToDelete(group.id) : deleteGroup(group.id)}><Trash2 size={16}/></button>
            </div>
          </div>
          {groupToDelete === group.id && <div className="editor-delete-group" role="group" aria-label="Confirm group deletion">
            <p>Delete “{group.name}” and its {items.length} {items.length === 1 ? 'criterion' : 'criteria'} from this draft?</p>
            <button type="button" className="button secondary small" onClick={() => setGroupToDelete(undefined)}>Keep group</button>
            <button type="button" className="button danger small" onClick={() => deleteGroup(group.id)}>Delete group and criteria</button>
          </div>}
          {items.map(item => <div className="editor-criterion-row" key={item.id}>
            <details className="editor-criterion" open={expanded.includes(item.id)}>
            <summary onClick={event => {event.preventDefault(); setExpanded(old => old.includes(item.id) ? old.filter(id => id !== item.id) : [...old, item.id])}}><strong>{item.name || 'Untitled'}</strong><span>Weight {item.weight}{item.severity === 'critical' && ' · critical'}</span></summary>
            <div className="editor-fields">
              <label className="editor-field">Criterion name<input value={item.name} maxLength={120} onChange={event => changeCriterion(item.id, {name: event.target.value})}/></label>
              <div className="editor-row">
                <label className="editor-field">Weight within group<input type="number" min={0} max={1000} step="any" value={item.weight} onChange={event => changeCriterion(item.id, {weight: Number(event.target.value)})}/></label>
                <label className="editor-field">Group<select value={item.group_id} onChange={event => changeCriterion(item.id, {group_id: event.target.value})}>{groups.map(group => <option key={group.id} value={group.id}>{group.name}</option>)}</select></label>
              </div>
              <div className="editor-row">
                <label className="editor-field">Evaluation method<select disabled={item.evaluator_type === 'code'} value={item.evaluator_type} onChange={event => changeCriterion(item.id, {evaluator_type: event.target.value, required_evidence: event.target.value === 'hybrid' ? ['message', 'knowledge'] : ['message']})}>
                  {item.evaluator_type === 'code' ? <option value="code">Built-in check</option> : <><option value="llm">Conversation</option><option value="hybrid">Conversation and knowledge base</option></>}
                </select></label>
                <label className="editor-field">Severity<select value={item.severity} onChange={event => changeCriterion(item.id, {severity: event.target.value})}><option value="standard">Standard</option><option value="critical">Critical</option></select></label>
              </div>
              {item.evaluator_type === 'code' && <p className="editor-help">This built-in check defines its own description and rubric. You can edit its name, group, weight, and severity.</p>}
              <label className="editor-field">What to evaluate<textarea rows={3} maxLength={3000} disabled={item.evaluator_type === 'code'} value={item.description} onChange={event => changeCriterion(item.id, {description: event.target.value})}/></label>
              <label className="editor-field">When it applies<textarea rows={2} maxLength={1500} disabled={item.evaluator_type === 'code'} value={item.applicability} onChange={event => changeCriterion(item.id, {applicability: event.target.value})}/></label>
              {Object.entries(rubricLabels).map(([key, label]) => <label className="editor-field" key={key}>{label}{['pass', 'fail'].includes(key) ? ' *' : ' · optional'}<textarea rows={2} maxLength={2000} disabled={item.evaluator_type === 'code'} value={item.rubric[key] || ''} onChange={event => changeCriterion(item.id, {rubric: {...item.rubric, [key]: event.target.value}})}/></label>)}
              <label className="editor-field">Human review<select value={item.review_policy} onChange={event => changeCriterion(item.id, {review_policy: event.target.value})}><option value="on_fail">On Fail or Partial</option><option value="always">Always</option></select></label>
            </div>
            </details>
            <button type="button" className="icon-button editor-delete-criterion" title="Delete criterion" aria-label={`Delete criterion ${item.name || 'Untitled'}`} onClick={() => deleteCriterion(item.id)}><Trash2 size={16}/></button>
          </div>)}
          {!items.length && <p className="editor-help empty-group">This group has no criteria yet.</p>}
          <button className="button secondary small" disabled={draft.criteria.length >= 40} onClick={() => addCriterion(group.id)}><Plus size={15}/> Add criterion</button>
        </section>
      })}
      <button className="button secondary" disabled={groups.length >= 20} onClick={() => update({...draft, groups: [...groups, {id: newId('group'), name: 'New group', weight: 0}]})}><Plus size={16}/> Add group</button>
      <div className="editor-footer"><span>{dirty ? 'Unsaved changes' : 'Scorecard saved'}</span><button className="button primary" onClick={save} disabled={busy || !dirty}><Save size={15}/> {busy ? 'Saving…' : 'Save version'}</button></div>
    </fieldset>
  </section>
}
