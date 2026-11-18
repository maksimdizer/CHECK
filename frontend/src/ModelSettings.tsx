import {useState, type FormEvent} from 'react'
import {CheckCircle2, Cloud, Monitor, Save} from 'lucide-react'
import {request, type CloudProvider, type EvaluationProfile, type LocalModel} from './api'

const cloudProviders: Record<CloudProvider, {name: string; modelHint: string}> = {
  openai: {name: 'OpenAI', modelHint: 'Model ID from OpenAI'},
  openrouter: {name: 'OpenRouter', modelHint: 'provider/model'},
  anthropic: {name: 'Anthropic (Claude)', modelHint: 'Claude model ID from Anthropic'},
  google: {name: 'Google Gemini', modelHint: 'Gemini model ID from Google AI Studio'},
}

const localModels = [
  {id: 'gemma4:12b-it-4bit-mlx', label: 'Gemma 4 12B IT · 4-bit MLX'},
  {id: 'gemma4:26b-a4b-3bit-mlx', label: 'Gemma 4 26B A4B · 3-bit MLX'},
]

export function EvaluationSettings({profile, save, close}: {
  profile: EvaluationProfile; save: (profile: EvaluationProfile) => void; close: () => void
}) {
  const [mode, setMode] = useState(profile.mode)
  const [local, setLocal] = useState(profile.local)
  const [customModel, setCustomModel] = useState(
    profile.local.model && !localModels.some(model => model.id === profile.local.model) ? profile.local.model : '',
  )
  const [modelChoice, setModelChoice] = useState(
    localModels.some(model => model.id === profile.local.model) ? profile.local.model : 'custom',
  )
  const [cloud, setCloud] = useState(profile.cloud)
  const [key, setKey] = useState('')
  const [busy, setBusy] = useState<'save' | 'test'>()
  const [feedback, setFeedback] = useState<{ok: boolean; message: string}>()
  const cloudHasKey = profile.configured_cloud_providers.includes(cloud.provider)

  const updateLocal = (patch: Partial<LocalModel>) => {
    setLocal(current => ({...current, ...patch}))
    setFeedback(undefined)
  }
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (busy) return
    const testing = (event.nativeEvent as SubmitEvent).submitter?.getAttribute('value') === 'test'
    setBusy(testing ? 'test' : 'save')
    setFeedback(undefined)
    const body = JSON.stringify({mode, local, cloud, ...(mode === 'cloud' && key.trim() ? {api_key: key.trim()} : {})})
    const headers = {'Content-Type': 'application/json'}
    try {
      if (testing) {
        await request('/settings/model/test', {method: 'POST', headers, body})
        setFeedback({ok: true, message: 'Model connection verified.'})
      } else {
        save(await request<EvaluationProfile>('/settings/model', {method: 'PUT', headers, body}))
      }
    } catch (error) {
      setFeedback({ok: false, message: error instanceof Error ? error.message : 'The request failed.'})
    } finally {
      setBusy(undefined)
    }
  }

  return <form className="form model-settings" onSubmit={submit}>
    <fieldset disabled={!!busy}>
      <legend>Evaluation mode</legend>
      <div className="mode-options">
        <label><input type="radio" name="model-mode" value="local" checked={mode === 'local'}
          onChange={() => {setMode('local'); setFeedback(undefined)}}/><Monitor size={17}/> Local</label>
        <label><input type="radio" name="model-mode" value="cloud" checked={mode === 'cloud'}
          onChange={() => {setMode('cloud'); setFeedback(undefined)}}/><Cloud size={17}/> Cloud</label>
      </div>

      {mode === 'local' ? <>
        <label>Local model
          <select value={modelChoice} onChange={event => {
            const choice = event.target.value
            setModelChoice(choice)
            updateLocal({model: choice === 'custom' ? customModel : choice})
          }}>
            {localModels.map(model => <option key={model.id} value={model.id}>{model.label}</option>)}
            <option value="custom">Custom model</option>
          </select>
        </label>
        {modelChoice === 'custom' && <label>Local model ID
          <input maxLength={200} value={customModel} placeholder="Model ID on your local server"
            onChange={event => {
              setCustomModel(event.target.value)
              updateLocal({model: event.target.value})
            }}/>
        </label>}
        <div className="model-fields">
          <label>Output tokens
            <input type="number" required min={1} max={32768} value={local.output_tokens || ''}
              onChange={event => updateLocal({output_tokens: Number(event.target.value)})}/>
          </label>
          <label>Context tokens
            <input type="number" min={512} max={262144} placeholder="Server default" value={local.context_tokens ?? ''}
              onChange={event => updateLocal({context_tokens: event.target.value ? Number(event.target.value) : null})}/>
          </label>
          <label>MLX reasoning tokens
            <input type="number" min={1} max={65536} placeholder="Server default" value={local.reasoning_tokens ?? ''}
              onChange={event => updateLocal({reasoning_tokens: event.target.value ? Number(event.target.value) : null})}/>
          </label>
        </div>
        <p className="model-note">For MLX, use a model ID registered on the local server and limits within its launch settings.</p>
        {!local.agent_mode && <label>Reasoning
          <select value={local.think == null ? 'default' : String(local.think)}
            onChange={event => updateLocal({think: event.target.value === 'default' ? null : event.target.value === 'true'})}>
            <option value="default">Server default</option><option value="true">Enabled</option><option value="false">Disabled</option>
          </select>
        </label>}
        <div className="model-toggles">
          <label><input type="checkbox" checked={local.agent_mode} onChange={event => updateLocal({
            agent_mode: event.target.checked, agent_quality: event.target.checked && local.agent_quality,
          })}/> Agent mode</label>
          <label><input type="checkbox" disabled={!local.agent_mode} checked={local.agent_quality}
            onChange={event => updateLocal({agent_quality: event.target.checked})}/> Quality mode</label>
        </div>
      </> : <>
        <label>Cloud provider
          <select value={cloud.provider} onChange={event => {
            setCloud({...cloud, provider: event.target.value as typeof cloud.provider, model: ''})
            setKey(''); setFeedback(undefined)
          }}>
            {Object.entries(cloudProviders).map(([id, provider]) => <option value={id} key={id}>{provider.name}</option>)}
          </select>
        </label>
        <label>Cloud model ID
          <input required maxLength={200} value={cloud.model} placeholder={cloudProviders[cloud.provider].modelHint}
            onChange={event => {setCloud({...cloud, model: event.target.value}); setFeedback(undefined)}}/>
        </label>
        <label>{cloudHasKey ? 'Replace API key' : 'API key'}
          <input type="password" required={!cloudHasKey} autoComplete="off" maxLength={4096} value={key}
            placeholder={cloudHasKey ? 'Leave blank to keep the saved key' : 'Paste your provider API key'}
            onChange={event => {setKey(event.target.value); setFeedback(undefined)}}/>
        </label>
        <label>Output tokens
          <input type="number" required min={16} max={32768} value={cloud.output_tokens || ''}
            onChange={event => {setCloud({...cloud, output_tokens: Number(event.target.value)}); setFeedback(undefined)}}/>
        </label>
        <p className="model-note">Choose a model with JSON output support. Cloud evaluations send conversation excerpts and relevant sources to {cloudProviders[cloud.provider].name}.</p>
      </>}
    </fieldset>
    {feedback && <div className={feedback.ok ? 'model-feedback' : 'form-error'} role={feedback.ok ? 'status' : 'alert'}>
      {feedback.ok && <CheckCircle2 size={16}/>} {feedback.message}
    </div>}
    <div className="model-actions">
      <button type="button" className="button secondary small" onClick={close} disabled={!!busy}>Cancel</button>
      <button type="submit" value="test" className="button secondary small" disabled={!!busy || !(mode === 'local' ? local.model.trim() : cloud.model.trim())}>
        {busy === 'test' ? 'Testing…' : 'Test connection'}
      </button>
      <button type="submit" className="button primary small" disabled={!!busy}><Save size={15}/>{busy === 'save' ? 'Saving…' : 'Save settings'}</button>
    </div>
  </form>
}
