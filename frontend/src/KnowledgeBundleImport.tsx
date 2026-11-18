import {useState} from 'react'
import {FolderOpen} from 'lucide-react'
import {post} from './api'

type Bundle = {index: string; documents: (Record<string, unknown> & {path: string; markdown: string})[]}
type Preview = {
  sections: {text: string; paths: string[]}[];
  documents: {path: string; document: string; version: number; fragment_count: number}[];
}

export function KnowledgeBundleImport({disabled, onPublished}: {disabled?: boolean; onPublished: () => void}) {
  const [bundle, setBundle] = useState<Bundle>()
  const [preview, setPreview] = useState<Preview>()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState('')

  async function load(files: File[]) {
    setBundle(undefined); setPreview(undefined); setError(''); setSaved(''); setBusy(true)
    try {
      const fileMap = new Map(files.map(file => [
        file.webkitRelativePath.split('/').slice(1).join('/'), file,
      ]))
      const indexFile = fileMap.get('index.md')
      const manifestFile = fileMap.get('manifest.json')
      if (!indexFile || !manifestFile) throw new Error('Choose the folder containing index.md and manifest.json.')
      const limit = 5 * 1024 * 1024
      if (indexFile.size + manifestFile.size > limit) throw new Error('The bundle exceeds 5 MB.')
      const read = async (file: File) => new TextDecoder('utf-8', {fatal: true}).decode(await file.arrayBuffer())
      const manifest = JSON.parse(await read(manifestFile))
      if (!Array.isArray(manifest?.documents) || !manifest.documents.length || manifest.documents.length > 100) {
        throw new Error('manifest.json must list 1–100 instruction versions under documents.')
      }
      const entries = []
      let total = indexFile.size + manifestFile.size
      for (const entry of manifest.documents) {
        const path = entry?.path
        if (typeof path !== 'string' || !path || !fileMap.has(path)) {
          throw new Error(`Instruction file not found: ${typeof path === 'string' ? path : '(missing path)'}`)
        }
        const file = fileMap.get(path)!
        total += file.size
        if (total > limit) throw new Error('The bundle exceeds 5 MB.')
        entries.push({entry, file})
      }
      const payload: Bundle = {
        index: await read(indexFile),
        documents: await Promise.all(entries.map(async ({entry, file}) => ({...entry, markdown: await read(file)}))),
      }
      if (new TextEncoder().encode(JSON.stringify(payload)).length > limit) throw new Error('The bundle exceeds 5 MB.')
      setPreview(await post<Preview>('/knowledge/bundle/preview', payload))
      setBundle(payload)
    } catch (exception) {setError(exception instanceof Error ? exception.message : 'Could not read the bundle.')}
    finally {setBusy(false)}
  }

  async function publish() {
    if (!bundle || busy || disabled) return
    setBusy(true); setError('')
    try {
      const result = await post<{documents: number; sections: number}>('/knowledge/bundle', bundle)
      setSaved(`Published ${result.documents} instruction versions and ${result.sections} index sections.`)
      setBundle(undefined); setPreview(undefined); onPublished()
    } catch (exception) {setError(exception instanceof Error ? exception.message : 'Could not publish the bundle.')}
    finally {setBusy(false)}
  }

  return <div className="form">
    <label className="file-field"><span><FolderOpen size={15}/> Select instruction folder</span>
      <input type="file" multiple disabled={busy || disabled} ref={node => {node?.setAttribute('webkitdirectory', '')}}
        onChange={event => {
          const files = Array.from(event.target.files ?? [])
          event.target.value = ''
          if (files.length) void load(files)
        }}/></label>
    <p>Choose a folder with index.md, manifest.json and the linked Markdown files. Publishing replaces the routing index and retains existing instruction versions.</p>
    {busy && <p role="status">{bundle ? 'Publishing bundle…' : 'Checking links and instructions…'}</p>}
    {preview && <>
      <details className="knowledge-fragments"><summary>Index sections ({preview.sections.length})</summary>
        {preview.sections.map(section => <div key={section.text + section.paths.join(',')}>
          <p>{section.text}</p><p>{section.paths.join('\n')}</p>
        </div>)}</details>
      <details className="knowledge-fragments"><summary>Instruction versions ({preview.documents.length})</summary>
        {preview.documents.map(doc => <div key={doc.path}>
          <strong>{doc.path}</strong><p>{doc.document} · v{doc.version} · {doc.fragment_count} fragments</p>
        </div>)}</details>
      <button className="button primary" disabled={busy || disabled} onClick={publish}>Publish bundle</button>
    </>}
    {saved && <p role="status">{saved}</p>}
    {error && <div className="form-error" role="alert">{error}</div>}
  </div>
}
