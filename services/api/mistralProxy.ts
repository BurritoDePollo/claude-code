/**
 * Mistral compatibility proxy for Claude Code.
 *
 * Translates the Anthropic Messages API (`POST /v1/messages`) into Mistral's
 * OpenAI-compatible chat completions endpoint and re-encodes the response as
 * Anthropic-style SSE events. Point Claude Code at this proxy via
 * `ANTHROPIC_BASE_URL=http://127.0.0.1:<port>` and a dummy `ANTHROPIC_API_KEY`
 * (the proxy itself authenticates to Mistral with `MISTRAL_API_KEY`).
 *
 * Self-contained: only Node built-ins, no dependencies on the rest of the
 * codebase. This keeps the proxy buildable and runnable in isolation.
 */

import { randomUUID } from 'node:crypto'
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'

// ---------- Anthropic <-> Mistral type sketches ----------

type AnthropicTextBlock = { type: 'text'; text: string; cache_control?: unknown }
type AnthropicToolUseBlock = {
  type: 'tool_use'
  id: string
  name: string
  input: unknown
}
type AnthropicToolResultBlock = {
  type: 'tool_result'
  tool_use_id: string
  content: string | Array<{ type: 'text'; text: string }>
  is_error?: boolean
}
type AnthropicImageBlock = {
  type: 'image'
  source: { type: 'base64'; media_type: string; data: string } | { type: 'url'; url: string }
}
type AnthropicThinkingBlock = { type: 'thinking'; thinking: string; signature?: string }
type AnthropicContentBlock =
  | AnthropicTextBlock
  | AnthropicToolUseBlock
  | AnthropicToolResultBlock
  | AnthropicImageBlock
  | AnthropicThinkingBlock
  | { type: string; [k: string]: unknown }

type AnthropicMessage = {
  role: 'user' | 'assistant'
  content: string | AnthropicContentBlock[]
}

type AnthropicTool = {
  name: string
  description?: string
  input_schema: Record<string, unknown>
}

type AnthropicRequest = {
  model: string
  max_tokens?: number
  system?: string | Array<AnthropicTextBlock>
  messages: AnthropicMessage[]
  tools?: AnthropicTool[]
  tool_choice?:
    | { type: 'auto' | 'any' }
    | { type: 'tool'; name: string }
    | { type: 'none' }
  temperature?: number
  top_p?: number
  stop_sequences?: string[]
  stream?: boolean
  metadata?: Record<string, unknown>
}

type MistralMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | Array<{ type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }>
  tool_calls?: Array<{
    id: string
    type: 'function'
    function: { name: string; arguments: string }
  }>
  tool_call_id?: string
  name?: string
}

type MistralRequest = {
  model: string
  messages: MistralMessage[]
  max_tokens?: number
  temperature?: number
  top_p?: number
  stop?: string[]
  stream?: boolean
  tools?: Array<{
    type: 'function'
    function: { name: string; description?: string; parameters: Record<string, unknown> }
  }>
  tool_choice?: 'auto' | 'any' | 'none' | { type: 'function'; function: { name: string } }
}

// ---------- Configuration ----------

export type ProxyConfig = {
  port: number
  host: string
  mistralBaseUrl: string
  mistralApiKey: string
  /** Maps an incoming Claude model name to a Mistral model. */
  modelMap: (claudeModel: string) => string
  /** Optional logger. Defaults to noop. */
  log?: (msg: string) => void
}

const DEFAULT_MODEL_MAP: Record<string, string> = {
  // Sensible defaults — override with MISTRAL_MODEL or MISTRAL_SMALL_MODEL.
  haiku: 'mistral-small-latest',
  sonnet: 'mistral-medium-latest',
  opus: 'mistral-large-latest',
}

export function defaultModelMap(claudeModel: string): string {
  const explicit = process.env.MISTRAL_MODEL
  if (explicit && !/haiku/i.test(claudeModel)) return explicit

  const small = process.env.MISTRAL_SMALL_MODEL
  if (small && /haiku/i.test(claudeModel)) return small

  const lower = claudeModel.toLowerCase()
  for (const [needle, target] of Object.entries(DEFAULT_MODEL_MAP)) {
    if (lower.includes(needle)) return target
  }
  return explicit || 'mistral-large-latest'
}

// ---------- Request translation ----------

function flattenSystem(system: AnthropicRequest['system']): string | undefined {
  if (!system) return undefined
  if (typeof system === 'string') return system
  return system
    .filter(b => b && b.type === 'text' && typeof b.text === 'string')
    .map(b => b.text)
    .join('\n\n')
}

function blocksToMistralContent(
  blocks: AnthropicContentBlock[],
): MistralMessage['content'] {
  const parts: Array<{ type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }> = []
  for (const block of blocks) {
    if (block.type === 'text') {
      parts.push({ type: 'text', text: (block as AnthropicTextBlock).text })
    } else if (block.type === 'image') {
      const src = (block as AnthropicImageBlock).source
      if (src.type === 'url') {
        parts.push({ type: 'image_url', image_url: { url: src.url } })
      } else {
        parts.push({
          type: 'image_url',
          image_url: { url: `data:${src.media_type};base64,${src.data}` },
        })
      }
    } else if (block.type === 'thinking') {
      // Mistral has no thinking channel — drop silently.
      continue
    }
    // tool_use / tool_result are handled separately by the caller.
  }
  if (parts.length === 1 && parts[0].type === 'text') return parts[0].text
  return parts.length ? parts : ''
}

export function translateRequest(
  req: AnthropicRequest,
  modelMap: (m: string) => string,
): MistralRequest {
  const out: MistralRequest = {
    model: modelMap(req.model),
    messages: [],
    max_tokens: req.max_tokens,
    temperature: req.temperature,
    top_p: req.top_p,
    stop: req.stop_sequences,
    stream: req.stream,
  }

  const system = flattenSystem(req.system)
  if (system) out.messages.push({ role: 'system', content: system })

  for (const msg of req.messages) {
    if (typeof msg.content === 'string') {
      out.messages.push({ role: msg.role, content: msg.content })
      continue
    }

    if (msg.role === 'user') {
      // Split into tool_results (each becomes its own `tool` message) and
      // remaining content (collapsed into one user message).
      const toolResults = msg.content.filter(
        (b): b is AnthropicToolResultBlock => b.type === 'tool_result',
      )
      const otherBlocks = msg.content.filter(b => b.type !== 'tool_result')

      for (const tr of toolResults) {
        const text =
          typeof tr.content === 'string'
            ? tr.content
            : tr.content
                .filter(c => c.type === 'text')
                .map(c => c.text)
                .join('\n')
        out.messages.push({
          role: 'tool',
          tool_call_id: tr.tool_use_id,
          content: tr.is_error ? `ERROR: ${text}` : text,
        })
      }

      if (otherBlocks.length > 0) {
        out.messages.push({
          role: 'user',
          content: blocksToMistralContent(otherBlocks),
        })
      }
      continue
    }

    // assistant
    const toolUses = msg.content.filter(
      (b): b is AnthropicToolUseBlock => b.type === 'tool_use',
    )
    const textBlocks = msg.content.filter(
      (b): b is AnthropicTextBlock => b.type === 'text',
    )
    const text = textBlocks.map(b => b.text).join('')

    const assistantMsg: MistralMessage = { role: 'assistant' }
    if (text) assistantMsg.content = text
    if (toolUses.length > 0) {
      assistantMsg.tool_calls = toolUses.map(tu => ({
        id: tu.id,
        type: 'function' as const,
        function: {
          name: tu.name,
          arguments: JSON.stringify(tu.input ?? {}),
        },
      }))
    }
    out.messages.push(assistantMsg)
  }

  if (req.tools && req.tools.length > 0) {
    out.tools = req.tools.map(t => ({
      type: 'function' as const,
      function: {
        name: t.name,
        description: t.description,
        parameters: t.input_schema,
      },
    }))
  }

  if (req.tool_choice) {
    if (req.tool_choice.type === 'auto') out.tool_choice = 'auto'
    else if (req.tool_choice.type === 'any') out.tool_choice = 'any'
    else if (req.tool_choice.type === 'none') out.tool_choice = 'none'
    else if (req.tool_choice.type === 'tool')
      out.tool_choice = {
        type: 'function',
        function: { name: req.tool_choice.name },
      }
  }

  return out
}

// ---------- Response translation (non-streaming) ----------

type MistralResponse = {
  id: string
  model: string
  choices: Array<{
    index: number
    message: {
      role: 'assistant'
      content: string | null
      tool_calls?: Array<{
        id: string
        type: 'function'
        function: { name: string; arguments: string }
      }>
    }
    finish_reason: 'stop' | 'length' | 'tool_calls' | string
  }>
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number }
}

function mapStopReason(
  fr: string | undefined,
): 'end_turn' | 'max_tokens' | 'tool_use' | 'stop_sequence' {
  if (fr === 'length') return 'max_tokens'
  if (fr === 'tool_calls') return 'tool_use'
  if (fr === 'stop_sequence') return 'stop_sequence'
  return 'end_turn'
}

export function translateResponse(
  m: MistralResponse,
  originalModel: string,
): unknown {
  const choice = m.choices[0]
  const content: AnthropicContentBlock[] = []
  if (choice?.message?.content) {
    content.push({ type: 'text', text: choice.message.content })
  }
  for (const tc of choice?.message?.tool_calls ?? []) {
    let parsed: unknown = {}
    try {
      parsed = tc.function.arguments ? JSON.parse(tc.function.arguments) : {}
    } catch {
      parsed = { _raw_arguments: tc.function.arguments }
    }
    content.push({ type: 'tool_use', id: tc.id, name: tc.function.name, input: parsed })
  }
  return {
    id: `msg_${m.id}`,
    type: 'message',
    role: 'assistant',
    model: originalModel,
    content,
    stop_reason: mapStopReason(choice?.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: m.usage?.prompt_tokens ?? 0,
      output_tokens: m.usage?.completion_tokens ?? 0,
    },
  }
}

// ---------- Streaming translation ----------

type MistralStreamChunk = {
  id: string
  choices: Array<{
    index: number
    delta: {
      role?: string
      content?: string | null
      tool_calls?: Array<{
        index: number
        id?: string
        function?: { name?: string; arguments?: string }
      }>
    }
    finish_reason?: string | null
  }>
  usage?: { prompt_tokens?: number; completion_tokens?: number }
}

class StreamTranslator {
  private messageId = `msg_${randomUUID().replace(/-/g, '').slice(0, 24)}`
  private startedMessage = false
  private textBlockOpen = false
  private textBlockIndex = -1
  private toolBlocks = new Map<number, { anthropicIndex: number; sentStart: boolean }>()
  private nextBlockIndex = 0
  private finishReason: string | undefined
  private inputTokens = 0
  private outputTokens = 0
  constructor(private originalModel: string) {}

  start(): string {
    this.startedMessage = true
    const event = {
      type: 'message_start',
      message: {
        id: this.messageId,
        type: 'message',
        role: 'assistant',
        model: this.originalModel,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    }
    return sse('message_start', event)
  }

  handleChunk(chunk: MistralStreamChunk): string {
    let out = ''
    if (!this.startedMessage) out += this.start()

    if (chunk.usage) {
      if (typeof chunk.usage.prompt_tokens === 'number')
        this.inputTokens = chunk.usage.prompt_tokens
      if (typeof chunk.usage.completion_tokens === 'number')
        this.outputTokens = chunk.usage.completion_tokens
    }

    const choice = chunk.choices?.[0]
    if (!choice) return out
    const delta = choice.delta ?? {}

    if (typeof delta.content === 'string' && delta.content.length > 0) {
      if (!this.textBlockOpen) {
        this.textBlockIndex = this.nextBlockIndex++
        this.textBlockOpen = true
        out += sse('content_block_start', {
          type: 'content_block_start',
          index: this.textBlockIndex,
          content_block: { type: 'text', text: '' },
        })
      }
      out += sse('content_block_delta', {
        type: 'content_block_delta',
        index: this.textBlockIndex,
        delta: { type: 'text_delta', text: delta.content },
      })
    }

    for (const tc of delta.tool_calls ?? []) {
      let entry = this.toolBlocks.get(tc.index)
      if (!entry) {
        if (this.textBlockOpen) {
          out += sse('content_block_stop', {
            type: 'content_block_stop',
            index: this.textBlockIndex,
          })
          this.textBlockOpen = false
        }
        entry = { anthropicIndex: this.nextBlockIndex++, sentStart: false }
        this.toolBlocks.set(tc.index, entry)
      }
      if (!entry.sentStart && tc.id && tc.function?.name) {
        entry.sentStart = true
        out += sse('content_block_start', {
          type: 'content_block_start',
          index: entry.anthropicIndex,
          content_block: {
            type: 'tool_use',
            id: tc.id,
            name: tc.function.name,
            input: {},
          },
        })
      }
      const argsDelta = tc.function?.arguments
      if (entry.sentStart && argsDelta) {
        out += sse('content_block_delta', {
          type: 'content_block_delta',
          index: entry.anthropicIndex,
          delta: { type: 'input_json_delta', partial_json: argsDelta },
        })
      }
    }

    if (choice.finish_reason) this.finishReason = choice.finish_reason
    return out
  }

  finish(): string {
    let out = ''
    if (!this.startedMessage) out += this.start()
    if (this.textBlockOpen) {
      out += sse('content_block_stop', {
        type: 'content_block_stop',
        index: this.textBlockIndex,
      })
      this.textBlockOpen = false
    }
    for (const entry of this.toolBlocks.values()) {
      if (entry.sentStart) {
        out += sse('content_block_stop', {
          type: 'content_block_stop',
          index: entry.anthropicIndex,
        })
      }
    }
    out += sse('message_delta', {
      type: 'message_delta',
      delta: {
        stop_reason: mapStopReason(this.finishReason),
        stop_sequence: null,
      },
      usage: { output_tokens: this.outputTokens },
    })
    out += sse('message_stop', { type: 'message_stop' })
    return out
  }
}

function sse(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
}

// ---------- Mistral SSE parser ----------

class SseLineBuffer {
  private buf = ''
  push(chunk: string): string[] {
    this.buf += chunk
    const events: string[] = []
    let idx: number
    // SSE events are separated by a blank line.
    while ((idx = this.buf.indexOf('\n\n')) !== -1) {
      events.push(this.buf.slice(0, idx))
      this.buf = this.buf.slice(idx + 2)
    }
    return events
  }
  flush(): string[] {
    if (!this.buf.trim()) return []
    const remaining = this.buf
    this.buf = ''
    return [remaining]
  }
}

function extractDataPayload(rawEvent: string): string | null {
  const lines = rawEvent.split('\n')
  const dataLines: string[] = []
  for (const line of lines) {
    if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
  }
  if (dataLines.length === 0) return null
  return dataLines.join('\n')
}

// ---------- HTTP server ----------

async function readBody(req: IncomingMessage): Promise<string> {
  return await new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    req.on('data', c => chunks.push(c))
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

function writeJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body)
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(payload),
  })
  res.end(payload)
}

function anthropicError(
  res: ServerResponse,
  status: number,
  type: string,
  message: string,
): void {
  writeJson(res, status, {
    type: 'error',
    error: { type, message },
  })
}

async function handleMessages(
  req: IncomingMessage,
  res: ServerResponse,
  cfg: ProxyConfig,
): Promise<void> {
  const body = await readBody(req)
  let parsed: AnthropicRequest
  try {
    parsed = JSON.parse(body)
  } catch {
    return anthropicError(res, 400, 'invalid_request_error', 'Invalid JSON body')
  }

  const stream = !!parsed.stream
  const upstream = translateRequest(parsed, cfg.modelMap)

  const url = new URL('/v1/chat/completions', cfg.mistralBaseUrl)
  cfg.log?.(`-> Mistral ${upstream.model} (stream=${stream})`)

  let upstreamRes: Response
  try {
    upstreamRes = await fetch(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        accept: stream ? 'text/event-stream' : 'application/json',
        authorization: `Bearer ${cfg.mistralApiKey}`,
      },
      body: JSON.stringify(upstream),
    })
  } catch (err) {
    return anthropicError(
      res,
      502,
      'api_error',
      `Mistral request failed: ${(err as Error).message}`,
    )
  }

  if (!upstreamRes.ok) {
    const errText = await upstreamRes.text()
    cfg.log?.(`<- Mistral error ${upstreamRes.status}: ${errText.slice(0, 500)}`)
    return anthropicError(
      res,
      upstreamRes.status,
      upstreamRes.status === 429 ? 'rate_limit_error' : 'api_error',
      errText || `Mistral returned ${upstreamRes.status}`,
    )
  }

  if (!stream) {
    const json = (await upstreamRes.json()) as MistralResponse
    return writeJson(res, 200, translateResponse(json, parsed.model))
  }

  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  })

  const translator = new StreamTranslator(parsed.model)
  res.write(translator.start())

  const reader = upstreamRes.body?.getReader()
  if (!reader) {
    res.write(translator.finish())
    res.end()
    return
  }

  const decoder = new TextDecoder()
  const buf = new SseLineBuffer()

  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      const text = decoder.decode(value, { stream: true })
      for (const event of buf.push(text)) {
        const data = extractDataPayload(event)
        if (data === null) continue
        if (data === '[DONE]') continue
        try {
          const chunk = JSON.parse(data) as MistralStreamChunk
          res.write(translator.handleChunk(chunk))
        } catch (err) {
          cfg.log?.(`stream parse error: ${(err as Error).message}`)
        }
      }
    }
    for (const event of buf.flush()) {
      const data = extractDataPayload(event)
      if (data && data !== '[DONE]') {
        try {
          res.write(translator.handleChunk(JSON.parse(data) as MistralStreamChunk))
        } catch {
          /* ignore */
        }
      }
    }
  } catch (err) {
    cfg.log?.(`stream read error: ${(err as Error).message}`)
  }

  res.write(translator.finish())
  res.end()
}

export function startMistralProxy(cfg: ProxyConfig): Promise<{ close: () => Promise<void> }> {
  const server = createServer((req, res) => {
    const url = req.url ?? ''
    if (req.method === 'POST' && (url === '/v1/messages' || url.startsWith('/v1/messages?'))) {
      handleMessages(req, res, cfg).catch(err => {
        cfg.log?.(`handler error: ${(err as Error).stack ?? err}`)
        if (!res.headersSent) {
          anthropicError(res, 500, 'api_error', (err as Error).message)
        } else {
          res.end()
        }
      })
      return
    }
    if (req.method === 'GET' && url === '/healthz') {
      writeJson(res, 200, { ok: true })
      return
    }
    anthropicError(res, 404, 'not_found_error', `No route for ${req.method} ${url}`)
  })

  return new Promise(resolve => {
    server.listen(cfg.port, cfg.host, () => {
      cfg.log?.(`Mistral proxy listening on http://${cfg.host}:${cfg.port}`)
      resolve({
        close: () =>
          new Promise<void>((res, rej) =>
            server.close(err => (err ? rej(err) : res())),
          ),
      })
    })
  })
}
