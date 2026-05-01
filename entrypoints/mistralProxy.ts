/**
 * CLI entrypoint that launches the Mistral compatibility proxy.
 *
 * Usage:
 *   node dist/entrypoints/mistralProxy.js
 *
 * Environment:
 *   MISTRAL_API_KEY        Required. API key for api.mistral.ai.
 *   MISTRAL_BASE_URL       Optional. Default https://api.mistral.ai
 *   MISTRAL_MODEL          Optional. Default mistral-large-latest. Used as the
 *                          target for non-Haiku Claude model names.
 *   MISTRAL_SMALL_MODEL    Optional. Default mistral-small-latest. Used when the
 *                          incoming Claude model name contains "haiku".
 *   MISTRAL_PROXY_PORT     Optional. Default 8787.
 *   MISTRAL_PROXY_HOST     Optional. Default 127.0.0.1.
 *   MISTRAL_PROXY_VERBOSE  Optional. Set truthy to log requests to stderr.
 *
 * Then point Claude Code at it:
 *   ANTHROPIC_BASE_URL=http://127.0.0.1:8787 \
 *   ANTHROPIC_API_KEY=sk-not-used \
 *   claude
 */

import { defaultModelMap, startMistralProxy } from '../services/api/mistralProxy.js'

function isTruthy(v: string | undefined): boolean {
  return !!v && !['0', 'false', 'no', ''].includes(v.toLowerCase())
}

async function main(): Promise<void> {
  const apiKey = process.env.MISTRAL_API_KEY
  if (!apiKey) {
    // biome-ignore lint/suspicious/noConsole: CLI entrypoint
    console.error('error: MISTRAL_API_KEY is required')
    process.exit(2)
  }

  const port = Number(process.env.MISTRAL_PROXY_PORT ?? 8787)
  const host = process.env.MISTRAL_PROXY_HOST ?? '127.0.0.1'
  const baseUrl = process.env.MISTRAL_BASE_URL ?? 'https://api.mistral.ai'
  const verbose = isTruthy(process.env.MISTRAL_PROXY_VERBOSE)

  const handle = await startMistralProxy({
    port,
    host,
    mistralBaseUrl: baseUrl,
    mistralApiKey: apiKey,
    modelMap: defaultModelMap,
    log: verbose
      ? // biome-ignore lint/suspicious/noConsole: CLI entrypoint
        msg => console.error(`[mistral-proxy] ${msg}`)
      : undefined,
  })

  const shutdown = async (): Promise<void> => {
    try {
      await handle.close()
    } finally {
      process.exit(0)
    }
  }
  process.on('SIGINT', shutdown)
  process.on('SIGTERM', shutdown)

  // biome-ignore lint/suspicious/noConsole: CLI entrypoint
  console.error(
    `Mistral proxy ready on http://${host}:${port}\n` +
      `Use it with: ANTHROPIC_BASE_URL=http://${host}:${port} ANTHROPIC_API_KEY=sk-not-used claude`,
  )
}

main().catch(err => {
  // biome-ignore lint/suspicious/noConsole: CLI entrypoint
  console.error(err)
  process.exit(1)
})
