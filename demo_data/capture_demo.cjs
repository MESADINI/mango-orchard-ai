/*
 * Walks the full demo flow in a real browser (headless Microsoft Edge or Chrome) against the running app and saves
 * a screenshot of every step to demo_data/screenshots/. Use them as the backup if the live app fails.
 * It also prints each scan's duration and any browser console errors or failed requests.
 * With --video it records the walkthrough to demo_data/screenshots/walkthrough.webm.
 *
 *   one-time:  cd demo_data && npm install --no-save playwright-core@1.47.2   (no browser download; uses Edge)
 *   run:       .\run_demo.ps1   then   node demo_data\capture_demo.cjs [--video] [--headed]
 */
const path = require('path')
const fs = require('fs')
const { chromium } = require('playwright-core')

const APP = process.env.DEMO_URL || 'http://localhost:3000'
const DIR = path.join(__dirname, 'screenshots')
const IMG = (name) => path.join(__dirname, 'images', name)
const args = process.argv.slice(2)
const problems = []
const timings = {}

async function main() {
  fs.mkdirSync(DIR, { recursive: true })
  const browser = await chromium.launch({ channel: 'msedge', headless: !args.includes('--headed') })
    .catch(() => chromium.launch({ channel: 'chrome', headless: !args.includes('--headed') }))
  const context = await browser.newContext({
    viewport: { width: 1366, height: 768 },
    ...(args.includes('--video') ? { recordVideo: { dir: DIR, size: { width: 1366, height: 768 } } } : {}),
  })
  const page = await context.newPage()
  const watch = (p) => {
    p.on('console', (m) => { if (m.type() === 'error') problems.push(`console: ${m.text()}`) })
    p.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`))
    p.on('response', (r) => { if (r.status() >= 400) problems.push(`HTTP ${r.status()} ${r.url()}`) })
  }
  watch(page)
  // The app scrolls inside <main>; for a full-page capture the fixed-height layout is unlocked temporarily.
  const unlock = '.h-screen{height:auto!important}.overflow-hidden{overflow:visible!important}' +
                 'main.overflow-y-auto{overflow:visible!important}'
  const shot = async (name, full = true) => {
    await page.waitForTimeout(400)
    const tall = full && await page.evaluate(() => {
      const m = document.querySelector('main'); return !!m && m.scrollHeight > m.clientHeight
    })
    const style = tall ? await page.addStyleTag({ content: unlock }) : null
    await page.screenshot({ path: path.join(DIR, `${name}.png`), fullPage: tall })
    if (style) await style.evaluate((el) => el.remove())
    console.log(`saved ${name}.png`)
  }

  // 1. Login
  await page.goto(`${APP}/auth`)
  await page.fill('#email', 'grower@mango.ai')
  await page.fill('#password', 'orchard123')
  await shot('01_login', false)
  await page.click('button[type=submit]')

  // 2. Dashboard (OHI history + weather risk forecast)
  await page.waitForURL(`${APP}/`)
  await page.waitForResponse((r) => r.url().includes('/api/weather/forecast'), { timeout: 30000 }).catch(() => {})
  await page.waitForTimeout(800)
  await shot('02_dashboard_weather')

  // 3-7. Scans
  const scan = async (mode, files, name) => {
    await page.click('a[href="/upload"]')
    await page.click(`button:has-text("${mode}")`)
    const inputs = page.locator('input[type=file]')
    for (let i = 0; i < files.length; i++) await inputs.nth(i).setInputFiles(IMG(files[i]))
    const t0 = Date.now()
    await Promise.all([
      page.waitForResponse((r) => r.url().includes('/api/analyze') && r.request().method() === 'POST', { timeout: 60000 }),
      page.click('button:has-text("Run Diagnostic")'),
    ])
    timings[name] = (Date.now() - t0) / 1000
    await page.waitForLoadState('networkidle')
    await shot(name)
  }
  await scan('Disease Only', ['03_bacterial_canker_1.jpg'], '03_leaf_scan_disease_gradcam_ohi_treatments')
  await scan('Pests Only', ['10_pest_trap_fruit_fly.jpg'], '04_pest_scan_unverified')
  await scan('Dual Scan', ['05_powdery_mildew_1.jpg', '10_pest_trap_fruit_fly.jpg'], '05_dual_scan')
  await scan('Disease Only', ['11_unsure_external.jpg'], '06_retake_photo')

  // 8. History and report
  await page.click('a[href="/history"]')
  await page.waitForSelector('text=Report')
  await shot('07_history')
  const [report] = await Promise.all([
    context.waitForEvent('page'),
    page.locator('button:has-text("Report")').first().click(),
  ])
  watch(report)
  await report.waitForLoadState('load')
  await report.waitForTimeout(800)
  await report.screenshot({ path: path.join(DIR, '08_report.png'), fullPage: true })
  console.log('saved 08_report.png')
  await report.close()

  // 9. Chatbot, English then Tamil
  await page.click('a[href="/chatbot"]')
  const ask = async (q) => {
    await page.fill('form input[type=text]', q)
    await Promise.all([
      page.waitForResponse((r) => r.url().includes('/api/chatbot')),
      page.click('form button[type=submit]'),
    ])
    await page.waitForTimeout(600)
  }
  await ask('How do I treat anthracnose?')
  await ask('What is the dose for powdery mildew?')
  await ask('How to control fruit fly?')
  await shot('09_chatbot_english', false)
  await page.click('button:has-text("தமிழ்")')
  await ask('ஆந்த்ராக்னோஸ் மருந்தளவு என்ன?')
  await shot('10_chatbot_tamil', false)

  await context.close()
  await browser.close()
  console.log('scan seconds:', JSON.stringify(timings))
  console.log(problems.length ? `PROBLEMS (${problems.length}):\n  ` + problems.join('\n  ') : 'no console errors or failed requests')
  process.exit(problems.length ? 2 : 0)
}

main().catch((e) => { console.error(e); process.exit(1) })
