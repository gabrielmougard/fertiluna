# Ad Setup Guide: Google AdSense on Cloudflare Workers

Complete step-by-step guide to monetize your site with Google AdSense,
optimized for a TikTok-community tool site with mobile-heavy traffic.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Revenue Estimates](#revenue-estimates)
- [Phase 1: Google AdSense Account](#phase-1-google-adsense-account)
- [Phase 2: Get Your Site Approved](#phase-2-get-your-site-approved)
- [Phase 3: Create Ad Units](#phase-3-create-ad-units)
- [Phase 4: Configure the Worker](#phase-4-configure-the-worker)
- [Phase 5: Set Up ads.txt](#phase-5-set-up-adstxt)
- [Phase 6: Deploy and Verify](#phase-6-deploy-and-verify)
- [Phase 7: Testing Locally](#phase-7-testing-locally)
- [Growth Roadmap](#growth-roadmap)
- [Appendix: Ad Sizes by Device](#appendix-ad-sizes-by-device)
- [Appendix: Troubleshooting](#appendix-troubleshooting)

---

## Architecture Overview

```
TikTok community member taps link
  |
  v
Cloudflare Worker receives request
  |
  |  1. Detects device type (mobile / tablet / desktop)
  |     from CF-Device-Type header or User-Agent
  |
  |  2. Renders HTML with the right ad layout:
  |     - Mobile:  top 320x50, inline 300x250, bottom 320x50 sticky
  |     - Tablet:  top 728x90, inline 300x250, bottom 728x90
  |     - Desktop: top 728x90, left 160x600, right 160x600, bottom 728x90
  |
  v
Browser loads AdSense script
  |
  |  3. AdSense runs a real-time auction for each ad slot.
  |     Google's demand pool includes:
  |     - Retargeting ads (advertisers targeting users who visited their site)
  |     - Contextual ads (matched to your page content)
  |     - Interest-based ads (based on user's browsing history)
  |
  |  4. Highest-paying ad wins each slot
  v
Ads displayed, you earn CPM/CPC revenue
```

**Does AdSense do retargeting?** Yes. When an advertiser runs a Google Ads
remarketing campaign, those ads are served through the Google Display Network,
which includes AdSense. You get retargeting ad revenue automatically -- no
extra setup needed.

---

## Revenue Estimates

Realistic estimates for a TikTok-community tool site (80%+ mobile traffic):

| Daily visits | Pageviews (1.5x) | Est. CPM | Daily revenue | Monthly |
|---|---|---|---|---|
| 1,000 | 1,500 | $1.50 | ~$2.25 | **~$68** |
| 3,000 | 4,500 | $1.80 | ~$8 | **~$243** |
| 5,000 | 7,500 | $2.00 | ~$15 | **~$450** |
| 10,000 | 15,000 | $2.20 | ~$33 | **~$990** |

**Why mobile CPMs are lower:** mobile screens are smaller, ad viewability is
lower, and competition for mobile inventory is less intense than desktop in
most niches. The 300x250 in-content rectangle is the exception -- it
performs well on mobile because users scroll directly past it.

**Per-slot breakdown (mobile):**

| Slot | Format | Est. CPM | Revenue share |
|------|--------|----------|--------------|
| Top 320x50 | Mobile banner | $0.30-0.50 | ~10% |
| Inline 300x250 | Medium rectangle | $2.50-4.00 | ~65% |
| Bottom 320x50 | Sticky anchor | $1.00-2.00 | ~25% |

The inline 300x250 is your primary money-maker on mobile. The sticky bottom
anchor has high viewability but low CPM due to its small size. The top banner
earns the least but establishes the "this site has ads" contract with users.

---

## Phase 1: Google AdSense Account

### 1.1 Sign Up

1. Go to [https://adsense.google.com](https://adsense.google.com)
2. Click **"Get started"**
3. Sign in with your Google account
4. Enter your website URL (must be a real domain, not `*.workers.dev`)
5. Select your payment country
6. Accept terms of service

### 1.2 Account Review

Google will review your application. This typically takes **1-3 days** but
can take up to 2 weeks. During review, they check:

- Your site has **original, substantial content** (not just lorem ipsum --
  replace the placeholder content with your actual tool before applying)
- Your site has basic pages: privacy policy, about/contact
- Your site is navigable and functional
- You are at least 18 years old
- Your content complies with [AdSense program policies](https://support.google.com/adsense/answer/48182)

### 1.3 Common Rejection Reasons

| Reason | Fix |
|--------|-----|
| "Insufficient content" | Add real content. Your tool should be functional with at least a few pages of useful text. |
| "Site not accessible" | Make sure your domain is live and not behind a password. |
| "Navigational issues" | Ensure the site has clear navigation, even if it's a single-page tool. |
| "Policy violation" | Review the content policies. No adult content, no copyrighted material, etc. |

You can reapply after fixing issues. There is no limit on reapplications.

---

## Phase 2: Get Your Site Approved

Before applying to AdSense, make sure your Worker is deployed on a real
domain (not `*.workers.dev`):

### 2.1 Domain Setup

1. Buy a domain (Cloudflare Registrar, Namecheap, etc.)
2. Add it to your Cloudflare account
3. In Cloudflare dashboard > Workers & Pages > your worker > Settings > Domains
4. Add your custom domain

### 2.2 Required Pages

Add these to your site (even minimal versions):

- **Privacy Policy** -- state that you use Google AdSense, which uses cookies
  for ad personalization. You can use a free privacy policy generator.
- **About / Contact** -- brief description of your tool and a way to contact
  you (email is fine).

### 2.3 Replace Placeholder Content

Your actual tool/content should be live before applying. AdSense reviewers
will reject a site with only lorem ipsum.

---

## Phase 3: Create Ad Units

Once approved, you need to create ad units in your AdSense dashboard. Each
ad unit gets a unique **slot ID** that you will configure in the Worker.

### 3.1 Navigate to Ad Units

1. In AdSense, go to **Ads > By ad unit > Display ads**
2. Click **"Create new ad unit"**

### 3.2 Create 5 Ad Units

| # | Ad unit name | Type | Size | Used for |
|---|---|---|---|---|
| 1 | `site_top` | Display | Fixed: 728x90 | Desktop/tablet top leaderboard |
| 2 | `site_left` | Display | Fixed: 160x600 | Desktop left sidebar |
| 3 | `site_right` | Display | Fixed: 160x600 | Desktop right sidebar |
| 4 | `site_bottom` | Display | Fixed: 728x90 | Desktop/tablet bottom leaderboard |
| 5 | `site_inline` | Display | Fixed: 300x250 | Mobile/tablet in-content rectangle |

For each:

1. Click **"Create new ad unit"** > choose **"Display ads"**
2. Name it as shown above
3. Under "Ad size," choose **Fixed size** and enter the dimensions
4. Click **"Create"**
5. On the next screen, you will see the ad code. You do NOT need the code --
   just note the **data-ad-slot** value (a ~10-digit number like `1234567890`).
   This is your slot ID.

### 3.3 Note All Slot IDs

Record each slot ID:

```
ADSENSE_SLOT_TOP=1234567890      (from site_top)
ADSENSE_SLOT_LEFT=1234567891     (from site_left)
ADSENSE_SLOT_RIGHT=1234567892    (from site_right)
ADSENSE_SLOT_BOTTOM=1234567893   (from site_bottom)
ADSENSE_SLOT_INLINE=1234567894   (from site_inline)
```

### 3.4 Your Publisher ID

Your AdSense publisher ID is shown in AdSense under **Account > Account
information**. It looks like `ca-pub-XXXXXXXXXXXXXXXX`. This is your
`ADSENSE_PUB_ID`.

---

## Phase 4: Configure the Worker

### 4.1 Local Development

Edit `.dev.vars`:

```env
DEV_MODE=true
ADSENSE_PUB_ID=ca-pub-XXXXXXXXXXXXXXXX
ADSENSE_SLOT_TOP=1234567890
ADSENSE_SLOT_LEFT=1234567891
ADSENSE_SLOT_RIGHT=1234567892
ADSENSE_SLOT_BOTTOM=1234567893
ADSENSE_SLOT_INLINE=1234567894
```

With `DEV_MODE=true`, the Worker shows mock ad placeholders -- no real
AdSense scripts load.

### 4.2 Production Secrets

```bash
npx wrangler secret put ADSENSE_PUB_ID
# Enter: ca-pub-XXXXXXXXXXXXXXXX

npx wrangler secret put ADSENSE_SLOT_TOP
# Enter the slot ID from site_top

npx wrangler secret put ADSENSE_SLOT_LEFT
npx wrangler secret put ADSENSE_SLOT_RIGHT
npx wrangler secret put ADSENSE_SLOT_BOTTOM
npx wrangler secret put ADSENSE_SLOT_INLINE
```

### 4.3 Deploy

```bash
npm run deploy
```

---

## Phase 5: Set Up ads.txt

### 5.1 What Is ads.txt?

`ads.txt` is a public file that tells ad exchanges "these networks are
authorized to sell ads on my site." Google requires it.

### 5.2 Find Your ads.txt Entry

In AdSense, go to **Sites > (your site) > ads.txt**. Google will show you
the line you need. It looks like:

```
google.com, pub-XXXXXXXXXXXXXXXX, DIRECT, f08c47fec0942fa0
```

### 5.3 Set It in the Worker

```bash
npx wrangler secret put ADS_TXT
# Paste: google.com, pub-XXXXXXXXXXXXXXXX, DIRECT, f08c47fec0942fa0
```

The Worker serves this at `/ads.txt` automatically.

### 5.4 Verify

After deploying, visit `https://yourdomain.com/ads.txt` and confirm the
entry is present.

---

## Phase 6: Deploy and Verify

### 6.1 Deploy

```bash
npm run deploy
```

### 6.2 Check Scripts Load

1. Visit your site on your real domain
2. Open browser DevTools (F12) > Network tab
3. Confirm you see a request to `pagead2.googlesyndication.com`
4. Confirm no JavaScript errors in the Console

### 6.3 Check Ads Render

- Ads may take a few minutes to start appearing after first deployment
- If you see blank spaces where ads should be, this is normal for the first
  few hours as AdSense learns your inventory
- AdSense may show a "vignette" or blank box initially

### 6.4 Verify on Mobile

Since most of your traffic is mobile:

1. Open your site on your phone (or use Chrome DevTools device emulation)
2. Confirm you see:
   - Slim 320x50 banner at the top
   - 300x250 rectangle in the content
   - 320x50 sticky banner at the bottom
3. Verify the sticky bottom banner doesn't obscure content (the template adds
   62px bottom padding to prevent this)

---

## Phase 7: Testing Locally

### 7.1 Dev Mode

```bash
npm run dev
```

Visit `http://localhost:8787`. You will see mock ad placeholders showing:
- Zone name (e.g., "Top Banner")
- Size (e.g., "320 x 50")
- Detected device type

### 7.2 Simulating Mobile Locally

The Worker detects device from the `User-Agent` header. To simulate mobile:

**Chrome DevTools:**
1. Open DevTools (F12)
2. Click the "Toggle device toolbar" icon (phone/tablet icon)
3. Select a mobile device (e.g., iPhone 14)
4. Reload the page

**Note:** Wrangler's local dev server passes the browser's User-Agent to
the Worker, so device emulation works correctly.

**cURL:**
```bash
curl -H "User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)" \
  http://localhost:8787
```

### 7.3 Why Real Ads Don't Show Locally

AdSense will not serve ads on `localhost` or non-approved domains. The mock
placeholders in dev mode exist for this reason -- they let you verify layout
and responsiveness without needing real ads.

---

## Growth Roadmap

What to do as your traffic grows:

### Stage 1: 1K-10K visits/day (you are here)

**Focus: grow traffic, not ad optimization.**

- Use AdSense only -- it's the right tool for this traffic level
- Spend time creating great tools for your TikTok community
- Share tools on TikTok to drive traffic
- Monitor AdSense dashboard for baseline CPMs

**Expected revenue: $70-$1,000/month**

### Stage 2: 10K-30K visits/day

**Optimization starts to matter.**

- Enable AdSense "Auto ads" experiment: let Google's AI test additional
  placements (it may find spots that convert well)
- A/B test the inline 300x250 position (try it after paragraph 1 vs.
  paragraph 2)
- Consider adding a second inline ad unit if content is long enough
  (at least 3 scroll-lengths of content between ads)
- Start building an email list or Discord from your TikTok community

**Expected revenue: $1,000-$3,000/month**

### Stage 3: 30K+ visits/day (~50K+ sessions/month)

**Apply to Mediavine.**

[Mediavine](https://www.mediavine.com/) is a premium ad management company
that replaces AdSense entirely. They:

- Manage all ad placements, formats, and demand for you
- Run header bidding with 20+ demand partners behind the scenes
- Pay 2-3x more than raw AdSense for the same traffic
- Require 50,000 sessions/month minimum

**Expected revenue: $3,000-$10,000+/month**

### Stage 4: 100K+ visits/day

**You have options:**

- **Stay with Mediavine / move to AdThrive** -- hands-off, premium CPMs
- **Build custom header bidding** with Prebid.js -- add Criteo, Amazon,
  Index Exchange, etc. alongside Google Ad Exchange. More work, potentially
  higher revenue.
- **Sell direct sponsorships** -- your TikTok community niche may attract
  brand deals that pay far more than programmatic ads

### When Does Criteo Make Sense?

Criteo becomes worth integrating when:

- You have **50K+ daily visits**
- A meaningful portion of your audience is **desktop** users who browse
  e-commerce sites (Criteo's retargeting pool)
- You have a **dedicated ad ops person** or agency to manage the GAM
  line items
- OR you use Mediavine/AdThrive, which include Criteo in their header
  bidding stack automatically

For a mobile-heavy TikTok community site, you are likely better off going
the Mediavine route at scale -- they handle Criteo integration for you.

---

## Appendix: Ad Sizes by Device

### Desktop (viewport >= 960px)

| Zone | Size | Format | Position |
|------|------|--------|----------|
| Top | 728x90 | Leaderboard | Static, above content |
| Left | 160x600 | Wide Skyscraper | Sticky sidebar |
| Right | 160x600 | Wide Skyscraper | Sticky sidebar |
| Bottom | 728x90 | Leaderboard | Static, below content |

### Tablet (viewport 768-959px)

| Zone | Size | Format | Position |
|------|------|--------|----------|
| Top | 728x90 | Leaderboard | Static, above content |
| Inline | 300x250 | Medium Rectangle | Between content sections |
| Bottom | 728x90 | Leaderboard | Static, below content |

### Mobile (viewport < 768px)

| Zone | Size | Format | Position |
|------|------|--------|----------|
| Top | 320x50 | Mobile Banner | Static, above content |
| Inline | 300x250 | Medium Rectangle | Between content sections |
| Bottom | 320x50 | Mobile Banner | Sticky anchor, fixed to bottom |

---

## Appendix: Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Ads not showing | AdSense not approved yet | Wait for approval email; check AdSense dashboard for status |
| Ads not showing on new domain | Domain not added in AdSense | Go to AdSense > Sites > Add site |
| Blank ad slots | AdSense is "learning" your inventory | Wait 24-48 hours; also check for ad blockers |
| "adsbygoogle.push() error" | Ad slot HTML mismatch | Verify slot IDs match what AdSense dashboard shows |
| Very low revenue | Low traffic or low-CPM geo | Focus on growing traffic; revenue scales with volume |
| Ads show on desktop but not mobile | Missing mobile slot IDs | Verify `ADSENSE_SLOT_INLINE` is set |
| Sticky bottom ad blocks content | CSS issue | Check that body has `padding-bottom: 62px` on mobile |
| ads.txt warning in AdSense | Missing or wrong ads.txt | Deploy the `ADS_TXT` secret and verify at `/ads.txt` |
| "Page-level ads" warning | AdSense wants you to enable auto ads | Optional: enable in AdSense dashboard under "Auto ads" |
