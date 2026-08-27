# Weinstein Stage Tracker

A weekly scanner for Stan Weinstein stage analysis across the S&P 500, S&P 400
and FTSE 350. It answers two questions every Saturday morning: which stocks broke
into Stage 2 or out of Stage 4 this week, and which ones are set up to do so next.

## What it actually measures

Weinstein's method is a weekly-chart method. Everything here is computed on
weekly bars, because the 30 week moving average, the Mansfield relative strength
line and the two-times-volume breakout rule are all defined that way. Running the
same logic on daily data produces a different and much noisier system.

Four inputs carry the model.

**The 30 week moving average and its slope.** The slope is measured as the
percentage change in the average itself over five weeks, then divided by weekly
ATR so a quiet utility and a volatile biotech are held to comparable standards of
"the average is rising".

**Mansfield relative strength.** The stock divided by its market index, then
divided by its own 52 week average of that ratio, minus one. Above zero is
Weinstein's precondition for buying a breakout. Below zero on an otherwise perfect
breakout is his usual reason to pass.

**Base geometry.** Support and resistance from the prior 30 weeks excluding the
current bar, the width of that range, how long price has stayed inside it, and how
much the last ten weeks have contracted within it.

**Volume.** The breakout week measured against the ten week average, taken as the
best of the breakout week and the two before it, because price often clears the
pivot a week or two after the accumulation surge. Breakdowns carry no volume
requirement, since selling can happen on an absence of bids.

## The market and the group

Weinstein's two contextual requirements are now part of the model rather than
notes about what it does not do.

The market regime is built from the index and from breadth together, because the
index alone is misleading: a cap weighted index can hold up on a handful of
megacaps while the median stock is already in Stage 4, which is exactly the
condition in which breakouts fail most. Breadth, the share of the scanned
universe above its own 30 week average, sees that directly and is weighted at 35
per cent against the index trend's 45. The result is a score from minus one
hundred to plus one hundred and a label from Bull to Bear, printed at the top of
the dashboard with the guidance that goes with it.

Sector strength is two separate questions and both are computed. Whether the
group beats the market is Mansfield relative strength on an equal weighted
composite of that sector's own constituents, measured against the same index the
stocks are measured against, so sectors and stocks sit on one scale. Whether the
stock beats its own group is the same measure applied against the composite.
Equal weighting is deliberate, since a cap weighted sector index tracks its two
largest members and the question here is what the group is doing.

Those two feed a multiplicative group factor: 1.15 for a leader in a leading
group, 0.70 for a laggard in a lagging group, 0.88 in between, and exactly 1.00
when there is no sector label, so an unclassified stock is ranked on its own
chart rather than quietly demoted. The factor scales readiness and drives the
signal grade. Grade A is a textbook break by a leader in a leading group with the
market behind it, C is a textbook break with the group or the market against it,
which is the case Weinstein explicitly tells you to pass on.

## The two outputs

**Stage classification** reads the chart in Weinstein's own order. Flat 30 week
average first, because a flat average is the definition of a transition, and only
then what came before it. A flat average after a decline is a Stage 1 floor. The
identical flat average after an advance is a Stage 3 ceiling. Prior trend is
measured from before the flat stretch began rather than over a trailing year that
may sit entirely inside it.

**Readiness scores** are the forward looking half. Each name gets a Stage 2 and a
Stage 4 readiness score out of one hundred, built from proximity to the pivot, the
turn in the moving average, relative strength, base quality and volume behaviour,
then scaled by a context factor. The context factor is the part that stops a Stage
3 top from ranking as the best Stage 2 candidate on the board, which it otherwise
does, because a range near its highs looks identical whichever way it is about to
resolve.

## Trade plans

Every candidate carries the levels an order ticket needs, and clicking its row on
the dashboard opens them: the buy trigger, a pullback entry zone, the initial
stop, two targets with their R multiples, and a position size.

The trigger sits a quarter of a weekly ATR above the pivot, floored at 0.4 per
cent, so a single tick through the base high does not count as a break. The stop
goes below the nearest ten week swing low and below the 30 week average, which is
Weinstein's "below the most recent significant low" rather than below the floor of
a base that may be two years old and thirty per cent down, and it is pulled in if
that would put it more than twelve per cent away.

Targets are the height of the base multiplied by a factor that grows with how
long the base took to form, from one times at a fresh base to three times at two
years. That scaling is the point rather than a refinement: Weinstein's own
objectives come from point and figure horizontal counts, whose entire mechanism is
that a wider base counts to a bigger target, and a flat one-times-height target is
arithmetically almost always under 1R because the stop is set by the same base the
target is measured from.

One thing the levels do not tell you is which order type to use, and it matters.
A resting buy stop fills the moment price touches the trigger on any day,
including on spikes that are back inside the base by Friday. Weinstein's rule is a
weekly close beyond the pivot on at least double volume, which means checking on
Friday and dealing on Monday. The price is the same either way, the fills are not:
the resting order takes more trades and more of them fail.

Each run also writes `docs/index_alerts.csv`, one row per price level with the
reason attached, so a fired alert says what it means rather than only that a
number was touched.

## Reading a row

Clicking any row opens a plain-English panel above the levels. It says what the
model thinks is happening in that stage, whether the sector and the market are
behind the name or against it, what the rules say to do today, and what would make
the idea wrong. Every sentence is derived from a value in that row, so nothing is
asserted that the scan did not measure, and where a component is weak the text
says so rather than rounding it into confidence.

That panel also carries a **Copy pre-trade check** button. It builds a filled-in
prompt containing the scan's own numbers for that ticker and asks an assistant to
try to talk you out of the trade: fetch current prices, since the scan is between
one and six days stale by the time you deal; re-derive the four Weinstein
conditions from a live chart rather than trusting the table; and look for what a
weekly OHLCV scan structurally cannot see, meaning earnings inside the window, a
takeover approach or placing, a consolidation that corrupts the price history the
base was measured from, dilution, liquidity and spread, and whether a UK line is
quoted in pence. It ends with a GO, NO or WAIT and the single strongest objection.

The template lives in `PROMPT.md` and in `weinstein/precheck.py`. The dashboard
ships one copy of it and fills it in the browser, so the file you can read and the
button you press cannot drift apart.

## Managing what you already hold

Put a `positions.csv` beside the project with the columns `ticker, entry_date,
entry_price, shares, initial_stop, side` and each weekly run reports what the plan
says to do with each one: hold, move the stop to breakeven, trim at the first
target, or exit.

The plan is rebuilt from the bar at your entry date rather than from the current
week, because the targets and the stop were set by the base that existed then.
Recomputing them from today would move the goalposts every week and make the plan
unfalsifiable.

The exit sequence is Weinstein's rather than a single stop: hold the initial stop
until one R of open gain, move to breakeven there, sell half at the first target,
trail the remainder half a weekly ATR under the 30 week average updated once a
week on the Friday close, and exit the rest on a weekly close through that average
while it has stopped rising.

## Calibration

Every run backtests itself, and now tests the two new layers as well as the
core signal. Confirmed signals are split by the market regime in force the week
they fired and by which group bucket they came from, so the claim that regime and
group matter is measured rather than asserted. If those splits come back flat,
the filters are costing signals without buying anything and should be loosened.

It measures what confirmed signals actually did next,
as a distribution of excess return over the stock's own index at four, thirteen and
twenty six weeks, plus the share stopped out inside thirteen weeks. It also buckets
every historic week by readiness decile and reports how often a confirmed break
followed within eight weeks. If the top decile does not clearly beat the bottom
decile, the readiness model is decoration and should be rebuilt rather than
trusted. That table is printed in the dashboard on every scan.

## Running it

```bash
pip install -r requirements.txt
./run_tests.sh                               # offline verification, no network needed
python -m weinstein.cli demo                 # offline, synthetic, proves the pipeline
python -m weinstein.cli scan                 # full live scan, writes docs/index.html
python -m weinstein.cli scan --indices SP500 # one index
python -m weinstein.cli scan --cache         # rescore from cached prices, no network
```

`scan` writes `docs/index.html`, a self-contained dashboard with no external
assets, and `docs/index_full.csv` with every scored field for every ticker.

## Automating it

`.github/workflows/weekly-scan.yml` runs the scan at 08:00 UTC every Saturday,
commits the regenerated dashboard, and pushes. Turn on GitHub Pages for the
`docs/` folder and the dashboard lives at a fixed URL that updates itself. Both
markets' weekly bars are final by that time.

## Audit

The system has been through a five-dimension adversarial review (lookahead,
indicator mathematics, scoring, backtest statistics, plan arithmetic) plus a
mechanical invariant harness in `tests/test_invariants.py`.

The harness enforces properties that must hold for every input rather than for
constructed cases. Causality is checked two ways: recomputing the whole feature
engine on truncated history and demanding bit-identical values up to the cut, and
perturbing one bar and demanding nothing earlier moves. Scale invariance checks
that multiplying every price by a constant leaves percentages, ratios and scores
unchanged while levels scale exactly, which catches mixed units. It also asserts
that no scored component ever rewards a missing input, that scores stay inside
their stated ranges, and that several thousand randomly generated trade plans all
satisfy stop-below-entry-below-targets, the risk cap, the sizing budget and alert
ordering.

The single most important finding: the readiness decile table, previously
described here as the test of whether the forward-looking half works, could not
fail. Its largest component is proximity to the level and its target is crossing
that level, so a pure random walk passed it at 28.7 sigma. It has been demoted to
a description of the sample and replaced by an AUC decomposition that reports the
score's ranking power, the ranking power of distance to the level on its own, and
the increment between them. The increment is the number that matters.

The calibration panel now also reports week-clustered confidence intervals and an
independent-episode count rather than a row count, because signals fire together
and hundreds of simultaneous breakouts are one draw rather than hundreds, and a
comparison budget stating how many statistics are on the page and what a single
one would have to clear to mean anything.

## How much history, and how wide a universe

Both questions have the same answer and it is not the obvious one.

Widening the universe buys almost no statistical power. Signals do not arrive
independently: hundreds fire in the same week and share whatever the market did
that week, so tripling the number of tickers triples the rows and leaves the
standard error essentially unchanged. Simulated across intra-week correlations
from 0.05 to 0.40, going from 1,250 names to 3,000 improves the standard error by
between 0 and 2 per cent. Going from eight years to twenty-five improves it by
about 43 per cent at every one of those correlations, because years add
independent episodes and tickers do not.

That argues for more history, except that more history makes survivorship worse in
exactly the same motion. Membership is scraped as it stands today, so at roughly
4.5 per cent annual index turnover about 69 per cent of today's constituents were
also members eight years ago, 50 per cent fifteen years ago and 32 per cent
twenty-five years ago. A twenty-five year window is twenty-five years of the
survivors' history, and bias does not shrink with sample size the way variance
does. Combining the two effects, total error is minimised somewhere between eight
and twenty-five years depending on how strong the bias actually is, which nobody
here knows.

So measure it rather than guessing:

```bash
python -m weinstein.cli horizons --years-list 8 15 25
```

That runs identical rules over three history lengths from one download and prints
the confirmed-breakout statistics side by side. The rules do not change and
membership is today's in every run, so if the measured edge improves as the window
lengthens, most of that improvement is the bias becoming visible. A flat comparison
is the reassuring result. A steeply improving one means the eight year numbers are
the honest ones.

Widening the universe is still worth doing for a different reason, and it is not a
data reason: small caps are less efficiently priced, so the method may simply work
better there. That is a separate hypothesis rather than more evidence for this one,
and it needs its own calibration, a liquidity filter, and the knowledge that
small-cap delisting rates make survivorship considerably worse than the numbers
above.

Candidate supply is not the constraint either way. At a confirmed-break rate of
roughly 1.5 per cent of the universe per week, 1,250 names already produce about 19
breaks and 4 grade A setups in a typical week, which is more than one person acting
on a weekly cycle can take.

## Known limits

Yahoo data through yfinance is free and unofficial. It has occasional bad bars,
survivorship bias in index membership scraped from current Wikipedia lists, and no
service guarantee. The backtest inherits all three, so treat its numbers as
indicative rather than precise, and treat any measured edge as an upper bound.

Index membership is current membership. A backtest over eight years therefore
tests today's constituents on their own history, which flatters results, because
the companies that fell out of the index are absent.

Sector composites are built from current index membership, so they inherit the
same survivorship bias as the stock universe and a historical sector composite is
really today's members' history. The lookahead test in
`tests/test_sectors_regime.py` proves the composite does not leak the future into
earlier weeks, which is a different and weaker guarantee than the composite being
a faithful record of what that sector actually was.

The thresholds in `stages.py` are Weinstein's textbook values where he gave one
and reasoned defaults where he did not. They are the first thing to revisit once
the calibration table has enough sample.
