# Customer Churn — Plain-Language Summary for Business Stakeholders

*A non-technical overview of what we built, what we learned, and what we recommend.
No statistics background needed. For the underlying detail see `findings.md`,
`methodology.md`, and `model_evaluation.md`.*

---

## The problem in one sentence

About **27 out of every 100 customers leave** ("churn"), and we want to know **who is
likely to leave next** so the business can step in *before* they go.

## Key Business Questions Answered:

**1. Who are our highest-risk customers?**

We built a prediction tool that looks at a customer's plan, tenure, charges, and services, and
produces a **churn risk score** for each customer — essentially "how likely is this
person to leave?" — along with a simple **Low / Medium / High** risk label.

It does **not** decide anything on its own. It produces a ranked list of at-risk
customers that the retention team can act on.

**2. What drives churn most?**

These came out of the data *and* were independently confirmed by the model. Each is
expressed as "share of that group who churned":

1. **Month-to-month contracts** — **43%** churn, vs. just **3%** for two-year
   contracts. By far the strongest signal.
2. **New customers (under ~1 year)** — **47%** churn in the first year, dropping to
   **~10%** after four years. Risk is concentrated early.
3. **Electronic-check payers** — **45%** churn, about **3×** higher than customers on
   automatic payment.
4. **Fiber-optic internet customers** — **42%** churn, vs. **19%** for DSL. Worth
   investigating the fiber experience or pricing.
5. **No add-on services** (no online security / no tech support) and **higher monthly
   bills** also push customers toward leaving.

**3. How accurate is this?**

- The tool correctly tells higher-risk from lower-risk customers about **83–85% of the
  time** (a standard quality measure called ROC-AUC; higher is better, 50% would be a
  coin flip). We confirmed this score is **stable**, not a fluke of one data sample.
- More importantly, we can **tune how aggressive it is** (see "A dial you can turn"
  below). At a balanced setting it **catches roughly 3 out of 4 customers who really do
  churn**, while keeping false alarms reasonable.

In short: it is a reliable early-warning system, not a crystal ball. It will miss some
churners and raise some false alarms — but it is far better than guessing or reacting
after customers have already left.

**4. What should we do?**

Based on our analysis, these actions will have the biggest impact:

1. **Target month-to-month customers** - 43% churn vs 3% for 2-year contracts
2. **Focus on new customers** - 47% churn in first year vs 10% after 4 years  
3. **Move customers to auto-pay** - Electronic check payers churn 3× more
4. **Investigate fiber experience** - 42% churn vs 19% for DSL

**5. How do we use this?**

The system is production-ready with:
- Ready-to-deploy API for integration with existing systems
- Threshold tuning for business needs (conservative to aggressive)
- Risk scoring system that can be used by retention teams

**Protective factors** (these keep customers around): long contracts, long tenure,
automatic payment, and having add-on services.

## A dial you can turn (risk sensitivity)

The tool gives each customer a probability of churning. We then choose a **cut-off**
for who counts as "at risk." This is a business choice, not a technical one:

- **Turn the dial more sensitive** → we flag more customers, **catch more real
  churners**, but also raise more false alarms (some flagged customers were not actually
  going to leave).
- **Turn the dial less sensitive** → fewer false alarms, but we **miss more real
  churners**.

Our recommended balanced setting catches about **77% of churners** while keeping
roughly half of the flagged customers as true churners. If a lost customer is far more
costly than a wasted retention offer (usually the case), lean toward the more sensitive
setting. Full guidance is in `usage.md`.

## What we recommend

**Quick wins**
- **Watch new customers closely.** The first 12 months are the danger zone — invest in
  onboarding and early check-ins.
- **Promote longer contracts.** Offer incentives to move month-to-month customers to
  one- or two-year plans (the single biggest lever).
- **Nudge customers onto automatic payment.** Electronic-check payers churn ~3× more.

**Worth investigating**
- **Fiber-optic experience.** These customers pay more and leave more — is it price,
  reliability, or expectations?
- **Bundle support services.** Customers without online security or tech support are
  more likely to leave; bundling may both reduce churn and add value.

## Important caveats (so the tool is used well)

- **It predicts, it does not explain cause.** "Month-to-month customers leave more" is a
  strong pattern, but only a real test (e.g. trying an incentive on a trial group) can
  prove an action *causes* customers to stay.
- **It will make mistakes.** Expect both misses and false alarms; the dial controls the
  balance, not whether mistakes happen.
- **It needs upkeep.** Customer behaviour changes over time, so the tool should be
  refreshed periodically with new data to stay accurate.
- **It reflects the data we have.** No information about competitors, outages, or
  customer-service interactions is included; those could matter too.

## Where the supporting detail lives

- **Charts** (risk breakdowns, model performance): the `results/` folder, e.g.
  `confusion_matrix.png`, `roc_curve.png`, `shap_summary.png`.
- **Detailed findings**: `docs/findings.md`.
- **How predictions are produced and tuned**: `docs/usage.md`.
- **Method and model details**: `docs/methodology.md` and `docs/model_evaluation.md`.

---

*Bottom line: we can reliably rank customers by churn risk and explain the main reasons
behind it. Acting early on month-to-month, new, electronic-check, and fiber-optic
customers is where the biggest retention gains are likely to come from.*
