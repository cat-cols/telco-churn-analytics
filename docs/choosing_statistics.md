# Choosing the Right Statistics: A Reasoning Guide

This guide explains *how to decide which statistical methods to use* for the
Telco churn project, and *why* each choice is appropriate. It contains no code
and no software references — only the statistical reasoning. The goal is to be
able to defend every number in this project from first principles.

---

## 1. Start with the question, not the method

Every statistic should answer a specific question. For this project there are
three distinct questions, and each calls for a different family of methods:

1. **Description** — "What does the customer base look like, and who churns?"
2. **Association** — "Which customer characteristics are related to churn, and
   how strongly?"
3. **Prediction** — "Can we rank customers by how likely they are to leave, and
   how good is that ranking?"

Choosing a method before knowing which of these you are answering is the most
common mistake. The rest of this guide works through each question in order.

---

## 2. Classify your variables first

The single most important step is to identify the *type* of each variable,
because the variable type dictates which statistics are valid.

- **Categorical (nominal)** — labels with no inherent order.
  Examples: contract type, payment method, internet service, churn (yes/no).
- **Ordinal** — categories with a meaningful order but unequal/unknown spacing.
  Example: tenure grouped into bands (0–12, 13–24, 25–48, 49–72 months).
- **Continuous (numeric)** — measured quantities on a real scale.
  Examples: tenure in months, monthly charges, total charges.

The target variable, **churn**, is **binary categorical** (yes/no). This single
fact shapes almost every downstream decision: it rules out methods that assume a
continuous outcome and points us toward methods for categorical outcomes.

---

## 3. Describing the data

Before testing anything, summarize each variable on its own terms.

- **Continuous variables** are summarized with a measure of center and a measure
  of spread. Use the **mean and standard deviation** when the distribution is
  roughly symmetric; prefer the **median and interquartile range** when it is
  skewed or has extreme values, because the median is resistant to outliers.
  Tenure and charges are skewed, so the median is often the more honest summary.
- **Categorical variables** are summarized with **counts and proportions** (e.g.,
  55% of customers are month-to-month). Proportions are the natural currency for
  categorical data.
- **The target** is summarized by its **base rate**: roughly 27% of customers
  churn. This base rate is the reference point for everything that follows — any
  claim of "high risk" only means something relative to this 27% baseline.

**Why this matters:** describing first prevents you from running a test on a
variable whose shape would invalidate the test's assumptions, and it gives you
the baseline against which all later comparisons are judged.

---

## 4. Measuring association: matching the test to the variable pair

"Association" questions always involve **two** variables. Choose the method based
on the *types* of the two variables involved.

### 4a. Categorical predictor vs. categorical outcome → Chi-square test

Most churn drivers (contract, payment method, internet service) are categorical,
and churn itself is categorical. The right tool for "are these two categorical
variables related?" is the **chi-square test of independence**.

- **The logic:** First assume the two variables are *unrelated* (the null
  hypothesis). Under that assumption, you can calculate how many customers you
  would *expect* in each cell of the cross-tabulation (e.g., churned
  month-to-month customers) purely from the row and column totals. The test then
  compares the **observed** counts to these **expected** counts. If the observed
  pattern is far from what independence would produce, the variables are
  associated.
- **What the result means:** a large chi-square value with a small p-value means
  the observed differences in churn rate across categories are very unlikely to
  be a fluke of sampling.
- **Why it fits here:** it makes no assumption about ordering or scale, which is
  correct for nominal variables like payment method.

**Assumption to respect:** the test relies on having a reasonable number of
customers in each cell (a common rule of thumb is an expected count of at least 5
per cell). With 7,043 customers spread over a handful of categories, this project
comfortably satisfies that condition — which is *why* the test is trustworthy
here.

### 4b. Why a small p-value is not enough → effect size

A p-value answers "is there an association?" but **not "how big is it?"** With a
large sample, even a trivial difference can be "statistically significant." This
project has thousands of customers, so significance is almost guaranteed and is
therefore not, by itself, impressive.

To report *strength*, pair every chi-square test with an **effect-size measure**.
For two categorical variables, the natural choice is **Cramér's V**, which
rescales the chi-square value to a 0-to-1 range (0 = no association, 1 = perfect
association) and is comparable across variables with different numbers of
categories.

**Why this is the responsible choice:** it lets you *rank* the drivers honestly.
Contract type and payment method may both be "significant," but Cramér's V reveals
that contract type is the stronger driver — which is the insight a business
actually needs for prioritization.

### 4c. Continuous predictor vs. categorical outcome → compare distributions

For continuous predictors like tenure or monthly charges against the binary churn
outcome, the question becomes "is the distribution of this number different for
churners vs. non-churners?" Approaches in increasing order of formality:

- **Compare summaries** (median tenure of churners vs. non-churners) — simple and
  often sufficient for a business narrative.
- **Compare whole distributions** side by side to see shape differences, not just
  center.
- **A formal two-group comparison** (a test of whether the average differs
  between churners and non-churners) can be used, but only if its assumptions
  hold; otherwise a rank-based comparison that does not assume a particular
  distribution shape is safer.

A pragmatic alternative used in this project is to **bucket the continuous
variable into ordered bands** (tenure bands) and then apply the chi-square logic.
This trades a little precision for interpretability — "47% of first-year
customers churn" is far more actionable to a business audience than a coefficient.

### 4d. Continuous predictor vs. continuous predictor → correlation

When both variables are continuous (e.g., tenure and total charges), use a
**correlation coefficient**, which measures how tightly two quantities move
together on a scale from −1 to +1.

- The main reason to do this here is **not** to study churn directly but to
  **detect redundancy** between predictors. Tenure and total charges are strongly
  correlated (longer-tenured customers have naturally paid more in total). Knowing
  this prevents you from treating two near-duplicate signals as independent
  evidence.
- Remember that correlation only captures **linear** co-movement and says nothing
  about causation.

---

## 5. From association to prediction

Knowing which factors relate to churn is useful, but ranking individual customers
by risk is a **prediction** problem with a **binary categorical outcome**. This
again rules out methods for continuous outcomes and points to **classification**.

The key statistical reframing: a good model does not output "yes/no" — it outputs
a **probability** of churn for each customer. How we judge that probability is the
crux of choosing the right evaluation statistics.

---

## 6. Choosing evaluation statistics for an imbalanced outcome

Because only ~27% of customers churn, the classes are **imbalanced**, and this
fact disqualifies the most intuitive metric.

### 6a. Why accuracy is the wrong headline metric

A model that predicts "no one churns" would be **73% accurate** simply because
73% of customers do not churn — while being completely useless, since it catches
zero churners. **Accuracy rewards the majority class** and hides failure on the
rare class we actually care about. This is *why* accuracy must not be the primary
metric here.

### 6b. Separating ranking quality from threshold choice

It helps to split evaluation into two independent questions:

1. **How well does the model rank customers by risk, regardless of any cutoff?**
2. **Where do we draw the line between "act" and "don't act"?**

**For ranking quality**, use a measure of how well the model orders churners
above non-churners across *all* possible cutoffs. This is the value of the
area-under-the-curve type metric (around 0.83 here): it can be read as the
probability that a randomly chosen churner is ranked as higher-risk than a
randomly chosen non-churner. It is **threshold-independent**, so it measures the
model's intrinsic discriminating power, and it is **robust to class imbalance**,
which is exactly why it is the right summary score for this project.

### 6c. Precision and recall: the business trade-off

Once you pick a cutoff, two complementary statistics describe the consequences:

- **Recall (sensitivity)** — of all customers who actually churn, what fraction
  did we flag? High recall means few churners slip through undetected.
- **Precision** — of all customers we flagged, what fraction actually churn? High
  precision means we waste little retention effort on customers who would have
  stayed anyway.

These two are in **tension**: flagging more customers raises recall but lowers
precision, and vice versa. There is no universally "correct" balance — it depends
on the relative cost of a missed churner versus a wasted retention offer.

**F1**, the harmonic mean of precision and recall, is a single summary when you
want to balance the two, but it should support the conversation, not replace it.

### 6d. Choosing the decision threshold by reasoning, not default

The default cutoff (treat anyone above 50% probability as a churner) is an
**arbitrary convention**, not a business decision. The statistically and
commercially sound approach is to **choose the threshold from the cost
trade-off**:

- If a missed churner is far more costly than a wasted incentive (often true in
  subscription businesses, where losing a customer is expensive), then **lower the
  threshold** to raise recall — accepting more false alarms to catch more real
  churners.
- This project lowers the cutoff so the model catches roughly three-quarters of
  churners, deliberately trading precision for recall because that matches the
  economics of retention.

The lesson: the threshold is a **policy lever**, and the right statistic to
optimize is the one tied to the business cost — not a textbook default.

---

## 7. Turning statistics into a financial estimate — honestly

The projected savings figure is a chain of explicit assumptions, not a measured
result. Sound reasoning here means **stating every assumption** so the estimate
can be challenged:

- the base churn rate and the number of customers it implies are lost,
- the fraction of churners the model identifies (recall at the chosen threshold),
- the fraction of *identified* churners a retention program actually saves,
- the average revenue retained per saved customer.

Multiplying these gives a projected saving and a return-on-investment ratio. The
discipline is to present this as a **scenario** ("under these assumptions…"), to
show the arithmetic, and to never describe a projection as an achieved outcome.
A defensible estimate is one a skeptic can recompute and argue with.

---

## 8. Cross-cutting principles

A few rules that governed every choice above:

- **Variable type drives method choice.** Categorical-vs-categorical → chi-square;
  continuous-vs-continuous → correlation; categorical outcome → classification.
- **Significance and strength are different questions.** Always pair a p-value
  with an effect size, especially with a large sample where everything looks
  "significant."
- **Match the metric to the cost structure.** Imbalance and asymmetric costs mean
  accuracy misleads; recall, precision, and a deliberate threshold tell the truth.
- **Separate intrinsic model quality from policy.** Ranking ability is a property
  of the model; the threshold is a business decision laid on top of it.
- **State assumptions for any projection.** A financial estimate is only as
  credible as the assumptions you are willing to defend out loud.

---

## 9. Quick reference

| Situation | Recommended approach | Why |
|---|---|---|
| Summarize a skewed numeric variable | Median + interquartile range | Resistant to outliers |
| Summarize a categorical variable | Counts + proportions | Natural for labels |
| Categorical driver vs. churn | Chi-square test + Cramér's V | Tests association and reports its strength |
| Numeric driver vs. churn | Compare distributions, or band it and use chi-square | Interpretable, assumption-light |
| Two numeric drivers | Correlation coefficient | Detects redundancy between predictors |
| Overall predictive quality | Threshold-independent ranking score | Robust to imbalance, no arbitrary cutoff |
| Operational performance | Recall, precision, F1 at a chosen threshold | Describes real-world consequences |
| Where to set the cutoff | Cost-based threshold selection | Treats the threshold as a business policy |
| Financial impact | Scenario with stated assumptions | Keeps projections honest and checkable |
