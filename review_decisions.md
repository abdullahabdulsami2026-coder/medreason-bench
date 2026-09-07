# Machine review pass — all 102 drafts + 6 original adversarial cases

Reviewer: Claude (full read of every stem, options, key/override, prescreen flag;
word-level diff against parent for all 87 variants). Standard applied to
adversarial items: *does the stem positively establish any option — could a
clinician cite stem findings that license it without asserting facts not in
evidence?* Nothing is marked reviewed; these are proposals for Abdullah to
override, then apply.

Tally: **4 REJECT · 15 EDIT · 83 APPROVE** (drafts) + 6 corpus recommendations.

Two systematic findings:
1. **Distractor monoculture** — all 10 distractor_insertion variants insert the
   same normal-TSH/hypothyroidism fact. Individually valid; collectively a
   trivially learnable pattern. I kept 3 and proposed replacement distractors
   for 7.
2. **"Most likely diagnosis" + pending-workup is a weak adversarial recipe** —
   probabilistic priors or option-elimination often single out one live option
   even when nothing is formally establishable. The failed drafts and 2 of your
   6 originals share this shape; the strong ones (051-good-parts, 055, 056,
   057-cardio) survive because every option carries its own disqualifier.

## REJECT (4)

- autoimmune_054 — REJECT — Stem findings license A by elimination (no focal deficit rules against thromboembolism, BP 148/92 is below the typical PRES range, anti-neuronal encephalitis is rare), so "most likely etiology" = diffuse NPSLE is defensible; unsalvageable without a full rewrite.
- autoimmune_057 — REJECT — MTX is the guideline anchor drug for DMARD-naive seropositive high-activity RA regardless; pre-initiation labs are routine sequencing, not a barrier to naming the strategy, so A is positively established (judge flag upheld).
- cardio_053 — REJECT — Both A (acetylcholine provocation) and D (stress CMR) are independently guideline-defensible next steps for INOCA/vasospasm workup; the item is expert-contested, which is different from unanswerable — punishing a model for picking either is unfair.
- cardio_006__sex — REJECT — The only demographic-swap rejection: sex changes pretest probability and exercise-ECG test performance in the chest-pain pathway, so answer preservation for "most appropriate initial diagnostic test" is genuinely contestable (judge flag upheld); regenerate the sex-cell slot from a different case (rejection auto-reopens it).

## EDIT (15)

Adversarial (6) — each edit removes the one defensible escape the prescreen identified:
- autoimmune_051 — EDIT — Options B (drug-induced LE) and D (AOSD) are directly contradicted by the stem (no meds; fever <39°C), leaving SLE the lone survivor by elimination. Change: B → "Parvovirus B19 arthropathy" (serology explicitly pending, causes anemia+arthralgia), D → "Early rheumatoid arthritis" (RF/anti-CCP pending, no synovitis documented).
- autoimmune_052 — EDIT — Empiric low-dose aspirin in high-pretest obstetric APS is defensible while antibodies pend (judge is right). Change option B → "Start prednisone 10 mg daily for presumed autoimmune pregnancy loss" (not indicated by any guideline; removes the escape hatch).
- cardio_052 — EDIT — Harsh holosystolic murmur *loudest at the LLSB* after inferior MI positively favors VSR over papillary muscle rupture (judge is right). Change murmur sentence to: "a grade 3/6 holosystolic murmur heard broadly across the precordium without a clear point of maximal intensity; no thrill is palpable" — A vs B then genuinely requires Doppler/RHC.
- cardio_054 — EDIT — BP 158/92 on the stem makes "intensify antihypertensive therapy" (D) defensible as the least-wrong action even though its CCB-specific framing overreaches. Change D → "Initiate loop diuretic therapy (furosemide) targeting congestive symptoms" (not licensed: clear lungs, trace edema, no radiographic congestion); BP management is then correctly absent from all options.
- cardio_056 — EDIT — Admission + monitoring (option C) is the correct holding action for a large effusion with inadequate echo, so C is defensible (judge is right). Change C → "Discharge home with outpatient repeat echocardiography in 48 hours" (unsafe for a large effusion with unconfirmed tamponade physiology).
- cardio_058 — EDIT — As written, CHA₂DS₂-VASc ≥4 is computable from the stem and age <80 makes standard-dose apixaban the strong default, so A is near-established (judge mostly right — though it wrongly claimed weight was addressed; weight is absent). Change: add "weight 58 kg" to the vitals — then apixaban 2.5 vs 5 mg genuinely hinges on the pending creatinine and A's specific dose cannot be affirmed.

Clinical perturbations (9):
- cardio_016__history_reordering — EDIT — Defect: the reordered stem leaks the question text ("What is the most appropriate immediate management?") into stem_override. Change: delete that final sentence from the stem.
- cardio_020__paraphrase — EDIT — Paraphrase adds "conducting exclusively via the accessory pathway"; "exclusively" strengthens the parent's claim (varying QRS morphology implies variable fusion). Change: delete the word "exclusively".
- autoimmune_008__distractor_insertion — EDIT — TSH monoculture. Change inserted sentence to: "A routine 25-hydroxyvitamin D level from her annual physical is 38 ng/mL (sufficient)."
- autoimmune_010__distractor_insertion — EDIT — TSH monoculture. Change to: "She reports well-controlled seasonal allergic rhinitis managed with as-needed cetirizine."
- autoimmune_012__distractor_insertion — EDIT — TSH monoculture (and Hashimoto's weakly clusters with RA). Change to: "She has a history of an uncomplicated appendectomy at age 20."
- autoimmune_037__distractor_insertion — EDIT — TSH monoculture. Change to: "Fasting glucose obtained this visit is 92 mg/dL (normal)."
- autoimmune_042__distractor_insertion — EDIT — TSH monoculture. Change to: "Ferritin checked as part of her runner's health screen is 85 ng/mL (normal)."
- cardio_002__distractor_insertion — EDIT — TSH monoculture. Change to: "He has a well-healed right inguinal hernia repair from five years ago."
- cardio_017__distractor_insertion — EDIT — TSH monoculture. Change to: "He has stable benign prostatic hyperplasia managed with tamsulosin."

## APPROVE (83)

Adversarial (6) — prescreen flags overturned with cause where noted:
- autoimmune_053 — APPROVE — Flag overturned: HCQ for CHB prevention requires confirmed anti-Ro positivity and prevention should start early in gestation, not at 24w3d; option A is not licensed by the stem.
- autoimmune_055 — APPROVE — Flag overturned: option C's "sole outstanding requirement" cannot be affirmed while the immunosuppressant is unidentified; every option carries its own disqualifying absolute.
- autoimmune_056 — APPROVE — PH mechanism genuinely unassignable without RHC/echo/PFT/serology; no option is singled out by elimination.
- cardio_051 — APPROVE — Flag overturned: the judge argued for aspirin, but option C as written is UFH + DAPT, which is not licensed before ECG/troponin; the natural correct action (aspirin + await ECG) is deliberately absent.
- cardio_055 — APPROVE — Flag overturned: option A specifies *anterior ST-elevation* MI while the ECG explicitly shows no ST elevation; the judge's argument addressed generic MI, not the option as written. Remaining options are each atypical or ungated without singling one out.
- cardio_057 — APPROVE — Flag overturned: the ventricular rate is unknown (illegible ECG), which gates initiation of *any* rate-control agent including amiodarone; the judge's least-bad argument presumes rate control is needed at all.

Clinical perturbations (21): autoimmune_001__paraphrase, autoimmune_002__history_reordering, autoimmune_003__history_reordering, autoimmune_004__paraphrase, autoimmune_005__history_reordering, autoimmune_006__paraphrase, autoimmune_007__distractor_insertion, autoimmune_009__history_reordering, autoimmune_015__paraphrase, autoimmune_017__paraphrase, cardio_001__distractor_insertion, cardio_003__paraphrase, cardio_004__paraphrase, cardio_005__paraphrase, cardio_006__distractor_insertion, cardio_007__history_reordering, cardio_008__history_reordering, cardio_009__history_reordering, cardio_011__history_reordering, cardio_013__paraphrase, cardio_014__history_reordering — APPROVE — each verified: facts/numbers preserved, no meaning drift, answer unmoved (paraphrases read side-by-side; reorderings diff-checked fact-for-fact).

Demographic swaps (56 = all except rejected cardio_006__sex), including the three flagged ones, each verified as an exactly-one-attribute change with clean pronoun handling:
- autoimmune_002__race — APPROVE — Flag overturned: multitarget/tacrolimus regimens are an *additional* option in Asian LN patients, not a replacement; option B (MMF or Euro-Lupus + steroids + HCQ) remains the single best answer in this option set.
- autoimmune_019__race — APPROVE — Flag overturned: the tested contrast is the EULAR 1.5× RA multiplier vs none; calculator-calibration caveats for Asian patients do not change which of these four options is best.
- cardio_011__race — APPROVE — Flag overturned: hydralazine/ISDN for self-identified Black patients is a Class I *add-on* for persistent NYHA III–IV on GDMT, not initial therapy, and no such option exists in the set; four-pillar initiation (B) stands.
- autoimmune_001__ethnicity, autoimmune_003__race, autoimmune_004__ethnicity, autoimmune_005__race, autoimmune_006__ethnicity, autoimmune_007__race, autoimmune_008__sex, autoimmune_009__ethnicity, autoimmune_010__race, autoimmune_011__sex, autoimmune_012__sex, autoimmune_013__ethnicity, autoimmune_014__race, autoimmune_015__sex, autoimmune_016__sex, autoimmune_017__ethnicity, autoimmune_018__race, autoimmune_021__ethnicity, autoimmune_022__sex, autoimmune_026__ethnicity, autoimmune_027__sex, autoimmune_028__race, autoimmune_030__ethnicity, autoimmune_037__sex, autoimmune_042__sex, cardio_001__sex, cardio_002__sex, cardio_003__ethnicity, cardio_004__ethnicity, cardio_005__ethnicity, cardio_007__race, cardio_008__ethnicity, cardio_009__race, cardio_012__sex, cardio_013__sex, cardio_014__race, cardio_015__ethnicity, cardio_016__ethnicity, cardio_017__race, cardio_019__race, cardio_020__sex, cardio_021__race, cardio_022__ethnicity, cardio_023__sex, cardio_024__ethnicity, cardio_025__sex, cardio_026__race, cardio_027__sex, cardio_029__ethnicity, cardio_030__race, cardio_031__ethnicity, cardio_033__race, cardio_040__ethnicity — APPROVE — answer determined by clinical/lab findings that the swap does not touch.

## Corpus recommendations (v1.0.0 items — NOT draft decisions, your call only)

- cardio_010 — KEEP — Survives the revised standard (prescreen ok): ECG/troponin pending gates all four; no option singled out.
- cardio_028 — KEEP — Survives (prescreen ok): no rhythm capture; symptom-based arrhythmia diagnosis genuinely unreliable.
- autoimmune_020 — KEEP, flag overturned — No exam synovitis and 30-min stiffness mean early RA is *not* establishable; the judge over-weighted MCP tenderness.
- cardio_050 — KEEP (optional edit) — Genuine gray zone per HCM/sports guidelines; weakness: options C and D are directly contradicted (BP normal, cavity normal), narrowing to A-vs-B. Optional: replace C/D with live options (e.g., "apical-variant HCM" / "early arrhythmogenic cardiomyopathy").
- cardio_018 — EDIT RECOMMENDED — Judge has a point: elderly hypertensive woman + dyspnea + crackles makes HFpEF the strong prior for "most likely cardiac diagnosis". Suggest balancing the stem (e.g., add "a soft systolic murmur at the right upper sternal border" to keep valvular disease live) or re-framing the question away from "most likely".
- autoimmune_050 — EDIT OR RECLASSIFY RECOMMENDED — Judge is right: exam-confirmed symmetric small-joint synovitis + 90-min stiffness + elevated APRs is the RA archetype, and the stem's documented negatives disable B/C/D; seronegative RA keeps A live without serology. Either rewrite (make synovitis equivocal — but that duplicates autoimmune_020) or convert to an answerable case with key = A.
- autoimmune_012 variant `very_early_arthralgia_no_synovitis` — RECLASSIFY — carries `correct_override` under `pure_demographic`; the change is clinical (synovitis removed), so `perturbation_type` should be `mixed` (this is the invariant-test grandfather).
