"""Metric implementations. Each module documents its formula in the top docstring.

Dataframe-based metrics share one contract — a *results frame* with one
row per (model, seed, item) built by ``run_eval.build_frame``:

    model, seed, item_id, case_id, is_variant, variant_id,
    perturbation_type, perturbation_subtype, changed_attr,
    answer_preserving, specialty, difficulty, correct, unanswerable,
    parsed_answer, abstained, answer_in_options, is_correct,
    confidence (0-1 float, NaN when unparsed), error

``correct`` is the item's own key (variants carry their override), so
``is_correct`` is always judged against the right answer for that row.
"""
