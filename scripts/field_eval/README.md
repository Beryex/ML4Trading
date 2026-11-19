# Field evaluation: pilot and order-rule variants

Scripts for the Field Evaluation Design and Model Revision assignment. Run them from the
repository root. Runs go to `runs/`, tables and recommendations to `results/field_eval/`, and the
extended dataset to `data_live/` and `data_pred/`; all of these are git-ignored.

## Pilot (`PILOT_PROTOCOL.md`)

```bash
# once: split the participants into the recommendation and control groups
python scripts/field_eval/assign_groups.py participants.csv --seed 20261118 > assignment.csv

# every evening after the close
export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...        # or --env-file path/to/.env
python scripts/field_eval/extend_data.py --src data --dst data_live
python scripts/field_eval/recommend.py --data-dir data_live
#   -> results/field_eval/recs/<date>.md, the message for the recommendation group
```

`extend_data.py` appends Alpaca SIP 30-minute regular-session bars after the dataset's last
session. On the overlap week it compares closes with the dataset, rescales the source for splits
made after the dataset ends, and rescales Alpaca's volume to the dataset's volume definition
(Alpaca reports about 1.47 times more). Over-the-counter listings have no Alpaca data and simply
end. `recommend.py` refits the frozen v29 configuration for the month of the next session, exactly
as the walk-forward does, and turns that session's predictions into whole-share positions with
the backtest's order rule. `ML4T_WORKERS` (default 4) caps the feature build's processes.

`decision_log_template.csv` lists the fields the participants' decision form collects.

## Order-rule variants (the TA's questions on the Initial ML Solution)

```bash
python scripts/field_eval/order_rule_variants.py            # test path, both frozen models
python scripts/field_eval/order_rule_variants_val.py v29    # validation, fold-mean net Sharpe
python scripts/field_eval/order_rule_variants_val.py P3b
```

The variants re-trade the saved predictions of `runs/report/v29` and `runs/report/p3b` with one
order-rule setting changed: long only, short only, a cost check of 2 or 3 times the round-trip
cost, and long only with 2 times. The unchanged rows reproduce the recorded test metrics and
validation Sharpes exactly.
