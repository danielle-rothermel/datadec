"""Print the source quotes for 64 distinct primary reproduction claims.

Run: uv run python scripts/repro/claims.py

Paper: DataDecide: How to Predict Best Pretraining Data with Small Experiments
Authors: Ian Magnusson et al. (2025). License: CC BY 4.0.
Source download: https://arxiv.org/src/2504.11393v2
Archive SHA-256: 20dc7aa3f920fe465ddf2e12d6f72fff6e8bb3567f53e34f5555a6da138542d1

Extract example_paper.tex and tables/pred_error.tex from that archive into
<repository>/data/raw/repro-paper/2504.11393v2/ before running.

Each claim owns one assertion and all its paper locations. IDs use the first
occurrence's number from PR #43's registry at commit
e56139f8a9539cca4c7d9847279b3b02d77d7f5c. Comments on merged entries record
which original claim entries they cover. Different assertions sharing a quote
remain separate, including task-specific and quantitative claims.

Quotes are read from the downloaded v2 source, preserving LaTeX macros.
Table rows give relative error followed by absolute error, both in percent.
Five targets derived from figure readings have no textual quote and are omitted:
DD-0356, DD-0368, DD-0369, DD-0413, DD-0414.
"""

from pathlib import Path

from repro.claims import Claim, QuoteLocation, read_quotes

PAPER_DIR = Path(__file__).resolve().parents[2] / "data/raw/repro-paper/2504.11393v2"

CLAIMS: dict[str, Claim] = {
    # Paper entries: DD-0010, DD-0192
    "DD-0010": Claim(
        statement="Single-scale recipe rankings provide a strong baseline for predicting 1B recipe rankings.",
        locations=(
            QuoteLocation("example_paper.tex", 239, 0, 199, "Abstract"),
            QuoteLocation(
                "example_paper.tex", 391, 267, 375, "Results / Scaling-law comparison"
            ),
        ),
    ),
    "DD-0011": Claim(
        statement="A 150M single-scale ranking gets approximately 80% of pairwise comparisons correct at 1B.",
        locations=(QuoteLocation("example_paper.tex", 239, 0, 199, "Abstract"),),
    ),
    # Paper entries: DD-0013, DD-0054, DD-0180, DD-0181
    "DD-0013": Claim(
        statement="The evaluated scaling-law methods do not exceed the single-scale compute-decision frontier.",
        locations=(
            QuoteLocation("example_paper.tex", 240, 0, 187, "Abstract"),
            QuoteLocation("example_paper.tex", 286, 43, 170, "Introduction"),
            QuoteLocation(
                "example_paper.tex",
                378,
                85,
                183,
                "Figure: scaling-law decision accuracy",
            ),
            QuoteLocation(
                "example_paper.tex", 384, 8, 114, "Results / Scaling-law comparison"
            ),
        ),
    ),
    "DD-0014": Claim(
        statement="Using continuous likelihood proxy metrics makes MMLU more than 80% predictable at the target 1B scale with 0.01% of target compute.",
        locations=(QuoteLocation("example_paper.tex", 240, 188, 437, "Abstract"),),
    ),
    "DD-0015": Claim(
        statement="Using continuous likelihood proxy metrics makes ARC more than 80% predictable at the target 1B scale with 0.01% of target compute.",
        locations=(QuoteLocation("example_paper.tex", 240, 188, 437, "Abstract"),),
    ),
    "DD-0016": Claim(
        statement="Using continuous likelihood proxy metrics makes HellaSwag more than 80% predictable at the target 1B scale with 0.01% of target compute.",
        locations=(QuoteLocation("example_paper.tex", 240, 188, 437, "Abstract"),),
    ),
    "DD-0017": Claim(
        statement="Using continuous likelihood proxy metrics makes MBPP more than 80% predictable at the target 1B scale with 0.01% of target compute.",
        locations=(QuoteLocation("example_paper.tex", 240, 188, 437, "Abstract"),),
    ),
    "DD-0018": Claim(
        statement="Using continuous likelihood proxy metrics makes HumanEval more than 80% predictable at the target 1B scale with 0.01% of target compute.",
        locations=(QuoteLocation("example_paper.tex", 240, 188, 437, "Abstract"),),
    ),
    # Paper entries: DD-0051, DD-0166
    "DD-0051": Claim(
        statement="The compute required for accurate recipe decisions differs substantially across tasks.",
        locations=(
            QuoteLocation("example_paper.tex", 285, 39, 130, "Introduction"),
            QuoteLocation(
                "example_paper.tex",
                367,
                138,
                213,
                "Results / Compute for data decisions",
            ),
        ),
    ),
    # Paper entries: DD-0052, DD-0167
    "DD-0052": Claim(
        statement="MMLU and ARC require less compute than HellaSwag to achieve comparable decision accuracy.",
        locations=(
            QuoteLocation("example_paper.tex", 285, 131, 271, "Introduction"),
            QuoteLocation(
                "example_paper.tex",
                367,
                214,
                290,
                "Results / Compute for data decisions",
            ),
        ),
    ),
    "DD-0053": Claim(
        statement="SocialIQA is difficult to predict at all examined scales.",
        locations=(QuoteLocation("example_paper.tex", 285, 131, 271, "Introduction"),),
    ),
    "DD-0055": Claim(
        statement="At small scales, continuous answer-likelihood metrics are better or equivalent decision predictors than target accuracy.",
        locations=(QuoteLocation("example_paper.tex", 287, 47, 208, "Introduction"),),
    ),
    # Paper entries: DD-0056, DD-0211
    "DD-0056": Claim(
        statement="Lower seed noise and wider between-recipe spread help explain better decision accuracy.",
        locations=(
            QuoteLocation("example_paper.tex", 288, 46, 231, "Introduction"),
            QuoteLocation(
                "example_paper.tex", 429, 8, 146, "Results / Benchmark predictability"
            ),
        ),
    ),
    "DD-0057": Claim(
        statement="Proxy metrics can improve the noise/spread traits that explain decision quality.",
        locations=(QuoteLocation("example_paper.tex", 288, 46, 231, "Introduction"),),
    ),
    "DD-0098": Claim(
        statement="At the 1B 5x Chinchilla scale, between-run standard deviation can reach 2 percentage points of accuracy for some recipes on most tasks.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 304, 420, 585, "Methods / The DataDecide suite"
            ),
        ),
    ),
    "DD-0119": Claim(
        statement="No explored multi-scale variant makes substantially better decisions than the baseline.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 324, 0, 232, "Methods / Prediction methods"
            ),
        ),
    ),
    "DD-0142": Claim(
        statement="All selected tasks except BoolQ receive non-trivial performance at the examined model scales.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                345,
                573,
                689,
                "Methods / Performance evaluation with OLMES",
            ),
        ),
    ),
    # Paper entries: DD-0148, DD-0174
    "DD-0148": Claim(
        statement="Recipe decision accuracy varies substantially between tasks at a given compute budget.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 350, 180, 374, "Figure: per-task decision accuracy"
            ),
            QuoteLocation(
                "example_paper.tex",
                372,
                248,
                514,
                "Results / Compute for data decisions",
            ),
        ),
    ),
    "DD-0149": Claim(
        statement="ARC Easy is predictable at small scales.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 350, 180, 374, "Figure: per-task decision accuracy"
            ),
        ),
    ),
    "DD-0150": Claim(
        statement="HellaSwag requires substantially more compute to predict than ARC Easy.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 350, 180, 374, "Figure: per-task decision accuracy"
            ),
        ),
    ),
    "DD-0164": Claim(
        statement="More compute makes better decisions.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 367, 8, 44, "Results / Compute for data decisions"
            ),
        ),
    ),
    "DD-0165": Claim(
        statement="Intermediate checkpoints make decisions as accurately as compute-equivalent final checkpoints.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                367,
                45,
                137,
                "Results / Compute for data decisions",
            ),
        ),
    ),
    "DD-0168": Claim(
        statement="The remaining OLMES tasks are markedly less reliable across examined scales.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                367,
                291,
                387,
                "Results / Compute for data decisions",
            ),
        ),
    ),
    "DD-0169": Claim(
        statement="Aggregate OLMES decision accuracy has a positive, roughly log-linear relationship with experimental compute.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 370, 0, 217, "Results / Compute for data decisions"
            ),
        ),
    ),
    "DD-0175": Claim(
        statement="ARC Easy remains predictable with five orders of magnitude less compute.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                372,
                248,
                514,
                "Results / Compute for data decisions",
            ),
        ),
    ),
    "DD-0176": Claim(
        statement="BoolQ exceeds trivial decision accuracy only at intermediate checkpoints of target runs.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                372,
                248,
                514,
                "Results / Compute for data decisions",
            ),
        ),
    ),
    "DD-0177": Claim(
        statement="HellaSwag has an insensitive region followed by a roughly log-linear rise after a compute threshold.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                372,
                515,
                673,
                "Results / Compute for data decisions",
            ),
        ),
    ),
    "DD-0178": Claim(
        statement="SocialIQA has an insensitive region followed by a roughly log-linear rise after a compute threshold.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                372,
                515,
                673,
                "Results / Compute for data decisions",
            ),
        ),
    ),
    "DD-0179": Claim(
        statement="WinoGrande has an insensitive region followed by a roughly log-linear rise after a compute threshold.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                372,
                515,
                673,
                "Results / Compute for data decisions",
            ),
        ),
    ),
    "DD-0189": Claim(
        statement="The 2-parameter and 3-parameter variants are among the top decision-accuracy methods.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 389, 400, 469, "Results / Scaling-law comparison"
            ),
        ),
    ),
    "DD-0194": Claim(
        statement="Observed scaling trends cross over frequently.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 391, 376, 517, "Results / Scaling-law comparison"
            ),
        ),
    ),
    "DD-0196": Claim(
        statement="At small scales, character-normalized Correct Prob and Total Prob are better or equivalent decision predictors than target Accuracy.",
        locations=(
            QuoteLocation("example_paper.tex", 400, 8, 238, "Results / Proxy metrics"),
        ),
    ),
    "DD-0197": Claim(
        statement="Proxy-metric trends are similar across length normalizations.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 404, 178, 317, "Results / Proxy metrics"
            ),
        ),
    ),
    "DD-0198": Claim(
        statement="Character normalization is empirically optimal for most observed tasks.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 404, 178, 317, "Results / Proxy metrics"
            ),
        ),
    ),
    "DD-0199": Claim(
        statement="Correct Prob or Total Prob matches or exceeds every other metric at most small scales.",
        locations=(
            QuoteLocation("example_paper.tex", 406, 0, 117, "Results / Proxy metrics"),
        ),
    ),
    "DD-0202": Claim(
        statement="Across tasks, proxy curves follow one of two distinct trend types.",
        locations=(
            QuoteLocation("example_paper.tex", 408, 0, 69, "Results / Proxy metrics"),
        ),
    ),
    "DD-0203": Claim(
        statement="In one trend type, proxy metrics are nearly indistinguishable and improve with compute.",
        locations=(
            QuoteLocation("example_paper.tex", 408, 70, 344, "Results / Proxy metrics"),
        ),
    ),
    "DD-0204": Claim(
        statement="In the other trend type, Correct Prob and Total Prob stay flat while other metrics rise toward them near full target compute.",
        locations=(
            QuoteLocation("example_paper.tex", 408, 70, 344, "Results / Proxy metrics"),
        ),
    ),
    "DD-0205": Claim(
        statement="Within the last order of magnitude below target compute, Accuracy and other metrics tend to overtake Correct Prob and Total Prob.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 408, 345, 553, "Results / Proxy metrics"
            ),
        ),
    ),
    "DD-0206": Claim(
        statement="Correct Prob and Total Prob sometimes decrease in decision accuracy near target compute.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 408, 345, 553, "Results / Proxy metrics"
            ),
        ),
    ),
    "DD-0207": Claim(
        statement="Norm Correct Prob and Margin trend with Accuracy and penalize probability on incorrect answers.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 408, 554, 732, "Results / Proxy metrics"
            ),
        ),
    ),
    "DD-0208": Claim(
        statement="Five tasks benefit at smaller scales from Correct Prob or Total Prob rather than Accuracy, Norm Correct Prob, or Margin.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                414,
                110,
                417,
                "Figure: proxy metric decision accuracy",
            ),
        ),
    ),
    "DD-0209": Claim(
        statement="At 150M with Correct Prob, HellaSwag succeeds with low run-to-run variance.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                422,
                81,
                267,
                "Figure: seed noise and recipe spread",
            ),
        ),
    ),
    "DD-0210": Claim(
        statement="At 150M with Correct Prob, SocialIQA has widely spread performance across data recipes.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                422,
                81,
                267,
                "Figure: seed noise and recipe spread",
            ),
        ),
    ),
    "DD-0212": Claim(
        statement="Correct Prob widens spread or reduces noise for many tasks.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 429, 147, 219, "Results / Benchmark predictability"
            ),
        ),
    ),
    # Paper entries: DD-0213, DD-0221
    "DD-0213": Claim(
        statement="Continuous correct-answer likelihood makes code recipe rankings predictable where accuracy-based predictors are near trivial.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 429, 220, 336, "Results / Benchmark predictability"
            ),
            QuoteLocation(
                "example_paper.tex",
                438,
                13,
                182,
                "Figure: math and code decision accuracy",
            ),
        ),
    ),
    "DD-0218": Claim(
        statement="MMLU predictability is characterized by low run-to-run noise.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 432, 599, 780, "Results / Benchmark predictability"
            ),
        ),
    ),
    "DD-0219": Claim(
        statement="ARC Easy predictability is characterized by wide recipe spread.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 432, 599, 780, "Results / Benchmark predictability"
            ),
        ),
    ),
    "DD-0220": Claim(
        statement="Correct Prob improvements often align with lower noise or wider spread.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 432, 781, 903, "Results / Benchmark predictability"
            ),
        ),
    ),
    "DD-0222": Claim(
        statement="Common math tasks stay near trivial decision accuracy regardless of proxy metric.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                438,
                183,
                270,
                "Figure: math and code decision accuracy",
            ),
        ),
    ),
    "DD-0224": Claim(
        statement="Correct Prob raises code-task decision accuracy from trivial to approximately 80%.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 442, 245, 351, "Results / Benchmark predictability"
            ),
        ),
    ),
    "DD-0225": Claim(
        statement="Correct Prob lets small code-task models exceed the noise floor while predicting large-scale Accuracy.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 442, 352, 491, "Results / Benchmark predictability"
            ),
        ),
    ),
    "DD-0226": Claim(
        statement="Minerva and GSM8K do not gain comparable predictability from switching the proxy to Correct Prob.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 442, 492, 621, "Results / Benchmark predictability"
            ),
        ),
    ),
    "DD-0227": Claim(
        statement="Minerva and GSM8K exceed 80% decision accuracy if the target metric is changed to Correct Prob.",
        locations=(
            QuoteLocation(
                "example_paper.tex", 442, 622, 823, "Results / Benchmark predictability"
            ),
        ),
    ),
    "DD-0301": Claim(
        statement="3-parameter with helpers and >50% checkpoints: relative error 5.6; absolute error 2.6.",
        locations=(
            QuoteLocation(
                "tables/pred_error.tex",
                6,
                0,
                63,
                "Table: scaling-law prediction errors",
            ),
        ),
    ),
    "DD-0302": Claim(
        statement="3-parameter with helper points: relative error 6.0; absolute error 2.8.",
        locations=(
            QuoteLocation(
                "tables/pred_error.tex",
                7,
                0,
                45,
                "Table: scaling-law prediction errors",
            ),
        ),
    ),
    "DD-0303": Claim(
        statement="3-parameter step 2 fit with >50% checkpoints: relative error 5.9; absolute error 2.9.",
        locations=(
            QuoteLocation(
                "tables/pred_error.tex",
                8,
                0,
                62,
                "Table: scaling-law prediction errors",
            ),
        ),
    ),
    "DD-0304": Claim(
        statement="3-parameter: relative error 6.5; absolute error 3.1.",
        locations=(
            QuoteLocation(
                "tables/pred_error.tex",
                9,
                0,
                26,
                "Table: scaling-law prediction errors",
            ),
        ),
    ),
    "DD-0305": Claim(
        statement="2-parameter: relative error 6.5; absolute error 3.2.",
        locations=(
            QuoteLocation(
                "tables/pred_error.tex",
                10,
                0,
                26,
                "Table: scaling-law prediction errors",
            ),
        ),
    ),
    "DD-0306": Claim(
        statement="5-parameter, single step: relative error 42.8; absolute error 17.4.",
        locations=(
            QuoteLocation(
                "tables/pred_error.tex",
                11,
                0,
                41,
                "Table: scaling-law prediction errors",
            ),
        ),
    ),
    "DD-0307": Claim(
        statement="3-parameter, single step: relative error 42.9; absolute error 42.3.",
        locations=(
            QuoteLocation(
                "tables/pred_error.tex",
                12,
                0,
                41,
                "Table: scaling-law prediction errors",
            ),
        ),
    ),
    "DD-0308": Claim(
        statement="5-parameter: relative error 230.8; absolute error 65.4.",
        locations=(
            QuoteLocation(
                "tables/pred_error.tex",
                13,
                0,
                29,
                "Table: scaling-law prediction errors",
            ),
        ),
    ),
    # Paper entries: DD-0311, DD-0330
    "DD-0311": Claim(
        statement="Prediction errors are comparable among scaling-law variants other than single-step and five-parameter methods.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                512,
                157,
                377,
                "Table: scaling-law prediction errors",
            ),
            QuoteLocation(
                "example_paper.tex", 564, 103, 230, "Appendix / Scaling-law variants"
            ),
        ),
    ),
    "DD-0312": Claim(
        statement="Single-step and 5-parameter error patterns roughly follow their compute-decision-frontier performance.",
        locations=(
            QuoteLocation(
                "example_paper.tex",
                512,
                157,
                377,
                "Table: scaling-law prediction errors",
            ),
        ),
    ),
}


def main() -> None:
    """Print each assertion once, followed by all its paper quotes and sections."""
    for claim_id, quotes in read_quotes(CLAIMS, PAPER_DIR).items():
        claim = CLAIMS[claim_id]
        print(f"{claim_id}: {claim.statement}")
        for location, quote in zip(claim.locations, quotes, strict=True):
            print(f"  {location.section} ({location.source_file}:{location.line})")
            print(quote)
        print()


if __name__ == "__main__":
    main()
