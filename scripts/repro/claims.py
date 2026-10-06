r"""Print the paper quotes for the 74 text-backed primary reproduction claims.

Run from the repository root: python3 scripts/repro/claims.py

Paper: DataDecide: How to Predict Best Pretraining Data with Small Experiments
Authors: Ian Magnusson et al. (2025). License: CC BY 4.0.
Source download: https://arxiv.org/src/2504.11393v2
Archive SHA-256: 20dc7aa3f920fe465ddf2e12d6f72fff6e8bb3567f53e34f5555a6da138542d1

Claim IDs retain the primary-target numbering from PR #43's registry at
commit e56139f8a9539cca4c7d9847279b3b02d77d7f5c. Repeated statements keep
separate IDs. Quotes are exact source substrings, including LaTeX macros;
source comments refer to files and lines inside the downloaded archive.
The table rows quote relative error followed by absolute error, both in percent.

Five targets derived from figure readings have no textual quote and are omitted:
DD-0356, DD-0368, DD-0369, DD-0413, DD-0414.
"""

CLAIMS: dict[str, str] = {
    # example_paper.tex:239
    "DD-0010": (
        r"We find that the ranking of models at a single, small size (e.g., 150M "
        r"parameters) is a strong baseline for predicting best models at our "
        r"larger target scale (1B) ($\sim80$\% of comparisons correct)."
    ),
    # example_paper.tex:239
    "DD-0011": (
        r"We find that the ranking of models at a single, small size (e.g., 150M "
        r"parameters) is a strong baseline for predicting best models at our "
        r"larger target scale (1B) ($\sim80$\% of comparisons correct)."
    ),
    # example_paper.tex:240
    "DD-0013": (
        r"No scaling law methods among \numScalingLaws{} baselines exceed the "
        r"compute-decision frontier of single-scale predictions, but \projSuite{} "
        r"can measure improvement in future scaling laws."
    ),
    # example_paper.tex:240
    "DD-0014": (
        r"We also identify that using continuous likelihood metrics as proxies in "
        r"small experiments makes benchmarks including \mmlu{}, \arc{}, "
        r"\hellaswag{}, \mbpp{}, and \humaneval{} $>80$\% predictable at the "
        r"target 1B scale with just 0.01\% of the compute."
    ),
    # example_paper.tex:240
    "DD-0015": (
        r"We also identify that using continuous likelihood metrics as proxies in "
        r"small experiments makes benchmarks including \mmlu{}, \arc{}, "
        r"\hellaswag{}, \mbpp{}, and \humaneval{} $>80$\% predictable at the "
        r"target 1B scale with just 0.01\% of the compute."
    ),
    # example_paper.tex:240
    "DD-0016": (
        r"We also identify that using continuous likelihood metrics as proxies in "
        r"small experiments makes benchmarks including \mmlu{}, \arc{}, "
        r"\hellaswag{}, \mbpp{}, and \humaneval{} $>80$\% predictable at the "
        r"target 1B scale with just 0.01\% of the compute."
    ),
    # example_paper.tex:240
    "DD-0017": (
        r"We also identify that using continuous likelihood metrics as proxies in "
        r"small experiments makes benchmarks including \mmlu{}, \arc{}, "
        r"\hellaswag{}, \mbpp{}, and \humaneval{} $>80$\% predictable at the "
        r"target 1B scale with just 0.01\% of the compute."
    ),
    # example_paper.tex:240
    "DD-0018": (
        r"We also identify that using continuous likelihood metrics as proxies in "
        r"small experiments makes benchmarks including \mmlu{}, \arc{}, "
        r"\hellaswag{}, \mbpp{}, and \humaneval{} $>80$\% predictable at the "
        r"target 1B scale with just 0.01\% of the compute."
    ),
    # example_paper.tex:285
    "DD-0051": (
        r"The amount of compute you need to allocate for a given \twoClass{} "
        r"depends heavily on task."
    ),
    # example_paper.tex:285
    "DD-0052": (
        r"\mmlu{} and \arc{} are much cheaper to predict than \hellaswag{} and "
        r"some tasks such as \socialiqa{} are difficult to predict at all scales."
    ),
    # example_paper.tex:285
    "DD-0053": (
        r"\mmlu{} and \arc{} are much cheaper to predict than \hellaswag{} and "
        r"some tasks such as \socialiqa{} are difficult to predict at all scales."
    ),
    # example_paper.tex:286
    "DD-0054": (
        r"\numScalingLaws{} baseline scaling law methods do not exceed the compute"
        r" to decision accuracy frontier set by \rankingMethod{}."
    ),
    # example_paper.tex:287
    "DD-0055": (
        r"At small scales, continuous metrics using answer likelihood are better "
        r"or equivalent predictors of decisions than using the same discrete "
        r"accuracy target metric."
    ),
    # example_paper.tex:288
    "DD-0056": (
        r"Better decisions can be explained in part by low run-to-run variance and"
        r" a wide spread of benchmark performance values for different data, "
        r"traits which can be improved by proxy metrics."
    ),
    # example_paper.tex:288
    "DD-0057": (
        r"Better decisions can be explained in part by low run-to-run variance and"
        r" a wide spread of benchmark performance values for different data, "
        r"traits which can be improved by proxy metrics."
    ),
    # example_paper.tex:304
    "DD-0098": (
        r"For instance, we find that the standard deviation between runs at the 1B"
        r" 5$\times C$ scale can be as high as $2\%$ points of accuracy for some "
        r"recipes on most tasks."
    ),
    # example_paper.tex:324
    "DD-0119": (
        r"We explore variations for a total of \numScalingLaws{} multi scale "
        r"approaches defined in Appendix~\ref{app:alternative-scaling-law-fits}; "
        r"none of these make for substantially better decisions than the method "
        r"defined in this section."
    ),
    # example_paper.tex:345
    "DD-0142": (
        r"These tasks are well suited for the model scales we examine with all but"
        r" \boolq{} receiving non-trivial performance."
    ),
    # example_paper.tex:350
    "DD-0148": (
        r"Specific tasks have very distinct ranges of sensitivity, with some like "
        r"\arcEasy{} being predictable at small scales and others like "
        r"\hellaswag{} requiring substantially more compute to predict."
    ),
    # example_paper.tex:350
    "DD-0149": (
        r"Specific tasks have very distinct ranges of sensitivity, with some like "
        r"\arcEasy{} being predictable at small scales and others like "
        r"\hellaswag{} requiring substantially more compute to predict."
    ),
    # example_paper.tex:350
    "DD-0150": (
        r"Specific tasks have very distinct ranges of sensitivity, with some like "
        r"\arcEasy{} being predictable at small scales and others like "
        r"\hellaswag{} requiring substantially more compute to predict."
    ),
    # example_paper.tex:367
    "DD-0164": (r"More compute makes better decisions."),
    # example_paper.tex:367
    "DD-0165": (
        r"Decisions from intermediate checkpoints are as good as compute "
        r"equivalent final checkpoints."
    ),
    # example_paper.tex:367
    "DD-0166": (
        r"The amount of compute needed to make good predictions varies between "
        r"tasks."
    ),
    # example_paper.tex:367
    "DD-0167": (
        r"\arc{} and \mmlu{} are predictable with much less compute than "
        r"\hellaswag{}."
    ),
    # example_paper.tex:367
    "DD-0168": (
        r"The rest of \olmes{} tasks give markedly less reliable predictions "
        r"across the scales we examine."
    ),
    # example_paper.tex:370
    "DD-0169": (
        r"First looking at the aggregation of all 10 \olmes{} tasks "
        r"(Figure~\ref{fig:accuracy_vs_compute} right), we see that there is a "
        r"positive and roughly log-linear relationship between experimental "
        r"compute and \twoClass{}."
    ),
    # example_paper.tex:372
    "DD-0174": (
        r"The predictive sensitivity of tasks at a given compute varies "
        r"significantly, with \arcEasy{} being consistently predictable with 5 "
        r"orders of magnitude less compute and \boolq{} only reaching beyond "
        r"trivial \twoClass{} for intermediate checkpoints of the target runs."
    ),
    # example_paper.tex:372
    "DD-0175": (
        r"The predictive sensitivity of tasks at a given compute varies "
        r"significantly, with \arcEasy{} being consistently predictable with 5 "
        r"orders of magnitude less compute and \boolq{} only reaching beyond "
        r"trivial \twoClass{} for intermediate checkpoints of the target runs."
    ),
    # example_paper.tex:372
    "DD-0176": (
        r"The predictive sensitivity of tasks at a given compute varies "
        r"significantly, with \arcEasy{} being consistently predictable with 5 "
        r"orders of magnitude less compute and \boolq{} only reaching beyond "
        r"trivial \twoClass{} for intermediate checkpoints of the target runs."
    ),
    # example_paper.tex:372
    "DD-0177": (
        r"\hellaswag{}, \socialiqa{}, \winogrande{} show distinct periods of "
        r"insensitivity followed by roughly log-linear increase after hitting some"
        r" compute threshold."
    ),
    # example_paper.tex:372
    "DD-0178": (
        r"\hellaswag{}, \socialiqa{}, \winogrande{} show distinct periods of "
        r"insensitivity followed by roughly log-linear increase after hitting some"
        r" compute threshold."
    ),
    # example_paper.tex:372
    "DD-0179": (
        r"\hellaswag{}, \socialiqa{}, \winogrande{} show distinct periods of "
        r"insensitivity followed by roughly log-linear increase after hitting some"
        r" compute threshold."
    ),
    # example_paper.tex:378
    "DD-0180": (
        r"At best, these approaches reach only the same compute to \twoClass{} "
        r"frontier as \rankingMethod{}."
    ),
    # example_paper.tex:384
    "DD-0181": (
        r"A selection of \numScalingLaws{} baseline scaling law methods are no "
        r"more efficient than \rankingMethod{}."
    ),
    # example_paper.tex:389
    "DD-0189": (
        r"The 2 and 3 parameter variants all achieve among the top \twoClass{}."
    ),
    # example_paper.tex:391
    "DD-0192": (
        r"Nevertheless \rankingMethod{} sets a high baseline \twoClass{}, implying"
        r" relatively little crossover occurs."
    ),
    # example_paper.tex:391
    "DD-0194": (
        r"It is difficult to distinguish evaluation variance from true crossovers,"
        r" but the scaling trends we empirically observe cross over frequently."
    ),
    # example_paper.tex:400
    "DD-0196": (
        r"At small scales, continuous metrics using the character normalized "
        r"likelihood of correct or all answer options serve as better or "
        r"equivalent predictors of decisions than using the same \primaryMetric{} "
        r"as used at the target scale."
    ),
    # example_paper.tex:404
    "DD-0197": (
        r"Metrics follow similar trends regardless of length normalization and "
        r"this one is empirically optimal for most of the tasks that we observe."
    ),
    # example_paper.tex:404
    "DD-0198": (
        r"Metrics follow similar trends regardless of length normalization and "
        r"this one is empirically optimal for most of the tasks that we observe."
    ),
    # example_paper.tex:406
    "DD-0199": (
        r"Using \correctProb{} or \totalProb{} leads to \twoClass{} at least as "
        r"good as any other metric for most small scales."
    ),
    # example_paper.tex:408
    "DD-0202": (
        r"We notice two very distinct types of trends over the different tasks."
    ),
    # example_paper.tex:408
    "DD-0203": (
        r"Either the different proxy metrics are nearly indistinguishable and "
        r"increase in \twoClass{} with compute or \correctProb{} and \totalProb{} "
        r"are flat with respect to scale and the other metrics only rise up to "
        r"that level of \twoClass{} towards the full target compute budget."
    ),
    # example_paper.tex:408
    "DD-0204": (
        r"Either the different proxy metrics are nearly indistinguishable and "
        r"increase in \twoClass{} with compute or \correctProb{} and \totalProb{} "
        r"are flat with respect to scale and the other metrics only rise up to "
        r"that level of \twoClass{} towards the full target compute budget."
    ),
    # example_paper.tex:408
    "DD-0205": (
        r"In the last order of magnitude below the target compute \primaryMetric{}"
        r" and the other metrics tend to overtake \correctProb{} and \totalProb{},"
        r" while these two metrics sometimes even decrease in \twoClass{}."
    ),
    # example_paper.tex:408
    "DD-0206": (
        r"In the last order of magnitude below the target compute \primaryMetric{}"
        r" and the other metrics tend to overtake \correctProb{} and \totalProb{},"
        r" while these two metrics sometimes even decrease in \twoClass{}."
    ),
    # example_paper.tex:408
    "DD-0207": (
        r"Notably these other metrics that trend with \primaryMetric{} include "
        r"continuous metrics that penalize probability assigned to incorrect "
        r"answers, \normCorrectProb{} and \margin{}."
    ),
    # example_paper.tex:414
    "DD-0208": (
        r"5 tasks benefit at smaller scales from using raw likelihood of answers "
        r"({\color{orange}\correctProb{}} and {\color{red}\totalProb{}}), as "
        r"opposed to discrete \primaryMetric{} or continuous metrics that penalize"
        r" probability on incorrect answers ({\color{purple}\normCorrectProb{}}, "
        r"{\color{green}\margin{}})."
    ),
    # example_paper.tex:422
    "DD-0209": (
        r"At 150M with \correctProb{} tasks like \hellaswag{} succeed with low "
        r"run-to-run variance and tasks like \socialiqa{} widely spread the "
        r"performance assigned to different pretraining data."
    ),
    # example_paper.tex:422
    "DD-0210": (
        r"At 150M with \correctProb{} tasks like \hellaswag{} succeed with low "
        r"run-to-run variance and tasks like \socialiqa{} widely spread the "
        r"performance assigned to different pretraining data."
    ),
    # example_paper.tex:429
    "DD-0211": (
        r"The \twoClass{} on a task is driven in part by low run-to-run variance "
        r"and a wide spread of performance values for different data recipes."
    ),
    # example_paper.tex:429
    "DD-0212": (
        r"Using \correctProb{} sees wider spreads or reduced noise for many tasks."
    ),
    # example_paper.tex:429
    "DD-0213": (
        r"Using this metric enables predicting rankings for code tasks that are "
        r"too hard for accuracy metrics at small scales."
    ),
    # example_paper.tex:432
    "DD-0218": (
        r"We see that some highly predictable tasks (e.g., \mmlu{}) are "
        r"characterized by having low run-to-run noise, while others (e.g., "
        r"\arcEasy{}) widely spread the different data recipes."
    ),
    # example_paper.tex:432
    "DD-0219": (
        r"We see that some highly predictable tasks (e.g., \mmlu{}) are "
        r"characterized by having low run-to-run noise, while others (e.g., "
        r"\arcEasy{}) widely spread the different data recipes."
    ),
    # example_paper.tex:432
    "DD-0220": (
        r"We also see that improvements from using \correctProb{} often align with"
        r" improvements in one of these two characteristics."
    ),
    # example_paper.tex:438
    "DD-0221": (
        r"Code tasks such as humaneval and MBPP go from trivial \twoClass{} to "
        r"largely predictable when using using continuous \correctProb{} instead "
        r"of discrete \primaryMetric{}."
    ),
    # example_paper.tex:438
    "DD-0222": (
        r"Meanwhile common math tasks remain near trivial decision accuracy "
        r"regardless of metric."
    ),
    # example_paper.tex:442
    "DD-0224": (
        r"Figure~\ref{fig:math_and_code} shows how \twoClass{} goes from trivial "
        r"to ~80\% when using \correctProb{}."
    ),
    # example_paper.tex:442
    "DD-0225": (
        r"The switch of metric allows small models to get above the noise floor "
        r"for these tasks, while still predicting large-scale accuracy metrics."
    ),
    # example_paper.tex:442
    "DD-0226": (
        r"Notably, two math benchmarks "
        r"\citep{lewkowycz2022solvingquantitativereasoningproblems, "
        r"cobbe2021gsm8k} do not see such a benefit."
    ),
    # example_paper.tex:442
    "DD-0227": (
        r"They do however give \twoClass{} above 80\% if we switch the "
        r"\textit{target metric} to \correctProb{}, raising a question for future "
        r"work to explore whether changing the target metric can be justified."
    ),
    # tables/pred_error.tex:6
    "DD-0301": (r"3-parameter with helpers and $>$50\% checkpoints & 5.6 & 2.6 \\"),
    # tables/pred_error.tex:7
    "DD-0302": (r"3-parameter with helper points & 6.0 & 2.8 \\"),
    # tables/pred_error.tex:8
    "DD-0303": (r"3-parameter step 2 fit with $>$50\% checkpoints & 5.9 & 2.9 \\"),
    # tables/pred_error.tex:9
    "DD-0304": (r"3-parameter & 6.5 & 3.1 \\"),
    # tables/pred_error.tex:10
    "DD-0305": (r"2-parameter & 6.5 & 3.2 \\"),
    # tables/pred_error.tex:11
    "DD-0306": (r"5-parameter, single step & 42.8 & 17.4 \\"),
    # tables/pred_error.tex:12
    "DD-0307": (r"3-parameter, single step & 42.9 & 42.3 \\"),
    # tables/pred_error.tex:13
    "DD-0308": (r"5-parameter & 230.8 & 65.4 \\"),
    # example_paper.tex:512
    "DD-0311": (
        r"We see that other than the single step and 5-parameter variants errors "
        r"are comparable, and these variants also roughly follow the "
        r"compute-decision frontier in "
        r"Figure~\ref{fig:all_scaling_laws_accuracy_vs_compute_shaded}."
    ),
    # example_paper.tex:512
    "DD-0312": (
        r"We see that other than the single step and 5-parameter variants errors "
        r"are comparable, and these variants also roughly follow the "
        r"compute-decision frontier in "
        r"Figure~\ref{fig:all_scaling_laws_accuracy_vs_compute_shaded}."
    ),
    # example_paper.tex:564
    "DD-0330": (
        r"As the best scaling laws variants are all roughly comparable to the "
        r"simple 3-parameter set up, we use this one as our baseline."
    ),
}


def main() -> None:
    """Print each claim ID followed by its exact paper quote."""
    for claim_id, quote in CLAIMS.items():
        print(f"{claim_id}\n{quote}\n")


if __name__ == "__main__":
    main()
