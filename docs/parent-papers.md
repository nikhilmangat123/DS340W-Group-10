# Parent-paper candidates

Project theme: reliability metrics for LLM-as-a-Judge. The working thesis is
that the field's headline reliability metrics measure whether a verdict
*stayed the same* under some perturbation, not whether it was *right*, and
that a deterministic, consistently wrong judge maximises all of them. The
planned contribution decomposes such metrics by correctness into four cells:
consistent+correct, consistent+wrong, flipped-to-correct, flipped-to-wrong.

## Assignment criteria

1. Peer-reviewed, published 2024 or later.
2. Detailed methodology, evaluation results and conclusions.
3. Evaluation datasets publicly accessible.
4. Code publicly available.

## Candidates

| # | Paper | Authors | Venue / year | URL | Why it is in scope | Status |
|---|---|---|---|---|---|---|
| 1 | **An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Model is not a General Substitute for GPT-4** | Hui Huang, Xingyuan Bu, Hongli Zhou, Yingqi Qu, Jing Liu, Muyun Yang, Bing Xu, Tiejun Zhao | Findings of the Association for Computational Linguistics: ACL 2025, pp. 5880–5895 | https://aclanthology.org/2025.findings-acl.306/ · code https://github.com/HuihuiChyan/UnlimitedJudge | Evaluates four fine-tuned judges against GPT-4 on eight public test sets and already computes a forward/reverse `consistency` metric alongside `agreement` on Auto-J, which is exactly the same-verdict-vs-correct pairing our decomposition needs. Code and all data are in one public repository and the metric path runs on CPU. | **Selected as first parent paper.** Meets all four criteria; see `decisions.md` D1. |
| 2 | Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge | Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, Nitesh V. Chawla, Xiangliang Zhang | ICLR 2025 (poster); arXiv 2410.02736 | https://arxiv.org/abs/2410.02736 · https://openreview.net/forum?id=3GTtZFiajM · https://llm-judge-bias.github.io/ | Defines the CALM framework and two metrics: **Robustness Rate** RR = (1/\|D\|) Σ 𝟙(yᵢ = ŷᵢ), the fraction of items whose verdict is unchanged after a bias is injected, and **Consistency Rate** CR = (1/\|D\|) Σ 𝟙(yᵢ = yᵢʳᵃⁿᵈ), the fraction unchanged on a second unperturbed run. Both are same-verdict metrics with no reference to correctness, which is the phenomenon the project targets. | Related work, not a parent paper. Peer-reviewed (ICLR 2025) and data are public, but the CALM evaluation code has not been released (fails criterion 4). See the availability check below. |
| 3 | Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge | Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, Soroush Vosoughi | Proceedings of the 14th IJCNLP and 4th AACL (IJCNLP-AACL 2025), pp. 292–314 | https://aclanthology.org/2025.ijcnlp-long.18/ | Introduces **position consistency**, the stability of a judge's preference when candidate order is swapped, over 150,000+ evaluation instances and 15 judges. Another same-verdict metric; directly comparable to upstream's Auto-J `consistency`. | Leading candidate for second parent paper, with caveats. Peer-reviewed. Analysis code is public but the paper's own link has expired and the ~150k judgment instances were not released. See the availability check below. |

## Rejected

| Paper | Authors | Venue / year | URL | Reason for rejection |
|---|---|---|---|---|
| Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias | Justin D. Norman, Michael U. Rivera, D. Alex Hughes (UC Berkeley School of Information) | arXiv 2606.19544, submitted 17 June 2026; no venue stated | https://arxiv.org/abs/2606.19544 | Fails criterion 1 (unrefereed preprint) and criterion 4: the paper states its evaluation library "will be released open source upon publication" and that "We will release the complete evaluation dataset and the [anonymized] code repository upon publication." Neither is available now. Its framing is nonetheless close to ours: it defines test-retest reliability as "the agreement of a judge with itself across independent re-evaluations of the same items ... a necessary but insufficient characteristic of a quality judge," and reports that test-retest reliability above 0.95 coexists with position bias above 0.10. It stays on the related-work list. |

## Code and data availability check (criteria 3 and 4)

Checked 2026-09-11 by following every link in each paper's PDF, its ACL
Anthology or arXiv page, its project website, the Hugging Face papers page
(where Papers with Code now redirects), and GitHub search and user profiles.
Repository metadata below comes from the GitHub API on that date.

### Ye et al. (ICLR 2025)

| | finding |
|---|---|
| Data (criterion 3) | **Public, no license.** The project site links https://github.com/Y0oMu/LLM-Judge-Bias-Dataset (created 2026-01-15, 4 stars). Contents: `base_datasets/` (alignment 439, fact-related 500, refinement-aware 150 entries) and `bias_datasets/` (sentiment, verbosity, fallacy-oversight, authority, refinement), all JSON, about 2,400 entries, plus one `promptTemplate.py`. No LICENSE file; the README says to consult the original sources (DPO sets, GSM8K, MATH, ScienceQA, CommonsenseQA, Quora-QuAD, TruthfulQA) for terms. |
| Code (criterion 4) | **Not public.** The CALM evaluation code that computes RR and CR is not in the dataset repository, not linked from https://llm-judge-bias.github.io/, not in the arXiv PDF (the only URL in the paper is the project site), not on the Hugging Face papers page, and not under the senior authors' GitHub accounts. The only code released is the prompt-template file. RR and CR are one-line formulas and trivial to reimplement, but the authors' implementation is unavailable. |
| Verdict | Fails criterion 4. Kept as related work. |

### Shi et al. (IJCNLP-AACL 2025)

| | finding |
|---|---|
| Code (criterion 4) | **Public, but reached by name match, not by the paper's link.** Appendix A ("Reproducibility") gives `https://anonymous.4open.science/r/Position-Bias-Analyzer-Demo-F7E3`, which now returns HTTP 410 `repository_expired`. A repository with the identical name exists at https://github.com/Slimshilin/Position-Bias-Analyzer-Demo (created 2024-12-14, 1 star, 16 Python files, no LICENSE). The owner's GitHub bio reads "Undergrad 25.5' at Dartmouth", consistent with first author Lin Shi and the Vosoughi lab at Dartmouth, but the README does not name the paper. It contains the analysis code (position consistency, preference fairness, judge agreement, length and win-rate factors) and scripts that run judges through OpenAI, Anthropic and Gemini API keys. |
| Data (criterion 3) | **Partial.** The repository ships a single 54 KB example slice, `QA_data/example/MTBench_data_sliced.xlsx`. The roughly 150,000 judgment instances analysed in the paper were not found anywhere. The source benchmarks are public: MT-Bench human judgments (https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) and DevBench (https://github.com/open-compass/DevBench). Regenerating the judgments cost the authors "approximately 3,000 USD" in API calls (Appendix A). |
| Verdict | Meets criterion 3 for the source benchmarks and criterion 4 for the analysis code, with two caveats that a write-up must state: the published link is dead and the live repository is unlicensed and not cited from the paper. Its metrics can be recomputed on this repository's Auto-J forward/reverse pairs, which avoids the missing judgment data. |

### Comparison with the selected paper

Huang et al. remains the only candidate whose peer-reviewed paper, code and
all evaluation data are in one public place and run end to end on CPU.

## Verification notes

Bibliographic details above were checked on 2026-09-10/11 against the ACL
Anthology pages (papers 1 and 3), the arXiv PDFs (papers 1, 2 and the rejected
paper) and the OpenReview API (paper 2's ICLR 2025 poster status). The arXiv
version of paper 2 is labelled "Preprint" in its header; the ICLR 2025
acceptance is recorded on OpenReview, not in the arXiv PDF. The RR/CR formulas
are quoted from the "Metrics" paragraph of paper 2's Section 3. Repository
facts in the availability check were read from the GitHub API and the
repositories' README files on 2026-09-11.
