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
| 2 | Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge | Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, Nitesh V. Chawla, Xiangliang Zhang | ICLR 2025 (poster); arXiv 2410.02736 | https://arxiv.org/abs/2410.02736 · https://openreview.net/forum?id=3GTtZFiajM · https://llm-judge-bias.github.io/ | Defines the CALM framework and two metrics: **Robustness Rate** RR = (1/\|D\|) Σ 𝟙(yᵢ = ŷᵢ), the fraction of items whose verdict is unchanged after a bias is injected, and **Consistency Rate** CR = (1/\|D\|) Σ 𝟙(yᵢ = yᵢʳᵃⁿᵈ), the fraction unchanged on a second unperturbed run. Both are same-verdict metrics with no reference to correctness, which is the phenomenon the project targets. | Under consideration as second parent paper. Peer-reviewed (ICLR 2025). Code/data availability to be checked against criteria 3–4 before selection. |
| 3 | Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge | Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, Soroush Vosoughi | Proceedings of the 14th IJCNLP and 4th AACL (IJCNLP-AACL 2025), pp. 292–314 | https://aclanthology.org/2025.ijcnlp-long.18/ | Introduces **position consistency**, the stability of a judge's preference when candidate order is swapped, over 150,000+ evaluation instances and 15 judges. Another same-verdict metric; directly comparable to upstream's Auto-J `consistency`. | Under consideration as second parent paper. Peer-reviewed. Code/data availability to be checked against criteria 3–4 before selection. |

## Rejected

| Paper | Authors | Venue / year | URL | Reason for rejection |
|---|---|---|---|---|
| Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias | Justin D. Norman, Michael U. Rivera, D. Alex Hughes (UC Berkeley School of Information) | arXiv 2606.19544, submitted 17 June 2026; no venue stated | https://arxiv.org/abs/2606.19544 | Fails criterion 1 (unrefereed preprint) and criterion 4: the paper states its evaluation library "will be released open source upon publication" and that "We will release the complete evaluation dataset and the [anonymized] code repository upon publication." Neither is available now. Its framing is nonetheless close to ours: it defines test-retest reliability as "the agreement of a judge with itself across independent re-evaluations of the same items ... a necessary but insufficient characteristic of a quality judge," and reports that test-retest reliability above 0.95 coexists with position bias above 0.10. It stays on the related-work list. |

## Verification notes

Bibliographic details above were checked on 2026-09-10/11 against the ACL
Anthology pages (papers 1 and 3), the arXiv PDFs (papers 1, 2 and the rejected
paper) and the OpenReview API (paper 2's ICLR 2025 poster status). The arXiv
version of paper 2 is labelled "Preprint" in its header; the ICLR 2025
acceptance is recorded on OpenReview, not in the arXiv PDF. The RR/CR formulas
are quoted from the "Metrics" paragraph of paper 2's Section 3.
