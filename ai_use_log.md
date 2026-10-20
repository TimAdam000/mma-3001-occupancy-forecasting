# AI Use Log

Tool used: Claude (Anthropic), via claude.ai, in a chat session across the project.
Student: Timothy Adam (34990143)

| Area | What the AI did | What I did |
|---|---|---|
| Project plan | Helped revise the plan after I found the environmental sensors could not be joined to the occupancy rooms. Produced the revised plan PDF. | Chose to keep the project occupancy-only, checked the dataset findings against my own Colab outputs. |
| Code (`src/cell0` to `cell7`) | Wrote the first versions of the loading, feature, model, error analysis and sensitivity scripts, and fixed errors I pasted back (DST error, file name, missing import). | Ran every script in Colab on the real data, reported errors, and confirmed the outputs match `reports/results_summary.md`. |
| Explanations | Explained F1, Random Forest, the 60/20/20 chronological split, and what each cell does. | Found the general premise of each concept from provided Colab files and Youtube - Filled in knowledge gaps using AI |
| GitHub setup | Gave step-by-step instructions for the repo, branches, commits and running from a fresh clone. | Did all the steps myself in the GitHub web interface and Colab. |
| Results file | Formatted my Colab outputs into `reports/results_summary.md`. | Supplied all numbers from my own runs. |
| README | Drafted the structure and wording. | Checked file names, run instructions and results against the repo, and edited the text. |
| Module, tests and docs | Rewrote the scripts as documented functions in `src/occupancy/`, wrote the pytest tests, and gave the commands for the test report and pdoc docs. | Uploaded the files, ran the tests and pdoc in Colab (25 passed), fixed a misplaced folder, and checked the outputs. |

## Verification
- Ran all scripts from a fresh `git clone` in Colab and the results matched.
- Compared the README results table to the Colab outputs.
- Ran all 25 pytest tests in Colab from a fresh clone; all passed.

## Limits of AI use
- AI output was not accepted without running it.
- I will not use AI during the presentation or Q&A.
