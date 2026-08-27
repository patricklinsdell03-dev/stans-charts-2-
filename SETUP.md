# Setup, and the one thing that goes wrong

## The failure you are most likely to hit

**Symptom:** the Actions tab never asks you to enable workflows, and after
setting up Pages there is no workflow to run.

**Cause:** the `.github` folder did not upload. GitHub's web uploader, and most
operating systems' unzip and drag-and-drop, silently skip folders whose name
begins with a dot. There is no error. The folder simply is not there.

**Confirm it in one step.** Open this URL, substituting your details:

    https://github.com/YOUR-NAME/YOUR-REPO/blob/main/.github/workflows/weekly-scan.yml

A 404 means the folder is missing, which is the whole problem. If the file loads,
the cause is something else and the checklist further down applies.

**Fix it from a browser, phone included.** Do not fight the uploader.

1. In the repository, choose **Add file**, then **Create new file**.
2. In the filename box type `.github/workflows/weekly-scan.yml` in full. Typing
   each `/` turns the preceding text into a folder as you go, which is how you
   create a dot-folder in the web interface.
3. Paste the contents of `WORKFLOW-COPY-ME.yml` from this project, minus its
   comment header, and commit.
4. Reload the Actions tab. The workflow appears, with a **Run workflow** button
   on the right.

That header note is why `WORKFLOW-COPY-ME.yml` sits in the root of this project
rather than only in `.github/`. Delete it once the real one is in place.

## If the file is there and it still does not run

Work down this list in order.

**Default branch.** Settings, then Branches. `workflow_dispatch` only offers a
Run workflow button for workflows on the default branch. If yours is `master`
rather than `main`, either rename it or change the Pages source to match.

**Actions permissions.** Settings, then Actions, then General. "Allow all
actions and reusable workflows" must be selected. A brand new account sometimes
also needs its email verified before any workflow will start.

**Write permission.** The workflow commits the regenerated dashboard back to the
repository. Settings, then Actions, then General, then Workflow permissions:
"Read and write permissions" must be selected, otherwise the run does all the
work and fails on the final push.

**Look at the run, not the tab.** If a run started and failed, the Actions tab
shows a red cross rather than nothing at all. Open it and read the failing step.
The scan itself takes fifteen to thirty minutes on the first run because it is
downloading 1,250 price histories.

## The rest of the setup

1. Free account at github.com.
2. New repository, public. GitHub Pages only serves public repositories on the
   free tier. If you would rather keep it private, skip the Pages step and take
   the dashboard from each run's artifacts instead.
3. Upload everything from inside the `weinstein-tracker` folder, keeping the
   folder structure. Assume `.github` did not make it and use the Create new
   file method above for the workflow.
4. Settings, then Pages. Source: Deploy from a branch. Branch `main`, folder
   `/docs`. Save.
5. Actions, then Weekly Weinstein scan, then Run workflow.
6. When it finishes the dashboard is at `yourname.github.io/yourrepo`.

Before the first real order, run `./run_tests.sh` once from a laptop. Five
suites, a few seconds, no network needed. It is the only evidence you have that
the arithmetic does what it claims.
