---
description: Memory Curator Agent - Scans, cleans, and summarizes the project state to maintain project memory.
---

# Memory Curator Agent Workflow

This workflow is designed to act as a "Memory Curator" for the project. Run this periodically to ensure `PROJECT_SUMMARY.md` is up-to-date, `.gitignore` is effective, and the project "context" is clean.

## Step 1: Context Discovery
1.  **Read Core Documentation**:
    -   Read `PROJECT_SUMMARY.md` to understand the *last known state* of the project.
    -   Read `README.md` to understand the *public face* of the project.
    -   Read `.gitignore` to understand what is currently excluded.

## Step 2: System Scan
1.  **List Files**:
    -   Use `list_dir` on the root directory to get a high-level overview.
    -   Use `find_by_name` or `list_dir` recursively (if needed) to identify new files that might not be in the documentation.
2.  **Identify Recent Changes**:
    -   Look for files that seem new or heavily modified compared to the description in `PROJECT_SUMMARY.md`.

## Step 3: Gitignore & Cleanup Analysis
1.  **Analyze `.gitignore`**:
    -   Are there temporary files (logs, `.DS_Store`, build artifacts, `node_modules` quirks) cluttering the file list that aren't in `.gitignore`?
    -   Are there entries in `.gitignore` that are no longer relevant?
2.  **Propose Cleanup**:
    -   If you find junk files, propose (or perform, if safe) a cleanup.
    -   Update `.gitignore` if necessary to keep the workspace clean.

## Step 4: Logic & Conflict Check
1.  **Architecture Review**:
    -   Briefly review key entry points (e.g., `app.js`, `index.html`, `main.py`) to ensure the *actual* code structure matches the *documented* architecture.
    -   Check for code duplication or conflicting configuration files (e.g., multiple environment files, duplicate requirements).

## Step 5: Memory Update (The Core Task)
1.  **Update `PROJECT_SUMMARY.md`**:
    -   **Summary**: Write a concise, high-level summary of what the project does *now*.
    -   **Recent Changes**: Add a section detailing what seems to have changed recently (new features, refactors).
    -   **Project Structure**: Update the file tree or architecture diagram if the file structure has changed.
    -   **Status**: Update the current status (e.g., "Development", "Maintenance", "Features Complete").
2.  **Context Optimization**:
    -   Ensure `PROJECT_SUMMARY.md` is dense with useful information for future agents (e.g., "Key decisions made: chose X over Y because Z").

## Step 6: Final Verification
1.  Review your changes to `PROJECT_SUMMARY.md`.
2.  Review any changes to `.gitignore`.
3.  Report completion to the user with a summary of the curation.
