# Code Change Log

Used to record every code change made in this repository.

## Entry Template

### YYYY-MM-DD HH:MM
- Summary:
- Files:
- Notes:

## Entries

### 2026-03-12 00:00
- Summary: Created the code change log document for future code update records.
- Files: `CODE_CHANGE_LOG.md`
- Notes: Starting now, each code change should also be recorded here.

### 2026-03-12 15:42
- Summary: Fixed auto-focus centering for portrait or narrow images so zoomed content stays centered on axes that do not fill the viewer.
- Files: `anno_refiner_app/src/ui/components.py`, `CODE_CHANGE_LOG.md`
- Notes: Also updated scrollbar visibility so horizontal or vertical sliders are hidden when there is no actual pan range on that axis.

### 2026-03-12 16:28
- Summary: Started XML annotation support on branch `codex/feature-xml-annotations` and added end-to-end `.xml` handling across analysis, dashboard visualization, annotator loading, and output saving.
- Files: `anno_refiner_app/src/core/label_utils.py`, `anno_refiner_app/src/core/analyzer.py`, `anno_refiner_app/src/core/file_manager.py`, `anno_refiner_app/src/ui/page_annotator.py`, `anno_refiner_app/src/ui/page_dashboard.py`, `CODE_CHANGE_LOG.md`
- Notes: Added Pascal VOC XML read/write support, generic label key/path resolution for `.txt/.xml`, XML-aware pending/statistics logic, and class-name mapping support for analysis and UI. Verified with `py_compile` and a local XML read/write smoke test.
