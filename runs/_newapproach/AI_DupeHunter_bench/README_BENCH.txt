AI DupeHunter -- bench copy (runs from source)
===============================================

Contents
  trek_gui.py            the application (start this)
  trek_cache.py          local/shared SQLite cache (compression, network-share tuning)
  trek_index.py          id -> module index
  trek_similarity.py     duplicate scoring (BM25 / embeddings / LLM judge)
  trek_projects.py       project list (Project/Campaign/Config ids, JWT token)
  trek_paths.py          where data files are written
  trek_log.py            activity log
  trek_theme.py          colour themes
  tal\KeywordDrivenBase\Addons\WorkspaceUpdate\TrekExportLinksAPI.py   TREK API client
  optimize_db.py         make a database fast on a network share (see below)
  dupehunter_icon.ico    window icon
  requirements.txt       Python packages
  trek_projects.json     your projects (bmw, HCP4) incl. JWT tokens -- keep private
  setup_bench.bat / run_bench.bat

Setup (once)
  1. Python 3.12 must be installed on the bench (Windows; TREK login uses Windows SSO).
  2. Double-click setup_bench.bat  (creates .venv and installs the packages).

Run
  Double-click run_bench.bat   (or:  .venv\Scripts\python trek_gui.py)

Database
  Created on first start NEXT TO THESE FILES (trek_cache.sqlite3, trek_activity.log,
  trek_settings.json). The "Cache Database" path was removed from the copied
  projects -- to use the shared database, open  Edit Project -> Cache Database
  and select the file on the share, e.g.
      \\iads220n\D\Users\Costi\Database\trek_cache.sqlite3

Testing the network speed-up
  1. Make the shared database network-ready once (app closed, on the PC that has
     the database locally):
        python optimize_db.py --copy-to D:\TREKflow\trek_cache_network.sqlite3
     then copy that file to the share.
  2. On the bench, point Edit Project -> Cache Database at the file on the share.
  3. Start the app: the window opens right away, the ID index loads in the
     background. Log -> "Index loaded in background: ... ids in X s" shows the time.
     Before the optimization this was ~125 s on the share.

Notes
  * Everyone using the shared database needs THIS version (older versions cannot
    read the compressed entries; they re-download them from TREK instead).
  * No trek_projects.json on the bench? The app asks for Project/Campaign/Config
    id and the JWT token on first start.
