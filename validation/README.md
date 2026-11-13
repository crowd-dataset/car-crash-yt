# Validation kit

Two reviewers label a stratified sample of crash and walking clips without
seeing the pipeline's labels; `analyse.py` then compares their answers with the
labels. The sample has two halves of 200 clips. So far both reviewers have
labelled the same half, so their answers can also be compared with each other;
the other half is still to be reviewed. Reviewers should
read [GUIDE.md](GUIDE.md) first: it defines every answer, with tie-break rules
and example clips; the review page links to it.

1. Draw the sample (already done; `sample.json` is committed, seed 2027):

       python validation/make_sample.py --crash-segments crash_segments.csv \
           --walking-mapping ../walking-yt/mapping.csv

2. Start the server and send the reviewers the address:

       python validation/server.py --host 0.0.0.0 --video-dir /path/to/videos

   Reviewers type their name. The first two names get the two halves; typing
   the same name later continues at the first unanswered clip. Clips that do
   not play on YouTube move to the end of that reviewer's list while the
   snippet is cut from `--video-dir` (full videos named `<video_id>.mp4`) or
   downloaded with yt-dlp (add `--cookies cookies.txt` if YouTube asks to sign
   in). `--retry-failed` tries failed downloads again. Progress:
   `http://<host>:8765/api/status`.

3. Compare with the newest labels:

       python validation/analyse.py --crash-segments crash_segments.csv

Answers are in `validation/data/state.json` (not in git: it has the reviewers'
names). Questions and options are in `questions.json`.
