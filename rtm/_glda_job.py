"""One Gibbs LDA model (called by rtm.lda.train_lda in a separate process, cwd = third_party/pyGibbsLDA2).

    python _glda_job.py <docs.txt> <T> <iters> <out_dir> <alpha> <beta>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "pyGibbsLDA2"))
sys.dont_write_bytecode = True
import gibbslda  # noqa: E402

docs, T, iters, out_dir = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
alpha, beta = float(sys.argv[5]), float(sys.argv[6])
texts = [l.split() for l in docs.read_text(encoding="utf-8").split("\n")]
out_dir.mkdir(parents=True, exist_ok=True)
g = gibbslda.CGibbsLDA()
g.SetInputDocs(texts)
g.SetParameters(alpha=alpha, beta=beta, topics=T, niters=iters, nitersave=iters, ldamethod="lda",
                fixtopics=0, granwnd=0, out_dir=str(out_dir))
g.StartCalculation()
