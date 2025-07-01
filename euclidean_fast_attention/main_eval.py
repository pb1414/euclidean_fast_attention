from euclidean_fast_attention.eval import run_evaluation

import sys
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'Workdir of the current run.')
flags.DEFINE_string('datafile', None, 'Path to the data file to evaluate.')
flags.DEFINE_string('eval_name', None, 'Name of the evaluation.')

def main(_):
    if FLAGS.workdir is None:
        print("Error: --workdir must be specified.", file=sys.stderr)
        sys.exit(1)
        
    run_evaluation(
        workdir=FLAGS.workdir,
        eval_name=FLAGS.eval_name,
        datafile=FLAGS.datafile
    )

if __name__ == '__main__':
    app.run(main)
