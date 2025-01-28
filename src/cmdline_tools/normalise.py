# Standard library
import argparse
import errno
import pathlib
import sys
import warnings

# Third party
import numpy as np

# Internal
import zignal


def main():
    parser = argparse.ArgumentParser(
        description="Normalise a wavfile")

    parser.add_argument("--version", action="version",
                        version="%(prog)s " + "%s" % zignal.__version__)

    parser.add_argument("filename", help="Input wav filename")

    parser.add_argument(
        "--peak",
        "-p",
        type=float,
        default=-1.0,
        help="Max dBFS peak gain after normalisation. (default: %(default)s)",
        )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output_normalised.wav",
        help="Output full path and filename. (default: %(default)s)",
        )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Don't print out progress. NOTE: might also hide " +
             "errors (default: %(default)s)",
        )

    args = parser.parse_args()

    fname_in    = pathlib.Path(args.filename).expanduser()
    fname_out   = pathlib.Path(args.output).expanduser().absolute()
    peak_gain   = args.peak
    quiet       = args.quiet

    if peak_gain > 0:
        print("Peak gain must be zero or negative, in dBFS", file=sys.stderr)
        return errno.EINVAL

    if quiet:
        warnings.simplefilter("ignore")

    if not quiet:
        print("input : %s" % fname_in)
        print("output: %s" % fname_out)

    x = zignal.WavFile(fname_in)

    x_peak_lin = x.peak()[0]
    x_peak_log = zignal.lin2db(np.absolute(x_peak_lin))
    if not quiet:
        print("Pre  [dB] [lin]: %s %s" % (
            np.array2string(
                x_peak_log, formatter={"float_kind": lambda x: "%6.1f" % x}),
            np.array2string(
                x_peak_lin, formatter={"float_kind": lambda x: "%7.4f" % x}),
            ))

    x.normalise()
    x.gain(peak_gain)

    x_peak_lin = x.peak()[0]
    x_peak_log = zignal.lin2db(np.absolute(x_peak_lin))
    if not quiet:
        print("Post [dB] [lin]: %s %s" % (
            np.array2string(
                x_peak_log, formatter={"float_kind": lambda x: "%6.1f" % x}),
            np.array2string(
                x_peak_lin, formatter={"float_kind": lambda x: "%7.4f" % x}),
            ))

    x.convert_to_float(32)
    x.write_wav_file(fname_out)


if __name__ == "__main__":
    # Below for debug and dev only, use the command line to pass these arguments
    sys.argv.append(("--help"))
    #sys.argv.append(("--version"))
    #sys.argv.append(("--quiet"))
    #sys.argv.append(("~/Music/somefile1.wav"))
    #sys.argv.extend(("--output", "output.wav",))
    #sys.argv.extend(("--peak", "-2.7",))
    #sys.argv.extend(("--peak", "0",))
    #sys.argv.extend(("--peak", "2",))

    main()
