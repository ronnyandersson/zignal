# Standard library
import argparse
import pathlib
import sys
import warnings

# Internal
import zignal


def main():
    parser = argparse.ArgumentParser(
        description="Combine .wav files to a multichannel .wav in 32 bit " +
                    "float. Each wav file can contain multiple channels.")

    parser.add_argument("--version", action="version",
                        version="%(prog)s " + "%s" % zignal.__version__)

    parser.add_argument(
        "filenames",
        nargs="+",
        help="Input wav filenames, can be multiple files",
        )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output_combined.wav",
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

    fname_out   = pathlib.Path(args.output).expanduser().absolute()
    filenames   = args.filenames
    quiet       = args.quiet

    if quiet:
        warnings.simplefilter("ignore")

    if not quiet:
        for filename in filenames:
            fname_in = pathlib.Path(filename).expanduser().absolute()
            print("input : %s" % fname_in)
        print("output: %s" % fname_out)

    wavfiles = []
    for filename in filenames:
        fname_in = pathlib.Path(filename).expanduser()
        wavfiles.append(zignal.WavFile(filename=fname_in))

    y = zignal.Audio(fs=wavfiles[0].fs)

    for wavfile in wavfiles:
        wavfile: zignal.Audio
        assert wavfiles[0].fs == wavfile.fs, "sample rates must match"
        y.append(wavfile)

    y.convert_to_float(32)
    if not quiet:
        print(y)
    y.write_wav_file(fname_out)


if __name__ == "__main__":
    # Below for debug and dev only, use the command line to pass these arguments
    sys.argv.append(("--help"))
    #sys.argv.append(("--version"))
    #sys.argv.append(("--quiet"))
    #sys.argv.extend((
    #    "~/Music/somefile1.wav",
    #    "~/Music/somefile2.wav",
    #    ))
    #sys.argv.extend(("--output", "output.wav",))

    main()
