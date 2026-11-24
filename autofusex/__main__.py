import sys
from autofusex.cli import main

if __name__ == "__main__":
    # Installed 0.2.4.x supervisors use this private launch command.
    # Keep it out of the afx console command and public help.
    if sys.argv[1:3] == ["daemon", "run"]:
        from autofusex.client.cli import main as internal_main
        raise SystemExit(internal_main(sys.argv[1:]))
    raise SystemExit(main())
