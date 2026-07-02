"""RelaiScreen shrink->grow restores the exact prior viewport (incl. trailing
blank lines) and stays correct when content changes while shrunk."""

from relai.screen import RelaiScreen

COLS, ROWS, PANEL = 80, 24, 10
APP = ROWS - PANEL


def feed_lines(screen, texts):
    import pyte
    stream = pyte.ByteStream(screen)
    for t in texts:
        stream.feed((t + "\r\n").encode())


def disp(screen):
    return [row.rstrip() for row in screen.display]


def check(label, cond):
    print(f"{'OK ' if cond else 'FAIL'}: {label}")
    return cond


def scenario_full_screen():
    s = RelaiScreen(COLS, ROWS)
    feed_lines(s, [f"line_{i:02d}" for i in range(30)])  # scrolls
    before, cur_before = disp(s), (s.cursor.y, s.cursor.x)
    s.resize(APP, COLS)      # open panel
    s.resize(ROWS, COLS)     # close panel
    after, cur_after = disp(s), (s.cursor.y, s.cursor.x)
    return check("full screen: viewport identical after toggle", after == before) and \
        check("full screen: cursor identical", cur_before == cur_after)


def scenario_partial_with_blanks():
    s = RelaiScreen(COLS, ROWS)
    import pyte
    stream = pyte.ByteStream(s)
    stream.feed(b"A0\r\nA1\r\nA2\r\nA3\r\nA4")  # 5 rows, cursor row 4, blanks below
    before, cur_before = disp(s), (s.cursor.y, s.cursor.x)
    s.resize(APP, COLS)
    s.resize(ROWS, COLS)
    after, cur_after = disp(s), (s.cursor.y, s.cursor.x)
    return check("partial: trailing blanks preserved", after == before) and \
        check("partial: cursor identical", cur_before == cur_after)


def scenario_changes_while_open():
    # relai path: full screen -> shrink -> new output arrives -> grow.
    s = RelaiScreen(COLS, ROWS)
    feed_lines(s, [f"orig_{i:02d}" for i in range(24)])
    s.resize(APP, COLS)                        # open panel
    feed_lines(s, [f"new_{i:02d}" for i in range(5)])  # output while shrunk
    s.resize(ROWS, COLS)                       # close panel
    # reference: same total output on an always-full screen.
    ref = RelaiScreen(COLS, ROWS)
    feed_lines(ref, [f"orig_{i:02d}" for i in range(24)] +
               [f"new_{i:02d}" for i in range(5)])
    return check("changed-while-open: matches always-full-size screen",
                 disp(s) == disp(ref)) and \
        check("changed-while-open: cursor matches",
              (s.cursor.y, s.cursor.x) == (ref.cursor.y, ref.cursor.x))


def main():
    ok = True
    ok &= scenario_full_screen()
    ok &= scenario_partial_with_blanks()
    ok &= scenario_changes_while_open()
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
