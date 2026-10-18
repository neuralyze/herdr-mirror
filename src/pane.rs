// herdr-mirror pane wrapper (data plane).
//
// Runs inside a local herdr pane and shows a remote herdr pane's terminal,
// live, over ssh. Read-only observe by default; escalates to a writable
// control session when the user types and releases back to observe.
//
//   herdr-mirror pane <ssh-target> <pane-target> [options]
//
// options:
//   --remote-bin PATH   remote herdr binary (default: PATH, then ~/.local/bin/herdr)
//   --cols N --rows N   observe request size (default 240x72; must be >= the
//                       remote PTY size or the server clips bottom rows away)
//   --dump              headless mode: print plain-text screen per frame
//   --session NAME      remote named session (passed as --session to herdr)
//   --control-idle N    auto-release control after N seconds idle (default 3600)
//   --always-control    start and stay in control: writable, no idle release,
//                       and sized to the local pane so it fills
//   --max-cols N        cap the size control asks the remote for (default:
//   --max-rows N        uncapped — control fills the local pane). Set for a
//                       remote with its own display: the remote keeps its own
//                       geometry and the rest of the local pane stays blank.
//
// Every stream gets its own direct ssh connection (no shared ControlMaster):
// isolated, and nothing persists to go stale on a flaky network.
//
// One owner of all state, message-driven: frames, keystrokes, timers, and
// ssh-child exits arrive on one channel; a session generation number tags
// every message so stale ones are dropped.

use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::ChildStdin;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::mpsc;
use tokio::time::Instant;

use crate::util::{err, Result};
use crate::grid::{Grid, Renderer};
use crate::foreground::Fg;
use crate::predict::Predictor;
use crate::select::{Released, Select};

// ---------------------------------------------------------------------------
// args

#[derive(Debug, Clone)]
pub struct Args {
    pub ssh_target: String,
    pub pane_target: String,
    /// Configured remote herdr path. `None` = auto-resolve on the remote
    /// (PATH, then `~/.local/bin/herdr`). See `config::remote_herdr_expr`.
    pub remote_bin: Option<String>,
    pub cols: usize,
    pub rows: usize,
    pub dump: bool,
    pub session: Option<String>,
    /// auto-release control after this much input idle; 0 disables
    pub control_idle_secs: u64,
    /// start and stay in control: writable, no idle release, and sized to the
    /// local pane so it fills. Set by the daemon from per-host config.
    pub always_control: bool,
    /// upper bound on the size control asks the remote for. `None` = uncapped
    /// (fill the local pane). Set by the daemon from per-host config; observe
    /// is never capped, since it doesn't resize anything.
    pub max_cols: Option<usize>,
    pub max_rows: Option<usize>,
    /// daemon's ssh ControlMaster socket for this host; foreground polls reuse it
    /// (`ssh -S <path>`) to skip a handshake. None → polls connect directly.
    ///
    pub ctl_path: Option<String>,
    /// container to exec into instead of ssh. `None` = ssh host.
    pub container: Option<ContainerArg>,
}

/// How the pane process should reach its container. The daemon passes a *ref*,
/// not a resolved id: the pane may outlive a rebuild, and ids change while the
/// folder label does not.
#[derive(Debug, Clone)]
pub struct ContainerArg {
    pub kind: crate::config::HostKind,
    pub docker_bin: String,
}

pub fn parse_args(argv: &[String]) -> Result<Args> {
    let mut args = Args {
        ssh_target: String::new(),
        pane_target: String::new(),
        remote_bin: None,
        cols: 240,
        rows: 72,
        dump: false,
        session: None,
        control_idle_secs: 3600,
        always_control: false,
        max_cols: None,
        max_rows: None,
        ctl_path: None,
        container: None,
    };
    let mut container_name: Option<String> = None;
    let mut container_folder: Option<String> = None;
    let mut docker_bin = "docker".to_string();
    let mut positional: Vec<String> = Vec::new();
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        let mut next = |flag: &str| -> Result<String> {
            it.next().cloned().ok_or_else(|| err(format!("{flag} needs a value")))
        };
        match a.as_str() {
            "--remote-bin" => args.remote_bin = Some(next("--remote-bin")?),
            "--cols" => {
                args.cols = next("--cols")?.parse().map_err(|_| err("--cols must be a number"))?;
            }
            "--rows" => {
                args.rows = next("--rows")?.parse().map_err(|_| err("--rows must be a number"))?;
            }
            "--session" => args.session = Some(next("--session")?),
            "--control-idle" => {
                args.control_idle_secs =
                    next("--control-idle")?.parse().map_err(|_| err("--control-idle must be a number"))?
            }
            "--always-control" => args.always_control = true,
            // 0 is unset here for the same reason config treats it that way:
            // a zero cap would ask the remote for a zero-column terminal, which
            // herdr rejects outright, killing the session twice over and
            // stranding the pane in "control unavailable" over a typo.
            "--max-cols" => {
                args.max_cols = Some(next("--max-cols")?.parse().map_err(|_| err("--max-cols must be a number"))?)
                    .filter(|&n| n > 0)
            }
            "--max-rows" => {
                args.max_rows = Some(next("--max-rows")?.parse().map_err(|_| err("--max-rows must be a number"))?)
                    .filter(|&n| n > 0)
            }
            "--ctl-path" => args.ctl_path = Some(next("--ctl-path")?),
            "--container" => container_name = Some(next("--container")?),
            "--container-folder" => container_folder = Some(next("--container-folder")?),
            "--docker-bin" => docker_bin = next("--docker-bin")?,
            "--dump" => args.dump = true,
            other if other.starts_with('-') => return Err(err(format!("unknown option: {other}"))),
            other => positional.push(other.to_string()),
        }
    }
    if positional.len() != 2 {
        return Err(err(
            "usage: herdr-mirror pane <ssh-target> <pane-target> [--remote-bin PATH] [--session NAME] [--cols N --rows N] [--max-cols N --max-rows N] [--dump]",
        ));
    }
    args.container = match (container_name, container_folder) {
        (Some(_), Some(_)) => return Err(err("--container and --container-folder are exclusive")),
        (Some(n), None) => {
            Some(ContainerArg { kind: crate::config::HostKind::DockerContainer(n), docker_bin })
        }
        (None, Some(f)) => {
            Some(ContainerArg { kind: crate::config::HostKind::DockerFolder(f), docker_bin })
        }
        (None, None) => None,
    };
    args.ssh_target = positional.remove(0);
    args.pane_target = positional.remove(0);
    Ok(args)
}

// ---------------------------------------------------------------------------
// remote session: one managed ssh/docker child across every mode transition

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Mode {
    Observe,
    Control,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Mode::Observe => "observe",
            Mode::Control => "control",
        }
    }
}

#[derive(Debug, Deserialize)]
struct Frame {
    seq: Option<u64>,
    full: Option<bool>,
    width: Option<usize>,
    height: Option<usize>,
    bytes: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TerminalState {
    request_id: u64,
    target: String,
    mode: Mode,
    cols: usize,
    rows: usize,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum StreamMessage {
    #[serde(rename = "terminal.frame")]
    Frame(Frame),
    #[serde(rename = "terminal.state")]
    State(TerminalState),
    #[serde(rename = "terminal.closed")]
    Closed { reason: Option<String> },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Copy)]
struct ModeRequest {
    request_id: u64,
    mode: Mode,
}

enum Msg {
    Remote { gen: u64, message: StreamMessage },
    SessionExit { gen: u64, reason: String },
    ProtocolError { gen: u64, reason: String },
    Stdin(Vec<u8>),
    StdinClosed,
    /// Failed polls retain the last classification; old generations cannot
    /// change a replacement stream's input routing.
    Foreground { gen: u64, value: Option<Fg> },
    Paste { epoch: u64, outcome: crate::paste::Outcome },
    Drop { epoch: u64, result: crate::paste::DropResult },
}

struct Session {
    gen: u64,
    process_group: i32,
    stdin: ChildStdin,
}

/// Stop the transport and every local helper it spawned (notably an ssh
/// ProxyCommand such as `aws ssm start-session`). Each transport is started as
/// its own process-group leader, so this cannot signal the pane wrapper itself.
fn kill_session_process_group(process_group: i32) {
    if process_group > 0 {
        unsafe { libc::kill(-process_group, libc::SIGKILL) };
    }
}

fn isolate_session_process_group(command: &mut tokio::process::Command) {
    command.process_group(0);
}

/// POSIX single-quote: an embedded ' can't break the remote shell parse.
pub(crate) fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

fn ssh_stream_args(ssh_target: &str, cmd: &str) -> Vec<String> {
    let mut argv: Vec<String> = crate::remote::SSH_COMMON_OPTS
        .iter()
        .map(|arg| (*arg).to_string())
        .collect();
    // A pane stream is long-lived and interactive. It must not inherit a
    // ControlPath from ~/.ssh/config: a stale shared master can hang
    // the stream before the remote command starts. `-S none` disables control
    // socket use without changing the daemon's intentional multiplexing.
    argv.extend(["-S".into(), "none".into(), ssh_target.into(), cmd.into()]);
    argv
}

fn managed_command(args: &Args, cols: usize, rows: usize) -> String {
    // Configured paths stay unquoted so remote-shell ~ expands; auto mode is an
    // `sh -c` resolver that takes the trailing words as "$@" (see
    // config::remote_herdr_expr).
    let bin = crate::config::remote_herdr_expr(
        args.remote_bin.as_deref(),
        args.session.as_deref(),
    );
    format!(
        "exec {} terminal session managed {} --cols {} --rows {}",
        bin,
        sh_quote(&args.pane_target),
        cols,
        rows
    )
}

/// Consume stdout in wire order, including state barriers between frames.
/// Keeping this shared with the real child reader makes fragmented-state and
/// frame-order tests exercise the consumer rather than a second parser.
async fn consume_stream(
    stdout: impl tokio::io::AsyncRead + Unpin,
    gen: u64,
    tx: &mpsc::Sender<Msg>,
) -> String {
    let mut reason = String::new();
    let mut lines = BufReader::new(stdout).lines();
    loop {
        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) => break,
            Err(e) => {
                reason = format!("managed terminal stream read failed: {e}");
                break;
            }
        };
        let message = match serde_json::from_str::<StreamMessage>(&line) {
            Ok(message) => message,
            Err(e) => {
                reason = format!("invalid managed terminal stream: {e}");
                let _ = tx.send(Msg::ProtocolError { gen, reason: reason.clone() }).await;
                break;
            }
        };
        if let StreamMessage::Closed { reason: Some(r) } = &message {
            reason = r.clone();
        }
        if tx.send(Msg::Remote { gen, message }).await.is_err() {
            break;
        }
    }
    reason
}

fn spawn_session(args: &Args, cols: usize, rows: usize, gen: u64, tx: mpsc::Sender<Msg>) -> Result<Session> {
    let cmd = managed_command(args, cols, rows);
    // ssh and docker differ only in how the command is carried; the streaming
    // contract (piped stdio, herdr's frames on stdout) is identical
    let mut builder = match &args.container {
        None => {
            let mut c = tokio::process::Command::new("ssh");
            c.args(ssh_stream_args(&args.ssh_target, &cmd));
            c
        }
        Some(ct) => {
            // resolve per spawn so a rebuilt container is picked up on
            // reconnect. Bounded: this runs on the pane's single-threaded
            // runtime, so a wedged Docker daemon must not be able to freeze
            // input, rendering or signal handling.
            let id = crate::docker::resolve_blocking(
                &ct.docker_bin,
                &ct.kind,
                Duration::from_secs(5),
            )?;
            let mut c = tokio::process::Command::new(&ct.docker_bin);
            // `sh -c` not `-lc`: match ssh's non-login remote shell
            c.args(["exec", "-i", &id, "sh", "-c", &cmd]);
            c
        }
    };
    // Keep the transport and all of its local descendants in one independently
    // killable group. Killing only ssh can orphan a ProxyCommand and leave its
    // remote terminal controller attached after the mirror pane closes.
    isolate_session_process_group(&mut builder);
    let mut child = builder
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let process_group = child.id().map(|p| p as i32).unwrap_or(0);
    let stdin = child.stdin.take().ok_or_else(|| err("no child stdin"))?;
    let stdout = child.stdout.take().ok_or_else(|| err("no child stdout"))?;
    let stderr = child.stderr.take().ok_or_else(|| err("no child stderr"))?;

    tokio::spawn(async move {
        // ssh errors arrive on stderr; the server's failure reason arrives as
        // a terminal.closed frame on STDOUT — capture both
        let err_tail: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let err_tail2 = err_tail.clone();
        let mut stderr_task = tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                let mut buf = err_tail2.lock().unwrap();
                buf.push_str(&l);
                buf.push('\n');
                if buf.len() > 400 {
                    let tail: String = buf.chars().rev().take(400).collect::<Vec<_>>().into_iter().rev().collect();
                    *buf = tail;
                }
            }
        });
        let close_reason = consume_stream(stdout, gen, &tx).await;
        let _ = child.wait().await;
        // A fast unsupported CLI can exit before its stderr reader is polled.
        // Drain its error before classifying the exit, without waiting forever
        // for a descendant that inherited the pipe.
        let _ = tokio::time::timeout(Duration::from_millis(100), &mut stderr_task).await;
        stderr_task.abort();
        let tail = err_tail.lock().unwrap().trim().to_string();
        let reason = if close_reason.is_empty() { tail } else { close_reason };
        let _ = tx.send(Msg::SessionExit { gen, reason }).await;
    });

    Ok(Session { gen, process_group, stdin })
}

// ---------------------------------------------------------------------------
// terminal plumbing

/// The layout herdr renders when a server has NO client attached: MIN_COLS x
/// MIN_ROWS from its headless server. Not a minimum — an attached client
/// smaller than this gets its real size — it is specifically the placeholder
/// used when nobody is watching.
///
/// Every derivation from it subtracts: sidebar, tab bar, splits, gaps, the
/// scrollbar gutter. So a pane born under that layout cannot EXCEED it on
/// either axis, which is what makes this a sound upper bound rather than a
/// shape to match. Shape matching is the trap: a phone lays out to the same
/// rectangle as the placeholder.
const HERDR_NO_CLIENT_LAYOUT: (usize, usize) = (80, 24);

/// Could this pane's size have come from a layout nobody is watching?
///
/// Strictly larger on either axis is provably a real viewport. `>` not `>=`:
/// herdr spawns restored panes at exactly 24x80, so `>=` would trust a
/// placeholder.
///
/// Consulted at birth to pick the initial mode. Deliberately NOT consulted on
/// the promotion path: a resize is taken as evidence of a client, which holds
/// in practice but is empirical rather than structural — herdr has one
/// clientless resize path (its first virtual render), so a pane created in the
/// instant before that render could in principle promote on a placeholder
/// resize. It self-heals the moment a client attaches.
fn size_is_trusted((cols, rows): (usize, usize)) -> bool {
    cols > HERDR_NO_CLIENT_LAYOUT.0 || rows > HERDR_NO_CLIENT_LAYOUT.1
}

/// Clamp a local terminal size to the per-host control caps. Split out from
/// `control_size` so the arithmetic is testable without an `App`.
fn cap_size(
    (cols, rows): (usize, usize),
    max_cols: Option<usize>,
    max_rows: Option<usize>,
) -> (usize, usize) {
    (
        max_cols.map_or(cols, |cap| cols.min(cap)),
        max_rows.map_or(rows, |cap| rows.min(cap)),
    )
}

/// Size to request for an observe stream. Split out from `App::observe_size` so
/// the floor is testable.
///
/// `--cols/--rows` are a floor, never an exact request. As a floor they still do
/// their original job: the request must be >= the remote PTY size or the server
/// clips its bottom rows away, and the daemon's numbers already carry a margin.
/// As an exact request they are wrong — the daemon samples the *remote* pane's
/// rect when it spawns the streamer, and a headless remote reports the no-client
/// placeholder, so the numbers are small. Control then resizes the remote pty to
/// this pane and nothing shrinks it back on release, so asking for the daemon's
/// numbers again would stream a crop of a screen that has since grown, painted
/// into the corner of a much larger pane.
fn observe_size_for(args: &Args, term: (usize, usize)) -> (usize, usize) {
    (args.cols.max(term.0), args.rows.max(term.1))
}

/// Desired mode after the initial Observe acknowledgement, never the CLI mode.
fn initial_mode(always_control: bool, size: (usize, usize)) -> Mode {
    if always_control && size_is_trusted(size) {
        Mode::Control
    } else {
        Mode::Observe
    }
}

fn term_size() -> (usize, usize) {
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0 {
            return (ws.ws_col as usize, ws.ws_row as usize);
        }
    }
    (80, 24)
}

struct RawMode {
    orig: libc::termios,
}

impl RawMode {
    fn enable() -> Option<RawMode> {
        unsafe {
            let mut orig: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut orig) != 0 {
                return None;
            }
            let mut raw = orig;
            libc::cfmakeraw(&mut raw);
            if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &raw) != 0 {
                return None;
            }
            Some(RawMode { orig })
        }
    }

    fn restore(&self) {
        unsafe {
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.orig);
        }
    }
}

fn write_stdout(s: &str) {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(s.as_bytes());
    let _ = out.flush();
}

/// Longest remote title relayed, in chars. Titles are one-line labels (an
/// agent's task summary); the bound only stops a hostile or broken remote from
/// pushing an arbitrarily large string through every herdr client's pane list.
const TITLE_MAX_CHARS: usize = 256;

/// Make a remote title safe to re-emit inside OSC 2. It travels from another
/// machine into the local terminal stream, where a control char ends the string
/// early — BEL, ESC (the start of ST), a C1 ST — and runs whatever follows as
/// terminal input. `is_control` covers all of C0, DEL and C1.
fn sanitize_title(raw: &str) -> String {
    raw.chars().filter(|c| !c.is_control()).take(TITLE_MAX_CHARS).collect()
}

/// The OSC 2 that makes this pane's title the relayed one, or `None` when
/// there is nothing to write: no relay for us, a relay addressed to a different
/// remote pane (a previous holder of this reused local id), or no change.
///
/// A remote pane with no title is written as an empty OSC 2, which herdr reads
/// as "no title" — rather than leaving the one the shell set before `exec`.
fn title_update(
    relayed: Option<crate::state::PaneTitle>,
    pane_target: &str,
    shown: &mut Option<String>,
) -> Option<String> {
    let relayed = relayed.filter(|t| t.remote == pane_target)?;
    let clean = sanitize_title(relayed.title.as_deref().unwrap_or(""));
    if shown.as_deref() == Some(clean.as_str()) {
        return None;
    }
    let seq = format!("\x1b]2;{clean}\x07");
    *shown = Some(clean);
    Some(seq)
}

/// Show the remote pane's title as this pane's own, so herdr's `terminal_title`
/// for the mirror is the remote's. Only a streamer drawing into a herdr pane
/// (tty, `HERDR_PANE_ID` known) has a title to set.
fn relay_title(
    state_dir: &std::path::Path,
    local_pane_id: Option<&str>,
    pane_target: &str,
    shown: &mut Option<String>,
) {
    let Some(id) = local_pane_id else { return };
    if let Some(seq) = title_update(crate::state::read_pane_title(state_dir, id), pane_target, shown) {
        write_stdout(&seq);
    }
}

/// One SGR mouse event: ESC [ < btn ; col ; row (M|m). Returns (btn, col, row,
/// press, total len) for a sequence starting at `bytes[at]`.
fn parse_mouse(bytes: &[u8], at: usize) -> Option<(u32, u32, u32, bool, usize)> {
    let rest = &bytes[at..];
    if rest.len() < 6 || rest[0] != 0x1b || rest[1] != b'[' || rest[2] != b'<' {
        return None;
    }
    let mut nums = [0u32; 3];
    let mut n = 0usize;
    let mut i = 3usize;
    let mut have_digit = false;
    while i < rest.len() && n < 3 {
        match rest[i] {
            b'0'..=b'9' => {
                // saturate: garbage digit runs on stdin must not overflow-panic
                nums[n] = nums[n].saturating_mul(10).saturating_add((rest[i] - b'0') as u32);
                have_digit = true;
                i += 1;
            }
            b';' if n < 2 && have_digit => {
                n += 1;
                have_digit = false;
                i += 1;
            }
            b'M' | b'm' if n == 2 && have_digit => {
                return Some((nums[0], nums[1], nums[2], rest[i] == b'M', i + 1));
            }
            _ => return None,
        }
    }
    None
}

const MOUSE_INPUT_TIMEOUT: Duration = Duration::from_millis(150);
const MAX_INCOMPLETE_MOUSE_BYTES: usize = 32;

#[derive(Debug, PartialEq)]
pub(crate) enum MouseSplit {
    Pending,
    Passthrough(Vec<u8>),
}

/// Find a possible SGR mouse sequence that ends at the current read boundary.
/// Only a short, syntactically valid prefix is held; ordinary input keeps
/// flowing to the existing routing code.
fn trailing_incomplete_mouse(bytes: &[u8]) -> Option<usize> {
    for (at, byte) in bytes.iter().enumerate() {
        if *byte != 0x1b {
            continue;
        }
        let rest = &bytes[at..];
        if rest == b"\x1b" || rest == b"\x1b[" {
            return Some(at);
        }
        if rest.starts_with(b"\x1b[<")
            && rest.len() <= MAX_INCOMPLETE_MOUSE_BYTES
            && rest[3..].iter().all(|b| b.is_ascii_digit() || *b == b';')
        {
            return Some(at);
        }
    }
    None
}

/// Hold only a trailing incomplete mouse sequence across stdin reads.
pub(crate) fn split_mouse(buf: &mut Vec<u8>, chunk: Vec<u8>) -> MouseSplit {
    let mut all = std::mem::take(buf);
    all.extend_from_slice(&chunk);
    let Some(at) = trailing_incomplete_mouse(&all) else {
        return MouseSplit::Passthrough(all);
    };
    let tail = all.split_off(at);
    *buf = tail;
    if all.is_empty() {
        MouseSplit::Pending
    } else {
        MouseSplit::Passthrough(all)
    }
}

/// A timed-out lone ESC is still a key; an incomplete mouse prefix is dropped
/// so it cannot leak as literal input into a remote shell.
fn flush_mouse(buf: &mut Vec<u8>) -> Option<Vec<u8>> {
    let pending = std::mem::take(buf);
    (pending == b"\x1b").then_some(pending)
}

/// How a parsed mouse event should be routed while in control mode.
#[derive(Debug, PartialEq, Eq)]
enum MouseAction {
    /// wheel: send as a semantic terminal.scroll (server decides app vs scrollback)
    Scroll { up: bool },
    /// click/drag on a remote TUI: forward the raw SGR sequence
    ForwardRaw,
    /// left press/drag/release: hand it to the plugin selector, which defers
    /// the press and (on a TUI) replays it if the gesture was a click rather
    /// than a drag (see src/select.rs)
    Select,
    /// unclassified, or a non-left button at a shell: drop so SGR never
    /// reaches a prompt that never enabled mouse reporting
    Drop,
}

/// Which physical button an SGR code names, with the modifier and motion flags
/// removed: 0/1/2 = left/middle/right, 4/5 = wheel up/down, 6/7 = wheel
/// left/right, 8.. = extra buttons.
///
/// The button number is *not* contiguous in the wire encoding: it is the low two
/// bits plus bit 6 (64) for the wheel set and bit 7 (128) for buttons 8-11,
/// while bits 2-4 carry shift/alt/ctrl and bit 5 carries motion. Masking only
/// the low two bits reads shift+wheel-up (68) as a left press, which is how the
/// wheel ends up driving the selection.
fn button_number(btn: u32) -> u32 {
    (btn & 0b11) + if btn & 64 != 0 { 4 } else { 0 } + if btn & 128 != 0 { 8 } else { 0 }
}

/// Wheel always scrolls semantically, regardless of the foreground
/// classification — the remote herdr server knows the real app's mouse mode
/// and is a better judge than this side's process-name heuristic (e.g. a TUI
/// that doesn't consume wheel events, like an agent CLI). Non-wheel
/// clicks/drags keep the existing foreground-based routing.
/// The left button goes to the plugin selector in every classified foreground
/// (TUI, agent, or shell), because a drag is the gesture with no substitute:
/// an app's click can be replayed after the fact, a selection cannot be
/// recovered. The grab is always held, so these events actually arrive — even
/// at a shell, where releasing it used to hand selection to herdr and starve
/// the wheel (#75).
///
/// A TUI gesture that never leaves its cell is replayed to the app as a real
/// click, so htop still sorts on a header click and lazygit still stages on a
/// file click. Agent CLIs get it too: they never enabled mouse reporting, but
/// claude and codex both discard the bytes cleanly, so withholding it only
/// stood to swallow clicks the day one of them grows mouse support. A shell
/// click is not replayed: the prompt never enabled mouse reporting, and the
/// bytes would dump into it.
///
/// The cost is an in-app *drag*: vim's mouse visual-select, a resize handle.
fn mouse_action(fg: Option<Fg>, btn: u32, press: bool) -> MouseAction {
    match button_number(btn) {
        // wheel up/down. Matched on the button number, not on `btn == 64`, so a
        // modified scroll still scrolls instead of falling through as a click.
        b @ (4 | 5) if press => MouseAction::Scroll { up: b == 4 },
        6 | 7 => MouseAction::Drop,
        0 if matches!(fg, Some(Fg::Agent) | Some(Fg::Mouse) | Some(Fg::Shell)) => {
            MouseAction::Select
        }
        // middle, right, wheel release and the extra buttons — TUI/agent only.
        // A shell never asked for these; forwarding them garbage the prompt.
        _ if matches!(fg, Some(Fg::Agent) | Some(Fg::Mouse)) => MouseAction::ForwardRaw,
        _ => MouseAction::Drop,
    }
}

fn contains_wheel_press(bytes: &[u8]) -> bool {
    let mut i = 0;
    while i < bytes.len() {
        if let Some((btn, _, _, press, len)) = parse_mouse(bytes, i) {
            if press && (btn == 64 || btn == 65) {
                return true;
            }
            i += len;
        } else {
            i += 1;
        }
    }
    false
}

/// Cap on a partially-received bracketed paste. Past this we stop waiting for
/// a terminator and flush what we have: a huge paste is still forwarded, it
/// just loses the drop treatment rather than buffering without bound.
const MAX_PASTE_BYTES: usize = 1024 * 1024;

const PASTE_START: &[u8] = b"\x1b[200~";
const PASTE_END: &[u8] = b"\x1b[201~";

/// Input held back while an upload is in flight (see `route_input`).
enum Queued {
    Raw(Vec<u8>),
    Body(Vec<u8>),
}

#[derive(Debug, PartialEq)]
pub(crate) enum PasteSplit {
    /// a paste has begun but not finished; nothing to do yet
    Pending,
    /// no paste involved: forward as-is
    Passthrough(Vec<u8>),
    Complete { before: Vec<u8>, body: Vec<u8>, after: Vec<u8> },
}

fn find_seq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Longest suffix of `hay` that is a *proper* prefix of `needle`, so a marker
/// straddling two reads is held rather than forwarded in pieces.
fn trailing_partial(hay: &[u8], needle: &[u8]) -> usize {
    (1..needle.len().min(hay.len()) + 1)
        .rev()
        .find(|&n| n < needle.len() && hay[hay.len() - n..] == needle[..n])
        .unwrap_or(0)
}

/// Pull the next complete bracketed paste out of `buf` + `chunk`.
///
/// Pure so the framing can be table-tested without a pane: every interesting
/// case (split reads, a START with no END, bytes either side, several pastes
/// in one read, the oversize flush) lives here rather than in the event loop.
pub(crate) fn split_paste(buf: &mut Vec<u8>, chunk: Vec<u8>) -> PasteSplit {
    // fast path for ordinary typing: nothing buffered and no marker starting
    if buf.is_empty()
        && find_seq(&chunk, PASTE_START).is_none()
        && trailing_partial(&chunk, PASTE_START) == 0
    {
        return PasteSplit::Passthrough(chunk);
    }
    buf.extend_from_slice(&chunk);

    // a paste that never terminates must not buffer without bound
    if buf.len() > MAX_PASTE_BYTES {
        return PasteSplit::Passthrough(std::mem::take(buf));
    }

    let Some(start) = find_seq(buf, PASTE_START) else {
        // hold only a possible partial marker; release everything before it
        let keep = trailing_partial(buf, PASTE_START);
        let cut = buf.len() - keep;
        if cut == 0 {
            return PasteSplit::Pending;
        }
        let tail = buf.split_off(cut);
        let head = std::mem::replace(buf, tail);
        return PasteSplit::Passthrough(head);
    };
    let Some(end) = find_seq(&buf[start..], PASTE_END).map(|i| start + i) else {
        return PasteSplit::Pending;
    };

    let all = std::mem::take(buf);
    PasteSplit::Complete {
        before: all[..start].to_vec(),
        body: all[start + PASTE_START.len()..end].to_vec(),
        after: all[end + PASTE_END.len()..].to_vec(),
    }
}

fn has_mouse_seq(bytes: &[u8]) -> bool {
    bytes.windows(3).any(|w| w == [0x1b, b'[', b'<'])
}
/// A mixed keyboard/mouse read still contains real input. A mouse report must
/// not make us discard the keystrokes beside it while observing.
fn has_keyboard_input(bytes: &[u8]) -> bool {
    let mut at = 0;
    while at < bytes.len() {
        match parse_mouse(bytes, at) {
            Some((_, _, _, _, len)) => at += len,
            None => return true,
        }
    }
    false
}

enum PendingInput {
    Raw(Vec<u8>),
    /// Paste/upload bytes are data: never re-interpret their shortcuts or mouse.
    Literal(Vec<u8>),
}


// ---------------------------------------------------------------------------
// the wrapper state machine

const BACKOFF: [u64; 4] = [1000, 2000, 5000, 10000];

/// Rung used once the remote pane is known to be gone. Deliberately still a
/// retry and not a stop: the daemon owns mirror lifecycle and reaps a pane
/// whose remote has been absent for two converge polls, so the streamer's job
/// here is to wait quietly for that rather than to decide it is finished. It
/// also keeps "gone" recoverable, which it sometimes is — a remote herdr
/// restarting renumbers pane ids while session restore runs.
const GONE_BACKOFF_MS: u64 = 60_000;
const QUICK_CONTROL_FAILURE: Duration = Duration::from_secs(4);


/// Is this failure "the pane we stream is gone", as opposed to any other
/// failure that happens to mention something missing?
///
/// Matches herdr's whole sentence (`headless.rs`: "terminal target {t} not
/// found") with OUR target in it, rather than loose substrings. That keeps it
/// off the remote-bin resolver's `exec: …/herdr: not found`, which means herdr
/// is absent on that host — a different problem with a different fix — and
/// stops a target being confused with one that merely shares its prefix
/// (`w1:p1` vs `w1:p10`, both real ids in herdr's base-32 alphabet).
///
/// Note the pane dying *underneath* a live stream reports differently
/// ("terminal attach ended: terminal {term_id} not found", carrying herdr's
/// internal terminal id, not ours). That deliberately does not match: the next
/// attempt asks for the pane target and gets the canonical sentence a second
/// later, so this fires one cycle behind rather than being loosened.
fn target_gone(reason: &str, pane_target: &str) -> bool {
    if pane_target.is_empty() {
        return false;
    }
    reason
        .to_ascii_lowercase()
        .contains(&format!("terminal target {} not found", pane_target.to_ascii_lowercase()))
}

/// Delay before the next attempt, and the ladder position to keep.
///
/// Pure so the rung and the ladder-resume are testable without a live pane.
/// A gone target does NOT consume a rung: if the pane comes back and later
/// fails transiently, that failure should start where the fast ladder left
/// off rather than at the top.
fn reconnect_delay(gone: bool, idx: usize) -> (u64, usize) {
    if gone {
        return (GONE_BACKOFF_MS, idx);
    }
    (BACKOFF[idx.min(BACKOFF.len() - 1)], idx + 1)
}

fn managed_unsupported(reason: &str) -> bool {
    let text = reason.to_ascii_lowercase();
    (text.contains("managed")
        && (text.contains("unrecognized") || text.contains("unknown")
            || text.contains("unexpected argument") || text.contains("unsupported")
            || text.contains("requires") || text.contains("require ")
            || text.contains("upgrade")))
        || (text.contains("terminal.session.")
            && (text.contains("unsupported") || text.contains("unknown")
                || text.contains("not supported")))
        || text.contains("without an initial terminal.state")
}

struct App {
    args: Args,
    tty: bool,
    grid: Grid,
    renderer: Renderer,
    tx: mpsc::Sender<Msg>,

    /// Last authoritative state, never the requested/speculated mode.
    mode: Mode,
    desired_mode: Mode,
    state_ready: bool,
    pending_mode: Option<ModeRequest>,
    /// Increasing across reconnects as well as transitions; generation is an
    /// additional guard, not a substitute for request correlation.
    next_request: u64,
    state_error: Option<String>,
    unsupported: bool,
    resize_pending: bool,
    session: Option<Session>,
    next_gen: u64,

    backoff_idx: usize,
    reconnect_at: Option<Instant>,
    control_failures: u32,
    control_since: Option<Instant>,
    control_sticky: bool,
    pending_input: Vec<PendingInput>,
    /// Invalidates clipboard/upload callbacks on release, denial or disconnect.
    input_epoch: u64,
    last_input: Instant,
    hint_clear_at: Option<Instant>,
    /// predictive local echo — draws keystrokes optimistically, frame-verified
    predict: Predictor,
    /// remote pane foreground classification, None=unknown (fail safe to local).
    /// Refreshed lazily on mouse activity; see `foreground::classify`.
    remote_fg: Option<Fg>,
    /// local drag-selection, driven by the left button the remote app would
    /// otherwise receive
    select: Select,
    /// screen rows the selection overlay covered on the last paint, so they can
    /// be repainted when it moves away
    last_select_rows: Option<(usize, usize)>,
    /// last time a foreground poll was kicked off (throttles the ssh handshakes)
    fg_poll_at: Option<Instant>,
    /// scheduled delayed re-poll to catch a foreground change the last input just
    /// caused (e.g. quitting a TUI back to a shell); bypasses the throttle
    settle_at: Option<Instant>,
    /// whether the local mouse grab (?1002h) is currently on. Always held so
    /// wheel events reach us as terminal.scroll even at a shell; the pane is
    /// on the alt screen and has no local scrollback, so a released grab
    /// cannot scroll (#75).
    mouse_grabbed: bool,
    /// whether the local pane is currently in application cursor mode (?1h), held
    /// to match the remote's so forwarded arrows arrive in the form it expects
    app_cursor_keys: bool,
    paste_inflight: bool,
    /// partially-received bracketed paste (see `intercept_paste`)
    paste_buf: Vec<u8>,
    /// partially-received SGR mouse sequence
    mouse_buf: Vec<u8>,
    mouse_flush_at: Option<Instant>,
    /// input held back while an upload is in flight, flushed in order after
    paste_queue: Vec<Queued>,
    /// the payload that started the in-flight upload, so it can be forwarded
    /// unchanged when every path turns out to exist on the remote already
    paste_original: Option<Vec<u8>>,
}

/// minimum spacing between foreground polls — each is an ssh handshake, so we
/// poll lazily (only around mouse activity) and no faster than this
const FG_POLL_INTERVAL: Duration = Duration::from_millis(1500);

/// after input settles, re-poll once this much later to catch a foreground
/// change the input caused (e.g. a TUI just exited); bypasses FG_POLL_INTERVAL
const SETTLE_DELAY: Duration = Duration::from_millis(350);

impl App {
    fn new(args: Args, tty: bool, tx: mpsc::Sender<Msg>) -> Self {
        Self {
            args,
            tty,
            grid: Grid::new(),
            renderer: Renderer::new(),
            tx,
            mode: Mode::Observe,
            desired_mode: Mode::Observe,
            state_ready: false,
            pending_mode: None,
            next_request: 0,
            state_error: None,
            unsupported: false,
            resize_pending: false,
            session: None,
            next_gen: 0,
            backoff_idx: 0,
            reconnect_at: None,
            control_failures: 0,
            control_since: None,
            control_sticky: false,
            pending_input: Vec::new(),
            input_epoch: 0,
            last_input: Instant::now(),
            hint_clear_at: None,
            predict: Predictor::new(),
            remote_fg: None,
            select: Select::new(),
            last_select_rows: None,
            fg_poll_at: None,
            settle_at: None,
            mouse_grabbed: tty,
            app_cursor_keys: false,
            paste_inflight: false,
            paste_buf: Vec::new(),
            mouse_buf: Vec::new(),
            mouse_flush_at: None,
            paste_queue: Vec::new(),
            paste_original: None,
        }
    }

    fn can_write(&self) -> bool {
        self.session.is_some() && self.state_ready && self.mode == Mode::Control
            && self.pending_mode.is_none() && self.desired_mode == Mode::Control
    }

    fn mode_label(&self) -> &'static str {
        if self.state_ready && self.mode == Mode::Control { "CONTROL" } else { "viewing only" }
    }

    fn mode_status(&self) -> String {
        if self.unsupported {
            return "managed sessions unavailable — upgrade the remote Herdr CLI and server to matching managed-session support; type to retry".into();
        }
        if self.session.is_none() {
            return self.state_error.clone().unwrap_or_else(|| "disconnected — reconnecting…".into());
        }
        if !self.state_ready {
            return "connecting — awaiting managed Observe acknowledgement".into();
        }
        let label = self.mode_label();
        if let Some(request) = self.pending_mode {
            return format!("{label} — {}", if request.mode == Mode::Control {
                "taking control…"
            } else {
                "releasing…"
            });
        }
        if let Some(error) = &self.state_error {
            return format!("{label} — {error}; {}", if self.mode == Mode::Control {
                "ctrl+\\ to retry release"
            } else {
                "type to retry"
            });
        }
        if self.paste_inflight {
            return format!("{label} — uploading…");
        }
        match self.mode {
            Mode::Observe => "viewing only — type to take control".into(),
            Mode::Control if self.args.always_control => "CONTROL".into(),
            Mode::Control => "CONTROL — ctrl+\\ to release".into(),
        }
    }

    fn restore_mode_status(&mut self) {
        self.renderer.status(&self.mode_status());
        self.hint_clear_at = None;
    }

    fn cancel_paste(&mut self) {
        self.input_epoch += 1;
        self.paste_inflight = false;
        self.paste_original = None;
        self.paste_queue.clear();
        self.paste_buf.clear();
        self.mouse_buf.clear();
        self.mouse_flush_at = None;
    }

    fn paint(&mut self) {
        if !self.tty {
            return;
        }
        let (cols, rows) = term_size();
        // Overlays paint outside the renderer's per-row cache, so the rows they
        // covered must be repainted or the old drawing survives underneath.
        // Predictions are scattered, so they take the whole pane; a selection is
        // a contiguous band, and during a drag it moves on every mouse motion,
        // so it repaints only the rows it just left and the ones it now covers.
        let reserved = self.renderer.status_rows();
        if self.predict.take_dirty() {
            self.renderer.invalidate();
        }
        if self.select.take_dirty() {
            let now = self.select.painted_rows(&self.grid, rows, reserved);
            for (top, bottom) in self.last_select_rows.into_iter().chain(now) {
                for r in top..=bottom {
                    self.renderer.invalidate_row(r);
                }
            }
        }
        self.last_select_rows = self.select.painted_rows(&self.grid, rows, reserved);
        let mut out = self.renderer.paint(&self.grid, cols, rows);
        // inject the overlays inside the synchronized-update block. Selection
        // goes last: it is the one the user is actively pointing at.
        let mut overlay = self.predict.overlay(&self.grid, cols, rows);
        let sel = self.select.overlay(&self.grid, cols, rows, reserved);
        overlay.push_str(&sel);
        if !overlay.is_empty() {
            // `paint` parks the cursor last; the overlays land after that, so
            // re-park or the cursor is left wherever the highlight ended and is
            // re-asserted there on every frame
            if !sel.is_empty() {
                overlay.push_str(&self.renderer.cursor_park(&self.grid, cols, rows));
            }
            const SYNC_END: &str = "\x1b[?2026l";
            if let Some(pos) = out.rfind(SYNC_END) {
                out.insert_str(pos, &overlay);
            } else {
                out.push_str(&overlay);
            }
        }
        write_stdout(&out);
    }

    fn hint(&mut self, text: &str) {
        self.hint_for(text, Duration::from_millis(1500));
    }

    /// A hint that stays up longer than the usual flash, for one the user did
    /// not cause by typing and so is not already watching for.
    fn hint_for(&mut self, text: &str, ttl: Duration) {
        self.renderer.status(&format!("{} — {text}", self.mode_label()));
        self.paint();
        self.hint_clear_at = Some(Instant::now() + ttl);
    }

    /// A hint with no expiry, for work whose duration we don't know: an upload
    /// can outlast the usual 1.5s and the pane would otherwise look idle while
    /// it is busy. Whoever set it replaces it when the work resolves.
    fn hint_sticky(&mut self, text: &str) {
        self.renderer.status(&format!("{} — {text}", self.mode_label()));
        self.paint();
        self.hint_clear_at = None;
    }

    /// Kick a background poll of the remote pane's foreground process, throttled
    /// so a mouse burst doesn't spawn an ssh per event. The result arrives as
    /// Msg::Foreground and updates `remote_is_shell`.
    fn spawn_foreground_poll(&mut self, force: bool) {
        let Some(gen) = self.session.as_ref().map(|s| s.gen) else { return };
        let now = Instant::now();
        if !force && self.fg_poll_at.is_some_and(|t| now.duration_since(t) < FG_POLL_INTERVAL) {
            return;
        }
        self.fg_poll_at = Some(now);
        let tx = self.tx.clone();
        let ssh = self.args.ssh_target.clone();
        let bin = self.args.remote_bin.clone();
        let session = self.args.session.clone();
        let pane = self.args.pane_target.clone();
        let ctl = self.args.ctl_path.clone();
        let container = self.args.container.clone();
        tokio::spawn(async move {
            let v = crate::foreground::poll(
                &ssh,
                bin.as_deref(),
                session.as_deref(),
                &pane,
                ctl.as_deref(),
                container.as_ref(),
            )
            .await;
            let _ = tx.send(Msg::Foreground { gen, value: v }).await;
        });
    }

    /// Hold the local mouse grab for the streamer's whole lifetime. The pane
    /// is always on the alt screen (no scrollback), so herdr's released-grab
    /// wheel routing has nothing to move; ?1007l already prevents the
    /// arrow-key hijacking that motivated the original shell-side release
    /// (#69). Text selection stays in the plugin selector.
    fn sync_mouse_grab(&mut self) {
        if !self.tty {
            return;
        }
        if self.mouse_grabbed {
            return;
        }
        self.mouse_grabbed = true;
        write_stdout("\x1b[?1002h\x1b[?1006h");
    }

    /// Match the local pane's cursor-key mode to the remote's, so the arrow bytes
    /// herdr hands us are already the ones the remote app expects.
    ///
    /// Frames carry no DEC modes (see grid.rs), so a remote app in application
    /// cursor mode (DECCKM, what terminfo `smkx` sets) never moves the local
    /// pane out of normal mode: herdr encodes Up as CSI A, we forward it
    /// verbatim, and the remote app is listening for SS3 A. Rather than rewrite
    /// the bytes in flight, put the LOCAL pane in the same mode and let herdr's
    /// own encoder produce the right form: it also covers Home/End and anything
    /// else whose encoding turns on this mode.
    ///
    /// The classification is the same shell/TUI proxy the mouse grab uses, for
    /// the same reason: the API exposes no input modes to ask for directly.
    fn sync_cursor_key_mode(&mut self) {
        if !self.tty {
            return;
        }
        // a shell prompt reads arrows in normal mode; a TUI is the case that
        // sets smkx, so mirror application mode unless we've confirmed a shell
        let want = matches!(self.remote_fg, Some(Fg::Agent) | Some(Fg::Mouse));
        if want == self.app_cursor_keys {
            return;
        }
        self.app_cursor_keys = want;
        write_stdout(if want { "\x1b[?1h" } else { "\x1b[?1l" });
    }

    fn observe_size(&self) -> (usize, usize) {
        let (cols, rows) = observe_size_for(&self.args, if self.tty { term_size() } else { (0, 0) });
        (cols.max(self.grid.width), rows.max(self.grid.height))
    }

    /// Size to enter (and stay in) control at. Control is authoritative on the
    /// remote — the server resizes the remote pty to whatever we ask for — so a
    /// host whose remote has its own display caps this and renders the remote at
    /// its own geometry, leaving the rest of the local pane blank rather than
    /// reflowing a screen someone over there is reading. Uncapped by default,
    /// which is the pre-existing fill-the-pane behaviour.
    fn control_size(&self) -> (usize, usize) {
        cap_size(term_size(), self.args.max_cols, self.args.max_rows)
    }

    /// FINAL detach only. Ordinary Observe transitions never close this pipe or
    /// replace the transport; EOF also releases any acquisition still in flight.
    async fn stop_session(&mut self) {
        if let Some(mut s) = self.session.take() {
            let _ = tokio::time::timeout(Duration::from_millis(150), async {
                let _ = s.stdin.write_all(b"{\"type\":\"terminal.release\"}\n").await;
                let _ = s.stdin.shutdown().await;
            }).await;
            tokio::time::sleep(Duration::from_millis(100)).await;
            kill_session_process_group(s.process_group);
        }
    }

    async fn connect(&mut self, desired: Mode) {
        self.mode = Mode::Observe;
        self.control_since = None;
        self.desired_mode = desired;
        self.state_ready = false;
        self.pending_mode = None;
        self.state_error = None;
        self.unsupported = false;
        self.resize_pending = false;
        self.predict = Predictor::new();
        self.select.clear();
        self.cancel_paste();
        let (cols, rows) = self.observe_size();
        if let Some(s) = self.session.take() {
            kill_session_process_group(s.process_group);
        }
        self.next_gen += 1;
        match spawn_session(&self.args, cols, rows, self.next_gen, self.tx.clone()) {
            Ok(s) => {
                self.session = Some(s);
                self.restore_mode_status();
                self.spawn_foreground_poll(false);
                self.paint();
            }
            Err(e) => self.schedule_reconnect(&e.to_string()),
        }
    }

    fn schedule_reconnect(&mut self, reason: &str) {
        let gone = target_gone(reason, &self.args.pane_target);
        let (delay, idx) = reconnect_delay(gone, self.backoff_idx);
        self.backoff_idx = idx;
        self.reconnect_at = Some(Instant::now() + Duration::from_millis(delay));
        self.state_error = Some(if gone {
            format!("remote pane {} is gone", self.args.pane_target)
        } else {
            let suffix = if reason.is_empty() { String::new() } else {
                format!(" — {}", sanitize_title(reason))
            };
            format!("disconnected — reconnecting in {}s (managed Observe){suffix}", delay / 1000)
        });
        self.restore_mode_status();
        self.paint();
    }


    async fn switch_mode(&mut self, mode: Mode) {
        self.request_mode(mode, false).await;
    }

    async fn request_mode(&mut self, mode: Mode, refresh: bool) {
        self.desired_mode = mode;
        if self.session.is_none() {
            self.reconnect_at = Some(Instant::now());
            return;
        }
        // Startup is itself acknowledged. An early keystroke (or explicit
        // always-control startup intent) waits for state 0, not merely spawn().
        if !self.state_ready {
            self.restore_mode_status();
            return;
        }
        if self.pending_mode.is_some_and(|p| p.mode == mode)
            || (!refresh && self.pending_mode.is_none() && self.mode == mode) {
            return;
        }
        self.next_request += 1;
        let request_id = self.next_request;
        self.pending_mode = Some(ModeRequest { request_id, mode });
        self.state_error = None;
        self.predict = Predictor::new();
        self.select.clear();
        self.renderer.invalidate();
        self.restore_mode_status();
        self.paint();
        let (cols, rows) = match mode {
            Mode::Control => self.control_size(),
            Mode::Observe => self.observe_size(),
        };
        self.write_message(json!({
            "type": if mode == Mode::Control { "terminal.acquire" } else { "terminal.observe" },
            "request_id": request_id, "cols": cols, "rows": rows,
        })).await;
    }

    async fn release_control(&mut self, idle: bool) {
        self.control_sticky = idle;
        self.pending_input.clear();
        self.cancel_paste();
        self.switch_mode(Mode::Observe).await;
    }

    async fn handle_resize(&mut self) {
        self.renderer.invalidate();
        // Preserve the explicit always-control option, but never auto-acquire
        // for the default Observe mode or retry a denied lease on window drags.
        if self.args.always_control && self.mode == Mode::Observe && !self.control_sticky {
            self.switch_mode(Mode::Control).await;
        }
        if self.can_write() {
            let (cols, rows) = self.control_size();
            self.send(json!({ "type": "terminal.resize", "cols": cols, "rows": rows })).await;
        } else if self.pending_mode.is_some() || !self.state_ready {
            self.resize_pending = true;
        } else if self.session.is_some() {
            // Observe geometry is a viewport request, never terminal.resize.
            self.request_mode(Mode::Observe, true).await;
        }
        self.paint();
    }

    async fn handle_remote(&mut self, gen: u64, message: StreamMessage) {
        if self.session.as_ref().map(|s| s.gen) != Some(gen) {
            return;
        }
        match message {
            StreamMessage::State(state) => self.handle_state(gen, state).await,
            StreamMessage::Frame(frame) if self.state_ready => self.handle_frame(frame),
            StreamMessage::Frame(_) => self.handle_exit(
                gen, "managed stream sent frames without an initial terminal.state; upgrade remote Herdr CLI and server",
            ),
            StreamMessage::Closed { reason } => self.handle_exit(
                gen, reason.as_deref().unwrap_or("remote terminal closed"),
            ),
            StreamMessage::Unknown => {}
        }
    }

    async fn handle_state(&mut self, gen: u64, state: TerminalState) {
        if self.session.as_ref().map(|s| s.gen) != Some(gen)
            || state.target != self.args.pane_target {
            return;
        }
        let requested = if !self.state_ready {
            if state.request_id != 0 { return; }
            Mode::Observe
        } else {
            let Some(request) = self.pending_mode else { return };
            if request.request_id != state.request_id { return; }
            request.mode
        };
        let initial = !self.state_ready;
        if let Some(error) = state.error.as_deref() {
            if initial || managed_unsupported(error) {
                self.handle_exit(gen, error);
                return;
            }
        }
        if state.error.is_none()
            && (state.cols == 0 || state.rows == 0 || state.mode != requested) {
            self.handle_exit(gen, "invalid managed terminal.state acknowledgement");
            return;
        }
        if !initial {
            self.pending_mode = None;
        }
        self.state_ready = true;
        if state.mode == Mode::Observe {
            self.control_since = None;
        } else if self.mode != Mode::Control {
            self.control_since = Some(Instant::now());
        }
        self.mode = state.mode;
        self.predict = Predictor::new();
        self.select.clear();
        if let Some(error) = state.error {
            // Denial is a real state reply, not a transport failure. Keep the
            // stream viewable and do not later replay rejected keystrokes.
            self.pending_input.clear();
            self.cancel_paste();
            self.desired_mode = state.mode;
            self.control_sticky = true;
            self.state_error = Some(format!("{} failed: {}", requested.as_str(), sanitize_title(&error)));
        } else {
            self.state_error = None;
            if initial && self.desired_mode == Mode::Control {
                self.switch_mode(Mode::Control).await;
                return;
            }
            self.desired_mode = state.mode;
            if self.mode == Mode::Control {
                self.last_input = Instant::now();
                if std::mem::take(&mut self.resize_pending) {
                    let (cols, rows) = self.control_size();
                    self.send(json!({ "type": "terminal.resize", "cols": cols, "rows": rows })).await;
                }
                // Take once; duplicate or superseded acknowledgements cannot
                // re-enter this flush. Raw input keeps mouse routing, while
                // pastes stay literal and never turn into local shortcuts.
                let pending = std::mem::take(&mut self.pending_input);
                for input in pending {
                    if !self.can_write() { break; }
                    match input {
                        PendingInput::Raw(bytes) => self.handle_stdin_inner(bytes).await,
                        PendingInput::Literal(bytes) => self.deliver_input(bytes).await,
                    }
                }
            } else if self.resize_pending {
                self.resize_pending = false;
                self.request_mode(Mode::Observe, true).await;
                return;
            }
        }
        self.restore_mode_status();
        self.paint();
    }

    fn handle_frame(&mut self, frame: Frame) {
        let Some(bytes) = &frame.bytes else { return };
        self.backoff_idx = 0;
        // Preserve the original transient status/viewport convention. Only the
        // state handler may announce CONTROL, never spawn or frame arrival.
        self.renderer.status("");
        let (fw, fh) = (
            frame.width.unwrap_or(self.grid.width),
            frame.height.unwrap_or(self.grid.height),
        );
        // A reflow or a full redraw replaces every cell, so a selection anchored
        // to grid coordinates would survive pointing at different text — and a
        // release afterwards would copy whatever landed there, or nothing, while
        // still showing a highlight that implies it worked.
        if (fw, fh) != (self.grid.width, self.grid.height) || frame.full == Some(true) {
            self.select.clear();
        }
        self.grid.resize(fw, fh);
        if frame.full == Some(true) {
            self.grid.clear();
        }
        if let Ok(decoded) = B64.decode(bytes) {
            self.grid.apply(&String::from_utf8_lossy(&decoded));
            // reconcile predictive echo against the authoritative frame
            self.predict.on_frame(&self.grid);
        }
        if self.args.dump {
            let lines: Vec<String> = self.grid.text_lines().into_iter().filter(|l| !l.is_empty()).collect();
            println!(
                "--- frame seq={:?} full={:?} {}x{} ---\n{}",
                frame.seq,
                frame.full,
                frame.width.unwrap_or(0),
                frame.height.unwrap_or(0),
                lines.join("\n")
            );
        } else {
            self.paint();
        }
    }

    fn handle_exit(&mut self, gen: u64, reason: &str) {
        if self.session.as_ref().map(|s| s.gen) != Some(gen) {
            return;
        }
        if self.mode == Mode::Control {
            self.control_failures = if self.control_since.is_some_and(|t| t.elapsed() < QUICK_CONTROL_FAILURE) {
                self.control_failures + 1
            } else {
                0
            };
            if self.control_failures >= 2 {
                self.control_failures = 0;
                self.control_sticky = true;
            }
        }
        self.control_since = None;
        if let Some(s) = self.session.take() {
            kill_session_process_group(s.process_group);
        }
        let discarded = !self.pending_input.is_empty() || self.paste_inflight;
        self.pending_input.clear();
        self.cancel_paste();
        self.mode = Mode::Observe;
        self.state_ready = false;
        self.pending_mode = None;
        self.resize_pending = false;
        self.predict = Predictor::new();
        self.select.clear();
        self.remote_fg = None;
        self.settle_at = None;
        self.desired_mode = if self.args.always_control && !self.control_sticky {
            initial_mode(true, term_size())
        } else {
            Mode::Observe
        };
        let reason = reason.lines().map(str::trim).rfind(|l| !l.is_empty()).unwrap_or("");
        self.unsupported = managed_unsupported(reason);
        if self.unsupported {
            self.reconnect_at = None;
            self.restore_mode_status();
            self.paint();
        } else {
            let reason = if discarded { format!("{reason}; unsent input discarded") } else { reason.into() };
            self.schedule_reconnect(&reason);
        }
    }

    async fn write_message(&mut self, msg: serde_json::Value) -> bool {
        let Some(s) = self.session.as_mut() else { return false };
        let gen = s.gen;
        let line = msg.to_string() + "\n";
        match s.stdin.write_all(line.as_bytes()).await {
            Ok(()) => true,
            Err(e) => {
                // Partial-write delivery is unknowable: never replay it into a
                // new connection, where it could execute twice.
                self.handle_exit(gen, &format!("managed terminal stdin failed: {e}"));
                false
            }
        }
    }

    async fn send(&mut self, msg: serde_json::Value) -> bool {
        if !self.can_write() { return false; }
        self.write_message(msg).await
    }

    /// Route ordinary input after preserving a partial SGR mouse sequence.
    async fn route_mouse_input(&mut self, bytes: Vec<u8>) {
        match split_mouse(&mut self.mouse_buf, bytes) {
            MouseSplit::Pending => {}
            MouseSplit::Passthrough(bytes) => self.route_input(bytes).await,
        }
        self.mouse_flush_at = (!self.mouse_buf.is_empty())
            .then(|| Instant::now() + MOUSE_INPUT_TIMEOUT);
    }

    /// Flush an unfinished mouse prefix after its short completion window.
    async fn flush_mouse_input(&mut self) {
        self.mouse_flush_at = None;
        if let Some(bytes) = flush_mouse(&mut self.mouse_buf) {
            self.route_input(bytes).await;
        }
    }

    /// Drain every complete paste in this chunk, in order.
    ///
    /// Deliberately a loop, not a one-shot: two drops land in a single read
    /// with nothing between them (a drop carries no terminator at all — which
    /// is precisely why `run` asks for DECSET 2004), so handling only the
    /// first would silently swallow the second, and leave its markers in the
    /// tail to be forwarded raw at the remote.
    async fn handle_stdin(&mut self, chunk: Vec<u8>) {
        let mut chunk = chunk;
        loop {
            match split_paste(&mut self.paste_buf, chunk) {
                PasteSplit::Pending => return,
                PasteSplit::Passthrough(bytes) => return self.route_mouse_input(bytes).await,
                PasteSplit::Complete { before, body, after } => {
                    self.route_mouse_input(before).await;
                    // A partial mouse sequence cannot continue into a paste;
                    // discard it before delivering the paste body directly.
                    if !self.mouse_buf.is_empty() {
                        self.mouse_buf.clear();
                        self.mouse_flush_at = None;
                    }
                    self.route_paste_body(body).await;
                    if after.is_empty() {
                        return;
                    }
                    chunk = after;
                }
            }
        }
    }

    /// Ordinary input, held back while an upload is in flight so the pasted
    /// remote paths cannot be overtaken by whatever was typed after them.
    async fn route_input(&mut self, bytes: Vec<u8>) {
        if !bytes.contains(&0x1c) {
            self.route_raw(bytes).await;
            return;
        }
        // A local release may share a read with typing, and must also cancel an
        // in-flight upload/acquire. Paste bodies bypass this path entirely.
        let mut start = 0;
        for (at, byte) in bytes.iter().enumerate() {
            if *byte == 0x1c {
                if start < at {
                    self.route_raw(bytes[start..at].to_vec()).await;
                }
                if !self.args.always_control {
                    self.release_control(false).await;
                }
                start = at + 1;
            }
        }
        if start < bytes.len() {
            self.route_raw(bytes[start..].to_vec()).await;
        }
    }

    async fn route_raw(&mut self, bytes: Vec<u8>) {
        if bytes.is_empty() { return; }
        if self.paste_inflight {
            self.paste_queue.push(Queued::Raw(bytes));
        } else {
            self.handle_stdin_inner(bytes).await;
        }
    }

    /// A complete paste body, markers already stripped. A file drop is
    /// uploaded; anything else is re-framed and forwarded as a paste.
    ///
    /// Re-framing is the whole point of getting here: the markers were only
    /// stripped so a drop could be recognised, and an ordinary paste that
    /// reaches the remote pty *without* them is not a paste any more — the
    /// remote app reads every newline in it as Enter, so a multi-line paste
    /// submits itself a line at a time (an agent's composer takes the first
    /// line and runs it). `deliver_input`, not `handle_stdin_inner`: a paste
    /// body is data, so its bytes must not be re-read as a Ctrl+V clipboard
    /// request, as mouse sequences, or as locally predicted keystrokes.
    async fn route_paste_body(&mut self, body: Vec<u8>) {
        if self.paste_inflight {
            self.paste_queue.push(Queued::Body(body));
            return;
        }
        // lossy only for the probe; the forward path keeps the original bytes
        let Some(paths) = crate::paste::dropped_paths(&String::from_utf8_lossy(&body)) else {
            self.deliver_input(crate::paste::bracketed_bytes(&body)).await;
            return;
        };

        self.paste_inflight = true;
        self.paste_original = Some(body);
        self.hint_sticky(&format!("uploading {} file(s)…", paths.len()));
        let tx = self.tx.clone();
        let ssh = self.args.ssh_target.clone();
        let ctl = self.args.ctl_path.clone();
        let container = self.args.container.clone();
        let epoch = self.input_epoch;
        tokio::spawn(async move {
            let result =
                crate::paste::files_to_remote(&paths, &ssh, ctl.as_deref(), container.as_ref())
                    .await;
            let _ = tx.send(Msg::Drop { epoch, result }).await;
        });
    }

    async fn handle_drop(&mut self, epoch: u64, result: crate::paste::DropResult) {
        if epoch != self.input_epoch || !self.paste_inflight { return; }
        self.paste_inflight = false;
        self.restore_mode_status();
        let original = self.paste_original.take();
        if let Some(text) = &result.text {
            self.deliver_input(crate::paste::bracketed(text)).await;
            self.hint(&format!("→ {text}"));
        } else if result.unchanged {
            // every path already exists over there, so the user meant those
            // files: forward what they actually dropped
            if let Some(body) = original {
                self.deliver_input(crate::paste::bracketed_bytes(&body)).await;
            }
        }
        if let Some(e) = result.error {
            self.hint(&format!("drop failed: {e}"));
        }
        self.drain_paste_queue().await;
    }

    /// Flush upload-blocked input in order. Either a file drop OR clipboard
    /// request can start another upload; a generation change cancels the rest.
    async fn drain_paste_queue(&mut self) {
        let epoch = self.input_epoch;
        let mut items = std::mem::take(&mut self.paste_queue).into_iter();
        while let Some(item) = items.next() {
            if epoch != self.input_epoch { return; }
            match item {
                Queued::Raw(b) => self.route_raw(b).await,
                Queued::Body(b) => self.route_paste_body(b).await,
            }
            if self.paste_inflight {
                self.paste_queue.extend(items);
                return;
            }
        }
    }

    async fn handle_stdin_inner(&mut self, buf: Vec<u8>) {
        if buf.is_empty() { return; }
        if buf == [0x1c] {
            if !self.args.always_control {
                self.release_control(false).await;
            }
            return;
        }
        if buf.len() == 1 && buf[0] == 0x16 && !self.paste_inflight {
            self.paste_inflight = true;
            let tx = self.tx.clone();
            let ssh = self.args.ssh_target.clone();
            let ctl = self.args.ctl_path.clone();
            let container = self.args.container.clone();
            let epoch = self.input_epoch;
            tokio::spawn(async move {
                let outcome =
                    crate::paste::clipboard_to_remote(&ssh, ctl.as_deref(), container.as_ref())
                        .await;
                let _ = tx.send(Msg::Paste { epoch, outcome }).await;
            });
            return;
        }
        if !self.can_write() {
            if has_mouse_seq(&buf) && !has_keyboard_input(&buf)
                && self.desired_mode != Mode::Control {
                // A stray wheel while glancing is read-only. After an idle
                // release, the wheel is real input and is retained through ack.
                if contains_wheel_press(&buf) && self.control_sticky {
                    self.control_sticky = false;
                    self.pending_input.push(PendingInput::Raw(buf));
                    self.switch_mode(Mode::Control).await;
                } else if contains_wheel_press(&buf) {
                    self.hint("read-only — type to take control");
                }
                return;
            }
            self.control_sticky = false;
            self.pending_input.push(PendingInput::Raw(buf));
            self.switch_mode(Mode::Control).await;
            return;
        }
        self.last_input = Instant::now();
        // refresh the foreground classification on any input while active.
        // the grab is always held, so mouse events also trigger this — that's
        // what catches a shell→TUI switch from a click, and the reverse from
        // `:q`. keyboard still does too, for people who never touch the mouse.
        self.spawn_foreground_poll(false);
        // and re-check shortly after input settles, to catch a change the input
        // just caused (e.g. `:q` quitting vim — the poll above still sees vim)
        self.settle_at = Some(Instant::now() + SETTLE_DELAY);
        // wheel becomes a semantic scroll (server decides app vs scrollback);
        // left-button drags go to the plugin selector; other TUI clicks/drags
        // forward to the remote pty. a shell never sees raw SGR.
        let mut rest: Vec<u8> = Vec::with_capacity(buf.len());
        let mut i = 0usize;
        let mut scrolls: Vec<serde_json::Value> = Vec::new();
        let mut sel_changed = false;
        let mut copy_span: Option<(crate::select::Pos, crate::select::Pos)> = None;
        while i < buf.len() {
            if let Some((btn, x, y, press, len)) = parse_mouse(&buf, i) {
                match mouse_action(self.remote_fg, btn, press) {
                    // Without a tty there is no grab, so this is unreachable in
                    // normal operation — but stdin could still be a pipe, and a
                    // selection there would be sized against a phantom viewport
                    // and would write OSC 52 into something that is not a
                    // terminal. Forward instead, which is what `main` did.
                    MouseAction::Select if !self.tty => {
                        rest.extend_from_slice(&buf[i..i + len]);
                    }
                    MouseAction::Select => {
                        let at = Select::locate(&self.grid, term_size().1, x, y);
                        let raw = &buf[i..i + len];
                        // motion flag (32) distinguishes a drag from the press
                        // that started it; `press` is the M/m final byte
                        match (press, btn & 32 != 0) {
                            (true, false) => self.select.press(at, raw),
                            (true, true) => self.select.drag(at),
                            (false, _) => match self.select.release(at, raw) {
                                // the clipboard holds one thing, so a second
                                // gesture in the same read legitimately wins
                                Released::Selection(span) => {
                                    copy_span = Some(span);
                                    // Finish this gesture before a later press in
                                    // the same read starts the next selection.
                                    sel_changed |= self.select.clear();
                                },
                                // It was a click, not a drag. TUI/agent get it
                                // (claude and codex discard the bytes cleanly).
                                // A shell does not: the prompt never enabled
                                // mouse reporting, and the bytes would dump
                                // into it.
                                Released::Click(bytes) => {
                                    if self.remote_fg != Some(Fg::Shell) {
                                        rest.extend_from_slice(&bytes);
                                    }
                                }
                                Released::Nothing => {}
                            },
                        }
                        sel_changed |= self.select.is_dirty();
                    }
                    MouseAction::Scroll { up } => {
                        // the viewport is about to move under the highlight,
                        // which is anchored to grid rows: leaving it up would
                        // paint reverse video over whatever scrolls into place
                        sel_changed |= self.select.clear();
                        scrolls.push(json!({
                            "type": "terminal.scroll",
                            "direction": if up { "up" } else { "down" },
                            "lines": 3,
                            "source": "wheel",
                            "column": x.saturating_sub(1),
                            "row": y.saturating_sub(1),
                            "modifiers": 0,
                        }));
                    }
                    MouseAction::ForwardRaw => rest.extend_from_slice(&buf[i..i + len]),
                    MouseAction::Drop => {}
                }
                i += len;
            } else {
                rest.push(buf[i]);
                i += 1;
            }
        }
        for s in scrolls {
            self.send(s).await;
        }
        if let Some((start, end)) = copy_span {
            let text = self.grid.selection_text(start, end);
            match crate::select::osc52(&text) {
                // no hint on success: herdr shows its own "copied to clipboard"
                // toast when it takes the OSC 52, so ours would be a duplicate
                Some(seq) => write_stdout(&seq),
                // too big for herdr to accept. Worth saying, because this is the
                // one path where nothing else reports: we never emit, so there
                // is no clipboard write for herdr to toast about.
                None if text.len() > 1024 => self.hint("selection too large to copy"),
                // an all-blank drag: leave the clipboard alone rather than
                // clearing it, and say nothing
                None => {}
            }
        }
        if !rest.is_empty() {
            // typing anywhere dismisses the highlight, the same as it would in a
            // local pane — otherwise it hangs over text the agent has redrawn.
            // `dismiss`, not `clear`: a press may have been buffered earlier in
            // this very read, and cancelling it would eat the click.
            sel_changed |= self.select.dismiss();
            let msg = json!({ "type": "terminal.input", "bytes": B64.encode(&rest) });
            let sent = self.send(msg).await;
            // No optimistic echo before acknowledgement or after failed writes.
            if sent && self.predict.on_input(&rest, &self.grid) {
                self.paint();
                sel_changed = false;
            }
        }
        if sel_changed {
            self.paint();
        }
    }

    async fn deliver_input(&mut self, buf: Vec<u8>) {
        if buf.is_empty() { return; }
        if !self.can_write() {
            self.control_sticky = false;
            self.pending_input.push(PendingInput::Literal(buf));
            self.switch_mode(Mode::Control).await;
            return;
        }
        self.last_input = Instant::now();
        self.send(json!({ "type": "terminal.input", "bytes": B64.encode(&buf) })).await;
    }

    async fn handle_paste(&mut self, epoch: u64, outcome: crate::paste::Outcome) {
        if epoch != self.input_epoch || !self.paste_inflight { return; }
        self.paste_inflight = false;
        self.restore_mode_status();
        match outcome {
            crate::paste::Outcome::NoImage => self.deliver_input(vec![0x16]).await,
            crate::paste::Outcome::Pasted(path) => {
                self.deliver_input(crate::paste::bracketed(&path)).await;
                self.hint(&format!("→ {path}"));
            }
            crate::paste::Outcome::Failed(e) => {
                self.hint(&format!("image paste failed: {e}"));
            }
        }
        self.drain_paste_queue().await;
    }
}

// ---------------------------------------------------------------------------
// main

/// Removes the streamer pidfile on any exit path out of `run` (stale files
/// from a hard kill are harmless — the daemon checks the pid is alive).
struct PidfileGuard(std::path::PathBuf);
impl Drop for PidfileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub async fn run(args: Args) -> Result<()> {
    let tty = !args.dump && unsafe { libc::isatty(libc::STDOUT_FILENO) } == 1;
    let local_pane_id = tty.then(|| std::env::var("HERDR_PANE_ID").ok()).flatten();
    let state_dir = crate::util::home_dir().join(".local").join("state").join("herdr-mirror");

    // Before either pidfile below exists: a pidfile is what tells the daemon
    // and the hooks they may SIGUSR1 us, and that signal's default disposition
    // is terminate. The daemon pokes a pane as soon as its first title lands,
    // which is typically while this streamer is still starting up.
    let mut sigusr1 = signal(SignalKind::user_defined1())?;
    // announce ourselves so the daemon can tell its typed `exec` took
    // (see util::streamer_pid_path); --dump is a human diagnostic, not a
    // daemon-spawned streamer, so it must not claim the slot
    let _pidfile = (!args.dump).then(|| {
        let path =
            crate::util::streamer_pid_path(&state_dir, &args.ssh_target, &args.pane_target);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(&path, std::process::id().to_string()).ok().map(|_| PidfileGuard(path))
    }).flatten();

    // Publish the pid before releasing the launch claim. Recovery always sees
    // at least one guard, even while herdr's process snapshot catches up.
    if _pidfile.is_some() {
        if let Some(id) = &local_pane_id {
            crate::util::clear_streamer_spawn_pending(&state_dir, id);
        }
    }

    // Say which local pane we are drawing, so anything holding only a herdr
    // pane id can find us. herdr hands every event hook the id of the pane it
    // is talking about and no way to write to it; this is how a hook reaches
    // the streamer sitting in that pane. HERDR_PANE_ID comes from herdr itself
    // and is inherited by whatever it starts in a pane, which is us.
    let _pane_pidfile = local_pane_id.as_deref().map(|id| {
        let path = crate::util::pane_pid_path(&state_dir, id);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(&path, std::process::id().to_string());
        // drop anything addressed to this pane before we existed: it describes
        // something that happened to a previous occupant
        let _ = crate::state::take_pane_hint(&state_dir, id);
        PidfileGuard(path)
    });
    let raw = if tty {
        // 1002/1006: button-event mouse tracking with SGR encoding, so wheel and
        // clicks reach us instead of scrolling the hosting pane's scrollback
        // 2004 (bracketed paste) is asked for so herdr frames a paste for us:
        // it only wraps one when the pane's app has enabled it, and a file
        // drop otherwise arrives as bare text with no terminator at all. The
        // framing is stripped only to recognise a drop and put back on the way
        // out (`route_paste_body`), so the remote app still sees a paste.
        // 1007 OFF (alternate scroll). The grab is always held, so mouse
        // reporting wins the routing and 1007 is never consulted. Kept as a
        // backstop: if the grab were ever lost, herdr would see "alt screen,
        // no mouse reporting, 1007 on" and type an Up/Down arrow per wheel
        // notch into us, which we'd forward, and a shell would walk command
        // history instead of scrolling (#69).
        write_stdout("\x1b[?1049h\x1b[2J\x1b[H\x1b[?1002h\x1b[?1006h\x1b[?2004h\x1b[?1007l");
        RawMode::enable()
    } else {
        None
    };

    let (tx, mut rx) = mpsc::channel::<Msg>(256);

    // stdin reader
    {
        let tx = tx.clone();
        tokio::spawn(async move {
            let mut stdin = tokio::io::stdin();
            let mut buf = [0u8; 1024];
            loop {
                match stdin.read(&mut buf).await {
                    Ok(0) | Err(_) => {
                        let _ = tx.send(Msg::StdinClosed).await;
                        break;
                    }
                    Ok(n) => {
                        if tx.send(Msg::Stdin(buf[..n].to_vec())).await.is_err() {
                            break;
                        }
                    }
                }
            }
        });
    }

    let mut app = App::new(args, tty, tx);
    // what this pane last wrote as its OSC 2 title, so repeats cost nothing
    let mut shown_title: Option<String> = None;
    relay_title(&state_dir, local_pane_id.as_deref(), &app.args.pane_target, &mut shown_title);
    // Control is authoritative on the remote: the server resizes the remote pty
    // to whatever we ask for, beating even a larger live client over there. So
    // entering Control with a size we cannot vouch for is what let a local herdr
    // with no client attached drag a healthy remote pane down to its 80x24
    // placeholder (#23). Observe never resizes anything, so it is the safe place
    // to wait: the first resize or keystroke proves a human and promotes us.
    // BEFORE connect: spawning the session awaits a process launch, and a
    // SIGWINCH arriving in that window is lost outright (its default disposition
    // is to be ignored). That window is exactly when a client attaching lays out
    // a freshly created pane — the resize we now promote on. Registered first,
    // tokio buffers it and delivers it once the loop starts.
    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;
    let mut sighup = signal(SignalKind::hangup())?; // pane closed — don't orphan the ssh child
    let mut sigwinch = signal(SignalKind::window_change())?;

    app.connect(initial_mode(app.args.always_control, term_size())).await;
    // the pane may have been laid out while the session was spawning; the signal
    // for that is buffered above, but check directly too
    if app.mode == Mode::Observe && initial_mode(app.args.always_control, term_size()) == Mode::Control
    {
        app.switch_mode(Mode::Control).await;
    } else if app.args.always_control && app.mode == Mode::Observe {
        // F3: otherwise the pane is inert with no explanation
        app.hint("read-only until this pane is sized — type to take control");
    }

    loop {
        // No mode-switch sleep: all transitions use the existing managed pipe.
        let idle_at = (app.can_write()
            && !app.args.always_control
            && app.args.control_idle_secs > 0)
            .then(|| app.last_input + Duration::from_secs(app.args.control_idle_secs));
        let sleep = crate::util::sleep_until_earliest([
            app.reconnect_at,
            app.hint_clear_at,
            idle_at,
            app.predict.deadline(),
            app.settle_at,
            app.mouse_flush_at,
        ]);

        tokio::select! {
            msg = rx.recv() => {
                match msg {
                    None | Some(Msg::StdinClosed) => break,
                    Some(Msg::Remote { gen, message }) => app.handle_remote(gen, message).await,
                    Some(Msg::SessionExit { gen, reason })
                        | Some(Msg::ProtocolError { gen, reason }) => app.handle_exit(gen, &reason),
                    Some(Msg::Stdin(buf)) => app.handle_stdin(buf).await,
                    // Keep the last good classification on failed/stale polls.
                    Some(Msg::Foreground { gen, value: v }) => if v.is_some()
                        && app.session.as_ref().map(|s| s.gen) == Some(gen) {
                        // a foreground change means the screen belongs to a
                        // different program now, so the old highlight points at
                        // text that is gone
                        if v != app.remote_fg && app.select.clear() {
                            app.paint();
                        }
                        app.remote_fg = v;
                        app.sync_mouse_grab();
                        app.sync_cursor_key_mode();
                    },
                    Some(Msg::Paste { epoch, outcome }) => app.handle_paste(epoch, outcome).await,
                    Some(Msg::Drop { epoch, result }) => app.handle_drop(epoch, result).await,
                }
            }
            _ = sigwinch.recv() => {
                app.handle_resize().await;
            }
            _ = sigusr1.recv() => {
                if let Some(id) = local_pane_id.as_deref() {
                    if let Some(msg) = crate::state::take_pane_hint(&state_dir, id) {
                        app.hint_for(&msg, Duration::from_secs(4));
                    }
                }
                // the daemon pokes the same way when the remote title changed
                relay_title(&state_dir, local_pane_id.as_deref(), &app.args.pane_target, &mut shown_title);
            }
            _ = sigterm.recv() => break,
            _ = sigint.recv() => break,
            _ = sighup.recv() => break,
            _ = sleep => {
                let now = Instant::now();
                if let Some(t) = app.reconnect_at {
                    if t <= now {
                        app.reconnect_at = None;
                        app.connect(app.desired_mode).await;
                    }
                }
                if app.hint_clear_at.is_some_and(|t| t <= now) {
                    app.hint_clear_at = None;
                    app.renderer.status("");
                    app.paint();
                }
                if idle_at.is_some_and(|t| t <= now) && app.can_write() {
                    app.release_control(true).await;
                }
                if app.settle_at.is_some_and(|t| t <= now) {
                    app.settle_at = None;
                    app.spawn_foreground_poll(true); // forced: bypass the throttle
                }
                if app.mouse_flush_at.is_some_and(|t| t <= now) {
                    app.flush_mouse_input().await;
                }
                if app.predict.deadline().is_some_and(|t| t <= now) {
                    app.predict.on_tick(); // wipe timed-out ghosts (no-echo prompts)
                    app.paint();
                }
            }
        }
    }

    // FINAL detach on local EOF, signal or pane close, including pending acquire.
    // Title relay and tty restoration remain independent of managed mode state.
    app.stop_session().await;
    if tty {
        // ?1l with the rest: leaving the hosting pane in application cursor mode
        // would misencode arrows for whatever runs there next
        // 1007 back on: it is a default-on mode we turned off, so leaving it
        // clear would silently change the wheel for whatever runs in this pane
        // after the streamer exits
        write_stdout("\x1b[?2004l\x1b[?1002l\x1b[?1006l\x1b[?1l\x1b[?1007h\x1b[?25h\x1b[?1049l");
    }
    if let Some(raw) = raw {
        raw.restore();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    type Commands = tokio::io::Lines<BufReader<tokio::process::ChildStdout>>;

    fn state(request_id: u64, mode: Mode, error: Option<&str>) -> TerminalState {
        TerminalState {
            request_id, target: "p1".into(), mode, cols: 80, rows: 24,
            error: error.map(str::to_string),
        }
    }

    fn frame(text: &str) -> Frame {
        Frame {
            seq: Some(1), full: Some(true), width: Some(80), height: Some(24),
            bytes: Some(B64.encode(text)),
        }
    }

    // A real pipe captures the production writer's JSON. It does not model
    // lease ownership or generate acknowledgements on behalf of the backend.
    async fn attach_sink(app: &mut App, gen: u64) -> (tokio::process::Child, Commands) {
        let mut command = tokio::process::Command::new("cat");
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).kill_on_drop(true);
        isolate_session_process_group(&mut command);
        let mut child = command.spawn().unwrap();
        let commands = BufReader::new(child.stdout.take().unwrap()).lines();
        app.session = Some(Session {
            gen, process_group: child.id().unwrap() as i32,
            stdin: child.stdin.take().unwrap(),
        });
        app.next_gen = gen;
        app.fg_poll_at = Some(Instant::now()); // Never probe SSH in unit tests.
        (child, commands)
    }

    async fn managed_app(ready: bool) -> (App, tokio::process::Child, Commands) {
        let args = parse_args(&["unused".into(), "p1".into()]).unwrap();
        let (tx, _rx) = mpsc::channel(256);
        let mut app = App::new(args, false, tx);
        let (child, commands) = attach_sink(&mut app, 1).await;
        app.state_ready = ready;
        app.restore_mode_status();
        (app, child, commands)
    }

    async fn command(commands: &mut Commands) -> serde_json::Value {
        let line = tokio::time::timeout(Duration::from_secs(1), commands.next_line())
            .await.expect("missing command").unwrap().expect("writer closed");
        serde_json::from_str(&line).unwrap()
    }

    async fn quiet(commands: &mut Commands) {
        assert!(tokio::time::timeout(Duration::from_millis(20), commands.next_line()).await.is_err(),
            "input or transition was written before its acknowledgement");
    }

    fn input_bytes(command: &serde_json::Value) -> Vec<u8> {
        assert_eq!(command["type"], "terminal.input");
        B64.decode(command["bytes"].as_str().unwrap()).unwrap()
    }

    async fn finish(
        mut app: App, mut child: tokio::process::Child, mut commands: Commands,
    ) -> Vec<serde_json::Value> {
        drop(app.session.take()); // Close the actual writer and read every last byte.
        drop(app);
        tokio::time::timeout(Duration::from_secs(1), child.wait()).await.unwrap().unwrap();
        let mut extra = Vec::new();
        while let Some(line) = commands.next_line().await.unwrap() {
            extra.push(serde_json::from_str(&line).unwrap());
        }
        extra
    }

    #[tokio::test]
    async fn consumer_preserves_fragmented_state_frame_and_denial_order() {
        let (mut writer, reader) = tokio::io::duplex(4096);
        let (tx, mut rx) = mpsc::channel(16);
        let consumer = tokio::spawn(async move { consume_stream(reader, 17, &tx).await });
        let wire = [
            json!({"type":"terminal.state","request_id":0,"target":"p1","mode":"observe","cols":80,"rows":24,"error":null}),
            json!({"type":"terminal.frame","seq":1,"full":true,"width":80,"height":24,"bytes":B64.encode("old")}),
            json!({"type":"terminal.state","request_id":4,"target":"p1","mode":"control","cols":100,"rows":30,"error":null}),
            json!({"type":"terminal.frame","seq":2,"full":true,"width":100,"height":30,"bytes":B64.encode("new")}),
            json!({"type":"terminal.state","request_id":5,"target":"p1","mode":"control","cols":100,"rows":30,"error":"release denied"}),
            json!({"type":"terminal.closed","reason":"ended"}),
        ].into_iter().map(|v| v.to_string() + "\n").collect::<String>();
        for bytes in wire.as_bytes().chunks(7) {
            writer.write_all(bytes).await.unwrap();
        }
        writer.shutdown().await.unwrap();
        drop(writer);
        let mut seen = Vec::new();
        while let Some(msg) = rx.recv().await {
            match msg {
                Msg::Remote { gen: 17, message: StreamMessage::State(s) } => {
                    seen.push(format!("state:{}:{}:{:?}", s.request_id, s.mode.as_str(), s.error));
                }
                Msg::Remote { gen: 17, message: StreamMessage::Frame(f) } => {
                    seen.push(format!("frame:{}", f.seq.unwrap()));
                }
                Msg::Remote { gen: 17, message: StreamMessage::Closed { reason } } => {
                    seen.push(reason.unwrap());
                }
                _ => panic!("unexpected consumer message"),
            }
        }
        assert_eq!(consumer.await.unwrap(), "ended");
        assert_eq!(seen, [
            "state:0:observe:None", "frame:1", "state:4:control:None", "frame:2",
            "state:5:control:Some(\"release denied\")", "ended",
        ]);
    }

    #[tokio::test]
    async fn consumer_rejects_malformed_states_instead_of_fabricating_ack() {
        let (tx, mut rx) = mpsc::channel(4);
        let bytes = b"{\"type\":\"terminal.state\",\"request_id\":1,\"target\":\"p1\",\"mode\":\"bogus\",\"cols\":80,\"rows\":24}\n";
        let reason = consume_stream(&bytes[..], 3, &tx).await;
        assert!(reason.contains("invalid managed terminal stream"));
        assert!(matches!(rx.recv().await, Some(Msg::ProtocolError { gen: 3, .. })));
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn first_input_waits_for_initial_state_and_exact_control_ack_once() {
        let (mut app, child, mut commands) = managed_app(false).await;
        let pid = child.id().unwrap() as i32;
        app.handle_stdin(b"\x1b[A".to_vec()).await;
        app.handle_stdin(b"b".to_vec()).await;
        assert_eq!(app.mode, Mode::Observe);
        assert!(!app.can_write());
        assert_eq!(app.pending_input.len(), 2);
        assert!(app.predict.deadline().is_none(), "no fake receipt by predictive echo");
        quiet(&mut commands).await;
        app.handle_remote(1, StreamMessage::State(state(0, Mode::Observe, None))).await;
        let acquire = command(&mut commands).await;
        assert_eq!(acquire["type"], "terminal.acquire");
        assert_eq!(acquire["request_id"], 1);
        assert!(acquire.get("takeover").is_none());
        assert!(app.mode_status().starts_with("viewing only"));
        app.handle_remote(0, StreamMessage::State(state(1, Mode::Control, None))).await;
        let mut wrong_target = state(1, Mode::Control, None);
        wrong_target.target = "other".into();
        app.handle_remote(1, StreamMessage::State(wrong_target)).await;
        app.handle_remote(1, StreamMessage::State(state(0, Mode::Observe, None))).await;
        app.handle_remote(1, StreamMessage::State(state(2, Mode::Control, None))).await;
        quiet(&mut commands).await;
        app.handle_remote(1, StreamMessage::State(state(1, Mode::Control, None))).await;
        assert_eq!(input_bytes(&command(&mut commands).await), b"\x1b[A");
        assert_eq!(input_bytes(&command(&mut commands).await), b"b");
        assert!(app.can_write());
        assert!(app.mode_status().starts_with("CONTROL"));
        assert_eq!(app.session.as_ref().unwrap().process_group, pid);
        app.handle_remote(1, StreamMessage::State(state(1, Mode::Control, None))).await;
        assert!(finish(app, child, commands).await.is_empty(), "duplicate ack replayed input");
    }

    #[tokio::test]
    async fn transitions_keep_one_stream_and_stale_observe_ack_cannot_flush_paste() {
        let (mut app, child, mut commands) = managed_app(true).await;
        let pid = child.id().unwrap() as i32;
        app.handle_stdin(b"first".to_vec()).await;
        assert_eq!(command(&mut commands).await["request_id"], 1);
        app.handle_remote(1, StreamMessage::State(state(1, Mode::Control, None))).await;
        assert_eq!(input_bytes(&command(&mut commands).await), b"first");
        app.handle_stdin(vec![0x1c]).await;
        let observe = command(&mut commands).await;
        assert_eq!(observe["type"], "terminal.observe");
        assert_eq!(observe["request_id"], 2);
        assert_eq!(app.mode, Mode::Control, "release is not acknowledged yet");
        assert!(!app.can_write());
        assert!(app.mode_status().contains("CONTROL — releasing"));
        let paste = crate::paste::bracketed_bytes(b"line one\n\x16\x1c\x1b[<64;1;1M\nline two");
        app.handle_stdin(paste.clone()).await;
        let acquire = command(&mut commands).await;
        assert_eq!(acquire["type"], "terminal.acquire");
        assert_eq!(acquire["request_id"], 3);
        app.handle_remote(1, StreamMessage::State(state(1, Mode::Control, None))).await;
        app.handle_remote(1, StreamMessage::State(state(2, Mode::Observe, None))).await;
        quiet(&mut commands).await;
        app.handle_remote(1, StreamMessage::State(state(3, Mode::Control, None))).await;
        assert_eq!(input_bytes(&command(&mut commands).await), paste);
        assert!(!app.paste_inflight, "literal Ctrl+V was interpreted as clipboard");
        app.handle_stdin(vec![0x1c]).await;
        assert_eq!(command(&mut commands).await["request_id"], 4);
        app.handle_remote(1, StreamMessage::State(state(4, Mode::Observe, None))).await;
        assert_eq!(app.mode, Mode::Observe);
        assert!(app.mode_status().starts_with("viewing only"));
        assert_eq!(app.session.as_ref().unwrap().process_group, pid);
        assert_eq!(app.next_gen, 1);
        assert!(finish(app, child, commands).await.is_empty());
    }

    #[tokio::test]
    async fn denied_acquire_keeps_viewable_stream_and_discards_rejected_input() {
        let (mut app, child, mut commands) = managed_app(true).await;
        let pid = child.id().unwrap() as i32;
        app.handle_stdin(b"rejected".to_vec()).await;
        command(&mut commands).await;
        app.handle_remote(1, StreamMessage::State(state(1, Mode::Observe, Some("owned by another client")))).await;
        assert!(app.pending_input.is_empty());
        assert!(!app.can_write());
        assert!(app.reconnect_at.is_none());
        assert!(app.control_sticky);
        assert!(app.mode_status().contains("owned by another client"));
        let rendered = app.renderer.paint(&app.grid, 100, 30);
        assert!(rendered.contains("viewing only") && rendered.contains("owned by another client"));
        app.handle_remote(1, StreamMessage::Frame(frame("still viewable"))).await;
        assert!(app.grid.text_lines()[0].contains("still viewable"));
        assert!(app.mode_status().contains("owned by another client"));
        app.handle_stdin(b"retry".to_vec()).await;
        let acquire = command(&mut commands).await;
        assert_eq!(acquire["request_id"], 2);
        app.handle_remote(1, StreamMessage::State(state(1, Mode::Control, None))).await;
        app.handle_remote(1, StreamMessage::State(state(2, Mode::Control, None))).await;
        assert_eq!(input_bytes(&command(&mut commands).await), b"retry");
        assert_eq!(app.session.as_ref().unwrap().process_group, pid);
        assert!(finish(app, child, commands).await.is_empty());
    }

    #[tokio::test]
    async fn denied_release_reports_authoritative_control_not_viewing() {
        let (mut app, child, mut commands) = managed_app(true).await;
        app.mode = Mode::Control;
        app.desired_mode = Mode::Control;
        app.release_control(false).await;
        assert_eq!(command(&mut commands).await["type"], "terminal.observe");
        app.handle_remote(1, StreamMessage::State(state(1, Mode::Control, Some("release refused")))).await;
        assert!(app.can_write());
        assert!(app.mode_status().starts_with("CONTROL"));
        assert!(app.mode_status().contains("release refused"));
        assert!(app.renderer.paint(&app.grid, 100, 30).contains("release refused"));
        app.handle_remote(1, StreamMessage::Frame(frame("owned"))).await;
        assert!(app.mode_status().starts_with("CONTROL"));
        assert!(finish(app, child, commands).await.is_empty());
    }

    #[tokio::test]
    async fn source_resize_is_only_sent_in_acknowledged_control() {
        let (mut app, child, mut commands) = managed_app(true).await;
        app.handle_resize().await;
        let view = command(&mut commands).await;
        assert_eq!(view["type"], "terminal.observe");
        assert_eq!(view["request_id"], 1);
        app.handle_stdin(b"x".to_vec()).await;
        assert_eq!(command(&mut commands).await["request_id"], 2);
        app.handle_resize().await;
        quiet(&mut commands).await;
        app.handle_remote(1, StreamMessage::State(state(1, Mode::Observe, None))).await;
        app.handle_remote(1, StreamMessage::State(state(2, Mode::Control, None))).await;
        assert_eq!(command(&mut commands).await["type"], "terminal.resize");
        assert_eq!(input_bytes(&command(&mut commands).await), b"x");
        app.release_control(false).await;
        assert_eq!(command(&mut commands).await["request_id"], 3);
        app.handle_resize().await;
        quiet(&mut commands).await;
        app.handle_remote(1, StreamMessage::State(state(3, Mode::Observe, None))).await;
        let view = command(&mut commands).await;
        assert_eq!(view["type"], "terminal.observe");
        assert_eq!(view["request_id"], 4);
        app.handle_remote(1, StreamMessage::State(state(4, Mode::Observe, None))).await;
        assert!(finish(app, child, commands).await.is_empty(), "resize escaped Observe gate");
    }

    #[tokio::test]
    async fn idle_release_and_first_wheel_retake_are_in_place_and_exactly_once() {
        let (mut app, child, mut commands) = managed_app(true).await;
        app.mode = Mode::Control;
        app.desired_mode = Mode::Control;
        app.release_control(true).await;
        assert_eq!(command(&mut commands).await["type"], "terminal.observe");
        assert_eq!(app.mode, Mode::Control);
        app.handle_remote(1, StreamMessage::State(state(1, Mode::Observe, None))).await;
        assert!(app.control_sticky);
        app.handle_stdin(b"\x1b[<64;7;9M".to_vec()).await;
        assert_eq!(command(&mut commands).await["request_id"], 2);
        app.handle_remote(1, StreamMessage::State(state(2, Mode::Control, None))).await;
        let wheel = command(&mut commands).await;
        assert_eq!(wheel["type"], "terminal.scroll");
        assert_eq!(wheel["direction"], "up");
        assert_eq!(wheel["column"], 6);
        assert_eq!(wheel["row"], 8);
        assert_eq!(app.next_gen, 1);
        assert!(finish(app, child, commands).await.is_empty());
    }

    #[tokio::test]
    async fn release_cancels_acquire_and_invalidates_paste_callbacks() {
        let (mut app, child, mut commands) = managed_app(true).await;
        app.handle_stdin(b"cancel".to_vec()).await;
        command(&mut commands).await;
        app.paste_inflight = true;
        let old_epoch = app.input_epoch;
        app.handle_stdin(vec![0x1c]).await;
        let observe = command(&mut commands).await;
        assert_eq!(observe["type"], "terminal.observe");
        assert_eq!(observe["request_id"], 2);
        assert!(app.pending_input.is_empty());
        app.handle_remote(1, StreamMessage::State(state(1, Mode::Control, None))).await;
        app.handle_remote(1, StreamMessage::State(state(2, Mode::Observe, None))).await;
        app.paste_inflight = true; // A newer upload must not be cleared by the old callback.
        app.handle_paste(old_epoch, crate::paste::Outcome::Pasted("/stale/image.png".into())).await;
        app.handle_drop(old_epoch, crate::paste::DropResult {
            text: Some("/stale/drop".into()), unchanged: false, error: None,
        }).await;
        assert!(app.paste_inflight);
        assert_eq!(app.mode, Mode::Observe);
        assert!(finish(app, child, commands).await.is_empty());
    }

    #[tokio::test]
    async fn clipboard_result_and_following_typing_keep_literal_order_through_ack() {
        let (mut app, child, mut commands) = managed_app(true).await;
        app.paste_inflight = true;
        let epoch = app.input_epoch;
        app.handle_stdin(b"after".to_vec()).await;
        assert_eq!(app.paste_queue.len(), 1);
        app.handle_paste(epoch, crate::paste::Outcome::NoImage).await;
        assert_eq!(command(&mut commands).await["type"], "terminal.acquire");
        assert_eq!(app.pending_input.len(), 2);
        app.handle_remote(1, StreamMessage::State(state(1, Mode::Control, None))).await;
        assert_eq!(input_bytes(&command(&mut commands).await), [0x16]);
        assert_eq!(input_bytes(&command(&mut commands).await), b"after");
        assert!(!app.paste_inflight);
        assert!(finish(app, child, commands).await.is_empty());
    }

    #[tokio::test]
    async fn reconnect_drops_old_input_and_correlates_new_generation_monotonically() {
        let (mut app, mut old_child, mut old_commands) = managed_app(true).await;
        app.handle_stdin(b"old unsent".to_vec()).await;
        assert_eq!(command(&mut old_commands).await["request_id"], 1);
        app.handle_exit(1, "transport lost");
        assert!(app.pending_input.is_empty());
        assert!(app.reconnect_at.is_some());
        assert!(app.mode_status().contains("unsent input discarded"));
        tokio::time::timeout(Duration::from_secs(1), old_child.wait()).await.unwrap().unwrap();
        let (child, mut commands) = attach_sink(&mut app, 2).await;
        app.reconnect_at = None;
        app.handle_stdin(b"new".to_vec()).await;
        app.handle_remote(2, StreamMessage::State(state(0, Mode::Observe, None))).await;
        assert_eq!(command(&mut commands).await["request_id"], 2);
        app.handle_remote(1, StreamMessage::State(state(2, Mode::Control, None))).await;
        app.handle_remote(1, StreamMessage::Frame(frame("stale screen"))).await;
        app.handle_exit(1, "stale exit");
        assert_eq!(app.session.as_ref().unwrap().gen, 2);
        assert_eq!(app.grid.width, 0);
        quiet(&mut commands).await;
        app.handle_remote(2, StreamMessage::State(state(2, Mode::Control, None))).await;
        assert_eq!(input_bytes(&command(&mut commands).await), b"new");
        assert!(finish(app, child, commands).await.is_empty());
        assert!(old_commands.next_line().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn unsupported_cli_and_unacknowledged_stream_require_matching_upgrade() {
        for reason in [
            "error: unrecognized subcommand 'managed'",
            "unsupported endpoint control kind terminal.session.set_mode",
            "managed terminal sessions require matching Herdr CLI and server upgrade",
        ] {
            let (mut app, child, commands) = managed_app(false).await;
            app.handle_remote(1, StreamMessage::Closed { reason: Some(reason.into()) }).await;
            assert!(app.unsupported, "{reason}");
            assert!(app.session.is_none());
            assert!(app.reconnect_at.is_none(), "must not cold-fallback");
            assert!(app.mode_status().contains("upgrade the remote Herdr CLI and server"));
            assert!(finish(app, child, commands).await.is_empty());
        }
        let (mut app, child, commands) = managed_app(false).await;
        app.handle_remote(1, StreamMessage::Frame(frame("not an acknowledgement"))).await;
        assert!(app.unsupported);
        assert_eq!(app.grid.width, 0);
        assert!(finish(app, child, commands).await.is_empty());
    }

    #[tokio::test]
    async fn explicit_always_control_intent_survives_initial_observe_ack_and_resize() {
        let (mut app, child, mut commands) = managed_app(false).await;
        app.args.always_control = true;
        app.desired_mode = initial_mode(true, (141, 44));
        assert_eq!(app.desired_mode, Mode::Control);
        app.handle_resize().await;
        quiet(&mut commands).await;
        app.handle_remote(1, StreamMessage::State(state(0, Mode::Observe, None))).await;
        assert_eq!(command(&mut commands).await["type"], "terminal.acquire");
        assert!(!app.can_write());
        app.handle_remote(1, StreamMessage::State(state(1, Mode::Control, None))).await;
        assert_eq!(command(&mut commands).await["type"], "terminal.resize");
        assert!(app.can_write());
        assert_eq!(app.mode_status(), "CONTROL");
        app.handle_stdin(vec![0x1c]).await; // Existing opt-in flag swallows release.
        assert!(finish(app, child, commands).await.is_empty());
    }

    #[tokio::test]
    async fn repeated_quick_control_losses_fall_back_without_auto_reacquire() {
        let (mut app, mut old_child, mut old_commands) = managed_app(true).await;
        app.args.always_control = true;
        app.handle_stdin(b"one".to_vec()).await;
        command(&mut old_commands).await;
        app.handle_remote(1, StreamMessage::State(state(1, Mode::Control, None))).await;
        command(&mut old_commands).await;
        app.handle_exit(1, "quick control loss");
        assert_eq!(app.control_failures, 1);
        tokio::time::timeout(Duration::from_secs(1), old_child.wait()).await.unwrap().unwrap();
        let (child, mut commands) = attach_sink(&mut app, 2).await;
        app.reconnect_at = None;
        app.handle_stdin(b"two".to_vec()).await;
        app.handle_remote(2, StreamMessage::State(state(0, Mode::Observe, None))).await;
        assert_eq!(command(&mut commands).await["request_id"], 2);
        app.handle_remote(2, StreamMessage::State(state(2, Mode::Control, None))).await;
        command(&mut commands).await;
        app.handle_exit(2, "second quick loss");
        assert!(app.control_sticky);
        assert_eq!(app.desired_mode, Mode::Observe);
        assert_eq!(app.control_failures, 0);
        assert!(finish(app, child, commands).await.is_empty());
    }

    #[tokio::test]
    async fn final_cleanup_detaches_pending_acquire_and_closes_stdin() {
        let (mut app, mut child, mut commands) = managed_app(true).await;
        app.handle_stdin(b"never acknowledged".to_vec()).await;
        assert_eq!(command(&mut commands).await["type"], "terminal.acquire");
        app.stop_session().await;
        assert!(app.session.is_none());
        let release = command(&mut commands).await;
        assert_eq!(release, json!({"type":"terminal.release"}));
        tokio::time::timeout(Duration::from_secs(1), child.wait()).await.unwrap().unwrap();
        assert!(commands.next_line().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn pane_transport_cleanup_terminates_proxy_command() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let pid_path = std::env::temp_dir().join(format!(
            "herdr-mirror-pane-proxy-{}-{nonce}.pid",
            std::process::id()
        ));
        let proxy = format!(
            "ProxyCommand=sh -c 'echo $$ > {}; exec sleep 30'",
            pid_path.display()
        );
        let mut command = tokio::process::Command::new("ssh");
        command
            .args([
                "-o",
                "BatchMode=yes",
                "-o",
                &proxy,
                "proxy-test.invalid",
                "true",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        isolate_session_process_group(&mut command);
        let mut child = command.spawn().unwrap();
        let process_group = child.id().unwrap() as i32;

        for _ in 0..40 {
            if pid_path.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let proxy_pid: i32 = std::fs::read_to_string(&pid_path)
            .unwrap()
            .trim()
            .parse()
            .unwrap();

        kill_session_process_group(process_group);
        tokio::time::timeout(Duration::from_secs(1), child.wait())
            .await
            .expect("ssh did not exit after its process group was killed")
            .unwrap();

        let mut proxy_survived = false;
        for _ in 0..20 {
            proxy_survived = unsafe { libc::kill(proxy_pid, 0) } == 0;
            if !proxy_survived {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        if proxy_survived {
            unsafe { libc::kill(proxy_pid, libc::SIGKILL) };
        }
        let _ = std::fs::remove_file(pid_path);
        assert!(!proxy_survived, "ProxyCommand survived pane transport cleanup");
    }

    /// Uncapped must stay byte-identical to the old `term_size()` call, or
    /// every existing headless-remote config silently changes behaviour.
    #[test]
    fn uncapped_control_size_is_the_local_size() {
        assert_eq!(cap_size((253, 50), None, None), (253, 50));
    }

    #[test]
    fn caps_only_bite_when_the_local_pane_is_bigger() {
        // the real case: local 253 cols vs a laptop that renders at 212
        assert_eq!(cap_size((253, 50), Some(212), Some(58)), (212, 50));
        // a local pane smaller than the cap is left alone — a cap is a ceiling,
        // never a demand for a size the local window can't show
        assert_eq!(cap_size((120, 30), Some(212), Some(58)), (120, 30));
        // one axis capped, the other free
        assert_eq!(cap_size((253, 50), Some(212), None), (212, 50));
        assert_eq!(cap_size((253, 90), None, Some(58)), (253, 58));
        // equal is not clamped away
        assert_eq!(cap_size((212, 58), Some(212), Some(58)), (212, 58));
    }

    #[test]
    fn wheel_always_semantic_scroll_even_on_tui_foreground() {
        // remote foreground classified as a TUI (e.g. `claude`) — wheel must
        // still produce a semantic scroll, not a raw forward, or it silently
        // does nothing when the TUI doesn't consume mouse wheel input
        assert_eq!(mouse_action(Some(Fg::Agent), 64, true), MouseAction::Scroll { up: true });
        assert_eq!(mouse_action(Some(Fg::Agent), 65, true), MouseAction::Scroll { up: false });
        // unclassified/shell foreground: wheel still scrolls
        assert_eq!(mouse_action(None, 64, true), MouseAction::Scroll { up: true });
        assert_eq!(mouse_action(Some(Fg::Shell), 65, true), MouseAction::Scroll { up: false });
        // the wheel never reaches the selection path at all, which is what
        // keeps PR #54's scroll regression impossible here by construction
        assert_eq!(mouse_action(Some(Fg::Agent), 0, true), MouseAction::Select);
        assert_eq!(mouse_action(Some(Fg::Shell), 0, true), MouseAction::Select); // shell drag-select
        assert_eq!(mouse_action(None, 0, true), MouseAction::Drop); // unclassified click
    }

    #[test]
    fn every_remote_tui_selects_on_drag_agent_or_not() {
        for fg in [Fg::Agent, Fg::Mouse] {
            assert_eq!(mouse_action(Some(fg), 0, true), MouseAction::Select, "{fg:?} press");
            assert_eq!(mouse_action(Some(fg), 32, true), MouseAction::Select, "{fg:?} drag");
            assert_eq!(mouse_action(Some(fg), 0, false), MouseAction::Select, "{fg:?} release");
            // other buttons still reach the app
            assert_eq!(mouse_action(Some(fg), 1, true), MouseAction::ForwardRaw);
            assert_eq!(mouse_action(Some(fg), 2, true), MouseAction::ForwardRaw);
        }
        // a shell holds the grab too, so the plugin selector runs; middle/right
        // stay dropped so SGR never hits a prompt. unknown fails safe.
        assert_eq!(mouse_action(Some(Fg::Shell), 0, true), MouseAction::Select);
        assert_eq!(mouse_action(Some(Fg::Shell), 32, true), MouseAction::Select);
        assert_eq!(mouse_action(Some(Fg::Shell), 0, false), MouseAction::Select);
        assert_eq!(mouse_action(Some(Fg::Shell), 1, true), MouseAction::Drop);
        assert_eq!(mouse_action(Some(Fg::Shell), 2, true), MouseAction::Drop);
        assert_eq!(mouse_action(None, 0, true), MouseAction::Drop);
    }

    #[test]
    fn a_modified_wheel_still_scrolls_instead_of_becoming_a_click() {
        // The button number is the low two bits PLUS bit 6, so shift+wheel-up is
        // 68 and `btn & 0b11` reads it as a left press: the wheel then drives the
        // selection, the scroll is lost, and a later left release replays the
        // wheel bytes to the app as a click.
        for mods in [4, 8, 16, 4 + 8, 4 + 16, 8 + 16, 4 + 8 + 16] {
            assert_eq!(
                mouse_action(Some(Fg::Agent), 64 + mods, true),
                MouseAction::Scroll { up: true },
                "shift/alt/ctrl + wheel-up (btn {})",
                64 + mods
            );
            assert_eq!(
                mouse_action(Some(Fg::Agent), 65 + mods, true),
                MouseAction::Scroll { up: false },
                "modified wheel-down (btn {})",
                65 + mods
            );
        }
        // horizontal wheel keeps dropping, modified or not
        assert_eq!(mouse_action(Some(Fg::Agent), 66 + 16, true), MouseAction::Drop);
        // buttons 8-11 carry bit 7, which the old mask also leaked
        assert_eq!(mouse_action(Some(Fg::Agent), 128, true), MouseAction::ForwardRaw);
        assert_eq!(mouse_action(Some(Fg::Agent), 128 + 4, true), MouseAction::ForwardRaw);
        // a wheel release is not a left release
        assert_eq!(mouse_action(Some(Fg::Agent), 64, false), MouseAction::ForwardRaw);
    }

    #[test]
    fn button_numbers_decode_the_split_encoding() {
        assert_eq!(button_number(0), 0); // left
        assert_eq!(button_number(32), 0); // left, dragging
        assert_eq!(button_number(16 + 32), 0); // ctrl + left drag
        assert_eq!(button_number(1), 1);
        assert_eq!(button_number(2), 2);
        assert_eq!(button_number(64), 4); // wheel up
        assert_eq!(button_number(68), 4); // shift + wheel up
        assert_eq!(button_number(65), 5);
        assert_eq!(button_number(66), 6);
        assert_eq!(button_number(128), 8);
        assert_eq!(button_number(131), 11);
    }

    #[test]
    fn mouse_parsing() {
        let seq = b"\x1b[<64;10;5M";
        let (btn, x, y, press, len) = parse_mouse(seq, 0).unwrap();
        assert_eq!((btn, x, y, press, len), (64, 10, 5, true, seq.len()));
        assert!(contains_wheel_press(seq));
        assert!(!contains_wheel_press(b"\x1b[<0;3;4M")); // click, not wheel
        assert!(!contains_wheel_press(b"\x1b[<64;10;5m")); // release, not press
        assert!(has_mouse_seq(b"xx\x1b[<0;1;1Myy"));
        assert!(!has_mouse_seq(b"plain text"));
    }

    #[test]
    fn mouse_sequence_split_across_reads_is_reassembled() {
        let mut buf = Vec::new();
        assert_eq!(
            split_mouse(&mut buf, b"pre\x1b[<0;3;2".to_vec()),
            MouseSplit::Passthrough(b"pre".to_vec())
        );
        assert_eq!(buf, b"\x1b[<0;3;2");
        assert_eq!(
            split_mouse(&mut buf, b"Mpost".to_vec()),
            MouseSplit::Passthrough(b"\x1b[<0;3;2Mpost".to_vec())
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn mouse_prefix_can_split_at_escape_and_csi_introducer() {
        let mut buf = Vec::new();
        assert_eq!(split_mouse(&mut buf, b"\x1b".to_vec()), MouseSplit::Pending);
        assert_eq!(split_mouse(&mut buf, b"[".to_vec()), MouseSplit::Pending);
        assert_eq!(
            split_mouse(&mut buf, b"<0;3;2M".to_vec()),
            MouseSplit::Passthrough(b"\x1b[<0;3;2M".to_vec())
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn timed_out_mouse_prefix_is_dropped_but_escape_survives() {
        let mut mouse = b"\x1b[<0;3;2".to_vec();
        assert_eq!(flush_mouse(&mut mouse), None);
        assert!(mouse.is_empty());

        let mut escape = b"\x1b".to_vec();
        assert_eq!(flush_mouse(&mut escape), Some(b"\x1b".to_vec()));
        assert!(escape.is_empty());
    }


    #[test]
    fn sh_quote_escapes_single_quotes() {
        assert_eq!(sh_quote("w9:p1"), "'w9:p1'");
        assert_eq!(sh_quote("a'b"), "'a'\\''b'");
        // overflow-proof mouse params: 11 digits saturate instead of panicking
        let (_, x, _, _, _) = parse_mouse(b"\x1b[<64;99999999999;1M", 0).unwrap();
        assert_eq!(x, u32::MAX);
    }

    #[test]
    fn observe_size_treats_daemon_sizes_as_a_floor() {
        // what the daemon spawns a streamer with for a headless remote: the
        // no-client placeholder rect plus its margin
        let argv: Vec<String> = ["work", "w1:p1", "--cols", "70", "--rows", "31"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let a = parse_args(&argv).unwrap();
        // control has already resized the remote pty to this pane, and release
        // does not shrink it back — observing at 70x31 would stream a crop
        assert_eq!(observe_size_for(&a, (314, 92)), (314, 92));
        // a pane smaller than the remote still gets the daemon's margin
        assert_eq!(observe_size_for(&a, (40, 20)), (70, 31));
        // --dump has no tty: exactly what was asked for
        assert_eq!(observe_size_for(&a, (0, 0)), (70, 31));
    }

    #[test]
    fn a_zero_cap_on_the_cli_is_unset_not_a_zero_request() {
        // herdr rejects a 0-column terminal, so a typo would kill the session
        // twice and strand the pane in "control unavailable"
        let argv: Vec<String> = ["h", "w1:p1", "--max-cols", "0", "--max-rows", "0"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let a = parse_args(&argv).unwrap();
        assert_eq!(a.max_cols, None);
        assert_eq!(a.max_rows, None);

        let argv: Vec<String> =
            ["h", "w1:p1", "--max-cols", "212"].iter().map(|s| s.to_string()).collect();
        assert_eq!(parse_args(&argv).unwrap().max_cols, Some(212));
    }

    #[test]
    fn arg_parsing() {
        let argv: Vec<String> =
            ["work", "w9:p1", "--remote-bin", "/opt/herdr", "--cols", "176", "--rows", "66"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        let a = parse_args(&argv).unwrap();
        assert_eq!(a.ssh_target, "work");
        assert_eq!(a.pane_target, "w9:p1");
        assert_eq!(a.remote_bin.as_deref(), Some("/opt/herdr"));
        assert_eq!((a.cols, a.rows), (176, 66));
        assert!(parse_args(&["onlyone".to_string()]).is_err());
        assert!(parse_args(&["a".into(), "b".into(), "--visibility-file".into(), "x".into()]).is_err());
    }

    // --- birth size trust (#23) ---
    //
    // herdr renders at 80x24 when no client is attached, and chrome only
    // subtracts, so anything larger in either axis is provably a real viewport.

    #[test]
    fn a_placeholder_sized_pane_is_never_trusted() {
        // what a mirror pane is born as when nobody is watching: 80x24 less a
        // 26-col sidebar and the tab bar
        assert!(!size_is_trusted((54, 23)));
        // and the extremes of that layout, in case chrome is configured away
        assert!(!size_is_trusted((80, 24)));
        assert!(!size_is_trusted((80, 23)));
    }

    #[test]
    fn an_ordinary_viewport_is_trusted_immediately() {
        assert!(size_is_trusted((141, 44)));
        assert!(size_is_trusted((133, 47)));
        // one axis is enough: a tall narrow pane cannot come from a 24-row floor
        assert!(size_is_trusted((60, 40)));
        assert!(size_is_trusted((200, 20)));
    }

    #[test]
    fn initial_mode_is_read_only_unless_the_size_vouches_for_itself() {
        // the whole composition: only a trusted size under always_control opens
        // writable, because Control is what can resize the remote
        assert_eq!(initial_mode(true, (141, 44)), Mode::Control);
        assert_eq!(initial_mode(true, (54, 23)), Mode::Observe, "placeholder-sized");
        // without always_control we start read-only regardless, as before
        assert_eq!(initial_mode(false, (141, 44)), Mode::Observe);
        assert_eq!(initial_mode(false, (54, 23)), Mode::Observe);
    }

    #[test]
    fn a_small_client_is_not_trusted_at_birth_and_must_earn_control() {
        // A phone (45x18 -> pane 44x16) and moshi (50x25 -> 49x23) are real
        // viewports, but at birth they are indistinguishable from the placeholder
        // — that is the whole reason shape matching failed. They start read-only
        // and the first resize or keystroke promotes them, rather than being
        // allowed to impose a size we cannot vouch for.
        assert!(!size_is_trusted((44, 16)));
        assert!(!size_is_trusted((49, 23)));
    }

    // --- paste framing -----------------------------------------------------
    // The bugs these pin were all real: a one-shot version dropped the second
    // drop in a read, leaked markers from the tail, and corrupted non-UTF-8.

    const S: &[u8] = b"\x1b[200~";
    const E: &[u8] = b"\x1b[201~";

    fn split(buf: &mut Vec<u8>, chunk: &[u8]) -> PasteSplit {
        split_paste(buf, chunk.to_vec())
    }

    fn seq(parts: &[&[u8]]) -> Vec<u8> {
        parts.concat()
    }

    #[test]
    fn ordinary_typing_passes_straight_through() {
        let mut buf = Vec::new();
        assert_eq!(split(&mut buf, b"a"), PasteSplit::Passthrough(b"a".to_vec()));
        assert_eq!(split(&mut buf, b"\x1b[A"), PasteSplit::Passthrough(b"\x1b[A".to_vec()));
        assert!(buf.is_empty(), "typing must not buffer");
    }

    #[test]
    fn paste_split_across_reads_reassembles() {
        let mut buf = Vec::new();
        assert_eq!(split(&mut buf, &seq(&[S, b"he"])), PasteSplit::Pending);
        assert_eq!(split(&mut buf, b"llo"), PasteSplit::Pending);
        assert_eq!(
            split(&mut buf, E),
            PasteSplit::Complete { before: vec![], body: b"hello".to_vec(), after: vec![] }
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn start_marker_split_across_reads_is_not_leaked() {
        // the 6-byte introducer straddling a read boundary must be held, not
        // forwarded in pieces (which would print ESC[20 at the remote)
        let mut buf = Vec::new();
        assert_eq!(split(&mut buf, b"\x1b[20"), PasteSplit::Pending);
        assert_eq!(
            split(&mut buf, &seq(&[b"0~/tmp/a.png", E])),
            PasteSplit::Complete { before: vec![], body: b"/tmp/a.png".to_vec(), after: vec![] }
        );
    }

    #[test]
    fn bytes_around_the_markers_are_preserved() {
        let mut buf = Vec::new();
        assert_eq!(
            split(&mut buf, &seq(&[b"pre", S, b"mid", E, b"post"])),
            PasteSplit::Complete {
                before: b"pre".to_vec(),
                body: b"mid".to_vec(),
                after: b"post".to_vec(),
            }
        );
    }

    #[test]
    fn two_pastes_in_one_read_are_drained_in_order() {
        // the regression: only the first was handled and the rest discarded,
        // so a second drop vanished and its markers reached the remote raw
        let mut buf = Vec::new();
        let PasteSplit::Complete { body, after, .. } =
            split(&mut buf, &seq(&[S, b"one", E, S, b"two", E]))
        else {
            panic!("expected first paste")
        };
        assert_eq!(body, b"one");
        assert_eq!(
            split(&mut buf, &after),
            PasteSplit::Complete { before: vec![], body: b"two".to_vec(), after: vec![] },
            "feeding the tail back must yield the second paste"
        );
    }

    #[test]
    fn keystroke_after_a_paste_survives() {
        let mut buf = Vec::new();
        let PasteSplit::Complete { after, .. } = split(&mut buf, &seq(&[S, b"/tmp/a", E, b"\r"]))
        else {
            panic!("expected paste")
        };
        assert_eq!(after, b"\r", "the trailing keystroke must not be eaten");
    }

    #[test]
    fn unterminated_paste_flushes_at_the_cap_without_duplicating() {
        let mut buf = Vec::new();
        assert_eq!(split(&mut buf, S), PasteSplit::Pending);
        let big = vec![b'x'; MAX_PASTE_BYTES + 1];
        let PasteSplit::Passthrough(out) = split(&mut buf, &big) else {
            panic!("expected a flush")
        };
        assert_eq!(out.len(), S.len() + big.len(), "every byte exactly once");
        assert!(buf.is_empty(), "buffer must not keep a copy");
    }

    #[test]
    fn non_utf8_paste_body_is_forwarded_byte_exact() {
        // the body is only lossily decoded to probe for paths; what gets
        // forwarded must be the original bytes
        let mut buf = Vec::new();
        let PasteSplit::Complete { body, .. } = split(&mut buf, &seq(&[S, b"caf\xe9", E])) else {
            panic!("expected paste")
        };
        assert_eq!(body, b"caf\xe9", "0xE9 must not become U+FFFD");
    }

    #[test]
    fn a_stripped_paste_is_reframed_byte_exact() {
        // what `route_paste_body` sends on: the framing comes off to recognise
        // a drop and has to go back on, or the remote app reads the newlines in
        // a multi-line paste as Enter and submits it a line at a time
        let body = b"first line\nsecond line".to_vec();
        let framed = crate::paste::bracketed_bytes(&body);
        assert_eq!(framed, seq(&[S, &body, E]));

        let mut buf = Vec::new();
        assert_eq!(
            split(&mut buf, &framed),
            PasteSplit::Complete { before: vec![], body, after: vec![] },
            "re-framing must be exactly what the splitter undoes"
        );
    }

    #[test]
    fn target_gone_matches_only_this_pane_being_gone() {
        // the real sentence, captured from herdr against a missing pane
        assert!(target_gone(
            "terminal session observe failed: terminal target w9Z:p99 not found",
            "w9Z:p99"
        ));
        assert!(target_gone(
            "terminal session control failed: terminal target w1:p1 not found",
            "w1:p1"
        ));

        // the false positive that matters most: herdr absent on the remote.
        // The auto-resolver execs `$(command -v herdr || echo ~/.local/bin/herdr)`,
        // so a host without herdr fails with a shell not-found — a different
        // problem that a slow rung would wrongly paper over.
        assert!(!target_gone("sh: 1: exec: /home/u/.local/bin/herdr: not found", "w9Z:p99"));

        // a target that merely shares our prefix: p1 and p10 are both real ids
        assert!(!target_gone(
            "terminal session observe failed: terminal target w1:p10 not found",
            "w1:p1"
        ));
        // ...and another pane's disappearance is not ours
        assert!(!target_gone(
            "terminal session observe failed: terminal target w1:p4 not found",
            "w9Z:p99"
        ));

        // ordinary transients stay on the fast ladder
        assert!(!target_gone("api timeout: session.snapshot", "w9Z:p99"));
        assert!(!target_gone("ssh timeout", "w9Z:p99"));
        assert!(!target_gone("", "w9Z:p99"));
        // an empty target must not turn `contains` into "matches everything"
        assert!(!target_gone("terminal target w1:p1 not found", ""));
    }

    #[test]
    fn a_gone_target_slows_down_without_consuming_the_ladder() {
        // the fix: 10s forever becomes one attempt a minute
        assert_eq!(reconnect_delay(true, 0), (GONE_BACKOFF_MS, 0));
        assert_eq!(reconnect_delay(true, 3), (GONE_BACKOFF_MS, 3));

        // the fast ladder is unchanged and still clamps at its last rung
        assert_eq!(reconnect_delay(false, 0), (1000, 1));
        assert_eq!(reconnect_delay(false, 1), (2000, 2));
        assert_eq!(reconnect_delay(false, 3), (10000, 4));
        assert_eq!(reconnect_delay(false, 99), (10000, 100));

        // a gone spell must not burn rungs: a transient afterwards resumes
        // where the ladder was, rather than restarting at 1s
        let (_, idx) = reconnect_delay(false, 0);
        let (_, idx) = reconnect_delay(true, idx);
        assert_eq!(reconnect_delay(false, idx), (2000, 2));
    }


    // Exercise the real input routing: a release and the next press can share
    // one stdin read. A local sink stands in for the session; no SSH is used.
    async fn selection_followed_by_press(next_drag: bool) {
        let args = parse_args(&["unused".into(), "p1".into()]).unwrap();
        let tty = true;
        let (tx, _rx) = mpsc::channel(256);
        let mut app = App::new(args, tty, tx);
        let mut child = tokio::process::Command::new("cat")
            .stdin(Stdio::piped()).stdout(Stdio::null()).kill_on_drop(true)
            .spawn().unwrap();
        app.session = Some(Session {
            gen: 1, process_group: child.id().unwrap() as i32,
            stdin: child.stdin.take().unwrap(),
        });
        app.mode = Mode::Control;
        app.desired_mode = Mode::Control;
        app.state_ready = true;
        app.remote_fg = Some(Fg::Mouse);
        // Keep foreground polling local to this fixture: suppress SSH probes.
        app.fg_poll_at = Some(Instant::now());
        // Blank cells exercise copy handling without writing to the clipboard.
        app.grid.resize(80, 24);
        app.handle_stdin(
            b"\x1b[<0;1;1M\x1b[<32;4;1M\x1b[<0;4;1m\x1b[<0;1;2M".to_vec(),
        ).await;
        let result = app.select.release(
            if next_drag { (1, 3) } else { (1, 0) }, b"release",
        );
        drop(app);
        child.kill().await.unwrap();
        let _ = child.wait().await;
        if next_drag {
            assert_eq!(result, Released::Selection(((1, 0), (1, 3))));
        } else {
            assert_eq!(result, Released::Click(b"\x1b[<0;1;2Mrelease".to_vec()));
        }
    }

    #[tokio::test]
    async fn copying_selection_preserves_next_click_in_same_read() {
        selection_followed_by_press(false).await;
    }

    #[tokio::test]
    async fn copying_selection_preserves_next_drag_in_same_read() {
        selection_followed_by_press(true).await;
    }

    fn relayed(remote: &str, title: Option<&str>) -> Option<crate::state::PaneTitle> {
        Some(crate::state::PaneTitle {
            host: "vibe1".into(),
            remote: remote.into(),
            title: title.map(str::to_string),
        })
    }

    #[test]
    fn a_remote_title_cannot_break_out_of_the_osc() {
        // BEL and ESC \ end an OSC; C1 ST (U+009C) does too for a terminal
        // that reads C1. Anything after them would run as local terminal input.
        let hostile = "task\x07\x1b]0;owned\x1b\\\u{9c}\x1b[2J done";
        assert_eq!(sanitize_title(hostile), "task]0;owned\\[2J done");
        let mut shown = None;
        let seq = title_update(relayed("wD:p3", Some(hostile)), "wD:p3", &mut shown).unwrap();
        let body = seq.strip_prefix("\x1b]2;").and_then(|s| s.strip_suffix('\x07')).unwrap();
        assert!(!body.chars().any(char::is_control), "{seq:?}");
    }

    #[test]
    fn a_remote_title_is_bounded_on_a_char_boundary() {
        let long = "π".repeat(TITLE_MAX_CHARS + 50);
        let clean = sanitize_title(&long);
        assert_eq!(clean.chars().count(), TITLE_MAX_CHARS);
        assert!(clean.chars().all(|c| c == 'π'));
    }

    #[test]
    fn the_remote_title_is_written_once_per_change() {
        let mut shown = None;
        assert_eq!(
            title_update(relayed("wD:p3", Some("π ⠴ jev-game")), "wD:p3", &mut shown).as_deref(),
            Some("\x1b]2;π ⠴ jev-game\x07")
        );
        // a poke for a hint, or a relay that repeats itself, writes nothing
        assert_eq!(title_update(relayed("wD:p3", Some("π ⠴ jev-game")), "wD:p3", &mut shown), None);
        assert_eq!(
            title_update(relayed("wD:p3", Some("π > jev-game")), "wD:p3", &mut shown).as_deref(),
            Some("\x1b]2;π > jev-game\x07")
        );
        // the remote dropping its title clears ours instead of keeping the last
        assert_eq!(title_update(relayed("wD:p3", None), "wD:p3", &mut shown).as_deref(), Some("\x1b]2;\x07"));
    }

    #[test]
    fn a_title_for_another_remote_pane_is_ignored() {
        // herdr reuses local pane ids: a record left for the previous holder of
        // this pane must not name the mirror now standing in it
        let mut shown = None;
        assert_eq!(title_update(relayed("wD:p1", Some("rpg designer")), "wD:p3", &mut shown), None);
        assert_eq!(shown, None);
        assert_eq!(title_update(None, "wD:p3", &mut shown), None);
    }

}
