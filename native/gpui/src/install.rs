//! Windows: a release unpacked from its zip installs itself where the guard can look after it (updates, restarts, rollback), when the person says so.
//! The layout is install.sh's: `<prefix>/builds/<tag>/` (the whole release), `<prefix>/CURRENT` (the tag), `<prefix>/backplane.exe` (the guard,
//! which with no arguments opens CURRENT's app: the Start-menu shortcut points at it, so it survives updates). `<prefix>` is
//! `%LOCALAPPDATA%\Backplane`. Uninstalling is docs/install.md's.
use std::path::{Path, PathBuf};

/// What an unpacked release has beside its client: the guard, the hub and the commit it was built from.
/* i18n-ok: file and env names */ pub const RELEASE_FILES: [&str; 3] = ["backplane.exe", "backplane-hub.exe", "COMMIT"];

/// Whether to ask: a client started with no `--url` (`pending`), not already run by a guard, not pointed at a hub by `BACKPLANE_HUB_BIN`
/// (a test or a dev run), from a folder that is an unpacked release (`has` says which files are beside it), and not already in `<prefix>/builds`.
pub fn offer(pending: bool, guarded: bool, hub_bin_env: bool, under_prefix: bool, has: impl Fn(&str) -> bool) -> bool {
    pending && !guarded && !hub_bin_env && !under_prefix && RELEASE_FILES.iter().all(|f| has(f))
}

/// `--install [DIR]`, the headless install (CI, scripts): no question, DIR or the default prefix. None when the arguments ask for none.
pub fn install_arg(args: &[String]) -> Option<Option<PathBuf>> {
    match args.get(1).map(String::as_str) { Some("--install") => Some(args.get(2).filter(|d| !d.starts_with("--")).map(PathBuf::from)), _ => None }
}

/// The tag of this build's folder: `v` and the version release.yml stamped into Cargo.toml (the updater names its builds the same way).
/* i18n-ok: file and env names */ pub fn tag() -> String { format!("v{}", env!("CARGO_PKG_VERSION")) }

/// Lays the release in `src` out under `prefix` and answers the installed client. The build is copied beside `builds/<tag>` and renamed into
/// place (whole or not at all); an existing `builds/<tag>` is kept. The guard is replaced by rename (Windows lets a running program be renamed,
/// not overwritten). CURRENT is written only when there is none: a later version on an installed machine moves through the guard
/// (`backplane hub update`), which checks it and can go back.
pub fn install(src: &Path, prefix: &Path, tag: &str) -> std::io::Result<PathBuf> {
    let builds = prefix.join("builds"); std::fs::create_dir_all(&builds)?;
    let dest = builds.join(tag);
    if !dest.join("backplane-gpui.exe").is_file() {
        let tmp = builds.join(format!(".install-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        copy_tree(src, &tmp)?;
        let _ = std::fs::remove_dir_all(&dest);
        std::fs::rename(&tmp, &dest)?;
    }
    let guard = prefix.join("backplane.exe");
    let fresh = prefix.join(format!(".backplane-{}.exe", std::process::id()));
    std::fs::copy(dest.join("backplane.exe"), &fresh)?;
    if guard.exists() { let old = prefix.join("backplane.exe.old"); let _ = std::fs::remove_file(&old); std::fs::rename(&guard, &old)?; }
    std::fs::rename(&fresh, &guard)?;
    /* i18n-ok: file and env names */ let current = prefix.join("CURRENT");
    if !current.is_file() { let t = prefix.join(".CURRENT.tmp"); std::fs::write(&t, format!("{tag}\n"))?; std::fs::rename(&t, &current)?; }
    // where no guard runs (the app uses a hub on another machine), nothing would move CURRENT: the app's next launch tries a newer build put
    // here by hand, as it tries one it fetched (appkeep.rs). A guard owns CURRENT when it runs.
    let cur = crate::appkeep::current(prefix);
    if !cur.is_empty() && crate::appkeep::newer(tag, &cur) && !crate::appkeep::guard_runs(prefix) {
        crate::appkeep::set_manual_next(prefix, tag)?;
    }
    Ok(dest.join("backplane-gpui.exe"))
}

fn copy_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)? {
        let e = e?; let t = e.file_type()?;
        if t.is_symlink() { continue } // a release has none; never follow one out of the folder
        if t.is_dir() { copy_tree(&e.path(), &to.join(e.file_name()))? } else { std::fs::copy(e.path(), to.join(e.file_name()))?; }
    }
    Ok(())
}

/// `backplane-gpui.exe --install [DIR]`: lay this unpacked release out in DIR (default `%LOCALAPPDATA%\Backplane`, which also gets the
/// Start-menu entry) with no question, print the installed client's path and end: 0 done, 2 not an unpacked release, 1 the copy failed.
/// Then `DIR\backplane.exe hub run --home H --port N --builds DIR -- --foreground` starts the guard for a home, as the client would.
#[cfg(windows)]
pub fn headless(dir: Option<PathBuf>) -> i32 {
    let Ok(me) = std::env::current_exe() else { return 1 };
    let Some(src) = me.parent() else { return 1 };
    if !RELEASE_FILES.iter().all(|f| src.join(f).is_file()) { eprintln!("--install: {} is not an unpacked release (it needs {})", src.display(), RELEASE_FILES.join(", ")); return 2 }
    let default = std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("Backplane"));
    let Some(prefix) = dir.or_else(|| default.clone()) else { eprintln!("--install: no folder given and no LOCALAPPDATA"); return 2 };
    match install(src, &prefix, &tag()) {
        Ok(app) => { if Some(&prefix) == default.as_ref() { shortcut(&prefix) } println!("{}", app.display()); 0 }
        Err(e) => { eprintln!("--install: {e}"); 1 }
    }
}

/// Prepare a tracked trial before spawning a manually installed newer build (`maybe_install`), acquiring an exclusive `clients.lock` and
/// ensuring no live trial or guard owns the prefix. Keeps `clients.lock` held until the child's PID is recorded or cleaned up.
pub(crate) fn prepare_upgrade_trial(prefix: &Path, tag: &str) -> Option<(String, std::fs::File)> {
    let lock = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(prefix.join("clients.lock")).ok()?;
    lock.try_lock().ok()?;
    let cur = crate::appkeep::current(prefix);
    if cur.is_empty() || !crate::appkeep::newer(tag, &cur) { return None; }
    let nonce = format!("inst-{}-{}", std::process::id(), crate::appkeep::now());
    crate::appkeep::begin_if_no_live_trial(prefix, tag, &cur, std::process::id(), &nonce).ok()?;
    Some((nonce, lock))
}

/// Windows, before the window opens: when `offer` says so, ask; on Install, install and start the installed client with the same arguments.
/// Answers true when this process should end (the installed one runs). Not now, or a failed install, runs this one as it is.
#[cfg(windows)]
pub fn maybe_install(pending: bool) -> bool {
    let Ok(me) = std::env::current_exe() else { return false };
    let Some(dir) = me.parent() else { return false };
    let Some(prefix) = std::env::var_os("LOCALAPPDATA").filter(|p| !p.is_empty()).map(|p| PathBuf::from(p).join("Backplane")) else { return false };
    let under = dir.starts_with(prefix.join("builds"));
    let guarded = crate::launch::guarded_install(&me).is_some();
    if !offer(pending, guarded, std::env::var_os("BACKPLANE_HUB_BIN").is_some(), under, |f| dir.join(f).is_file()) { return false }
    if !ask(&prefix) { return false }
    let t = tag();
    let app = match install(dir, &prefix, &t) {
        Ok(a) => a,
        Err(e) => { say(&crate::i18n::tf("gpui.install.failed", &[("dir", prefix.display().to_string().into()), ("why", e.to_string().into())])); return false }
    };
    shortcut(&prefix);
    use std::os::windows::process::CommandExt;
    let held = prepare_upgrade_trial(&prefix, &t);
    let mut cmd = std::process::Command::new(&app);
    cmd.args(std::env::args_os().skip(1)).creation_flags(0x0000_0200);
    if let Some((ref n, _)) = held { cmd.env(crate::selfupdate::TRIAL_ENV, n); }
    match cmd.spawn() {
        Ok(child) => {
            if let Some((ref n, lock)) = held {
                crate::appkeep::update_trial_pid(&prefix, &t, n, child.id());
                drop(lock);
            }
            true
        }
        Err(_) => {
            if let Some((ref n, lock)) = held {
                crate::appkeep::forget_if_nonce(&prefix, n);
                drop(lock);
            }
            false
        }
    }
}

#[cfg(windows)]
fn wide(s: &str) -> Vec<u16> { s.encode_utf16().chain(Some(0)).collect() }

/// One question with two buttons, Install and Not now (a task dialog: comctl32 v6, which gpui's manifest asks for).
#[cfg(windows)]
fn ask(prefix: &Path) -> bool {
    use windows_sys::Win32::UI::Controls::{TaskDialogIndirect, TASKDIALOGCONFIG, TASKDIALOG_BUTTON};
    let (title, main, body) = (wide("Backplane"), wide(&crate::i18n::tf("gpui.install.ask", &[("dir", prefix.display().to_string().into())])), wide(&crate::i18n::trs("Needed for updates. Not now runs it from this folder and asks again next time.")));
    let (b1, b2) = (wide(&crate::i18n::trs("Install")), wide(&crate::i18n::trs("Not now")));
    let buttons = [TASKDIALOG_BUTTON { nButtonID: 100, pszButtonText: b1.as_ptr() }, TASKDIALOG_BUTTON { nButtonID: 101, pszButtonText: b2.as_ptr() }];
    let mut c: TASKDIALOGCONFIG = unsafe { std::mem::zeroed() };
    c.cbSize = std::mem::size_of::<TASKDIALOGCONFIG>() as u32;
    c.pszWindowTitle = title.as_ptr(); c.pszMainInstruction = main.as_ptr(); c.pszContent = body.as_ptr();
    c.cButtons = buttons.len() as u32; c.pButtons = buttons.as_ptr(); c.nDefaultButton = 100;
    let mut pressed = 0i32;
    let hr = unsafe { TaskDialogIndirect(&c, &mut pressed, std::ptr::null_mut(), std::ptr::null_mut()) };
    hr == 0 && pressed == 100
}

#[cfg(windows)]
fn say(text: &str) {
    use windows_sys::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_OK};
    let (t, m) = (wide("Backplane"), wide(text));
    unsafe { MessageBoxW(std::ptr::null_mut(), m.as_ptr(), t.as_ptr(), MB_OK); }
}

/// The Start-menu entry: `<prefix>\backplane.exe` (the guard opens CURRENT's app), started minimised so its console stays out of sight.
#[cfg(windows)]
fn shortcut(prefix: &Path) {
    let Some(appdata) = std::env::var_os("APPDATA") else { return };
    /* i18n-ok: file and env names */ let lnk = PathBuf::from(appdata).join(r"Microsoft\Windows\Start Menu\Programs\Backplane.lnk");
    let q = |p: &Path| p.display().to_string().replace('\'', "''");
    /* i18n-ok: a PowerShell command */ let ps = format!("$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{}'); $s.TargetPath = '{}'; $s.WorkingDirectory = '{}'; $s.WindowStyle = 7; $s.IconLocation = '{},0'; $s.Description = 'Backplane'; $s.Save()",
        q(&lnk), q(&prefix.join("backplane.exe")), q(prefix), q(&prefix.join("backplane.exe")));
    use std::os::windows::process::CommandExt;
    let _ = std::process::Command::new("powershell.exe").args(["-NoProfile", "-NonInteractive", "-Command", &ps]).creation_flags(0x0800_0000).status();
}

fn contains_ci(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w.eq_ignore_ascii_case(needle))
}

fn contains_utf16le_ci(haystack: &[u8], needle: &str) -> bool {
    let pat: Vec<u8> = needle.encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
    if pat.is_empty() { return false }
    haystack.windows(pat.len()).any(|w| {
        w.chunks_exact(2).zip(pat.chunks_exact(2)).all(|(a, b)| {
            let ca = u16::from_le_bytes([a[0], a[1]]);
            let cb = u16::from_le_bytes([b[0], b[1]]);
            if ca < 128 && cb < 128 { (ca as u8).eq_ignore_ascii_case(&(cb as u8)) } else { ca == cb }
        })
    })
}

/// Fast pre-filter on raw `.lnk` bytes: true when the shortcut still references a versioned `builds\` path or `backplane-gpui`
/// (from pre-#1228 installs) so already-migrated `<prefix>\backplane.exe,0` shortcuts never spawn PowerShell.
pub fn shortcut_bytes_need_icon_repair(raw: &[u8]) -> bool {
    for pat in ["\\builds\\", "backplane-gpui"] { // i18n-ok: Windows shortcut byte signature
        if contains_ci(raw, pat.as_bytes()) || contains_utf16le_ci(raw, pat) { return true }
    }
    false
}

/// Chooses `<prefix>\backplane.exe` when it embeds an icon (survives future upgrades), else falls back to `current_exe` when it does.
pub fn choose_icon_exe(guard: PathBuf, guard_has: bool, app: PathBuf, app_has: bool) -> Option<PathBuf> {
    if guard_has { Some(guard) } else if app_has { Some(app) } else { None }
}

/// PowerShell snippet that updates `Backplane.lnk` `IconLocation` only if `TargetPath` points to `<prefix>\backplane.exe`.
pub fn shortcut_icon_repair_ps(lnk: &Path, prefix: &Path, icon_exe: &Path) -> String {
    let q = |p: &Path| p.display().to_string().replace('\'', "''");
    let guard = prefix.join("backplane.exe"); // i18n-ok: Windows binary name
    let want_icon = format!("{},0", q(icon_exe)); // i18n-ok: Windows icon location
    format!("$lnk = '{}'; $guard = '{}'; $want = '{}'; $sh = New-Object -ComObject WScript.Shell; $s = $sh.CreateShortcut($lnk); if ($s.TargetPath -ieq $guard -and $s.IconLocation -ine $want) {{ $s.IconLocation = $want; $s.Save() }}", q(lnk), q(&guard), want_icon) // i18n-ok: PowerShell shortcut migration
}

#[cfg(windows)]
fn exe_has_icon(path: &Path) -> bool {
    if !path.is_file() { return false }
    let w: Vec<u16> = path.as_os_str().to_string_lossy().encode_utf16().chain(Some(0)).collect();
    unsafe { windows_sys::Win32::UI::WindowsAndMessaging::PrivateExtractIconsW(w.as_ptr(), 0, 32, 32, std::ptr::null_mut(), std::ptr::null_mut(), 0, 0) > 0 }
}

/// Repairs a legacy Start-menu `Backplane.lnk` whose `IconLocation` still points into `builds\...` (or `backplane-gpui.exe`).
#[cfg(windows)]
pub fn repair_start_menu_shortcut_in(appdata: &Path, prefix: &Path, current_exe: &Path) -> bool {
    let lnk = appdata.join("Microsoft").join("Windows").join("Start Menu").join("Programs").join("Backplane.lnk"); // i18n-ok: Windows Start Menu path
    let Ok(raw) = std::fs::read(&lnk) else { return false };
    if !shortcut_bytes_need_icon_repair(&raw) { return false }
    let Some(icon_exe) = choose_icon_exe(prefix.join("backplane.exe"), exe_has_icon(&prefix.join("backplane.exe")), current_exe.to_path_buf(), exe_has_icon(current_exe)) else { return false }; // i18n-ok: Windows binary name
    let ps = shortcut_icon_repair_ps(&lnk, prefix, &icon_exe);
    use std::os::windows::process::CommandExt;
    std::process::Command::new("powershell.exe").args(["-NoProfile", "-NonInteractive", "-Command", &ps]).creation_flags(0x0800_0000).status().is_ok_and(|s| s.success()) // i18n-ok: PowerShell CLI args
}

/// Background self-heal on Windows when running from an installed `CURRENT` build: refreshes a stale `<prefix>\backplane.exe`
/// (e.g. upgraded from a pre-#1228/#1672 guard) and migrates any legacy Start-menu shortcut pointing at deleted/iconless builds.
#[cfg(windows)]
pub fn heal_installed_guard_and_shortcut() {
    let Ok(me) = std::env::current_exe() else { return };
    let Some((_, prefix)) = crate::launch::guarded_install(&me) else { return };
    let Some(tag) = me.parent().and_then(|d| d.file_name()).map(|n| n.to_string_lossy().into_owned()) else { return };
    let Some(appdata) = std::env::var_os("APPDATA").filter(|p| !p.is_empty()).map(PathBuf::from) else { return };
    let _ = std::thread::Builder::new().name("win-shortcut-repair".into()).spawn(move || { // i18n-ok: thread name
        if crate::appkeep::current(&prefix) == tag {
            crate::appkeep::refresh_guard(&prefix, &tag);
        }
        repair_start_menu_shortcut_in(&appdata, &prefix, &me);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tmp(n: &str) -> crate::tmp::TempDir { crate::tmp::dir(&format!("inst-{n}")) }

    #[test] fn only_an_unpacked_release_run_on_its_own_is_offered_the_install() {
        let all = |_: &str| true;
        assert!(offer(true, false, false, false, all));
        assert!(!offer(false, false, false, false, all), "--url: a hub was given");
        assert!(!offer(true, true, false, false, all), "already run by a guard");
        assert!(!offer(true, false, true, false, all), "BACKPLANE_HUB_BIN: a test or a dev run");
        assert!(!offer(true, false, false, true, all), "already in <prefix>/builds");
        assert!(!offer(true, false, false, false, |f: &str| f != "COMMIT"), "a build tree has no COMMIT beside the client");
        assert!(!offer(true, false, false, false, |f: &str| f != "backplane.exe"), "no guard to install");
    }

    #[test] fn install_takes_a_folder_or_none() {
        let a = |x: &[&str]| x.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(install_arg(&a(&["gpui", "--install", "D:/bp"])), Some(Some(PathBuf::from("D:/bp"))));
        assert_eq!(install_arg(&a(&["gpui", "--install"])), Some(None));
        assert_eq!(install_arg(&a(&["gpui", "--url", "x"])), None);
        assert_eq!(install_arg(&a(&["gpui"])), None);
    }
    #[test] fn the_install_is_install_sh_s_layout_and_keeps_what_is_there() {
        let (src, pre) = (tmp("src"), tmp("pre"));
        for f in ["backplane-gpui.exe", "backplane-hub.exe", "backplane.exe", "COMMIT"] { std::fs::write(src.join(f), f).unwrap(); }
        std::fs::create_dir_all(src.join("assets/icon")).unwrap(); std::fs::write(src.join("assets/icon/a.svg"), "svg").unwrap();
        std::fs::create_dir_all(src.join("licenses/rust")).unwrap(); std::fs::write(src.join("licenses/rust/hub.txt"), "notices").unwrap();
        let app = install(&src, &pre, "v0.15.0").unwrap();
        assert_eq!(app, pre.join("builds/v0.15.0/backplane-gpui.exe")); assert!(app.is_file());
        assert_eq!(std::fs::read_to_string(pre.join("builds/v0.15.0/assets/icon/a.svg")).unwrap(), "svg");
        assert_eq!(std::fs::read_to_string(pre.join("builds/v0.15.0/licenses/rust/hub.txt")).unwrap(), "notices", "the notices come with the build (Settings > About names them)");
        assert_eq!(std::fs::read_to_string(pre.join("backplane.exe")).unwrap(), "backplane.exe");
        assert_eq!(std::fs::read_to_string(pre.join("CURRENT")).unwrap(), "v0.15.0\n");
        // a second release: its build is added, the guard replaced, CURRENT left to the guard to move
        std::fs::write(src.join("backplane.exe"), "guard 2").unwrap();
        install(&src, &pre, "v0.15.1").unwrap();
        assert!(pre.join("builds/v0.15.1/backplane-gpui.exe").is_file() && pre.join("builds/v0.15.0/backplane-gpui.exe").is_file());
        assert_eq!(std::fs::read_to_string(pre.join("backplane.exe")).unwrap(), "guard 2");
        assert_eq!(std::fs::read_to_string(pre.join("CURRENT")).unwrap(), "v0.15.0\n", "the guard moves CURRENT, after its health check");
        assert_eq!(crate::appkeep::next(&pre), "v0.15.1", "or, with no guard, the app's next launch tries it");
        let left: Vec<String> = std::fs::read_dir(pre.join("builds")).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        assert!(left.iter().all(|n| !n.starts_with('.')), "no scratch left: {left:?}");
        #[cfg(windows)] assert!(crate::launch::guarded_install(&app).is_some(), "the installed client is run by the guard");
        let _ = (std::fs::remove_dir_all(&src), std::fs::remove_dir_all(&pre));
    }
    #[test] fn upgrade_trial_respects_guard_lock_and_promotes_without_guard() {
        let (src, pre) = (tmp("upg-src"), tmp("upg-pre"));
        for f in ["backplane-gpui.exe", "backplane-hub.exe", "backplane.exe", "COMMIT"] { std::fs::write(src.join(f), f).unwrap(); }
        install(&src, &pre, "v0.15.0").unwrap();
        assert_eq!(crate::appkeep::current(&pre), "v0.15.0");

        // When a guard holds .guard.lock, installer leaves CURRENT/next/trial alone
        let lock = std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(pre.join(".guard.lock")).unwrap();
        lock.lock_shared().unwrap();
        install(&src, &pre, "v0.16.0").unwrap();
        assert_eq!(crate::appkeep::next(&pre), "");
        assert!(prepare_upgrade_trial(&pre, "v0.16.0").is_none());
        drop(lock);

        // Without a guard, manual install writes manual next; while another window holds clients.lock, prepare_upgrade_trial defers
        install(&src, &pre, "v0.16.0").unwrap();
        assert_eq!(crate::appkeep::next(&pre), "v0.16.0");
        assert!(crate::appkeep::is_manual_next(&pre, "v0.16.0"));
        let shared = std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(pre.join("clients.lock")).unwrap();
        shared.lock_shared().unwrap();
        assert!(prepare_upgrade_trial(&pre, "v0.16.0").is_none(), "shared clients.lock blocks upgrade trial handoff");
        drop(shared);

        // With no competing window or guard, prepares a tracked trial and promotes cleanly
        let (nonce, held_lock) = prepare_upgrade_trial(&pre, "v0.16.0").unwrap();
        drop(held_lock);
        assert_eq!(crate::appkeep::opened(&pre, "v0.16.0", Some(&nonce)), Some("v0.16.0".into()));
        assert_eq!(crate::appkeep::current(&pre), "v0.16.0");
    }

    #[test] fn legacy_shortcut_icon_detection_and_target_selection() {
        let utf16 = |s: &str| s.encode_utf16().flat_map(|u| u.to_le_bytes()).collect::<Vec<u8>>();
        assert!(shortcut_bytes_need_icon_repair(br"C:\Users\u\AppData\Local\Backplane\builds\v0.15.4\backplane-gpui.exe"));
        assert!(shortcut_bytes_need_icon_repair(&utf16(r"C:\Users\u\AppData\Local\Backplane\BUILDS\v0.16.0\app.exe")));
        assert!(!shortcut_bytes_need_icon_repair(&utf16(r"C:\Users\u\AppData\Local\Backplane\backplane.exe")));
        let pre = PathBuf::from(r"C:\Users\u\AppData\Local\Backplane");
        let guard = pre.join("backplane.exe");
        let app = pre.join("builds").join("v0.16.4").join("backplane-gpui.exe");
        assert_eq!(choose_icon_exe(guard.clone(), true, app.clone(), true), Some(guard.clone()));
        assert_eq!(choose_icon_exe(guard.clone(), false, app.clone(), true), Some(app.clone()));
        assert_eq!(choose_icon_exe(guard.clone(), false, app, false), None);
        let ps = shortcut_icon_repair_ps(&pre.join("Backplane.lnk"), &pre, &guard);
        assert!(ps.contains("$s.TargetPath -ieq $guard") && ps.contains("backplane.exe,0"));
    }

    #[cfg(windows)]
    #[test] fn shortcut_icon_repair_migrates_legacy_builds_icon_in_temp_dir() {
        let root = tmp("lnk-repair");
        let appdata = root.join("appdata");
        let programs = appdata.join("Microsoft").join("Windows").join("Start Menu").join("Programs");
        std::fs::create_dir_all(&programs).unwrap();
        let lnk = programs.join("Backplane.lnk");
        let pre = root.join("Backplane");
        let cur_dir = pre.join("builds").join("v0.16.4");
        std::fs::create_dir_all(&cur_dir).unwrap();
        let me = std::env::current_exe().unwrap();
        assert!(exe_has_icon(&me), "gpui test exe should carry the embedded Backplane icon");
        let guard = pre.join("backplane.exe");
        let cur_app = cur_dir.join("backplane-gpui.exe");
        std::fs::copy(&me, &guard).unwrap();
        std::fs::copy(&me, &cur_app).unwrap();
        let legacy_icon = pre.join("builds").join("v0.15.4-nightly.20261115.2").join("backplane-gpui.exe");
        let q = |p: &Path| p.display().to_string().replace('\'', "''");
        let mk = format!("$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{}'); $s.TargetPath = '{}'; $s.IconLocation = '{},0'; $s.Save()",
            q(&lnk), q(&guard), q(&legacy_icon));
        assert!(std::process::Command::new("powershell.exe").args(["-NoProfile", "-NonInteractive", "-Command", &mk]).status().unwrap().success());
        assert!(repair_start_menu_shortcut_in(&appdata, &pre, &cur_app));
        let rd = format!("(New-Object -ComObject WScript.Shell).CreateShortcut('{}').IconLocation", q(&lnk));
        let out = std::process::Command::new("powershell.exe").args(["-NoProfile", "-NonInteractive", "-Command", &rd]).output().unwrap();
        let icon_loc = String::from_utf8_lossy(&out.stdout).trim().to_string();
        assert_eq!(icon_loc.to_ascii_lowercase(), format!("{},0", guard.display()).to_ascii_lowercase());
        assert!(!repair_start_menu_shortcut_in(&appdata, &pre, &cur_app), "already migrated shortcut must not re-run repair");
    }
}
