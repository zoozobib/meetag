use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    AppHandle, Emitter, Runtime,
};

pub fn create_tray<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<()> {
    let open_i = MenuItem::with_id(app, "open", "Open Window", true, None::<&str>)?;
    let start_i = MenuItem::with_id(app, "start", "Start Recording", true, None::<&str>)?;
    let stop_i = MenuItem::with_id(app, "stop", "Stop", true, None::<&str>)?;
    let quit_i = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

    let menu = Menu::with_items(app, &[&open_i, &start_i, &stop_i, &quit_i])?;

    let builder = TrayIconBuilder::with_id("main")
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(move |app, event| {
            let id = event.id.as_ref();
            match id {
                "open" => {
                    let _ = app.emit("tray-open-window", ());
                }
                "start" => {
                    let _ = app.emit("tray-record-start", ());
                }
                "stop" => {
                    let _ = app.emit("tray-record-stop", ());
                }
                "quit" => {
                    app.exit(0);
                }
                _ => {}
            }
        });

    // Use default app icon if available
    let builder = if let Some(icon) = app.default_window_icon() {
        builder.icon(icon.clone())
    } else {
        // Fallback or warning
        println!("⚠️ No default icon found for tray!");
        builder
    };

    builder.build(app)?;

    Ok(())
}
