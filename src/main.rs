use anyhow::{Context as _, Result};
use cairo::{Context as CairoContext, Format, ImageSurface};
use log::warn;
use pango::FontDescription;
use std::env;
use std::num::NonZeroU32;
use std::os::fd::AsRawFd;
use std::process::Command;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use winit::event_loop::OwnedDisplayHandle;
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalPosition, LogicalSize, PhysicalPosition, PhysicalSize},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    window::{Window, WindowAttributes, WindowId},
};
use xbar_core::{
    BarEffect, BarRuntime, ModelConfig, RuntimeUpdate, SharedEventNotifier, SharedTransport,
    logging::init as initialize_logging,
    presentation::{Point, PointerAction, PresentationConfig, Size},
    render::cairo::CairoBar,
};

const TRANSPORT_RETRY_INTERVAL: Duration = Duration::from_secs(2);

type SoftSurface = softbuffer::Surface<OwnedDisplayHandle, Rc<Window>>;

#[derive(Debug, Clone)]
enum UserEvent {
    Tick,
    SharedUpdated(Arc<AtomicBool>),
}

struct EventForwarder {
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Drop for EventForwarder {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take()
            && let Err(payload) = worker.join()
        {
            warn!("event forwarding thread panicked: {payload:?}");
        }
    }
}

fn spawn_tick_thread(proxy: EventLoopProxy<UserEvent>) -> EventForwarder {
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = Arc::clone(&stop);
    let worker = thread::spawn(move || {
        while !worker_stop.load(Ordering::Acquire) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_else(|_| Duration::from_secs(0));
            let subns = u64::from(now.subsec_nanos());
            thread::sleep(Duration::from_nanos(
                1_000_000_000_u64.saturating_sub(subns).max(1),
            ));
            if worker_stop.load(Ordering::Acquire) || proxy.send_event(UserEvent::Tick).is_err() {
                break;
            }
        }
    });
    EventForwarder {
        stop,
        worker: Some(worker),
    }
}

fn spawn_shared_thread(
    proxy: EventLoopProxy<UserEvent>,
    notifier: Option<SharedEventNotifier>,
) -> Option<EventForwarder> {
    notifier.map(|notifier| {
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        // The event-loop handler clears this only after it has drained the
        // transport, so at most one shared update can be queued at a time.
        let worker_pending = Arc::new(AtomicBool::new(false));
        let worker = thread::spawn(move || {
            let mut descriptor = libc::pollfd {
                fd: notifier.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            while !worker_stop.load(Ordering::Acquire) {
                descriptor.revents = 0;
                let ready = unsafe { libc::poll(&mut descriptor, 1, 250) };
                if ready < 0 {
                    let error = std::io::Error::last_os_error();
                    if error.raw_os_error() == Some(libc::EINTR) {
                        continue;
                    }
                    warn!("shared notifier poll failed: {error}");
                    break;
                }
                if ready == 0 {
                    continue;
                }
                if descriptor.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                    warn!("shared notifier fd became unusable: {}", descriptor.revents);
                    break;
                }
                if descriptor.revents & libc::POLLIN != 0 {
                    match notifier.drain() {
                        Ok(0) => {}
                        Ok(_) => {
                            if worker_pending
                                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                                .is_ok()
                            {
                                let event = UserEvent::SharedUpdated(Arc::clone(&worker_pending));
                                if proxy.send_event(event).is_err() {
                                    worker_pending.store(false, Ordering::Release);
                                    break;
                                }
                            }
                            while worker_pending.load(Ordering::Acquire)
                                && !worker_stop.load(Ordering::Acquire)
                            {
                                thread::sleep(Duration::from_millis(10));
                            }
                        }
                        Err(error) => {
                            warn!("shared notifier drain failed: {error}");
                            break;
                        }
                    }
                }
            }
        });
        EventForwarder {
            stop,
            worker: Some(worker),
        }
    })
}

struct CairoBackBuffer {
    width: u32,
    height: u32,
    image: ImageSurface,
}

impl CairoBackBuffer {
    fn new(width: u32, height: u32) -> Result<Self> {
        let image = ImageSurface::create(
            Format::ARgb32,
            i32::try_from(width).context("buffer width does not fit Cairo")?,
            i32::try_from(height).context("buffer height does not fit Cairo")?,
        )?;
        Ok(Self {
            width,
            height,
            image,
        })
    }

    fn ensure_size(&mut self, width: u32, height: u32) -> Result<()> {
        if self.width == width && self.height == height {
            return Ok(());
        }
        *self = Self::new(width, height)?;
        Ok(())
    }
}

struct App {
    window: Option<Rc<Window>>,
    window_id: Option<WindowId>,
    bar: CairoBar,
    scale_factor: f64,
    logical_size: LogicalSize<f64>,
    default_logical_size: LogicalSize<f64>,
    last_physical_size: PhysicalSize<u32>,
    last_cursor_pos: Option<Point>,
    back: Option<CairoBackBuffer>,
    soft_surface: Option<SoftSurface>,
    shared_path: String,
    last_transport_attempt: Instant,
}

impl App {
    fn new(
        bar: CairoBar,
        logical_size: LogicalSize<f64>,
        scale_factor: f64,
        shared_path: String,
    ) -> Self {
        Self {
            window: None,
            window_id: None,
            bar,
            scale_factor,
            logical_size,
            default_logical_size: logical_size,
            last_physical_size: PhysicalSize::new(
                logical_size.width.round() as u32,
                logical_size.height.round() as u32,
            ),
            last_cursor_pos: None,
            back: None,
            soft_surface: None,
            shared_path,
            last_transport_attempt: Instant::now(),
        }
    }

    fn redraw(&mut self) -> Result<()> {
        let width = self.last_physical_size.width;
        let height = self.last_physical_size.height;
        if self.window.is_none() || width == 0 || height == 0 {
            return Ok(());
        }

        let back = match self.back.as_mut() {
            Some(back) => {
                back.ensure_size(width, height)?;
                back
            }
            None => self.back.insert(CairoBackBuffer::new(width, height)?),
        };
        {
            let context = CairoContext::new(&back.image)?;
            context.scale(self.scale_factor, self.scale_factor);
            self.bar.render(
                &context,
                Size::new(
                    self.logical_size.width as f32,
                    self.logical_size.height as f32,
                ),
            )?;
        }
        back.image.flush();

        let stride = usize::try_from(back.image.stride())?;
        let data = back.image.data()?;
        let width_usize = width as usize;
        let height_usize = height as usize;
        let surface = match self.soft_surface.as_mut() {
            Some(surface) => surface,
            None => return Ok(()),
        };
        let mut target = surface
            .buffer_mut()
            .map_err(|error| anyhow::anyhow!("softbuffer buffer acquisition failed: {error}"))?;
        if target.len() < width_usize * height_usize {
            anyhow::bail!("softbuffer returned an undersized frame");
        }

        if stride == width_usize * 4 {
            let source: &[u32] = bytemuck::cast_slice(&data[..height_usize * stride]);
            target[..width_usize * height_usize].copy_from_slice(source);
        } else {
            for row in 0..height_usize {
                let start = row * stride;
                let source: &[u32] = bytemuck::cast_slice(&data[start..start + width_usize * 4]);
                target[row * width_usize..(row + 1) * width_usize].copy_from_slice(source);
            }
        }
        target
            .present()
            .map_err(|error| anyhow::anyhow!("softbuffer present failed: {error}"))?;
        Ok(())
    }

    fn request_redraw(&self) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn resize_surface(&mut self, size: PhysicalSize<u32>) {
        self.last_physical_size = size;
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.logical_size = size.to_logical(self.scale_factor);
        if let Some(surface) = self.soft_surface.as_mut()
            && let (Some(width), Some(height)) =
                (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
            && let Err(error) = surface.resize(width, height)
        {
            warn!("softbuffer resize failed: {error}");
        }
    }

    fn handle_pointer_action(&mut self, point: Point, action: PointerAction) {
        let update = self.bar.pointer_action(point, action);
        self.handle_runtime_update(update);
    }

    fn handle_runtime_update(&mut self, update: RuntimeUpdate) {
        let RuntimeUpdate {
            changes,
            platform_effects,
            issues,
        } = update;
        for issue in issues {
            warn!("xbar runtime issue: {issue:?}");
        }
        for effect in platform_effects {
            self.handle_platform_effect(effect);
        }
        if !changes.is_empty() {
            self.request_redraw();
        }
    }

    fn tick_and_poll(&mut self) {
        if !self.shared_path.is_empty()
            && self.bar.runtime().transport().is_none()
            && self.last_transport_attempt.elapsed() >= TRANSPORT_RETRY_INTERVAL
        {
            self.last_transport_attempt = Instant::now();
            match SharedTransport::open(&self.shared_path) {
                Ok(transport) => {
                    self.bar.runtime_mut().set_transport(Some(transport));
                    log::debug!("reconnected WM transport at {}", self.shared_path);
                }
                Err(error) => log::debug!("WM transport is still unavailable: {error}"),
            }
        }

        let mut update = self.bar.tick();
        update.merge(self.bar.poll_transport());
        self.handle_runtime_update(update);
    }

    fn handle_platform_effect(&mut self, effect: BarEffect) {
        match effect {
            BarEffect::ApplyMonitorGeometry(geometry) => self.apply_monitor_geometry(geometry),
            BarEffect::ClearMonitorGeometry => {
                if let Some(window) = &self.window {
                    window.set_outer_position(LogicalPosition::new(0.0, 0.0));
                    let _ = window.request_inner_size(self.default_logical_size);
                }
            }
            BarEffect::Screenshot => spawn_program("flameshot", &["gui"]),
            BarEffect::OpenAudioControl => spawn_program("pavucontrol", &[]),
            BarEffect::WindowManager(_)
            | BarEffect::ToggleMute
            | BarEffect::AdjustVolume(_)
            | BarEffect::AdjustBrightness(_)
            | BarEffect::RefreshBattery => {
                warn!("no frontend adapter handled platform effect: {effect:?}");
            }
        }
    }

    fn apply_monitor_geometry(&self, geometry: xbar_core::MonitorGeometry) {
        if let Some(window) = &self.window {
            let height = (f64::from(self.bar.config().bar_height) * self.scale_factor)
                .round()
                .clamp(1.0, f64::from(u32::MAX)) as u32;
            window.set_outer_position(PhysicalPosition::new(geometry.x, geometry.y));
            let _ = window.request_inner_size(PhysicalSize::new(geometry.width, height));
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let primary = event_loop
            .primary_monitor()
            .or_else(|| event_loop.available_monitors().next());
        self.scale_factor = primary
            .as_ref()
            .map_or(1.0, |monitor| monitor.scale_factor());
        let screen_size = primary
            .as_ref()
            .map_or(PhysicalSize::new(1920, 1080), |monitor| monitor.size());
        self.logical_size = LogicalSize::new(
            f64::from(screen_size.width) / self.scale_factor,
            f64::from(self.bar.config().bar_height),
        );
        self.default_logical_size = self.logical_size;

        let attributes = WindowAttributes::default()
            .with_title("winit_softbuffer_bar")
            .with_inner_size(self.logical_size)
            .with_decorations(false)
            .with_resizable(true)
            .with_visible(true)
            .with_transparent(false);
        let window = Rc::new(
            event_loop
                .create_window(attributes)
                .expect("create_window failed"),
        );
        let context = softbuffer::Context::new(event_loop.owned_display_handle())
            .map_err(|error| anyhow::anyhow!("softbuffer context initialization failed: {error}"))
            .expect("softbuffer context");
        let mut surface = SoftSurface::new(&context, Rc::clone(&window))
            .map_err(|error| anyhow::anyhow!("softbuffer surface initialization failed: {error}"))
            .expect("softbuffer surface");
        let size = window.inner_size();
        if let (Some(width), Some(height)) =
            (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        {
            surface
                .resize(width, height)
                .expect("initial softbuffer resize failed");
        }

        self.window_id = Some(window.id());
        self.window = Some(window);
        self.last_physical_size = size;
        self.back = Some(
            CairoBackBuffer::new(size.width.max(1), size.height.max(1))
                .expect("Cairo back buffer creation failed"),
        );
        self.soft_surface = Some(surface);

        let tick = self.bar.tick();
        self.handle_runtime_update(tick);
        let shared = self.bar.poll_transport();
        self.handle_runtime_update(shared);
        self.request_redraw();
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Tick => self.tick_and_poll(),
            UserEvent::SharedUpdated(pending) => {
                let update = self.bar.poll_transport();
                self.handle_runtime_update(update);
                pending.store(false, Ordering::Release);
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.scale_factor = self
                    .window
                    .as_ref()
                    .map_or(self.scale_factor, |window| window.scale_factor());
                self.resize_surface(size);
                self.request_redraw();
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor = scale_factor;
                if let Some(window) = &self.window {
                    self.resize_surface(window.inner_size());
                }
                if let Some(geometry) = self.bar.runtime().view().geometry {
                    self.apply_monitor_geometry(geometry);
                }
                self.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = position.to_logical::<f64>(self.scale_factor);
                let point = Point::new(position.x as f32, position.y as f32);
                self.last_cursor_pos = Some(point);
                if self.bar.pointer_motion(point) {
                    self.request_redraw();
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.last_cursor_pos = None;
                if self.bar.pointer_leave() {
                    self.request_redraw();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                use winit::event::{ElementState, MouseButton};
                if state == ElementState::Pressed
                    && let Some(point) = self.last_cursor_pos
                {
                    let action = match button {
                        MouseButton::Left => Some(PointerAction::Primary),
                        MouseButton::Right => Some(PointerAction::Secondary),
                        MouseButton::Middle
                        | MouseButton::Back
                        | MouseButton::Forward
                        | MouseButton::Other(_) => None,
                    };
                    if let Some(action) = action {
                        self.handle_pointer_action(point, action);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                use winit::event::MouseScrollDelta;
                if let Some(point) = self.last_cursor_pos {
                    let vertical = match delta {
                        MouseScrollDelta::LineDelta(_, value) => f64::from(value),
                        MouseScrollDelta::PixelDelta(position) => position.y,
                    };
                    let action = if vertical > 0.0 {
                        Some(PointerAction::ScrollUp)
                    } else if vertical < 0.0 {
                        Some(PointerAction::ScrollDown)
                    } else {
                        None
                    };
                    if let Some(action) = action {
                        self.handle_pointer_action(point, action);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.redraw() {
                    warn!("redraw failed: {error}");
                }
            }
            _ => {}
        }
    }
}

fn spawn_program(program: &str, args: &[&str]) {
    let program = program.to_owned();
    let args = args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>();
    thread::spawn(move || {
        if let Err(error) = Command::new(&program).args(&args).status() {
            warn!("failed to run {program}: {error}");
        }
    });
}

fn main() -> Result<()> {
    let shared_path = env::args().skip(1).last().unwrap_or_default();
    initialize_logging("winit_softbuffer_bar", &shared_path)?;

    let transport = if shared_path.is_empty() {
        None
    } else {
        Some(
            SharedTransport::open(&shared_path)
                .with_context(|| format!("failed to open shared transport {shared_path}"))?,
        )
    };
    let notifier = transport
        .as_ref()
        .map(|transport| transport.notifier(true))
        .transpose()
        .context("failed to start shared transport notifier")?;
    let runtime = BarRuntime::with_transport(ModelConfig::default(), transport)?;
    let presentation = PresentationConfig {
        bar_height: 38.0,
        ..PresentationConfig::default()
    };
    let font = env::var("XBAR_FONT").unwrap_or_else(|_| "monospace 11".to_owned());
    let bar = CairoBar::new(runtime, presentation, FontDescription::from_string(&font));

    let event_loop: EventLoop<UserEvent> = EventLoop::with_user_event().build()?;
    let proxy = event_loop.create_proxy();
    let _tick_forwarder = spawn_tick_thread(proxy.clone());
    let _shared_forwarder = spawn_shared_thread(proxy, notifier);

    let mut app = App::new(bar, LogicalSize::new(800.0, 38.0), 1.0, shared_path);
    event_loop.run_app(&mut app)?;
    Ok(())
}
