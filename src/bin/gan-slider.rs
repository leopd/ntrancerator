//! GAN Slider — manipulate StyleGAN2 latent space with APC mini mk2 sliders.
//!
//! Reads 8 slider values from the APC, projects them into the GAN's 512-dim
//! z-space via a random linear projection, generates an image, and displays it
//! in a wgpu window.  The 8 buttons above the sliders re-randomize the
//! projection direction for the corresponding slider.
//!
//! FPS is displayed in the window title.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use midir::{MidiInput, MidiInputPort};
use ntrancerator::gan::{GanClient, SliderProjection};

/// APC mini mk2: sliders are CC 48..=55 (we use 8), buttons above are
/// note-on messages for notes 64..=71.
const CC_FIRST: u8 = 48;
const CC_LAST: u8 = 55;
const NUM_SLIDERS: usize = 8;
const BTN_NOTE_FIRST: u8 = 64;
const BTN_NOTE_LAST: u8 = 71;

/// Shared state between the MIDI callback and the render loop.
struct MidiState {
    sliders: [u8; NUM_SLIDERS],
    /// Button press counter per slider — incremented on note-on, the render
    /// loop checks for changes and re-randomizes.
    button_presses: [u64; NUM_SLIDERS],
}

#[derive(Parser)]
#[command(name = "gan-slider")]
struct Cli {
    /// Path to the pygan directory.
    #[arg(long, default_value = "pygan")]
    pygan_dir: PathBuf,

    /// Path to the model .pkl file (relative to pygan_dir or absolute).
    #[arg(long, default_value = "models/metfaces.pkl")]
    model: PathBuf,

    /// Truncation psi.
    #[arg(long, default_value_t = 0.7)]
    trunc: f32,

    /// MIDI port name substring.
    #[arg(long, default_value = "APC mini mk2")]
    port: String,

    /// List MIDI ports and exit.
    #[arg(short, long)]
    list: bool,

    /// RNG seed for initial projection.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Monitor selection (name substring or index).
    #[arg(long)]
    monitor: Option<String>,

    /// Start in fullscreen mode.
    #[arg(long)]
    fullscreen: bool,
}

fn find_port(midi_in: &MidiInput, needle: &str) -> Result<MidiInputPort> {
    let needle_lower = needle.to_lowercase();
    for port in midi_in.ports() {
        let name = midi_in
            .port_name(&port)
            .unwrap_or_else(|_| "(unknown)".into());
        if name.to_lowercase().contains(&needle_lower) {
            return Ok(port);
        }
    }
    bail!(
        "no MIDI port matching {:?}. Run with --list to see available ports.",
        needle
    );
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    let midi_in = MidiInput::new("gan-slider").context("failed to create MIDI input")?;

    if cli.list {
        let ports = midi_in.ports();
        if ports.is_empty() {
            println!("No MIDI input ports found.");
        } else {
            println!("MIDI input ports:");
            for (i, port) in ports.iter().enumerate() {
                let name = midi_in
                    .port_name(port)
                    .unwrap_or_else(|_| "(unknown)".into());
                println!("  {i}: {name}");
            }
        }
        return Ok(());
    }

    // --- MIDI setup ---
    let port = find_port(&midi_in, &cli.port)?;
    let port_name = midi_in
        .port_name(&port)
        .unwrap_or_else(|_| "(unknown)".into());
    println!("MIDI: connected to {port_name}");

    let state = Arc::new(Mutex::new(MidiState {
        sliders: [64; NUM_SLIDERS], // start at midpoint
        button_presses: [0; NUM_SLIDERS],
    }));
    let state_cb = Arc::clone(&state);

    let _conn = midi_in
        .connect(
            &port,
            "gan-slider-read",
            move |_ts, msg, _| {
                if msg.len() < 3 {
                    return;
                }
                let status = msg[0] & 0xF0;
                let data1 = msg[1];
                let data2 = msg[2];

                if let Ok(mut s) = state_cb.lock() {
                    // CC message: slider update
                    if status == 0xB0 && (CC_FIRST..=CC_LAST).contains(&data1) {
                        let idx = (data1 - CC_FIRST) as usize;
                        s.sliders[idx] = data2;
                    }
                    // Note-On message: button press (re-randomize)
                    if status == 0x90
                        && (BTN_NOTE_FIRST..=BTN_NOTE_LAST).contains(&data1)
                        && data2 > 0
                    {
                        let idx = (data1 - BTN_NOTE_FIRST) as usize;
                        if idx < NUM_SLIDERS {
                            s.button_presses[idx] += 1;
                        }
                    }
                }
            },
            (),
        )
        .map_err(|e| anyhow::anyhow!("failed to connect to MIDI port: {e}"))?;

    // --- GAN setup ---
    println!("Starting GAN server...");
    let gan = GanClient::spawn(&cli.pygan_dir, &cli.model, cli.trunc)?;
    let info = gan.info();
    println!(
        "GAN ready: z_dim={}, img={}x{}x{}",
        info.z_dim, info.img_size, info.img_size, info.img_channels
    );

    // --- Projection setup ---
    let projection = SliderProjection::new(info.z_dim as usize, NUM_SLIDERS, cli.seed);
    let last_button_presses = [0u64; NUM_SLIDERS];
    let btn_seed_counter = cli.seed + 1000;

    // --- wgpu/winit setup ---
    use winit::application::ApplicationHandler;
    use winit::event::{ElementState, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
    use winit::keyboard::{Key, NamedKey};
    use winit::window::{Fullscreen, Window, WindowId};

    struct GanGpu {
        surface: wgpu::Surface<'static>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: wgpu::SurfaceConfiguration,
        pipeline: wgpu::RenderPipeline,
        bind_group: wgpu::BindGroup,
        texture: wgpu::Texture,
        img_size: u32,
    }

    impl GanGpu {
        fn new(window: Arc<Window>, img_size: u32) -> Result<Self> {
            let size = window.inner_size();
            let (w, h) = (size.width.max(1), size.height.max(1));

            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN | wgpu::Backends::PRIMARY,
                ..Default::default()
            });
            let surface = instance.create_surface(window.clone())?;
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }))
            .ok_or_else(|| anyhow::anyhow!("no suitable GPU adapter"))?;

            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gan-slider"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            ))?;

            let mut config = surface
                .get_default_config(&adapter, w, h)
                .ok_or_else(|| anyhow::anyhow!("surface not supported"))?;
            config.present_mode = wgpu::PresentMode::Mailbox;
            config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
            surface.configure(&device, &config);

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gan-image"),
                size: wgpu::Extent3d {
                    width: img_size,
                    height: img_size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let tex_view = texture.create_view(&Default::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("gan-sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gan-image-shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../render/shaders/gan_image.wgsl").into(),
                ),
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gan-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gan-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&tex_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gan-layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gan-pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

            Ok(Self {
                surface,
                device,
                queue,
                config,
                pipeline,
                bind_group,
                texture,
                img_size,
            })
        }

        fn resize(&mut self, w: u32, h: u32) {
            if w > 0 && h > 0 {
                self.config.width = w;
                self.config.height = h;
                self.surface.configure(&self.device, &self.config);
            }
        }

        /// Upload RGB pixels (HWC, row-major) and convert to RGBA for the GPU texture.
        fn upload_image(&self, rgb: &[u8]) {
            let n = self.img_size as usize;
            let mut rgba = vec![255u8; n * n * 4];
            for i in 0..(n * n) {
                rgba[i * 4] = rgb[i * 3];
                rgba[i * 4 + 1] = rgb[i * 3 + 1];
                rgba[i * 4 + 2] = rgb[i * 3 + 2];
                // alpha stays 255
            }
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.img_size),
                    rows_per_image: Some(self.img_size),
                },
                wgpu::Extent3d {
                    width: self.img_size,
                    height: self.img_size,
                    depth_or_array_layers: 1,
                },
            );
        }

        fn render(&self) -> Result<(), wgpu::SurfaceError> {
            let frame = self.surface.get_current_texture()?;
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gan-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            self.queue.submit(Some(enc.finish()));
            frame.present();
            Ok(())
        }
    }

    struct App {
        gan: GanClient,
        projection: SliderProjection,
        state: Arc<Mutex<MidiState>>,
        last_button_presses: [u64; NUM_SLIDERS],
        btn_seed_counter: u64,
        fullscreen: bool,
        #[allow(dead_code)]
        monitor_request: Option<String>,

        window: Option<Arc<Window>>,
        gpu: Option<GanGpu>,

        // FPS tracking
        frame_count: u32,
        fps_timer: Instant,
        current_fps: f32,

        // Reusable image buffer
        img_buf: Vec<u8>,
    }

    impl App {
        fn pump_gan(&mut self) {
            // Check for button presses → re-randomize
            if let Ok(s) = self.state.lock() {
                for i in 0..NUM_SLIDERS {
                    if s.button_presses[i] > self.last_button_presses[i] {
                        self.last_button_presses[i] = s.button_presses[i];
                        self.btn_seed_counter += 1;
                        self.projection
                            .rerandomize_slider(i, self.btn_seed_counter);
                        log::info!("Re-randomized slider {i} direction");
                    }
                }
            }

            // Project sliders to z-vector
            let sliders = {
                let s = self.state.lock().unwrap();
                s.sliders
            };
            let z = self.projection.project(&sliders);

            // Generate image
            match self.gan.generate_into(&z, &mut self.img_buf) {
                Ok(()) => {
                    if let Some(gpu) = &self.gpu {
                        gpu.upload_image(&self.img_buf);
                    }
                }
                Err(e) => {
                    log::error!("GAN generation failed: {e:#}");
                }
            }

            // FPS tracking
            self.frame_count += 1;
            let elapsed = self.fps_timer.elapsed().as_secs_f32();
            if elapsed >= 1.0 {
                self.current_fps = self.frame_count as f32 / elapsed;
                self.frame_count = 0;
                self.fps_timer = Instant::now();
                if let Some(w) = &self.window {
                    w.set_title(&format!(
                        "GAN Slider — {:.1} FPS",
                        self.current_fps
                    ));
                }
            }
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.gpu.is_some() {
                return;
            }
            let mut attrs = Window::default_attributes().with_title("GAN Slider — starting...");
            if self.fullscreen {
                attrs = attrs.with_fullscreen(Some(Fullscreen::Borderless(None)));
            }
            let window = match event_loop.create_window(attrs) {
                Ok(w) => Arc::new(w),
                Err(e) => {
                    log::error!("failed to create window: {e}");
                    event_loop.exit();
                    return;
                }
            };

            let img_size = self.gan.info().img_size;
            match GanGpu::new(window.clone(), img_size) {
                Ok(gpu) => {
                    self.gpu = Some(gpu);
                    self.window = Some(window);
                }
                Err(e) => {
                    log::error!("GPU init failed: {e:#}");
                    event_loop.exit();
                }
            }
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => {
                    if let Some(gpu) = &mut self.gpu {
                        gpu.resize(size.width, size.height);
                    }
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed =>
                {
                    match event.logical_key.as_ref() {
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        Key::Character(c) => match c.to_ascii_lowercase().as_str() {
                            "q" => event_loop.exit(),
                            "f" => {
                                self.fullscreen = !self.fullscreen;
                                if let Some(w) = &self.window {
                                    w.set_fullscreen(
                                        self.fullscreen
                                            .then(|| Fullscreen::Borderless(None)),
                                    );
                                }
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
                WindowEvent::RedrawRequested => {
                    self.pump_gan();
                    if let Some(gpu) = &self.gpu {
                        match gpu.render() {
                            Ok(()) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                if let Some(g) = &mut self.gpu {
                                    let (w, h) = (g.config.width, g.config.height);
                                    g.resize(w, h);
                                }
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("out of memory");
                                event_loop.exit();
                            }
                            Err(e) => log::warn!("render error: {e}"),
                        }
                    }
                }
                _ => {}
            }
        }

        fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
    }

    let img_bytes = info.image_bytes();
    let mut app = App {
        gan,
        projection,
        state,
        last_button_presses,
        btn_seed_counter,
        fullscreen: cli.fullscreen,
        monitor_request: cli.monitor,
        window: None,
        gpu: None,
        frame_count: 0,
        fps_timer: Instant::now(),
        current_fps: 0.0,
        img_buf: vec![0u8; img_bytes],
    };

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app)?;

    Ok(())
}
