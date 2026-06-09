//! wgpu/winit render driver (spec §8, §9).
//!
//! Owns the `winit` event loop, the `wgpu` device/surface, the DSP producer, and
//! the spectrogram history texture. Each redraw: drain the analysis ring →
//! produce any new STFT columns → upload them to the history texture at the ring
//! cursor → render one full-screen pass (log-frequency map + colormap).
//!
//! Only compiled with the `gui` feature. It cannot be exercised headlessly, so
//! the testable math it relies on lives in `render::{mapping, colormap, history}`.

use crate::audio::RingSource;
use crate::config::{Colormap, Config};
use crate::dsp::StftProducer;
use crate::render::{colormap, history::HistoryCursor};
use anyhow::{anyhow, Result};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::monitor::MonitorHandle;
use winit::window::{Fullscreen, Window, WindowId};

/// Fixed history width in columns (spec §8: a fixed value such as 2048).
const HISTORY_WIDTH: u32 = 2048;

/// Resolve the `--monitor` request against winit's enumerated outputs, warning
/// and falling back to winit's default (current monitor) on no match. Returns
/// `None` to mean "let winit choose" (e.g. single-output targets like `cage`).
fn select_monitor(event_loop: &ActiveEventLoop, requested: Option<&str>) -> Option<MonitorHandle> {
    let monitors: Vec<MonitorHandle> = event_loop.available_monitors().collect();
    if monitors.is_empty() {
        return None;
    }
    let names: Vec<String> = monitors
        .iter()
        .map(|m| m.name().unwrap_or_default())
        .collect();
    let primary = event_loop
        .primary_monitor()
        .and_then(|p| monitors.iter().position(|m| *m == p))
        .unwrap_or(0);

    match crate::render::monitor::match_monitor_index(&names, primary, requested) {
        Some(idx) => Some(monitors[idx].clone()),
        None => {
            if let Some(req) = requested {
                log::warn!("no monitor matched '{req}'; using the default output");
            }
            None
        }
    }
}

/// Uniforms shared with `spectrogram.wgsl`. Field order/padding must match.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    cursor_offset: f32,
    width: f32,
    num_bins: f32,
    fft_size: f32,
    sample_rate: f32,
    db_floor: f32,
    db_ceiling: f32,
    freq_min: f32,
    freq_max: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// GPU resources, created once the window exists (in `resumed`).
struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surf_config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    history_tex: wgpu::Texture,
    lut_tex: wgpu::Texture,
    uniform_buf: wgpu::Buffer,
    num_bins: u32,
}

impl Gpu {
    async fn new(window: Arc<Window>, params: Params, num_bins: u32) -> Result<Gpu> {
        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow!("no suitable GPU adapter found"))?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("spectro-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        let mut surf_config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| anyhow!("surface not supported by adapter"))?;
        surf_config.present_mode = wgpu::PresentMode::Fifo; // vsync (spec §9)
        surf_config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        surface.configure(&device, &surf_config);

        // --- history texture (R32Float dB), initialized to the dB floor ---
        let history_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("history"),
            size: wgpu::Extent3d {
                width: HISTORY_WIDTH,
                height: num_bins,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let clear = vec![params.db_floor; (HISTORY_WIDTH * num_bins) as usize];
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &history_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&clear),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * HISTORY_WIDTH),
                rows_per_image: Some(num_bins),
            },
            wgpu::Extent3d {
                width: HISTORY_WIDTH,
                height: num_bins,
                depth_or_array_layers: 1,
            },
        );
        let history_view = history_tex.create_view(&Default::default());

        // --- colormap LUT (256x1 Rgba8Unorm) ---
        let lut_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colormap-lut"),
            size: wgpu::Extent3d {
                width: colormap::LUT_LEN as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let lut_view = lut_tex.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lut-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&params));

        // --- pipeline ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spectrogram"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/spectrogram.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spectro-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spectro-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spectro-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("spectro-pipeline"),
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
                    format: surf_config.format,
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

        let gpu = Gpu {
            surface,
            device,
            queue,
            surf_config,
            pipeline,
            bind_group,
            history_tex,
            lut_tex,
            uniform_buf,
            num_bins,
        };
        gpu.upload_lut(Colormap::Inferno); // overwritten by App with the real choice
        Ok(gpu)
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.surf_config.width = width;
            self.surf_config.height = height;
            self.surface.configure(&self.device, &self.surf_config);
        }
    }

    /// Write one dB column into history texture column `x`.
    fn upload_column(&self, x: u32, column: &[f32]) {
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.history_tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(column),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(self.num_bins),
            },
            wgpu::Extent3d {
                width: 1,
                height: self.num_bins,
                depth_or_array_layers: 1,
            },
        );
    }

    fn upload_lut(&self, cm: Colormap) {
        let lut = colormap::lut_rgba8(cm);
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.lut_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &lut,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * colormap::LUT_LEN as u32),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: colormap::LUT_LEN as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    fn set_params(&self, params: &Params) {
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(params));
    }

    fn render(&self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("spectro-pass"),
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
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

/// The winit application: ties the source, DSP, and GPU together.
struct App {
    source: Box<dyn RingSource>,
    producer: StftProducer,
    cursor: HistoryCursor,
    params: Params,
    colormap: Colormap,
    fullscreen: bool,
    /// Raw `--monitor` request; resolved to `monitor` once the event loop runs.
    monitor_request: Option<String>,
    /// Output to target for fullscreen; `None` means winit's current monitor.
    monitor: Option<MonitorHandle>,

    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,

    // Reused scratch (no per-frame allocation in steady state).
    drain_buf: Vec<f32>,
    col_buf: Vec<f32>,
}

impl App {
    fn new(config: Config, source: Box<dyn RingSource>) -> Self {
        let sample_rate = source.sample_rate();
        let fft_size = config.fft_size;
        let hop = config.hop_size();
        let producer = StftProducer::new(fft_size, hop, config.window, sample_rate);
        let num_bins = producer.num_bins();

        let params = Params {
            cursor_offset: 0.0,
            width: HISTORY_WIDTH as f32,
            num_bins: num_bins as f32,
            fft_size: fft_size as f32,
            sample_rate: sample_rate as f32,
            db_floor: config.db_floor,
            db_ceiling: config.db_ceiling,
            freq_min: config.freq_min,
            freq_max: config.effective_freq_max(sample_rate),
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        Self {
            colormap: config.colormap,
            fullscreen: config.fullscreen,
            monitor_request: config.monitor,
            monitor: None,
            cursor: HistoryCursor::new(HISTORY_WIDTH),
            source,
            producer,
            params,
            window: None,
            gpu: None,
            drain_buf: Vec::new(),
            col_buf: Vec::new(),
        }
    }

    /// Drain the ring, produce columns, upload them, and update the cursor.
    fn pump_audio(&mut self) {
        self.drain_buf.clear();
        while let Ok(s) = self.source.consumer().pop() {
            self.drain_buf.push(s);
        }
        if self.drain_buf.is_empty() {
            return;
        }

        self.col_buf.clear();
        let col_buf = &mut self.col_buf;
        self.producer.process(&self.drain_buf, |col| {
            col_buf.extend_from_slice(col);
        });

        let Some(gpu) = self.gpu.as_ref() else {
            return;
        };
        let bins = self.producer.num_bins();
        for column in self.col_buf.chunks_exact(bins) {
            let x = self.cursor.advance();
            gpu.upload_column(x, column);
        }
        self.params.cursor_offset = self.cursor.uv_offset();
        gpu.set_params(&self.params);
    }

    fn toggle_fullscreen(&mut self) {
        self.fullscreen = !self.fullscreen;
        if let Some(w) = &self.window {
            let target = self.monitor.clone();
            w.set_fullscreen(self.fullscreen.then(|| Fullscreen::Borderless(target)));
        }
    }

    fn adjust_db_floor(&mut self, delta: f32) {
        // Keep floor strictly below ceiling.
        self.params.db_floor = (self.params.db_floor + delta).min(self.params.db_ceiling - 1.0);
        if let Some(gpu) = &self.gpu {
            gpu.set_params(&self.params);
        }
        log::info!("db_floor = {:.1}", self.params.db_floor);
    }

    fn cycle_colormap(&mut self) {
        self.colormap = self.colormap.next();
        if let Some(gpu) = &self.gpu {
            gpu.upload_lut(self.colormap);
        }
        log::info!("colormap = {:?}", self.colormap);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return; // already initialized
        }
        self.monitor = select_monitor(event_loop, self.monitor_request.as_deref());
        let mut attrs = Window::default_attributes().with_title("N-Trancerator — spectrogram");
        if self.fullscreen {
            attrs = attrs.with_fullscreen(Some(Fullscreen::Borderless(self.monitor.clone())));
        }
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };
        let num_bins = self.producer.num_bins() as u32;
        match pollster::block_on(Gpu::new(window.clone(), self.params, num_bins)) {
            Ok(gpu) => {
                gpu.upload_lut(self.colormap);
                self.gpu = Some(gpu);
                self.window = Some(window);
            }
            Err(e) => {
                log::error!("GPU initialization failed: {e:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Character(c) => match c.to_ascii_lowercase().as_str() {
                        "q" => event_loop.exit(),
                        "f" => self.toggle_fullscreen(),
                        "c" => self.cycle_colormap(),
                        "[" => self.adjust_db_floor(-5.0),
                        "]" => self.adjust_db_floor(5.0),
                        _ => {}
                    },
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                self.pump_audio();
                if let Some(gpu) = &self.gpu {
                    match gpu.render() {
                        Ok(()) => {}
                        // Surface lost/outdated: reconfigure and try again next frame.
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let (w, h) = (gpu.surf_config.width, gpu.surf_config.height);
                            if let Some(g) = &mut self.gpu {
                                g.resize(w, h);
                            }
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("surface out of memory; exiting");
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
        // Drive continuous redraws (render is vsync-capped by Fifo).
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

/// Run the visualizer event loop with the given source until the user quits.
pub fn run(config: Config, source: Box<dyn RingSource>) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(config, source);
    event_loop.run_app(&mut app)?;
    Ok(())
}
