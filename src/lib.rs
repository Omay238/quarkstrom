//! Quarkstrom
//! Quarkstrom = Quark + Maestrom
//!
//! A rendering engine for particle physics engines.

#![warn(missing_docs)]

/// Collection of GUI helpers
pub mod gui;

pub use egui;
pub use wgpu;
pub use winit;
pub use winit_input_helper;

use crate::gui::GuiHandler;

use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use ultraviolet::Vec2;

use wgpu::{
    StoreOp,
    util::DeviceExt
};
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::*,
    event_loop::EventLoopBuilder,
    window::{Window, WindowAttributes},
    application::ApplicationHandler,
    event_loop::ActiveEventLoop,
    window::WindowId
};
use winit_input_helper::WinitInputHelper;

#[repr(C)]
#[derive(Copy, Clone)]
struct View {
    position: Vec2,
    scale: f32,
    x: u16,
    y: u16,
}

unsafe impl Pod for View {}
unsafe impl Zeroable for View {}

#[repr(C)]
#[derive(Clone, Copy)]
struct Rect {
    min: Vec2,
    max: Vec2,
    color: [u8; 4],
}

unsafe impl Pod for Rect {}
unsafe impl Zeroable for Rect {}

impl Rect {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Unorm8x4];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Rect>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    pos: Vec2,
    color: [u8; 4],
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Unorm8x4];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Instance {
    position: Vec2,
    radius: f32,
    color: [u8; 4],
}

unsafe impl Pod for Instance {}
unsafe impl Zeroable for Instance {}

impl Instance {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32, 2 => Unorm8x4];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Instance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

struct State {
    window: Arc<Window>,
    surface: Arc<wgpu::Surface<'static>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    rects: u32,
    rect_buffer: wgpu::Buffer,
    vertices: u32,
    vertex_buffer: wgpu::Buffer,
    instances: u32,
    instance_buffer: wgpu::Buffer,
    rect_render_pipeline: wgpu::RenderPipeline,
    line_render_pipeline: wgpu::RenderPipeline,
    circle_render_pipeline: wgpu::RenderPipeline,
    view: View,
    view_buffer: wgpu::Buffer,
    view_bind_group: wgpu::BindGroup,

    gui: GuiHandler,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
        });

        let surface = Arc::new(instance.create_surface(window.clone()).unwrap());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::default(),
                required_limits: wgpu::Limits::default(),
                label: None,
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code assumes an sRGB surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync, // Could be surface_caps.present_modes[0] but Intel Arc A770 go brrr.
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let gui = GuiHandler::new(window.clone().as_ref(), config.format, &device);

        let circle_shader = device.create_shader_module(wgpu::include_wgsl!("circle_shader.wgsl"));
        let line_shader = device.create_shader_module(wgpu::include_wgsl!("line_shader.wgsl"));
        let rect_shader = device.create_shader_module(wgpu::include_wgsl!("rect_shader.wgsl"));

        let view = View {
            position: Vec2::zero(),
            scale: 1.0,
            x: config.width as u16,
            y: config.height as u16,
        };

        let view_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("View Buffer"),
            contents: bytemuck::cast_slice(&[view]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let view_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("View Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let view_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("View Bind Group"),
            layout: &view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_buffer.as_entire_binding(),
            }],
        });

        let rect_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&view_bind_group_layout],
                push_constant_ranges: &[],
            });

        let rect_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&rect_render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &rect_shader,
                entry_point: Option::from("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Rect::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &rect_shader,
                entry_point: Option::from("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                front_face: wgpu::FrontFace::Ccw,
                conservative: false,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let line_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&view_bind_group_layout],
                push_constant_ranges: &[],
            });

        let line_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&line_render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: Option::from("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: Option::from("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let circle_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&view_bind_group_layout],
                push_constant_ranges: &[],
            });

        let circle_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&circle_render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &circle_shader,
                    entry_point: Option::from("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Instance::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &circle_shader,
                    entry_point: Option::from("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                    polygon_mode: wgpu::PolygonMode::Fill,
                    // Requires Features::DEPTH_CLIP_CONTROL
                    unclipped_depth: false,
                    // Requires Features::CONSERVATIVE_RASTERIZATION
                    conservative: false,
                },
                depth_stencil: None, // 1.
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        let rects = 0;

        let rect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rect Buffer"),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            size: 1 << 28,
            mapped_at_creation: false,
        });

        let vertices = 0;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            size: 1 << 28,
            mapped_at_creation: false,
        });

        let instances = 0;

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            size: 1 << 28,
            mapped_at_creation: false,
        });

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            rects,
            rect_buffer,
            vertices,
            vertex_buffer,
            instances,
            instance_buffer,
            rect_render_pipeline,
            line_render_pipeline,
            circle_render_pipeline,
            view,
            view_buffer,
            view_bind_group,
            gui,
        }
    }

    /// Get the window from the state
    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            self.view.x = new_size.width as u16;
            self.view.y = new_size.height as u16;
        }
    }

    fn set_rects(&mut self, rects: &[Rect]) {
        self.rects = rects.len() as u32;
        self.queue
            .write_buffer(&self.rect_buffer, 0, bytemuck::cast_slice(rects));
    }

    fn set_vertices(&mut self, vertices: &[Vertex]) {
        self.vertices = vertices.len() as u32;
        self.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
    }

    fn set_instances(&mut self, instances: &[Instance]) {
        self.instances = instances.len() as u32;
        self.queue
            .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
    }

    fn input(&mut self, event: &Event<()>) -> bool {
        // If gui doesn't want exclusive access and it's time to update
        !self.gui.handle_event(event, &self.window)
    }

    fn render(&mut self, gui: &mut dyn FnMut(&egui::Context)) -> Result<(), wgpu::SurfaceError> {
        self.queue
            .write_buffer(&self.view_buffer, 0, bytemuck::cast_slice(&[self.view]));

        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let (clipped_primitives, screen_descriptor) =
            self.gui
                .render(&self.device, &self.queue, &self.window, &mut encoder, gui);

        {
            let mut render_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[
                        // This is what @location(0) in the fragment shader targets
                        Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();

            render_pass.set_pipeline(&self.rect_render_pipeline);
            render_pass.set_bind_group(0, &self.view_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.rect_buffer.slice(..));
            render_pass.draw(0..4, 0..self.rects);

            render_pass.set_pipeline(&self.line_render_pipeline);
            render_pass.set_bind_group(0, &self.view_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.vertices, 0..1);

            render_pass.set_pipeline(&self.circle_render_pipeline);
            render_pass.set_bind_group(0, &self.view_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
            render_pass.draw(0..3, 0..self.instances);

            self.gui
                .renderer
                .render(&mut render_pass, &clipped_primitives, &screen_descriptor);
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[derive(Clone, Copy)]
/// An enum specifying the mode for the window
/// Windowed or Fullscreen
/// If windowed, two numbers correspond to the width and the height of the window
pub enum WindowMode {
    Windowed(u32, u32),
    Fullscreen,
}

#[derive(Clone, Copy)]
/// Configuration for a Quarkstrom
pub struct Config {
    /// A WindowMode enum that the window will be created with
    pub window_mode: WindowMode,
}

pub struct RenderContext {
    pos: Vec2,
    scale: f32,
    circles: Vec<Instance>,
    lines: Vec<Vertex>,
    rects: Vec<Rect>,
}

impl RenderContext {
    fn new() -> Self {
        Self {
            pos: Vec2::zero(),
            scale: 1.0,
            circles: Vec::new(),
            lines: Vec::new(),
            rects: Vec::new(),
        }
    }

    pub fn set_view_pos(&mut self, pos: Vec2) {
        self.pos = pos;
    }

    pub fn set_view_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    pub fn clear_rects(&mut self) {
        self.rects.clear();
    }

    pub fn clear_lines(&mut self) {
        self.lines.clear();
    }

    pub fn clear_circles(&mut self) {
        self.circles.clear();
    }

    pub fn draw_circle(&mut self, position: Vec2, radius: f32, color: [u8; 4]) {
        self.circles.push(Instance {
            position,
            radius,
            color,
        });
    }

    pub fn draw_line(&mut self, src: Vec2, dst: Vec2, color: [u8; 4]) {
        self.lines.push(Vertex { pos: src, color });
        self.lines.push(Vertex { pos: dst, color });
    }

    pub fn draw_rect(&mut self, min: Vec2, max: Vec2, color: [u8; 4]) {
        self.rects.push(Rect { min, max, color });
    }
}

pub trait Renderer {
    fn new() -> Self;
    fn input(&mut self, input: &WinitInputHelper, width: u16, height: u16);
    fn render(&mut self, ctx: &mut RenderContext);
    fn gui(&mut self, ctx: &egui::Context);
}

struct AppHandler<R: Renderer> {
    config: Config,
    state: Option<State>,
    input: Option<WinitInputHelper>,
    renderer: Option<R>,
    render_ctx: Option<RenderContext>,
}

impl<R: Renderer> ApplicationHandler<()> for AppHandler<R> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut builder = WindowAttributes::default().with_title("Quarkstrom");

        match self.config.window_mode {
            WindowMode::Windowed(width, height) => {
                //Set window size
                builder = builder.with_inner_size(PhysicalSize::new(width, height));

                //If a primary monitor can be found, position the window in the middle

                if let Some(monitor) = event_loop.primary_monitor() {
                    let size = monitor.size();
                    let position = PhysicalPosition::new(
                        (size.width - width) as i32 / 2,
                        (size.height - height) as i32 / 2,
                    );
                    builder = builder.with_position(position);
                }
            }
            WindowMode::Fullscreen => {
                builder =
                    builder.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
            }
        }

        let window = event_loop
            .create_window(builder)
            .expect("Failed to create window.");

        self.state = Some(pollster::block_on(State::new(Arc::new(window))));
        self.input = Some(WinitInputHelper::new());
        self.renderer = Some(R::new());
        self.render_ctx = Some(RenderContext::new());
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self.state.is_none() {
            return;
        }
        let state = self.state.as_mut().unwrap();
        let input = self.input.as_mut().unwrap();
        let renderer = self.renderer.as_mut().unwrap();
        let mut render_ctx = self.render_ctx.as_mut().unwrap();

        if window_id == state.window().id() {
            let egui_event = Event::WindowEvent {
                window_id,
                event: event.clone(),
            };
            let egui_passed = state.input(&egui_event);

            if egui_passed {
                input.process_window_event(&event);
            }
        }

        if window_id == state.window().id() {
            match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            state: ElementState::Pressed,
                            logical_key:
                                winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
                            ..
                        },
                    ..
                } => return,
                WindowEvent::Resized(physical_size) => {
                    state.resize(physical_size);
                }
                WindowEvent::ScaleFactorChanged { .. } => {
                    state.resize(state.window.inner_size());
                }

                WindowEvent::RedrawRequested if window_id == state.window().id() => {
                    renderer.input(&input, state.view.x, state.view.y);
                    renderer.render(&mut render_ctx);
                    state.view.position = render_ctx.pos;
                    state.view.scale = render_ctx.scale;
                    state.set_instances(&render_ctx.circles);
                    state.set_vertices(&render_ctx.lines);
                    state.set_rects(&render_ctx.rects);

                    match state.render(&mut |ctx| renderer.gui(ctx)) {
                        Ok(_) => {}
                        // Reconfigure the surface if lost
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        // The system is out of memory, we should probably quit
                        Err(wgpu::SurfaceError::OutOfMemory) => return,
                        // All other errors (Outdated, Timeout) should be resolved by the next frame
                        Err(e) => eprintln!("{:?}", e),
                    }
                }
                _ => {}
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if self.state.is_none() {
            return;
        }

        let input = self.input.as_mut().unwrap();

        input.process_device_event(&event);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.state.as_ref().unwrap().window.request_redraw();
    }
}

/// The function to run the project
/// Takes in a Quarkstrom Config
pub fn run<R>(config: Config)
where
    R: Renderer + 'static,
{
    let event_loop = EventLoopBuilder::default().build();

    event_loop
        .unwrap()
        .run_app(&mut AppHandler::<R> {
            config,
            state: None,
            input: None,
            renderer: None,
            render_ctx: None,
        })
        .unwrap();
}
