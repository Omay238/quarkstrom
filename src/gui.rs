use egui::{ClippedPrimitive, Context};
use egui_wgpu::{RendererOptions, ScreenDescriptor};
use wgpu::CommandEncoder;

pub struct GuiHandler {
    ctx: Context,
    pub renderer: egui_wgpu::Renderer,
    state: egui_winit::State,
}

impl GuiHandler {
    pub fn new(
        window: &winit::window::Window,
        format: wgpu::TextureFormat,
        device: &wgpu::Device,
    ) -> Self {
        let ctx = Context::default();
        let state =
            egui_winit::State::new(ctx.clone(), ctx.viewport_id(), &window, None, None, None);

        let renderer = egui_wgpu::Renderer::new(device, format, RendererOptions::default());

        Self {
            ctx,
            renderer,
            state,
        }
    }

    pub fn handle_event(
        &mut self,
        event: &winit::event::Event<()>,
        window: &winit::window::Window,
    ) -> bool {
        match event {
            winit::event::Event::WindowEvent {
                window_id: _,
                event,
            } => self.state.on_window_event(window, event).consumed,
            _ => false,
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window: &winit::window::Window,
        encoder: &mut CommandEncoder,
        gui: &mut dyn FnMut(&Context),
    ) -> (Vec<ClippedPrimitive>, ScreenDescriptor) {
        let raw_input: egui::RawInput = self.state.take_egui_input(window);

        let full_output = self.ctx.run(raw_input, |ctx| {
            gui(ctx);
        });

        self.state
            .handle_platform_output(window, full_output.platform_output);

        let size = window.inner_size();
        let pixels_per_point = full_output.pixels_per_point;
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point,
        };

        let clipped_primitives = self.ctx.tessellate(full_output.shapes, pixels_per_point);

        self.renderer.update_buffers(
            device,
            queue,
            encoder,
            &clipped_primitives,
            &screen_descriptor,
        );
        for (tex_id, img_delta) in full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, tex_id, &img_delta);
        }
        for tex_id in full_output.textures_delta.free {
            self.renderer.free_texture(&tex_id);
        }

        (clipped_primitives, screen_descriptor)
    }
}
