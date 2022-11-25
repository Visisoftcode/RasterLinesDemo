/* License: MIT */

use sdl2;
use sdl2::rect::Rect;
use sdl2::pixels::PixelFormatEnum;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use std::time::{Duration,Instant};
use std::thread::sleep;

use ndarray as nd;

use core::f32::consts::PI;

use nalgebra as na;
use na::clamp;

// Our canvas data type
type ImageRGB = nd::Array3<u8>; // [channels x height x width]

type Vec2f = na::SVector<f32,2>;
type Vec3f = na::SVector<f32,3>;



const FPS: f32 = 60.0;

const SCALE: usize = 1; // scale of pixels

const WIDTH: usize = 1280/SCALE;
const HEIGHT: usize = 720/SCALE;

const TITLE: &str = "generativeart"; // title of the window



struct LineIterator {
	start: isize, // start of main coordinate
	end: isize, // end of main coordinate
	
	other_start: f32, // start of opposite coordinate
	factor: f32, // multiplier for opposite coordinate
	
	i: isize, // state of the iterator
}

impl LineIterator {
	fn new(start: isize,end: isize,other_start: f32,factor: f32)->Self {
		Self {
			start,
			end,
			other_start,
			factor,
			i: 0,
		}
	}
}

impl Iterator for LineIterator {
	type Item = (isize,f32);
	
	fn next(&mut self)->Option<Self::Item> {
		let end_bigger = self.end > self.start;
		
		let iternum = if end_bigger {self.end - self.start} else {self.start - self.end}; // + 1 may be added, but there will be some artifacts
		
		if self.i == iternum {
			return None;
		}
		
		let iterdel = std::cmp::max(iternum,1) as f32;
		
		let fcc = self.other_start + self.factor*((self.i as f32 + 0.5)/iterdel);
		
		let c = self.start + if end_bigger {self.i} else {-self.i};
		
		self.i += 1;
		
		Some((c,fcc))
	}
}



fn antialiasing_values(ant_coord: f32)->(isize,isize,f32) {
	let rem = ant_coord%1.0;
	
	if rem >= 0.5 {
		(ant_coord as isize,
		 (ant_coord + 1.0) as isize,
		 1.0 - (rem - 0.5))
	}
	else {
		((ant_coord - 1.0) as isize,
		 (ant_coord) as isize,
		 0.5 - rem)
	}
}



fn alpha_blend(src: &[u8; 3],add: &[u8; 3],alpha: f32)->[u8; 3] {
	[clamp((src[0] as f32*(1.0 - alpha) + add[0] as f32*alpha) as isize,0,255) as u8,
	 clamp((src[1] as f32*(1.0 - alpha) + add[1] as f32*alpha) as isize,0,255) as u8,
	 clamp((src[2] as f32*(1.0 - alpha) + add[2] as f32*alpha) as isize,0,255) as u8]
}



struct Canvas {
	data: ImageRGB,
	smooth: bool, // to render smooth lines
}

impl Canvas {
	fn new(width: usize,height: usize,smooth: bool)->Self {
		Self {
			data: ImageRGB::zeros((3,height,width)),
			smooth,
		}
	}
	
	fn clear(&mut self,color: &Vec3f) { // clear with a color
		let dims = self.data.dim();
		
		for y in 0..dims.1 {
			for x in 0..dims.2 {
				self.data[(0,y,x)] = (color.x*255.0) as u8;
				self.data[(1,y,x)] = (color.y*255.0) as u8;
				self.data[(2,y,x)] = (color.z*255.0) as u8;
			}
		}
	}
	
	fn set_pixel(&mut self,x : isize,y: isize,color: [u8; 3]) {
		let dims = self.data.dim();
		
		if x >= 0 && x < dims.2 as isize && y >= 0 && y < dims.1 as isize {
			let x = x as usize;
			let y = y as usize;
			
			self.data[(0,y,x)] = color[0];
			self.data[(1,y,x)] = color[1];
			self.data[(2,y,x)] = color[2];
		}
	}
	
	fn get_pixel(&self,x: isize,y: isize)->[u8; 3] {
		let dims = self.data.dim();
		
		if x >= 0 && x < dims.2 as isize && y >= 0 && y < dims.1 as isize {
			let x = x as usize;
			let y = y as usize;
			
			[self.data[(0,y,x)],self.data[(1,y,x)],self.data[(2,y,x)]]
		}
		else {
			[0,0,0]
		}
	}
	
	// draw line using two points: points[0] and points[1]
	fn line(&mut self,points: &[Vec2f],color: &[u8; 3]) {
		let p0 = points[0].component_mul(&Vec2f::new(WIDTH as f32,HEIGHT as f32));
		let p1 = points[1].component_mul(&Vec2f::new(WIDTH as f32,HEIGHT as f32));
		
		let d = p1 - p0;
		
		if d.x.abs() > d.y.abs() { // iterating over x
			let li = LineIterator::new(p0.x as isize,p1.x as isize,p0.y,d.y);
			
			for (x,fy) in li {
				let y = fy as isize;
				
				if !self.smooth {
					self.set_pixel(x,y,*color);
				}
				else {
					let (by,ty,bvalue) = antialiasing_values(fy);
					
					let bcolor = alpha_blend(
						&self.get_pixel(x,by),
						color,
						bvalue
					);
					self.set_pixel(x,by,bcolor);
					
					let tcolor = alpha_blend(
						&self.get_pixel(x,ty),
						color,
						1.0 - bvalue
					);
					self.set_pixel(x,ty,tcolor);
				}
			}
		}
		else { // iterating over y
			let li = LineIterator::new(p0.y as isize,p1.y as isize,p0.x,d.x);
			
			for (y,fx) in li {
				let x = fx as isize;
				
				if !self.smooth {
					self.set_pixel(x,y,*color);
				}
				else {
					let (lx,rx,lvalue) = antialiasing_values(fx);
					
					let lcolor = alpha_blend(
						&self.get_pixel(lx,y),
						color,
						lvalue
					);
					self.set_pixel(lx,y,lcolor);
					
					let rcolor = alpha_blend(
						&self.get_pixel(rx,y),
						color,
						1.0 - lvalue
					);
					self.set_pixel(rx,y,rcolor);
				}
			}
		}
	}
	
	// draw line strip defined by points
	fn line_strip(&mut self,points: &[Vec2f],color: &[u8; 3]) {
		for i in 0..(points.len() - 1) {
			self.line(&points[i..],color);
		}
	}
}



// generates smoother line using Chaikin's algorithm
fn chaikins_line(input: &[Vec2f])->Vec<Vec2f> {
	input.iter()
		.take(input.len() - 1)
		.zip(
			input.iter()
				.skip(1)
		)
		.map(|(p0,p1)| {
			let dir = p1 - p0;
			let np0 = p0 + dir*0.25;
			let np1 = p0 + dir*0.75;
			[np0,np1]
		})
		.flatten()
		.collect()
}

// generates smooth line using chaikins_line function
fn chaikins_smooth(input: &[Vec2f],num: usize)->Vec<Vec2f> {
	if num == 0 {panic!("Error, num must be more than zero")}
	
	let value = chaikins_line(input);
	let value = (1..num).fold(value,|acc, _| chaikins_line(acc.as_slice()));
	
	let p0 = input[0];
	let p1 = input[input.len() - 1];
	
	std::iter::once(p0).chain(
		value.into_iter().chain(
			std::iter::once(p1)
		)
	).collect()
}



// argument multiplier, phase, value multiplier

const COEFFICIENTS_RED: &[[f32; 3]] = &[
	[1.0,  0.00, 1.0],
	[5.0,  0.05, 0.5],
	[8.0,  0.00, 0.2],
	[1.0,  0.01, 1.0],
	[0.5,  0.30, 2.5],
	[15.0,  0.20, 0.7],
];

const COEFFICIENTS_GREEN: &[[f32; 3]] = &[
	[0.5,  1.00,-1.2],
	[5.0,  0.00, 0.07],
	[5.5,  0.00, 0.08],
	[3.2,  0.70, 0.05],
];



// this is a sum of sin(x*coef[0] + coef[1])*coef[2]
// x may be from 0.0 to 1.0, internally it is multiplied by 2 pi
fn complex_function(in_x: f32, coefs: &[[f32; 3]])->f32 {
	let x = in_x*2.*PI;
	
	coefs.iter()
		.map(|coef| {
			(coef[0]*x + coef[1]).sin()*coef[2]
		})
		.fold(0., |acc, value| acc + value)
}


// not pretty function to generate a red strip
fn gen_red_strip(j: usize, time: f32)->Vec<Vec2f> {
	let coefs: Vec<[f32; 3]> = COEFFICIENTS_RED.iter()
		.enumerate()
		.map(|(ci,coef)| {
			let timefactor = j as f32/8. + ci as f32/12.;
			[coef[0],coef[1] + ci as f32/(8.*(1.3 + (time*timefactor).sin())) + j as f32/14.,coef[2]*(0.5 + timefactor/2.*(ci as f32/5.))]
		})
		.collect();
	
	let strip: Vec<Vec2f> = (0..=50)
		.map(|i| {
			let t = i as f32/50.;
			Vec2f::new(t,0.5 + complex_function(t,coefs.as_slice())/24.)
		})
		.collect();
	
	strip
}


// not pretty function to generate a green strip
fn gen_green_strip(j: usize,time: f32)->Vec<Vec2f> {
	let coefs: Vec<[f32; 3]> = COEFFICIENTS_GREEN.iter()
		.enumerate()
			.map(|(ci,coef)| {
				let timefactor = j as f32/12. + ci as f32/16.;
				[coef[0],coef[1] + ci as f32/(8.*(1.3 + (time*timefactor).sin())) + j as f32/17.,coef[2]*(0.5 + timefactor/2.*(ci as f32/7.))]
			})
			.collect();
	
	let strip: Vec<Vec2f> = (0..=50)
		.map(|i| {
			let t = i as f32/50.;
			Vec2f::new(t,0.5 + complex_function(t,coefs.as_slice())/2.)
		})
		.collect();
	
	strip
}


// function to draw to out canvas
fn draw(canvas: &mut Canvas,time: f32) {
	for j in 0..15 {
		let strip = gen_red_strip(j,time);
		let strip = chaikins_smooth(strip.as_slice(),4);
		canvas.line_strip(strip.as_slice(),&[255,0,255]);
	}
	
	for j in 0..12 {
		let strip = gen_green_strip(j,time);
		let strip = chaikins_smooth(strip.as_slice(),4);
		canvas.line_strip(strip.as_slice(),&[15,255,15]);
	}
}



// main
fn main() {
	let sdl_context = sdl2::init().unwrap();
	let video_subsystem = sdl_context.video().unwrap();
	
	let window = video_subsystem.window(TITLE,(WIDTH*SCALE) as u32,(HEIGHT*SCALE) as u32)
		.position_centered()
		.build()
		.unwrap();
	
	let mut canvas = window.into_canvas().build().unwrap();
	
	let texture_creator = canvas.texture_creator();
	let mut texture = texture_creator.create_texture_streaming(PixelFormatEnum::RGB24,(WIDTH*SCALE) as u32,(HEIGHT*SCALE) as u32).unwrap();
	
	
	
	let mut plane = Canvas::new(WIDTH,HEIGHT,true); // generate out Canvas
	
	let background = Vec3f::new(0.05,0.15,0.2); // clear color
	
	
	
	let start = Instant::now(); // start time of simulation
	
	'mainloop: loop {
		let now = Instant::now();
		
		let mut event_pump = sdl_context.event_pump().unwrap();
		for event in event_pump.poll_iter() { // process events
			match event {
				Event::Quit {..} |
				Event::KeyDown {keycode: Some(Keycode::Escape), .. } => {
					break 'mainloop;
				},
				_ => {},
			}
		}
		
		plane.clear(&background); // clear plane
		draw(&mut plane,start.elapsed().as_secs_f32()); // draw to plane
		
		let result = texture.with_lock(None, |buffer: &mut [u8], pitch: usize| { // copy data from out plane to SDL2
			for y in 0..HEIGHT {
				for x in 0..WIDTH {
					for sy in 0..SCALE {
						for sx in 0..SCALE {
							let lx = x*SCALE + sx;
							let ly = (HEIGHT - y - 1)*SCALE + sy;
							
							let offset = ly*pitch + lx*3;
							
							buffer[offset + 0] = plane.data[[0,y,x]];
							buffer[offset + 1] = plane.data[[1,y,x]];
							buffer[offset + 2] = plane.data[[2,y,x]];
						}
					}
				}
			}
		});
		if let Err(e) = result {
			println!("{}",e);
		}
		
		let result = canvas.copy(&texture,None,Some(Rect::new(0,0,(WIDTH*SCALE) as u32,(HEIGHT*SCALE) as u32))); // copy to SDL2 canvas
		if let Err(e) = result {
			println!("{}",e);
		}
		
		canvas.present(); // present on window
		
		let time = now.elapsed();
		println!("elapsed: {}, max FPS: {}",time.as_secs_f32(),1./time.as_secs_f32());
		if time < Duration::from_secs_f32(1.0/FPS) {
			sleep(Duration::from_secs_f32(1.0/FPS) - time); // sleep such time, so with good performance we will get FPS frames per second
		}
	}
}
