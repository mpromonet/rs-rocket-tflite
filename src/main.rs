/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
** -------------------------------------------------------------------------*/

use std::env::args;
use std::io::Cursor;

use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, InterpreterBuilder};
use image::io::Reader;

use serde::Serialize;

use actix_web::{get, post, web, App, HttpServer, Responder};
use actix_files as fs;

struct AppState {
        model: String,
}

#[derive(Serialize)]
struct Point {
    X: i32,
    Y: i32,
}

#[derive(Serialize)]
struct Rectangle {
	Min       : Point,
	Max       : Point,
}

#[derive(Serialize)]
struct Item {
	Box       : Rectangle,
	Score     : f32,
	ClassID   : i32,
	ClassName : String,
}

#[get("/models")]
async fn models(data: web::Data<AppState>) -> impl Responder {
    let filename = data.model.as_str();
    format!("[\"{}\"]",filename)
}

#[post("/invoke/{model}")]
async fn invoke(data: web::Data<AppState>, body: web::Bytes) -> impl Responder {
    let reader = Reader::new(Cursor::new(body)).with_guessed_format().unwrap();

    let mut img = reader.decode().expect("Failed to read image");
    println!("image size: {}x{}", img.width(), img.height());

    let filename = data.model.as_str();
    let model = FlatBufferModel::build_from_file(filename).unwrap();
    let resolver = BuiltinOpResolver::default();

    let builder = InterpreterBuilder::new(&model, &resolver).unwrap();
    let mut interpreter = builder.build().unwrap();
    interpreter.allocate_tensors().unwrap();

    let inputs = interpreter.inputs().to_vec();
    let input_index = inputs[0];

    let info = interpreter.tensor_info(input_index).unwrap();
    println!("tensor size: {:?}", info.dims);
    if info.dims[3] == 1 {
        img = img.grayscale();
    }
    let resized_img = img.resize_exact(info.dims[1].try_into().unwrap(), info.dims[2].try_into().unwrap(), image::imageops::FilterType::Nearest);
    interpreter.tensor_data_mut(input_index).unwrap().copy_from_slice(&resized_img.into_bytes());

    interpreter.invoke().unwrap();

    let outputs = interpreter.outputs().to_vec();
    let output_index = outputs[0];
    let _output: &[u8] = interpreter.tensor_data(output_index).unwrap();

    let out_info = interpreter.tensor_info(output_index).unwrap();
    format!("{:?}",out_info);

    let items: Vec<Item> = Vec::new();
    web::Json(items)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    assert_eq!(args().len(), 2, "minimal <tflite model>");

    let filename = args().nth(1).unwrap();

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(AppState {
                model: filename.to_owned(),
            }))
            .data(web::PayloadConfig::new(1 << 24))
            .service(models)
            .service(invoke)
            .service(web::redirect("/", "/index.html"))
            .service(fs::Files::new("/", "./static").show_files_listing())
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
