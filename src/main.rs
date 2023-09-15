use std::env::args;

use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, InterpreterBuilder};

use actix_web::{get, post, web, App, HttpServer, Responder};
use actix_files as fs;

struct AppState {
        model: String,
}

#[get("/models")]
async fn models(data: web::Data<AppState>) -> impl Responder {
    let filename = data.model.as_str();
    format!("[\"{}\"]",filename)
}

#[post("/invoke")]
async fn invoke(data: web::Data<AppState>) -> impl Responder {
    let filename = data.model.as_str();
    let model = FlatBufferModel::build_from_file(filename).unwrap();
    let resolver = BuiltinOpResolver::default();

    let builder = InterpreterBuilder::new(&model, &resolver).unwrap();
    let mut interpreter = builder.build().unwrap();
    interpreter.allocate_tensors().unwrap();
    interpreter.print_state();
    format!("{}",interpreter.inputs().len())
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
            .service(models)
            .service(invoke)
            .service(web::redirect("/", "/index.html"))
            .service(fs::Files::new("/", "./static").show_files_listing())
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
