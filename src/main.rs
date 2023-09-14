use std::env::args;

use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, Interpreter, InterpreterBuilder, Result};

use rocket::{get, State};

#[macro_use]
extern crate rocket;

#[get("/")]
fn index() -> &'static str {
    "Hello, world!"
}

#[get("/invoke")]
fn invoke(interpreter: State<&Interpreter>) -> &'static str {
    interpreter.print_state()
}

#[launch]
fn rocket() -> _ {
    assert_eq!(args().len(), 2, "minimal <tflite model>");

    let filename = args().nth(1).unwrap();

    let model = FlatBufferModel::build_from_file(filename).unwrap();
    let resolver = BuiltinOpResolver::default();

    let builder = InterpreterBuilder::new(&model, &resolver).unwrap();
    let mut interpreter = builder.build().unwrap();
    interpreter.allocate_tensors().unwrap();


    rocket::build().manage(&interpreter)
                    .mount("/", routes![index,invoke])
}
